from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Dict

import torch

from .config import Config
from .data_utils import split_and_scale
from .evaluate import regression_metrics, collect_predictions, save_metrics
from .model import ComponentSumModel, TotalBaselineModel
from .train import set_seed, build_loss, run_source_epoch, run_target_epoch, EarlyStopper


def clone_cfg(cfg: Config) -> Config:
    return copy.deepcopy(cfg)


def run_total_only_epoch(model, loader, optimizer, device: str, loss_fn, train: bool = True) -> Dict[str, float]:
    model.train() if train else model.eval()
    total_loss = 0.0
    n = 0

    for batch in loader:
        x = batch["x"].to(device)
        y_total = batch["total"].to(device)

        if train:
            optimizer.zero_grad()

        out = model(x)
        pred_total = out["total"]
        loss = loss_fn(pred_total, y_total)

        if train:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

    return {"loss": total_loss / max(n, 1)}


def train_full_method(cfg: Config, data) -> Dict[str, float]:
    model = ComponentSumModel(
        in_dim=len(cfg.input_cols),
        num_components=len(cfg.component_cols),
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    ).to(cfg.device)

    loss_fn = build_loss("huber")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    stopper = EarlyStopper(patience=cfg.early_stop_patience)
    for _ in range(cfg.epochs_pretrain):
        run_source_epoch(model, data.source_train_loader, optimizer, cfg.device, loss_fn, cfg, train=True)
        val_stats = run_source_epoch(model, data.source_val_loader, optimizer, cfg.device, loss_fn, cfg, train=False)
        if stopper.step(val_stats["loss"], model):
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr * 0.5, weight_decay=cfg.weight_decay)
    stopper = EarlyStopper(patience=cfg.early_stop_patience)
    for _ in range(cfg.epochs_finetune):
        run_target_epoch(model, data.source_train_loader, data.target_train_loader, optimizer, cfg.device, loss_fn, cfg, train=True)
        val_stats = run_target_epoch(model, data.source_val_loader, data.target_val_loader, optimizer, cfg.device, loss_fn, cfg, train=False)
        if stopper.step(val_stats["loss"], model):
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    tgt_pred = collect_predictions(model, data.target_val_loader, cfg.device)
    return regression_metrics(tgt_pred["true_total"], tgt_pred["pred_total"])


def train_target_only_baseline(cfg: Config, data) -> Dict[str, float]:
    model = TotalBaselineModel(
        in_dim=len(cfg.input_cols),
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    ).to(cfg.device)

    loss_fn = build_loss("huber")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    stopper = EarlyStopper(patience=cfg.early_stop_patience)

    for _ in range(cfg.epochs_finetune):
        run_total_only_epoch(model, data.target_train_loader, optimizer, cfg.device, loss_fn, train=True)
        val_stats = run_total_only_epoch(model, data.target_val_loader, optimizer, cfg.device, loss_fn, train=False)
        if stopper.step(val_stats["loss"], model):
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    tgt_pred = collect_predictions(model, data.target_val_loader, cfg.device)
    return regression_metrics(tgt_pred["true_total"], tgt_pred["pred_total"])


def train_ablation_no_component_supervision(cfg: Config, data) -> Dict[str, float]:
    ab_cfg = clone_cfg(cfg)
    ab_cfg.lambda_src_components = 0.0
    return train_full_method(ab_cfg, data)


def train_ablation_no_source_transfer(cfg: Config, data) -> Dict[str, float]:
    model = ComponentSumModel(
        in_dim=len(cfg.input_cols),
        num_components=len(cfg.component_cols),
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    ).to(cfg.device)

    loss_fn = build_loss("huber")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    stopper = EarlyStopper(patience=cfg.early_stop_patience)

    for _ in range(cfg.epochs_finetune):
        run_total_only_epoch(model, data.target_train_loader, optimizer, cfg.device, loss_fn, train=True)
        val_stats = run_total_only_epoch(model, data.target_val_loader, optimizer, cfg.device, loss_fn, train=False)
        if stopper.step(val_stats["loss"], model):
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    tgt_pred = collect_predictions(model, data.target_val_loader, cfg.device)
    return regression_metrics(tgt_pred["true_total"], tgt_pred["pred_total"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline + ablation experiments.")
    parser.add_argument("--output", type=str, default="outputs/experiments/metrics.json")
    parser.add_argument("--epochs-pretrain", type=int, default=120)
    parser.add_argument("--epochs-finetune", type=int, default=180)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    cfg.epochs_pretrain = args.epochs_pretrain
    cfg.epochs_finetune = args.epochs_finetune
    cfg.early_stop_patience = args.patience
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.device = args.device

    if cfg.device == "cpu" and torch.cuda.is_available():
        cfg.device = "cuda"

    set_seed(cfg.random_seed)
    data = split_and_scale(cfg)

    results = {
        "full_method": train_full_method(cfg, data),
        "baseline_target_only_total_model": train_target_only_baseline(cfg, data),
        "ablation_no_component_supervision": train_ablation_no_component_supervision(cfg, data),
        "ablation_no_source_transfer": train_ablation_no_source_transfer(cfg, data),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_metrics(results, args.output)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()