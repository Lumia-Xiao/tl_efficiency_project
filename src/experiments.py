from __future__ import annotations

import argparse
import copy
import os
import random
from dataclasses import replace
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .data_utils import split_and_scale
from .evaluate import collect_predictions, regression_metrics, save_metrics
from .model import ComponentSumModel
from .train import EarlyStopper, build_loss, run_source_epoch, run_target_epoch


class TotalOnlyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        total = self.head(self.backbone(x))
        return {"total": total}


@torch.no_grad()
def collect_total_predictions(model: nn.Module, data_loader, device: str) -> Dict[str, np.ndarray]:
    model.eval()
    pred_total, true_total = [], []
    for batch in data_loader:
        x = batch["x"].to(device)
        out = model(x)
        pred_total.append(out["total"].cpu().numpy())
        true_total.append(batch["total"].cpu().numpy())
    return {"pred_total": np.vstack(pred_total), "true_total": np.vstack(true_total)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        loss = loss_fn(out["total"], y_total)

        if train:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

    return {"loss": total_loss / max(n, 1)}


def train_proposed(cfg: Config) -> Tuple[Dict, Dict]:
    data = split_and_scale(cfg)
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

    if cfg.freeze_backbone_in_finetune:
        model.freeze_backbone()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr * 0.5, weight_decay=cfg.weight_decay)

    stopper = EarlyStopper(patience=cfg.early_stop_patience)
    for _ in range(cfg.epochs_finetune):
        run_target_epoch(model, data.source_train_loader, data.target_train_loader, optimizer, cfg.device, loss_fn, cfg, train=True)
        val_stats = run_target_epoch(model, data.source_val_loader, data.target_val_loader, optimizer, cfg.device, loss_fn, cfg, train=False)
        if stopper.step(val_stats["loss"], model):
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    tgt_pred = collect_predictions(model, data.target_val_loader, cfg.device)
    src_pred = collect_predictions(model, data.source_val_loader, cfg.device)
    metrics = {
        "target_val_total": regression_metrics(tgt_pred["true_total"], tgt_pred["pred_total"]),
        "source_val_total": regression_metrics(src_pred["true_total"], src_pred["pred_total"]),
    }
    return metrics, {
        "name": "proposed",
        "uses_source_transfer": True,
        "uses_component_supervision": True,
        "uses_sum_constraint": True,
    }


def train_ablation_no_component_supervision(cfg: Config) -> Tuple[Dict, Dict]:
    ab_cfg = replace(cfg, lambda_src_components=0.0)
    metrics, meta = train_proposed(ab_cfg)
    meta["name"] = "ablation_no_component_supervision"
    meta["uses_component_supervision"] = False
    return metrics, meta


def train_baseline_target_only(cfg: Config) -> Tuple[Dict, Dict]:
    data = split_and_scale(cfg)
    model = TotalOnlyMLP(in_dim=len(cfg.input_cols), hidden_dims=cfg.hidden_dims, dropout=cfg.dropout).to(cfg.device)
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

    tgt_pred = collect_total_predictions(model, data.target_val_loader, cfg.device)
    metrics = {
        "target_val_total": regression_metrics(tgt_pred["true_total"], tgt_pred["pred_total"]),
    }
    return metrics, {
        "name": "baseline_target_only_total_regression",
        "uses_source_transfer": False,
        "uses_component_supervision": False,
        "uses_sum_constraint": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline + ablation experiments against the proposed method.")
    parser.add_argument("--output", type=str, default="outputs/experiment_comparison.json")
    parser.add_argument("--epochs-pretrain", type=int, default=200)
    parser.add_argument("--epochs-finetune", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        epochs_pretrain=args.epochs_pretrain,
        epochs_finetune=args.epochs_finetune,
        early_stop_patience=args.patience,
        random_seed=args.seed,
        device=args.device,
    )

    if cfg.device == "cpu" and torch.cuda.is_available():
        cfg.device = "cuda"

    set_seed(cfg.random_seed)

    experiments = [
        ("baseline_target_only_total_regression", train_baseline_target_only),
        ("ablation_no_component_supervision", train_ablation_no_component_supervision),
        ("proposed", train_proposed),
    ]

    results: Dict[str, Dict] = {}
    for name, fn in experiments:
        set_seed(cfg.random_seed)
        metrics, meta = fn(copy.deepcopy(cfg))
        results[name] = {
            "meta": meta,
            "metrics": metrics,
        }
        tgt = metrics["target_val_total"]
        print(
            f"[{name}] target MAE={tgt['MAE']:.4f} RMSE={tgt['RMSE']:.4f} "
            f"MAPE={tgt['MAPE_pct']:.2f}% R2={tgt['R2']:.4f}"
        )

    base_rmse = results["baseline_target_only_total_regression"]["metrics"]["target_val_total"]["RMSE"]
    prop_rmse = results["proposed"]["metrics"]["target_val_total"]["RMSE"]
    results["summary"] = {
        "target_rmse_improvement_vs_baseline_pct": (base_rmse - prop_rmse) / max(base_rmse, 1e-12) * 100.0,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_metrics(results, args.output)
    print(f"Saved experiment comparison to: {args.output}")


if __name__ == "__main__":
    main()