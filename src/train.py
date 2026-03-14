from __future__ import annotations

import argparse
import copy
import json
import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .data_utils import split_and_scale, save_scaler
from .evaluate import regression_metrics, collect_predictions, save_metrics, plot_true_vs_pred, plot_component_means
from .model import ComponentSumModel


class EarlyStopper:
    def __init__(self, patience: int = 30):
        self.patience = patience
        self.best = None
        self.count = 0
        self.best_state = None

    def step(self, value: float, model: torch.nn.Module) -> bool:
        if self.best is None or value < self.best:
            self.best = value
            self.count = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.count += 1
        return self.count >= self.patience


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loss(name: str = "huber"):
    if name == "mae":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    return nn.HuberLoss(delta=1.0)


def run_source_epoch(model, loader, optimizer, device: str, loss_fn, cfg: Config, train: bool = True) -> Dict[str, float]:
    model.train() if train else model.eval()
    total_loss = 0.0
    n = 0

    for batch in loader:
        x = batch["x"].to(device)
        y_comp = batch["comp"].to(device)
        y_total = batch["total"].to(device)

        if train:
            optimizer.zero_grad()

        out = model(x)
        loss_comp = loss_fn(out["components"], y_comp)
        loss_total = loss_fn(out["total"], y_total)
        loss = cfg.lambda_src_components * loss_comp + cfg.lambda_src_total * loss_total

        if train:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

    return {"loss": total_loss / max(n, 1)}


def run_target_epoch(model, src_loader, tgt_loader, optimizer, device: str, loss_fn, cfg: Config, train: bool = True) -> Dict[str, float]:
    model.train() if train else model.eval()
    total_loss = 0.0
    total_src = 0.0
    total_tgt = 0.0
    n = 0

    tgt_iter = iter(tgt_loader)
    for src_batch in src_loader:
        try:
            tgt_batch = next(tgt_iter)
        except StopIteration:
            tgt_iter = iter(tgt_loader)
            tgt_batch = next(tgt_iter)

        x_s = src_batch["x"].to(device)
        y_comp_s = src_batch["comp"].to(device)
        y_total_s = src_batch["total"].to(device)

        x_t = tgt_batch["x"].to(device)
        y_total_t = tgt_batch["total"].to(device)

        if train:
            optimizer.zero_grad()

        out_s = model(x_s)
        out_t = model(x_t)

        src_loss = (
            cfg.lambda_src_components * loss_fn(out_s["components"], y_comp_s)
            + cfg.lambda_src_total * loss_fn(out_s["total"], y_total_s)
        )
        tgt_loss = cfg.lambda_tgt_total * loss_fn(out_t["total"], y_total_t)
        loss = src_loss + tgt_loss

        if train:
            loss.backward()
            optimizer.step()

        bs = x_s.size(0)
        total_loss += loss.item() * bs
        total_src += src_loss.item() * bs
        total_tgt += tgt_loss.item() * bs
        n += bs

    return {
        "loss": total_loss / max(n, 1),
        "src_loss": total_src / max(n, 1),
        "tgt_loss": total_tgt / max(n, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-pretrain", type=int, default=None)
    parser.add_argument("--epochs-finetune", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--freeze-backbone-in-finetune", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    if args.epochs_pretrain is not None:
        cfg.epochs_pretrain = args.epochs_pretrain
    if args.epochs_finetune is not None:
        cfg.epochs_finetune = args.epochs_finetune
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.device is not None:
        cfg.device = args.device
    if args.freeze_backbone_in_finetune:
        cfg.freeze_backbone_in_finetune = True

    if cfg.device == "cpu" and torch.cuda.is_available():
        cfg.device = "cuda"

    set_seed(cfg.random_seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    data = split_and_scale(cfg)
    save_scaler(data.scaler, cfg.output_dir)

    model = ComponentSumModel(
        in_dim=len(cfg.input_cols),
        num_components=len(cfg.component_cols),
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    ).to(cfg.device)

    loss_fn = build_loss("huber")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Stage 1: source pretraining
    stopper = EarlyStopper(patience=cfg.early_stop_patience)
    history = {"pretrain": [], "finetune": []}
    for epoch in range(1, cfg.epochs_pretrain + 1):
        train_stats = run_source_epoch(model, data.source_train_loader, optimizer, cfg.device, loss_fn, cfg, train=True)
        val_stats = run_source_epoch(model, data.source_val_loader, optimizer, cfg.device, loss_fn, cfg, train=False)
        history["pretrain"].append({"epoch": epoch, "train": train_stats, "val": val_stats})
        if epoch % 20 == 0 or epoch == 1:
            print(f"[Pretrain] epoch={epoch:03d} train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f}")
        if stopper.step(val_stats["loss"], model):
            print(f"Early stop in pretraining at epoch {epoch}")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_pretrained.pt"))

    # Stage 2: target weak supervision fine-tuning
    if cfg.freeze_backbone_in_finetune:
        model.freeze_backbone()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr * 0.5, weight_decay=cfg.weight_decay)

    stopper = EarlyStopper(patience=cfg.early_stop_patience)
    for epoch in range(1, cfg.epochs_finetune + 1):
        train_stats = run_target_epoch(model, data.source_train_loader, data.target_train_loader, optimizer, cfg.device, loss_fn, cfg, train=True)
        val_stats = run_target_epoch(model, data.source_val_loader, data.target_val_loader, optimizer, cfg.device, loss_fn, cfg, train=False)
        history["finetune"].append({"epoch": epoch, "train": train_stats, "val": val_stats})
        if epoch % 20 == 0 or epoch == 1:
            print(
                f"[Finetune] epoch={epoch:03d} train_loss={train_stats['loss']:.4f} "
                f"val_loss={val_stats['loss']:.4f} val_tgt={val_stats['tgt_loss']:.4f}"
            )
        if stopper.step(val_stats["loss"], model):
            print(f"Early stop in fine-tuning at epoch {epoch}")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_finetuned.pt"))

    # Evaluation
    src_pred = collect_predictions(model, data.source_val_loader, cfg.device)
    tgt_pred = collect_predictions(model, data.target_val_loader, cfg.device)

    metrics = {
        "source_val_total": regression_metrics(src_pred["true_total"], src_pred["pred_total"]),
        "target_val_total": regression_metrics(tgt_pred["true_total"], tgt_pred["pred_total"]),
    }
    if "true_components" in src_pred:
        comp_metrics = {}
        for i, name in enumerate(cfg.component_cols):
            comp_metrics[name] = regression_metrics(src_pred["true_components"][:, i], src_pred["pred_components"][:, i])
        metrics["source_val_components"] = comp_metrics

    save_metrics(metrics, os.path.join(cfg.output_dir, "metrics.json"))
    with open(os.path.join(cfg.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_true_vs_pred(src_pred["true_total"].reshape(-1), src_pred["pred_total"].reshape(-1),
                      title="Source val total loss", path=os.path.join(cfg.output_dir, "source_val_total.png"))
    plot_true_vs_pred(tgt_pred["true_total"].reshape(-1), tgt_pred["pred_total"].reshape(-1),
                      title="Target val total loss", path=os.path.join(cfg.output_dir, "target_val_total.png"))
    plot_component_means(tgt_pred["pred_components"], cfg.component_cols, os.path.join(cfg.output_dir, "target_component_means.png"))

    print("\nFinished training.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
