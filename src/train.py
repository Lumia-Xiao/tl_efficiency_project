from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import Config
    from src.data_utils import split_and_scale, save_scaler
    from src.evaluate import regression_metrics, collect_predictions, save_metrics, plot_true_vs_pred, plot_component_means
    from src.model import ComponentSumModel
else:
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


def components_to_share_and_clr(components: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.clamp(components.sum(dim=1, keepdim=True), min=eps)
    share = torch.clamp(components / total, min=eps)
    log_share = torch.log(share)
    clr = log_share - log_share.mean(dim=1, keepdim=True)
    return share, clr


def batch_covariance(x: torch.Tensor) -> torch.Tensor:
    if x.size(0) <= 1:
        return torch.zeros(x.size(1), x.size(1), device=x.device, dtype=x.dtype)
    centered = x - x.mean(dim=0, keepdim=True)
    return centered.transpose(0, 1) @ centered / (x.size(0) - 1)


def source_relation_loss(pred_components: torch.Tensor, true_components: torch.Tensor, cfg: Config) -> torch.Tensor:
    _, pred_clr = components_to_share_and_clr(pred_components, cfg.relation_eps)
    _, true_clr = components_to_share_and_clr(true_components, cfg.relation_eps)
    return F.smooth_l1_loss(pred_clr, true_clr)


def source_component_supervision_loss(
    pred_components: torch.Tensor, true_components: torch.Tensor, loss_fn, cfg: Config
) -> torch.Tensor:
    true_total = torch.clamp(true_components.sum(dim=1, keepdim=True), min=cfg.relation_eps)
    pred_share, _ = components_to_share_and_clr(pred_components, cfg.relation_eps)
    true_share, _ = components_to_share_and_clr(true_components, cfg.relation_eps)
    share_loss = F.smooth_l1_loss(pred_share, true_share)
    pred_norm = pred_components / true_total
    true_norm = true_components / true_total
    norm_loss = loss_fn(pred_norm, true_norm)
    return (
        cfg.component_share_loss_weight * share_loss
        + cfg.component_norm_loss_weight * norm_loss
    )


def target_relation_prior_loss(
    pred_components: torch.Tensor, relation_prior: Dict[str, torch.Tensor], cfg: Config, scale: float = 1.0
) -> Dict[str, torch.Tensor]:
    share, pred_clr = components_to_share_and_clr(pred_components, cfg.relation_eps)
    mean_loss = F.smooth_l1_loss(pred_clr.mean(dim=0), relation_prior["clr_mean"])
    cov_loss = F.smooth_l1_loss(batch_covariance(pred_clr), relation_prior["clr_cov"])
    share_loss = F.smooth_l1_loss(share.mean(dim=0), relation_prior["share_mean"])
    return {
        "loss": scale * (
            cfg.lambda_tgt_relation_mean * (mean_loss + share_loss) + cfg.lambda_tgt_relation_cov * cov_loss
        ),
        "mean_loss": mean_loss,
        "cov_loss": cov_loss,
        "share_loss": share_loss,
    }


def build_relation_prior_tensors(relation_prior: Dict[str, np.ndarray], device: str) -> Dict[str, torch.Tensor]:
    return {
        name: torch.tensor(value, dtype=torch.float32, device=device)
        for name, value in relation_prior.items()
    }


def get_source_loss_weights(cfg: Config, stage: str, epoch: int | None = None, total_epochs: int | None = None) -> Dict[str, float]:
    if stage == "finetune":
        return {
            "components": cfg.lambda_src_components_finetune,
            "total": cfg.lambda_src_total_finetune,
            "relation": cfg.lambda_src_relation_finetune,
        }
    component_weight = cfg.lambda_src_components
    if epoch is not None and total_epochs is not None and total_epochs > 0:
        progress = min(max(epoch / total_epochs, 0.0), 1.0)
        component_weight = cfg.lambda_src_components * ((1.0 - progress) ** cfg.src_component_decay_power)
    return {
        "components": component_weight,
        "total": cfg.lambda_src_total,
        "relation": cfg.lambda_src_relation,
    }


def get_target_relation_scale(cfg: Config, epoch: int | None, total_epochs: int | None) -> float:
    if epoch is None or total_epochs is None:
        return 1.0
    warmup_epochs = max(1, min(cfg.tgt_relation_warmup_epochs, total_epochs))
    return min(epoch / warmup_epochs, 1.0)


def run_source_epoch(
    model, loader, optimizer, device: str, loss_fn, cfg: Config, train: bool = True,
    epoch: int | None = None, total_epochs: int | None = None
) -> Dict[str, float]:
    model.train() if train else model.eval()
    total_loss = 0.0
    total_comp = 0.0
    total_total = 0.0
    total_relation = 0.0
    n = 0
    weights = get_source_loss_weights(cfg, stage="pretrain", epoch=epoch, total_epochs=total_epochs)

    for batch in loader:
        x = batch["x"].to(device)
        y_comp = batch["comp"].to(device)
        y_total = batch["total"].to(device)

        if train:
            optimizer.zero_grad()

        out = model(x)
        loss_comp = source_component_supervision_loss(out["components"], y_comp, loss_fn, cfg)
        loss_total = loss_fn(out["total"], y_total)
        loss_relation = source_relation_loss(out["components"], y_comp, cfg)
        loss = (
            weights["components"] * loss_comp
            + weights["total"] * loss_total
            + weights["relation"] * loss_relation
        )

        if train:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_comp += loss_comp.item() * bs
        total_total += loss_total.item() * bs
        total_relation += loss_relation.item() * bs
        n += bs

    return {
        "loss": total_loss / max(n, 1),
        "comp_loss": total_comp / max(n, 1),
        "total_loss": total_total / max(n, 1),
        "relation_loss": total_relation / max(n, 1),
        "comp_weight": weights["components"],
    }


def run_target_epoch(
    model, src_loader, tgt_loader, optimizer, device: str, loss_fn, cfg: Config,
    relation_prior: Dict[str, torch.Tensor], train: bool = True,
    epoch: int | None = None, total_epochs: int | None = None
) -> Dict[str, float]:
    model.train() if train else model.eval()
    total_loss = 0.0
    total_src = 0.0
    total_tgt = 0.0
    total_src_relation = 0.0
    total_tgt_relation = 0.0
    n = 0
    src_weights = get_source_loss_weights(cfg, stage="finetune")
    tgt_relation_scale = get_target_relation_scale(cfg, epoch, total_epochs)

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
            src_weights["components"] * source_component_supervision_loss(out_s["components"], y_comp_s, loss_fn, cfg)
            + src_weights["total"] * loss_fn(out_s["total"], y_total_s)
        )
        src_relation = src_weights["relation"] * source_relation_loss(out_s["components"], y_comp_s, cfg)
        tgt_loss = cfg.lambda_tgt_total * loss_fn(out_t["total"], y_total_t)
        tgt_relation = target_relation_prior_loss(
            out_t["components"], relation_prior, cfg, scale=tgt_relation_scale
        )["loss"]
        loss = src_loss + src_relation + tgt_loss + tgt_relation

        if train:
            loss.backward()
            optimizer.step()

        bs = x_s.size(0)
        total_loss += loss.item() * bs
        total_src += src_loss.item() * bs
        total_tgt += tgt_loss.item() * bs
        total_src_relation += src_relation.item() * bs
        total_tgt_relation += tgt_relation.item() * bs
        n += bs

    return {
        "loss": total_loss / max(n, 1),
        "src_loss": total_src / max(n, 1),
        "tgt_loss": total_tgt / max(n, 1),
        "src_relation_loss": total_src_relation / max(n, 1),
        "tgt_relation_loss": total_tgt_relation / max(n, 1),
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
    relation_prior = build_relation_prior_tensors(data.relation_prior, cfg.device)
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
        train_stats = run_source_epoch(
            model, data.source_train_loader, optimizer, cfg.device, loss_fn, cfg,
            train=True, epoch=epoch, total_epochs=cfg.epochs_pretrain
        )
        val_stats = run_source_epoch(
            model, data.source_val_loader, optimizer, cfg.device, loss_fn, cfg,
            train=False, epoch=epoch, total_epochs=cfg.epochs_pretrain
        )
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
        train_stats = run_target_epoch(
            model, data.source_train_loader, data.target_train_loader, optimizer, cfg.device, loss_fn, cfg,
            relation_prior, train=True, epoch=epoch, total_epochs=cfg.epochs_finetune
        )
        val_stats = run_target_epoch(
            model, data.source_val_loader, data.target_val_loader, optimizer, cfg.device, loss_fn, cfg,
            relation_prior, train=False, epoch=epoch, total_epochs=cfg.epochs_finetune
        )
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
