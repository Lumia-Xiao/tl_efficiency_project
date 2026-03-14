from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"MAE": mae, "RMSE": rmse, "MAPE_pct": mape, "R2": r2}


@torch.no_grad()
def collect_predictions(model, data_loader, device: str) -> Dict[str, np.ndarray]:
    model.eval()
    comps, totals, y_total = [], [], []
    has_comp = False
    y_comp = []

    for batch in data_loader:
        x = batch["x"].to(device)
        out = model(x)
        comps.append(out["components"].cpu().numpy())
        totals.append(out["total"].cpu().numpy())
        y_total.append(batch["total"].cpu().numpy())
        if "comp" in batch:
            has_comp = True
            y_comp.append(batch["comp"].cpu().numpy())

    result = {
        "pred_components": np.vstack(comps),
        "pred_total": np.vstack(totals),
        "true_total": np.vstack(y_total),
    }
    if has_comp:
        result["true_components"] = np.vstack(y_comp)
    return result


def save_metrics(metrics: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def plot_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.8)
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    plt.plot([vmin, vmax], [vmin, vmax])
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_component_means(pred_components: np.ndarray, comp_names: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    means = pred_components.mean(axis=0)
    plt.figure(figsize=(7, 4))
    plt.bar(comp_names, means)
    plt.ylabel("Mean predicted loss (W)")
    plt.title("Average predicted loss components")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
