from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, messagebox

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.model import ComponentSumModel


@dataclass
class LoadedModel:
    name: str
    model: ComponentSumModel


def _checkpoint_map(cfg: Config) -> Dict[str, str]:
    return {
        "simulation_domain": os.path.join(cfg.output_dir, "best_pretrained.pt"),
        "experiment_domain": os.path.join(cfg.output_dir, "best_finetuned.pt"),
    }


def load_models(cfg: Config, device: str) -> tuple[List[LoadedModel], object]:
    scaler_path = os.path.join(cfg.output_dir, "scaler.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}. Run training first.")
    scaler = joblib.load(scaler_path)

    models: List[LoadedModel] = []
    for domain_name, ckpt_path in _checkpoint_map(cfg).items():
        if not os.path.exists(ckpt_path):
            continue
        model = ComponentSumModel(
            in_dim=len(cfg.input_cols),
            num_components=len(cfg.component_cols),
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        ).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        models.append(LoadedModel(name=domain_name, model=model))

    if not models:
        raise FileNotFoundError(
            "No checkpoints found. Expected at least one of: "
            f"{', '.join(_checkpoint_map(cfg).values())}. Run training first."
        )

    return models, scaler


def predict_row(models: List[LoadedModel], scaler, values: List[float], cfg: Config, device: str) -> Dict[str, Dict[str, float]]:
    x_np = np.asarray(values, dtype=np.float32).reshape(1, -1)
    x_scaled = scaler.transform(x_np)
    x = torch.tensor(x_scaled, dtype=torch.float32, device=device)

    results: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for loaded in models:
            out = loaded.model(x)
            comp = out["components"].detach().cpu().numpy().reshape(-1)
            total = float(out["total"].detach().cpu().numpy().reshape(-1)[0])
            row = {name: float(val) for name, val in zip(cfg.component_cols, comp)}
            row[cfg.total_col] = total
            results[loaded.name] = row
    return results


class LossPredictorGUI:
    def __init__(self, root: tk.Tk, cfg: Config, device: str):
        self.root = root
        self.cfg = cfg
        self.device = device
        self.root.title("Transformer Loss Component Predictor")
        self.root.geometry("980x520")

        self.models, self.scaler = load_models(cfg, device)

        top = ttk.Frame(root, padding=12)
        top.pack(fill=tk.BOTH, expand=True)

        info = ttk.Label(
            top,
            text=(
                "Enter Vin, Vo, D1, D2, DT, Fs, Po.\n"
                "Outputs are predicted PIron, PCond, PCopp, PSw, Ploss for both checkpoints:\n"
                "simulation_domain (best_pretrained.pt) and experiment_domain (best_finetuned.pt)."
            ),
            justify=tk.LEFT,
        )
        info.pack(anchor=tk.W, pady=(0, 10))

        form = ttk.Frame(top)
        form.pack(fill=tk.X)

        self.entries: Dict[str, tk.StringVar] = {}
        for idx, col in enumerate(cfg.input_cols):
            r, c = divmod(idx, 4)
            label = ttk.Label(form, text=col, width=8)
            label.grid(row=r * 2, column=c, sticky="w", padx=6, pady=(4, 0))
            var = tk.StringVar(value="0")
            entry = ttk.Entry(form, textvariable=var, width=16)
            entry.grid(row=r * 2 + 1, column=c, sticky="we", padx=6, pady=(0, 8))
            self.entries[col] = var

        btn_frame = ttk.Frame(top)
        btn_frame.pack(fill=tk.X, pady=(6, 8))
        ttk.Button(btn_frame, text="Predict", command=self.on_predict).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Clear", command=self.on_clear).pack(side=tk.LEFT, padx=8)

        columns = ["domain"] + cfg.component_cols + [cfg.total_col]
        self.table = ttk.Treeview(top, columns=columns, show="headings", height=6)
        for col in columns:
            self.table.heading(col, text=col)
            self.table.column(col, width=145, anchor=tk.CENTER)
        self.table.pack(fill=tk.BOTH, expand=True)

        status_names = ", ".join(m.name for m in self.models)
        self.status = ttk.Label(top, text=f"Loaded models: {status_names}")
        self.status.pack(anchor=tk.W, pady=(8, 0))

    def on_clear(self) -> None:
        for v in self.entries.values():
            v.set("0")
        for item in self.table.get_children():
            self.table.delete(item)

    def on_predict(self) -> None:
        try:
            values = [float(self.entries[col].get().strip()) for col in self.cfg.input_cols]
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values for all inputs.")
            return

        try:
            results = predict_row(self.models, self.scaler, values, self.cfg, self.device)
        except Exception as exc:  # runtime inference/display error
            messagebox.showerror("Prediction Error", str(exc))
            return

        for item in self.table.get_children():
            self.table.delete(item)

        for domain_name in ["simulation_domain", "experiment_domain"]:
            if domain_name not in results:
                continue
            row = results[domain_name]
            vals = [domain_name] + [f"{row[name]:.6f}" for name in self.cfg.component_cols + [self.cfg.total_col]]
            self.table.insert("", tk.END, values=vals)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI for transformer loss component prediction.")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    device = args.device
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"

    try:
        root = tk.Tk()
        LossPredictorGUI(root, cfg=cfg, device=device)
    except Exception as exc:
        print(f"Failed to start GUI: {exc}")
        raise

    root.mainloop()


if __name__ == "__main__":
    main()
