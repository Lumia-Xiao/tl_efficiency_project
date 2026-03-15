from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.model import ComponentSumModel


@dataclass
class LoadedModel:
    name: str
    model: ComponentSumModel


@dataclass
class ArtifactBundle:
    scaler: object | None
    models: List[LoadedModel]
    missing: List[str]


def _default_paths(cfg: Config) -> Dict[str, str]:
    return {
        "scaler": os.path.join(cfg.output_dir, "scaler.joblib"),
        "simulation_domain": os.path.join(cfg.output_dir, "best_pretrained.pt"),
        "experiment_domain": os.path.join(cfg.output_dir, "best_finetuned.pt"),
    }


def load_artifacts(cfg: Config, device: str, paths: Dict[str, str]) -> ArtifactBundle:
    missing: List[str] = []

    scaler = None
    scaler_path = paths["scaler"]
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        missing.append(f"Scaler not found: {scaler_path}")

    models: List[LoadedModel] = []
    for domain_name in ["simulation_domain", "experiment_domain"]:
        ckpt_path = paths[domain_name]
        if not os.path.exists(ckpt_path):
            missing.append(f"Checkpoint for {domain_name} not found: {ckpt_path}")
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

    return ArtifactBundle(scaler=scaler, models=models, missing=missing)


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
    def __init__(self, root: tk.Tk, cfg: Config, device: str, scaler_path: Optional[str], sim_ckpt_path: Optional[str], exp_ckpt_path: Optional[str]):
        self.root = root
        self.cfg = cfg
        self.device = device
        self.root.title("Transformer Loss Component Predictor")
        self.root.geometry("1060x620")

        defaults = _default_paths(cfg)
        self.paths = {
            "scaler": scaler_path or defaults["scaler"],
            "simulation_domain": sim_ckpt_path or defaults["simulation_domain"],
            "experiment_domain": exp_ckpt_path or defaults["experiment_domain"],
        }

        self.models: List[LoadedModel] = []
        self.scaler = None

        top = ttk.Frame(root, padding=12)
        top.pack(fill=tk.BOTH, expand=True)

        info = ttk.Label(
            top,
            text=(
                "Enter Vin, Vo, D1, D2, DT, Fs, Po.\n"
                "Predicts PIron, PCond, PCopp, PSw, Ploss for simulation_domain and experiment_domain.\n"
                "If artifacts are missing, choose paths below and click 'Load Artifacts'."
            ),
            justify=tk.LEFT,
        )
        info.pack(anchor=tk.W, pady=(0, 10))

        path_frame = ttk.LabelFrame(top, text="Model Artifacts", padding=8)
        path_frame.pack(fill=tk.X, pady=(0, 10))
        self.path_vars: Dict[str, tk.StringVar] = {}

        self._build_path_row(path_frame, "scaler", "Scaler (.joblib)", 0)
        self._build_path_row(path_frame, "simulation_domain", "Simulation checkpoint (.pt)", 1)
        self._build_path_row(path_frame, "experiment_domain", "Experiment checkpoint (.pt)", 2)

        ttk.Button(path_frame, text="Load Artifacts", command=self.on_load_artifacts).grid(row=3, column=2, sticky="e", pady=(8, 0))

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
        self.predict_btn = ttk.Button(btn_frame, text="Predict", command=self.on_predict)
        self.predict_btn.pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Clear", command=self.on_clear).pack(side=tk.LEFT, padx=8)

        columns = ["domain"] + cfg.component_cols + [cfg.total_col]
        self.table = ttk.Treeview(top, columns=columns, show="headings", height=8)
        for col in columns:
            self.table.heading(col, text=col)
            self.table.column(col, width=145, anchor=tk.CENTER)
        self.table.pack(fill=tk.BOTH, expand=True)

        self.status = ttk.Label(top, text="")
        self.status.pack(anchor=tk.W, pady=(8, 0))

        self.on_load_artifacts(show_dialog=False)

    def _build_path_row(self, parent, key: str, label_text: str, row: int) -> None:
        ttk.Label(parent, text=label_text, width=28).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        var = tk.StringVar(value=self.paths[key])
        self.path_vars[key] = var
        ttk.Entry(parent, textvariable=var, width=90).grid(row=row, column=1, sticky="we", pady=4)
        ttk.Button(parent, text="Browse", command=lambda k=key: self.on_browse(k)).grid(row=row, column=2, padx=(8, 0), pady=4)

    def on_browse(self, key: str) -> None:
        if key == "scaler":
            selected = filedialog.askopenfilename(filetypes=[("Joblib files", "*.joblib"), ("All files", "*.*")])
        else:
            selected = filedialog.askopenfilename(filetypes=[("PyTorch checkpoints", "*.pt"), ("All files", "*.*")])
        if selected:
            self.path_vars[key].set(selected)

    def on_load_artifacts(self, show_dialog: bool = True) -> None:
        for key in self.paths:
            self.paths[key] = self.path_vars[key].get().strip()

        bundle = load_artifacts(self.cfg, self.device, self.paths)
        self.models = bundle.models
        self.scaler = bundle.scaler

        ready = self.scaler is not None and len(self.models) > 0
        self.predict_btn.state(["!disabled"] if ready else ["disabled"])

        loaded_domains = ", ".join(m.name for m in self.models) if self.models else "none"
        if bundle.missing:
            msg = " | ".join(bundle.missing)
            self.status.config(text=f"Loaded domains: {loaded_domains}. Missing: {msg}")
            if show_dialog:
                messagebox.showwarning("Artifacts Missing", msg)
        else:
            self.status.config(text=f"Loaded domains: {loaded_domains}. Ready for prediction.")
            if show_dialog:
                messagebox.showinfo("Artifacts Loaded", "All artifacts loaded successfully.")

    def on_clear(self) -> None:
        for v in self.entries.values():
            v.set("0")
        for item in self.table.get_children():
            self.table.delete(item)

    def on_predict(self) -> None:
        if self.scaler is None or not self.models:
            messagebox.showwarning("Artifacts Missing", "Please load scaler and at least one checkpoint first.")
            return

        try:
            values = [float(self.entries[col].get().strip()) for col in self.cfg.input_cols]
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values for all inputs.")
            return

        try:
            results = predict_row(self.models, self.scaler, values, self.cfg, self.device)
        except Exception as exc:
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
    parser.add_argument("--scaler-path", type=str, default=None)
    parser.add_argument("--simulation-ckpt", type=str, default=None)
    parser.add_argument("--experiment-ckpt", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    device = args.device
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"

    root = tk.Tk()
    LossPredictorGUI(
        root,
        cfg=cfg,
        device=device,
        scaler_path=args.scaler_path,
        sim_ckpt_path=args.simulation_ckpt,
        exp_ckpt_path=args.experiment_ckpt,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
