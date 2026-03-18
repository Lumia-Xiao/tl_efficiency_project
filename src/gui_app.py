from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Work around OpenMP runtime duplication crashes that can happen on Windows/PyCharm
# when NumPy/Torch pull in different OpenMP-linked binaries in the same process.
if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.model import ComponentSumModel


INPUT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Vlv": (300.0, 500.0),
    "Vhv": (600.0, 800.0),
    "D": (0.0, 1.0),
    "fsw": (50000.0, 100000.0),
    "deadtime_s": (2e-7, 4e-7),
    "Pout": (-20000.0, 20000.0),
}



def _format_bound(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:g}"
    if 0 < abs(value) < 1e-3:
        return f"{value:.1e}"
    return f"{value:g}"


def _default_for_input(col: str) -> str:
    low, high = INPUT_BOUNDS.get(col, (0.0, 0.0))
    default = (low + high) / 2.0 if high > low else 0.0
    return f"{default:g}"


def _range_text(col: str) -> str:
    low, high = INPUT_BOUNDS.get(col, (0.0, 0.0))
    return f"[{_format_bound(low)}, {_format_bound(high)}]"

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

def validate_inputs(values: Dict[str, float]) -> tuple[bool, str]:
    for key, (low, high) in INPUT_BOUNDS.items():
        value = values[key]
        if value < low or value > high:
            return False, f"{key}={value} is out of scope. Allowed range: [{low}, {high}]"

    if values["Vlv"] >= values["Vhv"]:
        return False, f"Input is out of scope: require Vlv < Vhv, but got Vlv={values['Vlv']}, Vhv={values['Vhv']}."

    return True, ""


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
        self.root.title("20-kW Buck-Boost Converter Loss Predictor")
        self.root.geometry("1230x880")

        self._configure_fonts()

        defaults = _default_paths(cfg)
        self.paths = {
            "scaler": scaler_path or defaults["scaler"],
            "simulation_domain": sim_ckpt_path or defaults["simulation_domain"],
            "experiment_domain": exp_ckpt_path or defaults["experiment_domain"],
        }

        self.models: List[LoadedModel] = []
        self.scaler = None

        top = ttk.Frame(root, padding=4)
        top.pack(fill=tk.BOTH, expand=True)

        info = ttk.Label(
            top,
            text=(
                "Input range limits: Vlv[300,500], Vhv[600,800], Vlv<Vhv, D[0,1], fsw[50000,100000], deadtime_s[2e-7,4e-7], Pout[-20000,20000].\n"
                "If any input is out of scope, prediction is blocked and marked as out of scope."
            ),
            justify=tk.LEFT,
        )
        info.pack(anchor=tk.W, pady=(0, 2))

        path_frame = ttk.LabelFrame(top, text="Model Artifacts", padding=2)
        path_frame.pack(fill=tk.X, pady=(0, 2))
        self.path_vars: Dict[str, tk.StringVar] = {}

        self._build_path_row(path_frame, "scaler", "Scaler (.joblib)", 0)
        self._build_path_row(path_frame, "simulation_domain", "Simulation checkpoint (.pt)", 1)
        self._build_path_row(path_frame, "experiment_domain", "Experiment checkpoint (.pt)", 2)

        ttk.Button(path_frame, text="Load Artifacts", command=self.on_load_artifacts).grid(row=3, column=2, sticky="e", pady=(2, 0))

        form = ttk.LabelFrame(top, text="Inputs", padding=2)
        form.pack(fill=tk.X)

        self.entries: Dict[str, tk.StringVar] = {}
        for idx, col in enumerate(cfg.input_cols):
            field = ttk.Frame(form)
            field.grid(row=0, column=idx, sticky="w", padx=4, pady=(2, 4))
            ttk.Label(field, text=col, width=4).pack(anchor="w")
            var = tk.StringVar(value=_default_for_input(col))
            ttk.Entry(field, textvariable=var, width=14).pack(anchor="w", pady=(2, 0))
            ttk.Label(field, text=f"range: {_range_text(col)}", foreground="#555555").pack(anchor="w")
            self.entries[col] = var

        btn_frame = ttk.Frame(top)
        btn_frame.pack(fill=tk.X, pady=(4, 4))
        self.predict_btn = ttk.Button(btn_frame, text="Predict", command=self.on_predict)
        self.predict_btn.pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Clear", command=self.on_clear).pack(side=tk.LEFT, padx=4)

        columns = ["domain"] + cfg.component_cols + [cfg.total_col]
        self.table = ttk.Treeview(top, columns=columns, show="headings", height=2)
        for col in columns:
            self.table.heading(col, text=col)
            self.table.column(col, width=170, anchor=tk.CENTER)
        self.table.pack(fill=tk.X, expand=False)

        chart_frame = ttk.Frame(top)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        self.fig = Figure(figsize=(16, 4.8), dpi=100)
        self.ax_pie_sim = self.fig.add_subplot(1, 2, 1)
        self.ax_pie_exp = self.fig.add_subplot(1, 2, 2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_empty_charts()

        self.status = ttk.Label(top, text="")
        self.status.pack(anchor=tk.W, pady=(4, 0))

        self.on_load_artifacts(show_dialog=False)

    def _configure_fonts(self) -> None:
        self.root.option_add("*Font", "SegoeUI 12")
        style = ttk.Style()
        style.configure("Treeview", rowheight=28, font=("SegoeUI", 12))
        style.configure("Treeview.Heading", font=("SegoeUI", 12, "bold"))
        style.configure("TLabel", font=("SegoeUI", 12))
        style.configure("TButton", font=("SegoeUI", 12))
        style.configure("TLabelframe.Label", font=("SegoeUI", 12, "bold"))

    def _draw_empty_charts(self) -> None:
        self.ax_pie_sim.clear()
        self.ax_pie_exp.clear()

        self.ax_pie_sim.set_title("Simulation component share")
        self.ax_pie_sim.text(0.5, 0.5, "Run prediction to view chart", ha="center", va="center",
                             transform=self.ax_pie_sim.transAxes)

        self.ax_pie_exp.set_title("Experiment component share")
        self.ax_pie_exp.text(0.5, 0.5, "Run prediction to view chart", ha="center", va="center",
                             transform=self.ax_pie_exp.transAxes)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _render_charts(self, results: Dict[str, Dict[str, float]]) -> None:
        domains = [d for d in ["simulation_domain", "experiment_domain"] if d in results]
        if not domains:
            self._draw_empty_charts()
            return

        self.ax_pie_sim.clear()
        self.ax_pie_exp.clear()

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        components = self.cfg.component_cols

        def draw_component_pie(ax, domain: str, title: str) -> None:
            if domain not in results:
                ax.set_title(title)
                ax.text(0.5, 0.5, "Not available", ha="center", va="center", transform=ax.transAxes)
                return

            values = np.array([results[domain][comp] for comp in components], dtype=float)
            total = float(values.sum())
            labels = [f"{comp} ({val:.3f}W)" for comp, val in zip(components, values)]
            if total <= 0:
                ax.set_title(title)
                ax.text(0.5, 0.5, "Total <= 0", ha="center", va="center", transform=ax.transAxes)
                return

            ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
            ax.set_title(f"{title} (total={total:.3f}W)")

        draw_component_pie(self.ax_pie_sim, "simulation_domain", "Simulation component share")
        draw_component_pie(self.ax_pie_exp, "experiment_domain", "Experiment component share")

        self.fig.tight_layout()
        self.canvas.draw_idle()

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
        for col, v in self.entries.items():
            v.set(_default_for_input(col))
        for item in self.table.get_children():
            self.table.delete(item)
        self._draw_empty_charts()

    def on_predict(self) -> None:
        if self.scaler is None or not self.models:
            messagebox.showwarning("Artifacts Missing", "Please load scaler and at least one checkpoint first.")
            return

        try:
            value_map = {col: float(self.entries[col].get().strip()) for col in self.cfg.input_cols}
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values for all inputs.")
            return

        in_scope, msg = validate_inputs(value_map)
        if not in_scope:
            for item in self.table.get_children():
                self.table.delete(item)
            self._draw_empty_charts()
            self.status.config(text=f"Out of scope: {msg}")
            messagebox.showwarning("Out of Scope", msg)
            return

        try:
            values = [value_map[col] for col in self.cfg.input_cols]
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

        self._render_charts(results)

        if "simulation_domain" in results and "experiment_domain" in results:
            sim = results["simulation_domain"][self.cfg.total_col]
            exp = results["experiment_domain"][self.cfg.total_col]
            rel = ((exp - sim) / max(abs(sim), 1e-8)) * 100.0
            self.status.config(text=f"Predicted totals -> simulation: {sim:.4f} W, experiment: {exp:.4f} W, difference: {exp-sim:+.4f} W ({rel:+.2f}%).")
        else:
            self.status.config(text="Prediction complete for available domain(s).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI for buck-boost converter loss component prediction.")
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