from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from .config import Config


class LossDataset(Dataset):
    def __init__(self, x: np.ndarray, comp: np.ndarray | None, total: np.ndarray | None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.comp = None if comp is None else torch.tensor(comp, dtype=torch.float32)
        self.total = None if total is None else torch.tensor(total, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        item = {"x": self.x[idx]}
        if self.comp is not None:
            item["comp"] = self.comp[idx]
        if self.total is not None:
            item["total"] = self.total[idx]
        return item


@dataclass
class DataBundle:
    source_train_loader: DataLoader
    source_val_loader: DataLoader
    target_train_loader: DataLoader
    target_val_loader: DataLoader
    source_train_df: pd.DataFrame
    source_val_df: pd.DataFrame
    target_train_df: pd.DataFrame
    target_val_df: pd.DataFrame
    scaler: StandardScaler


def load_csvs(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    project_root = Path(__file__).resolve().parent.parent
    src_path = Path(cfg.source_csv)
    tgt_path = Path(cfg.target_csv)
    if not src_path.is_absolute():
        src_path = project_root / src_path
    if not tgt_path.is_absolute():
        tgt_path = project_root / tgt_path
    src_df = pd.read_csv(src_path)
    tgt_df = pd.read_csv(tgt_path)
    return src_df, tgt_df


def split_and_scale(cfg: Config) -> DataBundle:
    src_df, tgt_df = load_csvs(cfg)

    src_train_df, src_val_df = train_test_split(
        src_df,
        test_size=cfg.source_val_ratio,
        random_state=cfg.random_seed,
        shuffle=True,
    )
    tgt_train_df, tgt_val_df = train_test_split(
        tgt_df,
        test_size=cfg.target_val_ratio,
        random_state=cfg.random_seed,
        shuffle=True,
    )

    scaler = StandardScaler()
    scaler.fit(src_train_df[cfg.input_cols].values)

    def _build(df: pd.DataFrame, has_components: bool) -> LossDataset:
        x = scaler.transform(df[cfg.input_cols].values)
        comp = df[cfg.component_cols].values if has_components else None
        total = df[cfg.total_col].values
        return LossDataset(x=x, comp=comp, total=total)

    source_train_loader = DataLoader(
        _build(src_train_df, has_components=True),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    source_val_loader = DataLoader(
        _build(src_val_df, has_components=True),
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    target_train_loader = DataLoader(
        _build(tgt_train_df, has_components=False),
        batch_size=min(cfg.batch_size, len(tgt_train_df)),
        shuffle=True,
    )
    target_val_loader = DataLoader(
        _build(tgt_val_df, has_components=False),
        batch_size=min(cfg.batch_size, len(tgt_val_df)),
        shuffle=False,
    )

    return DataBundle(
        source_train_loader=source_train_loader,
        source_val_loader=source_val_loader,
        target_train_loader=target_train_loader,
        target_val_loader=target_val_loader,
        source_train_df=src_train_df,
        source_val_df=src_val_df,
        target_train_df=tgt_train_df,
        target_val_df=tgt_val_df,
        scaler=scaler,
    )


def save_scaler(scaler: StandardScaler, output_dir: str) -> None:
    import joblib
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
