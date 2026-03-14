from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    source_csv: str = "data/source_domain.csv"
    target_csv: str = "data/target_domain.csv"
    output_dir: str = "outputs"

    input_cols: List[str] = field(default_factory=lambda: [
        "Vin", "Vo", "D1", "D2", "DT", "Fs", "Po"
    ])
    component_cols: List[str] = field(default_factory=lambda: [
        "PIron", "PCond", "PCopp", "PSw"
    ])
    total_col: str = "Ploss"

    source_val_ratio: float = 0.2
    target_val_ratio: float = 0.2
    random_seed: int = 42

    batch_size: int = 256
    hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 64])
    dropout: float = 0.1

    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs_pretrain: int = 300
    epochs_finetune: int = 400
    early_stop_patience: int = 40

    lambda_src_components: float = 1.0
    lambda_src_total: float = 0.5
    lambda_tgt_total: float = 1.0

    freeze_backbone_in_finetune: bool = False
    device: str = "cpu"
