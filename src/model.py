from __future__ import annotations

from typing import List, Dict

import torch
import torch.nn as nn


class MLPBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ComponentSumModel(nn.Module):
    def __init__(self, in_dim: int, num_components: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        self.backbone = MLPBackbone(in_dim=in_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.head = nn.Linear(self.backbone.out_dim, num_components)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        comp_raw = self.head(feat)
        comp_pred = self.softplus(comp_raw)
        total_pred = comp_pred.sum(dim=1, keepdim=True)
        return {
            "components": comp_pred,
            "total": total_pred,
        }

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False


class TotalBaselineModel(nn.Module):
    """A baseline model that predicts total loss directly."""

    def __init__(self, in_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        self.backbone = MLPBackbone(in_dim=in_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.head = nn.Linear(self.backbone.out_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        total_pred = self.head(feat)
        return {"total": total_pred}
