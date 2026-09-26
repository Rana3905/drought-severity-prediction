from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


PROJ_SIZE = 24
HIDDEN_SIZE = 64
N_LAYERS = 1
DROPOUT = 0.4
SOIL_PROJ = 8
DEFAULT_SOIL_SIZE = 31


class LSTMAttention(nn.Module):
    """Shared LSTM + additive attention model for the AlfaFlow scenarios.

    The architecture constants intentionally match the Kazakhstan engineered
    feature model: projection 24, hidden size 64, and soil/static projection 8.
    Scenarios without real soil/static features should pass a zero-filled
    placeholder matrix with the same soil_size used at construction time.
    """

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        soil_size: int = DEFAULT_SOIL_SIZE,
        proj_size: int = PROJ_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        n_layers: int = N_LAYERS,
        dropout: float = DROPOUT,
        soil_proj: int = SOIL_PROJ,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.soil_size = soil_size
        self.n_classes = n_classes

        self.proj = nn.Sequential(
            nn.Linear(input_size, proj_size),
            nn.LayerNorm(proj_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(
            proj_size,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attn_w = nn.Linear(hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.soil_proj = nn.Sequential(
            nn.Linear(soil_size, soil_proj),
            nn.LayerNorm(soil_proj),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.fc1 = nn.Linear(hidden_size + soil_proj, 32)
        self.fc2 = nn.Linear(32, n_classes)
        self.relu = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        soil: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, n_features = x.shape
        if n_features != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {n_features}")
        if soil.shape[-1] != self.soil_size:
            raise ValueError(f"Expected {self.soil_size} soil/static features, got {soil.shape[-1]}")

        x_proj = self.proj(x.reshape(batch_size * seq_len, n_features)).reshape(
            batch_size,
            seq_len,
            -1,
        )
        out, _ = self.lstm(x_proj)
        scores = self.attn_v(torch.tanh(self.attn_w(out))).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = (out * weights.unsqueeze(-1)).sum(dim=1)
        soil_context = self.soil_proj(soil)
        combined = torch.cat([self.drop(context), soil_context], dim=1)
        logits = self.fc2(self.relu(self.fc1(combined)))
        return (logits, weights) if return_attn else logits


def make_zero_static(n_samples: int, soil_size: int = DEFAULT_SOIL_SIZE) -> np.ndarray:
    return np.zeros((n_samples, soil_size), dtype=np.float32)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
