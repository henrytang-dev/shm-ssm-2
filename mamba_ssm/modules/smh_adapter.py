import math
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _is_enabled(cfg: Optional[Dict]) -> bool:
    if cfg is None:
        return False
    enabled = cfg.get("enabled", True)
    return bool(enabled)


class SaliencyGate(nn.Module):
    """Maps scalar saliency scores to token-wise gates in [0, 1]."""

    def __init__(
        self,
        d_model: int,
        hidden_dims: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
        clamp_range: Optional[Sequence[float]] = None,
        train_only: bool = True,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims or [])
        layers = []
        in_dim = 1
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, d_model))
        self.proj = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        if clamp_range is not None and len(clamp_range) == 2:
            self.register_buffer("_clamp_min", torch.tensor(float(clamp_range[0])), persistent=False)
            self.register_buffer("_clamp_max", torch.tensor(float(clamp_range[1])), persistent=False)
        else:
            self._clamp_min = None
            self._clamp_max = None
        self.train_only = train_only

    def forward(self, hidden_states: torch.Tensor, saliency_scores: Optional[torch.Tensor]) -> torch.Tensor:
        if saliency_scores is None:
            return hidden_states
        if self.train_only and not self.training:
            return hidden_states
        if saliency_scores.dim() == 2:
            saliency_scores = saliency_scores.unsqueeze(-1)
        if saliency_scores.size(-1) != 1:
            raise ValueError("saliency_scores must have a trailing dimension of size 1.")
        scores = saliency_scores.to(dtype=hidden_states.dtype)
        gates = torch.sigmoid(self.proj(scores))
        if self.dropout is not None:
            gates = self.dropout(gates)
        if self._clamp_min is not None and self._clamp_max is not None:
            gates = torch.clamp(gates, min=float(self._clamp_min), max=float(self._clamp_max))
        return hidden_states * gates


class SelectiveMemory(nn.Module):
    """Local window attention that compresses nearby states then re-injects them."""

    def __init__(
        self,
        d_model: int,
        window_sizes: Iterable[int],
        memory_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        sizes = sorted({int(max(1, w)) for w in window_sizes})
        if not sizes:
            raise ValueError("window_sizes must contain at least one positive integer.")
        self.window_sizes = tuple(sizes)
        self.memory_dim = memory_dim or d_model
        self.scale = 1.0 / math.sqrt(self.memory_dim)
        self.q_proj = nn.Linear(d_model, self.memory_dim)
        self.k_proj = nn.Linear(d_model, self.memory_dim)
        self.v_proj = nn.Linear(d_model, self.memory_dim)
        self.out_proj = nn.Linear(self.memory_dim, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must be a (batch, seqlen, dim) tensor.")
        contexts = []
        queries = self.q_proj(hidden_states)
        for window in self.window_sizes:
            patches = self._gather_patches(hidden_states, window)  # (B, L, W, D)
            if patches.size(-1) != hidden_states.size(-1):
                # Ensure the last dimension corresponds to the feature dim
                patches = patches.transpose(-1, -2)
            keys = self.k_proj(patches)
            values = self.v_proj(patches)
            attn_scores = torch.einsum("bld,blwd->blw", queries, keys) * self.scale
            attn = torch.softmax(attn_scores, dim=-1)
            if self.dropout is not None:
                attn = self.dropout(attn)
            context = torch.einsum("blw,blwd->bld", attn, values)
            contexts.append(context)
        merged = torch.stack(contexts, dim=0).mean(0) if len(contexts) > 1 else contexts[0]
        return hidden_states + self.out_proj(merged)

    @staticmethod
    def _gather_patches(hidden_states: torch.Tensor, window: int) -> torch.Tensor:
        if window <= 1:
            return hidden_states.unsqueeze(-2)
        seq_len = hidden_states.size(1)
        window = min(window, seq_len)
        pad = window - 1
        padded = F.pad(hidden_states, (0, 0, pad, 0))
        patches = padded.unfold(dimension=1, size=window, step=1)
        if patches.size(1) > seq_len:
            patches = patches[:, -seq_len:]
        return patches.contiguous()


class SMHAdapter(nn.Module):
    """Combines saliency gating and selective memory as lightweight adapters."""

    def __init__(
        self,
        d_model: int,
        saliency_gate: Optional[Dict] = None,
        selective_memory: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.saliency_gate = (
            SaliencyGate(d_model, **{k: v for k, v in (saliency_gate or {}).items() if k != "enabled"})
            if _is_enabled(saliency_gate)
            else None
        )
        if _is_enabled(selective_memory):
            selective_kwargs = {k: v for k, v in (selective_memory or {}).items() if k != "enabled"}
            selective_kwargs.setdefault("window_sizes", (8,))
            self.selective_memory = SelectiveMemory(d_model, **selective_kwargs)
        else:
            self.selective_memory = None

    def forward(self, hidden_states: torch.Tensor, saliency_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.saliency_gate is not None:
            hidden_states = self.saliency_gate(hidden_states, saliency_scores=saliency_scores)
        if self.selective_memory is not None:
            hidden_states = self.selective_memory(hidden_states)
        return hidden_states

    @property
    def is_active(self) -> bool:
        return (self.saliency_gate is not None) or (self.selective_memory is not None)
