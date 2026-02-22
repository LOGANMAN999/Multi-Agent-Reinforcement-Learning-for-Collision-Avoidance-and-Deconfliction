"""Gated MPNN student policy (PyTorch Geometric).

This module defines a node-level regression GNN for multi-agent navigation.
It consumes a PyG Data/Batch object with fields:
  - x: (num_nodes, d_x) node features
  - edge_index: (2, num_edges)
  - edge_attr: (num_edges, d_e) edge features

It outputs:
  - actions: (num_nodes, action_dim) where action_dim=2 by default

Design:
  - Node encoder MLP
  - 3x Gated message passing layers (mean aggregation)
  - Residual updates + LayerNorm
  - Output head with tanh squashing and max_speed scaling

Notes:
  - This file only defines the model. Training / data loading handled elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import MessagePassing
except Exception as e:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required for this model. Install PyTorch Geometric and dependencies."
    ) from e


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, *, num_layers: int = 2, dropout: float = 0.0) -> nn.Sequential:
    """Simple MLP with ReLU and optional dropout."""
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    layers: list[nn.Module] = []
    if num_layers == 1:
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    layers.append(nn.Linear(in_dim, hidden_dim))
    layers.append(nn.ReLU())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class GatedMPNNLayer(MessagePassing):
    """A gated message passing layer.

    Message for edge i<-j:
      edge_emb = EdgeMLP(edge_attr)
      gate = sigmoid(GateMLP(edge_attr))   # scalar gate per edge
      msg = gate * (W_h h_j + edge_emb)

    Aggregate by mean over neighbors.

    Update with residual MLP:
      h_{new} = LayerNorm(h + UpdateMLP([h, agg]))

    This design makes it easy for the model to downweight occluded edges via the
    `blocked` feature, while still allowing those edges to exist.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(aggr="mean")
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        # Edge embedding used inside the message.
        self.edge_mlp = _mlp(edge_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)

        # Scalar gate per edge.
        self.gate_mlp = _mlp(edge_dim, hidden_dim, 1, num_layers=2, dropout=dropout)

        # Transform neighbor node embedding.
        self.neigh_lin = nn.Linear(hidden_dim, hidden_dim)

        # Update MLP after aggregation.
        self.update_mlp = _mlp(2 * hidden_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = float(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # propagate returns aggregated messages per node
        agg = self.propagate(edge_index=edge_index, x=h, edge_attr=edge_attr)
        upd = self.update_mlp(torch.cat([h, agg], dim=-1))
        if self.dropout > 0:
            upd = F.dropout(upd, p=self.dropout, training=self.training)
        h_new = self.norm(h + upd)
        return h_new

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        edge_emb = self.edge_mlp(edge_attr)  # (E, H)
        gate = torch.sigmoid(self.gate_mlp(edge_attr))  # (E, 1)
        msg = gate * (self.neigh_lin(x_j) + edge_emb)
        return msg


@dataclass
class StudentGNNConfig:
    node_dim: int
    edge_dim: int
    hidden_dim: int = 128
    num_layers: int = 3
    action_dim: int = 2
    dropout: float = 0.0
    max_speed: float = 1.5  # should match preprocessing / env action scaling


class StudentGNN(nn.Module):
    """Gated MPNN policy producing per-node continuous actions."""

    def __init__(self, cfg: StudentGNNConfig) -> None:
        super().__init__()
        if cfg.num_layers != 3:
            # User requested 3 layers; keep config flexible but guard accidental mismatch.
            raise ValueError(f"Expected num_layers=3 for baseline, got {cfg.num_layers}")

        self.cfg = cfg

        self.node_encoder = _mlp(cfg.node_dim, cfg.hidden_dim, cfg.hidden_dim, num_layers=2, dropout=cfg.dropout)

        self.layers = nn.ModuleList(
            [GatedMPNNLayer(cfg.hidden_dim, cfg.edge_dim, dropout=cfg.dropout) for _ in range(cfg.num_layers)]
        )

        self.head = _mlp(cfg.hidden_dim, cfg.hidden_dim, cfg.action_dim, num_layers=2, dropout=cfg.dropout)

        # Optional: small initial output scale can help early stability.
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: torch_geometric.data.Data or Batch with x, edge_index, edge_attr.

        Returns:
            actions: (num_nodes, action_dim)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.node_encoder(x)
        for layer in self.layers:
            h = layer(h, edge_index=edge_index, edge_attr=edge_attr)

        raw = self.head(h) * self.output_scale
        # Bound actions for stable rollout; training target should be normalized consistently.
        actions = self.cfg.max_speed * torch.tanh(raw)
        return actions


def load_cfg_from_stats(stats_json_path: str | Path, *, node_dim: int, edge_dim: int, hidden_dim: int = 128) -> StudentGNNConfig:
    """Convenience: load constants like max_speed from stats.json.

    Expects stats.json to contain a key 'max_speed' if you saved it during preprocessing.
    """
    p = Path(stats_json_path)
    with p.open("r", encoding="utf-8") as f:
        d = json.load(f)

    max_speed = float(d.get("max_speed", 1.5))
    return StudentGNNConfig(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        action_dim=2,
        dropout=0.0,
        max_speed=max_speed,
    )
