"""Gated MPNN + per-agent GRU student policy.

Baseline (`gnn_student_model.py`) maps a single PyG graph (agents as nodes) to
per-agent actions. This variant adds memory via a GRUCell per agent.

Provides:
- step(data_t, h_prev=None, mask=None) -> (actions, h_next)
- forward(data, ...) alias for step(...)[0]
- forward_sequence(x, edge_index, edge_attr, mask=None, h0=None) -> (actions_seq, h_last)

Sequence training expects:
  x: (B, L, N, Dx)
  mask: (B, L, N) optional but recommended (1=active)
  edge_index[t][b]: (2, E_{b,t}) list-of-lists
  edge_attr[t][b]:  (E_{b,t}, De) list-of-lists

You should mask the loss by `mask` and (optionally) reset hidden states when
mask==0. This model resets hidden states to zero for inactive agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import json
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
    """Same gated message passing layer as the baseline model."""

    def __init__(self, hidden_dim: int, edge_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__(aggr="mean")
        self.edge_mlp = _mlp(edge_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.gate_mlp = _mlp(edge_dim, hidden_dim, 1, num_layers=2, dropout=dropout)
        self.neigh_lin = nn.Linear(hidden_dim, hidden_dim)
        self.update_mlp = _mlp(2 * hidden_dim, hidden_dim, hidden_dim, num_layers=2, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = float(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        agg = self.propagate(edge_index=edge_index, x=h, edge_attr=edge_attr)
        upd = self.update_mlp(torch.cat([h, agg], dim=-1))
        if self.dropout > 0:
            upd = F.dropout(upd, p=self.dropout, training=self.training)
        return self.norm(h + upd)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        edge_emb = self.edge_mlp(edge_attr)
        gate = torch.sigmoid(self.gate_mlp(edge_attr))
        return gate * (self.neigh_lin(x_j) + edge_emb)


@dataclass
class StudentGNNGRUConfig:
    node_dim: int
    edge_dim: int
    gnn_hidden_dim: int = 128
    num_layers: int = 3
    gru_hidden_dim: int = 128
    action_dim: int = 2
    dropout: float = 0.0
    max_speed: float = 1.5


class StudentGNNGRU(nn.Module):
    """Gated MPNN encoder + per-agent GRUCell + action head."""

    def __init__(self, cfg: StudentGNNGRUConfig) -> None:
        super().__init__()
        if cfg.num_layers != 3:
            raise ValueError(f"Expected num_layers=3 for baseline parity, got {cfg.num_layers}")

        self.cfg = cfg

        self.node_encoder = _mlp(cfg.node_dim, cfg.gnn_hidden_dim, cfg.gnn_hidden_dim, num_layers=2, dropout=cfg.dropout)
        self.layers = nn.ModuleList(
            [GatedMPNNLayer(cfg.gnn_hidden_dim, cfg.edge_dim, dropout=cfg.dropout) for _ in range(cfg.num_layers)]
        )

        # If GRU hidden size differs from GNN size, project GNN outputs.
        if cfg.gru_hidden_dim != cfg.gnn_hidden_dim:
            self.to_gru = nn.Linear(cfg.gnn_hidden_dim, cfg.gru_hidden_dim)
        else:
            self.to_gru = nn.Identity()

        self.gru_cell = nn.GRUCell(input_size=cfg.gru_hidden_dim, hidden_size=cfg.gru_hidden_dim)
        self.head = _mlp(cfg.gru_hidden_dim, cfg.gru_hidden_dim, cfg.action_dim, num_layers=2, dropout=cfg.dropout)

        # Optional: small initial output scale can help early stability.
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    # ----------------------- core computations -----------------------
    def _encode_graph(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Encode node features with GNN, returning node embeddings (num_nodes, H)."""
        h = self.node_encoder(x)
        for layer in self.layers:
            h = layer(h, edge_index=edge_index, edge_attr=edge_attr)
        h = self.to_gru(h)
        return h

    def _apply_mask(self, t: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return t
        # Accept bool or float masks.
        m = mask.to(dtype=t.dtype)
        while m.dim() < t.dim():
            m = m.unsqueeze(-1)
        return t * m

    # ----------------------- public API -----------------------
    @torch.no_grad()
    def init_hidden(self, num_nodes: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Convenience: initialize hidden state to zeros."""
        dev = device if device is not None else next(self.parameters()).device
        return torch.zeros(num_nodes, self.cfg.gru_hidden_dim, device=dev)

    def step(
        self,
        data,
        *,
        h_prev: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One-step inference/training on a single PyG Data/Batch.

        Args:
            data: PyG Data/Batch with x, edge_index, edge_attr.
            h_prev: (num_nodes, H_gru) previous hidden state. If None, zeros.
            mask: (num_nodes,) optional 0/1 active mask. If provided, hidden
                  state is reset to zero where mask==0.

        Returns:
            actions: (num_nodes, action_dim)
            h_next:  (num_nodes, H_gru)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        z = self._encode_graph(x, edge_index, edge_attr)  # (num_nodes, H_gru)

        if h_prev is None:
            h_prev = torch.zeros(z.size(0), self.cfg.gru_hidden_dim, device=z.device, dtype=z.dtype)

        # If masked, we do not update inactive agents and we reset their state to 0.
        if mask is not None:
            mask_f = mask.to(dtype=z.dtype)
            z = z * mask_f.unsqueeze(-1)
            h_prev = h_prev * mask_f.unsqueeze(-1)

        h_next = self.gru_cell(z, h_prev)

        if mask is not None:
            # Keep inactive agents at zero (prevents stale memory).
            h_next = h_next * mask_f.unsqueeze(-1)

        raw = self.head(h_next) * self.output_scale
        actions = self.cfg.max_speed * torch.tanh(raw)
        return actions, h_next

    def forward(self, data, *, h_prev: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compatibility forward: returns actions only."""
        actions, _ = self.step(data, h_prev=h_prev, mask=mask)
        return actions

    # ----------------------- sequence training helper -----------------------
    @staticmethod
    def _batch_edges(
        edge_index_b: Sequence[torch.Tensor],
        edge_attr_b: Sequence[torch.Tensor],
        n_pad: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a block-diagonal batch of edges.

        edge_index_b/edge_attr_b are sequences of length B. Each edge_index_b[b]
        is (2, E_b) and refers to nodes in [0, n_pad).

        Returns concatenated (edge_index, edge_attr) for a batch graph with
        B*n_pad nodes.
        """
        ei_all: List[torch.Tensor] = []
        ea_all: List[torch.Tensor] = []
        offset = 0
        for ei, ea in zip(edge_index_b, edge_attr_b):
            if ei.numel() == 0:
                offset += n_pad
                continue
            ei_all.append(ei.to(device) + offset)
            ea_all.append(ea.to(device))
            offset += n_pad

        if len(ei_all) == 0:
            # No edges anywhere.
            return (
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0, edge_attr_b[0].size(-1)), dtype=torch.float32, device=device)
                if len(edge_attr_b) > 0 and edge_attr_b[0].dim() == 2
                else torch.empty((0, 0), dtype=torch.float32, device=device),
            )

        edge_index = torch.cat(ei_all, dim=1)
        edge_attr = torch.cat(ea_all, dim=0)
        return edge_index, edge_attr

    def forward_sequence(
        self,
        x: torch.Tensor,
        edge_index: Sequence[Sequence[torch.Tensor]],
        edge_attr: Sequence[Sequence[torch.Tensor]],
        *,
        mask: Optional[torch.Tensor] = None,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a batch of fixed-length sequences.

        Args:
            x: (B, L, N, Dx)
            edge_index: list length L, each item is list length B of (2,E) tensors
            edge_attr:  list length L, each item is list length B of (E,De) tensors
            mask: (B, L, N) optional 0/1 active mask
            h0: (B, N, H_gru) optional initial hidden state

        Returns:
            actions_seq: (B, L, N, action_dim)
            h_last:      (B, N, H_gru)
        """
        if x.dim() != 4:
            raise ValueError(f"Expected x shape (B,L,N,Dx), got {tuple(x.shape)}")
        B, L, N, Dx = x.shape
        if len(edge_index) != L or len(edge_attr) != L:
            raise ValueError("edge_index and edge_attr must be length L")

        device = x.device

        if h0 is None:
            h = torch.zeros((B, N, self.cfg.gru_hidden_dim), device=device, dtype=x.dtype)
        else:
            if h0.shape != (B, N, self.cfg.gru_hidden_dim):
                raise ValueError(f"Expected h0 shape {(B,N,self.cfg.gru_hidden_dim)}, got {tuple(h0.shape)}")
            h = h0

        actions_out = torch.zeros((B, L, N, self.cfg.action_dim), device=device, dtype=x.dtype)

        # Unroll over time.
        for t in range(L):
            x_t = x[:, t]  # (B,N,Dx)
            x_nodes = x_t.reshape(B * N, Dx)

            # Build block-diagonal edges for this timestep.
            ei_t, ea_t = self._batch_edges(edge_index[t], edge_attr[t], N, device=device)

            z = self._encode_graph(x_nodes, ei_t, ea_t)  # (B*N, H_gru)
            z = z.reshape(B, N, self.cfg.gru_hidden_dim)

            if mask is not None:
                m = mask[:, t].to(dtype=x.dtype)  # (B,N)
                z = z * m.unsqueeze(-1)
                h = h * m.unsqueeze(-1)  # reset inactive before update
            else:
                m = None

            # GRU update per agent: flatten (B*N, H)
            h_flat = h.reshape(B * N, self.cfg.gru_hidden_dim)
            z_flat = z.reshape(B * N, self.cfg.gru_hidden_dim)
            h_next = self.gru_cell(z_flat, h_flat).reshape(B, N, self.cfg.gru_hidden_dim)

            if m is not None:
                h_next = h_next * m.unsqueeze(-1)  # keep inactive at zero

            h = h_next

            raw = self.head(h.reshape(B * N, self.cfg.gru_hidden_dim)) * self.output_scale
            act = (self.cfg.max_speed * torch.tanh(raw)).reshape(B, N, self.cfg.action_dim)
            actions_out[:, t] = act

        return actions_out, h


def load_cfg_from_stats(
    stats_json_path: str | Path,
    *,
    node_dim: int,
    edge_dim: int,
    gnn_hidden_dim: int = 128,
    gru_hidden_dim: int = 128,
) -> StudentGNNGRUConfig:
    """Convenience: load constants like max_speed from stats.json."""
    p = Path(stats_json_path)
    with p.open("r", encoding="utf-8") as f:
        d = json.load(f)
    max_speed = float(d.get("max_speed", 1.5))
    return StudentGNNGRUConfig(
        node_dim=node_dim,
        edge_dim=edge_dim,
        gnn_hidden_dim=gnn_hidden_dim,
        num_layers=3,
        gru_hidden_dim=gru_hidden_dim,
        action_dim=2,
        dropout=0.0,
        max_speed=max_speed,
    )
