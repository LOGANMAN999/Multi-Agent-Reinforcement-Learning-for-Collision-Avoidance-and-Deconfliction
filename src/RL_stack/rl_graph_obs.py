
"""
rl_graph_obs.py

Graph observation builder for RL rollouts, compatible with the GRU student model.

Why this file exists
--------------------
Your GRU student policy expects PyG `Data` objects with node/edge features that match
`build_student_dataset_gru.py` and `student_gnn_controller_gru.py`.

In RL you will be generating those inputs online from `sim_env.MultiRobotEnv` (or your
own env with the same API: positions/goals/walls + lidar_scan_all + segment_blocked_by_walls).

This module centralizes that logic so:
- dataset generation, supervised training, RL rollout, and controller inference all agree
  on feature definitions and ordering;
- you can reuse the same normalization statistics JSON produced by build_student_dataset_gru.py.

Feature conventions
-------------------
Node features (Dx = 2 + 1 + R + 1 + 1 + 1):
  x = [rel_goal(2), dist_goal(1), lidar(R), goal_blocked(1), jump_count(1), active(1)]

Edge features (De = 2 + 1 + 1 + 1):
  e = [dp(2), dist(1), blocked_by_walls(1), active_j(1)]

All geometric vectors are normalized by world_size.
LiDAR distances are normalized by lidar_max_range and clipped to [0,1].

Normalization
-------------
If you provide normalization stats (node_mean/std, edge_mean/std), we z-score the
features exactly like the supervised pipeline.

Typical RL usage
----------------
builder = GraphObsBuilder.from_stats_json(".../stats.json", device="cuda")
data_t, aux = builder.build(env)

Where `data_t` is a torch_geometric.data.Data object ready for StudentGNNGRU.step(...)
and `aux` includes lightweight diagnostics you may want for reward shaping
(e.g., min_lidar per agent).

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from pathlib import Path
import sys

# File is: repo/src/RL_stack/<this_file>.py
_THIS_DIR = Path(__file__).resolve().parent          # repo/src/RL_stack
_SRC_DIR = _THIS_DIR.parent                          # repo/src

# Make `controllers`, `sim_env`, `data_building`, etc. importable
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# (Optional) also allow importing other RL_stack files without package prefix
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def goal_cone_jump_count(
    lidar_ranges: np.ndarray,  # (N,R) raw distances (NOT normalized)
    goal_vec: np.ndarray,      # (N,2) in world units
    *,
    cone_opening_deg: float = 30.0,
    jump_thresh: float = 3.0,
    ray_angles: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Count adjacent LiDAR "range jumps" inside a cone centered on the goal direction.

    This matches the logic used in student_gnn_controller_gru.py / build_student_dataset_gru.py.
    Returns shape (N,1) float32.
    """
    N, R = lidar_ranges.shape
    if ray_angles is None:
        ray_angles = 2.0 * np.pi * (np.arange(R, dtype=np.float32) / float(R))

    theta_g = np.arctan2(goal_vec[:, 1], goal_vec[:, 0]).astype(np.float32)
    theta_g = (theta_g + 2.0 * np.pi) % (2.0 * np.pi)

    half_angle = np.deg2rad(cone_opening_deg * 0.5).astype(np.float32)
    jump_counts = np.zeros((N,), dtype=np.float32)

    goal_norm = np.linalg.norm(goal_vec, axis=1)
    valid = goal_norm > 1e-6

    delta = _wrap_to_pi(ray_angles[None, :] - theta_g[:, None])
    in_cone = np.abs(delta) <= half_angle

    for i in range(N):
        if not valid[i]:
            continue
        idx = np.nonzero(in_cone[i])[0]
        if idx.size < 2:
            continue
        r = lidar_ranges[i, idx]
        diffs = np.abs(np.diff(r))
        jump_counts[i] = float(np.sum(diffs >= jump_thresh))

    return jump_counts.reshape(N, 1).astype(np.float32)


@dataclass
class GraphObsConfig:
    # Geometry / scaling
    world_size: float = 10.0

    # LiDAR
    lidar_n_rays: int = 64
    lidar_max_range: float = 20.0

    # Graph edges
    edge_radius: float = 3.0  # world units
    include_self_edges: bool = False
    directed_edges: bool = True

    # Jump-count feature
    cone_opening_deg: float = 30.0
    jump_thresh: float = 3.0


class GraphObsBuilder:
    """
    Builds PyG Data objects from a live env state.

    You can optionally supply normalization statistics; if present, inputs are z-scored.
    """

    def __init__(
        self,
        cfg: GraphObsConfig,
        *,
        device: str = "cpu",
        node_mean: Optional[torch.Tensor] = None,
        node_std: Optional[torch.Tensor] = None,
        edge_mean: Optional[torch.Tensor] = None,
        edge_std: Optional[torch.Tensor] = None,
    ):
        self.cfg = cfg
        self.device = device

        self.node_mean = node_mean
        self.node_std = node_std
        self.edge_mean = edge_mean
        self.edge_std = edge_std

        self._ray_angles = 2.0 * np.pi * (np.arange(cfg.lidar_n_rays, dtype=np.float32) / float(cfg.lidar_n_rays))

    @classmethod
    def from_stats_json(
        cls,
        stats_json: str | Path,
        *,
        cfg: Optional[GraphObsConfig] = None,
        device: Optional[str] = None,
    ) -> "GraphObsBuilder":
        """
        Load normalization stats produced by build_student_dataset_gru.py.

        stats_json must contain: node_mean, node_std, edge_mean, edge_std.
        """
        import json

        if cfg is None:
            cfg = GraphObsConfig()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        stats_json = Path(stats_json)
        with open(stats_json, "r", encoding="utf-8") as f:
            stats = json.load(f)

        node_mean = torch.tensor(stats["node_mean"], dtype=torch.float32, device=device)
        node_std = torch.tensor(stats["node_std"], dtype=torch.float32, device=device)
        edge_mean = torch.tensor(stats["edge_mean"], dtype=torch.float32, device=device)
        edge_std = torch.tensor(stats["edge_std"], dtype=torch.float32, device=device)

        return cls(cfg, device=device, node_mean=node_mean, node_std=node_std, edge_mean=edge_mean, edge_std=edge_std)

    def build(self, env: Any) -> Tuple[Data, Dict[str, torch.Tensor]]:
        """
        Build a normalized PyG Data graph for the current env state.

        Returns:
            data: torch_geometric.data.Data with x, edge_index, edge_attr, mask
            aux:  dict of per-agent tensors useful for reward shaping / logging
        """
        pos = np.asarray(env.positions, dtype=np.float32)  # (N,2)
        goals = np.asarray(env.goals, dtype=np.float32)    # (N,2)
        N = int(pos.shape[0])

        # active mask
        d_goal = np.linalg.norm(pos - goals, axis=1)
        active = d_goal > float(getattr(env, "goal_tolerance", 0.3))
        active_f = active.astype(np.float32).reshape(N, 1)

        # ----- node features -----
        rel_goal = (goals - pos) / float(self.cfg.world_size)
        dist_goal = np.linalg.norm(rel_goal, axis=1, keepdims=True)

        # LiDAR: use env.lidar_scan_all if available, else per-agent lidar_scan
        if hasattr(env, "lidar_scan_all"):
            lidar_raw = env.lidar_scan_all(n_rays=self.cfg.lidar_n_rays, max_range=self.cfg.lidar_max_range).astype(np.float32)
        else:
            lidar_raw = np.stack([env.lidar_scan(pos[i], n_rays=self.cfg.lidar_n_rays, max_range=self.cfg.lidar_max_range) for i in range(N)], axis=0).astype(np.float32)

        lidar = np.clip(lidar_raw / float(self.cfg.lidar_max_range), 0.0, 1.0)

        # goal_blocked: segment from agent to its goal intersects any wall
        goal_blocked = np.zeros((N, 1), dtype=np.float32)
        if hasattr(env, "segment_blocked_by_walls"):
            for i in range(N):
                goal_blocked[i, 0] = 1.0 if env.segment_blocked_by_walls(pos[i], goals[i]) else 0.0

        # jump_count in goal cone (uses raw ranges)
        jump_count = goal_cone_jump_count(
            lidar_ranges=lidar_raw,
            goal_vec=(goals - pos),
            cone_opening_deg=self.cfg.cone_opening_deg,
            jump_thresh=self.cfg.jump_thresh,
            ray_angles=self._ray_angles,
        )

        x = np.concatenate([rel_goal, dist_goal, lidar, goal_blocked, jump_count, active_f], axis=1).astype(np.float32)

        x_t = torch.from_numpy(x).to(self.device)

        if self.node_mean is not None and self.node_std is not None:
            x_t = (x_t - self.node_mean) / (self.node_std + 1e-12)

        # ----- edges (radius graph) -----
        src, dst = self._radius_edges(pos, radius=float(self.cfg.edge_radius), include_self=self.cfg.include_self_edges)
        if src.size == 0:
            edge_index_t = torch.empty((2, 0), dtype=torch.long, device=self.device)
            # infer edge_dim if possible
            edge_dim = int(self.edge_mean.numel()) if self.edge_mean is not None else 5
            edge_attr_t = torch.empty((0, edge_dim), dtype=torch.float32, device=self.device)
        else:
            dp = (pos[dst] - pos[src]) / float(self.cfg.world_size)
            dist = np.linalg.norm(dp, axis=1, keepdims=True).astype(np.float32)

            blocked = np.zeros((src.shape[0], 1), dtype=np.float32)
            if hasattr(env, "segment_blocked_by_walls"):
                for k in range(src.shape[0]):
                    blocked[k, 0] = 1.0 if env.segment_blocked_by_walls(pos[src[k]], pos[dst[k]]) else 0.0
            active_j = active[dst].astype(np.float32).reshape(-1, 1)

            e = np.concatenate([dp.astype(np.float32), dist, blocked, active_j], axis=1).astype(np.float32)
            e_t = torch.from_numpy(e).to(self.device)

            if self.edge_mean is not None and self.edge_std is not None:
                e_t = (e_t - self.edge_mean) / (self.edge_std + 1e-12)

            edge_index_t = torch.from_numpy(np.stack([src, dst], axis=0)).long().to(self.device)
            edge_attr_t = e_t

        mask_t = torch.from_numpy(active.astype(np.bool_)).to(self.device)

        data = Data(x=x_t, edge_index=edge_index_t, edge_attr=edge_attr_t, mask=mask_t)

        aux: Dict[str, torch.Tensor] = {
            "active": mask_t,
            "dist_to_goal": torch.from_numpy(d_goal.astype(np.float32)).to(self.device),
            "min_lidar": torch.from_numpy(np.min(lidar_raw, axis=1).astype(np.float32)).to(self.device),
            "lidar_raw": torch.from_numpy(lidar_raw).to(self.device),
        }
        return data, aux

    @staticmethod
    def _radius_edges(pos: np.ndarray, radius: float, include_self: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Directed radius graph (src->dst)."""
        N = pos.shape[0]
        diff = pos[:, None, :] - pos[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        mask = d2 <= (radius * radius)
        if not include_self:
            np.fill_diagonal(mask, False)
        src, dst = np.where(mask)
        return src.astype(np.int64), dst.astype(np.int64)
