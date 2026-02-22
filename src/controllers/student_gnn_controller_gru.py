# src/controllers/student_gnn_controller_gru.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json
import numpy as np
import torch

from torch_geometric.data import Data

from gnn_student_model_gru import StudentGNNGRU, StudentGNNGRUConfig


@dataclass
class StudentPolicyPaths:
    stats_json: Path
    checkpoint_pt: Path


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _goal_cone_jump_count(
    lidar_ranges: np.ndarray,  # (N,R) raw distances
    goal_vec: np.ndarray,      # (N,2)
    *,
    cone_opening_deg: float = 30.0,
    jump_thresh: float = 3.0,
    ray_angles: np.ndarray | None = None,
) -> np.ndarray:
    """Count adjacent lidar range jumps inside a cone centered on the goal direction."""
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

    return jump_counts.reshape(N, 1)


class StudentGNNGRUController:
    """Stateful controller wrapper (keeps a per-agent GRU hidden state).

    Node features must match build_student_dataset_gru.py:
      x = [rel_goal(2), dist_goal(1), lidar(64), goal_blocked(1), jump_count(1), active(1)]
    Edge features:
      e = [dp(2), dist(1), blocked(1), active_j(1)]

    The GRU model is trained to output *normalized* actions (targets are actions/max_speed),
    so we multiply by env.max_speed before returning.
    """

    def __init__(
        self,
        paths: StudentPolicyPaths,
        *,
        device: Optional[str] = None,
        edge_radius: float = 3.0,
        lidar_n_rays: int = 64,
        lidar_max_range: float = 20.0,
        world_size: float = 10.0,
        max_speed: float = 1.5,
        gnn_hidden_dim: int = 128,
        gru_hidden_dim: int = 128,
    ):
        self.paths = paths
        self.edge_radius = float(edge_radius)
        self.lidar_n_rays = int(lidar_n_rays)
        self.lidar_max_range = float(lidar_max_range)
        self.world_size = float(world_size)
        self.max_speed = float(max_speed)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        with open(paths.stats_json, "r", encoding="utf-8") as f:
            stats = json.load(f)

        # Normalization vectors produced by build_student_dataset*.py
        self.node_mean = torch.tensor(stats["node_mean"], dtype=torch.float32, device=self.device)
        self.node_std = torch.tensor(stats["node_std"], dtype=torch.float32, device=self.device)
        self.edge_mean = torch.tensor(stats["edge_mean"], dtype=torch.float32, device=self.device)
        self.edge_std = torch.tensor(stats["edge_std"], dtype=torch.float32, device=self.device)

        node_dim = int(self.node_mean.numel())
        edge_dim = int(self.edge_mean.numel())

        # IMPORTANT: targets are normalized, so cfg.max_speed=1.0.
        cfg = StudentGNNGRUConfig(
            node_dim=node_dim,
            edge_dim=edge_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            num_layers=3,
            gru_hidden_dim=gru_hidden_dim,
            action_dim=2,
            dropout=0.0,
            max_speed=1.0,
        )
        self.model = StudentGNNGRU(cfg).to(self.device)

        ckpt = torch.load(paths.checkpoint_pt, map_location=self.device)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.eval()

        # Per-agent recurrent state (initialized lazily on first call)
        self._h: Optional[torch.Tensor] = None
        self._last_n: Optional[int] = None

        # Precompute ray angles used by jump_count feature.
        self._ray_angles = 2.0 * np.pi * (np.arange(self.lidar_n_rays, dtype=np.float32) / float(self.lidar_n_rays))

    def reset(self) -> None:
        """Reset GRU memory (call at episode start)."""
        self._h = None
        self._last_n = None

    @torch.no_grad()
    def __call__(self, env) -> np.ndarray:
        pos = np.asarray(env.positions, dtype=np.float32)
        goals = np.asarray(env.goals, dtype=np.float32)
        N = pos.shape[0]

        # active mask
        d_goal = np.linalg.norm(pos - goals, axis=1)
        active = d_goal > float(env.goal_tolerance)
        active_t = torch.from_numpy(active).to(self.device)

        # initialize hidden if needed / if agent count changed
        if self._h is None or self._last_n != N:
            self._h = torch.zeros((N, self.model.cfg.gru_hidden_dim), device=self.device, dtype=torch.float32)
            self._last_n = N

        # ----- node features -----
        rel_goal = (goals - pos) / self.world_size
        dist_goal = np.linalg.norm(rel_goal, axis=1, keepdims=True)

        lidar_raw = env.lidar_scan_all(n_rays=self.lidar_n_rays, max_range=self.lidar_max_range).astype(np.float32)
        lidar = np.clip(lidar_raw / self.lidar_max_range, 0.0, 1.0)

        # goal_blocked: whether straight line to goal intersects any wall segment
        goal_blocked = np.zeros((N, 1), dtype=np.float32)
        if hasattr(env, "segment_blocked_by_walls"):
            for i in range(N):
                goal_blocked[i, 0] = 1.0 if env.segment_blocked_by_walls(pos[i], goals[i]) else 0.0

        # jump_count in goal cone (uses *raw* lidar distances)
        jump_count = _goal_cone_jump_count(
            lidar_ranges=lidar_raw,
            goal_vec=(goals - pos),
            cone_opening_deg=30.0,
            jump_thresh=3.0,
            ray_angles=self._ray_angles,
        ).astype(np.float32)

        active_f = active.astype(np.float32).reshape(N, 1)

        x = np.concatenate([rel_goal, dist_goal, lidar, goal_blocked, jump_count, active_f], axis=1).astype(np.float32)
        x_t = torch.from_numpy(x).to(self.device)
        x_t = (x_t - self.node_mean) / self.node_std

        # ----- edges -----
        src, dst = self._radius_edges(pos, self.edge_radius)
        if src.size == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.empty((0, int(self.edge_mean.numel())), dtype=torch.float32, device=self.device)
        else:
            dp = (pos[dst] - pos[src]) / self.world_size
            dist = np.linalg.norm(dp, axis=1, keepdims=True).astype(np.float32)

            blocked = np.zeros((src.shape[0], 1), dtype=np.float32)
            for k in range(src.shape[0]):
                blocked[k, 0] = 1.0 if env.segment_blocked_by_walls(pos[src[k]], pos[dst[k]]) else 0.0
            active_j = active[dst].astype(np.float32).reshape(-1, 1)

            e = np.concatenate([dp, dist, blocked, active_j], axis=1).astype(np.float32)
            e_t = torch.from_numpy(e).to(self.device)
            e_t = (e_t - self.edge_mean) / self.edge_std

            edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long().to(self.device)
            edge_attr = e_t

        data = Data(x=x_t, edge_index=edge_index, edge_attr=edge_attr)

        # GRU step
        a_norm, h_next = self.model.step(data, h_prev=self._h, mask=active_t)
        self._h = h_next

        # Convert normalized action -> physical units
        a = a_norm * self.max_speed
        a[~active_t] = 0.0
        return a.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _radius_edges(pos: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """Directed radius graph, excluding self edges."""
        N = pos.shape[0]
        diff = pos[:, None, :] - pos[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        mask = (d2 <= radius * radius)
        np.fill_diagonal(mask, False)
        src, dst = np.where(mask)
        return src.astype(np.int64), dst.astype(np.int64)
