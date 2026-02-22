# src/controllers/student_gnn_controller.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json
import numpy as np
import torch

from torch_geometric.data import Data

from gnn_student_model import StudentGNN, StudentGNNConfig


@dataclass
class StudentPolicyPaths:
    stats_json: Path
    checkpoint_pt: Path


class StudentGNNController:
    """
    Controller wrapper around the trained PyG student model.

    It builds the same node/edge features you used in preprocessing:
      node x = [rel_goal(2), dist_goal(1), lidar(64), active(1)] then z-score
      edge e = [dp(2), dist(1), blocked(1), active_j(1)] then z-score

    IMPORTANT: Your training targets y were normalized by max_speed (y = actions/max_speed).
    Your StudentGNN model currently outputs ~that normalized action (because it learns tanh(raw)=y/1.5).
    So for environment stepping (physical velocities), we multiply by env.max_speed.
    """

    def __init__(
        self,
        paths: StudentPolicyPaths,
        *,
        device: Optional[str] = None,
        edge_radius: float = 3.0,          # MUST match dataset_generator.py cfg.edge_radius
        lidar_n_rays: int = 64,            # MUST match preprocessing
        lidar_max_range: float = 8.0,      # MUST match preprocessing (or meta)
        world_size: float = 10.0,          # MUST match preprocessing (or meta)
        max_speed: float = 1.5,            # env max_speed (for converting normalized -> physical)
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

        # Load stats
        with open(paths.stats_json, "r", encoding="utf-8") as f:
            stats = json.load(f)

        self.node_mean = torch.tensor(stats["node_mean"], dtype=torch.float32, device=self.device)
        self.node_std  = torch.tensor(stats["node_std"],  dtype=torch.float32, device=self.device)
        self.edge_mean = torch.tensor(stats["edge_mean"], dtype=torch.float32, device=self.device)
        self.edge_std  = torch.tensor(stats["edge_std"],  dtype=torch.float32, device=self.device)

        node_dim = int(self.node_mean.numel())
        edge_dim = int(self.edge_mean.numel())

        # Build model + load checkpoint
        cfg = StudentGNNConfig(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=128, num_layers=3, max_speed=1.5)
        self.model = StudentGNN(cfg).to(self.device)

        ckpt = torch.load(paths.checkpoint_pt, map_location=self.device)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, env) -> np.ndarray:
        """
        env must have:
          - env.positions (N,2)
          - env.goals (N,2)
          - env.goal_tolerance
          - env.lidar_scan_all(n_rays, max_range)
          - env.segment_blocked_by_walls(p, q)
        Returns: actions (N,2) in *physical* units to pass into env.step
        """
        pos = np.asarray(env.positions, dtype=np.float32)
        goals = np.asarray(env.goals, dtype=np.float32)
        N = pos.shape[0]

        # active mask (match dataset)
        d_goal = np.linalg.norm(pos - goals, axis=1)
        active = d_goal > float(env.goal_tolerance)

        # node features

        """
        rel_goal = (goals - pos) / self.world_size                 # (N,2)
        dist_goal = np.linalg.norm(rel_goal, axis=1, keepdims=True) # (N,1)
        lidar = env.lidar_scan_all(n_rays=self.lidar_n_rays, max_range=self.lidar_max_range).astype(np.float32)
        lidar = np.clip(lidar / self.lidar_max_range, 0.0, 1.0)     # (N,R)
        active_f = active.astype(np.float32).reshape(N, 1)          # (N,1)

        x = np.concatenate([rel_goal, dist_goal, lidar, active_f], axis=1).astype(np.float32)
        """

                # node features (MUST match training order):
        # [rel_goal(2), dist_goal(1), lidar_norm(R), goal_blocked(1), jump_count(1), active(1)]

        rel_goal_vec = (goals - pos)                     # (N,2) raw
        rel_goal = rel_goal_vec / self.world_size        # (N,2)
        dist_goal = np.linalg.norm(rel_goal, axis=1, keepdims=True)  # (N,1)

        # Lidar: keep RAW for jump_count, and normalized for x
        lidar_raw = env.lidar_scan_all(
            n_rays=self.lidar_n_rays,
            max_range=self.lidar_max_range
        ).astype(np.float32)                             # (N,R) raw distances
        lidar_norm = np.clip(lidar_raw / self.lidar_max_range, 0.0, 1.0)

        # goal blocked flag: segment agent->goal intersects walls?
        goal_blocked = np.zeros((N, 1), dtype=np.float32)
        for i in range(N):
            goal_blocked[i, 0] = 1.0 if env.segment_blocked_by_walls(pos[i], goals[i]) else 0.0

        # jump count in 30° cone (total opening), threshold >= 3.0
        jump_count = self._goal_cone_jump_count(
            lidar_raw=lidar_raw,
            goal_vec=rel_goal_vec,
            cone_opening_deg=30.0,
            jump_thresh=3.0,
        ).astype(np.float32)                             # (N,1)

        active_f = active.astype(np.float32).reshape(N, 1)

        x = np.concatenate(
            [rel_goal, dist_goal, lidar_norm, goal_blocked, jump_count, active_f],
            axis=1
        ).astype(np.float32)

        x_t = torch.from_numpy(x).to(self.device)
        x_t = (x_t - self.node_mean) / self.node_std

        # edges: radius graph over positions
        src, dst = self._radius_edges(pos, self.edge_radius)

        if src.size == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_attr  = torch.empty((0, int(self.edge_mean.numel())), dtype=torch.float32, device=self.device)
        else:
            dp = (pos[dst] - pos[src]) / self.world_size                      # (E,2)
            dist = np.linalg.norm(dp, axis=1, keepdims=True).astype(np.float32)  # (E,1)

            # blocked + active_j
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

        # Model output is ~normalized action (because y was actions/max_speed).
        a_norm = self.model(data)                 # (N,2) ~ in [-1,1]
        a = a_norm * self.max_speed               # convert to physical units for env.step

        # If inactive, do nothing
        if not np.all(active):
            a[~torch.from_numpy(active).to(self.device)] = 0.0

        return a.detach().cpu().numpy().astype(np.float32)
    

    @staticmethod
    def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    def _goal_cone_jump_count(
        self,
        lidar_raw: np.ndarray,   # (N,R) raw ranges in same units as map
        goal_vec: np.ndarray,    # (N,2) goals - pos
        *,
        cone_opening_deg: float = 30.0,  # total opening (±15°)
        jump_thresh: float = 3.0,
    ) -> np.ndarray:
        """
        Count adjacent lidar discontinuities within a cone around the goal direction:
            count of i where |r[i+1] - r[i]| >= jump_thresh
        Returns shape (N,1) float32.
        """
        N, R = lidar_raw.shape
        ray_angles = 2.0 * np.pi * (np.arange(R, dtype=np.float32) / float(R))  # [0,2π)

        theta_g = np.arctan2(goal_vec[:, 1], goal_vec[:, 0]).astype(np.float32)
        theta_g = (theta_g + 2.0 * np.pi) % (2.0 * np.pi)

        half_angle = float(np.deg2rad(cone_opening_deg * 0.5))

        delta = self._wrap_to_pi(ray_angles[None, :] - theta_g[:, None])
        in_cone = np.abs(delta) <= half_angle

        jump_counts = np.zeros((N,), dtype=np.float32)
        valid = (np.linalg.norm(goal_vec, axis=1) > 1e-6)

        for i in range(N):
            if not valid[i]:
                continue
            idx = np.nonzero(in_cone[i])[0]
            if idx.size < 2:
                continue
            r = lidar_raw[i, idx]
            diffs = np.abs(np.diff(r))
            jump_counts[i] = float(np.sum(diffs >= jump_thresh))

        return jump_counts.reshape(N, 1)


    @staticmethod
    def _radius_edges(pos: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """Directed radius graph, excluding self edges."""
        N = pos.shape[0]
        diff = pos[:, None, :] - pos[None, :, :]          # (N,N,2)
        d2 = np.sum(diff * diff, axis=-1)                 # (N,N)
        mask = (d2 <= radius * radius)
        np.fill_diagonal(mask, False)
        src, dst = np.where(mask)
        return src.astype(np.int64), dst.astype(np.int64)
