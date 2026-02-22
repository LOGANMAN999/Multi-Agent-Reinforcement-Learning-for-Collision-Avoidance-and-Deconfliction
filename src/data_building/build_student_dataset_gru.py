"""build_student_dataset.py

Create PyTorch Geometric training samples from the saved episode *.npz files.

Design goals (per user requirements):
  - Use ALL episode folders under datasets/il_dataset (episodes*, including walls+obstacles, etc.).
  - Remove expert-action leakage from edge features (do NOT use dv or closing speed).
  - Keep radius-based edges as stored in each episode (edge_index), even when line-of-sight is blocked.
    We still add a boolean edge feature `blocked_by_walls` (computed from walls_xyxy).
  - Do NOT replace with velocity estimates from state history.
  - Normalize relevant inputs for stable training.

Outputs:
  datasets/il_dataset/processed_student_v1/
    train/shard_000.pt, shard_001.pt, ...
    val/...
    test/...
    stats.json   (normalization stats)
    manifest.json (what was processed)

Each saved shard is either:
  - (default) a list of torch_geometric.data.Data objects with fields:
  - (when --sequence_mode) a list of python dicts containing fixed-length sequences suitable for GRU training.

Default per-timestep Data fields:
  x:         (N, Dx) float
  edge_index:(2, E)  long
  edge_attr: (E, De) float
  y:         (N, 2)  float   (normalized expert actions)
  mask:      (N,)    bool    (active agents at time t)
  meta: dict-like python object with episode path + t (kept small)

Run:
  python build_student_dataset.py \
    --dataset_root datasets/il_dataset \
    --out_dir datasets/il_dataset/processed_student_v1
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _safe_std(var: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.sqrt(np.maximum(var, 0.0) + eps)


def _segment_intersects_walls_vectorized(
    p1: np.ndarray,  # (E,2)
    p2: np.ndarray,  # (E,2)
    walls_xyxy: np.ndarray,  # (M,4)
) -> np.ndarray:
    """Return blocked flags for each segment (p1[e], p2[e]) against any wall.

    Uses the same ccw-based intersection test as sim_env.segment_blocked_by_walls,
    but vectorized over walls (and edges).

    Output: (E,) bool
    """
    if walls_xyxy.size == 0:
        return np.zeros((p1.shape[0],), dtype=bool)

    a1 = walls_xyxy[:, 0:2]  # (M,2)
    a2 = walls_xyxy[:, 2:4]  # (M,2)

    # Broadcast to (E,M,2)
    P1 = p1[:, None, :]
    P2 = p2[:, None, :]
    A1 = a1[None, :, :]
    A2 = a2[None, :, :]

    # ccw(A,B,C) = (Cy-Ay)*(Bx-Ax) > (By-Ay)*(Cx-Ax)
    def ccw(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
        return (C[..., 1] - A[..., 1]) * (B[..., 0] - A[..., 0]) > (B[..., 1] - A[..., 1]) * (C[..., 0] - A[..., 0])

    c1 = ccw(P1, A1, A2)
    c2 = ccw(P2, A1, A2)
    c3 = ccw(P1, P2, A1)
    c4 = ccw(P1, P2, A2)

    inter = (c1 != c2) & (c3 != c4)  # (E,M)
    return np.any(inter, axis=1)

def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi

def _goal_cone_jump_count(
    lidar_ranges: np.ndarray,  # (N,R) raw distances
    goal_vec: np.ndarray,      # (N,2)
    *,
    cone_opening_deg: float = 30.0,  # total opening (±15°)
    jump_thresh: float = 3.0,
    ray_angles: np.ndarray | None = None,
) -> np.ndarray:
    """
    Count how many times adjacent lidar rays inside the goal cone have
    |r[i+1]-r[i]| >= jump_thresh.
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

    return jump_counts.reshape(N, 1)



@dataclass
class NormalizationStats:
    node_mean: List[float]
    node_std: List[float]
    edge_mean: List[float]
    edge_std: List[float]
    # for reference
    node_feature_names: List[str]
    edge_feature_names: List[str]


def find_episode_dirs(dataset_root: Path) -> List[Path]:
    """Return episode directories to process (all folders beginning with 'episodes')."""
    dirs = []
    for p in sorted(dataset_root.iterdir()):
        if p.is_dir() and p.name.startswith("episodes"):
            dirs.append(p)
    return dirs


def meta_for_episode_dir(dataset_root: Path, episode_dir: Path) -> Dict:
    """Pick the appropriate meta JSON for an episode directory."""
    # episodes_walls+obstacles -> meta_walls+obstacles.json
    # episodes_higher_agents -> meta_higher_agents.json
    # episodes -> meta.json
    suffix = episode_dir.name[len("episodes"):]
    meta_candidates = []
    if suffix:
        # strip leading underscore
        if suffix.startswith("_"):
            suffix = suffix[1:]
        meta_candidates.append(dataset_root / f"meta_{suffix}.json")
    meta_candidates.append(dataset_root / "meta.json")

    for p in meta_candidates:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def list_episode_files(episode_dir: Path) -> List[Path]:
    return sorted(episode_dir.glob("episode_*.npz"))


def split_files(files: List[Path], seed: int, train_frac: float, val_frac: float) -> Tuple[List[Path], List[Path], List[Path]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n = len(files)
    n_train = int(math.floor(train_frac * n))
    n_val = int(math.floor(val_frac * n))
    train = [files[i] for i in idx[:n_train]]
    val = [files[i] for i in idx[n_train:n_train + n_val]]
    test = [files[i] for i in idx[n_train + n_val:]]
    return train, val, test


def compute_stats(
    episode_files: List[Path],
    dataset_root: Path,
    world_scale_default: float = 10.0,
    max_speed: float = 1.5,
) -> NormalizationStats:
    """
    Compute z-score stats from TRAIN episodes only.

    Node x order used everywhere:
      [rel_goal_x, rel_goal_y, dist_goal, lidar_0..R-1, goal_blocked, jump_count, active]

    We compute mean/std only for continuous dims:
      rel_goal_x, rel_goal_y, dist_goal, lidar_*, jump_count
    and set binaries (goal_blocked, active) to mean=0,std=1 so normalization is a no-op.
    """
    node_sum = None
    node_sumsq = None
    node_count = 0

    # Edge features unchanged: [dp_x, dp_y, dist, blocked, active_j]
    edge_sum = np.zeros((3,), dtype=np.float64)
    edge_sumsq = np.zeros((3,), dtype=np.float64)
    edge_count = 0

    node_feature_names: List[str] = []
    edge_feature_names = ["dp_x", "dp_y", "dist", "blocked", "active_j"]

    for ep_path in episode_files:
        meta = meta_for_episode_dir(dataset_root, ep_path.parent)
        world_size = float(meta.get("world_size", world_scale_default))
        lidar_max_range = float(meta.get("lidar_max_range", 8.0))

        ep = np.load(ep_path, allow_pickle=True)

        # Use YOUR actual keys here (match build_samples_from_episode)
        positions = ep["obs_positions"].astype(np.float32)      # (T+1,N,2)
        goals = ep["goals"].astype(np.float32)                  # (N,2)
        active = ep["obs_active"].astype(bool)                  # (T+1,N)
        lidar = ep["obs_lidar_ranges"].astype(np.float32)       # (T+1,N,R)
        actions = ep["actions_expert"].astype(np.float32)       # (T,N,2)
        edge_index_obj = ep["edge_index"]                       # (T,) object
        walls_xyxy = ep["walls_xyxy"].astype(np.float32)        # (M,4)

        T = actions.shape[0]
        R = lidar.shape[2]

        # continuous dims = 2+1 + R + 1(jump_count) = R+4
        if node_sum is None:
            node_feature_names = (
                ["rel_goal_x", "rel_goal_y", "dist_goal"]
                + [f"lidar_{k}" for k in range(R)]
                + ["goal_blocked", "jump_count", "active"]
            )
            node_sum = np.zeros((R + 4,), dtype=np.float64)
            node_sumsq = np.zeros((R + 4,), dtype=np.float64)

        for t in range(T):
            pos_t = positions[t]  # (N,2)
            rel_goal = (goals - pos_t) / world_size                    # (N,2)
            dist_goal = np.linalg.norm(rel_goal, axis=1, keepdims=True) # (N,1)

            # lidar normalized (continuous)
            lidar_t = np.clip(lidar[t] / lidar_max_range, 0.0, 1.0)     # (N,R)

            # jump_count computed on RAW ranges (threshold in raw units)
            jump_count = _goal_cone_jump_count(
                lidar_ranges=lidar[t],               # raw (N,R)
                goal_vec=(goals - pos_t),            # raw (N,2)
                cone_opening_deg=30.0,
                jump_thresh=3.0,
            ).astype(np.float32)                      # (N,1)

            # continuous node vector for stats: [rel_goal(2), dist(1), lidar(R), jump_count(1)]
            node_cont = np.concatenate([rel_goal, dist_goal, lidar_t, jump_count], axis=1)  # (N, R+4)

            mask = active[t]
            if np.any(mask):
                X = node_cont[mask].astype(np.float64)
                node_sum += X.sum(axis=0)
                node_sumsq += (X * X).sum(axis=0)
                node_count += X.shape[0]

            # edge continuous stats unchanged
            edge_index = np.asarray(edge_index_obj[t], dtype=np.int64)
            if edge_index.size == 0:
                continue
            src = edge_index[0]
            dst = edge_index[1]
            dp = (pos_t[dst] - pos_t[src]) / world_size
            dist = np.linalg.norm(dp, axis=1, keepdims=True)
            cont = np.concatenate([dp, dist], axis=1).astype(np.float64)  # (E,3)

            edge_sum += cont.sum(axis=0)
            edge_sumsq += (cont * cont).sum(axis=0)
            edge_count += cont.shape[0]

    if node_sum is None or node_count == 0:
        raise RuntimeError("No active nodes found in training set to compute normalization stats.")

    node_mean_cont = node_sum / node_count
    node_var_cont = node_sumsq / node_count - node_mean_cont * node_mean_cont
    node_std_cont = _safe_std(node_var_cont)

    if edge_count == 0:
        edge_mean = np.zeros((3,), dtype=np.float64)
        edge_std = np.ones((3,), dtype=np.float64)
    else:
        edge_mean = edge_sum / edge_count
        edge_var = edge_sumsq / edge_count - edge_mean * edge_mean
        edge_std = _safe_std(edge_var)

    # Edge full: pad binary flags blocked, active_j
    edge_mean_full = np.array([edge_mean[0], edge_mean[1], edge_mean[2], 0.0, 0.0], dtype=np.float64)
    edge_std_full = np.array([edge_std[0], edge_std[1], edge_std[2], 1.0, 1.0], dtype=np.float64)

    # Node full: insert goal_blocked (mean=0,std=1), append active (mean=0,std=1)
    # Order: [rel_goal(2), dist(1), lidar(R), goal_blocked, jump_count, active]
    base = 3 + R
    jump_mean = node_mean_cont[base]      # last entry of cont
    jump_std = node_std_cont[base]

    node_mean_full = np.concatenate([
        node_mean_cont[:base],            # rel_goal + dist + lidar
        np.array([0.0], dtype=np.float64),# goal_blocked (no-op)
        np.array([jump_mean], dtype=np.float64),
        np.array([0.0], dtype=np.float64) # active (no-op)
    ])
    node_std_full = np.concatenate([
        node_std_cont[:base],
        np.array([1.0], dtype=np.float64),
        np.array([jump_std], dtype=np.float64),
        np.array([1.0], dtype=np.float64)
    ])

    # Sanity check: dims must match x dim = R + 6
    assert node_mean_full.shape[0] == (R + 6), (node_mean_full.shape, R)
    assert node_std_full.shape[0] == (R + 6), (node_std_full.shape, R)

    return NormalizationStats(
        node_mean=node_mean_full.tolist(),
        node_std=node_std_full.tolist(),
        edge_mean=edge_mean_full.tolist(),
        edge_std=edge_std_full.tolist(),
        node_feature_names=node_feature_names,
        edge_feature_names=edge_feature_names,
    )



def build_samples_from_episode(
    ep_path: Path,
    dataset_root: Path,
    stats: NormalizationStats,
    world_scale_default: float = 10.0,
    max_speed: float = 1.5,
):
    """Yield PyG Data samples for each timestep in one episode."""
    import torch
    from torch_geometric.data import Data

    meta = meta_for_episode_dir(dataset_root, ep_path.parent)
    world_size = float(meta.get("world_size", world_scale_default))
    lidar_max_range = float(meta.get("lidar_max_range", 8.0))

    ep = np.load(ep_path, allow_pickle=True)
    positions = ep["obs_positions"].astype(np.float32)
    goals = ep["goals"].astype(np.float32)
    active = ep["obs_active"].astype(bool)
    lidar = ep["obs_lidar_ranges"].astype(np.float32)
    actions = ep["actions_expert"].astype(np.float32)
    edge_index_obj = ep["edge_index"]
    walls_xyxy = ep["walls_xyxy"].astype(np.float32)

    T = actions.shape[0]
    node_mean = np.asarray(stats.node_mean, dtype=np.float32)
    node_std = np.asarray(stats.node_std, dtype=np.float32)
    edge_mean = np.asarray(stats.edge_mean, dtype=np.float32)
    edge_std = np.asarray(stats.edge_std, dtype=np.float32)

    for t in range(T):
        pos_t = positions[t]  # (N,2)
        N = pos_t.shape[0]

        # Node features
        rel_goal = (goals - pos_t) / world_size
        dist_goal = np.linalg.norm(rel_goal, axis=1, keepdims=True)
        lidar_t = np.clip(lidar[t] / lidar_max_range, 0.0, 1.0)

        goal_blocked = _segment_intersects_walls_vectorized(pos_t, goals, walls_xyxy)\
            .astype(np.float32).reshape(N, 1)

        jump_count = _goal_cone_jump_count(
            lidar_ranges=lidar[t],
            goal_vec=(goals - pos_t),
            cone_opening_deg=30.0,
            jump_thresh=3.0,
        ).astype(np.float32)


        active_t = active[t].astype(np.float32).reshape(N, 1)
        x = np.concatenate([rel_goal, dist_goal, lidar_t, goal_blocked, jump_count, active_t], axis=1).astype(np.float32)

        # Z-score normalize (active flag unaffected by mean/std=0/1)
        x = (x - node_mean) / node_std

        # Target actions (normalized by max_speed)
        y = actions[t] / max_speed

        # Edge structure (stored)
        edge_index = np.asarray(edge_index_obj[t], dtype=np.int64)
        if edge_index.size == 0:
            edge_index_t = torch.empty((2, 0), dtype=torch.long)
            edge_attr_t = torch.empty((0, len(stats.edge_feature_names)), dtype=torch.float32)
        else:
            src = edge_index[0]
            dst = edge_index[1]
            dp = (pos_t[dst] - pos_t[src]) / world_size
            dist = np.linalg.norm(dp, axis=1, keepdims=True)

            # blocked_by_walls: vectorized intersection test
            blocked = _segment_intersects_walls_vectorized(pos_t[src], pos_t[dst], walls_xyxy).astype(np.float32).reshape(-1, 1)
            active_j = active[t, dst].astype(np.float32).reshape(-1, 1)

            edge_attr = np.concatenate([dp, dist, blocked, active_j], axis=1).astype(np.float32)
            edge_attr = (edge_attr - edge_mean) / edge_std

            edge_index_t = torch.from_numpy(edge_index)
            edge_attr_t = torch.from_numpy(edge_attr)

        data = Data(
            x=torch.from_numpy(x),
            edge_index=edge_index_t,
            edge_attr=edge_attr_t,
            y=torch.from_numpy(y.astype(np.float32)),
            mask=torch.from_numpy(active[t].astype(np.bool_)),
        )
        # Keep minimal debug metadata (stored as python object)
        data.meta = {
            "episode": str(ep_path),
            "t": int(t),
        }
        yield data


def scan_max_agents(episode_files: List[Path]) -> int:
    """Scan episode files and return the maximum number of agents (N) found.

    This is useful for building fixed-shape (T, N_max, ·) tensors when training
    recurrent models (e.g., GRU per agent) with standard batching.
    """
    max_n = 0
    for p in episode_files:
        ep = np.load(p, allow_pickle=True)
        # obs_positions: (T+1, N, 2)
        n = int(ep["obs_positions"].shape[1])
        if n > max_n:
            max_n = n
    return max_n


def build_sequence_samples_from_episode(
    ep_path: Path,
    dataset_root: Path,
    stats: NormalizationStats,
    *,
    seq_len: int,
    stride: int,
    max_agents: int | None = None,
    pad_to_max_agents: bool = True,
    world_scale_default: float = 10.0,
    max_speed: float = 1.5,
):
    """Yield sequence samples suitable for per-agent recurrent models (e.g., GRU).

    Each yielded sample is a python dict (pickleable via torch.save) with:
      x:         (L, N_pad, Dx) float32
      y:         (L, N_pad, 2)  float32   (normalized expert actions)
      mask:      (L, N_pad)     bool      (active agents)
      edge_index: list[LongTensor] length L, each (2, E_t)
      edge_attr:  list[FloatTensor] length L, each (E_t, De)
      seq_mask:  (L,) bool  (True for valid timesteps; all True here since we drop short tails)
      meta: dict (episode path + t0 + length + N)
    Notes:
      - We keep per-timestep graphs (edge_index/edge_attr) as a list because E varies with time.
      - If pad_to_max_agents=True, tensors are padded on the agent dimension to N_pad=max_agents.
        Padded agents have mask=False and zero x/y (after normalization).
    """
    import torch

    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    meta = meta_for_episode_dir(dataset_root, ep_path.parent)
    world_size = float(meta.get("world_size", world_scale_default))
    lidar_max_range = float(meta.get("lidar_max_range", 8.0))

    ep = np.load(ep_path, allow_pickle=True)
    positions = ep["obs_positions"].astype(np.float32)          # (T+1,N,2)
    goals = ep["goals"].astype(np.float32)                      # (N,2)
    active = ep["obs_active"].astype(bool)                      # (T+1,N)
    lidar = ep["obs_lidar_ranges"].astype(np.float32)           # (T+1,N,R)
    actions = ep["actions_expert"].astype(np.float32)           # (T,N,2)
    edge_index_obj = ep["edge_index"]                           # (T,) object
    walls_xyxy = ep["walls_xyxy"].astype(np.float32)            # (M,4)

    T = int(actions.shape[0])
    N = int(goals.shape[0])
    R = int(lidar.shape[2])

    node_mean = np.asarray(stats.node_mean, dtype=np.float32)
    node_std = np.asarray(stats.node_std, dtype=np.float32)
    edge_mean = np.asarray(stats.edge_mean, dtype=np.float32)
    edge_std = np.asarray(stats.edge_std, dtype=np.float32)

    if pad_to_max_agents:
        if max_agents is None:
            raise ValueError("max_agents must be provided when pad_to_max_agents=True")
        N_pad = int(max_agents)
        if N_pad < N:
            raise ValueError(f"max_agents ({N_pad}) < episode agent count ({N})")
    else:
        N_pad = N

    # Drop short tails to keep fixed-length sequences (easier batching).
    for t0 in range(0, T - seq_len + 1, stride):
        xs = np.zeros((seq_len, N_pad, R + 6), dtype=np.float32)
        ys = np.zeros((seq_len, N_pad, 2), dtype=np.float32)
        ms = np.zeros((seq_len, N_pad), dtype=np.bool_)
        edge_index_list: List[torch.Tensor] = []
        edge_attr_list: List[torch.Tensor] = []

        for k in range(seq_len):
            t = t0 + k
            pos_t = positions[t]  # (N,2)

            rel_goal = (goals - pos_t) / world_size
            dist_goal = np.linalg.norm(rel_goal, axis=1, keepdims=True)
            lidar_t = np.clip(lidar[t] / lidar_max_range, 0.0, 1.0)

            goal_blocked = _segment_intersects_walls_vectorized(pos_t, goals, walls_xyxy).astype(np.float32).reshape(N, 1)

            jump_count = _goal_cone_jump_count(
                lidar_ranges=lidar[t],
                goal_vec=(goals - pos_t),
                cone_opening_deg=30.0,
                jump_thresh=3.0,
            ).astype(np.float32)

            active_t = active[t].astype(np.float32).reshape(N, 1)

            x = np.concatenate([rel_goal, dist_goal, lidar_t, goal_blocked, jump_count, active_t], axis=1).astype(np.float32)
            x = (x - node_mean) / node_std

            y = (actions[t] / max_speed).astype(np.float32)

            if N_pad == N:
                xs[k] = x
                ys[k] = y
                ms[k] = active[t]
            else:
                xs[k, :N] = x
                ys[k, :N] = y
                ms[k, :N] = active[t]

            edge_index = np.asarray(edge_index_obj[t], dtype=np.int64)
            if edge_index.size == 0:
                edge_index_t = torch.empty((2, 0), dtype=torch.long)
                edge_attr_t = torch.empty((0, len(stats.edge_feature_names)), dtype=torch.float32)
            else:
                src = edge_index[0]
                dst = edge_index[1]
                dp = (pos_t[dst] - pos_t[src]) / world_size
                dist = np.linalg.norm(dp, axis=1, keepdims=True)

                blocked = _segment_intersects_walls_vectorized(pos_t[src], pos_t[dst], walls_xyxy).astype(np.float32).reshape(-1, 1)
                active_j = active[t, dst].astype(np.float32).reshape(-1, 1)

                edge_attr = np.concatenate([dp, dist, blocked, active_j], axis=1).astype(np.float32)
                edge_attr = (edge_attr - edge_mean) / edge_std

                edge_index_t = torch.from_numpy(edge_index)
                edge_attr_t = torch.from_numpy(edge_attr)

            edge_index_list.append(edge_index_t)
            edge_attr_list.append(edge_attr_t)

        sample = {
            "x": torch.from_numpy(xs),
            "y": torch.from_numpy(ys),
            "mask": torch.from_numpy(ms),
            "edge_index": edge_index_list,
            "edge_attr": edge_attr_list,
            "seq_mask": torch.ones((seq_len,), dtype=torch.bool),
            "meta": {
                "episode": str(ep_path),
                "t0": int(t0),
                "len": int(seq_len),
                "N": int(N),
                "N_pad": int(N_pad),
            },
        }
        yield sample


def write_shards(samples, out_dir: Path, shard_size: int) -> int:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    buf = []
    for s in samples:
        buf.append(s)
        if len(buf) >= shard_size:
            p = out_dir / f"shard_{shard_idx:03d}.pt"
            torch.save(buf, p)
            shard_idx += 1
            buf = []
    if buf:
        p = out_dir / f"shard_{shard_idx:03d}.pt"
        torch.save(buf, p)
        shard_idx += 1
    return shard_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, default="datasets/il_dataset")
    ap.add_argument("--out_dir", type=str, default="datasets/il_dataset/processed_student_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--shard_size", type=int, default=2000)
    ap.add_argument("--sequence_mode", action="store_true", default=True,
                    help="If set, build fixed-length sequence samples for recurrent (GRU) training.")
    ap.add_argument("--seq_len", type=int, default=32,
                    help="Sequence length (timesteps) when --sequence_mode is set.")
    ap.add_argument("--seq_stride", type=int, default=24,
                    help="Stride between consecutive sequences when --sequence_mode is set.")
    ap.add_argument("--pad_agents_to_max", default=True, action="store_true",
                    help="Pad agent dimension to the global max agent count for easy batching (recommended).")
    ap.add_argument("--no_pad_agents_to_max", action="store_true",
                    help="Disable agent padding (sequences keep per-episode N; batching needs a custom collate).")

    ap.add_argument("--max_speed", type=float, default=1.5)
    ap.add_argument("--world_size_default", type=float, default=10.0)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    episode_dirs = find_episode_dirs(dataset_root)
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found under {dataset_root}")

    all_files: List[Path] = []
    per_dir: Dict[str, int] = {}
    for d in episode_dirs:
        files = list_episode_files(d)
        per_dir[d.name] = len(files)
        all_files.extend(files)

    if not all_files:
        raise FileNotFoundError("Found episode directories but no episode_*.npz files.")

    all_files = sorted(all_files)
    train_files, val_files, test_files = split_files(all_files, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac)

    print(f"Found {len(all_files)} episodes across {len(episode_dirs)} dirs: {per_dir}")
    print(f"Split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    # 1) Compute normalization stats on TRAIN split
    
    print("Computing normalization stats on train split...")
    stats = compute_stats(
        train_files,
        dataset_root=dataset_root,
        world_scale_default=args.world_size_default,
        max_speed=args.max_speed,
    )
    (out_root / "stats.json").write_text(json.dumps(stats.__dict__, indent=2), encoding="utf-8")


    # 2) Build and write shards
    pad_agents = args.pad_agents_to_max and (not args.no_pad_agents_to_max)
    max_agents = None
    if args.sequence_mode and pad_agents:
        print("Scanning for max agent count for padding...")
        max_agents = scan_max_agents(all_files)
        print(f"Max agents across all episodes: {max_agents}")

    def iter_samples(files: List[Path]):
        for ep_path in files:
            if not args.sequence_mode:
                yield from build_samples_from_episode(
                    ep_path,
                    dataset_root=dataset_root,
                    stats=stats,
                    world_scale_default=args.world_size_default,
                    max_speed=args.max_speed,
                )
            else:
                yield from build_sequence_samples_from_episode(
                    ep_path,
                    dataset_root=dataset_root,
                    stats=stats,
                    seq_len=args.seq_len,
                    stride=args.seq_stride,
                    max_agents=max_agents,
                    pad_to_max_agents=pad_agents,
                    world_scale_default=args.world_size_default,
                    max_speed=args.max_speed,
                )

    print("Writing train shards...")
    n_train_shards = write_shards(iter_samples(train_files), out_root / "train", shard_size=args.shard_size)
    print("Writing val shards...")
    n_val_shards = write_shards(iter_samples(val_files), out_root / "val", shard_size=args.shard_size)
    print("Writing test shards...")
    n_test_shards = write_shards(iter_samples(test_files), out_root / "test", shard_size=args.shard_size)

    manifest = {
        "dataset_root": str(dataset_root),
        "out_dir": str(out_root),
        "episode_dirs": [str(d) for d in episode_dirs],
        "episode_counts": per_dir,
        "n_episodes": len(all_files),
        "splits": {
            "train": [str(p) for p in train_files],
            "val": [str(p) for p in val_files],
            "test": [str(p) for p in test_files],
        },
        "n_shards": {"train": n_train_shards, "val": n_val_shards, "test": n_test_shards},
        "notes": {
            "sequence_mode": bool(args.sequence_mode),
            "seq_len": int(args.seq_len) if args.sequence_mode else None,
            "seq_stride": int(args.seq_stride) if args.sequence_mode else None,
            "pad_agents_to_max": bool(pad_agents) if args.sequence_mode else None,
            "max_agents": int(max_agents) if (args.sequence_mode and pad_agents) else None,

            "node_features": stats.node_feature_names,
            "edge_features": stats.edge_feature_names,
            "target": "actions[t] normalized by max_speed",
            "edge_attr_action_leakage": "removed (no dv, no closing speed)",
            "blocked_edges": "kept; blocked_by_walls is a binary edge feature",
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Done. Wrote processed dataset to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
