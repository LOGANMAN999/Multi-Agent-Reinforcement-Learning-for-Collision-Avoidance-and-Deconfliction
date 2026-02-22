"""
dataset_generator.py

Episode runner + logging scaffold for imitation learning (GNN-friendly).

This file is intentionally modular:
- Environment + map creation are handled via imports from sim_env.py and map_generation.py
- Recording is handled by EpisodeRecorder
- Graph construction is handled by GraphBuilder
- Saving format is NPZ (one file per episode) with JSON sidecar metadata.

Next steps (we'll iterate):
- Decide a canonical observation tensor for node features (agent-centric vs world)
- Decide exact edge features and radius/KNN policy for graph construction
- Add filtering: only keep episodes successful within <= max_steps
- Add batched runs + diagnostics summary file
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# --- Imports from your existing sandbox files ---------------------------------
# These must be importable when you run `python dataset_generator.py` from the repo root.
# If your repo uses a package layout, we'll adjust these imports in the next iteration.

from sim_env import MultiRobotEnv, Wall  # environment + geometry
from .map_generation import generate_map, MapSpec  # random map generator


# --- Configuration ------------------------------------------------------------

@dataclass
class DatasetGenConfig:
    # Episode generation
    seed: int = 6969
    n_episodes: int = 1000
    max_steps: int = 300

    # Map parameters (per episode we sample counts uniformly in provided ranges)
    world_size: float = 10.0
    obstacle_size: float = 0.7
    obstacle_count_range: Tuple[int, int] = (10, 15)  # inclusive
    wall_count_range: Tuple[int, int] = (5, 20)      # inclusive
    wall_len_range: Tuple[float, float] = (2.0, 8.0)
    margin: float = 0.8

    # Agents
    n_agents_range: Tuple[int, int] = (6, 15)        # inclusive

    # LiDAR logging
    lidar_n_rays: int = 64
    lidar_max_range: float = 20.0

    # Graph logging (GNN)
    edge_radius: float = 3.0   # in world units; tune later
    include_self_edges: bool = False
    directed_edges: bool = True

    # Output
    out_dir: str = "datasets/il_dataset"


# --- Helpers: map conversion --------------------------------------------------

def mapspec_to_walls(map_spec: MapSpec) -> List[Wall]:
    """
    Convert map_generation.MapSpec into sim_env.Wall segments.
    Note: MapSpec includes obstacles as boxes; MapSpec.to_segments() converts them to segments.
    """
    walls: List[Wall] = []
    for s in map_spec.all_wall_segments():
        walls.append(Wall(float(s.x1), float(s.y1), float(s.x2), float(s.y2)))
    return walls


def sample_episode_mapspec(
    rng: np.random.Generator,
    cfg: DatasetGenConfig,
    episode_idx: int,
) -> Tuple[MapSpec, int, int]:
    """
    Sample obstacle/wall counts and generate a new MapSpec.
    Returns: (map_spec, n_obstacles, n_walls)
    """
    n_obs = int(rng.integers(cfg.obstacle_count_range[0], cfg.obstacle_count_range[1] + 1))
    n_walls = int(rng.integers(cfg.wall_count_range[0], cfg.wall_count_range[1] + 1))

    map_id = f"ep{episode_idx:06d}_obs{n_obs}_walls{n_walls}"
    # We want dense + tight spaces => use walls_and_obstacles consistently.
    category = "walls_only"

    map_spec = generate_map(
        rng=rng,
        category=category,
        map_id=map_id,
        world_size=cfg.world_size,
        n_obstacles=n_obs,
        obstacle_size=cfg.obstacle_size,
        wall_count=n_walls,
        wall_len_range=cfg.wall_len_range,
        margin=cfg.margin,
    )
    return map_spec, n_obs, n_walls

def make_boundary_walls(world_size: float) -> List[Wall]:
    w = float(world_size)
    return [
        Wall(-w, -w, -w,  w),
        Wall( w, -w,  w,  w),
        Wall(-w, -w,  w, -w),
        Wall(-w,  w,  w,  w),
    ]



def reset_env_with_walls(env: MultiRobotEnv, walls: List[Wall], n_agents: int) -> None:
    """
    Initialize env state using a custom wall list (bypasses env.reset(map_type,...)).
    Uses env's internal samplers to place starts/goals away from walls and each other.
    """
    env.walls = make_boundary_walls(env.world_size) + list(walls)
    env.map_type = None
    env.n_agents = int(n_agents)
    env.t = 0
    env.positions = env._sample_non_colliding_points(env.n_agents)
    env.goals = env._sample_non_colliding_points(env.n_agents)


# --- Graph builder (GNN edges) ------------------------------------------------

class GraphBuilder:
    """
    Builds per-timestep graph structure from positions (and optional extras).

    Current design: radius graph (edges between pairs with ||p_i - p_j|| <= edge_radius).
    Output format matches common GNN stacks (e.g., PyG):
      edge_index: (2, E) int64
      edge_attr:  (E, F) float32

    We'll iterate on edge features; for now, we include a small, stable set.
    """

    def __init__(self, edge_radius: float, directed: bool = True, include_self: bool = False):
        self.edge_radius = float(edge_radius)
        self.directed = bool(directed)
        self.include_self = bool(include_self)

    def build(
        self,
        positions: np.ndarray,        # (N,2)
        actions: np.ndarray,          # (N,2) interpreted as velocity commands
        active: np.ndarray,           # (N,) bool
        env: MultiRobotEnv,
    ) -> Tuple[np.ndarray, np.ndarray]:
        N = int(positions.shape[0])
        # Pairwise distances
        diff = positions[:, None, :] - positions[None, :, :]   # (N,N,2)
        d2 = np.sum(diff * diff, axis=-1)                      # (N,N)
        mask = d2 <= (self.edge_radius ** 2)

        if not self.include_self:
            np.fill_diagonal(mask, False)

        # Build edges
        src_list: List[int] = []
        dst_list: List[int] = []
        attr_list: List[List[float]] = []

        for i in range(N):
            for j in range(N):
                if not mask[i, j]:
                    continue
                # Undirected: only take i<j then add both directions (optional)
                if not self.directed and j <= i:
                    continue

                dp = positions[j] - positions[i]
                dv = actions[j] - actions[i]
                dist = float(np.sqrt(d2[i, j]) + 1e-12)
                closing = float(np.dot(dv, dp) / dist)  # positive means separating in this convention

                blocked = 1.0 if env.segment_blocked_by_walls(positions[i], positions[j]) else 0.0
                active_j = 1.0 if bool(active[j]) else 0.0

                # Edge features (F=8):
                # [dx, dy, dvx, dvy, dist, closing_speed, blocked_by_walls, active_j]
                attr_list.append([float(dp[0]), float(dp[1]), float(dv[0]), float(dv[1]),
                                  dist, closing, blocked, active_j])
                src_list.append(i)
                dst_list.append(j)

                if self.directed is False:
                    # If undirected, also add reverse with negated relative vectors.
                    attr_list.append([float(-dp[0]), float(-dp[1]), float(-dv[0]), float(-dv[1]),
                                      dist, closing, blocked, 1.0 if bool(active[i]) else 0.0])
                    src_list.append(j)
                    dst_list.append(i)

        edge_index = np.array([src_list, dst_list], dtype=np.int64)  # (2,E)
        edge_attr = np.array(attr_list, dtype=np.float32)            # (E,F)
        return edge_index, edge_attr


# --- Episode recorder ---------------------------------------------------------

class EpisodeRecorder:
    """
    Stores per-timestep arrays and variable-sized graph data, then saves to disk.
    """

    def __init__(self, n_agents: int, lidar_n_rays: int):
        self.n_agents = int(n_agents)
        self.lidar_n_rays = int(lidar_n_rays)

        # State/action time series
        self.positions: List[np.ndarray] = []   # each (N,2)
        self.actions: List[np.ndarray] = []     # each (N,2)
        self.active: List[np.ndarray] = []      # each (N,) bool
        self.lidar: List[np.ndarray] = []       # each (N,R)

        # Graph per timestep (variable E)
        self.edge_index: List[np.ndarray] = []  # each (2,E)
        self.edge_attr: List[np.ndarray] = []   # each (E,F)

        # Episode-level
        self.goals: Optional[np.ndarray] = None # (N,2)
        self.walls_xyxy: Optional[np.ndarray] = None  # (M,4)
        self.episode_meta: Dict[str, Any] = {}

    def capture_initial(self, env: MultiRobotEnv, lidar_n_rays: int, lidar_max_range: float) -> None:
        assert env.positions is not None and env.goals is not None
        self.goals = np.asarray(env.goals, dtype=np.float32).copy()

        # Save walls for reproducibility/training visualization
        walls = []
        for w in env.walls:
            walls.append([float(w.x1), float(w.y1), float(w.x2), float(w.y2)])
        self.walls_xyxy = np.asarray(walls, dtype=np.float32)

        # Capture t=0 state
        self.positions.append(np.asarray(env.positions, dtype=np.float32).copy())
        active0 = np.linalg.norm(env.positions - env.goals, axis=1) > env.goal_tolerance
        self.active.append(active0.astype(bool))

        scans0 = env.lidar_scan_all(n_rays=lidar_n_rays, max_range=lidar_max_range)
        self.lidar.append(np.asarray(scans0, dtype=np.float32).copy())

    def capture_step(
        self,
        env: MultiRobotEnv,
        actions: np.ndarray,
        graph: Tuple[np.ndarray, np.ndarray],
        lidar_n_rays: int,
        lidar_max_range: float,
    ) -> None:
        assert env.positions is not None and env.goals is not None

        self.actions.append(np.asarray(actions, dtype=np.float32).copy())

        # state after the env.step call
        self.positions.append(np.asarray(env.positions, dtype=np.float32).copy())
        active_t = np.linalg.norm(env.positions - env.goals, axis=1) > env.goal_tolerance
        self.active.append(active_t.astype(bool))

        scans = env.lidar_scan_all(n_rays=lidar_n_rays, max_range=lidar_max_range)
        self.lidar.append(np.asarray(scans, dtype=np.float32).copy())

        edge_index, edge_attr = graph
        self.edge_index.append(edge_index.astype(np.int64, copy=True))
        self.edge_attr.append(edge_attr.astype(np.float32, copy=True))

    def to_npz_payload(self) -> Dict[str, Any]:
        """
        NPZ-friendly dict.
        We store both full trajectories and explicit (obs_t, action_t, next_obs) slices
        to avoid any indexing ambiguity during training.
        """
        assert self.goals is not None and self.walls_xyxy is not None

        positions_all = np.stack(self.positions, axis=0)  # (T+1,N,2)
        actions = (
            np.stack(self.actions, axis=0)
            if self.actions
            else np.zeros((0, self.n_agents, 2), dtype=np.float32)
        )
        active_all = np.stack(self.active, axis=0)        # (T+1,N)
        lidar_all = np.stack(self.lidar, axis=0)          # (T+1,N,R)

        # Graph per step aligns with obs_t (pre-step)
        #edge_index_obj = np.array(self.edge_index, dtype=object)  # length T
        #edge_attr_obj = np.array(self.edge_attr, dtype=object)    # length T

        edge_index_obj = np.empty(len(self.edge_index), dtype=object)
        for k, ei in enumerate(self.edge_index):
            edge_index_obj[k] = ei

        edge_attr_obj = np.empty(len(self.edge_attr), dtype=object)
        for k, ea in enumerate(self.edge_attr):
            edge_attr_obj[k] = ea

        # Explicit supervised tuples:
        obs_positions = positions_all[:-1]   # (T,N,2)
        next_positions = positions_all[1:]   # (T,N,2)
        obs_active = active_all[:-1]         # (T,N)
        next_active = active_all[1:]         # (T,N)
        obs_lidar = lidar_all[:-1]           # (T,N,R)
        next_lidar = lidar_all[1:]           # (T,N,R)

        return {
            # Episode-level reconstruction
            "goals": self.goals,
            "walls_xyxy": self.walls_xyxy,

            # Full trajectories (sometimes convenient)
            "positions_all": positions_all,
            "active_all": active_all,
            "lidar_all": lidar_all,

            # Supervised learning slices (unambiguous)
            "obs_positions": obs_positions,
            "obs_active": obs_active,
            "obs_lidar_ranges": obs_lidar,
            "actions_expert": actions,
            "next_positions": next_positions,
            "next_active": next_active,
            "next_lidar_ranges": next_lidar,

            # GNN per-step graph aligned to obs_t
            "edge_index": edge_index_obj,
            "edge_attr": edge_attr_obj,

            "episode_meta_json": json.dumps(self.episode_meta),
        }



# --- Rollout ------------------------------------------------------------------

def run_one_episode(
    *,
    rng: np.random.Generator,
    cfg: DatasetGenConfig,
    episode_idx: int,
    controller: Any,
) -> Tuple[bool, int, Dict[str, Any], Optional[EpisodeRecorder]]:
    """
    Run a single episode and (optionally) return a recorder with collected data.

    Returns:
        success: bool
        steps: int
        info: dict (episode diagnostics)
        recorder: EpisodeRecorder if success else None
    """
    # sample map + agent count
    map_spec, n_obs, n_walls = sample_episode_mapspec(rng, cfg, episode_idx)
    n_agents = int(rng.integers(cfg.n_agents_range[0], cfg.n_agents_range[1] + 1))

    env = MultiRobotEnv(world_size=cfg.world_size)
    walls = mapspec_to_walls(map_spec)
    reset_env_with_walls(env, walls, n_agents)

    # Episode metadata to allow later reconstruction / auditing
    episode_meta = {
        "episode_idx": episode_idx,
        "map_id": map_spec.map_id,
        "category": map_spec.category,
        "world_size": cfg.world_size,
        "obstacle_size": cfg.obstacle_size,
        "n_obstacles": n_obs,
        "n_walls": n_walls,
        "n_agents": n_agents,

        # env dynamics/termination knobs (log them even if you don't plan to change)
        "dt": getattr(env, "dt", None),
        "max_speed": getattr(env, "max_speed", None),
        "robot_radius": getattr(env, "robot_radius", None),
        "goal_tolerance": getattr(env, "goal_tolerance", None),

        # sensing / graph knobs
        "lidar_n_rays": cfg.lidar_n_rays,
        "lidar_max_range": cfg.lidar_max_range,
        "edge_radius": cfg.edge_radius,
        "directed_edges": cfg.directed_edges,
        "include_self_edges": cfg.include_self_edges,
    }


    graph_builder = GraphBuilder(edge_radius=cfg.edge_radius,
                                 directed=cfg.directed_edges,
                                 include_self=cfg.include_self_edges)

    rec = EpisodeRecorder(n_agents=n_agents, lidar_n_rays=cfg.lidar_n_rays)
    rec.episode_meta = episode_meta
    rec.capture_initial(env, lidar_n_rays=cfg.lidar_n_rays, lidar_max_range=cfg.lidar_max_range)

    done = False
    info = {}
    steps = 0

    while not done and steps < cfg.max_steps:
        # Controller returns (N,2) desired velocities.
        actions = controller(env)

        # Build graph from pre-step state and chosen action (expert context).
        # Note: actions doubles as "velocity" signal in this env.
        pos = np.asarray(env.positions, dtype=float)
        act = np.asarray(actions, dtype=float)
        active = np.linalg.norm(pos - env.goals, axis=1) > env.goal_tolerance
        graph = graph_builder.build(pos, act, active, env)

        _, _, done, info = env.step(actions)
        rec.capture_step(env, actions, graph, lidar_n_rays=cfg.lidar_n_rays, lidar_max_range=cfg.lidar_max_range)

        steps += 1

    success = bool(info.get("reached_all", False)) and not bool(info.get("collision_with_wall", False)) and not bool(info.get("collision_between_robots", False)) and steps <= cfg.max_steps
    diag = {
        "episode_idx": episode_idx,
        "steps": steps,
        "n_agents": n_agents,
        "n_obstacles": n_obs,
        "n_walls": n_walls,
        "success": success,
        "info": {k: bool(v) for k, v in (info or {}).items()},
        "map_id": map_spec.map_id,
        "category": map_spec.category,
    }

    return success, steps, diag, (rec if success else None)


# --- Controller import ---------------------------------------------------------

def make_expert_controller():
    """
    Import the controller from the sandbox files.

    We keep this as a function to avoid hard-coding your repo layout. If your repo has
    a controllers/ package, we can swap the import in one place.
    """
    try:
        # If you have a package layout in your repo.
        from controllers.astar_global_local import AStarGlobalLocalController  # type: ignore
        return AStarGlobalLocalController()
    except Exception:
        # Fallback for the sandbox where astar_global_local.py is at project root.
        from astar_global_local import AStarGlobalLocalController  # type: ignore
        return AStarGlobalLocalController()


# --- Saving -------------------------------------------------------------------

def save_episode_npz(out_dir: Path, episode_idx: int, recorder: EpisodeRecorder, diag: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_path = out_dir / f"episode_{episode_idx:06d}.npz"
    payload = recorder.to_npz_payload()
    np.savez_compressed(ep_path, **payload)

    # Write a small JSON sidecar for easy inspection (optional).
    sidecar = out_dir / f"episode_{episode_idx:06d}.json"
    sidecar.write_text(json.dumps(diag, indent=2))
    return ep_path


# --- Main ---------------------------------------------------------------------

def main(cfg: DatasetGenConfig) -> None:
    rng = np.random.default_rng(cfg.seed)
    out_root = Path(cfg.out_dir)
    episodes_dir = out_root / "episodes_walls"
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    controller = make_expert_controller()

    run_meta = asdict(cfg)
    (out_root / "meta_walls.json").write_text(json.dumps(run_meta, indent=2))

    total = 0
    success_ct = 0

    log_path = logs_dir / "run_log_walls.jsonl"
    fail_path = logs_dir / "failures_walls.jsonl"

    with log_path.open("a", encoding="utf-8") as lf, fail_path.open("a", encoding="utf-8") as ff:
        for ep in range(cfg.n_episodes):
            total += 1
            success, steps, diag, rec = run_one_episode(rng=rng, cfg=cfg, episode_idx=ep, controller=controller)

            lf.write(json.dumps(diag) + "\n")
            lf.flush()

            if success and rec is not None:
                success_ct += 1
                save_episode_npz(episodes_dir, ep, rec, diag)
            else:
                ff.write(json.dumps(diag) + "\n")
                ff.flush()

    pct = 100.0 * success_ct / max(total, 1)
    print(f"Done. Success: {success_ct}/{total} ({pct:.1f}%). Dataset at: {out_root.resolve()}")


if __name__ == "__main__":
    # Lightweight CLI: tweak later as needed.
    cfg = DatasetGenConfig()
    main(cfg)
