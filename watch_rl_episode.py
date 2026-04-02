#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path
import sys
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sim_env import MultiRobotEnv, Wall
from data_building.map_generation import generate_mapset
from controllers.harmonic_navigation import HarmonicNavigationController
from controllers.astar_global_local import AStarGlobalLocalController

from RL_stack.gat_deconfliction_policy import GATDeconflictionPolicy
from RL_stack.gat_graph_builder import INTERACTION_RADIUS
from controllers.gat_deconfliction_controller import (
    GATDeconflictionController,
    ASTAR_MODE_STEPS,
    GOAL_PROXIMITY_STOP,
    SCAN_SAFETY_MARGIN,
)


def _compute_proximity_edges(
    positions: np.ndarray,   # [N, 2]
    active_mask: np.ndarray, # [N] bool
    radius: float,
    env=None,                # MultiRobotEnv — if provided, filters wall-blocked edges
) -> np.ndarray:
    
    active_idx = np.where(active_mask)[0]
    if len(active_idx) < 2:
        return np.zeros((2, 0), dtype=np.int64)
    pos = positions[active_idx]                           # [M, 2]
    diff = pos[:, None, :] - pos[None, :, :]             # [M, M, 2]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))            # [M, M]
    np.fill_diagonal(dist, np.inf)
    i_local, j_local = np.where(dist < radius)
    src = active_idx[i_local]
    dst = active_idx[j_local]

    if env is not None and len(src) > 0:
        keep = np.array([
            not env.segment_blocked_by_walls(positions[s], positions[d])
            for s, d in zip(src.tolist(), dst.tolist())
        ], dtype=bool)
        src = src[keep]
        dst = dst[keep]

    if len(src) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    return np.stack([src, dst], axis=0).astype(np.int64)


# 10 visually distinct colors (matplotlib tab10)
_AGENT_COLORS = plt.cm.tab10.colors


@dataclass
class EpisodeFrame:
    positions: np.ndarray           # [N, 2]
    goals: np.ndarray               # [N, 2]
    active: np.ndarray              # [N] bool
    astar_velocities: np.ndarray    # [N, 2]
    corrected_velocities: np.ndarray  # [N, 2]  priority-protocol adjusted
    t: int
    priority_scores: np.ndarray = field(default_factory=lambda: np.array([]))  # [N]
    yield_mask: np.ndarray = field(default_factory=lambda: np.array([]))       # [N] bool
    goals_reached: int = 0


@dataclass
class EpisodeRecording:
    frames: List[EpisodeFrame]
    n_agents: int
    world_size: float
    walls: List[Dict]  # serializable wall data
    seed: int
    n_agents_final: int
    max_t: int


def save_episode(recording: EpisodeRecording, filepath: str) -> None:
   
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    frame_data = {
        "positions":             np.array([f.positions for f in recording.frames]),
        "goals":                 np.array([f.goals for f in recording.frames]),
        "active":                np.array([f.active for f in recording.frames]),
        "astar_velocities":      np.array([f.astar_velocities for f in recording.frames]),
        "corrected_velocities":  np.array([f.corrected_velocities for f in recording.frames]),
        "timesteps":             np.array([f.t for f in recording.frames]),
        "priority_scores":       np.array([f.priority_scores for f in recording.frames]),
        "yield_mask":            np.array([f.yield_mask for f in recording.frames]),
        "goals_reached":         np.array([f.goals_reached for f in recording.frames], dtype=np.int32),
    }
    np.savez_compressed(str(filepath.with_suffix(".npz")), **frame_data)

    metadata = {
        "n_agents": recording.n_agents,
        "world_size": recording.world_size,
        "walls": recording.walls,
        "seed": recording.seed,
        "n_agents_final": recording.n_agents_final,
        "max_t": recording.max_t,
        "n_frames": len(recording.frames),
    }
    with open(filepath.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved episode to {filepath.with_suffix('.*')} ({len(recording.frames)} frames)")


def load_episode(filepath: str) -> tuple:
    """
    Load a recorded episode from disk.

    Args:
        filepath: Base path (without extension).

    Returns:
        (recording, env_template): EpisodeRecording and simple env-like object for rendering.
    """
    filepath = Path(filepath)

    with open(filepath.with_suffix(".json"), "r") as f:
        metadata = json.load(f)

    npz_data = np.load(filepath.with_suffix(".npz"))
    positions            = npz_data["positions"]
    goals                = npz_data["goals"]
    active               = npz_data["active"]
    astar_velocities     = npz_data["astar_velocities"]
    corrected_velocities = npz_data["corrected_velocities"]
    timesteps            = npz_data["timesteps"]
    N = metadata["n_agents"]

    # priority_scores / yield_mask may be absent in recordings made before this update
    priority_scores = npz_data["priority_scores"] if "priority_scores" in npz_data else np.zeros((len(timesteps), N))
    yield_mask      = npz_data["yield_mask"]      if "yield_mask"      in npz_data else np.zeros((len(timesteps), N), dtype=bool)
    goals_reached   = npz_data["goals_reached"]   if "goals_reached"   in npz_data else np.zeros(len(timesteps), dtype=np.int32)

    frames = []
    for i in range(len(timesteps)):
        frame = EpisodeFrame(
            positions=positions[i],
            goals=goals[i],
            active=active[i],
            astar_velocities=astar_velocities[i],
            corrected_velocities=corrected_velocities[i],
            t=int(timesteps[i]),
            priority_scores=priority_scores[i],
            yield_mask=yield_mask[i].astype(bool),
            goals_reached=int(goals_reached[i]),
        )
        frames.append(frame)

    from sim_env import Wall
    walls = [Wall(w["x1"], w["y1"], w["x2"], w["y2"]) for w in metadata["walls"]]

    recording = EpisodeRecording(
        frames=frames,
        n_agents=metadata["n_agents"],
        world_size=metadata["world_size"],
        walls=metadata["walls"],
        seed=metadata["seed"],
        n_agents_final=metadata["n_agents_final"],
        max_t=metadata["max_t"],
    )

    class EnvTemplate:
        pass

    env_template = EnvTemplate()
    env_template.n_agents = metadata["n_agents"]
    env_template.world_size = metadata["world_size"]
    env_template.walls = walls

    print(f"Loaded episode from {filepath.with_suffix('.*')} ({len(frames)} frames)")
    return recording, env_template


def load_gat_checkpoint(
    ckpt_path: str,
    device: torch.device,
    hidden_dim: int = 128,
    n_heads: int = 4,
) -> GATDeconflictionPolicy:
    """
    Load a GAT checkpoint and instantiate the policy.

    Args:
        ckpt_path: Path to checkpoint file.
        device: torch device.
        hidden_dim: GATDeconflictionPolicy hidden dimension.
        n_heads: Number of attention heads.

    Returns:
        policy: Loaded GATDeconflictionPolicy on device.
    """
    policy = GATDeconflictionPolicy(
        node_dim=12,
        edge_dim=9,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
        policy.load_state_dict(ckpt["policy_state_dict"], strict=True)
    elif isinstance(ckpt, dict) and "policy_state" in ckpt:
        policy.load_state_dict(ckpt["policy_state"], strict=True)
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        policy.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        policy.load_state_dict(ckpt, strict=True)

    policy.eval()
    return policy


def build_test_env(
    seed: int = 4243,
    chosen_idx: int = 6,
    n_agents: int = 20,
    world_size: float = 10.0,
) -> MultiRobotEnv:
    ms = generate_mapset(
        seed=seed,
        per_category=3,
        world_size=world_size,
        n_obstacles=15,
        obstacle_size=0.7,
        wall_count=12,
        wall_len_range=(2.0, 8.0),
        margin=0.8,
    )
    chosen_map = ms.maps[chosen_idx % len(ms.maps)]

    env = MultiRobotEnv(world_size=chosen_map.world_size)
    segs = chosen_map.all_wall_segments()
    env.walls = [Wall(s.x1, s.y1, s.x2, s.y2) for s in segs]

    env.map_type = None
    env.n_agents = n_agents
    env.t = 0
    env.positions = env._sample_non_colliding_points(n_agents)
    env.goals = env._sample_non_colliding_points(n_agents)
    env._get_obs()

    return env


def prune_harmonic_colliders(env: MultiRobotEnv) -> int:
    """
    Run the harmonic flow scan and permanently remove agents whose trajectory
    would hit a wall from the environment.

    Uses the same lookahead parameters as GATDeconflictionController.reset().
    Mutates env.positions, env.goals, and env.n_agents in-place.

    Returns:
        Number of agents removed.
    """
    controller = HarmonicNavigationController()
    controller.reset(env)

    positions = np.array(env.positions, dtype=np.float32)
    goals     = np.array(env.goals,     dtype=np.float32)
    dt        = float(getattr(env, "dt", 0.1))

    hits = controller.simulate_flow_hits_wall(
        positions=positions,
        dt=dt,
        lookahead_steps=ASTAR_MODE_STEPS,
        goals=goals,
        goal_proximity_stop=GOAL_PROXIMITY_STOP,
        safety_margin=SCAN_SAFETY_MARGIN,
    )

    keep = ~hits
    n_removed = int(hits.sum())

    if n_removed > 0:
        env.positions = env.positions[keep]
        env.goals     = env.goals[keep]
        env.n_agents  = int(keep.sum())
        env._get_obs()

    return n_removed


def render_frame(
    env_or_template: object,
    active: np.ndarray,
    astar_velocities: np.ndarray,
    corrected_velocities: np.ndarray,
    positions: np.ndarray,
    goals: np.ndarray,
    t: int,
    pause: float = 0.03,
    dual_mode: bool = True,
    yield_mask: Optional[np.ndarray] = None,
    priority_scores: Optional[np.ndarray] = None,
    edge_index: Optional[np.ndarray] = None,
    show_graph: bool = False,
    baseline_label: str = "harmonic",
    astar_mode: Optional[np.ndarray] = None,
    goals_reached: int = 0,
):

    plt.clf()
    ax = plt.gca()
    N = len(active)

    if yield_mask is None:
        yield_mask = np.zeros(N, dtype=bool)

    # Draw walls
    for w in env_or_template.walls:
        ax.plot([w.x1, w.x2], [w.y1, w.y2], "k-", linewidth=2, zorder=1)

    # ------------------------------------------------------------------
    # Interaction graph overlay
    # ------------------------------------------------------------------
    if show_graph:
        # Interaction-radius discs — always drawn for every active agent
        for i in range(N):
            if active[i]:
                disc = plt.Circle(
                    (float(positions[i, 0]), float(positions[i, 1])),
                    INTERACTION_RADIUS,
                    color=_AGENT_COLORS[i % 10],
                    fill=True, alpha=0.12, zorder=2,
                )
                ax.add_patch(disc)

        # Edge lines — only when edges exist
        if edge_index is not None and edge_index.shape[1] > 0:
            src_arr, dst_arr = edge_index[0], edge_index[1]
            drawn = set()
            for s, d in zip(src_arr.tolist(), dst_arr.tolist()):
                pair = (min(s, d), max(s, d))
                if pair in drawn:
                    continue
                drawn.add(pair)
                if active[s] and active[d]:
                    ax.plot(
                        [positions[s, 0], positions[d, 0]],
                        [positions[s, 1], positions[d, 1]],
                        "-", color="red", linewidth=1.5, alpha=0.75, zorder=3,
                    )

    # ------------------------------------------------------------------
    # Agents, goals, velocity arrows
    # ------------------------------------------------------------------
    for i in range(N):
        c = _AGENT_COLORS[i % 10]
        alpha = 1.0 if active[i] else 0.30

        px, py = float(positions[i, 0]), float(positions[i, 1])
        gx, gy = float(goals[i, 0]), float(goals[i, 1])

        # Robot body (filled circle)
        ax.scatter(px, py, color=c, s=90, alpha=alpha, zorder=5)

        # Yielding indicator: hollow ring around the agent dot
        if active[i] and yield_mask[i]:
            ax.scatter(px, py, facecolors="none", edgecolors=c,
                       s=220, linewidths=1.5, alpha=0.85, zorder=6)

        # Goal (star)
        ax.scatter(gx, gy, color=c, marker="*", s=220, alpha=alpha, zorder=5)

        # Velocity arrows
        if active[i]:
            if dual_mode:
                # Harmonic velocity (dashed, semi-transparent)
                vx_a, vy_a = astar_velocities[i]
                ax.arrow(
                    px, py, vx_a * 0.3, vy_a * 0.3,
                    head_width=0.1, head_length=0.08,
                    fc=c, ec=c, alpha=0.4, linestyle="--", linewidth=0.8, zorder=4,
                )
                # Priority-adjusted velocity (solid)
                vx_c, vy_c = corrected_velocities[i]
                ax.arrow(
                    px, py, vx_c * 0.3, vy_c * 0.3,
                    head_width=0.15, head_length=0.12,
                    fc=c, ec=c, alpha=0.8, linewidth=1.2, zorder=5,
                )
            else:
                vx_a, vy_a = astar_velocities[i]
                ax.arrow(
                    px, py, vx_a * 0.3, vy_a * 0.3,
                    head_width=0.15, head_length=0.12,
                    fc=c, ec=c, alpha=0.8, linewidth=1.2, zorder=5,
                )

    ax.set_aspect("equal")
    ax.set_xlim(-env_or_template.world_size, env_or_template.world_size)
    ax.set_ylim(-env_or_template.world_size, env_or_template.world_size)

    n_yielding = int(yield_mask[active].sum()) if dual_mode else 0
    n_astar = int(astar_mode.sum()) if astar_mode is not None else 0
    if dual_mode:
        astar_str = f"  A*={n_astar}/{N}" if n_astar > 0 else ""
        title_suffix = f"yield={n_yielding}{astar_str}"
    else:
        title_suffix = f"{baseline_label} baseline"
    ax.set_title(f"Step {t}   active {int(active.sum())}/{N}   goals {goals_reached}/{N}   {title_suffix}")
    ax.grid(True, alpha=0.35)
    plt.pause(pause)


def run_episode(
    env: MultiRobotEnv,
    controller: GATDeconflictionController,
    max_steps: int = 300,
    pause: float = 0.04,
    record_path: str = None,
    show_graph: bool = True,
):
    N = env.n_agents
    controller.reset(env, n_agents=N)

    collided_mask     = np.zeros(N, dtype=bool)
    goal_reached_ever = np.zeros(N, dtype=bool)
    frames = []
    wall_data = [{"x1": w.x1, "y1": w.y1, "x2": w.x2, "y2": w.y2} for w in env.walls]

    if not record_path:
        plt.ion()
        plt.figure(figsize=(8, 8))

    for t in range(max_steps):
        active_mask = ~collided_mask  # goal-reached agents remain active in graph

        velocities, info = controller.act(env, active_mask=active_mask)

        astar_velocities = info["astar_velocities"]                                      # [N, 2]
        priority_scores  = info.get("priority_scores", np.zeros(N))                   # [N]
        yield_mask       = info.get("yield_mask", np.zeros(N, dtype=bool))            # [N]
        edge_index       = info.get("edge_index", None)                               # [2, E] or None
        astar_mode_info  = info.get("astar_mode", np.zeros(N, dtype=bool))            # [N]

        # Only collided agents get zero velocity; goal-reached keep moving
        velocities[collided_mask] = 0.0

        env.step(velocities)

        # Collision detection — already-inactive agents cannot newly collide
        wall_cols, robot_cols = env.check_per_agent_collisions_vec(env.positions)
        wall_cols[collided_mask]  = False
        robot_cols[collided_mask] = False
        collided_agents = wall_cols | robot_cols
        collided_mask  |= collided_agents

        # Goal detection — only for active (non-collided) agents
        active_mask = ~collided_mask
        dists = np.linalg.norm(env.positions - env.goals, axis=1)
        goals_reached_step = active_mask & (dists < env.goal_tolerance)
        goal_reached_ever |= goals_reached_step

        all_done = bool(np.all(goal_reached_ever | collided_mask))

        frame = EpisodeFrame(
            positions=env.positions.copy(),
            goals=env.goals.copy(),
            active=active_mask.copy(),
            astar_velocities=astar_velocities.copy(),
            corrected_velocities=velocities.copy(),
            t=t,
            priority_scores=priority_scores.copy(),
            yield_mask=yield_mask.copy(),
            goals_reached=int(goal_reached_ever.sum()),
        )
        frames.append(frame)

        if not record_path:
            render_frame(
                env, active_mask, astar_velocities, velocities,
                env.positions, env.goals, t,
                pause=pause, dual_mode=True,
                yield_mask=yield_mask,
                priority_scores=priority_scores,
                edge_index=edge_index,
                show_graph=show_graph,
                astar_mode=astar_mode_info,
                goals_reached=int(goal_reached_ever.sum()),
            )

        if all_done:
            print(f"Episode finished at step {t}: all agents done")
            break

    if record_path:
        recording = EpisodeRecording(
            frames=frames,
            n_agents=N,
            world_size=env.world_size,
            walls=wall_data,
            seed=0,
            n_agents_final=int(active_mask.sum()),
            max_t=t,
        )
        save_episode(recording, record_path)
    else:
        plt.ioff()
        plt.show()


def run_episode_harmonic(
    env: MultiRobotEnv,
    controller: HarmonicNavigationController,
    max_steps: int = 300,
    pause: float = 0.04,
    record_path: str = None,
    show_graph: bool = True,
):

    N = env.n_agents
    collided_mask     = np.zeros(N, dtype=bool)
    goal_reached_ever = np.zeros(N, dtype=bool)

    controller.reset(env)

    frames = []
    wall_data = [{"x1": w.x1, "y1": w.y1, "x2": w.x2, "y2": w.y2} for w in env.walls]

    if not record_path:
        plt.ion()
        plt.figure(figsize=(8, 8))

    for t in range(max_steps):
        active_mask = ~collided_mask  # goal-reached agents remain active

        velocities = controller(env)
        # Only collided agents get zero velocity; goal-reached keep moving
        velocities[collided_mask] = 0.0

        env.step(velocities)

        # Collision detection — already-inactive agents cannot newly collide
        wall_cols, robot_cols = env.check_per_agent_collisions_vec(env.positions)
        wall_cols[collided_mask]  = False
        robot_cols[collided_mask] = False
        collided_agents = wall_cols | robot_cols
        collided_mask  |= collided_agents

        # Goal detection — only for active (non-collided) agents
        active_mask = ~collided_mask
        dists = np.linalg.norm(env.positions - env.goals, axis=1)
        goals_reached_step = active_mask & (dists < env.goal_tolerance)
        goal_reached_ever |= goals_reached_step

        all_done = bool(np.all(goal_reached_ever | collided_mask))

        edge_index = (
            _compute_proximity_edges(env.positions, active_mask, INTERACTION_RADIUS, env=env)
            if show_graph else None
        )

        frame = EpisodeFrame(
            positions=env.positions.copy(),
            goals=env.goals.copy(),
            active=active_mask.copy(),
            astar_velocities=velocities.copy(),
            corrected_velocities=velocities.copy(),
            t=t,
            priority_scores=np.zeros(N),
            yield_mask=np.zeros(N, dtype=bool),
            goals_reached=int(goal_reached_ever.sum()),
        )
        frames.append(frame)

        if not record_path:
            render_frame(
                env, active_mask, velocities, velocities,
                env.positions, env.goals, t,
                pause=pause, dual_mode=False,
                edge_index=edge_index,
                show_graph=show_graph,
                goals_reached=int(goal_reached_ever.sum()),
            )

        if all_done:
            print(f"Episode finished at step {t}: all agents done")
            break

    if record_path:
        recording = EpisodeRecording(
            frames=frames,
            n_agents=N,
            world_size=env.world_size,
            walls=wall_data,
            seed=0,
            n_agents_final=int(active_mask.sum()),
            max_t=t,
        )
        save_episode(recording, record_path)
    else:
        plt.ioff()
        plt.show()


def run_episode_astar(
    env: MultiRobotEnv,
    controller: AStarGlobalLocalController,
    max_steps: int = 300,
    pause: float = 0.04,
    record_path: str = None,
    show_graph: bool = True,
):

    N = env.n_agents
    collided_mask     = np.zeros(N, dtype=bool)
    goal_reached_ever = np.zeros(N, dtype=bool)

    frames = []
    wall_data = [{"x1": w.x1, "y1": w.y1, "x2": w.x2, "y2": w.y2} for w in env.walls]

    if not record_path:
        plt.ion()
        plt.figure(figsize=(8, 8))

    for t in range(max_steps):
        active_mask = ~collided_mask

        velocities = controller(env)
        velocities[collided_mask] = 0.0

        env.step(velocities)

        wall_cols, robot_cols = env.check_per_agent_collisions_vec(env.positions)
        wall_cols[collided_mask]  = False
        robot_cols[collided_mask] = False
        collided_agents = wall_cols | robot_cols
        collided_mask  |= collided_agents

        active_mask = ~collided_mask
        dists = np.linalg.norm(env.positions - env.goals, axis=1)
        goals_reached_step = active_mask & (dists < env.goal_tolerance)
        goal_reached_ever |= goals_reached_step

        all_done = bool(np.all(goal_reached_ever | collided_mask))

        edge_index = (
            _compute_proximity_edges(env.positions, active_mask, INTERACTION_RADIUS, env=env)
            if show_graph else None
        )

        frame = EpisodeFrame(
            positions=env.positions.copy(),
            goals=env.goals.copy(),
            active=active_mask.copy(),
            astar_velocities=velocities.copy(),
            corrected_velocities=velocities.copy(),
            t=t,
            priority_scores=np.zeros(N),
            yield_mask=np.zeros(N, dtype=bool),
            goals_reached=int(goal_reached_ever.sum()),
        )
        frames.append(frame)

        if not record_path:
            render_frame(
                env, active_mask, velocities, velocities,
                env.positions, env.goals, t,
                pause=pause, dual_mode=False,
                edge_index=edge_index,
                show_graph=show_graph,
                baseline_label="A*",
                goals_reached=int(goal_reached_ever.sum()),
            )

        if all_done:
            print(f"Episode finished at step {t}: all agents done")
            break

    if record_path:
        recording = EpisodeRecording(
            frames=frames,
            n_agents=N,
            world_size=env.world_size,
            walls=wall_data,
            seed=0,
            n_agents_final=int(active_mask.sum()),
            max_t=t,
        )
        save_episode(recording, record_path)
    else:
        plt.ioff()
        plt.show()


def playback_episode(
    recording: EpisodeRecording,
    env_template: object,
    pause: float = 0.04,
    dual_mode: bool = True,
):

    plt.ion()
    plt.figure(figsize=(8, 8))

    for frame in recording.frames:
        render_frame(
            env_template,
            frame.active,
            frame.astar_velocities,
            frame.corrected_velocities,
            frame.positions,
            frame.goals,
            frame.t,
            pause=pause,
            dual_mode=dual_mode,
            yield_mask=frame.yield_mask,
            priority_scores=frame.priority_scores,
            goals_reached=frame.goals_reached,
        )

    plt.ioff()
    plt.show()


def export_gif(
    recording: EpisodeRecording,
    env_template: object,
    output_path: str,
    fps: int = 20,
    dual_mode: bool = True,
):

    import io
    import matplotlib
    from PIL import Image

    output_path = str(Path(output_path).with_suffix(".gif"))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Switch to non-interactive Agg backend so render_frame's plt.pause(0)
    # doesn't try to update a display window during offscreen rendering.
    _prev_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt_agg  # noqa: re-import after backend switch

    fig = _plt_agg.figure(figsize=(8, 8))
    pil_frames = []

    print(f"Rendering {len(recording.frames)} frames...")
    for i, frame in enumerate(recording.frames):
        fig.clf()
        ax = fig.add_subplot(111)
        N = len(frame.active)
        yield_mask = frame.yield_mask if frame.yield_mask is not None else np.zeros(N, dtype=bool)

        for w in env_template.walls:
            ax.plot([w.x1, w.x2], [w.y1, w.y2], "k-", linewidth=2, zorder=1)

        for j in range(N):
            c = _AGENT_COLORS[j % 10]
            pos = frame.positions[j]
            goal = frame.goals[j]
            is_active = bool(frame.active[j])
            alpha = 1.0 if is_active else 0.25

            ax.scatter(*goal, marker="*", s=180, color=c, zorder=3, alpha=alpha)
            circle = _plt_agg.Circle(pos, 0.25, color=c, alpha=0.7 * alpha, zorder=4)
            ax.add_patch(circle)

            if is_active:
                harm = frame.astar_velocities[j]
                corr = frame.corrected_velocities[j]
                if dual_mode and np.linalg.norm(harm) > 1e-6:
                    ax.annotate("", xy=pos + harm * 0.5, xytext=pos,
                                arrowprops=dict(arrowstyle="->", color=c,
                                                linestyle="dashed", lw=1.2, alpha=0.6))
                if np.linalg.norm(corr) > 1e-6:
                    ax.annotate("", xy=pos + corr * 0.5, xytext=pos,
                                arrowprops=dict(arrowstyle="->", color=c, lw=1.5))
                if yield_mask[j]:
                    ring = _plt_agg.Circle(pos, 0.35, fill=False, edgecolor=c,
                                           linewidth=2, linestyle=":", zorder=5)
                    ax.add_patch(ring)

        ax.set_aspect("equal")
        ax.set_xlim(-env_template.world_size, env_template.world_size)
        ax.set_ylim(-env_template.world_size, env_template.world_size)
        ax.set_title(f"Step {frame.t}   active {int(frame.active.sum())}/{N}   goals {frame.goals_reached}/{N}")
        ax.grid(True, alpha=0.35)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf).convert("RGB").quantize(colors=256)
        pil_frames.append(img)
        buf.close()

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(recording.frames)} frames rendered")

    _plt_agg.close(fig)
    matplotlib.use(_prev_backend)

    if not pil_frames:
        print("No frames to export.")
        return

    duration_ms = max(1, 1000 // fps)
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    size_kb = Path(output_path).stat().st_size // 1024
    print(f"GIF saved to {output_path}  ({len(pil_frames)} frames @ {fps} fps, {size_kb} KB)")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize GAT deconfliction policy or A* baseline in simulation. "
            "Three modes: live rendering, recording (precompute), or playback. "
            "Recordings saved to: runs/episode_recordings/"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sheaf_deconfliction_v2/policy_ep669.pt", #ep165 is a good one to watch
        help="Path to trained GAT checkpoint.",
    )
    parser.add_argument(
        "--n_agents",
        type=int,
        default=30,
        help="Number of agents.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=9,
        help="RNG seed for environment generation.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=250,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.01,
        help="Pause duration between frames (seconds).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy (mean actions).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (sample actions).",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="GATDeconflictionPolicy hidden dimension.",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=4,
        help="Number of GAT attention heads.",
    )
    parser.add_argument(
        "--harmonic_only",
        action="store_true",
        default=False,
        help="Run harmonic navigation baseline instead of GAT RL policy.",
    )
    parser.add_argument(
        "--astar_only",
        action="store_true",
        default=False,
        help="Run pure A* controller baseline (no RL model).",
    )
    parser.add_argument(
        "--astar_base",
        action="store_true",
        default=False,
        help="Use A* as the base policy under the GAT RL deconfliction model (instead of harmonic).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=True,
        help="Play episode live without recording (default). Ignored if --record or --playback is set.",
    )
    parser.add_argument(
        "--no_show_graph",
        action="store_true",
        default=False,
        help="Disable interaction graph overlay (radius discs + edge lines). Graph is shown by default.",
    )
    parser.add_argument(
        "--record",
        type=str,
        default= None,
        metavar="PATH",
        help=(
            "Save episode recording to this path (without extension). "
            "Creates .npz (binary) and .json (metadata) files in runs/episode_recordings/. "
            "Examples: 'harmonic_baseline/harmonic_n20_s4241' or "
            "'gat_deconfliction/policy_ep350_n20_s4241'. Directories are auto-created."
        ),
    )
    parser.add_argument(
        "--playback",
        type=str,
        default= None, #"harmonic_baseline/policy_ep165_n30_s3.npz"
        metavar="PATH",
        help=(
            "Load and playback a recorded episode (without extension). "
            "Loads from runs/episode_recordings/. "
            "Example: 'gat_deconfliction/policy_ep350_n20_s4241'."
        ),
    )
    parser.add_argument(
        "--gif",
        type=str,
        default= None,
        metavar="PATH",
        help=(
            "Export playback as a GIF instead of showing live. "
            "Use with --playback. Path is relative to runs/episode_recordings/ "
            "if not absolute. Extension is added automatically. "
            "Example: --playback foo/bar --gif foo/bar"
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for GIF export (used with --gif).",
    )
    parser.add_argument(
        "--prune_harmonic_colliders",
        action="store_true",
        default=True,
        help=(
            "Before running, scan each agent's harmonic flow trajectory and "
            "permanently remove agents (and their goals) whose path would hit "
            "a wall. Reduces n_agents to only those with clean harmonic starts."
        ),
    )

    args = parser.parse_args()

    # Handle paths for recording/playback: prepend runs/episode_recordings/ if relative
    if args.record:
        record_path = args.record if args.record.startswith("/") or ":" in args.record else f"runs/episode_recordings/{args.record}"
        args.record = record_path

    if args.playback:
        p = args.playback
        if p.startswith("/") or ":" in p or p.startswith("runs/"):
            playback_path = p
        else:
            playback_path = f"runs/episode_recordings/{p}"
        args.playback = playback_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Playback mode (no simulation needed)
    if args.playback:
        print(f"Loading recorded episode from {args.playback}")
        recording, env_template = load_episode(args.playback)
        if args.gif:
            gif_path = args.gif if args.gif.startswith("/") or ":" in args.gif else f"runs/{args.gif}"
            export_gif(recording, env_template, gif_path, fps=args.fps, dual_mode=True)
        else:
            playback_episode(recording, env_template, pause=args.pause, dual_mode=True)
        return

    # Build environment
    print(f"Building environment with {args.n_agents} agents (seed={args.seed})")
    env = build_test_env(
        seed=args.seed,
        n_agents=args.n_agents,
    )

    if args.prune_harmonic_colliders:
        n_removed = prune_harmonic_colliders(env)
        print(f"  Pruned {n_removed} agents with wall-colliding harmonic flows "
              f"({env.n_agents} remaining)")

    show_graph = not args.no_show_graph
    print(f"  Interaction graph overlay: {show_graph}")

    if args.astar_only:
        mode_str = "A* controller baseline"
        if args.record:
            mode_str += " (recording)"
        print(f"Starting {mode_str} visualization...")
        controller = AStarGlobalLocalController()
        run_episode_astar(
            env, controller,
            max_steps=args.max_steps,
            pause=args.pause,
            record_path=args.record,
            show_graph=show_graph,
        )
    elif args.harmonic_only:
        mode_str = "harmonic navigation baseline"
        if args.record:
            mode_str += " (recording)"
        print(f"Starting {mode_str} visualization...")
        controller = HarmonicNavigationController()
        run_episode_harmonic(
            env, controller,
            max_steps=args.max_steps,
            pause=args.pause,
            record_path=args.record,
            show_graph=show_graph,
        )
    else:
        print(f"Loading checkpoint from {args.checkpoint}")
        policy = load_gat_checkpoint(
            args.checkpoint,
            device=device,
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
        )

        deterministic = not args.stochastic
        base_controller = AStarGlobalLocalController() if args.astar_base else None
        controller = GATDeconflictionController(
            policy=policy,
            device=device,
            deterministic=deterministic,
            base_controller=base_controller,
            # Hybrid fallback: only active when using the harmonic base (not --astar_base)
            enable_astar_fallback=(not args.astar_base),
        )

        mode_str = "GAT + A* base" if args.astar_base else "GAT + harmonic base"
        if args.record:
            mode_str += " (recording)"
        print(f"Starting {mode_str} visualization...")
        print(f"  Deterministic: {deterministic}")

        run_episode(
            env, controller,
            max_steps=args.max_steps,
            pause=args.pause,
            record_path=args.record,
            show_graph=show_graph,
        )


if __name__ == "__main__":
    np.random.seed(33)
    main()
