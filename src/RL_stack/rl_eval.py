#!/usr/bin/env python3
"""rl_eval.py

Deterministic evaluation harness for the GRU GNN policy / RL Actor-Critic.

This is the diagnostics companion to `train_student_rl_ppo_gru.py`.
It runs fixed-seed evaluation episodes, logs episode-level metrics, and writes
summary JSON + per-episode CSV.

Compatible with:
  - sim_env.py: MultiRobotEnv, MapType
  - rl_graph_obs.py: GraphObsBuilder, GraphObsConfig
  - rl_actor_critic_gru.py: GraphActorCriticGRU, ActorCriticConfig
  - rl_rewards.py: RewardComputer, RewardConfig

Checkpoint support
------------------
Loads either:
  1) RL checkpoints produced by GraphActorCriticGRU.save(...): keys
     {actor_state, critic_state, log_std, cfg}
  2) Supervised student checkpoints produced by train_student_gnn_gru.py:
     raw state_dict or a dict with key "model_state".

Evaluation logic
----------------
- Uses deterministic actions (no exploration sampling).
- Ignores env.step's built-in reward and done.
- Episode ends on:
    * team_success (all agents reached goals and no collisions), OR
    * all_inactive (all agents masked due to reaching or collisions), OR
    * max_steps.

Outputs
-------
Writes:
  - <out_dir>/summary.json
  - <out_dir>/episodes.csv

Example
-------
python rl_eval.py \
  --ckpt runs/rl_wall_finetune_v1/checkpoints/best_rl.pt \
  --stats_json datasets/il_dataset/processed_student_gru_v1/stats.json \
  --episodes 50 --max_steps 400 --n_agents_min 4 --n_agents_max 10 --map_type mixed \
  --out_dir runs/rl_wall_finetune_v1/eval_seed0
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import torch

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

# --- local imports: ensure this directory is on sys.path ---
_THIS_DIR = Path(__file__).resolve().parent
import sys

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sim_env import MapType, MultiRobotEnv  # noqa: E402
from rl_graph_obs import GraphObsBuilder, GraphObsConfig  # noqa: E402
from rl_actor_critic_gru import ActorCriticConfig, GraphActorCriticGRU  # noqa: E402
from rl_rewards import RewardConfig, RewardComputer  # noqa: E402

from data_building.map_generation import generate_map as generate_mapspec
from data_building.dataset_generator import mapspec_to_walls, reset_env_with_walls



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_map_type(s: str) -> Optional[MapType]:
    s = s.strip().lower()
    if s in ("mixed", "random_mix", "randmix", "any", "all"):
        return None
    for mt in MapType:
        if s == mt.value.lower() or s == mt.name.lower():
            return mt
    raise ValueError(
        f"Unknown map_type '{s}'. Options: mixed, " + ", ".join([m.value for m in MapType])
    )


def load_stats_dims(stats_json: Optional[str]) -> Tuple[int, int]:
    """Infer (node_dim, edge_dim) from stats.json if provided, else defaults."""
    if stats_json is None:
        # Defaults for this repo's GRU student: Dx=70, De=5 (see rl_graph_obs.py)
        return 70, 5
    with open(stats_json, "r", encoding="utf-8") as f:
        st = json.load(f)
    node_dim = int(len(st["node_mean"]))
    edge_dim = int(len(st["edge_mean"]))
    return node_dim, edge_dim


def build_obs_builder(stats_json: Optional[str], cfg: GraphObsConfig, device: str) -> GraphObsBuilder:
    if stats_json is None:
        return GraphObsBuilder(cfg, device=device)
    return GraphObsBuilder.from_stats_json(stats_json, cfg=cfg, device=device)


def load_policy_checkpoint(ckpt_path: str, ac: GraphActorCriticGRU) -> Dict[str, Any]:
    """Load either an RL actor-critic checkpoint or a supervised student checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta: Dict[str, Any] = {"ckpt_path": ckpt_path}

    # RL checkpoint format (GraphActorCriticGRU.save)
    if isinstance(ckpt, dict) and ("actor_state" in ckpt or "critic_state" in ckpt or "log_std" in ckpt):
        if "actor_state" in ckpt:
            ac.actor.load_state_dict(ckpt["actor_state"], strict=True)
        if "critic_state" in ckpt:
            ac.value_head.load_state_dict(ckpt["critic_state"], strict=True)
        if "log_std" in ckpt:
            with torch.no_grad():
                ac.log_std.copy_(torch.as_tensor(ckpt["log_std"], dtype=ac.log_std.dtype))
        meta["format"] = "rl_actor_critic"
        meta["cfg_in_ckpt"] = ckpt.get("cfg", None)
        return meta

    # Supervised checkpoint format (train_student_gnn_gru.py)
    ac.load_supervised_student(ckpt_path, strict=True)
    meta["format"] = "supervised_student"
    return meta


def to_numpy_aux(aux: Dict[str, torch.Tensor], active_override: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Convert GraphObsBuilder aux tensors to numpy for RewardComputer."""
    out: Dict[str, np.ndarray] = {
        "dist_to_goal": aux["dist_to_goal"].detach().cpu().numpy(),
        "min_lidar": aux["min_lidar"].detach().cpu().numpy(),
    }
    if active_override is not None:
        out["active"] = active_override.astype(np.bool_)
    elif "active" in aux:
        out["active"] = aux["active"].detach().cpu().numpy().astype(np.bool_)
    return out


@dataclass
class MapGenConfig:
    map_source: str = "sim_env"  # "sim_env" or "mapspec"

    # MapSpec params (only used if map_source == "mapspec")
    mapspec_category: str = "walls_and_obstacles"
    mapspec_n_obstacles: int = 12
    mapspec_obstacle_size: float = 0.7
    mapspec_wall_count: int = 8
    mapspec_wall_len_min: float = 2.0
    mapspec_wall_len_max: float = 8.0
    mapspec_margin: float = 0.8


@dataclass
class EpisodeResult:
    episode_idx: int
    seed: int
    map_type: str
    n_agents: int
    steps: int

    team_success: bool
    any_collision: bool
    wall_collision_any: bool
    agent_collision_any: bool

    mean_dist: float
    mean_min_lidar: float
    min_min_lidar: float


@torch.no_grad()
def run_one_episode(
    env: MultiRobotEnv,
    obs_builder: GraphObsBuilder,
    ac: GraphActorCriticGRU,
    reward_cfg: RewardConfig,
    *,
    device: torch.device,
    episode_idx: int,
    seed: int,
    map_type: Optional[MapType],
    map_cfg: MapGenConfig,
    n_agents: int,
    max_steps: int,
    render: bool = False,
) -> EpisodeResult:
    """Run one deterministic episode and return episode-level metrics."""
    ac.eval()

    mt = map_type if map_type is not None else random.choice(list(MapType))
    #env.reset(mt, n_agents=n_agents)

    if map_cfg.map_source == "mapspec":
        rng = np.random.default_rng(seed)  # deterministic per-episode

        map_spec = generate_mapspec(
            rng=rng,
            category=map_cfg.mapspec_category,
            map_id=f"eval_ep{episode_idx:06d}",
            world_size=float(env.world_size),
            n_obstacles=int(map_cfg.mapspec_n_obstacles),
            obstacle_size=float(map_cfg.mapspec_obstacle_size),
            wall_count=int(map_cfg.mapspec_wall_count),
            wall_len_range=(float(map_cfg.mapspec_wall_len_min), float(map_cfg.mapspec_wall_len_max)),
            margin=float(map_cfg.mapspec_margin),
        )
        walls = mapspec_to_walls(map_spec)
        reset_env_with_walls(env, walls, n_agents=n_agents)
        _ = env._get_obs()  # optional
    else:
        env.reset(mt, n_agents=n_agents)

    rewarder = RewardComputer(n_agents=n_agents, cfg=reward_cfg)
    rewarder.reset(env)
    h = ac.init_hidden(n_agents, device=device)

    prev_aux_np: Optional[Dict[str, np.ndarray]] = None

    any_wall = False
    any_agent = False
    any_collision = False
    team_success = False

    dist_sum = 0.0
    lidar_sum = 0.0
    count = 0
    min_min_lidar = float("inf")

    for t in range(max_steps):
        data_t, aux_t = obs_builder.build(env)

        # Use RewardComputer's active mask for acting (not builder's goal-only active).
        active_np = rewarder.active.copy()
        mask_t = torch.from_numpy(active_np.astype(np.bool_)).to(device=device)

        data_t = data_t.to(device)
        data_t.mask = mask_t

        action, _logp, _value, h_next = ac.act(data_t, h_prev=h, mask=mask_t, deterministic=True)
        action_np = action.detach().cpu().numpy()

        # Step env with physical velocities
        env.step(action_np * float(env.max_speed))

        _, aux_post = obs_builder.build(env)
        aux_np = to_numpy_aux(aux_post, active_override=active_np)
        _r, _done_agent, info_r = rewarder.step(env, aux_np, prev_aux_np, action_np)

        prev_aux_np = aux_np
        h = h_next

        any_wall = any_wall or bool(np.any(info_r["wall_collision"]))
        any_agent = any_agent or bool(np.any(info_r["agent_collision"]))
        any_collision = any_collision or bool(info_r["any_collision"])
        team_success = team_success or bool(info_r["team_success"])

        dist_sum += float(info_r["mean_dist"])
        lidar_sum += float(info_r["mean_min_lidar"])
        count += 1
        min_min_lidar = min(min_min_lidar, float(np.min(aux_np["min_lidar"])))

        if render:
            env.render()

        if info_r["team_success"] or info_r["all_inactive"]:
            steps = t + 1
            break
    else:
        steps = max_steps

    mean_dist = dist_sum / max(1, count)
    mean_min_lidar = lidar_sum / max(1, count)

    return EpisodeResult(
        episode_idx=episode_idx,
        seed=seed,
        map_type=mt.value,
        n_agents=n_agents,
        steps=steps,
        team_success=team_success,
        any_collision=any_collision,
        wall_collision_any=any_wall,
        agent_collision_any=any_agent,
        mean_dist=mean_dist,
        mean_min_lidar=mean_min_lidar,
        min_min_lidar=min_min_lidar if np.isfinite(min_min_lidar) else float("nan"),
    )


def summarize(results: List[EpisodeResult]) -> Dict[str, Any]:
    if not results:
        return {}

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    success = [1.0 if r.team_success else 0.0 for r in results]
    wall = [1.0 if r.wall_collision_any else 0.0 for r in results]
    agent = [1.0 if r.agent_collision_any else 0.0 for r in results]
    coll = [1.0 if r.any_collision else 0.0 for r in results]

    return {
        "n_episodes": len(results),
        "success_rate": _mean(success),
        "wall_hit_rate": _mean(wall),
        "agent_hit_rate": _mean(agent),
        "any_collision_rate": _mean(coll),
        "avg_steps": _mean([float(r.steps) for r in results]),
        "avg_mean_dist": _mean([float(r.mean_dist) for r in results]),
        "avg_mean_min_lidar": _mean([float(r.mean_min_lidar) for r in results]),
        "avg_min_min_lidar": _mean([float(r.min_min_lidar) for r in results]),
    }


def write_csv(path: Path, results: List[EpisodeResult]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to RL checkpoint (actor_critic*.pt) or supervised best.pt")
    ap.add_argument("--stats_json", type=str, default=None, help="stats.json for feature normalization and dim inference")
    ap.add_argument("--out_dir", type=str, default="runs/eval", help="Output directory for summary.json and episodes.csv")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--render", action="store_true", help="Render with matplotlib (slow)")

    ap.add_argument("--map_type", type=str, default="mixed", help="One of: mixed, random, maze, maze_with_random")
    ap.add_argument("--map_source", type=str, default="mapspec", choices=["sim_env", "mapspec"])

    # If using MapSpec maps:
    ap.add_argument("--mapspec_category", type=str, default="walls_and_obstacles")
    ap.add_argument("--mapspec_n_obstacles", type=int, default=12)
    ap.add_argument("--mapspec_obstacle_size", type=float, default=0.7)
    ap.add_argument("--mapspec_wall_count", type=int, default=8)
    ap.add_argument("--mapspec_wall_len_min", type=float, default=2.0)
    ap.add_argument("--mapspec_wall_len_max", type=float, default=8.0)
    ap.add_argument("--mapspec_margin", type=float, default=0.8)
    ap.add_argument("--n_agents", type=int, default=None, help="Fixed number of agents. If omitted, sample from [min,max]")
    ap.add_argument("--n_agents_min", type=int, default=4)
    ap.add_argument("--n_agents_max", type=int, default=10)

    ap.add_argument("--world_size", type=float, default=10.0)
    ap.add_argument("--lidar_n_rays", type=int, default=64)
    ap.add_argument("--lidar_max_range", type=float, default=8.0)
    ap.add_argument("--edge_radius", type=float, default=3.0)

    # Reward weights (defaults match RewardConfig)
    ap.add_argument("--goal_reward", type=float, default=RewardConfig.goal_reward)
    ap.add_argument("--wall_collision_penalty", type=float, default=RewardConfig.wall_collision_penalty)
    ap.add_argument("--agent_collision_penalty", type=float, default=RewardConfig.agent_collision_penalty)
    ap.add_argument("--team_success_reward", type=float, default=RewardConfig.team_success_reward)
    ap.add_argument("--time_penalty", type=float, default=RewardConfig.time_penalty)
    ap.add_argument("--progress_weight", type=float, default=RewardConfig.progress_weight)
    ap.add_argument("--wall_proximity_weight", type=float, default=RewardConfig.wall_proximity_weight)
    ap.add_argument("--wall_safe_dist", type=float, default=RewardConfig.wall_safe_dist)

    args = ap.parse_args()

    map_cfg = MapGenConfig(
    map_source=args.map_source,
    mapspec_category=args.mapspec_category,
    mapspec_n_obstacles=args.mapspec_n_obstacles,
    mapspec_obstacle_size=args.mapspec_obstacle_size,
    mapspec_wall_count=args.mapspec_wall_count,
    mapspec_wall_len_min=args.mapspec_wall_len_min,
    mapspec_wall_len_max=args.mapspec_wall_len_max,
    mapspec_margin=args.mapspec_margin,
)

    set_seed(int(args.seed))

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    map_type = parse_map_type(args.map_type)

    if args.n_agents is not None:
        n_min = n_max = int(args.n_agents)
    else:
        n_min, n_max = int(args.n_agents_min), int(args.n_agents_max)
        if n_min <= 0 or n_max < n_min:
            raise ValueError("Invalid n_agents range")

    obs_cfg = GraphObsConfig(
        world_size=float(args.world_size),
        lidar_n_rays=int(args.lidar_n_rays),
        lidar_max_range=float(args.lidar_max_range),
        edge_radius=float(args.edge_radius),
    )
    obs_builder = build_obs_builder(args.stats_json, obs_cfg, device=str(device))

    node_dim, edge_dim = load_stats_dims(args.stats_json)
    ac_cfg = ActorCriticConfig(node_dim=node_dim, edge_dim=edge_dim)
    ac = GraphActorCriticGRU(ac_cfg).to(device)
    ckpt_meta = load_policy_checkpoint(args.ckpt, ac)

    r_cfg = RewardConfig(
        goal_reward=float(args.goal_reward),
        wall_collision_penalty=float(args.wall_collision_penalty),
        agent_collision_penalty=float(args.agent_collision_penalty),
        team_success_reward=float(args.team_success_reward),
        time_penalty=float(args.time_penalty),
        progress_weight=float(args.progress_weight),
        wall_proximity_weight=float(args.wall_proximity_weight),
        wall_safe_dist=float(args.wall_safe_dist),
    )

    env = MultiRobotEnv(world_size=float(args.world_size))

    results: List[EpisodeResult] = []
    for ep in range(int(args.episodes)):
        ep_seed = int(args.seed) + 1000 * ep
        set_seed(ep_seed)

        n_agents = random.randint(n_min, n_max)
        res = run_one_episode(
            env,
            obs_builder,
            ac,
            r_cfg,
            device=device,
            episode_idx=ep,
            seed=ep_seed,
            map_type=map_type,
            map_cfg=map_cfg,
            n_agents=n_agents,
            max_steps=int(args.max_steps),
            render=bool(args.render),
        )
        results.append(res)

    summ = summarize(results)

    payload = {
        "args": vars(args),
        "checkpoint": ckpt_meta,
        "obs_cfg": obs_cfg.__dict__,
        "ac_cfg": ac_cfg.__dict__,
        "reward_cfg": r_cfg.__dict__,
        "summary": summ,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if results:
        write_csv(out_dir / "episodes.csv", results)

    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
