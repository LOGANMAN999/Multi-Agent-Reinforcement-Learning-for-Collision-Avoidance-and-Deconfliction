
"""train_student_rl_ppo_gru.py

Recurrent PPO fine-tuning for the GRU student policy on live rollouts from sim_env.py.

This script is the "trainer" component of the RL stack. It ties together:

- Environment: sim_env.MultiRobotEnv
- Online graph observation: rl_graph_obs.GraphObsBuilder
- Actor-critic wrapper: rl_actor_critic_gru.GraphActorCriticGRU
- Reward + per-agent termination/masking: rl_rewards.RewardComputer
- Recurrent rollout storage + GAE + minibatches: rl_rollout_buffer_gru.RolloutBufferGRU

Design choices (aligned with your repo + earlier requirements)
------------------------------------------------------------
- One shared policy controls all agents; it is executed per-agent but with GNN message passing.
- Per-agent termination: agents that reach goals or collide are masked out; the episode continues.
- We IGNORE env.step's scalar reward and done flag; we compute rewards/dones ourselves.
- Actions are in the student's normalized units (roughly [-1,1]); env.step uses physical velocity
  so we multiply by env.max_speed when stepping.

This is an MVP trainer aimed at correctness and ease of iteration, not max throughput.
Once it works, it is straightforward to vectorize environments and/or batch graphs with PyG Batch.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

try:
    from torch_geometric.data import Data
except Exception as e:  # pragma: no cover
    raise ImportError("torch_geometric is required to run RL training") from e

# --- local imports: ensure this directory is on sys.path ---
_THIS_DIR = Path(__file__).resolve().parent          # repo/src/RL_stack
_SRC_DIR = _THIS_DIR.parent                          # repo/src

# Make `controllers`, `sim_env`, `data_building`, etc. importable
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# (Optional) also allow importing other RL_stack files without package prefix
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sim_env import MapType, MultiRobotEnv  # noqa: E402
from rl_graph_obs import GraphObsBuilder, GraphObsConfig  # noqa: E402
from rl_actor_critic_gru import ActorCriticConfig, GraphActorCriticGRU  # noqa: E402
from rl_rewards import RewardComputer, RewardConfig  # noqa: E402
from rl_rollout_buffer_gru import RolloutBufferGRU  # noqa: E402

from data_building.map_generation import generate_map as generate_mapspec  # MapSpec sampler
from data_building.dataset_generator import mapspec_to_walls, reset_env_with_walls


# ----------------------------- utils -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_numpy_aux(aux: Dict[str, torch.Tensor], active_override: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Convert aux tensors to numpy arrays for RewardComputer."""
    out: Dict[str, np.ndarray] = {
        "dist_to_goal": aux["dist_to_goal"].detach().cpu().numpy(),
        "min_lidar": aux["min_lidar"].detach().cpu().numpy(),
    }
    if active_override is not None:
        out["active"] = active_override.astype(np.bool_)
    elif "active" in aux:
        # builder active (goal-based) if no override
        out["active"] = aux["active"].detach().cpu().numpy().astype(np.bool_)
    return out


def build_data_from_step(
    x_bn: torch.Tensor,                 # (N,Dx)
    edge_index: torch.Tensor,           # (2,E)
    edge_attr: torch.Tensor,            # (E,De)
    mask_n: torch.Tensor,               # (N,)
) -> Data:
    return Data(x=x_bn, edge_index=edge_index, edge_attr=edge_attr, mask=mask_n)


# ----------------------------- evaluation -----------------------------

@torch.no_grad()
def evaluate_policy(
    env: MultiRobotEnv,
    obs_builder: GraphObsBuilder,
    ac: GraphActorCriticGRU,
    reward_cfg: RewardConfig,
    *,
    device: torch.device,
    n_eval_episodes: int = 10,
    max_steps: int = 400,
    n_agents: int = 8,
    map_source: str = "mapspec",
    seed: int = 0,
    mapspec_category: str = "walls_and_obstacles",
    mapspec_n_obstacles: int = 12,
    mapspec_obstacle_size: float = 0.7,
    mapspec_wall_count: int = 8,
    mapspec_wall_len_min: float = 2.0,
    mapspec_wall_len_max: float = 8.0,
    mapspec_margin: float = 0.8,
) -> Dict[str, float]:
    """Run deterministic evaluation episodes.

    Important invariants (kept consistent with the training loop):
      - We act only on `rewarder.active` (not builder's goal-only active).
      - We step the env with *physical* velocities: action * env.max_speed.
      - We compute rewards from *post-step* aux (distance/min_lidar at s_{t+1}).
      - We support MapSpec-generated maps (map_source='mapspec') so eval matches training.
    """
    ac.eval()

    succ = 0
    wall_hits = 0
    agent_hits = 0
    mean_steps = []

    for ep_idx in range(int(n_eval_episodes)):
        # --- reset env ---
        if map_source == "mapspec":
            rng = np.random.default_rng(int(seed) + 10007 * (ep_idx + 1))
            map_spec = generate_mapspec(
                rng=rng,
                category=mapspec_category,
                map_id=f"rl_eval_ep{ep_idx:04d}",
                world_size=float(env.world_size),
                n_obstacles=int(mapspec_n_obstacles),
                obstacle_size=float(mapspec_obstacle_size),
                wall_count=int(mapspec_wall_count),
                wall_len_range=(float(mapspec_wall_len_min), float(mapspec_wall_len_max)),
                margin=float(mapspec_margin),
            )
            walls = mapspec_to_walls(map_spec)
            reset_env_with_walls(env, walls, n_agents=int(n_agents))
            _ = env._get_obs()  # ensures obs buffers are fresh
        else:
            # Fallback: use built-in sim_env map types
            mt = random.choice(list(MapType))
            env.reset(mt, n_agents=int(n_agents))

        rewarder = RewardComputer(n_agents=int(n_agents), cfg=reward_cfg)
        rewarder.reset(env)
        h = ac.init_hidden(int(n_agents), device=device)

        prev_aux_np = None
        episode_steps = 0

        while episode_steps < int(max_steps):
            # Build graph from current env state
            data_t, _aux_pre = obs_builder.build(env)

            # Act only for currently-active agents (rewarder mask)
            active_np = rewarder.active.copy()
            mask_t = torch.from_numpy(active_np.astype(np.bool_)).to(device=device)

            data_t = data_t.to(device)
            data_t.mask = mask_t

            action, _logp, _value, h_next = ac.act(
                data_t, h_prev=h, mask=mask_t, deterministic=True
            )
            action_np = action.detach().cpu().numpy()

            # Step env with physical velocities
            env.step(action_np * float(env.max_speed))

            # Post-step aux for reward shaping (s_{t+1})
            _data_post, aux_post = obs_builder.build(env)
            aux_np = to_numpy_aux(aux_post, active_override=active_np)

            _r, _done_agent, info_r = rewarder.step(env, aux_np, prev_aux_np, action_np)

            wall_hits += int(np.any(info_r["wall_collision"]))
            agent_hits += int(np.any(info_r["agent_collision"]))

            h = h_next
            prev_aux_np = aux_np
            episode_steps += 1

            if info_r["team_success"]:
                succ += 1
                break
            if info_r["all_inactive"]:
                break

        mean_steps.append(episode_steps)

    return {
        "success_rate": succ / max(1, int(n_eval_episodes)),
        "wall_hit_rate": wall_hits / max(1, int(n_eval_episodes)),
        "agent_hit_rate": agent_hits / max(1, int(n_eval_episodes)),
        "avg_steps": float(np.mean(mean_steps)) if mean_steps else 0.0,
    }


# ----------------------------- PPO update -----------------------------

def ppo_update_on_batch(
    ac: GraphActorCriticGRU,
    batch,
    *,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    """One PPO gradient step over a SequenceBatch (recurrent unroll inside)."""
    ac.train()

    B, L, N, _Dx = batch.x.shape
    A = batch.actions.shape[-1]

    # We'll unroll per sequence, per timestep, calling evaluate_actions on each graph.
    logp_new = torch.zeros((B, L, N), device=batch.x.device, dtype=torch.float32)
    entropy = torch.zeros((B, L, N), device=batch.x.device, dtype=torch.float32)
    value_new = torch.zeros((B, L, N), device=batch.x.device, dtype=torch.float32)

    for b in range(B):
        h = batch.h0[b]  # (N,H)
        for t in range(L):
            data_t = build_data_from_step(
                batch.x[b, t],
                batch.edge_index[t][b],
                batch.edge_attr[t][b],
                batch.mask[b, t],
            ).to(batch.x.device)

            lp, ent, v, h_next = ac.evaluate_actions(
                data_t,
                batch.actions[b, t],
                h_prev=h,
                mask=batch.mask[b, t],
            )
            logp_new[b, t] = lp
            entropy[b, t] = ent
            value_new[b, t] = v
            h = h_next

    # Masks
    m = batch.mask.to(dtype=torch.float32)
    denom = m.sum().clamp_min(1.0)

    # Policy loss
    ratio = torch.exp(logp_new - batch.logp_old)
    surr1 = ratio * batch.advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch.advantages
    policy_loss = -torch.minimum(surr1, surr2)
    policy_loss = (policy_loss * m).sum() / denom

    # Value loss
    v_loss = 0.5 * ((value_new - batch.returns) ** 2)
    v_loss = (v_loss * m).sum() / denom

    # Entropy bonus (maximize entropy => subtract negative)
    ent = (entropy * m).sum() / denom

    loss = policy_loss + vf_coef * v_loss - ent_coef * ent

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(v_loss.detach().cpu()),
        "entropy": float(ent.detach().cpu()),
        "approx_kl": float(((batch.logp_old - logp_new) * m).sum().detach().cpu() / denom.detach().cpu()),
    }


# ----------------------------- main training loop -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recurrent PPO fine-tuning for StudentGNNGRU")

    # Initialization
    p.add_argument("--supervised_ckpt", type=str,
                default="datasets/il_dataset/processed_student_v1/checkpoints/best.pt",
                help="Path to supervised best.pt to initialize actor.")
    p.add_argument("--stats_json", type=str,
                default="datasets/il_dataset/processed_student_v1/stats.json", help="Path to stats.json (node/edge mean/std).")

    # Environment
    p.add_argument("--n_agents", type=int, default=4)
    #p.add_argument("--map_type", type=str, default="random", choices=[m.value for m in MapType] + ["mixed"])
    p.add_argument("--map_source", type=str, default="mapspec", choices=["sim_env", "mapspec"])
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--mapspec_category", type=str, default="walls_and_obstacles")
    p.add_argument("--mapspec_n_obstacles", type=int, default=12)
    p.add_argument("--mapspec_obstacle_size", type=float, default=0.7)
    p.add_argument("--mapspec_wall_count", type=int, default=5)
    p.add_argument("--mapspec_wall_len_min", type=float, default=2.0)
    p.add_argument("--mapspec_wall_len_max", type=float, default=8.0)
    p.add_argument("--mapspec_margin", type=float, default=0.8)

    # PPO rollout + update
    p.add_argument("--horizon", type=int, default=256, help="Steps per rollout (buffer horizon).")
    p.add_argument("--seq_len", type=int, default=32, help="Truncated BPTT length.")
    p.add_argument("--stride", type=int, default=None, help="Sequence stride for minibatches. Default: seq_len.")
    p.add_argument("--batch_size", type=int, default=4, help="#sequences per minibatch.")
    p.add_argument("--ppo_epochs", type=int, default=4, help="Optimization epochs per rollout.")
    p.add_argument("--updates", type=int, default=200)

    # PPO hyperparameters
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Reward weights (override defaults)
    p.add_argument("--goal_reward", type=float, default=10.0)
    p.add_argument("--wall_collision_penalty", type=float, default=20.0)
    p.add_argument("--agent_collision_penalty", type=float, default=15.0)
    p.add_argument("--team_success_reward", type=float, default=20.0)
    p.add_argument("--time_penalty", type=float, default=0.01)
    p.add_argument("--progress_weight", type=float, default=1.0)
    p.add_argument("--wall_proximity_weight", type=float, default=0.5)
    p.add_argument("--wall_safe_dist", type=float, default=0.75)

    # Logging / saving
    p.add_argument("--run_dir", type=str, default="runs/rl_ppo_gru")
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--eval_every", type=int, default=25)
    p.add_argument("--n_eval_episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    # Save config for reproducibility
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    # Build env
    env = MultiRobotEnv()

    # Graph obs builder (uses stats.json normalization)
    obs_builder = GraphObsBuilder.from_stats_json(args.stats_json, cfg=GraphObsConfig(), device=str(device))

    # Actor-critic
    node_dim = int(obs_builder.node_mean.numel()) if obs_builder.node_mean is not None else 70
    edge_dim = int(obs_builder.edge_mean.numel()) if obs_builder.edge_mean is not None else 5

    ac_cfg = ActorCriticConfig(
        node_dim=node_dim,
        edge_dim=edge_dim,
        gnn_hidden_dim=128,
        gru_hidden_dim=128,
        num_layers=3,
        dropout=0.0,
        action_dim=2,
        max_speed=1.0,
    )
    ac = GraphActorCriticGRU(ac_cfg).to(device)
    ac.load_supervised_student(args.supervised_ckpt, strict=True)

    optimizer = Adam(ac.parameters(), lr=args.lr)

    # Reward config
    reward_cfg = RewardConfig(
        goal_reward=args.goal_reward,
        wall_collision_penalty=args.wall_collision_penalty,
        agent_collision_penalty=args.agent_collision_penalty,
        team_success_reward=args.team_success_reward,
        time_penalty=args.time_penalty,
        progress_weight=args.progress_weight,
        wall_proximity_weight=args.wall_proximity_weight,
        wall_safe_dist=args.wall_safe_dist,
    )

    # Rollout buffer
    buf = RolloutBufferGRU(
        horizon=args.horizon,
        num_agents=args.n_agents,
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=ac_cfg.gru_hidden_dim,
        action_dim=ac_cfg.action_dim,
        store_on_cpu=True,
    )

    best_success = -1.0

    for upd in range(1, args.updates + 1):
        # --- collect one rollout (single episode segment) ---
        buf.reset()

        
        #env.reset(mt, n_agents=args.n_agents)
        if args.map_source == "mapspec":
            # Deterministic per-update RNG (still random, but reproducible)
            rng = np.random.default_rng(args.seed + 10007 * upd)

            map_spec = generate_mapspec(
                rng=rng,
                category=args.mapspec_category,
                map_id=f"rl_upd{upd:06d}",
                world_size=float(env.world_size),
                n_obstacles=int(args.mapspec_n_obstacles),
                obstacle_size=float(args.mapspec_obstacle_size),
                wall_count=int(args.mapspec_wall_count),
                wall_len_range=(float(args.mapspec_wall_len_min), float(args.mapspec_wall_len_max)),
                margin=float(args.mapspec_margin),
            )
            walls = mapspec_to_walls(map_spec)   # MapSpec -> List[Wall]
            reset_env_with_walls(env, walls, n_agents=args.n_agents)
            _ = env._get_obs()  # optional; ensures obs is consistent immediately

        rewarder = RewardComputer(n_agents=args.n_agents, cfg=reward_cfg)
        rewarder.reset(env)

        h = ac.init_hidden(args.n_agents, device=device)
        prev_aux_np = None

        # episode metrics
        ep_return = np.zeros((args.n_agents,), dtype=np.float32)
        ep_wall = 0
        ep_agent = 0
        ep_goal = 0
        team_success = False

        for t in range(args.horizon):
            data_t, aux_t = obs_builder.build(env)

            active_np = rewarder.active.copy()
            mask_t = torch.from_numpy(active_np.astype(np.bool_)).to(device=device)

            data_t = data_t.to(device)
            data_t.mask = mask_t

            action, logp, value, h_next = ac.act(data_t, h_prev=h, mask=mask_t, deterministic=False)

            action_np = action.detach().cpu().numpy()
            # Step env (physical vel)
            _obs, _r, _done, _info = env.step(action_np * float(env.max_speed))

            _, aux_post = obs_builder.build(env)
            aux_np = to_numpy_aux(aux_post, active_override=active_np)

            rewards, done_agent, info_r = rewarder.step(env, aux_np, prev_aux_np, action_np)

            # metrics
            ep_return += rewards.astype(np.float32)
            ep_wall += int(np.any(info_r["wall_collision"]))
            ep_agent += int(np.any(info_r["agent_collision"]))
            ep_goal += int(np.any(info_r["reached"]))
            team_success = team_success or bool(info_r["team_success"])

            # store to buffer (store h BEFORE update)
            buf.add(
                data=Data(x=data_t.x.detach().cpu(), edge_index=data_t.edge_index.detach().cpu(), edge_attr=data_t.edge_attr.detach().cpu(), mask=data_t.mask.detach().cpu()),
                h_prev=h.detach().cpu(),
                action=action.detach().cpu(),
                logp=logp.detach().cpu(),
                value=value.detach().cpu(),
                reward=rewards,
                done=done_agent,
            )

            h = h_next
            prev_aux_np = aux_np

            if info_r["all_inactive"]:
                break

        # Bootstrap last value
        with torch.no_grad():
            data_last, aux_last = obs_builder.build(env)
            active_np = rewarder.active.copy()
            mask_last = torch.from_numpy(active_np.astype(np.bool_)).to(device=device)
            data_last = data_last.to(device)
            data_last.mask = mask_last
            _a, _lp, v_last, _h = ac.act(data_last, h_prev=h, mask=mask_last, deterministic=True)
        buf.compute_returns_and_advantages(gamma=args.gamma, lam=args.lam, last_value=v_last.detach().cpu())

        # --- PPO update ---
        logs = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        n_steps = 0

        for _epoch in range(args.ppo_epochs):
            for batch in buf.iter_minibatches(seq_len=args.seq_len, batch_size=args.batch_size, device=device, shuffle=True, stride=args.stride):
                out = ppo_update_on_batch(
                    ac,
                    batch,
                    clip_eps=args.clip_eps,
                    vf_coef=args.vf_coef,
                    ent_coef=args.ent_coef,
                    max_grad_norm=args.max_grad_norm,
                    optimizer=optimizer,
                )
                for k in logs:
                    logs[k] += out[k]
                n_steps += 1

        for k in logs:
            logs[k] /= max(1, n_steps)

        # --- periodic eval + checkpoint ---
        if (upd % args.eval_every) == 0 or upd == 1:
            eval_metrics = evaluate_policy(
                env,
                obs_builder,
                ac,
                reward_cfg,
                device=device,
                n_eval_episodes=args.n_eval_episodes,
                max_steps=args.max_steps,
                n_agents=args.n_agents,
                map_source=args.map_source,
                seed=args.seed + 424242 * upd,
                mapspec_category=args.mapspec_category,
                mapspec_n_obstacles=args.mapspec_n_obstacles,
                mapspec_obstacle_size=args.mapspec_obstacle_size,
                mapspec_wall_count=args.mapspec_wall_count,
                mapspec_wall_len_min=args.mapspec_wall_len_min,
                mapspec_wall_len_max=args.mapspec_wall_len_max,
                mapspec_margin=args.mapspec_margin,
            )
            # Save "best" by success rate
            if eval_metrics["success_rate"] > best_success:
                best_success = eval_metrics["success_rate"]
                ac.save(ckpt_dir / "best_rl.pt", extra={"update": upd, "eval": eval_metrics, "reward_cfg": asdict(reward_cfg)})

            # Always save latest
            ac.save(ckpt_dir / "latest_rl.pt", extra={"update": upd, "eval": eval_metrics, "reward_cfg": asdict(reward_cfg)})

            print(
                f"[upd {upd:04d}] "
                f"ret(mean)={float(np.mean(ep_return)):.2f} goals={ep_goal} wallE={ep_wall} agentE={ep_agent} "
                f"team_success={int(team_success)} | "
                f"loss={logs['loss']:.3f} pi={logs['policy_loss']:.3f} v={logs['value_loss']:.3f} "
                f"ent={logs['entropy']:.3f} kl={logs['approx_kl']:.4f} | "
                f"eval_succ={eval_metrics['success_rate']:.2f} eval_wall={eval_metrics['wall_hit_rate']:.2f} eval_agent={eval_metrics['agent_hit_rate']:.2f} "
            )
        elif (upd % args.save_every) == 0:
            ac.save(ckpt_dir / f"ckpt_upd{upd:04d}.pt", extra={"update": upd, "reward_cfg": asdict(reward_cfg)})
            print(
                f"[upd {upd:04d}] "
                f"ret(mean)={float(np.mean(ep_return)):.2f} goals={ep_goal} wallE={ep_wall} agentE={ep_agent} "
                f"loss={logs['loss']:.3f} pi={logs['policy_loss']:.3f} v={logs['value_loss']:.3f} ent={logs['entropy']:.3f}"
            )

    print(f"Done. Best eval success_rate={best_success:.3f}. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()
