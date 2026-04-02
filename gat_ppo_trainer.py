import os
import csv
import time
import collections
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from RL_stack.gat_graph_builder import build_graph, INTERACTION_RADIUS
from RL_stack.gat_deconfliction_policy import GATDeconflictionPolicy
from RL_stack.priority_protocol import (
    HarmonicPriorityManager,
    compute_connected_components,
)
from controllers.gat_deconfliction_controller import compute_repulsion_params
from RL_stack.deconfliction_reward import compute_rewards_harmonic
from RL_stack.gat_rollout_buffer import GATEpisodeBuffer
from controllers.harmonic_navigation import HarmonicNavigationController
from sim_env import MultiRobotEnv, Wall

import sys as _sys, os as _os
_THIS_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _THIS_DIR not in _sys.path:
    _sys.path.insert(0, _THIS_DIR)
from data_building.map_generation import generate_map as generate_map_spec

# Harmonic flow scan parameters — must match watch_rl_episode / gat_deconfliction_controller
_PRUNE_LOOKAHEAD_STEPS = 100
_PRUNE_GOAL_STOP       = 1.0
_PRUNE_SAFETY_MARGIN   = 0.15


DEFAULT_CONFIG = {
    # Rollout: collect complete episodes until total steps >= n_steps
    "n_steps":           1000,
    # PPO update
    "n_epochs":          10,
    "gamma":             0.99,
    "gae_lambda":        0.95,
    "clip_epsilon":      0.2,
    "value_coef":        0.5,
    "entropy_coef":      0.01,
    "sheaf_coef":        0.05,
    "max_grad_norm":     0.5,
    "target_kl":         0.02,
    "lr":                1e-3,
    "lr_schedule":       "constant",
    # Environment
    "n_agents":          45,
    "interaction_radius": INTERACTION_RADIUS,
    "curriculum":        [],
    # Logging / saving
    "log_dir":           "runs/sheaf_deconfliction",
    "save_dir":          "checkpoints/sheaf_deconfliction_v2",
    "save_every":        10,
    "log_every":         10,
    "total_episodes":    1000,
    "max_episode_steps": 450,
    # Architecture
    "hidden_dim":        128,
    "n_heads":           4,
    "world_size":        10.0,
    # Map generation
    "map_category":      "walls_and_obstacles",
    "map_n_obstacles":   20,
    "map_obstacle_size": 0.7,
    "map_wall_count":    15,
    "map_wall_len_min":  1.5,
    "map_wall_len_max":  4.0,
    "map_margin":        0.8,
    "map_base_seed":     41,
    # Reward shaping
    "progress_coef":     1.0,    # scale for per-step distance-reduction reward
    # Repulsion modulation
    "base_repulsion_radius":   1.5,
    "base_repulsion_strength": 5.0,
}


def _manual_reset_env(
    env: MultiRobotEnv,
    n_agents: int,
    config: Dict,
    episode_seed: int = 0,
) -> np.ndarray:
   
    seed = int(config.get("map_base_seed", 42)) + episode_seed
    rng  = np.random.default_rng(seed)

    map_spec = generate_map_spec(
        rng=rng,
        category=config.get("map_category", "walls_and_obstacles"),
        map_id=episode_seed,
        world_size=float(config.get("world_size", 10.0)),
        n_obstacles=int(config.get("map_n_obstacles", 3)),
        obstacle_size=float(config.get("map_obstacle_size", 0.5)),
        wall_count=int(config.get("map_wall_count", 2)),
        wall_len_range=(
            float(config.get("map_wall_len_min", 1.5)),
            float(config.get("map_wall_len_max", 4.0)),
        ),
        margin=float(config.get("map_margin", 0.8)),
    )
    env.walls     = [Wall(s.x1, s.y1, s.x2, s.y2) for s in map_spec.all_wall_segments()]
    env.map_type  = None
    env.n_agents  = n_agents
    env.t         = 0
    env.positions = env._sample_non_colliding_points(n_agents)
    env.goals     = env._sample_non_colliding_points(n_agents)
    return env._get_obs()


def _compute_headings_vec(
    velocities: np.ndarray,
    prev_headings: Optional[np.ndarray] = None,
) -> np.ndarray:
    speeds = np.linalg.norm(velocities, axis=1)
    new_h  = np.arctan2(velocities[:, 1], velocities[:, 0])
    moving = speeds > 1e-6
    if prev_headings is not None:
        return np.where(moving, new_h, prev_headings)
    return np.where(moving, new_h, 0.0)


def _attach_component_sizes(
    graph,
    edge_index_np: np.ndarray,
    N: int,
    at_goal: np.ndarray,
    active_mask: np.ndarray,
    device: torch.device,
) -> None:
    """Compute component sizes and attach as graph.component_sizes tensor."""
    _, component_map = compute_connected_components(
        edge_index_np, N, at_goal, active_mask
    )
    component_sizes = np.zeros(N, dtype=np.int64)
    for members in component_map.values():
        sz = len(members)
        for idx in members:
            component_sizes[idx] = sz
    graph.component_sizes = torch.tensor(component_sizes, dtype=torch.long, device=device)


class GATDeconflictionTrainer:

    def __init__(self, policy: GATDeconflictionPolicy, config: Dict):
        self.policy  = policy
        self.config  = config
        self.device  = next(policy.parameters()).device

        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=float(config["lr"])
        )

        total_updates = max(1, int(config.get("total_episodes", 1000)) //
                           max(1, int(config["n_steps"] // config.get("max_episode_steps", 400))))
        if config.get("lr_schedule") == "linear":
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.1,
                total_iters=max(1, total_updates),
            )
        else:
            self.lr_scheduler = None

        self.curriculum_stages    = config.get("curriculum", [])
        self.curriculum_stage_idx = 0
        self.n_agents             = int(config.get("n_agents", 4))
        if self.curriculum_stages:
            self.n_agents = int(self.curriculum_stages[0]["n_agents"])

        self.episode_count           = 0
        self.update_count            = 0
        self.last_logged_episode     = 0
        self.last_saved_episode      = 0
        self.recent_collision_rates: collections.deque = collections.deque(maxlen=100)

        self.buffer = GATEpisodeBuffer(
            n_steps    = int(config["n_steps"]),
            n_agents   = self.n_agents,
            gamma      = float(config["gamma"]),
            gae_lambda = float(config["gae_lambda"]),
        )

        self.env              = MultiRobotEnv(world_size=float(config.get("world_size", 10.0)))
        self.controller       = HarmonicNavigationController()
        self.priority_manager = HarmonicPriorityManager()

        log_dir  = config.get("log_dir",  "runs/sheaf_deconfliction")
        save_dir = config.get("save_dir", "checkpoints/sheaf_deconfliction")
        os.makedirs(log_dir,  exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        self.log_dir  = log_dir
        self.save_dir = save_dir

        self.csv_episode_path = os.path.join(log_dir, "episode_metrics.csv")
        self.csv_update_path  = os.path.join(log_dir, "update_metrics.csv")
        self._init_csv_files()

        self.writer = None
        if _TENSORBOARD_AVAILABLE:
            try:
                self.writer = SummaryWriter(log_dir=log_dir)
            except Exception:
                self.writer = None

    def _init_csv_files(self):
        ep_headers = [
            "episode", "mean_total_reward", "goals_reached", "goal_completion_rate",
            "collision_count", "collision_rate", "timeout_rate",
            "mean_episode_length", "n_agents", "curriculum_stage",
        ]
        upd_headers = [
            "update", "policy_loss", "value_loss", "entropy", "sheaf_loss",
            "approx_kl", "clip_fraction", "explained_variance",
        ]
        if not os.path.exists(self.csv_episode_path):
            with open(self.csv_episode_path, "w", newline="") as f:
                csv.writer(f).writerow(ep_headers)
        if not os.path.exists(self.csv_update_path):
            with open(self.csv_update_path, "w", newline="") as f:
                csv.writer(f).writerow(upd_headers)

    def _advance_curriculum(self):
        if self.curriculum_stage_idx >= len(self.curriculum_stages) - 1:
            return
        threshold = self.curriculum_stages[self.curriculum_stage_idx].get("collision_rate_threshold")
        if threshold is None:
            return
        if len(self.recent_collision_rates) < 20:
            return
        mean_cr = float(np.mean(list(self.recent_collision_rates)))
        if mean_cr < threshold:
            self.curriculum_stage_idx += 1
            new_stage = self.curriculum_stages[self.curriculum_stage_idx]
            self.n_agents = int(new_stage["n_agents"])
            self.buffer = GATEpisodeBuffer(
                n_steps    = int(self.config["n_steps"]),
                n_agents   = self.n_agents,
                gamma      = float(self.config["gamma"]),
                gae_lambda = float(self.config["gae_lambda"]),
            )
            print(f"[Curriculum] Stage {self.curriculum_stage_idx}: "
                  f"n_agents={self.n_agents} (collision_rate={mean_cr:.3f})")

    def _reset_episode(self) -> int:

        _manual_reset_env(self.env, self.n_agents, self.config, episode_seed=self.episode_count)

        # Harmonic flow scan — remove agents whose solo path would hit a wall.
        # Copied verbatim from watch_rl_episode.prune_harmonic_colliders().
        prune_ctrl = HarmonicNavigationController()
        prune_ctrl.reset(self.env)
        positions = np.array(self.env.positions, dtype=np.float32)
        goals     = np.array(self.env.goals,     dtype=np.float32)
        dt        = float(getattr(self.env, "dt", 0.1))
        hits = prune_ctrl.simulate_flow_hits_wall(
            positions=positions,
            dt=dt,
            lookahead_steps=_PRUNE_LOOKAHEAD_STEPS,
            goals=goals,
            goal_proximity_stop=_PRUNE_GOAL_STOP,
            safety_margin=_PRUNE_SAFETY_MARGIN,
        )
        keep = ~hits
        if int(hits.sum()) > 0:
            self.env.positions = self.env.positions[keep]
            self.env.goals     = self.env.goals[keep]
            self.env.n_agents  = int(keep.sum())
            self.env._get_obs()

        self.controller.reset(self.env)
        self.priority_manager.reset()
        return self.env.n_agents

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(self) -> Dict:
        max_ep_steps = int(self.config.get("max_episode_steps", 400))

        self.buffer.clear()
        self.policy.eval()

        log_every = int(self.config.get("log_every", 10))

        ep_rewards_list    = []
        ep_lengths_list    = []
        ep_goals_list      = []
        ep_collisions_list = []
        ep_timeouts_list   = []
        ep_n_agents_list   = []   # post-pruning N per episode

        while not self.buffer.is_ready:

            # ----------------------------------------------------------------
            # Episode initialisation
            # ----------------------------------------------------------------
            N = self._reset_episode()   # actual agent count after pruning
            self.buffer.n_agents = N    # keep GAE reshape in sync
            self.buffer.start_episode()
            hidden           = self.policy.init_hidden(N, self.device)
            agent_velocities = np.zeros((N, 2), dtype=float)
            agent_headings   = np.zeros(N, dtype=float)
            time_since_prog  = np.zeros(N, dtype=int)
            best_dist_to_goal = np.linalg.norm(
                self.env.positions - self.env.goals, axis=1
            )

            ep_steps         = 0
            ep_total_reward  = np.zeros(N, dtype=float)
            ep_goals_reached = np.zeros(N, dtype=bool)
            ep_collision     = False
            ep_timeout       = False
            collided_mask    = np.zeros(N, dtype=bool)
            at_goal          = np.zeros(N, dtype=bool)   # first-arrival only
            goal_reward_given = np.zeros(N, dtype=bool)
            episode_done     = False

            # ----------------------------------------------------------------
            # Episode step loop
            # ----------------------------------------------------------------
            while not episode_done:

                active_mask = ~collided_mask   # active = not collided

                # Harmonic preferred velocities for ALL agents
                harmonic_vels = np.asarray(self.controller(self.env), dtype=float)

                # Graph: edges only between active AND not-at-goal agents
                graph_active = active_mask & ~at_goal
                graph = build_graph(
                    self.env, harmonic_vels, agent_velocities,
                    agent_headings, time_since_prog, active_mask=graph_active,
                ).to(self.device)

                edge_idx_np = (
                    graph.edge_index.cpu().numpy()
                    if graph.edge_index.numel() > 0
                    else np.zeros((2, 0), dtype=np.int64)
                )

                # Attach component sizes for sheaf policy input
                _attach_component_sizes(
                    graph, edge_idx_np, N, at_goal, active_mask, self.device
                )

                active_t = torch.tensor(
                    active_mask.astype(float), dtype=torch.float32, device=self.device
                )

                # Policy forward (no grad during collection)
                with torch.no_grad():
                    priority_mean, values_raw, new_hidden, _, _ = \
                        self.policy.forward(graph, hidden, active_mask=active_t)

                    std  = self.policy.log_std.exp()
                    dist = torch.distributions.Normal(priority_mean.squeeze(-1), std)

                    priority_sampled = dist.rsample()       # [N]
                    log_probs        = dist.log_prob(priority_sampled)  # [N]
                    values           = values_raw.squeeze(-1)           # [N]

                actions = {"priority_score": priority_sampled}

                # Priority protocol -> final velocities
                priority_np = priority_sampled.cpu().numpy()

                ranks, component_sizes_arr, proto_info = self.priority_manager.step(
                    priority_scores = priority_np,
                    edge_index_np   = edge_idx_np,
                    N               = N,
                    active_mask     = active_mask,
                    at_goal         = at_goal,
                )

                repulsion_radii, repulsion_strengths = compute_repulsion_params(
                    ranks           = ranks,
                    component_sizes = component_sizes_arr,
                    n_agents        = N,
                    base_radius     = float(self.config.get("base_repulsion_radius", 1.5)),
                    base_strength   = float(self.config.get("base_repulsion_strength", 5.0)),
                    at_goal         = at_goal,
                )

                final_vels = np.asarray(
                    self.controller(
                        self.env,
                        repulsion_radii     = repulsion_radii,
                        repulsion_strengths = repulsion_strengths,
                    ),
                    dtype=float,
                )
                final_vels[collided_mask] = 0.0

                prev_positions     = self.env.positions.copy()
                prev_dist_to_goal = np.linalg.norm(prev_positions - self.env.goals, axis=1)
                self.env.step(final_vels)
                curr_positions = self.env.positions.copy()

                new_agent_vels = (curr_positions - prev_positions) / self.env.dt
                new_headings   = _compute_headings_vec(new_agent_vels, agent_headings)

                # Collision check — already-inactive agents cannot newly collide
                wall_cols, robot_cols = self.env.check_per_agent_collisions_vec(curr_positions)
                wall_cols[collided_mask]  = False
                robot_cols[collided_mask] = False
                collided_agents = wall_cols | robot_cols
                collided_mask  |= collided_agents

                # Goal check — only active (non-collided) agents
                active_mask        = ~collided_mask
                dists              = np.linalg.norm(curr_positions - self.env.goals, axis=1)
                goals_reached_step = active_mask & (dists < self.env.goal_tolerance)

                # First-arrival only: gate with goal_reward_given
                newly_rewarded     = goals_reached_step & ~goal_reward_given
                goal_reward_given |= newly_rewarded

                # at_goal persists once set
                at_goal |= newly_rewarded

                # Episode ends when every agent has reached goal or collided
                all_done = bool(np.all(at_goal | collided_mask))

                # Progress tracking
                improved          = active_mask & (dists < best_dist_to_goal - 0.01)
                best_dist_to_goal = np.where(improved, dists, best_dist_to_goal)
                time_since_prog   = np.where(
                    improved, 0, np.where(active_mask, time_since_prog + 1, 0)
                ).astype(int)

                ep_steps   += 1
                is_timeout  = (ep_steps >= max_ep_steps) and not all_done

                perfect_episode = all_done and not np.any(collided_mask)

                reward_dict = compute_rewards_harmonic(
                    active_agents      = active_mask,
                    newly_reached_goal = newly_rewarded,
                    collided_agents    = collided_agents,
                    n_agents           = N,
                    prev_dist_to_goal  = prev_dist_to_goal,
                    curr_dist_to_goal  = dists,
                    progress_coef      = float(self.config.get("progress_coef", 1.0)),
                    goal_reached_ever  = at_goal,
                    is_timeout         = is_timeout,
                    perfect_episode    = perfect_episode,
                )
                rewards_np = reward_dict["per_agent"]

                ep_total_reward  += rewards_np
                ep_goals_reached |= newly_rewarded
                if np.any(collided_agents): ep_collision = True
                if is_timeout:              ep_timeout   = True

                self.buffer.add_step(
                    graph       = graph.cpu(),
                    active_mask = active_mask,
                    actions     = {k: v.cpu() for k, v in actions.items()},
                    log_probs   = log_probs.cpu(),
                    values      = values.cpu(),
                    rewards     = rewards_np,
                )

                # Advance GRU; freeze collided agents at zero
                hidden         = new_hidden.detach()
                not_collided_t = torch.tensor(
                    active_mask.astype(float), dtype=torch.float32, device=self.device
                )
                hidden = hidden * not_collided_t.unsqueeze(-1)

                agent_velocities = new_agent_vels
                agent_headings   = new_headings

                episode_done = all_done or (ep_steps >= max_ep_steps)

                if self.writer is not None and ep_steps % 50 == 0:
                    try:
                        for k, v in proto_info.items():
                            if isinstance(v, (int, float)):
                                self.writer.add_scalar(
                                    f"protocol/{k}", float(v),
                                    self.episode_count * 1000 + ep_steps,
                                )
                    except Exception:
                        pass

            # ----------------------------------------------------------------
            # Episode complete
            # ----------------------------------------------------------------
            self.buffer.finish_episode()
            self.episode_count += 1

            if (self.episode_count - self.last_logged_episode >= log_every):
                col_r = float(np.mean(ep_collisions_list[-log_every:])) if ep_collisions_list else float(ep_collision)
                rew_r = float(np.mean(ep_rewards_list[-log_every:]))    if ep_rewards_list    else float(np.mean(ep_total_reward))
                gcr   = float(np.mean([g / self.n_agents for g in ep_goals_list[-log_every:]])) if ep_goals_list else 0.0
                print(
                    f"[Ep {self.episode_count:5d}] "
                    f"reward={rew_r:7.2f}  "
                    f"goals={gcr:.2f}  "
                    f"col={col_r:.2f}  "
                    f"steps={self.buffer.total_steps}  "
                    f"n={N}/{self.n_agents}  (collecting...)"
                )
                self.last_logged_episode = self.episode_count

            self.recent_collision_rates.append(float(ep_collision))
            ep_rewards_list.append(float(np.mean(ep_total_reward)))
            ep_lengths_list.append(ep_steps)
            ep_goals_list.append(int(np.sum(ep_goals_reached)))
            ep_collisions_list.append(int(ep_collision))
            ep_timeouts_list.append(int(ep_timeout))
            ep_n_agents_list.append(N)

        mean_n_pruned = float(np.mean(ep_n_agents_list)) if ep_n_agents_list else float(self.n_agents)
        return {
            "mean_total_reward":    float(np.mean(ep_rewards_list)),
            "goals_reached":        float(np.mean(ep_goals_list)),
            "goal_completion_rate": float(np.mean([g / n for g, n in zip(ep_goals_list, ep_n_agents_list)])),
            "collision_count":      float(sum(ep_collisions_list)),
            "collision_rate":       float(np.mean(ep_collisions_list)),
            "timeout_rate":         float(np.mean(ep_timeouts_list)),
            "mean_episode_length":  float(np.mean(ep_lengths_list)),
            "n_agents":             self.n_agents,
            "n_agents_pruned":      mean_n_pruned,
            "curriculum_stage":     self.curriculum_stage_idx,
        }

    # ------------------------------------------------------------------
    # Policy update — full-sequence GRU replay
    # ------------------------------------------------------------------

    def update_policy(self) -> Dict:
        self.policy.train()
        clip_eps  = float(self.config["clip_epsilon"])
        c_v       = float(self.config["value_coef"])
        c_e       = float(self.config["entropy_coef"])
        c_s       = float(self.config.get("sheaf_coef", 0.01))
        max_grad  = float(self.config["max_grad_norm"])
        n_epochs  = int(self.config["n_epochs"])
        target_kl = float(self.config.get("target_kl", float("inf")))

        # Global advantage normalisation
        all_advs = torch.cat(
            [ep["advantages"].reshape(-1) for ep in self.buffer.episodes]
        )
        adv_mean = float(all_advs.mean())
        adv_std  = float(all_advs.std() + 1e-8)

        all_policy_losses, all_value_losses = [], []
        all_entropies, all_kls, all_clip_fracs, all_evs = [], [], [], []
        all_sheaf_losses = []
        kl_exceeded = False

        for _ in range(n_epochs):
            if kl_exceeded:
                break
            for ep in self.buffer.iter_episodes():
                T      = ep["length"]
                # Use per-episode N (may differ from self.n_agents after pruning)
                N      = len(ep["active_masks"][0])
                device = self.device

                h = self.policy.init_hidden(N, device)

                new_lp_list  = []
                new_ent_list = []
                new_val_list = []
                sheaf_list   = []

                for t in range(T):
                    g        = ep["graphs"][t].to(device)
                    a_dict_t = {k: ep["actions"][t][k].to(device)
                                for k in ep["actions"][t]}
                    active_t = torch.tensor(
                        ep["active_masks"][t], dtype=torch.float32, device=device
                    )

                    # Ensure component_sizes is on device (graphs stored on cpu)
                    if not hasattr(g, "component_sizes") or g.component_sizes is None:
                        g.component_sizes = torch.zeros(N, dtype=torch.long, device=device)
                    else:
                        g.component_sizes = g.component_sizes.to(device)

                    lp, ent, val, h, sheaf_loss = self.policy.evaluate_actions(
                        g, h, a_dict_t, active_mask=active_t
                    )
                    new_lp_list.append(lp)
                    new_ent_list.append(ent)
                    new_val_list.append(val)
                    sheaf_list.append(sheaf_loss)

                new_log_probs = torch.stack(new_lp_list)   # [T, N]
                new_entropy   = torch.stack(new_ent_list)  # [T, N]
                new_values    = torch.stack(new_val_list)  # [T, N]
                sheaf_loss_mean = torch.stack(sheaf_list).mean()

                old_log_probs = ep["log_probs_old"].to(device)
                advantages    = (ep["advantages"].to(device) - adv_mean) / adv_std
                returns       = ep["returns"].to(device)

                active_masks_t = torch.stack([
                    torch.tensor(ep["active_masks"][t], dtype=torch.float32, device=device)
                    for t in range(T)
                ])

                old_lp_f = old_log_probs.reshape(-1)
                new_lp_f = new_log_probs.reshape(-1)
                adv_f    = advantages.reshape(-1)
                ret_f    = returns.reshape(-1)
                val_f    = new_values.reshape(-1)
                ent_f    = new_entropy.reshape(-1)
                active_f = active_masks_t.reshape(-1)
                n_active = active_f.sum().clamp(min=1.0)

                ratio  = torch.exp(new_lp_f - old_lp_f)
                surr1  = ratio * adv_f
                surr2  = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_f
                policy_loss  = -(torch.min(surr1, surr2) * active_f).sum() / n_active
                value_loss   =  0.5 * (((val_f - ret_f) ** 2) * active_f).sum() / n_active
                entropy_loss = -(ent_f * active_f).sum() / n_active

                total_loss = (policy_loss
                              + c_v * value_loss
                              + c_e * entropy_loss
                              + c_s * sheaf_loss_mean)

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((old_lp_f - new_lp_f) * active_f).sum().item() / n_active.item()
                    clip_frac = (((ratio - 1.0).abs() > clip_eps).float() * active_f).sum().item() / n_active.item()
                    ret_active = ret_f[active_f > 0]
                    val_active = val_f[active_f > 0]
                    var_ret    = ret_active.var().item()
                    ev = (1.0 - (ret_active - val_active).var().item() / var_ret
                          if var_ret > 1e-8 else 0.0)

                all_policy_losses.append(float(policy_loss.item()))
                all_value_losses.append(float(value_loss.item()))
                all_entropies.append(float(-entropy_loss.item()))
                all_kls.append(float(approx_kl))
                all_clip_fracs.append(float(clip_frac))
                all_evs.append(float(ev))
                all_sheaf_losses.append(float(sheaf_loss_mean.item()))

                if approx_kl > target_kl:
                    kl_exceeded = True
                    break

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.update_count += 1

        if self.writer is not None:
            try:
                self.writer.add_scalar("loss/sheaf", float(np.mean(all_sheaf_losses)), self.update_count)
            except Exception:
                pass

        def _mean(lst): return float(np.mean(lst)) if lst else 0.0
        return {
            "policy_loss":        _mean(all_policy_losses),
            "value_loss":         _mean(all_value_losses),
            "entropy":            _mean(all_entropies),
            "sheaf_loss":         _mean(all_sheaf_losses),
            "approx_kl":          _mean(all_kls),
            "clip_fraction":      _mean(all_clip_fracs),
            "explained_variance": _mean(all_evs),
        }

    # ------------------------------------------------------------------
    # Logging / saving / training loop
    # ------------------------------------------------------------------

    def _log_episode(self, stats: Dict, episode: int):
        row = [
            episode,
            stats.get("mean_total_reward", 0),
            stats.get("goals_reached", 0),
            stats.get("goal_completion_rate", 0),
            stats.get("collision_count", 0),
            stats.get("collision_rate", 0),
            stats.get("timeout_rate", 0),
            stats.get("mean_episode_length", 0),
            stats.get("n_agents", self.n_agents),
            stats.get("curriculum_stage", self.curriculum_stage_idx),
        ]
        with open(self.csv_episode_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        if self.writer is not None:
            try:
                for k, v in stats.items():
                    self.writer.add_scalar(f"episode/{k}", float(v), episode)
            except Exception:
                pass

    def _log_update(self, stats: Dict, update: int):
        row = [
            update,
            stats.get("policy_loss", 0), stats.get("value_loss", 0),
            stats.get("entropy", 0),     stats.get("sheaf_loss", 0),
            stats.get("approx_kl", 0),   stats.get("clip_fraction", 0),
            stats.get("explained_variance", 0),
        ]
        with open(self.csv_update_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        if self.writer is not None:
            try:
                for k, v in stats.items():
                    self.writer.add_scalar(f"update/{k}", float(v), update)
            except Exception:
                pass

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.episode_count        = int(ckpt.get("episode_count",        0))
        self.update_count         = int(ckpt.get("update_count",         0))
        self.curriculum_stage_idx = int(ckpt.get("curriculum_stage_idx", 0))
        self.n_agents             = int(ckpt.get("n_agents",             self.n_agents))
        self.buffer = GATEpisodeBuffer(
            n_steps    = int(self.config["n_steps"]),
            n_agents   = self.n_agents,
            gamma      = float(self.config["gamma"]),
            gae_lambda = float(self.config["gae_lambda"]),
        )
        self.last_logged_episode = self.episode_count
        self.last_saved_episode  = self.episode_count
        print(f"[Sheaf Trainer] Resumed from '{path}' "
              f"(episode {self.episode_count}, update {self.update_count})")

    def save_checkpoint(self, tag: str = "latest"):
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, f"policy_{tag}.pt")
        torch.save({
            "policy_state_dict":    self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode_count":        self.episode_count,
            "update_count":         self.update_count,
            "curriculum_stage_idx": self.curriculum_stage_idx,
            "n_agents":             self.n_agents,
            "config":               self.config,
        }, path)
        return path

    def train(self, total_episodes: int):
        save_every = int(self.config.get("save_every", 50))
        sheaf_coef = float(self.config.get("sheaf_coef", 0.01))

        base_r  = self.config.get("base_repulsion_radius",   1.5)
        base_s  = self.config.get("base_repulsion_strength", 5.0)
        print("=" * 56)
        print("  SHEAF DECONFLICTION TRAINING V2")
        print("=" * 56)
        print(f"  Action space:       Priority score per agent (R^1, unbounded)")
        print(f"  Velocity protocol:  NONE — velocities unmodified")
        print(f"  Repulsion protocol: Per-agent radius and strength scaled by rank")
        print(f"                      increment[i] = (n - rank[i]) / n  [0,1]")
        print(f"                      radius[i]   = {base_r} + increment[i]")
        print(f"                      strength[i] = {base_s} + increment[i]")
        print(f"                      rank 0 (highest) → largest radius + strength")
        print(f"                      rank n-1 (lowest) → smallest radius + strength")
        print(f"  Architecture:       2 × SheafLayer (stalk_dim = hidden_dim // 2)")
        print(f"                      + GRUCell + priority_head (zero init)")
        print(f"  Sheaf loss coef:    {sheaf_coef}")
        print(f"  Priority cache:     Active — recomputes only on topology change")
        print(f"  A* controller:      NOT USED")
        print(f"  Checkpoint dir:     {self.save_dir}")
        print(f"  Rewards:            collision=-300, goal=+25, clean_sweep=+300")
        print(f"                      progress_coef={self.config.get('progress_coef', 1.0)}")
        print(f"  n_agents:           {self.n_agents}")
        print(f"  device:             {self.device}")
        print("=" * 56)

        while self.episode_count < total_episodes:
            ep_stats     = self.collect_rollout()
            update_stats = self.update_policy()
            self._advance_curriculum()

            self._log_episode(ep_stats, self.episode_count)
            self._log_update(update_stats, self.update_count)
            print(
                f"[Update {self.update_count:4d} | Ep {self.episode_count:5d}] "
                f"reward={ep_stats['mean_total_reward']:7.2f}  "
                f"goals={ep_stats['goal_completion_rate']:.2f}  "
                f"col={ep_stats['collision_rate']:.2f}  "
                f"sheaf={update_stats['sheaf_loss']:.4f}  "
                f"kl={update_stats['approx_kl']:.4f}  "
                f"ev={update_stats['explained_variance']:.3f}  "
                f"n={int(ep_stats['n_agents_pruned'])}/{self.n_agents}"
            )
            self.last_logged_episode = self.episode_count

            if self.episode_count - self.last_saved_episode >= save_every:
                path = self.save_checkpoint(tag=f"ep{self.episode_count}")
                print(f"  Saved: {path}")
                self.last_saved_episode = self.episode_count

        self.save_checkpoint(tag="final")
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                pass
        print(f"[Sheaf Trainer] Done. {self.episode_count} episodes, {self.update_count} updates.")
