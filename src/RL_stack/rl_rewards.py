"""rl_rewards.py

Reward and termination logic for multi-agent RL fine-tuning.

This module is designed for your current stack:
- Environment: `sim_env.MultiRobotEnv` (or API-compatible)
- Online graph builder: `rl_graph_obs.GraphObsBuilder` (which returns `aux`)

Requirements implemented (as requested):
- Per-agent rewards and per-agent termination; DO NOT terminate the whole episode on collisions.
- Individual rewards for reaching goals.
- Individual penalties for wall collisions and agent-agent collisions.
- Small time decay while active.
- Dense shaping via Euclidean distance-to-goal decrease.
- Small wall-proximity penalty using min LiDAR distance.
- Uniform team reward when the episode is successfully completed (all agents reach goals).
- Agents that are done (goal/collision) are masked from further shaping/time penalties.

Mathematical note on the progress term:
We use a potential-based shaping style term:  k * (d_prev - d_curr).
For a discount factor gamma, the classic potential-based shaping is
    r' = r + gamma*Phi(s') - Phi(s)
With Phi(s) = -d(s), this yields a dense signal proportional to distance decrease.
Even when you do not implement the exact gamma form, the distance-decrease term is still
a practical dense shaping signal that empirically improves learning.

The reward computer keeps a per-agent `active` mask; you should reset it at env.reset().

Typical use in an RL loop:

    rewarder = RewardComputer(n_agents=env.n_agents)
    rewarder.reset(env)
    prev_aux = None
    for t in range(T):
        data_t, aux = obs_builder.build(env)
        action, logp, value, h_next = ac.act(data_t, h, aux['active'])
        env.step(action * env.max_speed)
        r, done_agent, info = rewarder.step(env, aux, prev_aux, action)
        prev_aux = aux
        h = h_next

The rewarder does NOT call env.step; it assumes env.step already happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

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


@dataclass
class RewardConfig:
    """Weights and thresholds for reward computation."""

    # Terminal events
    goal_reward: float = 10.0
    wall_collision_penalty: float = 20.0
    agent_collision_penalty: float = 15.0

    # Episode-level team reward (granted once when all agents reach goals)
    team_success_reward: float = 20.0

    # Dense shaping while active
    time_penalty: float = 0.01
    progress_weight: float = 1.0

    # Wall proximity shaping
    wall_proximity_weight: float = 0.5
    wall_safe_dist: float = 0.5  # in same units as LiDAR distances


class RewardComputer:
    """Compute per-agent rewards and per-agent done flags.

    The environment's built-in `done` and `reward` are ignored; this class computes
    reward and termination from env state.

    Important: sim_env.MultiRobotEnv's helpers for collision checks are "private" (leading underscore),
    but Python still allows calling them. We use them to implement per-agent collisions.
    """

    def __init__(self, n_agents: int, cfg: Optional[RewardConfig] = None):
        self.n_agents = int(n_agents)
        self.cfg = cfg or RewardConfig()
        self.active = np.ones((self.n_agents,), dtype=bool)
        self._team_reward_given = False
        self._any_collision = False

    def reset(self, env=None, n_agents: Optional[int] = None):
        """Reset internal masks at episode start."""
        if n_agents is not None:
            self.n_agents = int(n_agents)
        self.active = np.ones((self.n_agents,), dtype=bool)
        self._team_reward_given = False
        self._any_collision = False

    # ------------------------- collision + goal checks -------------------------

    def _goal_reached_mask(self, env) -> np.ndarray:
        pos = np.asarray(env.positions, dtype=float)
        goals = np.asarray(env.goals, dtype=float)
        d = np.linalg.norm(pos - goals, axis=1)
        return d < float(env.goal_tolerance)

    def _wall_collision_mask(self, env) -> np.ndarray:
        # An agent collides with wall if within robot_radius of any wall segment.
        pos = np.asarray(env.positions, dtype=float)
        out = np.zeros((self.n_agents,), dtype=bool)
        for i in range(self.n_agents):
            # env._point_near_any_wall expects margin
            out[i] = bool(env._point_near_any_wall(pos[i], margin=float(env.robot_radius)))
        return out

    def _agent_collision_mask(self, env) -> np.ndarray:
        pos = np.asarray(env.positions, dtype=float)
        out = np.zeros((self.n_agents,), dtype=bool)
        rr = float(env.robot_radius)
        thr = 2.0 * rr
        for i in range(self.n_agents):
            if not self.active[i]:
                continue
            for j in range(i + 1, self.n_agents):
                if not self.active[j]:
                    continue
                if float(np.linalg.norm(pos[i] - pos[j])) < thr:
                    out[i] = True
                    out[j] = True
        return out

    # ------------------------------- main step --------------------------------

    def step(
        self,
        env,
        aux: Dict[str, np.ndarray],
        prev_aux: Optional[Dict[str, np.ndarray]],
        actions: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        """Compute rewards after env has stepped.

        Args:
            env: MultiRobotEnv (or compatible). Must expose positions, goals, robot_radius, goal_tolerance.
            aux: dict returned by rl_graph_obs.GraphObsBuilder.build(env). Must include:
                 - 'dist_to_goal': (N,) float
                 - 'min_lidar': (N,) float
                 - 'active': (N,) bool or float (optional; if present, will be ANDed with internal mask)
            prev_aux: previous step's aux dict or None at t=0.
            actions: optional (N,2) actions in normalized units [-1,1] or in env units; not used here.

        Returns:
            rewards: (N,) float per agent.
            done_agent: (N,) bool per agent termination at this step.
            info: diagnostics dict.
        """
        N = self.n_agents
        cfg = self.cfg

        # Ensure arrays
        dist = np.asarray(aux.get("dist_to_goal"), dtype=float).reshape(N)
        min_lidar = np.asarray(aux.get("min_lidar"), dtype=float).reshape(N)

        # Combine builder active (if provided) with internal active mask
        if "active" in aux:
            builder_active = np.asarray(aux["active"]).reshape(N)
            builder_active = builder_active.astype(bool) if builder_active.dtype != bool else builder_active
            self.active = np.logical_and(self.active, builder_active)

        active = self.active.copy()

        # Terminal event masks
        reached = self._goal_reached_mask(env) & active
        wall_col = self._wall_collision_mask(env) & active
        agent_col = self._agent_collision_mask(env) & active

        done_agent = reached | wall_col | agent_col

        if np.any(wall_col) or np.any(agent_col):
            self._any_collision = True

        # Base rewards
        rewards = np.zeros((N,), dtype=float)

        # Time penalty while active (before applying done)
        rewards[active] -= cfg.time_penalty

        # Progress shaping (distance decrease) while active
        if prev_aux is not None and "dist_to_goal" in prev_aux:
            prev_dist = np.asarray(prev_aux["dist_to_goal"], dtype=float).reshape(N)
            # Positive if you moved closer
            delta = prev_dist - dist
            rewards[active] += cfg.progress_weight * delta[active]

        # Wall proximity penalty (soft), while active
        if cfg.wall_proximity_weight != 0.0 and cfg.wall_safe_dist > 0.0:
            gap = np.maximum(0.0, cfg.wall_safe_dist - min_lidar)
            rewards[active] -= cfg.wall_proximity_weight * (gap[active] ** 2)

        # Terminal bonuses/penalties (only once, at transition)
        rewards[reached] += cfg.goal_reward
        rewards[wall_col] -= cfg.wall_collision_penalty
        rewards[agent_col] -= cfg.agent_collision_penalty

        # Update active mask: remove done agents
        self.active[done_agent] = False

        # Team success reward: give once when all agents reached goals (not just "done")
        # We define success as every agent reached its goal at some point (i.e., no active agents remain)
        # AND no agent is removed due to collision.
        # To detect this robustly, we check whether all agents are within goal tolerance *now* OR have
        # previously reached. Since we don't store "reached_history" separately, we grant team reward
        # when the *last remaining active* agent reaches goal and no collisions occurred this step.
        team_success = False
        if (not self._team_reward_given) and (not self._any_collision) and (not np.any(self.active)):
            # If episode ended because last event was reaching (and not a collision), treat as success.
            # If the last event was collision (wall/agent), do NOT grant team reward.
            if np.any(reached) and (not np.any(wall_col)) and (not np.any(agent_col)):
                team_success = True
                rewards += cfg.team_success_reward
                self._team_reward_given = True

        info: Dict[str, object] = {
            "active": active,
            "done_agent": done_agent,
            "reached": reached,
            "wall_collision": wall_col,
            "agent_collision": agent_col,
            "team_success": team_success,
            "any_collision": bool(self._any_collision),
            "all_inactive": (not np.any(self.active)),
            "mean_dist": float(np.mean(dist)),
            "mean_min_lidar": float(np.mean(min_lidar)),
        }

        return rewards, done_agent, info
