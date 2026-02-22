# src/controllers/apf_lagrangian.py

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ApfLagrangianConfig:
    # --- LiDAR settings ---
    num_lidar_beams: int = 32      # number of beams over 360°
    lidar_max_range: float = 3.0   # max sensing range

    # --- APF nominal controller gains ---
    k_goal: float = 2.5            # attraction toward goal
    k_wall: float = 3.0            # wall repulsion (via LiDAR)
    k_agent: float = 2.0           # agent-agent repulsion

    # weight for preferring open directions from LiDAR
    k_free: float = 5.0          # try 0.5–1.0 to start
    free_space_exp: float = 10.0  # exponent alpha in (r/R)^alpha

    # Agent-agent repulsion cutoff (for APF)
    agent_sense_radius: float = 3.0

    # --- Safety distances for refinement (if None, use multiples of env.robot_radius) ---
    safe_dist_wall: Optional[float] = None
    safe_dist_agent: Optional[float] = None

    # --- Penalty / Lagrangian weights (refinement step) ---
    lambda_wall: float = 20.0
    lambda_agent: float = 20.0

    # --- Gradient-based refinement parameters ---
    step_size: float = 0.05
    num_iters: int = 8


def _compute_lidar_scans(env, positions: np.ndarray, config: ApfLagrangianConfig) -> np.ndarray:
    """
    Compute LiDAR scans for all agents against static walls.

    Returns:
        ranges: (N, K) array where ranges[i, k] is the distance along beam k
                from agent i to the closest wall intersection, capped at lidar_max_range.
    """
    walls = env.walls
    N = positions.shape[0]
    K = config.num_lidar_beams
    R_max = config.lidar_max_range

    # Precompute beam directions (unit vectors) for 360° coverage
    angles = np.linspace(0.0, 2.0 * np.pi, num=K, endpoint=False)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (K, 2)

    ranges = np.full((N, K), R_max, dtype=float)

    # For each agent, cast K rays and intersect with each wall segment
    for i in range(N):
        origin = positions[i]
        for k in range(K):
            d = dirs[k]
            min_t = R_max

            # Ray: origin + t * d, t >= 0
            for w in walls:
                a = np.array([w.x1, w.y1], dtype=float)
                b = np.array([w.x2, w.y2], dtype=float)
                ab = b - a

                # Solve origin + t d = a + u ab, t >= 0, u in [0,1]
                # This is 2x2 linear system: [ -d  ab ] [t; u] = a - origin
                M = np.column_stack((d, -ab))
                rhs = a - origin

                det = np.linalg.det(M)
                if abs(det) < 1e-9:
                    continue  # parallel or almost parallel; skip

                invM = np.linalg.inv(M)
                t, u = invM @ rhs

                if t >= 0.0 and 0.0 <= u <= 1.0:
                    if t < min_t:
                        min_t = t

            ranges[i, k] = min(min_t, R_max)

    return ranges


def apf_nominal_control(env, config: ApfLagrangianConfig) -> np.ndarray:
    """
    Compute a nominal APF-style control for all agents using:
      - attraction to goal,
      - repulsion from other agents (distance-based),
      - repulsion from walls via LiDAR beams.

    This is a purely local, geometric steering law; the Lagrangian refinement
    step will make it safer with respect to hard distance constraints.
    """
    positions = env.positions
    goals = env.goals

    if positions is None or goals is None:
        raise ValueError("Environment must have positions and goals set before calling APF controller.")

    positions = np.asarray(positions, dtype=float)  # (N, 2)
    goals = np.asarray(goals, dtype=float)          # (N, 2)
    N = positions.shape[0]

    # Attractive term: go to goal
    u = config.k_goal * (goals - positions)

    # --- Agent-agent repulsion (APF-style) ---
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            diff = positions[i] - positions[j]
            dist = np.linalg.norm(diff)
            if 0.0 < dist < config.agent_sense_radius:
                # repulsion ~ 1/dist^2 along the line connecting the agents
                u[i] += config.k_agent * diff / (dist**2 + 1e-6)

    # --- Wall repulsion via LiDAR ---
    K = config.num_lidar_beams
    R_max = config.lidar_max_range
    angles = np.linspace(0.0, 2.0 * np.pi, num=K, endpoint=False)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (K, 2)

    lidar_ranges = _compute_lidar_scans(env, positions, config)  # (N, K)

    alpha = config.free_space_exp
    for i in range(N):
        r_i = lidar_ranges[i]                        # (K,)
        # normalize ranges to [0,1]
        w = (r_i / R_max) ** alpha                   # (K,)
        # weighted average of beam directions
        v_free = (w[:, None] * dirs).sum(axis=0) / (w.sum() + 1e-9)  # (2,)

        # add free-space bias
        u[i] += config.k_free * v_free

    # 2) Wall repulsion: for beams that actually hit something, push opposite the beam
    for i in range(N):
        for k in range(K):
            r = lidar_ranges[i, k]
            if r < R_max:  # something is hit along this ray
                r_safe = max(r, 1e-2)
                dir_vec = dirs[k]  # direction toward obstacle along this beam
                u[i] -= config.k_wall * dir_vec / (r_safe**2)

    # Clip by env.max_speed if available
    max_speed = getattr(env, "max_speed", None)
    if max_speed is not None:
        speeds = np.linalg.norm(u, axis=1, keepdims=True)
        mask = speeds > max_speed
        u = np.where(mask, u * max_speed / (speeds + 1e-9), u)

    return u


def refine_control_lagrangian(env, u_nom: np.ndarray, config: ApfLagrangianConfig) -> np.ndarray:
    """
    One-step refinement of APF control using a penalty/Lagrangian-style update.

    We approximately minimize:

        J(u) = 0.5 * ||u - u_nom||^2
               + lambda_wall  * sum_i,k  [max(0, d_safe_wall  - d_iwall)^2]
               + lambda_agent * sum_(i<j)[max(0, d_safe_agent - d_ij)^2]

    where distances are computed at the predicted next positions:

        p_i^+ = p_i + dt * u_i

    This is NOT a full-blown MPC; it's a single-step local repair that
    nudges the control away from impending collisions while staying
    close to the APF suggestion.
    """
    positions = env.positions
    walls = env.walls

    if positions is None:
        raise ValueError("Environment must have positions set before refinement.")

    positions = np.asarray(positions, dtype=float)
    u_nom = np.asarray(u_nom, dtype=float)
    u = u_nom.copy()

    N = positions.shape[0]
    dt = getattr(env, "dt", 0.1)

    # Choose safety distances if not specified
    robot_radius = getattr(env, "robot_radius", 0.3)
    safe_wall = config.safe_dist_wall if config.safe_dist_wall is not None else 2.0 * robot_radius
    safe_agent = config.safe_dist_agent if config.safe_dist_agent is not None else 2.5 * robot_radius

    for _ in range(config.num_iters):
        # Predicted next positions under current u
        p_next = positions + dt * u

        # Gradient wrt u from quadratic term
        grad_u = (u - u_nom)
        # Gradient wrt p_next from penalties
        grad_p = np.zeros_like(p_next)

        # --- Wall penalties ---
        for i in range(N):
            p = p_next[i]
            for w in walls:
                a = np.array([w.x1, w.y1], dtype=float)
                b = np.array([w.x2, w.y2], dtype=float)
                d = env._distance_point_to_segment(p, a, b)
                if d < safe_wall:
                    # Closest point on segment
                    ab = b - a
                    if np.allclose(ab, 0.0):
                        proj = a
                    else:
                        t = np.clip(
                            np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9),
                            0.0, 1.0
                        )
                            # normal points from wall to agent
                        proj = a + t * ab

                    normal = p - proj
                    n_norm = np.linalg.norm(normal)
                    if n_norm > 1e-6:
                        normal /= n_norm
                    else:
                        normal[:] = 0.0

                    # Penalty term: 0.5 * lambda_wall * (safe_wall - d)^2
                    # dJ/dd = lambda_wall * (d - safe_wall)
                    coeff = config.lambda_wall * (d - safe_wall)
                    grad_p[i] += coeff * normal  # gradient wrt p_next

        # --- Agent-agent penalties ---
        for i in range(N):
            for j in range(i + 1, N):
                diff = p_next[i] - p_next[j]
                dist = np.linalg.norm(diff)
                if 0.0 < dist < safe_agent:
                    # Penalty: 0.5 * lambda_agent * (safe_agent - dist)^2
                    # dJ/dd = lambda_agent * (dist - safe_agent)
                    n_ij = diff / (dist + 1e-9)
                    coeff = config.lambda_agent * (dist - safe_agent)

                    grad_p[i] += coeff * n_ij
                    grad_p[j] -= coeff * n_ij  # opposite sign

        # Chain rule: p_next = positions + dt * u  =>  dJ/du += dt * dJ/dp_next
        grad_u += dt * grad_p

        # Gradient descent step
        u = u - config.step_size * grad_u

        # Re-enforce max_speed after each refinement step
        max_speed = getattr(env, "max_speed", None)
        if max_speed is not None:
            speeds = np.linalg.norm(u, axis=1, keepdims=True)
            mask = speeds > max_speed
            u = np.where(mask, u * max_speed / (speeds + 1e-9), u)

    return u


class ApfLagrangianController:
    """
    High-level controller object.

    Usage:
        controller = ApfLagrangianController()
        actions = controller(env)  # returns refined velocities (N, 2)
    """

    def __init__(self, config: Optional[ApfLagrangianConfig] = None):
        if config is None:
            config = ApfLagrangianConfig()
        self.config = config

    def compute_actions(self, env) -> np.ndarray:
        u_nom = apf_nominal_control(env, self.config)
        u_ref = refine_control_lagrangian(env, u_nom, self.config)
        return u_ref

    def __call__(self, env) -> np.ndarray:
        return self.compute_actions(env)
