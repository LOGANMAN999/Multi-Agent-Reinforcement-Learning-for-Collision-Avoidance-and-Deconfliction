"""
# Phase 1 Vectorization Header
# File: src/sim_env.py
# Changes:
#   - Added _wall_endpoints() to extract [W,2] arrays once per episode reset.
#   - Added _pts_to_segments_dist_batched(): vectorized [N,W] point-to-segment
#     distances using NumPy broadcasting — zero Python loops over agents or walls.
#   - Added _pairwise_dist_matrix(): cdist-based [N,N] pairwise distance matrix.
#   - Replaced _check_wall_collisions inner loop with vectorized call.
#   - Replaced _check_robot_collisions nested loop with vectorized call.
#   - Added check_per_agent_collisions_vec(positions) → ([N] bool, [N] bool)
#     for use by the trainer (returns per-agent flags, not single bool).
#   - Replaced lidar_scan_all() inner loop with vectorized ray-segment
#     intersection over [N,K,W] simultaneously.
#   - Added _batched_ray_wall_distances_per_agent() static helper used by
#     build_graph for directional wall distances.
#   - Added __main__ throughput benchmark (1.6 above original).
# Why: eliminate all O(N) and O(N²) Python loops from the simulation hot path.
"""
from __future__ import annotations
import enum
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
#from controllers.student_gnn_controller import StudentGNNController, StudentPolicyPaths
from controllers.student_gnn_controller_gru import StudentGNNGRUController, StudentPolicyPaths

from controllers.astar_global_local import AStarGlobalLocalController
from controllers.harmonic_navigation import HarmonicNavigationController  # harmonic potential field nav
from data_building.map_generation import generate_mapset 
from safety_filter import safety_filter
from controllers.bug_controller import BugController



# --- Basic geometry types ----------------------------------------------------


@dataclass
class Wall:
    """Axis-aligned wall segment in 2D."""
    x1: float
    y1: float
    x2: float
    y2: float


class MapType(enum.Enum):
    RANDOM = "random"
    MAZE = "maze"
    MAZE_WITH_RANDOM = "maze_with_random"


# --- Map generators ----------------------------------------------------------


WORLD_SIZE = 10.0  # square world: [-WORLD_SIZE, WORLD_SIZE]^2


def generate_random_obstacles(num_obstacles: int = 8) -> List[Wall]:
    """
    Generate 'dot-like' obstacles by making tiny vertical or horizontal walls.
    This roughly matches Fig. 5(a) in the paper.
    """
    walls: List[Wall] = []
    for _ in range(num_obstacles):
        x = np.random.uniform(-WORLD_SIZE * 0.8, WORLD_SIZE * 0.8)
        y = np.random.uniform(-WORLD_SIZE * 0.8, WORLD_SIZE * 0.8)

        # small line segment to look like a point-ish obstacle
        if np.random.rand() < 0.5:
            # short vertical segment
            walls.append(Wall(x, y - 0.2, x, y + 0.2))
        else:
            # short horizontal segment
            walls.append(Wall(x - 0.2, y, x + 0.2, y))

    return walls


def generate_maze_walls() -> List[Wall]:
    """
    Simple hand-crafted 'maze' as a few long axis-aligned segments.
    Meant to resemble Fig. 5(b) qualitatively.
    """
    w = WORLD_SIZE
    walls = [
        # outer boundary
        Wall(-w, -w, -w, w),
        Wall(w, -w, w, w),
        Wall(-w, -w, w, -w),
        Wall(-w, w, w, w),
        # inner maze-like walls (tweak as you like)
        Wall(0, 6, 0,-6),
        #Wall(-3, -w, -3, -1),
        #Wall(3, 1, 3, w),
    ]
    return walls


def generate_map(map_type: MapType) -> List[Wall]:
    """
    Master function to generate obstacle walls for a given map type.
    """
    if map_type == MapType.RANDOM:
        return generate_random_obstacles()

    if map_type == MapType.MAZE:
        return generate_maze_walls()

    if map_type == MapType.MAZE_WITH_RANDOM:
        maze = generate_maze_walls()
        random_walls = generate_random_obstacles(num_obstacles=8)
        return maze + random_walls

    raise ValueError(f"Unknown map type: {map_type}")


# --- Visualization helper ----------------------------------------------------


def plot_map(walls: List[Wall], ax=None, title: str = ""):
    if ax is None:
        fig, ax = plt.subplots()

    # draw walls
    for w in walls:
        ax.plot([w.x1, w.x2], [w.y1, w.y2], linewidth=2)

    ax.set_aspect("equal")
    ax.set_xlim(-WORLD_SIZE, WORLD_SIZE)
    ax.set_ylim(-WORLD_SIZE, WORLD_SIZE)
    ax.set_title(title)
    ax.grid(True)


def debug_draw_lidar(env, agent_i=0, n_rays=32, max_range=20.0):
        pos = np.asarray(env.positions[agent_i], dtype=float)
        d = env.lidar_scan(pos, n_rays=n_rays, max_range=max_range)

        ax = plt.gca()
        # draw only every 'step' rays to reduce clutter
        for k in range(n_rays):
            theta = 2*np.pi*(k/n_rays)
            r = np.array([np.cos(theta), np.sin(theta)])
            end = pos + d[k]*r
            ax.plot([pos[0], end[0]], [pos[1], end[1]], linewidth=1)


# --- Multi-robot environment --------------------------------------------------

class MultiRobotEnv:
    """
    Simple 2D multi-robot environment on top of our wall maps.

    - Robots are discs with radius self.robot_radius.
    - Actions are desired velocities in R^2 per robot.
    - Dynamics: p_{t+1} = p_t + dt * clip(v, max_speed).
    """

    def __init__(
        self,
        world_size: float = WORLD_SIZE,
        dt: float = 0.1,
        robot_radius: float = 0.25,
        max_speed: float = 1.5,
        goal_tolerance: float = 0.3,
        action_mode: str = "velocity",
    ):
        """
        action_mode : "velocity" (default) or "waypoint".
          - "velocity": step() expects (N,2) velocity commands (original behaviour).
          - "waypoint": step() internally converts waypoint offsets → velocities via PD.
            The conversion is:  v = k_p * (waypoint - pos),  k_p = max_speed / r_max.
            r_max defaults to 2.0 (WaypointConfig default); override via step(..., r_max=).
            The velocity output is then clipped to max_speed and passed through the
            normal dynamics — no external safety filter call is made here.
            Callers are responsible for running safety_filter_with_count() before
            calling step() if they want safe velocities with intervention counts.
        """
        assert action_mode in ("velocity", "waypoint"), \
            f"action_mode must be 'velocity' or 'waypoint', got {action_mode!r}"
        self.world_size = world_size
        self.dt = dt
        self.robot_radius = robot_radius
        self.max_speed = max_speed
        self.goal_tolerance = goal_tolerance
        self.action_mode = action_mode

        self.walls: List[Wall] = []
        self.map_type: MapType | None = None

        self.n_agents: int = 0
        self.positions: np.ndarray | None = None  # shape (N, 2)
        self.goals: np.ndarray | None = None      # shape (N, 2)
        self.t: int = 0

    # ------------- public API -------------------------------------------------

    def reset(self, map_type: MapType, n_agents: int):
        """
        Sample a new map and random start/goal positions.

        Returns an observation array, here [x, y, gx, gy] per agent.
        """
        self.map_type = map_type
        self.walls = generate_map(map_type)
        self.n_agents = n_agents
        self.t = 0

        self.positions = self._sample_non_colliding_points(n_agents)
        self.goals = self._sample_non_colliding_points(n_agents)

        # --- Build harmonic navigation fields, one per goal (agent i -> goal i) ---
        self.nav_fields = []
        for i in range(self.n_agents):
            g = np.asarray(self.goals[i], dtype=float)
            phi = self.navigator.compute_potential_for_goal(g)
            self.nav_fields.append(phi)

        return self._get_obs()



    def _pd_waypoint_to_vel(
        self,
        waypoints: np.ndarray,
        r_max: float = 2.0,
    ) -> np.ndarray:
        """
        Proportional controller: convert (N,2) absolute waypoint positions → (N,2) velocities.

        Formula:  v = k_p * (waypoint - pos),  k_p = max_speed / r_max
        The proportional gain k_p ensures that a waypoint exactly r_max away
        from the agent produces a velocity of max_speed — no overshoot.
        Output is NOT clipped here; step() handles max_speed clipping.

        Args:
            waypoints: (N,2) absolute waypoint positions in world coordinates.
            r_max:     waypoint radius bound used when the RL policy was trained (default 2.0).

        Returns:
            velocities: (N,2) proportional velocity commands.
        """
        assert self.positions is not None
        k_p = self.max_speed / r_max
        return k_p * (waypoints - self.positions)

    def step(self, actions: np.ndarray, r_max: float = 2.0):
        """
        Step the environment forward one time-step.

        actions: np.ndarray of shape (N, 2).
          - action_mode="velocity": raw velocity commands (m/s).
          - action_mode="waypoint": absolute waypoint positions; converted to
            velocities internally via _pd_waypoint_to_vel(actions, r_max).

        r_max: waypoint bound [m] — only used when action_mode="waypoint".

        Returns:
            obs: (N, 4) array
            reward: float (negative mean distance to goal)
            done: bool (episode finished)
            info: dict (extra diagnostics)
        """
        assert self.positions is not None and self.goals is not None
        assert actions.shape == (self.n_agents, 2)

        # Convert waypoints → velocities when in waypoint mode.
        if self.action_mode == "waypoint":
            actions = self._pd_waypoint_to_vel(actions, r_max=r_max)

        # Clip speeds
        speeds = np.linalg.norm(actions, axis=1, keepdims=True)
        too_fast = speeds > self.max_speed
        actions = np.where(too_fast, actions * self.max_speed / speeds, actions)

        # Integrate positions
        new_positions = self.positions + self.dt * actions

        # Keep inside world bounds
        w = self.world_size
        new_positions[:, 0] = np.clip(new_positions[:, 0], -w, w)
        new_positions[:, 1] = np.clip(new_positions[:, 1], -w, w)

        # Check collisions with walls and other robots
        collision_with_wall = self._check_wall_collisions(new_positions)
        collision_between_robots = self._check_robot_collisions(new_positions)

        self.positions = new_positions
        self.t += 1

        # Compute distances to goals and reward
        dists = np.linalg.norm(self.positions - self.goals, axis=1)
        reward = -float(np.mean(dists))

        # Done if any collision or all agents near their goals
        reached_all = bool(np.all(dists < self.goal_tolerance))
        done = bool(collision_with_wall or collision_between_robots or reached_all)

        info = {
            "collision_with_wall": collision_with_wall,
            "collision_between_robots": collision_between_robots,
            "reached_all": reached_all,
        }

        return self._get_obs(), reward, done, info
    


    def render(self):
        """
        Simple matplotlib render of walls, robot positions, and goals.
        """
        assert self.positions is not None and self.goals is not None

        plt.clf()
        ax = plt.gca()

        # draw walls
        for w in self.walls:
            ax.plot([w.x1, w.x2], [w.y1, w.y2], "k-", linewidth=2)

        # draw robots
        xs, ys = self.positions[:, 0], self.positions[:, 1]
        ax.scatter(xs, ys, c="tab:blue", label="robots")

        # draw goals
        gx, gy = self.goals[:, 0], self.goals[:, 1]
        ax.scatter(gx, gy, c="tab:orange", marker="x", label="goals")

        ax.set_aspect("equal")
        ax.set_xlim(-self.world_size, self.world_size)
        ax.set_ylim(-self.world_size, self.world_size)
        ax.set_title(f"t = {self.t}")
        ax.grid(True)

        # optional: keep legend small
        if self.t == 0:
            ax.legend(loc="upper right")

        plt.pause(0.01)


    # ------------- vectorized collision helpers (zero agent loops) -----------

    def _wall_endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ([W,2], [W,2]) arrays of wall start/end points, or empty arrays."""
        if not self.walls:
            empty = np.zeros((0, 2), dtype=float)
            return empty, empty
        a = np.array([[w.x1, w.y1] for w in self.walls], dtype=float)
        b = np.array([[w.x2, w.y2] for w in self.walls], dtype=float)
        return a, b

    @staticmethod
    def _pts_to_segments_dist_batched(
        points: np.ndarray,    # [N, 2]
        a_pts: np.ndarray,     # [W, 2] segment start points
        b_pts: np.ndarray,     # [W, 2] segment end points
    ) -> np.ndarray:
        """
        Vectorized point-to-segment distance for N points and W segments.

        Derivation:
          For segment ab, the closest point to p is proj = a + t*(b-a),
          where t = dot(p-a, b-a) / dot(b-a, b-a), clamped to [0,1].
          Distance = ||p - proj||.

        Broadcasting plan:
          ab[w]       = b[w] - a[w]          shape [W, 2]
          ap[n, w]    = points[n] - a[w]      shape [N, W, 2]
          t[n, w]     = dot(ap, ab) / dot(ab,ab)  shape [N, W]
          proj[n, w]  = a[w] + t[n,w]*ab[w]  shape [N, W, 2]
          dist[n, w]  = ||points[n] - proj[n,w]|| shape [N, W]

        Returns:
            dist: [N, W] float array of distances.
        """
        if a_pts.shape[0] == 0:
            return np.zeros((points.shape[0], 0), dtype=float)

        ab = b_pts - a_pts                                    # [W, 2]
        ab_sq = np.sum(ab * ab, axis=1)                       # [W]

        # ap[n, w] = points[n] - a_pts[w]
        ap = points[:, None, :] - a_pts[None, :, :]           # [N, W, 2]

        # t[n, w] = dot(ap[n,w], ab[w]) / ab_sq[w]
        dot_ap_ab = np.sum(ap * ab[None, :, :], axis=2)       # [N, W]
        t = dot_ap_ab / (ab_sq[None, :] + 1e-12)              # [N, W]
        t = np.clip(t, 0.0, 1.0)                              # [N, W]

        # proj[n, w, :] = a_pts[w] + t[n,w] * ab[w]
        proj = a_pts[None, :, :] + t[:, :, None] * ab[None, :, :]  # [N, W, 2]

        diff = points[:, None, :] - proj                      # [N, W, 2]
        dist = np.sqrt(np.sum(diff * diff, axis=2))            # [N, W]
        return dist

    @staticmethod
    def _pairwise_dist_matrix(positions: np.ndarray) -> np.ndarray:
        """Return [N, N] Euclidean pairwise distance matrix (diagonal = 0)."""
        diff = positions[:, None, :] - positions[None, :, :]  # [N, N, 2]
        return np.sqrt(np.sum(diff * diff, axis=2))            # [N, N]

    def check_per_agent_collisions_vec(
        self, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized per-agent collision detection.  Zero Python loops.

        All agents (active and inactive/frozen) are treated as physical obstacles.
        The caller is responsible for filtering results by active_mask to prevent
        re-counting already-inactive agents.

        Returns:
            wall_cols:  [N] bool — True if agent i collides with any wall.
            robot_cols: [N] bool — True if agent i is within collision distance of any other agent.
        """
        N = positions.shape[0]
        a_pts, b_pts = self._wall_endpoints()

        # --- wall collisions ---
        if a_pts.shape[0] > 0:
            dist_w = self._pts_to_segments_dist_batched(positions, a_pts, b_pts)  # [N, W]
            wall_cols = np.any(dist_w < self.robot_radius, axis=1)               # [N]
        else:
            wall_cols = np.zeros(N, dtype=bool)

        # --- robot-robot collisions ---
        dist_rr = self._pairwise_dist_matrix(positions)        # [N, N]
        np.fill_diagonal(dist_rr, np.inf)
        robot_cols = np.any(dist_rr < 2.0 * self.robot_radius, axis=1)           # [N]

        return wall_cols, robot_cols

    @staticmethod
    def _batched_ray_wall_distances_per_agent(
        positions: np.ndarray,           # [N, 2]
        directions_per_agent: np.ndarray,  # [N, D, 2] unit direction vectors
        a_pts: np.ndarray,               # [W, 2] wall start points
        b_pts: np.ndarray,               # [W, 2] wall end points
        max_range: float,
    ) -> np.ndarray:
        """
        Vectorized ray–wall intersection distances for N agents × D directions × W walls.

        Uses the 2-D ray–segment intersection formula via cross products:
          Given ray: p + t·r_hat (t >= 0)
          Segment:   a + u·(b - a)  (u in [0,1])
          rxs  = cross2(r_hat, s)  where s = b - a
          t    = cross2(q_p, s) / rxs  where q_p = a - p
          u    = cross2(q_p, r_hat) / rxs
          Valid: |rxs| > eps  AND  t >= 0  AND  0 <= u <= 1

        Broadcasting plan (dims: N=agents, D=directions, W=walls):
          s[w]          = b[w] - a[w]             [W, 2]
          q_p[n, w]     = a[w] - pos[n]            [N, W, 2]
          rxs[n, d, w]  = cross2(dir[n,d], s[w])  [N, D, W]
          cross_qp_s[n, w] = cross2(q_p[n,w], s[w])  [N, W]  (no d-dependence)
          cross_qp_r[n, d, w] = cross2(q_p[n,w], dir[n,d])  [N, D, W]
          t[n, d, w]    = cross_qp_s[n,w] / rxs[n,d,w]     [N, D, W]
          u[n, d, w]    = cross_qp_r[n,d,w] / rxs[n,d,w]   [N, D, W]

        Returns: [N, D] distances clipped to [0, max_range].
        """
        N, D = positions.shape[0], directions_per_agent.shape[1]
        W = a_pts.shape[0]

        if W == 0:
            return np.full((N, D), max_range, dtype=float)

        s = b_pts - a_pts                               # [W, 2]

        # q_p[n, w] = a_pts[w] - positions[n]
        q_p = a_pts[None, :, :] - positions[:, None, :]  # [N, W, 2]

        # cross_qp_s[n, w] = q_p[n,w,0]*s[w,1] - q_p[n,w,1]*s[w,0]
        cross_qp_s = q_p[:, :, 0] * s[None, :, 1] - q_p[:, :, 1] * s[None, :, 0]  # [N, W]

        # rxs[n, d, w] = dir[n,d,0]*s[w,1] - dir[n,d,1]*s[w,0]
        rxs = (directions_per_agent[:, :, None, 0] * s[None, None, :, 1] -
               directions_per_agent[:, :, None, 1] * s[None, None, :, 0])  # [N, D, W]

        eps = 1e-9
        parallel = np.abs(rxs) < eps                         # [N, D, W]
        rxs_safe = np.where(parallel, eps, rxs)

        # t[n, d, w] = cross_qp_s[n, w] / rxs[n, d, w]
        t = cross_qp_s[:, None, :] / rxs_safe               # [N, D, W]

        # cross_qp_r[n, d, w] = q_p[n,w,0]*dir[n,d,1] - q_p[n,w,1]*dir[n,d,0]
        cross_qp_r = (q_p[:, None, :, 0] * directions_per_agent[:, :, None, 1] -
                      q_p[:, None, :, 1] * directions_per_agent[:, :, None, 0])  # [N, D, W]
        u = cross_qp_r / rxs_safe                           # [N, D, W]

        valid = (~parallel) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)  # [N, D, W]
        t_filtered = np.where(valid, t, max_range)           # [N, D, W]

        dists = t_filtered.min(axis=2)                       # [N, D]
        return np.clip(dists, 0.0, max_range)

    # ------------- helpers ----------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """
        Simple observation: [x, y, gx, gy] per agent.
        Later we can replace/augment this with LiDAR, neighbor info, etc.
        """
        assert self.positions is not None and self.goals is not None
        return np.concatenate([self.positions, self.goals], axis=1)

    def _sample_non_colliding_points(self, n: int) -> np.ndarray:
        """
        Sample n points that are not too close to walls or each other.
        Quick-and-dirty; good enough to get started.
        """
        pts: list[Tuple[float, float]] = []
        w = self.world_size
        max_tries = 10_000
        tries = 0
        while len(pts) < n and tries < max_tries:
            tries += 1
            x = np.random.uniform(-0.8 * w, 0.8 * w)
            y = np.random.uniform(-0.8 * w, 0.8 * w)
            p = np.array([x, y])

            # away from walls
            if self._point_near_any_wall(p, margin=self.robot_radius * 2.0):
                continue

            # away from existing points
            if pts:
                arr = np.stack(pts, axis=0)
                d = np.linalg.norm(arr - p, axis=1)
                if np.any(d < self.robot_radius * 3.0):
                    continue

            pts.append((x, y))

        if len(pts) < n:
            raise RuntimeError("Could not sample enough non-colliding points")

        return np.array(pts, dtype=float)

    def _point_near_any_wall(self, p: np.ndarray, margin: float) -> bool:
        for w in self.walls:
            if self._distance_point_to_segment(p, np.array([w.x1, w.y1]), np.array([w.x2, w.y2])) < margin:
                return True
        return False

    def _check_wall_collisions(self, positions: np.ndarray) -> bool:
        """Return True if any agent collides with any wall.  Zero Python loops."""
        a_pts, b_pts = self._wall_endpoints()
        if a_pts.shape[0] == 0:
            return False
        dist_w = self._pts_to_segments_dist_batched(positions, a_pts, b_pts)  # [N, W]
        return bool(np.any(dist_w < self.robot_radius))

    def _check_robot_collisions(self, positions: np.ndarray) -> bool:
        """Return True if any pair of robots collides.  Zero Python loops."""
        dist_rr = self._pairwise_dist_matrix(positions)  # [N, N]
        np.fill_diagonal(dist_rr, np.inf)
        return bool(np.any(dist_rr < 2.0 * self.robot_radius))
    

    @staticmethod
    def _cross2(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    def _ray_segment_intersection_dist(
        self,
        p: np.ndarray,          # ray origin
        r_hat: np.ndarray,      # ray direction (unit)
        a: np.ndarray,          # segment start
        b: np.ndarray,          # segment end
        eps: float = 1e-9,
    ) -> float | None:
        """
        Return distance t >= 0 where p + t*r_hat intersects segment ab, else None.
        Uses 2D ray-segment intersection via cross products.
        """
        s = b - a
        rxs = self._cross2(r_hat, s)
        if abs(rxs) < eps:
            return None  # parallel

        q_p = a - p
        t = self._cross2(q_p, s) / rxs
        u = self._cross2(q_p, r_hat) / rxs

        if t >= 0.0 and 0.0 <= u <= 1.0:
            return float(t)  # since r_hat is unit, t is distance
        return None

    def lidar_scan(
        self,
        pos: np.ndarray,
        n_rays: int = 64,
        max_range: float = 20.0,
        angle_offset: float = 0.0,
    ) -> np.ndarray:
        """
        Simple 2D LiDAR-lite: cast n_rays uniformly over [0, 2π).
        Returns distances shape (n_rays,), each in [0, max_range].
        """
        dists = np.full((n_rays,), max_range, dtype=float)

        # Pre-pack wall endpoints
        for k in range(n_rays):
            theta = angle_offset + 2.0 * np.pi * (k / n_rays)
            r_hat = np.array([np.cos(theta), np.sin(theta)], dtype=float)

            best = max_range
            for w in self.walls:
                a = np.array([w.x1, w.y1], dtype=float)
                b = np.array([w.x2, w.y2], dtype=float)
                t = self._ray_segment_intersection_dist(pos, r_hat, a, b)
                if t is not None and t < best:
                    best = t

            dists[k] = best

        return dists

    def lidar_scan_all(
        self,
        n_rays: int = 64,
        max_range: float = 20.0,
        center_angles: np.ndarray | None = None,
        cone_half_angle: float = np.pi,
    ) -> np.ndarray:
        """
        Returns LiDAR distances for all agents simultaneously: shape (N, n_rays).

        Vectorized over N agents × n_rays × W walls using
        _batched_ray_wall_distances_per_agent.  Zero Python loops at runtime.

        Parameters
        ----------
        center_angles : None (default)
            Full 360° sweep.  All agents share identical evenly-spaced ray directions.
        center_angles : np.ndarray [N]
            Per-agent forward-cone scan.  Each agent fires n_rays evenly spread
            across [center - cone_half_angle, center + cone_half_angle].
            Used by BugController for directional wall detection.
        cone_half_angle : float
            Half-width of the cone in radians.  Ignored when center_angles is None.
            Default π = full 360° (same as no cone).
        """
        assert self.positions is not None
        N = self.n_agents
        a_pts, b_pts = self._wall_endpoints()

        if center_angles is None:
            # Full 360° — all agents share the same ray set (original behaviour).
            angles = 2.0 * np.pi * np.arange(n_rays) / n_rays          # [K]
            r_hats = np.stack([np.cos(angles), np.sin(angles)], axis=1) # [K, 2]
            directions_per_agent = np.broadcast_to(
                r_hats[None, :, :], (N, n_rays, 2)
            ).copy()
        else:
            # Per-agent cone scan: each agent gets its own ray direction set.
            c = np.asarray(center_angles, dtype=float)                  # [N]
            offsets = (np.zeros(1) if n_rays == 1
                       else np.linspace(-cone_half_angle, cone_half_angle, n_rays))
            ray_angles = c[:, None] + offsets[None, :]                  # [N, K]
            directions_per_agent = np.stack(
                [np.cos(ray_angles), np.sin(ray_angles)], axis=2        # [N, K, 2]
            )

        return self._batched_ray_wall_distances_per_agent(
            self.positions, directions_per_agent, a_pts, b_pts, max_range
        )  # [N, n_rays]

    def goal_visibility(
        self,
        i: int,
        max_range: float = 8.0,
        eps: float = 1e-6,
    ) -> tuple[bool, float]:
        """
        Returns (visible, dist_to_first_wall_along_goal_dir).
        visible=True iff first wall along direction is farther than goal distance.
        """
        assert self.positions is not None and self.goals is not None
        p = np.asarray(self.positions[i], dtype=float)
        g = np.asarray(self.goals[i], dtype=float)
        v = g - p
        dist_goal = float(np.linalg.norm(v))
        if dist_goal < eps:
            return True, max_range

        r_hat = v / dist_goal
        best = max_range
        for w in self.walls:
            a = np.array([w.x1, w.y1], dtype=float)
            b = np.array([w.x2, w.y2], dtype=float)
            t = self._ray_segment_intersection_dist(p, r_hat, a, b)
            if t is not None and t < best:
                best = t

        visible = (best >= dist_goal - 1e-3)  # small tolerance
        return visible, best

    def segment_blocked_by_walls(self, p: np.ndarray, q: np.ndarray) -> bool:
        """
        True if segment pq intersects any wall segment (thin walls).
        Useful for agent-agent 'visibility' edges.
        """
        # Segment intersection test in 2D: pq with ab
        def seg_intersect(p1, p2, a1, a2) -> bool:
            # orientation / ccw test
            def ccw(A, B, C):
                return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
            return (ccw(p1, a1, a2) != ccw(p2, a1, a2)) and (ccw(p1, p2, a1) != ccw(p1, p2, a2))

        p1 = np.asarray(p, dtype=float)
        p2 = np.asarray(q, dtype=float)
        for w in self.walls:
            a1 = np.array([w.x1, w.y1], dtype=float)
            a2 = np.array([w.x2, w.y2], dtype=float)
            if seg_intersect(p1, p2, a1, a2):
                return True
        return False


    @staticmethod
    def _distance_point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """
        Euclidean distance from point p to line segment ab.
        """
        ab = b - a
        if np.allclose(ab, 0):
            return float(np.linalg.norm(p - a))
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))
    
# --- Quick manual test -------------------------------------------------------


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Throughput benchmark (Phase 1.6)
    # Run: python src/sim_env.py
    # -------------------------------------------------------------------------
    import time as _time
    _env_bench = MultiRobotEnv(world_size=10.0)
    _env_bench.walls = [
        Wall(-8, -8, -8, 8), Wall(8, -8, 8, 8),
        Wall(-8, -8, 8, -8), Wall(-8, 8, 8, 8),
        Wall(0, 4, 0, -4),
    ]
    _env_bench.n_agents = 8
    _env_bench.t = 0
    _env_bench.positions = np.array([
        [-6, -6], [-6, 6], [6, -6], [6, 6],
        [-3, 0],  [3, 0],  [0, -3], [0, 3],
    ], dtype=float)
    _env_bench.goals = _env_bench.positions[::-1].copy()
    _n_steps = 10_000
    _actions = np.zeros((8, 2), dtype=float)
    _t0 = _time.perf_counter()
    for _ in range(_n_steps):
        _env_bench.step(_actions)
    _elapsed = _time.perf_counter() - _t0
    print(f"Throughput: {_n_steps / _elapsed:.1f} steps/sec")
    print(f"Per step:   {_elapsed / _n_steps * 1000:.3f} ms")

    # Quick visual test of maps
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    plot_map(generate_map(MapType.RANDOM), ax=axes[0], title="Random")
    plot_map(generate_map(MapType.MAZE), ax=axes[1], title="Maze")
    plot_map(generate_map(MapType.MAZE_WITH_RANDOM), ax=axes[2], title="Maze + Random")
    plt.tight_layout()
    plt.show()
    """

    # Quick test of environment dynamics

    """
    env = MultiRobotEnv()
    obs = env.reset(MapType.RANDOM, n_agents=12)
    controller = AStarGlobalLocalController()

    plt.ion()
    for t in range(300):
       actions = controller(env)
       obs, reward, done, info = env.step(actions)
       env.render()
       if done:
           print("done", t, info)
           break
    plt.ioff()
    plt.show()

    """

    np.random.seed(33)

    # -----------------------------
    # Create a few maps to test
    # -----------------------------
    ms = generate_mapset(
        seed=43,
        per_category=3,     # 3 per category -> 9 total
        world_size=10.0,
        n_obstacles=5,      # if you switched obstacles to "dot walls", this becomes dot-count
        obstacle_size=0.7,  # irrelevant if you're not using box obstacles anymore
        wall_count=15,
        wall_len_range=(2.0, 8.0),
        margin=0.8,
    )

    # Pick one map to simulate
    chosen_idx = 6  # change 0..len(ms.maps)-1
    chosen_map = ms.maps[chosen_idx]

    # -----------------------------
    # Build env with these walls
    # -----------------------------
    env = MultiRobotEnv(world_size=chosen_map.world_size)

    # Inject walls from the MapSpec into the env
    segs = chosen_map.all_wall_segments()  # <-- this is your method
    env.walls = [Wall(s.x1, s.y1, s.x2, s.y2) for s in segs]

    # Manual reset (avoid env.reset(MapType...) because it regenerates walls)
    n_agents = 5
    env.map_type = None
    env.n_agents = n_agents
    env.t = 0
    env.positions = env._sample_non_colliding_points(n_agents)
    env.goals = env._sample_non_colliding_points(n_agents)
    obs = env._get_obs()

    # Controller
    controller = AStarGlobalLocalController()
    #controller = HarmonicNavigationController(); controller.reset(env)  # harmonic nav (call reset after each env.reset())
    # Bug controller (tangent bug, reactive only, no global map):
    # Re-instantiate every episode — walls are baked in at construction time.
    # from controllers.bug_controller import BugController
    """
    controller = BugController(
        walls          = env.walls,
        n_agents       = env.n_agents,
        max_speed      = env.max_speed,
        goal_tolerance = env.goal_tolerance,
        robot_radius   = env.robot_radius,
        world_size     = env.world_size,
    )
    controller.reset()
    """

    paths = StudentPolicyPaths(
    stats_json=Path("datasets/il_dataset/processed_student_v1/stats.json"),
    checkpoint_pt=Path("datasets/il_dataset/processed_student_v1/checkpoints/best.pt"),
    )

    student = StudentGNNGRUController(
        paths,
        edge_radius=3.0,
        lidar_n_rays=64,
        lidar_max_range=20.0,
        world_size=10.0,
        max_speed=1.5,
    )

    # -----------------------------
    # Sim loop
    # -----------------------------
    plt.ion()
    for t in range(600):
        actions = controller(env)
        #actions = student(env)
        #actions = safety_filter(env, actions)
        obs, reward, done, info = env.step(actions)
        env.render()
        #debug_draw_lidar(env, agent_i=0, n_rays=64, max_range=20.0)
        #plt.pause(0.01)
        if done:
            print("done", t, info)
            break
    plt.ioff()
    plt.show()