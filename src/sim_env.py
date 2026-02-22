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
from data_building.map_generation import generate_mapset 



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
    ):
        self.world_size = world_size
        self.dt = dt
        self.robot_radius = robot_radius
        self.max_speed = max_speed
        self.goal_tolerance = goal_tolerance

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



    def step(self, actions: np.ndarray):
        """
        Step the environment forward one time-step.

        actions: np.ndarray of shape (N, 2) (desired velocities)

        Returns:
            obs: (N, 4) array
            reward: float (negative mean distance to goal)
            done: bool (episode finished)
            info: dict (extra diagnostics)
        """
        assert self.positions is not None and self.goals is not None
        assert actions.shape == (self.n_agents, 2)

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
        for p in positions:
            if self._point_near_any_wall(p, margin=self.robot_radius):
                return True
        return False

    def _check_robot_collisions(self, positions: np.ndarray) -> bool:
        n = positions.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(positions[i] - positions[j]) < 2 * self.robot_radius:
                    return True
        return False
    

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
    ) -> np.ndarray:
        """
        Returns LiDAR-lite for all agents: shape (N, n_rays).
        """
        assert self.positions is not None
        scans = np.zeros((self.n_agents, n_rays), dtype=float)
        for i in range(self.n_agents):
            scans[i] = self.lidar_scan(self.positions[i], n_rays=n_rays, max_range=max_range)
        return scans

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
        seed=23424,
        per_category=3,     # 3 per category -> 9 total
        world_size=10.0,
        n_obstacles=5,      # if you switched obstacles to "dot walls", this becomes dot-count
        obstacle_size=0.7,  # irrelevant if you're not using box obstacles anymore
        wall_count=5,
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
    for t in range(300):
        #actions = controller(env)
        actions = student(env)
        obs, reward, done, info = env.step(actions)
        env.render()
        #debug_draw_lidar(env, agent_i=0, n_rays=64, max_range=20.0)
        #plt.pause(0.01)
        if done:
            print("done", t, info)
            break
    plt.ioff()
    plt.show()