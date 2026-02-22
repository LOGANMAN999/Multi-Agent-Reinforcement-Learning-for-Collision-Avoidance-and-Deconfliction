# src/controllers/harmonic_navigation.py

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np


@dataclass
class HarmonicNavConfig:
    # Grid resolution in x (width) and y (height) directions
    grid_width: int = 64
    grid_height: int = 64

    # Dirichlet boundary values
    boundary_value: float = 1.0  # value on walls / outer boundary
    goal_value: float = 0.0      # value at goal region

    # Physical thickness of obstacles (in world units) when rasterizing walls
    obstacle_thickness: float = 0.5

    # Radius around goal treated as "goal region" (Dirichlet φ = goal_value)
    goal_radius: float = 0.4

    # Iterative solver parameters
    max_iters: int = 1000
    tol: float = 1e-3
    sor_omega: float = 1.5  # relaxation factor in (1,2) for SOR; 1.0 = plain Gauss–Seidel

    # World bounds override (if None, we try to infer from env)
    world_bounds: Optional[Tuple[float, float, float, float]] = None
    # (xmin, xmax, ymin, ymax)


class HarmonicNavigator:
    """
    Compute and use harmonic navigation functions (navigation potentials)
    on a rectangular grid for 2D maps with line-segment walls.

    Typical usage:
        nav = HarmonicNavigator(env, config)
        phi = nav.compute_potential_for_goal(goal)

        # later, for an agent at position pos:
        direction = nav.navigation_direction(pos, phi)

    You can store one phi per distinct goal and reuse it for multiple agents.
    """

    def __init__(self, env, config: Optional[HarmonicNavConfig] = None):
        if config is None:
            config = HarmonicNavConfig()
        self.env = env
        self.config = config

        # Determine world bounds
        self.xmin, self.xmax, self.ymin, self.ymax = self._compute_world_bounds(env, config)
        self.width = config.grid_width
        self.height = config.grid_height

        # Pre-compute grid coordinates
        self.dx = (self.xmax - self.xmin) / (self.width - 1)
        self.dy = (self.ymax - self.ymin) / (self.height - 1)

        xs = np.linspace(self.xmin, self.xmax, self.width)
        ys = np.linspace(self.ymin, self.ymax, self.height)
        self.grid_x, self.grid_y = np.meshgrid(xs, ys)  # shape (H, W)

        # Cache for potentials per goal, if you want
        self._phi_cache: Dict[Tuple[float, float], np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_potential_for_goal(self, goal: np.ndarray) -> np.ndarray:
        """
        Compute the harmonic navigation potential φ for a given goal.

        Args:
            goal: np.array of shape (2,) = (x_goal, y_goal) in world coordinates.

        Returns:
            phi: 2D array of shape (H, W) with harmonic potential values.
        """
        goal_key = (float(goal[0]), float(goal[1]))
        if goal_key in self._phi_cache:
            return self._phi_cache[goal_key]

        # 1) Rasterize obstacles and domain boundary
        is_obstacle = self._rasterize_obstacles()
        is_boundary = self._compute_boundary_mask(is_obstacle)

        # 2) Goal region mask
        is_goal = self._compute_goal_mask(goal)

        # 3) Initialize phi
        phi = np.full((self.height, self.width), self.config.boundary_value, dtype=float)

        # Set boundary / obstacle and goal Dirichlet values
        phi[is_obstacle] = self.config.boundary_value
        phi[is_boundary] = self.config.boundary_value
        phi[is_goal] = self.config.goal_value

        # Interior cells (free space, not boundary, not goal) can be initialized to something in between
        interior_mask = ~(is_obstacle | is_boundary | is_goal)
        phi[interior_mask] = 0.5 * (self.config.boundary_value + self.config.goal_value)

        # 4) Solve Laplace equation by SOR / Gauss-Seidel
        phi = self._solve_harmonic(phi, is_obstacle, is_boundary, is_goal)

        self._phi_cache[goal_key] = phi
        return phi

    def navigation_direction(self, pos: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Get the navigation direction -∇φ at a given world position.

        Args:
            pos: np.array shape (2,) = (x, y)
            phi: 2D potential array for the corresponding goal

        Returns:
            direction: np.array shape (2,). Normalized; returns zero vector if grad ~ 0.
        """
        i, j = self._world_to_grid_indices(pos)

        # Clamp indices to avoid out-of-bounds
        i = np.clip(i, 1, self.height - 2)
        j = np.clip(j, 1, self.width - 2)

        # Central differences
        dphi_dx = (phi[i, j + 1] - phi[i, j - 1]) / (2.0 * self.dx)
        dphi_dy = (phi[i + 1, j] - phi[i - 1, j]) / (2.0 * self.dy)

        grad = np.array([dphi_dx, dphi_dy], dtype=float)
        v = -grad  # navigation direction is downhill

        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return np.zeros(2, dtype=float)
        return v / norm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_world_bounds(self, env, config: HarmonicNavConfig) -> Tuple[float, float, float, float]:
        """
        Determine world bounds [xmin, xmax, ymin, ymax].

        Priority:
          1) config.world_bounds if provided
          2) try env.xmin/xmax/ymin/ymax
          3) fall back to [0, env.width] x [0, env.height]
        """
        if config.world_bounds is not None:
            return config.world_bounds

        # Try env attributes
        xmin = getattr(env, "xmin", None)
        xmax = getattr(env, "xmax", None)
        ymin = getattr(env, "ymin", None)
        ymax = getattr(env, "ymax", None)

        if None not in (xmin, xmax, ymin, ymax):
            return float(xmin), float(xmax), float(ymin), float(ymax)

        # Fallback: use env.width, env.height if present
        width = getattr(env, "width", 10.0)
        height = getattr(env, "height", 10.0)
        return 0.0, float(width), 0.0, float(height)

    def _rasterize_obstacles(self) -> np.ndarray:
        """
        Rasterize line-segment walls into an obstacle mask on the grid.

        Returns:
            is_obstacle: (H, W) bool array, True where cell is treated as obstacle.
        """
        H, W = self.height, self.width
        is_obstacle = np.zeros((H, W), dtype=bool)

        walls = getattr(self.env, "walls", [])
        if not walls:
            return is_obstacle

        # Use either env._distance_point_to_segment if available, or a local implementation.
        has_env_distance = hasattr(self.env, "_distance_point_to_segment")

        for idx_w, w in enumerate(walls):
            a = np.array([w.x1, w.y1], dtype=float)
            b = np.array([w.x2, w.y2], dtype=float)

            # Rough bounding box in grid space to avoid scanning whole grid per wall
            xmin = min(a[0], b[0]) - self.config.obstacle_thickness
            xmax = max(a[0], b[0]) + self.config.obstacle_thickness
            ymin = min(a[1], b[1]) - self.config.obstacle_thickness
            ymax = max(a[1], b[1]) + self.config.obstacle_thickness

            i_min, j_min = self._world_to_grid_indices(np.array([xmin, ymin]))
            i_max, j_max = self._world_to_grid_indices(np.array([xmax, ymax]))

            i_min = np.clip(min(i_min, i_max), 0, H - 1)
            i_max = np.clip(max(i_min, i_max), 0, H - 1)
            j_min = np.clip(min(j_min, j_max), 0, W - 1)
            j_max = np.clip(max(j_min, j_max), 0, W - 1)

            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    px = self.grid_x[i, j]
                    py = self.grid_y[i, j]
                    p = np.array([px, py], dtype=float)

                    if has_env_distance:
                        d = self.env._distance_point_to_segment(p, a, b)
                    else:
                        d = self._distance_point_to_segment_local(p, a, b)

                    if d <= 0.5 * self.config.obstacle_thickness:
                        is_obstacle[i, j] = True

        return is_obstacle

    def _compute_boundary_mask(self, is_obstacle: np.ndarray) -> np.ndarray:
        """
        Mark outer domain boundary as boundary cells (Dirichlet).
        Obstacles are already handled separately.
        """
        H, W = is_obstacle.shape
        is_boundary = np.zeros((H, W), dtype=bool)

        # Outer ring of the grid is boundary
        is_boundary[0, :] = True
        is_boundary[-1, :] = True
        is_boundary[:, 0] = True
        is_boundary[:, -1] = True

        # We don't unset obstacle cells here; they stay obstacles with boundary value.
        return is_boundary

    def _compute_goal_mask(self, goal: np.ndarray) -> np.ndarray:
        """
        Mark cells within goal_radius of goal as goal region (Dirichlet φ = goal_value).
        """
        gx, gy = goal
        dx = self.grid_x - gx
        dy = self.grid_y - gy
        dist = np.sqrt(dx * dx + dy * dy)
        is_goal = dist <= self.config.goal_radius
        return is_goal

    def _solve_harmonic(
        self,
        phi: np.ndarray,
        is_obstacle: np.ndarray,
        is_boundary: np.ndarray,
        is_goal: np.ndarray
    ) -> np.ndarray:
        """
        Solve the discrete Laplace equation with SOR:

            φ[i,j] = 0.25 * (φ[i+1,j] + φ[i-1,j] + φ[i,j+1] + φ[i,j-1])

        with φ fixed on boundary, obstacles, and goal region.

        Args:
            phi: initial potential array (H, W)
            is_obstacle, is_boundary, is_goal: boolean masks

        Returns:
            phi: converged potential.
        """
        H, W = phi.shape
        omega = self.config.sor_omega

        # Fixed cells: do not update obstacles, boundaries, or goal cells.
        is_fixed = is_obstacle | is_boundary | is_goal

        for it in range(self.config.max_iters):
            max_delta = 0.0

            # Gauss-Seidel sweep over interior (excluding outermost ring)
            for i in range(1, H - 1):
                for j in range(1, W - 1):
                    if is_fixed[i, j]:
                        continue

                    # 5-point stencil
                    neighbor_avg = 0.25 * (
                        phi[i + 1, j] +
                        phi[i - 1, j] +
                        phi[i, j + 1] +
                        phi[i, j - 1]
                    )

                    new_val = (1.0 - omega) * phi[i, j] + omega * neighbor_avg
                    delta = abs(new_val - phi[i, j])
                    if delta > max_delta:
                        max_delta = delta

                    phi[i, j] = new_val

            if max_delta < self.config.tol:
                # print(f"[HarmonicNavigator] Converged in {it+1} iterations, max_delta={max_delta:.3e}")
                break

        return phi

    def _world_to_grid_indices(self, pos: np.ndarray) -> Tuple[int, int]:
        """
        Map world coordinates (x, y) to grid indices (i, j).
        i indexes rows (y), j indexes columns (x).
        """
        x, y = float(pos[0]), float(pos[1])

        # Normalize to [0,1] across bounds
        u = (x - self.xmin) / (self.xmax - self.xmin + 1e-9)
        v = (y - self.ymin) / (self.ymax - self.ymin + 1e-9)

        j = int(round(u * (self.width - 1)))
        i = int(round(v * (self.height - 1)))
        return i, j

    @staticmethod
    def _distance_point_to_segment_local(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """
        Local fallback for distance from point p to segment ab.
        """
        ab = b - a
        denom = np.dot(ab, ab)
        if denom < 1e-12:
            return float(np.linalg.norm(p - a))

        t = np.dot(p - a, ab) / denom
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))
