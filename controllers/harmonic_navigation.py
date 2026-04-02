"""
Harmonic Potential Field Controller
====================================
Solves Laplace's equation (∇²φ = 0) on a 128×128 grid for the current map.
Goal cells are set to 0 (sink), boundary/obstacle cells to 1 (source).
The resulting potential has no local minima — agents follow the negative gradient
from their current position toward the goal.

Interface matches AStarGlobalLocalController:
    controller = HarmonicNavigationController()
    controller.reset(env)            # call once per episode (builds all fields)
    velocities = controller(env)     # call each step → (N, 2) float velocities

Grid parameters (validated for this env):
    world_size = 10.0  → world spans [-10, 10]² = 20 × 20 m
    grid_size  = 128   → cell_size ≈ 0.156 m
    inflate    = 0.25 m (= robot_radius) → ~1.6 cells; matches the physical agent
                 size so the Laplace field deflects agents before collision range

Solver priority:
    1. scipy.sparse.linalg.spsolve  (direct, ~50–150 ms for 128²)
    2. Numpy vectorized Gauss-Seidel SOR  (fallback, 1 500 iters, ω = 1.8)

File: src/controllers/harmonic_navigation.py
Dependencies: numpy, (optional) scipy
"""

from __future__ import annotations

import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular import at runtime — sim_env imports this file
    from sim_env import MultiRobotEnv

# ---------------------------------------------------------------------------
# Grid / solver constants
# ---------------------------------------------------------------------------

GRID_SIZE: int = 128          # grid cells per axis (higher = better narrow-passage handling)
INFLATE_RADIUS_WORLD: float = 0.25   # world-units to inflate obstacles (= robot_radius = 0.25 m)
                                      # Matches the physical agent size so the Laplace gradient
                                      # deflects agents before they reach collision range.
SOR_OMEGA: float = 1.8        # SOR relaxation factor  (1 < ω < 2 for convergence)
SOR_MAX_ITERS: int = 1500     # SOR iteration cap
SOR_TOL: float = 1e-4         # SOR convergence tolerance (max-norm residual)


# ---------------------------------------------------------------------------
# Helper: build obstacle mask from env walls
# ---------------------------------------------------------------------------

def _build_obstacle_mask(walls, world_size: float, grid_size: int,
                         inflate_radius: float) -> np.ndarray:
    """
    Returns a boolean [H, W] array where True = obstacle cell.

    Outer border row/column is always obstacle (world boundary).
    Any cell whose centre is within `inflate_radius` world-units of any wall
    segment is marked as obstacle (configuration-space inflation).

    Args:
        walls:          list of Wall(x1, y1, x2, y2) objects
        world_size:     half-extent of square world  (world spans [-ws, ws]²)
        grid_size:      number of cells per axis (H = W = grid_size)
        inflate_radius: minimum clearance in world units

    Returns:
        mask: bool [grid_size, grid_size]  (row = y, col = x)
    """
    H = W = grid_size
    cell_size = (2.0 * world_size) / grid_size  # world units per cell

    # Grid cell centres — shape [H, W, 2]
    xs = np.linspace(-world_size + cell_size * 0.5,
                      world_size - cell_size * 0.5, W)
    ys = np.linspace(-world_size + cell_size * 0.5,
                      world_size - cell_size * 0.5, H)
    cx, cy = np.meshgrid(xs, ys)          # [H, W] each
    pts = np.stack([cx.ravel(), cy.ravel()], axis=1)  # [H*W, 2]

    mask = np.zeros(H * W, dtype=bool)

    if walls:
        # Wall endpoints → [W_walls, 4]
        wall_arr = np.array([[w.x1, w.y1, w.x2, w.y2] for w in walls],
                            dtype=np.float64)  # [n_walls, 4]
        A = wall_arr[:, :2]   # [n_walls, 2]
        B = wall_arr[:, 2:]   # [n_walls, 2]
        AB = B - A            # [n_walls, 2]
        AB_sq = np.sum(AB * AB, axis=1, keepdims=True)  # [n_walls, 1]
        AB_sq = np.maximum(AB_sq, 1e-12)

        # Point-to-segment distance: vectorised over all cells × all walls
        # pts: [P, 2], A: [n_walls, 2]
        # AP: [P, n_walls, 2]
        AP = pts[:, None, :] - A[None, :, :]        # [P, n_walls, 2]
        t = np.sum(AP * AB[None], axis=2) / AB_sq.T  # [P, n_walls]
        t = np.clip(t, 0.0, 1.0)
        closest = A[None] + t[:, :, None] * AB[None]  # [P, n_walls, 2]
        diff = pts[:, None, :] - closest               # [P, n_walls, 2]
        dists = np.sqrt(np.sum(diff * diff, axis=2))   # [P, n_walls]
        min_dists = dists.min(axis=1)                  # [P]
        mask |= (min_dists < inflate_radius)

    mask = mask.reshape(H, W)

    # World boundary → always obstacle
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    return mask


# ---------------------------------------------------------------------------
# Helper: solve Laplace equation on the free cells
# ---------------------------------------------------------------------------

def _solve_laplace(obstacle_mask: np.ndarray,
                   goal_row: int, goal_col: int) -> np.ndarray:
    """
    Solves ∇²φ = 0 with:
        φ = 0  at goal cell  (sink)
        φ = 1  at all obstacle/boundary cells  (source)

    Tries scipy sparse direct solver first; falls back to SOR.

    Args:
        obstacle_mask: bool [H, W]
        goal_row, goal_col: grid indices of the goal

    Returns:
        phi: float32 [H, W]  potential field, 0 at goal, 1 at walls
    """
    H, W = obstacle_mask.shape

    # Clamp goal into a free cell (nearest free if goal is inside obstacle)
    if obstacle_mask[goal_row, goal_col]:
        free_rows, free_cols = np.where(~obstacle_mask)
        idx = np.argmin((free_rows - goal_row) ** 2 + (free_cols - goal_col) ** 2)
        goal_row, goal_col = int(free_rows[idx]), int(free_cols[idx])

    # Identify free cells (non-obstacle, non-goal)
    goal_mask = np.zeros((H, W), dtype=bool)
    goal_mask[goal_row, goal_col] = True

    free_mask = ~obstacle_mask & ~goal_mask   # cells with unknown φ
    free_idx = np.flatnonzero(free_mask)       # [F]
    F = len(free_idx)

    phi = np.ones((H, W), dtype=np.float64)
    phi[goal_row, goal_col] = 0.0

    if F == 0:
        return phi.astype(np.float32)

    # Try sparse direct solver (fully vectorised matrix construction — no Python cell loop)
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        cell_to_var = np.full(H * W, -1, dtype=np.int32)
        cell_to_var[free_idx] = np.arange(F, dtype=np.int32)

        phi_flat = phi.ravel()
        rows_f = free_idx // W   # [F] row index of each free cell
        cols_f = free_idx % W    # [F] col index of each free cell
        rhs = np.zeros(F, dtype=np.float64)

        # Diagonal: 4 for every free cell
        diag_rows = np.arange(F, dtype=np.int32)
        diag_cols = np.arange(F, dtype=np.int32)
        diag_data = np.full(F, 4.0)

        off_rows, off_cols, off_data = [], [], []

        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = rows_f + dr           # [F] neighbour row
            nc = cols_f + dc           # [F] neighbour col

            in_bounds = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
            flat_nb = nr * W + nc      # [F] flat neighbour index
            flat_nb = np.where(in_bounds, flat_nb, 0)   # safe index (masked below)

            var_nb = cell_to_var[flat_nb]               # [F] variable index or -1

            # Free neighbour → off-diagonal -1 entry
            is_free_nb = in_bounds & (var_nb >= 0)
            k_idx = diag_rows[is_free_nb]
            off_rows.append(k_idx)
            off_cols.append(var_nb[is_free_nb])
            off_data.append(np.full(k_idx.shape, -1.0))

            # Fixed neighbour (obstacle=1 or goal=0) → contributes to RHS
            is_fixed_nb = in_bounds & (var_nb < 0)
            np.add.at(rhs, diag_rows[is_fixed_nb], phi_flat[flat_nb[is_fixed_nb]])

        all_rows = np.concatenate([diag_rows] + off_rows)
        all_cols = np.concatenate([diag_cols] + off_cols)
        all_data = np.concatenate([diag_data] + off_data)

        A_mat = sp.csr_matrix((all_data, (all_rows, all_cols)), shape=(F, F))
        x = spla.spsolve(A_mat, rhs)
        phi.ravel()[free_idx] = x

    except Exception:
        # Fallback: vectorised Gauss-Seidel SOR
        phi = _solve_sor(obstacle_mask, goal_mask, phi)

    return phi.astype(np.float32)


def _solve_sor(obstacle_mask: np.ndarray,
               goal_mask: np.ndarray,
               phi: np.ndarray) -> np.ndarray:
    """
    Successive Over-Relaxation (SOR) solver for Laplace on the free cells.
    Vectorised checkerboard update — no Python loops over cells.
    """
    H, W = phi.shape
    fixed = obstacle_mask | goal_mask

    r, c = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    even = ((r + c) % 2 == 0) & ~fixed
    odd  = ((r + c) % 2 == 1) & ~fixed

    for _ in range(SOR_MAX_ITERS):
        # --- update even cells ---
        phi_up    = np.roll(phi, -1, axis=0)
        phi_down  = np.roll(phi,  1, axis=0)
        phi_left  = np.roll(phi, -1, axis=1)
        phi_right = np.roll(phi,  1, axis=1)
        avg = 0.25 * (phi_up + phi_down + phi_left + phi_right)
        phi_new_even = phi + SOR_OMEGA * (avg - phi)
        phi = np.where(even, phi_new_even, phi)
        # re-apply BCs
        phi[fixed] = np.where(obstacle_mask[fixed], 1.0, 0.0)

        # --- update odd cells ---
        phi_up    = np.roll(phi, -1, axis=0)
        phi_down  = np.roll(phi,  1, axis=0)
        phi_left  = np.roll(phi, -1, axis=1)
        phi_right = np.roll(phi,  1, axis=1)
        avg = 0.25 * (phi_up + phi_down + phi_left + phi_right)
        phi_new_odd = phi + SOR_OMEGA * (avg - phi)
        residual = np.max(np.abs(phi_new_odd[odd] - phi[odd])) if odd.any() else 0.0
        phi = np.where(odd, phi_new_odd, phi)
        phi[fixed] = np.where(obstacle_mask[fixed], 1.0, 0.0)

        if residual < SOR_TOL:
            break

    return phi


# ---------------------------------------------------------------------------
# Helper: compute gradient field from potential
# ---------------------------------------------------------------------------

def _compute_gradient(phi: np.ndarray,
                      obstacle_mask: np.ndarray) -> np.ndarray:
    """
    Central finite-difference gradient of φ.

    Returns:
        grad: float32 [H, W, 2]  where grad[r, c] = [dφ/dx, dφ/dy]
              zeroed at obstacle cells.
    """
    H, W = phi.shape
    cell_size = 1.0  # normalised; caller will scale by actual cell_size

    gx = np.zeros((H, W), dtype=np.float32)
    gy = np.zeros((H, W), dtype=np.float32)

    # Interior: central differences
    gx[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * cell_size)
    gy[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * cell_size)

    # Borders: forward / backward differences
    gx[:, 0]  = (phi[:, 1]  - phi[:, 0])   / cell_size
    gx[:, -1] = (phi[:, -1] - phi[:, -2])  / cell_size
    gy[0, :]  = (phi[1, :]  - phi[0, :])   / cell_size
    gy[-1, :] = (phi[-1, :] - phi[-2, :])  / cell_size

    # Zero gradient inside obstacles
    gx[obstacle_mask] = 0.0
    gy[obstacle_mask] = 0.0

    return np.stack([gx, gy], axis=-1)  # [H, W, 2]


# ---------------------------------------------------------------------------
# Helper: bilinear interpolation of a [H, W, 2] field at world positions
# ---------------------------------------------------------------------------

def _bilinear_sample(field: np.ndarray,
                     world_pos: np.ndarray,
                     world_size: float,
                     grid_size: int) -> np.ndarray:
    """
    Bilinearly interpolates `field` ([H, W, D]) at world positions.

    Args:
        field:      [H, W, D]
        world_pos:  [N, 2] in world coords  ([-world_size, world_size]²)
        world_size: half-extent of world
        grid_size:  cells per axis

    Returns:
        values: [N, D] interpolated field values
    """
    H, W = field.shape[:2]
    D = field.shape[2]
    cell_size = (2.0 * world_size) / grid_size

    # World → fractional grid index (row = y-axis, col = x-axis)
    col_f = (world_pos[:, 0] + world_size) / cell_size - 0.5   # [N]
    row_f = (world_pos[:, 1] + world_size) / cell_size - 0.5   # [N]

    col_f = np.clip(col_f, 0.0, W - 1.0 - 1e-6)
    row_f = np.clip(row_f, 0.0, H - 1.0 - 1e-6)

    r0 = np.floor(row_f).astype(int)
    c0 = np.floor(col_f).astype(int)
    r1 = np.minimum(r0 + 1, H - 1)
    c1 = np.minimum(c0 + 1, W - 1)

    dr = (row_f - r0)[:, None]   # [N, 1]
    dc = (col_f - c0)[:, None]   # [N, 1]

    # Bilinear weights
    v00 = field[r0, c0]  # [N, D]
    v01 = field[r0, c1]
    v10 = field[r1, c0]
    v11 = field[r1, c1]

    return ((1 - dr) * (1 - dc) * v00 +
            (1 - dr) *      dc  * v01 +
                 dr  * (1 - dc) * v10 +
                 dr  *      dc  * v11)


# ---------------------------------------------------------------------------
# Main controller class
# ---------------------------------------------------------------------------

class HarmonicNavigationController:
    """
    Harmonic potential field navigation controller.

    Computes one Laplace field per agent goal at the start of each episode
    (call `reset(env)`), then provides velocity commands each step via
    `__call__(env)`.

    Usage:
        controller = HarmonicNavigationController()
        controller.reset(env)            # once per episode
        velocities = controller(env)     # each step → (N, 2) np.float32

    Attributes:
        grid_size:          cells per axis (default 256)
        max_speed:          velocity magnitude cap (matched to env at reset)
        repulsion_radius:   agents within this world-unit distance repel each other
        repulsion_strength: peak repulsive force magnitude (world units/s)
        repulsion_sigma:    Gaussian falloff distance (world units)
        _obstacle_mask:     bool [H, W] — rebuilt each reset
        _gradients:         list of [H, W, 2] gradient fields, one per agent
    """

    def __init__(self,
                 grid_size: int = GRID_SIZE,
                 repulsion_radius: float = 1.5,
                 repulsion_strength: float = 5.0,
                 repulsion_sigma: float = 0.5):
        self.grid_size = grid_size
        self.max_speed: float = 1.5
        self.repulsion_radius = repulsion_radius
        self.repulsion_strength = repulsion_strength
        self.repulsion_sigma = repulsion_sigma
        self._obstacle_mask: Optional[np.ndarray] = None
        self._gradients: list = []
        self._world_size: float = 10.0
        self._cell_size: float = (2.0 * 10.0) / grid_size
        self._walls: list = []   # stored so simulate_flow_hits_wall can rebuild with wider margin

    # ------------------------------------------------------------------
    # Reset: called once per episode to (re)build fields for all goals
    # ------------------------------------------------------------------

    def reset(self, env: "MultiRobotEnv") -> None:
        """
        Builds the obstacle mask and solves one Laplace field per agent goal.

        Args:
            env: MultiRobotEnv instance after reset() has been called
                 (env.walls, env.goals, env.n_agents, env.world_size must exist)
        """
        ws = getattr(env, "world_size", 10.0)
        self._world_size = ws
        self._cell_size = (2.0 * ws) / self.grid_size
        self.max_speed = getattr(env, "max_speed", 1.5)

        # Store walls for on-demand wider-margin scan masks
        self._walls = list(getattr(env, "walls", []))

        # Build shared obstacle mask (walls + boundary inflation)
        self._obstacle_mask = _build_obstacle_mask(
            walls=self._walls,
            world_size=ws,
            grid_size=self.grid_size,
            inflate_radius=INFLATE_RADIUS_WORLD,
        )

        # Solve one field per agent goal
        n_agents = env.n_agents
        goals = env.goals  # [N, 2] world coords
        self._gradients = []
        for i in range(n_agents):
            gx, gy = self._world_to_grid(goals[i, 0], goals[i, 1])
            phi = _solve_laplace(self._obstacle_mask, gy, gx)
            grad = _compute_gradient(phi, self._obstacle_mask)
            # Scale gradient from grid units to world units
            grad /= self._cell_size
            self._gradients.append(grad)

    # ------------------------------------------------------------------
    # Step: returns (N, 2) velocity commands
    # ------------------------------------------------------------------

    def __call__(self, env: "MultiRobotEnv", repulsion_radii=None, repulsion_strengths=None) -> np.ndarray:
        """
        Returns velocity commands for all agents.

        Each agent's velocity is the sum of:
          1. Harmonic field velocity  — steers toward goal, no local minima
          2. Agent repulsion          — Gaussian pushback from nearby agents

        Agents that have no field (e.g., reset() not yet called) fall back
        to direct-to-goal heading at max_speed.

        Args:
            env: MultiRobotEnv instance

        Returns:
            velocities: float32 [N, 2]
        """
        n_agents = env.n_agents
        positions = env.positions  # [N, 2]
        goals = env.goals          # [N, 2]
        active = getattr(env, "active", np.ones(n_agents, dtype=bool))

        velocities = np.zeros((n_agents, 2), dtype=np.float32)

        if not self._gradients or len(self._gradients) != n_agents:
            return self._direct_to_goal(positions, goals, active)

        ws = self._world_size
        gs = self.grid_size
        pos_arr = positions.astype(np.float64)  # [N, 2]

        # ------------------------------------------------------------------
        # 1. Harmonic field velocities (per-agent gradient lookup)
        # ------------------------------------------------------------------
        for i in range(n_agents):
            if not active[i]:
                continue

            grad_i = _bilinear_sample(
                self._gradients[i],
                pos_arr[i:i+1],
                ws, gs,
            )[0]  # [2]

            vel = -grad_i.astype(np.float32)
            mag = float(np.linalg.norm(vel))
            if mag > 1e-6:
                vel = vel * (self.max_speed / mag)
            else:
                # Gradient vanished — fall back to direct heading
                to_goal = goals[i] - positions[i]
                dist = np.linalg.norm(to_goal)
                if dist > 1e-6:
                    vel = (to_goal / dist * self.max_speed).astype(np.float32)

            velocities[i] = vel

        # ------------------------------------------------------------------
        # 2. Agent-agent repulsion  (vectorised over all N×N pairs)
        # ------------------------------------------------------------------
        if n_agents > 1:
            pos_f = positions.astype(np.float32)           # [N, 2]

            # Displacement vectors pointing away from each neighbour
            diffs = pos_f[:, None, :] - pos_f[None, :, :] # [N, N, 2]
            dists = np.linalg.norm(diffs, axis=2)          # [N, N]

            # CHANGED: was scalar self.repulsion_radius, now per-agent array
            _radii = (
                np.asarray(repulsion_radii, dtype=np.float32)
                if repulsion_radii is not None
                else np.full(n_agents, self.repulsion_radius, dtype=np.float32)
            )

            # CHANGED: was scalar self.repulsion_strength, now per-agent array
            _strengths = (
                np.asarray(repulsion_strengths, dtype=np.float32)
                if repulsion_strengths is not None
                else np.full(n_agents, self.repulsion_strength, dtype=np.float32)
            )

            if np.any(_strengths > 0.0):
                # CHANGED: per-source radius — agent i repelled by j if within j's radius
                in_range = (dists > 1e-6) & (dists < _radii[None, :])
                in_range &= active[:, None] & active[None, :]  # both must be active

                # CHANGED: per-source strength — j's strength drives repulsion weight
                weights = _strengths[None, :] * np.exp(
                    -dists / self.repulsion_sigma
                )                                              # [N, N]
                weights[~in_range] = 0.0

                # Weighted unit direction vectors
                unit_dirs = diffs / np.maximum(dists[:, :, None], 1e-6)  # [N, N, 2]
                repulsion = (weights[:, :, None] * unit_dirs).sum(axis=1) # [N, 2]

                velocities += repulsion.astype(np.float32)

        # ------------------------------------------------------------------
        # 3. Re-clip combined velocity to max_speed; zero inactive agents
        # ------------------------------------------------------------------
        mags = np.linalg.norm(velocities, axis=1, keepdims=True)       # [N, 1]
        scale = np.where(mags > self.max_speed, self.max_speed / mags, 1.0)
        velocities = (velocities * scale).astype(np.float32)
        velocities[~active] = 0.0

        return velocities

    # ------------------------------------------------------------------
    # get_preferred_velocity: for GNN node feature extraction
    # ------------------------------------------------------------------

    def get_preferred_velocity(self, agent_idx: int,
                               pos: np.ndarray,
                               goal: np.ndarray) -> np.ndarray:
        """
        Returns the harmonic preferred velocity [2] for a single agent
        at an arbitrary position.  Used by graph builders to populate the
        `preferred_velocity` node feature without re-solving the field.

        Args:
            agent_idx: index into self._gradients
            pos:       [2] world position
            goal:      [2] world position (unused if field is cached)

        Returns:
            vel: [2] float32 velocity vector (capped at max_speed)
        """
        if not self._gradients or agent_idx >= len(self._gradients):
            to_goal = (goal - pos).astype(np.float32)
            dist = np.linalg.norm(to_goal)
            if dist > 1e-6:
                return to_goal * (self.max_speed / dist)
            return np.zeros(2, dtype=np.float32)

        grad = _bilinear_sample(
            self._gradients[agent_idx],
            np.array([pos], dtype=np.float64),
            self._world_size, self.grid_size,
        )[0]
        vel = -grad.astype(np.float32)
        mag = float(np.linalg.norm(vel))
        if mag > 1e-6:
            return vel * (self.max_speed / mag)
        to_goal = (goal - pos).astype(np.float32)
        dist = np.linalg.norm(to_goal)
        if dist > 1e-6:
            return to_goal * (self.max_speed / dist)
        return np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------
    # simulate_flow_hits_wall: 20-step lookahead for hybrid controller
    # ------------------------------------------------------------------

    # Number of intermediate obstacle checks performed between each dt step.
    # Catches agents that would step over a thin (1-cell) wall without landing
    # on an obstacle cell at the coarse dt resolution.
    _FLOW_SUBSTEPS: int = 32

    def simulate_flow_hits_wall(
        self,
        positions: np.ndarray,
        dt: float,
        lookahead_steps: int = 20,
        goals: Optional[np.ndarray] = None,
        goal_proximity_stop: float = 1.0,
        safety_margin: float = 0.0,
    ) -> np.ndarray:
        """
        Simulate each agent's harmonic-field trajectory for ``lookahead_steps``
        discrete steps and detect wall collisions.

        The simulated trajectory is identical to the real trajectory an agent
        would follow under the harmonic controller alone: same bilinear gradient
        lookup, same max_speed normalisation, same dt.  Agent-agent repulsion is
        NOT modelled — this is a pure wall-geometry check.

        Each dt step is divided into ``_FLOW_SUBSTEPS`` sub-steps so that thin
        (1-cell) walls are caught even when an agent would otherwise step clean
        over them at the coarse dt resolution.

        ``safety_margin`` widens the obstacle mask used for the scan (without
        changing the Laplace field) to catch agents whose trajectory passes
        through a rasterization gap in a diagonal wall (~1 grid cell ≈ 0.156 m).

        Args:
            positions:           [N, 2] current world positions.
            dt:                  Simulation timestep; should match env.dt.
            lookahead_steps:     Number of forward steps to simulate (default 20).
            goals:               [N, 2] goal positions.  When the simulated
                                 trajectory comes within ``goal_proximity_stop``
                                 of the goal, the lookahead stops early to avoid
                                 false positives when the goal is near a wall.
            goal_proximity_stop: Distance threshold (world units) for early
                                 termination.  Default 1.0 m.
            safety_margin:       Extra clearance (world units) added to the
                                 obstacle inflation radius for this scan only,
                                 to catch diagonal-wall rasterization gaps.
                                 Does NOT change the Laplace field.  Default 0.0.

        Returns:
            hits: [N] bool — True if agent i's simulated trajectory enters an
                  inflated obstacle cell within ``lookahead_steps`` steps.
        """
        N = len(positions)
        hits = np.zeros(N, dtype=bool)

        if self._obstacle_mask is None or len(self._gradients) < N:
            return hits

        ws = self._world_size
        gs = self.grid_size
        cs = self._cell_size
        n_sub = self._FLOW_SUBSTEPS
        sub_dt = 1.0 / n_sub          # fractional advance per sub-step

        # Build a wider obstacle mask for the scan if a safety margin is requested.
        # This catches agents whose solo path skims a wall within repulsion range.
        if safety_margin > 0.0:
            scan_mask = _build_obstacle_mask(
                self._walls, ws, gs, INFLATE_RADIUS_WORLD + safety_margin
            )
        else:
            scan_mask = self._obstacle_mask

        for i in range(N):
            pos = np.array(positions[i], dtype=np.float64)

            # Skip agents already within goal_proximity_stop of their goal —
            # the path approaching a goal that sits near a wall is legitimate.
            if goals is not None:
                if np.linalg.norm(pos - goals[i]) < goal_proximity_stop:
                    continue

            agent_hit = False
            for _ in range(lookahead_steps):
                # Check current position before computing the next step.
                col = int(np.clip((pos[0] + ws) / cs, 0, gs - 1))
                row = int(np.clip((pos[1] + ws) / cs, 0, gs - 1))
                if scan_mask[row, col]:
                    agent_hit = True
                    break

                # Bilinear-interpolated gradient at the simulated position
                # (same call path as __call__ uses for the real trajectory).
                grad = _bilinear_sample(
                    self._gradients[i],
                    pos.reshape(1, 2),
                    ws, gs,
                )[0]                         # [2]

                vel = -grad.astype(np.float64)
                mag = np.linalg.norm(vel)
                if mag > 1e-6:
                    vel *= self.max_speed / mag
                else:
                    # Gradient vanished — mirror HarmonicNavigationController.__call__:
                    # fall back to direct-to-goal heading.  This is the exact path the
                    # real controller takes, and it can drive an agent straight into a
                    # wall when the goal is behind it (long-wall bisector case).
                    if goals is not None:
                        to_goal = goals[i] - pos
                        dist_g = np.linalg.norm(to_goal)
                        if dist_g > 1e-6:
                            vel = (to_goal / dist_g) * self.max_speed
                        else:
                            break   # at goal, stop
                    else:
                        break       # no goal info, stop

                # Sub-step collision checks along the full dt arc.
                # Checks fractions 1/n_sub, 2/n_sub, …, 1 of the step so that
                # a thin wall cannot be skipped over between whole-step samples.
                step = dt * vel
                sub_hit = False
                for k in range(1, n_sub + 1):
                    interp = pos + (k * sub_dt) * step
                    c = int(np.clip((interp[0] + ws) / cs, 0, gs - 1))
                    r = int(np.clip((interp[1] + ws) / cs, 0, gs - 1))
                    if scan_mask[r, c]:
                        sub_hit = True
                        break
                if sub_hit:
                    agent_hit = True
                    break

                pos += step

                # Early-stop once the simulated position is close to the goal.
                if goals is not None:
                    if np.linalg.norm(pos - goals[i]) < goal_proximity_stop:
                        break

            hits[i] = agent_hit

        return hits

    # ------------------------------------------------------------------
    # compute_potential_for_goal: compatibility with sim_env.py reference
    # ------------------------------------------------------------------

    def compute_potential_for_goal(self, goal: np.ndarray) -> np.ndarray:
        """
        Solves and returns the Laplace potential [H, W] for a single goal
        without caching.  Provided for compatibility with any code that
        calls `self.navigator.compute_potential_for_goal(goal)`.

        Args:
            goal: [2] world coordinates

        Returns:
            phi: float32 [H, W]
        """
        if self._obstacle_mask is None:
            raise RuntimeError("Call reset(env) before compute_potential_for_goal.")
        gx, gy = self._world_to_grid(goal[0], goal[1])
        return _solve_laplace(self._obstacle_mask, gy, gx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _world_to_grid(self, wx: float, wy: float):
        """World coords → (col, row) integer grid indices."""
        cs = self._cell_size
        ws = self._world_size
        col = int(np.clip((wx + ws) / cs, 0, self.grid_size - 1))
        row = int(np.clip((wy + ws) / cs, 0, self.grid_size - 1))
        return col, row

    @staticmethod
    def _direct_to_goal(positions: np.ndarray,
                        goals: np.ndarray,
                        active: np.ndarray,
                        max_speed: float = 1.5) -> np.ndarray:
        """Simple direct-to-goal fallback when fields are unavailable."""
        to_goal = goals - positions          # [N, 2]
        dists = np.linalg.norm(to_goal, axis=1, keepdims=True)  # [N, 1]
        safe_dists = np.maximum(dists, 1e-6)
        vels = to_goal / safe_dists * max_speed  # [N, 2]
        vels[~active] = 0.0
        return vels.astype(np.float32)
