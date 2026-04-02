"""
Static visualization of harmonic potential field and A* path for a single agent.

Generates two side-by-side plots for a fixed map, start position, and goal:
  Left  — Harmonic potential field: red near obstacles/walls (phi ~= 1), blue
           in free space (phi ~= 0), with normalized navigation-direction arrows.
  Right — Occupancy grid used by A* with the planned (smoothed) path overlaid.

Map geometry is generated via data_building/map_generation.py (no hard-coded
maze templates). Use --category to pick the map type.

Usage:
    python visualization/visualize_nav_functions.py
    python visualization/visualize_nav_functions.py --category walls_only
    python visualization/visualize_nav_functions.py --category obstacles_only --n_obstacles 15
    python visualization/visualize_nav_functions.py --start -6 -6 --goal 6 6
    python visualization/visualize_nav_functions.py --no_smooth --save nav_vis.png
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

from data_building.map_generation import generate_map, Segment, MapSpec
from controllers.harmonic_navigation import (
    _build_obstacle_mask,
    _solve_laplace,
    _compute_gradient,
    GRID_SIZE,
    INFLATE_RADIUS_WORLD,
)
from controllers.astar_global_local import AStarGlobalLocalController, AStarConfig


WORLD_SIZE = 10.0  # world spans [-WORLD_SIZE, WORLD_SIZE]²


# ---------------------------------------------------------------------------
# Tiny mock env so controller helpers work without a running simulation
# ---------------------------------------------------------------------------

class _MockEnv:
    """Minimal env-like namespace with the attributes controllers need."""
    def __init__(self, walls, start, goal, world_size=WORLD_SIZE,
                 max_speed=1.5, robot_radius=0.25, dt=0.1):
        self.walls        = walls
        self.positions    = np.array([start], dtype=float)
        self.goals        = np.array([goal],  dtype=float)
        self.n_agents     = 1
        self.world_size   = world_size
        self.max_speed    = max_speed
        self.robot_radius = robot_radius
        self.goal_tolerance = 0.5
        self.dt = dt
        self.t  = 0


# ---------------------------------------------------------------------------
# Map generation wrapper
# ---------------------------------------------------------------------------

def build_map(category: str, seed: int,
              n_obstacles: int, wall_count: int,
              wall_len_min: float, wall_len_max: float,
              world_size: float = WORLD_SIZE) -> MapSpec:
    """Generate one MapSpec via data_building/map_generation.py."""
    rng = np.random.default_rng(seed)
    return generate_map(
        rng=rng,
        category=category,
        map_id=f"{category}_{seed}",
        world_size=world_size,
        n_obstacles=n_obstacles,
        obstacle_size=0.7,
        wall_count=wall_count,
        wall_len_range=(wall_len_min, wall_len_max),
        margin=0.8,
    )


# ---------------------------------------------------------------------------
# Harmonic field helpers
# ---------------------------------------------------------------------------

def _world_to_grid_coords(wx, wy, world_size, grid_size):
    """World (x, y) -> (col, row) integer grid indices."""
    cell_size = (2.0 * world_size) / grid_size
    col = int(np.clip((wx + world_size) / cell_size, 0, grid_size - 1))
    row = int(np.clip((wy + world_size) / cell_size, 0, grid_size - 1))
    return col, row


def compute_harmonic_field(walls, goal,
                           world_size=WORLD_SIZE, grid_size=GRID_SIZE):
    """
    Build obstacle mask, solve Laplace equation, compute gradient.

    Returns:
        phi:      float32 [H, W]  potential (0 at goal, 1 at obstacles)
        grad:     float32 [H, W, 2]  world-unit gradient [gx, gy]
        obs_mask: bool    [H, W]
        extent:   [xmin, xmax, ymin, ymax] for imshow
    """
    obs_mask = _build_obstacle_mask(walls, world_size, grid_size, INFLATE_RADIUS_WORLD)
    goal_col, goal_row = _world_to_grid_coords(goal[0], goal[1], world_size, grid_size)
    phi = _solve_laplace(obs_mask, goal_row, goal_col)

    cell_size = (2.0 * world_size) / grid_size
    grad = _compute_gradient(phi, obs_mask) / cell_size  # world units

    extent = [-world_size, world_size, -world_size, world_size]
    return phi, grad, obs_mask, extent


# ---------------------------------------------------------------------------
# A* path helpers
# ---------------------------------------------------------------------------

def compute_astar_path(walls, start, goal,
                       world_size=WORLD_SIZE, smooth=True):
    """
    Build occupancy grid and run A* for a single agent.

    Returns:
        grid:       bool [H, W]  True = obstacle
        path_world: list of [2] world-coord waypoints (may be empty)
    """
    env  = _MockEnv(walls, start, goal, world_size=world_size)
    ctrl = AStarGlobalLocalController(astar_config=AStarConfig(grid_size=128))
    ctrl._build_grid(env)

    grid = ctrl._grid
    H, W = grid.shape

    start_ij = AStarGlobalLocalController._world_to_grid(
        np.array(start), world_size, H, W)
    goal_ij = AStarGlobalLocalController._world_to_grid(
        np.array(goal),  world_size, H, W)

    if grid[start_ij]:
        start_ij = ctrl._nearest_free_cell(grid, start_ij) or start_ij
    if grid[goal_ij]:
        goal_ij  = ctrl._nearest_free_cell(grid, goal_ij)  or goal_ij

    path_cells = ctrl._astar(grid, start_ij, goal_ij)
    path_world = [
        AStarGlobalLocalController._grid_cell_to_world(ij, world_size, H, W)
        for ij in path_cells
    ]

    if smooth and len(path_world) > 2:
        path_world = ctrl._smooth_path_los(path_world, env)

    return grid, path_world


# ---------------------------------------------------------------------------
# Shared wall drawing helper
# ---------------------------------------------------------------------------

def _draw_walls(ax, walls, color="#111111", lw=2.0, zorder=3):
    for w in walls:
        ax.plot([w.x1, w.x2], [w.y1, w.y2],
                color=color, lw=lw, solid_capstyle="round", zorder=zorder)


# ---------------------------------------------------------------------------
# Left plot: harmonic potential field — streamline visualization
# ---------------------------------------------------------------------------

def plot_harmonic_field(ax, walls, start, goal,
                        world_size=WORLD_SIZE, grid_size=GRID_SIZE,
                        stream_density=1.8):
    """
    Visualize the harmonic potential field as streamlines of -grad(phi).

    Each line is a path an agent would actually follow from that starting
    point. Lines are colored by gradient magnitude: bright where the field
    pushes strongly (near goal or tight passages), dark in flat regions.
    A faint phi heatmap in the background gives spatial context.
    """
    phi, grad, obs_mask, extent = compute_harmonic_field(
        walls, goal, world_size, grid_size)

    H, W = phi.shape

    # Navigation velocity field: -grad(phi), zeroed inside obstacles.
    # streamplot expects U[row, col] and V[row, col] where row 0 = y_min.
    nav_x = -grad[:, :, 0].copy()   # x-component of -grad phi  [H, W]
    nav_y = -grad[:, :, 1].copy()   # y-component of -grad phi  [H, W]
    nav_x[obs_mask] = 0.0
    nav_y[obs_mask] = 0.0

    # Gradient magnitude used to color each streamline segment
    speed = np.sqrt(nav_x**2 + nav_y**2)

    # 1-D coordinate arrays that streamplot requires
    xs = np.linspace(-world_size, world_size, W)
    ys = np.linspace(-world_size, world_size, H)

    # --- solid gray fill for obstacle cells ---
    obs_rgba = np.zeros((H, W, 4), dtype=np.float32)
    obs_rgba[obs_mask] = [0.25, 0.25, 0.25, 1.0]
    ax.imshow(obs_rgba, origin="lower", extent=extent,
              aspect="equal", zorder=1)

    # --- streamlines colored by gradient magnitude ---
    strm = ax.streamplot(
        xs, ys,
        nav_x, nav_y,
        color=speed,
        cmap="plasma",
        linewidth=1.2,
        density=stream_density,
        arrowsize=1.0,
        zorder=2,
    )

    _draw_walls(ax, walls, color="#111111", lw=2.0, zorder=3)

    ax.plot(*start, "o", color="#00aac8", ms=10, mew=1.5, mec="#005a6e",
            label="Start", zorder=5)
    ax.plot(*goal,  "*", color="#ff6f00", ms=14, mew=1.5, mec="#8b3a00",
            label="Goal",  zorder=5)

    cbar = plt.colorbar(strm.lines, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Gradient magnitude  ||∇φ||", fontsize=9)

    ax.set_facecolor("white")
    ax.set_xlim(-world_size, world_size)
    ax.set_ylim(-world_size, world_size)
    ax.set_title("Harmonic Potential Field — Streamlines", fontsize=12, fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper left", fontsize=8)


# ---------------------------------------------------------------------------
# Right plot: A* path on occupancy grid
# ---------------------------------------------------------------------------

def plot_astar_path(ax, walls, start, goal,
                    world_size=WORLD_SIZE, smooth=True):
    """Occupancy grid used by A* with the planned path drawn over it."""
    grid, path_world = compute_astar_path(walls, start, goal, world_size, smooth)

    ax.imshow(
        grid.astype(float),
        origin="lower",
        extent=[-world_size, world_size, -world_size, world_size],
        cmap="Greys",
        vmin=0.0, vmax=1.5,
        aspect="equal",
        zorder=0,
    )

    _draw_walls(ax, walls, zorder=2)

    if path_world:
        px = [p[0] for p in path_world]
        py = [p[1] for p in path_world]
        ax.plot(px, py, "-", color="#00c853", lw=2.2,
                label="A* path", zorder=3)
        if len(px) > 2:
            ax.plot(px[1:-1], py[1:-1], "o", color="#00c853", ms=3, zorder=4)
    else:
        ax.text(0, 0, "No path found", ha="center", va="center",
                fontsize=11, color="red", zorder=5)

    ax.plot(*start, "o", color="#00e5ff", ms=10, mew=2, mec="white",
            label="Start", zorder=5)
    ax.plot(*goal,  "*", color="#ff6f00", ms=14, mew=1.5, mec="white",
            label="Goal",  zorder=5)

    ax.set_xlim(-world_size, world_size)
    ax.set_ylim(-world_size, world_size)
    title = "A* Path (smoothed)" if smooth else "A* Path (raw)"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper left", fontsize=8)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--category", default="walls_and_obstacles",
                   choices=["obstacles_only", "walls_only", "walls_and_obstacles"],
                   help="Map category (default: walls_and_obstacles)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for map generation (default: 42)")
    p.add_argument("--n_obstacles", type=int, default=8,
                   help="Number of dot obstacles (default: 8)")
    p.add_argument("--wall_count", type=int, default=12,
                   help="Number of random walls (default: 12)")
    p.add_argument("--wall_len_min", type=float, default=1.5,
                   help="Minimum wall length in world units (default: 1.5)")
    p.add_argument("--wall_len_max", type=float, default=4.0,
                   help="Maximum wall length in world units (default: 4.0)")
    p.add_argument("--start", type=float, nargs=2, default=[-7.0, -7.0],
                   metavar=("X", "Y"), help="Agent start position (default: -7 -7)")
    p.add_argument("--goal",  type=float, nargs=2, default=[ 7.0,  7.0],
                   metavar=("X", "Y"), help="Goal position (default: 7 7)")
    p.add_argument("--grid_size", type=int, default=GRID_SIZE,
                   help=f"Harmonic field grid resolution (default: {GRID_SIZE})")
    p.add_argument("--stream_density", type=float, default=1.8,
                   help="Streamline density for harmonic plot (default: 1.8)")
    p.add_argument("--no_smooth", action="store_true",
                   help="Skip line-of-sight smoothing on the A* path")
    p.add_argument("--save", type=str, default=None, metavar="FILE",
                   help="Save figure to FILE instead of displaying it")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Generating map: category={args.category}, seed={args.seed}")
    map_spec = build_map(
        category=args.category,
        seed=args.seed,
        n_obstacles=args.n_obstacles,
        wall_count=args.wall_count,
        wall_len_min=args.wall_len_min,
        wall_len_max=args.wall_len_max,
    )
    walls = map_spec.all_wall_segments()  # List[Segment] — same x1/y1/x2/y2 interface
    start = tuple(args.start)
    goal  = tuple(args.goal)
    print(f"start={start}  goal={goal}  |  {len(walls)} wall segments")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.suptitle(
        f"Navigation Function Comparison  —  {args.category}",
        fontsize=13, fontweight="bold",
    )

    print("Solving harmonic field ...", end=" ", flush=True)
    plot_harmonic_field(
        axes[0], walls, start, goal,
        world_size=WORLD_SIZE,
        grid_size=args.grid_size,
        stream_density=args.stream_density,
    )
    print("done.")

    print("Running A* ...", end=" ", flush=True)
    plot_astar_path(
        axes[1], walls, start, goal,
        world_size=WORLD_SIZE,
        smooth=not args.no_smooth,
    )
    print("done.")

    # rect=[left, bottom, right, top] — top=0.94 leaves room for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
