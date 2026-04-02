"""
tda/map_geometry.py

Map-level geometric and topological metrics derived from the wall distance field.

All functions operate on:
    walls_xyxy : np.ndarray  shape (M, 4) — [x1, y1, x2, y2] per wall segment
    world_size : float       — map spans [-world_size, +world_size]^2

Public API
----------
compute_distance_field(walls_xyxy, world_size, grid_n) -> d [P]
compute_G_eps(d, world_size, eps)                      -> G [len(eps)]
compute_connectivity_curve(d, grid_n, eps)             -> C [len(eps)]
bottleneck_score(C, eps, r_robot)                      -> float
tightness_ratio(d, world_size, r_robot)                -> float
total_wall_length(walls_xyxy)                          -> float
map_summary(walls_xyxy, world_size, grid_n, r_robot)   -> dict
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import label as ndlabel


# ---------------------------------------------------------------------------
# Distance field
# ---------------------------------------------------------------------------

def _seg_dist2(pts: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Squared distance from each point in pts (P,2) to segment seg (4,)."""
    x1, y1, x2, y2 = seg.astype(np.float64)
    abx, aby = x2 - x1, y2 - y1
    apx = pts[:, 0] - x1
    apy = pts[:, 1] - y1
    denom = abx * abx + aby * aby
    if denom <= 1e-14:
        return apx * apx + apy * apy
    t = np.clip((apx * abx + apy * aby) / denom, 0.0, 1.0)
    cx = x1 + t * abx
    cy = y1 + t * aby
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    return dx * dx + dy * dy


def compute_distance_field(
    walls_xyxy: np.ndarray,
    world_size: float,
    grid_n: int = 200,
) -> np.ndarray:
    """
    Compute min-distance-to-any-wall at each grid point.

    Returns d of shape (grid_n*grid_n,) — row-major (xy indexing).
    Grid spans [-world_size, +world_size] along both axes.
    """
    w = float(world_size)
    xs = np.linspace(-w, w, grid_n, dtype=np.float64)
    ys = np.linspace(-w, w, grid_n, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)

    min_d2 = np.full(pts.shape[0], np.inf, dtype=np.float64)
    for seg in walls_xyxy:
        d2 = _seg_dist2(pts, seg)
        np.minimum(min_d2, d2, out=min_d2)

    return np.sqrt(min_d2)


# ---------------------------------------------------------------------------
# Clearance-volume profile G(eps)
# ---------------------------------------------------------------------------

def compute_G_eps(
    d: np.ndarray,
    world_size: float,
    eps: np.ndarray,
) -> np.ndarray:
    """
    G(eps) = area of the map with distance-to-wall < eps.
    F(eps) = total_area - G(eps) = navigable area at clearance eps.
    """
    area_total = (2.0 * world_size) ** 2
    d_sorted = np.sort(d)
    counts = np.searchsorted(d_sorted, eps, side="left")
    return area_total * (counts / d_sorted.size)


# ---------------------------------------------------------------------------
# Free-space connectivity curve C(eps)
# ---------------------------------------------------------------------------

def compute_connectivity_curve(
    d: np.ndarray,
    grid_n: int,
    eps: np.ndarray,
) -> np.ndarray:
    """
    C(eps) = number of 4-connected components of {x : d(x) > eps}.

    Captures how the navigable space fragments as we require more clearance.
    Bottlenecks appear as jumps in C(eps): a passage of width 2*eps_0 causes
    C to jump from 1 to 2 at eps = eps_0.

    Returns integer array of shape (len(eps),).
    """
    d_grid = d.reshape(grid_n, grid_n)
    struct = np.ones((3, 3), dtype=np.int32)  # 8-connectivity for label
    counts = np.empty(len(eps), dtype=np.int32)
    for k, e in enumerate(eps):
        mask = (d_grid > e).astype(np.int32)
        if mask.sum() == 0:
            counts[k] = 0
        else:
            _, n = ndlabel(mask, structure=struct)
            counts[k] = n
    return counts


# ---------------------------------------------------------------------------
# Derived scalar metrics
# ---------------------------------------------------------------------------

def bottleneck_score(
    C: np.ndarray,
    eps: np.ndarray,
    r_robot: float = 0.25,
) -> float:
    """
    Total excess connectivity above 1 for eps >= r_robot.

    score = integral of max(C(eps) - 1, 0) d_eps  for eps in [r_robot, eps_max]

    0 for a fully open map.  Higher = more / narrower bottlenecks.
    Units: world-space length (meters of clearance-scale weighted by fragmentation).
    """
    mask = eps >= r_robot
    if mask.sum() < 2:
        return 0.0
    excess = np.maximum(C[mask].astype(np.float64) - 1.0, 0.0)
    return float(np.trapz(excess, eps[mask]))


def tightness_ratio(
    d: np.ndarray,
    world_size: float,
    r_robot: float = 0.25,
) -> float:
    """
    Fraction of navigable space (d > r_robot) in the "marginal clearance" band
    r_robot < d < 2*r_robot.

    High tightness_ratio -> agents frequently operate near walls -> priority
    ordering matters more (less room to manoeuvre around each other).
    """
    navigable = d > r_robot
    marginal  = (d > r_robot) & (d < 2 * r_robot)
    n_nav = navigable.sum()
    if n_nav == 0:
        return 0.0
    return float(marginal.sum() / n_nav)


def total_wall_length(walls_xyxy: np.ndarray) -> float:
    """Sum of Euclidean lengths of all wall segments."""
    dx = walls_xyxy[:, 2] - walls_xyxy[:, 0]
    dy = walls_xyxy[:, 3] - walls_xyxy[:, 1]
    return float(np.sqrt(dx * dx + dy * dy).sum())


# ---------------------------------------------------------------------------
# Convenience: compute all map metrics at once
# ---------------------------------------------------------------------------

def map_summary(
    walls_xyxy: np.ndarray,
    world_size: float,
    grid_n: int = 200,
    r_robot: float = 0.25,
    n_eps: int = 40,
    eps_max: float = 3.0,
) -> dict:
    """
    Compute all map-level metrics and return as a flat dict.

    Keys returned:
        wall_length          — total wall segment length
        free_area_fraction   — fraction of map with d > r_robot
        tightness_ratio      — marginal-clearance fraction of navigable area
        bottleneck_score     — integral of excess connectivity C(eps)-1
        peak_g_deriv_eps     — eps at which dG/deps is largest (characteristic scale)
        n_components_at_r    — C(r_robot): components when requiring robot clearance
    """
    d     = compute_distance_field(walls_xyxy, world_size, grid_n)
    eps   = np.linspace(0.0, eps_max, n_eps)
    G     = compute_G_eps(d, world_size, eps)
    C     = compute_connectivity_curve(d, grid_n, eps)

    area_total = (2.0 * world_size) ** 2
    free_area  = float(((area_total - G[-1]) + (area_total - G[0])) / 2)  # rough
    free_frac  = float((d > r_robot).sum() / d.size)

    # dG/deps — peak indicates the dominant constriction scale
    dG = np.gradient(G, eps)
    peak_idx = int(np.argmax(dG[eps >= 0.1]))  # skip eps near 0
    peak_eps = float(eps[eps >= 0.1][peak_idx])

    # C at eps = r_robot
    r_idx = int(np.searchsorted(eps, r_robot))
    n_comp_at_r = int(C[min(r_idx, len(C) - 1)])

    return {
        "wall_length":         total_wall_length(walls_xyxy),
        "free_area_fraction":  free_frac,
        "tightness_ratio":     tightness_ratio(d, world_size, r_robot),
        "bottleneck_score":    bottleneck_score(C, eps, r_robot),
        "peak_g_deriv_eps":    peak_eps,
        "n_components_at_r":   n_comp_at_r,
        # store curves for later use
        "_eps": eps,
        "_G":   G,
        "_C":   C,
    }
