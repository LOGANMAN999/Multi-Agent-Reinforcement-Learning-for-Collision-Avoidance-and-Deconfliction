from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_world_size(dataset_root: Path, override: float | None) -> float:
    if override is not None:
        return float(override)

    # Try the run-specific meta first, then generic meta.json
    for name in ["meta_walls+obstacles.json", "meta.json"]:
        p = dataset_root / name
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            if "world_size" in d:
                return float(d["world_size"])

    raise ValueError(
        "Could not infer world_size from meta_walls+obstacles.json or meta.json. "
        "Pass --world_size explicitly."
    )


def point_to_segment_dist2(points_xy: np.ndarray, seg_xyxy: np.ndarray) -> np.ndarray:
    """
    points_xy: (P,2)
    seg_xyxy: (4,) = [x1,y1,x2,y2]
    Returns dist^2: (P,)
    """
    x1, y1, x2, y2 = seg_xyxy.astype(np.float64)
    ax, ay = x1, y1
    bx, by = x2, y2

    abx = bx - ax
    aby = by - ay
    apx = points_xy[:, 0] - ax
    apy = points_xy[:, 1] - ay

    denom = abx * abx + aby * aby
    # handle degenerate segments (shouldn't happen, but safe)
    if denom <= 1e-14:
        return apx * apx + apy * apy

    t = (apx * abx + apy * aby) / denom
    t = np.clip(t, 0.0, 1.0)

    cx = ax + t * abx
    cy = ay + t * aby

    dx = points_xy[:, 0] - cx
    dy = points_xy[:, 1] - cy
    return dx * dx + dy * dy


def compute_min_distance_field(
    walls_xyxy: np.ndarray, world_size: float, grid_n: int
) -> np.ndarray:
    """
    Returns min distance d(x) at grid points as a flat array of length P = grid_n^2.
    """
    w = float(world_size)
    xs = np.linspace(-w, w, grid_n, dtype=np.float64)
    ys = np.linspace(-w, w, grid_n, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)  # (P,2)

    min_d2 = np.full((pts.shape[0],), np.inf, dtype=np.float64)
    for seg in walls_xyxy:
        d2 = point_to_segment_dist2(pts, seg)
        min_d2 = np.minimum(min_d2, d2)

    return np.sqrt(min_d2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_root",
        type=str,
        default="datasets/il_dataset",
        help="Dataset root (contains episodes_walls+obstacles/ and meta*.json)",
    )
    ap.add_argument(
        "--episode",
        type=str,
        default=None,
        help="Path to a specific episode npz. If omitted, uses --episode_idx or the first file found.",
    )
    ap.add_argument(
        "--episode_idx",
        type=int,
        default="000015",
        help="Episode index to load (episode_XXXXXX.npz). Ignored if --episode is given.",
    )
    ap.add_argument(
        "--world_size",
        type=float,
        default=None,
        help="Override world_size (default: inferred from meta json).",
    )
    ap.add_argument(
        "--grid_n",
        type=int,
        default=400,
        help="Grid resolution per axis (area approx improves with higher values).",
    )
    ap.add_argument(
        "--eps_max",
        type=float,
        default=5,
        help="Max epsilon to plot. Default: min(5.0, max distance on grid).",
    )
    ap.add_argument(
        "--n_eps",
        type=int,
        default=200,
        help="Number of epsilon samples on [0, eps_max].",
    )
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    episodes_dir = dataset_root / "episodes_walls+obstacles"

    world_size = load_world_size(dataset_root, args.world_size)

    # Pick episode file
    if args.episode is not None:
        ep_path = Path(args.episode)
    elif args.episode_idx is not None:
        ep_path = episodes_dir / f"episode_{args.episode_idx:06d}.npz"
    else:
        candidates = sorted(episodes_dir.glob("episode_*.npz"))
        if not candidates:
            raise FileNotFoundError(f"No episode_*.npz found in {episodes_dir}")
        ep_path = candidates[0]

    if not ep_path.exists():
        raise FileNotFoundError(f"Episode file not found: {ep_path}")

    ep = np.load(ep_path, allow_pickle=True)
    if "walls_xyxy" not in ep.files:
        raise KeyError(f"{ep_path} does not contain 'walls_xyxy'. Found keys: {ep.files}")

    walls_xyxy = np.asarray(ep["walls_xyxy"], dtype=np.float64)
    if walls_xyxy.ndim != 2 or walls_xyxy.shape[1] != 4:
        raise ValueError(f"walls_xyxy should have shape (M,4). Got {walls_xyxy.shape}")

    print(f"Episode: {ep_path}")
    print(f"world_size={world_size}, grid_n={args.grid_n}, walls={walls_xyxy.shape[0]} segments")

    # Compute distance-to-nearest-segment at grid points
    d = compute_min_distance_field(walls_xyxy, world_size, args.grid_n)

    # Choose eps_max
    d_max = float(np.max(d))
    eps_max = float(args.eps_max) if args.eps_max is not None else min(5.0, d_max)
    eps = np.linspace(0.0, eps_max, args.n_eps, dtype=np.float64)

    # G(eps) = area within distance < eps of the walls
    area_total = (2.0 * world_size) ** 2
    # Sort distances so we can compute counts quickly via searchsorted
    d_sorted = np.sort(d)
    P = d_sorted.size
    counts = np.searchsorted(d_sorted, eps, side="left")  # # of points with d < eps
    G = area_total * (counts / P)

    # Plot
    F = area_total - G
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(eps, F)
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"$G(\varepsilon)$  (area with distance $< \varepsilon$)")
    plt.title("Clearance-volume profile from walls_xyxy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
