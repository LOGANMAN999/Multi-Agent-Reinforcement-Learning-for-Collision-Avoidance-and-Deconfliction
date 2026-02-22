# map_generation.py
from __future__ import annotations
import matplotlib.pyplot as plt

from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional
import json
import math
import numpy as np


# ----------------------------
# Geometry primitives
# ----------------------------

@dataclass(frozen=True)
class Segment:
    """Wall segment in world coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class BoxObstacle:
    """Axis-aligned square/rectangle obstacle (we'll mainly use square)."""
    cx: float
    cy: float
    w: float
    h: float

    def corners(self) -> Tuple[float, float, float, float]:
        xmin = self.cx - self.w / 2.0
        xmax = self.cx + self.w / 2.0
        ymin = self.cy - self.h / 2.0
        ymax = self.cy + self.h / 2.0
        return xmin, ymin, xmax, ymax

    def to_segments(self) -> List[Segment]:
        xmin, ymin, xmax, ymax = self.corners()
        return [
            Segment(xmin, ymin, xmax, ymin),
            Segment(xmax, ymin, xmax, ymax),
            Segment(xmax, ymax, xmin, ymax),
            Segment(xmin, ymax, xmin, ymin),
        ]


# ----------------------------
# Map specification + serialization
# ----------------------------

@dataclass
class MapSpec:
    """
    A reusable map container.

    category:
        - "obstacles_only"
        - "walls_only"
        - "walls_and_obstacles"
    """
    map_id: str
    category: str
    world_size: float

    # Keep obstacles as boxes (for bookkeeping/metrics) AND as wall segments (for env/controller use)
    obstacles: List[BoxObstacle]
    n_obstacles: int

    # Additional "free" walls (not including the obstacle box perimeters)
    walls: List[Segment]

    # Optional metadata for later (difficulty metrics can be written here)
    meta: Dict[str, Any]

    def all_wall_segments(self) -> List[Segment]:
        segs: List[Segment] = []
        for ob in self.obstacles:
            segs.extend(ob.to_segments())
        segs.extend(self.walls)
        return segs

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # dataclasses -> plain dicts
        d["obstacles"] = [asdict(o) for o in self.obstacles]
        d["walls"] = [asdict(s) for s in self.walls]
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MapSpec":
        obs = [BoxObstacle(**o) for o in d["obstacles"]]
        walls = [Segment(**s) for s in d["walls"]]
        return MapSpec(
            map_id=d["map_id"],
            category=d["category"],
            world_size=float(d["world_size"]),
            obstacles=obs,
            n_obstacles=int(d["n_obstacles"]),
            walls=walls,
            meta=dict(d.get("meta", {})),
        )


@dataclass
class MapSet:
    maps: List[MapSpec]

    def to_json(self, path: str) -> None:
        payload = {"maps": [m.to_dict() for m in self.maps]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def from_json(path: str) -> "MapSet":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        maps = [MapSpec.from_dict(m) for m in payload["maps"]]
        return MapSet(maps=maps)


# ----------------------------
# Sampling helpers
# ----------------------------

def _rand_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))



def _sample_dot_obstacles_as_segments(
    rng: np.random.Generator,
    world_size: float,
    num_obstacles: int,
    half_len: float = 0.2,
    margin_frac: float = 0.8,
) -> List[Segment]:
    """
    Generate 'dot-like' obstacles as tiny horizontal/vertical segments.
    Similar to your generate_random_obstacles() sketch.
    """
    segs: List[Segment] = []
    lo = -world_size * margin_frac
    hi =  world_size * margin_frac

    for _ in range(num_obstacles):
        x = float(rng.uniform(lo, hi))
        y = float(rng.uniform(lo, hi))

        if float(rng.random()) < 0.5:
            # short vertical
            segs.append(Segment(x, y - half_len, x, y + half_len))
        else:
            # short horizontal
            segs.append(Segment(x - half_len, y, x + half_len, y))

    return segs



def _intervals_overlap(a1: float, a2: float, b1: float, b2: float, pad: float = 0.0) -> bool:
    """True if [a1,a2] overlaps [b1,b2] with optional padding."""
    lo1, hi1 = (a1, a2) if a1 <= a2 else (a2, a1)
    lo2, hi2 = (b1, b2) if b1 <= b2 else (b2, b1)
    return (lo1 <= hi2 + pad) and (lo2 <= hi1 + pad)

def _sample_axis_aligned_walls(
    rng: np.random.Generator,
    world_size: float,
    wall_count: int,
    min_wall_len: float,
    max_wall_len: float,
    margin: float,
    min_parallel_sep: float = 0.8,   # <-- NEW: minimum spacing between parallel walls
    overlap_pad: float = 0.0,        # <-- NEW: treat "near overlap" as overlap
    max_tries: int = 20_000          # <-- NEW: avoid infinite loops
) -> List[Segment]:
    """
    Random axis-aligned internal walls (not boundary walls), with a minimum distance
    enforced between *parallel* walls that overlap along their length.
    """
    walls: List[Segment] = []
    lo = -world_size + margin
    hi =  world_size - margin

    tries = 0
    while len(walls) < wall_count and tries < max_tries:
        tries += 1

        horizontal = bool(rng.integers(0, 2))
        L = float(rng.uniform(min_wall_len, max_wall_len))

        if horizontal:
            y = float(rng.uniform(lo, hi))
            x_center = float(rng.uniform(lo, hi))
            x1 = max(lo, x_center - L / 2.0)
            x2 = min(hi, x_center + L / 2.0)
            cand = Segment(x1, y, x2, y)

            # Enforce min spacing vs existing horizontal walls with overlapping x-ranges
            ok = True
            for w in walls:
                if abs(w.y1 - w.y2) < 1e-9:  # w is horizontal
                    if _intervals_overlap(cand.x1, cand.x2, w.x1, w.x2, pad=overlap_pad):
                        if abs(cand.y1 - w.y1) < min_parallel_sep:
                            ok = False
                            break
            if not ok:
                continue

            walls.append(cand)

        else:
            x = float(rng.uniform(lo, hi))
            y_center = float(rng.uniform(lo, hi))
            y1 = max(lo, y_center - L / 2.0)
            y2 = min(hi, y_center + L / 2.0)
            cand = Segment(x, y1, x, y2)

            # Enforce min spacing vs existing vertical walls with overlapping y-ranges
            ok = True
            for w in walls:
                if abs(w.x1 - w.x2) < 1e-9:  # w is vertical
                    if _intervals_overlap(cand.y1, cand.y2, w.y1, w.y2, pad=overlap_pad):
                        if abs(cand.x1 - w.x1) < min_parallel_sep:
                            ok = False
                            break
            if not ok:
                continue

            walls.append(cand)

    # If we couldn’t place enough walls, you can either:
    #  - accept fewer walls (current behavior), or
    #  - relax min_parallel_sep and retry outside this function.
    return walls



# ----------------------------
# Public API
# ----------------------------

def generate_map(
    rng: np.random.Generator,
    category: str,
    map_id: str,
    world_size: float,
    n_obstacles: int,
    obstacle_size: float,
    wall_count: int,
    wall_len_range: Tuple[float, float],
    margin: float,
    min_obstacle_center_sep: Optional[float] = None,
) -> MapSpec:
    """
    Generate one map. Categories:
      - "obstacles_only"
      - "walls_only"
      - "walls_and_obstacles"
    """
    if category not in ("obstacles_only", "walls_only", "walls_and_obstacles"):
        raise ValueError(f"Unknown category: {category}")

    if min_obstacle_center_sep is None:
        # default: mild separation to avoid near-total overlap
        min_obstacle_center_sep = 1.25 * obstacle_size

    obstacles = []
    walls: List[Segment] = []
    dot_segs: List[Segment] = []

    if category in ("obstacles_only", "walls_and_obstacles"):
        dot_segs = _sample_dot_obstacles_as_segments(
        rng=rng,
        world_size=world_size,
        num_obstacles=n_obstacles,
        half_len=0.2,        # <-- change size here
        margin_frac=0.8,
    )

    if category in ("walls_only", "walls_and_obstacles"):
        minL, maxL = wall_len_range
        walls = _sample_axis_aligned_walls(
            rng=rng,
            world_size=world_size,
            wall_count=wall_count,
            min_wall_len=minL,
            max_wall_len=maxL,
            margin=margin,
        )
    walls = list(walls) + list(dot_segs)

    return MapSpec(
        map_id=map_id,
        category=category,
        world_size=float(world_size),
        obstacles=obstacles,
        n_obstacles=len(obstacles),
        walls=walls,
        meta={
            "obstacle_size": float(obstacle_size),
            "wall_count": int(wall_count),
            "wall_len_range": [float(wall_len_range[0]), float(wall_len_range[1])],
            "margin": float(margin),
        },
    )



def plot_mapspec(map_spec, ax=None, title: str = ""):
    """
    Visualize a MapSpec (from map_generation.py) similarly to sim_env.py.
    Draws all segments: obstacle perimeters + explicit walls.
    """
    if ax is None:
        fig, ax = plt.subplots()

    segs = map_spec.all_wall_segments()
    for s in segs:
        ax.plot([s.x1, s.x2], [s.y1, s.y2], "k-", linewidth=2)

    w = map_spec.world_size
    ax.set_aspect("equal")
    ax.set_xlim(-w, w)
    ax.set_ylim(-w, w)
    ax.grid(True)

    if not title:
        title = f"{map_spec.map_id} ({map_spec.category})"
    ax.set_title(title)
    return ax



def generate_mapset(
    seed: int = 0,
    per_category: int = 5,
    world_size: float = 10.0,
    n_obstacles: int = 10,
    obstacle_size: float = 0.7,
    wall_count: int = 6,
    wall_len_range: Tuple[float, float] = (2.0, 8.0),
    margin: float = 0.8,
) -> MapSet:
    """
    Create a small batch of maps across all 3 categories for inspection.
    """
    rng = np.random.default_rng(seed)
    maps: List[MapSpec] = []
    categories = ["obstacles_only", "walls_only", "walls_and_obstacles"]

    for cat in categories:
        for k in range(per_category):
            mid = f"{cat}_{seed}_{k:03d}"
            m = generate_map(
                rng=rng,
                category=cat,
                map_id=mid,
                world_size=world_size,
                n_obstacles=n_obstacles,
                obstacle_size=obstacle_size,
                wall_count=wall_count,
                wall_len_range=wall_len_range,
                margin=margin,
            )
            maps.append(m)

    return MapSet(maps=maps)


def map_to_env_walls(map_spec: MapSpec) -> List[Tuple[float, float, float, float]]:
    """
    Returns wall segments as tuples, which is a safe interchange format.
    If your env expects objects with x1,y1,x2,y2, you can wrap these later.
    """
    return [s.as_tuple() for s in map_spec.all_wall_segments()]


# Example usage:
#   ms = generate_mapset(seed=1, per_category=3)
#   ms.to_json("maps_small.json")
#   ms2 = MapSet.from_json("maps_small.json")
#   walls = map_to_env_walls(ms2.maps[0])


if __name__ == "__main__":
    # Generate a small batch and visualize a handful (no simulation).
    ms = generate_mapset(
        seed=2,
        per_category=3,   # 3 maps per category => 9 total
        world_size=10.0,
        n_obstacles=4,
        obstacle_size=0.7,
        wall_count=15,
        wall_len_range=(2.0, 8.0),
        margin=1.2,
    )



    # Show up to 9 maps in a 3x3 grid
    maps_to_show = ms.maps[:9]
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for idx, m in enumerate(maps_to_show):
        r, c = divmod(idx, cols)
        plot_mapspec(m, ax=axes[r, c])

    # Turn off any unused axes
    for idx in range(len(maps_to_show), rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()