"""
tda/agent_metrics.py

Agent-configuration metrics derived from initial positions and goal locations.

These capture the traffic-flow structure of each episode independently of the
map geometry.  High-conflict configurations require good priority ordering
regardless of map bottlenecks.

Public API
----------
cross_traffic_fraction(starts, goals, interaction_radius) -> float
wasserstein_dist(starts, goals)                           -> float
path_crossing_density(starts, goals, interaction_radius)  -> float
initial_pair_density(starts, interaction_radius)          -> float
mean_path_length(starts, goals)                           -> float
agent_summary(starts, goals, interaction_radius)          -> dict
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Cross-traffic fraction
# ---------------------------------------------------------------------------

def cross_traffic_fraction(
    starts: np.ndarray,
    goals: np.ndarray,
    interaction_radius: float = 2.0,
) -> float:
    """
    Fraction of nearby agent pairs whose goal-direction vectors are opposed
    (cosine < 0, i.e. heading toward each other).

    'Nearby' = within interaction_radius of each other at t=0.

    High values -> many head-on encounters -> priority ordering critical.
    Returns 0.0 if no nearby pairs exist.
    """
    N = len(starts)
    dirs = goals - starts                                         # [N, 2]
    norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    dirs_unit = dirs / norms                                      # [N, 2]

    pair_dists = cdist(starts, starts)                            # [N, N]
    n_opposed = 0
    n_pairs   = 0
    for i in range(N):
        for j in range(i + 1, N):
            if pair_dists[i, j] < interaction_radius:
                cos_angle = dirs_unit[i] @ dirs_unit[j]
                n_pairs  += 1
                if cos_angle < 0:
                    n_opposed += 1

    return float(n_opposed / n_pairs) if n_pairs > 0 else 0.0


# ---------------------------------------------------------------------------
# Wasserstein-1 distance between start and goal distributions
# ---------------------------------------------------------------------------

def wasserstein_dist(
    starts: np.ndarray,
    goals: np.ndarray,
) -> float:
    """
    Wasserstein-1 (earth mover) distance between the start and goal position
    distributions, computed via linear assignment (exact for equal-size sets).

    High W1 -> agents need to travel far across the map -> more cross-traffic.
    Normalised by N so it's comparable across different agent counts.
    """
    cost = cdist(starts, goals, metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean())


# ---------------------------------------------------------------------------
# Path crossing density
# ---------------------------------------------------------------------------

def _min_dist_straight_paths(
    s_i: np.ndarray, g_i: np.ndarray,
    s_j: np.ndarray, g_j: np.ndarray,
) -> float:
    """
    Minimum distance between two agents travelling at the same fractional
    speed along their straight-line start->goal paths.

    Parameterisation: p(t) = s + t*(g - s), t in [0,1].
    Minimises |p_i(t) - p_j(t)|^2 over t.
    """
    a = s_i - s_j                           # relative position at t=0
    b = (g_i - s_i) - (g_j - s_j)          # relative velocity direction
    b_sq = float(b @ b)
    if b_sq < 1e-10:
        return float(np.linalg.norm(a))
    t_star = float(np.clip(-(a @ b) / b_sq, 0.0, 1.0))
    closest = a + t_star * b
    return float(np.linalg.norm(closest))


def path_crossing_density(
    starts: np.ndarray,
    goals: np.ndarray,
    interaction_radius: float = 2.0,
) -> float:
    """
    Fraction of agent pairs whose straight-line paths come within
    interaction_radius/2 of each other at the same journey-fraction t.

    This is a proxy for "intended collision density" before the harmonic
    field routes agents around each other.

    High values -> more conflict events predicted -> sheaf priority matters more.
    """
    N = len(starts)
    if N < 2:
        return 0.0
    threshold = interaction_radius / 2.0
    n_crossing = 0
    n_total    = 0
    for i in range(N):
        for j in range(i + 1, N):
            d = _min_dist_straight_paths(starts[i], goals[i], starts[j], goals[j])
            n_total += 1
            if d < threshold:
                n_crossing += 1
    return float(n_crossing / n_total)


# ---------------------------------------------------------------------------
# Initial pair density (graph density at t=0)
# ---------------------------------------------------------------------------

def initial_pair_density(
    starts: np.ndarray,
    interaction_radius: float = 2.0,
) -> float:
    """
    Fraction of agent pairs within interaction_radius at t=0.

    Equivalent to the edge density of the communication graph at episode start.
    High density -> priority protocol active immediately for most agents.
    """
    N = len(starts)
    if N < 2:
        return 0.0
    dists = cdist(starts, starts)
    n_pairs  = N * (N - 1) // 2
    n_nearby = int((dists < interaction_radius).sum() - N) // 2
    return float(n_nearby / n_pairs)


# ---------------------------------------------------------------------------
# Mean straight-line path length
# ---------------------------------------------------------------------------

def mean_path_length(starts: np.ndarray, goals: np.ndarray) -> float:
    """Mean Euclidean distance from each agent's start to its goal."""
    return float(np.linalg.norm(goals - starts, axis=1).mean())


# ---------------------------------------------------------------------------
# Convenience: compute all agent metrics at once
# ---------------------------------------------------------------------------

def agent_summary(
    starts: np.ndarray,
    goals: np.ndarray,
    interaction_radius: float = 2.0,
) -> dict:
    """
    Compute all agent-configuration metrics and return as a flat dict.

    Keys returned:
        cross_traffic_fraction  — fraction of nearby pairs heading toward each other
        wasserstein_dist        — mean optimal transport cost start->goal
        path_crossing_density   — fraction of pairs with crossing straight-line paths
        initial_pair_density    — graph edge density at t=0
        mean_path_length        — average start-to-goal Euclidean distance
        n_agents                — number of agents
    """
    return {
        "cross_traffic_fraction": cross_traffic_fraction(starts, goals, interaction_radius),
        "wasserstein_dist":       wasserstein_dist(starts, goals),
        "path_crossing_density":  path_crossing_density(starts, goals, interaction_radius),
        "initial_pair_density":   initial_pair_density(starts, interaction_radius),
        "mean_path_length":       mean_path_length(starts, goals),
        "n_agents":               len(starts),
    }
