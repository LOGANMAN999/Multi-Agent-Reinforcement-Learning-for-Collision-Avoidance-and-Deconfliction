import sys
import csv
import time
from pathlib import Path
from datetime import datetime

# Add src/ so watch_rl_episode and all its deps are importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless — no display window during batch runs

from src.watch_rl_episode import (
    build_test_env,
    prune_harmonic_colliders,
    run_episode,
    load_gat_checkpoint,
    GATDeconflictionController,
)
from src.RL_stack.gat_graph_builder import build_graph
from src.RL_stack.priority_protocol import HarmonicPriorityManager, compute_connected_components
from src.controllers.gat_deconfliction_controller import compute_repulsion_params, CONTROLLER_CONFIG
from src.controllers.harmonic_navigation import HarmonicNavigationController


# ---------------------------------------------------------------------------
# Config — mirrors the defaults in watch_rl_episode.py
# ---------------------------------------------------------------------------
N_EPISODES      = 100
N_AGENTS        = 45
MAX_STEPS       = 450
CHECKPOINT      = "checkpoints/sheaf_deconfliction_v2/policy_ep669.pt"
HIDDEN_DIM      = 128
N_HEADS         = 4
PRUNE_COLLIDERS = True   # run harmonic flow scan and remove wall-colliding agents

TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR  = Path(f"runs/benchmark_{TIMESTAMP}")
SHEAF_DIR   = OUTPUT_DIR / "sheaf"
RANDOM_DIR  = OUTPUT_DIR / "random"

CSV_COLUMNS = [
    "episode", "seed",
    "n_agents_requested",
    "n_agents_pruned_sheaf",
    "n_agents_pruned_random",
    # Sheaf outcomes
    "sheaf_goals_reached", "sheaf_collisions", "sheaf_steps",
    "sheaf_goal_rate", "sheaf_collision_rate",
    # Random-priority outcomes
    "random_goals_reached", "random_collisions", "random_steps",
    "random_goal_rate", "random_collision_rate",
]


# ---------------------------------------------------------------------------
# Random-priority controller
# ---------------------------------------------------------------------------

class RandomPriorityController:

    def __init__(self, device: torch.device, interaction_radius: float = 0.5):
        self.device              = device
        self.interaction_radius  = interaction_radius
        self.harmonic_controller = HarmonicNavigationController()
        self.priority_manager    = HarmonicPriorityManager()
        self.prev_dist_to_goal   = None
        self.time_since_progress = None
        self.prev_velocities     = None

    def reset(self, env, n_agents: int):
        self.priority_manager.reset()
        if hasattr(self.harmonic_controller, "reset"):
            self.harmonic_controller.reset(env)
        positions = np.array(env.positions, dtype=np.float32)
        goals     = np.array(env.goals,     dtype=np.float32)
        self.prev_dist_to_goal   = np.linalg.norm(positions - goals, axis=1)
        self.time_since_progress = np.zeros(n_agents, dtype=np.int32)
        self.prev_velocities     = np.zeros((n_agents, 2), dtype=np.float32)

    def act(
        self,
        env,
        active_mask: Optional[np.ndarray] = None,
        at_goal: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        N = env.n_agents
        if active_mask is None:
            active_mask = np.ones(N, dtype=bool)
        if at_goal is None:
            at_goal = np.zeros(N, dtype=bool)

        positions = np.array(env.positions, dtype=np.float32)
        goals     = np.array(env.goals,     dtype=np.float32)

        # 1. Harmonic base velocities
        harmonic_vels = self.harmonic_controller(env)
        speeds   = np.linalg.norm(harmonic_vels, axis=1)
        headings = np.where(
            speeds > 1e-6,
            np.arctan2(harmonic_vels[:, 1], harmonic_vels[:, 0]),
            0.0,
        )
        curr_dist = np.linalg.norm(positions - goals, axis=1)
        improved  = curr_dist < self.prev_dist_to_goal - 1e-6
        self.time_since_progress = np.where(improved, 0, self.time_since_progress + 1)
        self.prev_dist_to_goal   = curr_dist.copy()

        # 2. Build communication graph
        graph_active_mask = active_mask & ~at_goal
        graph = build_graph(
            env=env,
            harmonic_velocities=harmonic_vels,
            agent_velocities=self.prev_velocities,
            agent_headings=headings,
            time_since_progress=self.time_since_progress,
            active_mask=graph_active_mask,
        )
        edge_index_np = (
            graph.edge_index.cpu().numpy()
            if graph.edge_index.numel() > 0
            else np.zeros((2, 0), dtype=np.int64)
        )

        # 3. Connected components
        _, component_map = compute_connected_components(
            edge_index_np, N, at_goal, active_mask
        )
        component_sizes = np.zeros(N, dtype=np.int64)
        if component_map:
            for members in component_map.values():
                sz = len(members)
                for idx in members:
                    component_sizes[idx] = sz

        # 4. Random priority scores — replaces the policy forward pass
        priority_scores = np.random.randn(N).astype(np.float32)

        # 5. Priority ranks
        ranks, component_sizes_arr, protocol_info = self.priority_manager.step(
            priority_scores=priority_scores,
            edge_index_np=edge_index_np,
            N=N,
            active_mask=active_mask,
            at_goal=at_goal,
        )

        # 6. Per-agent repulsion parameters from rank
        repulsion_radii, repulsion_strengths = compute_repulsion_params(
            ranks=ranks,
            component_sizes=component_sizes_arr,
            n_agents=N,
            base_radius=CONTROLLER_CONFIG["base_repulsion_radius"],
            base_strength=CONTROLLER_CONFIG["base_repulsion_strength"],
            at_goal=at_goal,
        )

        # 7. Final velocities with priority-modulated repulsion
        final_vels = np.asarray(
            self.harmonic_controller(
                env,
                repulsion_radii=repulsion_radii,
                repulsion_strengths=repulsion_strengths,
            ),
            dtype=np.float32,
        )
        final_vels[~active_mask] = 0.0
        self.prev_velocities = final_vels.copy()

        info = {
            "astar_velocities": harmonic_vels,
            "priority_scores":  priority_scores,
            "yield_mask":       ranks > 0,
            "edge_index":       edge_index_np,
            "protocol_info":    protocol_info,
            "log_probs":        np.zeros(N),
            "values":           np.zeros(N),
            "astar_mode":       np.zeros(N, dtype=bool),
            "frozen_mask":      np.zeros(N, dtype=bool),
        }
        return final_vels, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_episode_outcome(npz_path: Path) -> dict:

    data = np.load(str(npz_path))
    n_agents     = int(data["active"].shape[1])
    goals        = int(data["goals_reached"][-1])
    n_active     = int(data["active"][-1].sum())
    n_collided   = n_agents - n_active
    steps        = int(data["timesteps"][-1])
    return {
        "goals_reached":    goals,
        "n_collided":       n_collided,
        "n_agents":         n_agents,
        "steps":            steps,
        "goal_rate":        round(goals  / max(n_agents, 1), 4),
        "collision_rate":   round(n_collided / max(n_agents, 1), 4),
    }


def print_summary(results: list, output_path: Path) -> None:
    def mean(key):
        return float(np.mean([r[key] for r in results]))

    lines = [
        "=" * 60,
        f"BENCHMARK RESULTS  ({len(results)} episodes, N_AGENTS={N_AGENTS})",
        f"Checkpoint: {CHECKPOINT}",
        f"Pruning: {PRUNE_COLLIDERS}",
        "=" * 60,
        f"  {'Metric':<30}  {'Sheaf':>8}  {'Random':>10}",
        f"  {'-'*30}  {'-'*8}  {'-'*10}",
        f"  {'Goal reach rate':<30}  {mean('sheaf_goal_rate'):>8.3f}  {mean('random_goal_rate'):>10.3f}",
        f"  {'Collision rate':<30}  {mean('sheaf_collision_rate'):>8.3f}  {mean('random_collision_rate'):>10.3f}",
        f"  {'Avg goals reached':<30}  {mean('sheaf_goals_reached'):>8.2f}  {mean('random_goals_reached'):>10.2f}",
        f"  {'Avg collisions':<30}  {mean('sheaf_collisions'):>8.2f}  {mean('random_collisions'):>10.2f}",
        f"  {'Avg steps':<30}  {mean('sheaf_steps'):>8.1f}  {mean('random_steps'):>10.1f}",
        "=" * 60,
    ]
    text = "\n".join(lines)
    print("\n" + text)
    output_path.write_text(text + "\n")
    print(f"\n  Summary written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    SHEAF_DIR.mkdir(parents=True, exist_ok=True)
    RANDOM_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device:      {device}")
    print(f"Output directory:  {OUTPUT_DIR}")
    print(f"Episodes:          {N_EPISODES}  (seeds 0 – {N_EPISODES - 1})")
    print(f"Agents:            {N_AGENTS}  (before pruning)")
    print(f"Max steps:         {MAX_STEPS}")
    print(f"Checkpoint:        {CHECKPOINT}")
    print(f"Pruning:           {PRUNE_COLLIDERS}")
    print()

    # Load sheaf policy once and reuse across episodes
    policy = load_gat_checkpoint(
        CHECKPOINT,
        device=device,
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
    )

    results = []
    t_start = time.time()

    for ep in range(N_EPISODES):
        seed = ep
        ep_t0 = time.time()

        # ----------------------------------------------------------------
        # Build two identical envs from the same seed (one per arm).
        # np.random.seed is reset before each call so both envs get the
        # same agent positions.
        # ----------------------------------------------------------------
        np.random.seed(seed)
        env_sheaf  = build_test_env(seed=seed, n_agents=N_AGENTS)
        np.random.seed(seed)
        env_random = build_test_env(seed=seed, n_agents=N_AGENTS)
        n_before = env_sheaf.n_agents  # same for both before pruning

        if PRUNE_COLLIDERS:
            prune_harmonic_colliders(env_sheaf)
            prune_harmonic_colliders(env_random)

        n_sheaf  = env_sheaf.n_agents
        n_random = env_random.n_agents

        # ----------------------------------------------------------------
        # Sheaf run — record to disk, no display
        # ----------------------------------------------------------------
        sheaf_path = str(SHEAF_DIR / f"ep{ep:04d}_s{seed}")
        controller_sheaf = GATDeconflictionController(
            policy=policy,
            device=device,
            deterministic=True,
            enable_astar_fallback=True,
        )
        run_episode(
            env_sheaf, controller_sheaf,
            max_steps=MAX_STEPS,
            pause=0,
            record_path=sheaf_path,
            show_graph=False,
        )

        # ----------------------------------------------------------------
        # Random-priority run — record to disk, no display
        # ----------------------------------------------------------------
        np.random.seed(seed)   # reset so random scores aren't seeded by sheaf run
        random_path = str(RANDOM_DIR / f"ep{ep:04d}_s{seed}")
        controller_random = RandomPriorityController(device=device)
        run_episode(
            env_random, controller_random,
            max_steps=MAX_STEPS,
            pause=0,
            record_path=random_path,
            show_graph=False,
        )

        # ----------------------------------------------------------------
        # Read outcomes from saved NPZs
        # ----------------------------------------------------------------
        sheaf_stats  = read_episode_outcome(Path(sheaf_path  + ".npz"))
        random_stats = read_episode_outcome(Path(random_path + ".npz"))

        row = {
            "episode":                   ep,
            "seed":                      seed,
            "n_agents_requested":        n_before,
            "n_agents_pruned_sheaf":     n_sheaf,
            "n_agents_pruned_random":    n_random,
            # Sheaf
            "sheaf_goals_reached":       sheaf_stats["goals_reached"],
            "sheaf_collisions":          sheaf_stats["n_collided"],
            "sheaf_steps":               sheaf_stats["steps"],
            "sheaf_goal_rate":           sheaf_stats["goal_rate"],
            "sheaf_collision_rate":      sheaf_stats["collision_rate"],
            # Random
            "random_goals_reached":      random_stats["goals_reached"],
            "random_collisions":         random_stats["n_collided"],
            "random_steps":              random_stats["steps"],
            "random_goal_rate":          random_stats["goal_rate"],
            "random_collision_rate":     random_stats["collision_rate"],
        }
        results.append(row)

        ep_time = time.time() - ep_t0
        print(
            f"[{ep+1:3d}/{N_EPISODES}] seed={seed:3d}  "
            f"Sheaf:  goals={sheaf_stats['goals_reached']:2d}/{n_sheaf}  "
            f"col={sheaf_stats['n_collided']}  "
            f"steps={sheaf_stats['steps']:3d}  |  "
            f"Random: goals={random_stats['goals_reached']:2d}/{n_random}  "
            f"col={random_stats['n_collided']}  "
            f"steps={random_stats['steps']:3d}  "
            f"({ep_time:.1f}s)"
        )

    # ----------------------------------------------------------------
    # Save CSV
    # ----------------------------------------------------------------
    csv_path = OUTPUT_DIR / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults CSV saved to {csv_path}")

    total_time = time.time() - t_start
    print(f"Total wall time: {total_time/60:.1f} min")

    # ----------------------------------------------------------------
    # Aggregate summary
    # ----------------------------------------------------------------
    print_summary(results, OUTPUT_DIR / "summary.txt")


if __name__ == "__main__":
    main()
