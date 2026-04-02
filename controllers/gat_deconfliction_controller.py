import numpy as np
import torch
from typing import Dict, Optional, Tuple

from RL_stack.gat_deconfliction_policy import GATDeconflictionPolicy
from RL_stack.gat_graph_builder import build_graph
from RL_stack.priority_protocol import (
    HarmonicPriorityManager,
    compute_connected_components,
)
from controllers.harmonic_navigation import HarmonicNavigationController


# Parameters used by the harmonic flow lookahead (shared with watch_rl_episode.py)
ASTAR_MODE_STEPS:      int   = 100
GOAL_PROXIMITY_STOP:   float = 1.0
SCAN_SAFETY_MARGIN:    float = 0.15

CONTROLLER_CONFIG = {
    "base_repulsion_radius":   1.5,
    "base_repulsion_strength": 5.0,
}


def compute_repulsion_params(
    ranks: np.ndarray,
    component_sizes: np.ndarray,
    n_agents: int,
    base_radius: float,
    base_strength: float,
    at_goal: np.ndarray,
) -> tuple:

    # Default increment=1.0 for at_goal and unranked agents
    effective_increment = np.ones(n_agents, dtype=np.float64)

    eligible = ~at_goal
    cs = component_sizes[eligible].astype(np.float64).clip(min=1.0)
    effective_increment[eligible] = (
        cs - ranks[eligible].astype(np.float64)
    ) / cs

    repulsion_radii     = base_radius   + effective_increment
    repulsion_strengths = base_strength + effective_increment

    return repulsion_radii, repulsion_strengths


class GATDeconflictionController:

    def __init__(
        self,
        policy: GATDeconflictionPolicy,
        device: torch.device,
        interaction_radius: float = 0.5,
        deterministic: bool = True,
        base_controller=None,
        enable_astar_fallback: bool = False,   # retained for API compat; ignored
    ):
        del enable_astar_fallback  # no A* in this controller
        self.policy             = policy
        self.device             = device
        self.deterministic      = deterministic
        self.interaction_radius = interaction_radius

        self.harmonic_controller = (
            base_controller if base_controller is not None
            else HarmonicNavigationController()
        )
        self.priority_manager = HarmonicPriorityManager()

        # Policy hidden states (set on reset)
        self.hidden_states = None

        # Progress tracking for graph features
        self.prev_dist_to_goal  = None
        self.time_since_progress = None
        self.prev_velocities    = None

    # ------------------------------------------------------------------
    # Pruning helper — copied verbatim from watch_rl_episode.py
    # ------------------------------------------------------------------

    @staticmethod
    def prune_harmonic_colliders(env) -> int:

        controller = HarmonicNavigationController()
        controller.reset(env)

        positions = np.array(env.positions, dtype=np.float32)
        goals     = np.array(env.goals,     dtype=np.float32)
        dt        = float(getattr(env, "dt", 0.1))

        hits = controller.simulate_flow_hits_wall(
            positions=positions,
            dt=dt,
            lookahead_steps=ASTAR_MODE_STEPS,
            goals=goals,
            goal_proximity_stop=GOAL_PROXIMITY_STOP,
            safety_margin=SCAN_SAFETY_MARGIN,
        )

        keep      = ~hits
        n_removed = int(hits.sum())

        if n_removed > 0:
            env.positions = env.positions[keep]
            env.goals     = env.goals[keep]
            env.n_agents  = int(keep.sum())
            env._get_obs()

        return n_removed

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env, n_agents: int):
        """Initialise hidden states and base controller."""
        self.hidden_states = self.policy.init_hidden(n_agents, device=self.device)
        self.priority_manager.reset()
        if hasattr(self.harmonic_controller, "reset"):
            self.harmonic_controller.reset(env)

        positions = np.array(env.positions, dtype=np.float32)
        goals     = np.array(env.goals,     dtype=np.float32)
        self.prev_dist_to_goal   = np.linalg.norm(positions - goals, axis=1)
        self.time_since_progress = np.zeros(n_agents, dtype=np.int32)
        self.prev_velocities     = np.zeros((n_agents, 2), dtype=np.float32)

    # ------------------------------------------------------------------
    # Act
    # ------------------------------------------------------------------

    def act(
        self,
        env,
        active_mask: Optional[np.ndarray] = None,
        at_goal: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:

        if self.hidden_states is None:
            raise RuntimeError("Must call reset() before act()")

        N = env.n_agents
        if active_mask is None:
            active_mask = np.ones(N, dtype=bool)
        if at_goal is None:
            at_goal = np.zeros(N, dtype=bool)

        positions = np.array(env.positions, dtype=np.float32)
        goals     = np.array(env.goals,     dtype=np.float32)

        # ----------------------------------------------------------------
        # 1. Harmonic base velocities
        # ----------------------------------------------------------------
        harmonic_vels = self.harmonic_controller(env)   # [N, 2]

        # Headings derived from harmonic preferred direction
        speeds   = np.linalg.norm(harmonic_vels, axis=1)
        headings = np.where(
            speeds > 1e-6,
            np.arctan2(harmonic_vels[:, 1], harmonic_vels[:, 0]),
            0.0,
        )

        # Progress tracking (for graph node feature dim-7)
        curr_dist = np.linalg.norm(positions - goals, axis=1)
        improved  = curr_dist < self.prev_dist_to_goal - 1e-6
        self.time_since_progress = np.where(improved, 0, self.time_since_progress + 1)
        self.prev_dist_to_goal   = curr_dist.copy()

        # ----------------------------------------------------------------
        # 2. Build communication graph — at_goal agents become isolated nodes
        #    (active_mask passed in is ~collided; at_goal agents still "active"
        #    in the sense of not collided, but we exclude them from edges)
        # ----------------------------------------------------------------
        # Graph edges only form between agents that are active AND not at goal
        graph_active_mask = active_mask & ~at_goal

        graph = build_graph(
            env=env,
            harmonic_velocities=harmonic_vels,
            agent_velocities=self.prev_velocities,
            agent_headings=headings,
            time_since_progress=self.time_since_progress,
            active_mask=graph_active_mask,
        )

        # ----------------------------------------------------------------
        # 3. Compute connected components and attach component_sizes to graph
        # ----------------------------------------------------------------
        edge_index_np = (
            graph.edge_index.cpu().numpy()
            if graph.edge_index.numel() > 0
            else np.zeros((2, 0), dtype=np.int64)
        )

        _, component_map = compute_connected_components(
            edge_index_np, N, at_goal, active_mask
        )

        # Vectorised component-size lookup
        component_sizes = np.zeros(N, dtype=np.int64)
        if component_map:
            for members in component_map.values():
                sz = len(members)
                for idx in members:
                    component_sizes[idx] = sz

        graph.component_sizes = torch.tensor(
            component_sizes, dtype=torch.long, device=self.device
        )
        graph = graph.to(self.device)

        active_tensor = torch.from_numpy(active_mask.astype(np.float32)).to(self.device)

        # ----------------------------------------------------------------
        # 4. Policy forward pass
        # ----------------------------------------------------------------
        with torch.no_grad():
            actions, log_probs, values, self.hidden_states = self.policy.act(
                graph=graph,
                hidden_states=self.hidden_states,
                active_mask=active_tensor,
                deterministic=self.deterministic,
            )

        priority_scores = actions["priority_score"].cpu().numpy()   # [N]

        # ----------------------------------------------------------------
        # 5. Priority ranks (no velocity scaling)
        # ----------------------------------------------------------------
        ranks, component_sizes_arr, protocol_info = self.priority_manager.step(
            priority_scores=priority_scores,
            edge_index_np=edge_index_np,
            N=N,
            active_mask=active_mask,
            at_goal=at_goal,
        )

        # ----------------------------------------------------------------
        # 6. Per-agent repulsion parameters based on priority rank
        # ----------------------------------------------------------------
        repulsion_radii, repulsion_strengths = compute_repulsion_params(
            ranks=ranks,
            component_sizes=component_sizes_arr,
            n_agents=N,
            base_radius=CONTROLLER_CONFIG["base_repulsion_radius"],
            base_strength=CONTROLLER_CONFIG["base_repulsion_strength"],
            at_goal=at_goal,
        )

        # ----------------------------------------------------------------
        # 7. Final velocities with priority-modulated repulsion field
        # ----------------------------------------------------------------
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
            "astar_velocities": harmonic_vels,      # uniform-repulsion harmonic (render compat)
            "priority_scores":  priority_scores,
            "yield_mask":       ranks > 0,           # agents with rank > 0 have reduced repulsion
            "edge_index":       edge_index_np,
            "protocol_info":    protocol_info,
            "log_probs":        log_probs.cpu().numpy(),
            "values":           values.cpu().numpy(),
            "astar_mode":       np.zeros(N, dtype=bool),
            "frozen_mask":      np.zeros(N, dtype=bool),
        }
        return final_vels, info
