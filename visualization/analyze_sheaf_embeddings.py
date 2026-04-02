import sys
import os
import json
import math
from pathlib import Path

# ── Resolve src/ imports ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# ── Dependency checks ─────────────────────────────────────────────────────────
_missing = []
try:
    import numpy as np
except ImportError:
    _missing.append("numpy  →  pip install numpy")
try:
    import torch
except ImportError:
    _missing.append("torch  →  pip install torch")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
except ImportError:
    _missing.append("matplotlib  →  pip install matplotlib")
try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import pairwise_distances
    _SKLEARN = True
except ImportError:
    _missing.append("scikit-learn  →  pip install scikit-learn")
    _SKLEARN = False
try:
    import umap as umap_lib
    UMAP_AVAILABLE = True
except ImportError:
    print("[warn] UMAP not installed — falling back to PCA only.  pip install umap-learn")
    UMAP_AVAILABLE = False

if _missing:
    print("Missing required packages:")
    for m in _missing:
        print(f"  {m}")
    sys.exit(1)

try:
    from torch_geometric.data import Data
except ImportError:
    print("torch_geometric required.  pip install torch_geometric")
    sys.exit(1)

# ── Project imports ───────────────────────────────────────────────────────────
from RL_stack.gat_deconfliction_policy import GATDeconflictionPolicy
from RL_stack.gat_graph_builder import build_graph, INTERACTION_RADIUS
from RL_stack.priority_protocol import compute_connected_components
from sim_env import MultiRobotEnv

# =============================================================================
# Phase 1: Configuration constants
# =============================================================================

CHECKPOINT_PATH = r"checkpoints/sheaf_deconfliction_v2/policy_ep669.pt"

EPISODE_JSON = (
    r"C:\Users\Logan\Desktop\Python Programs\multi_robot_nav"
    r"\runs\benchmark_20260323_093528\harmonic\ep0095_s95.json"
)
EPISODE_NPZ = (
    r"C:\Users\Logan\Desktop\Python Programs\multi_robot_nav"
    r"\runs\benchmark_20260323_093528\harmonic\ep0095_s95.npz"
)

OUTPUT_DIR = str(_ROOT / "visualization" / "sheaf_analysis_output")

# ── Analysis parameters ───────────────────────────────────────────────────────
UMAP_N_NEIGHBORS     = 30
UMAP_MIN_DIST        = 0.1
UMAP_METRIC          = "cosine"
PCA_N_COMPONENTS     = 10
RANDOM_STATE         = 42
SCATTER_POINT_SIZE   = 4
SCATTER_ALPHA        = 0.5
FIGURE_DPI           = 150

MAX_SPEED    = 1.5
ROBOT_RADIUS = 0.25


# =============================================================================
# Phase 2: Data loading
# =============================================================================

class _WallSeg:
    __slots__ = ("x1", "y1", "x2", "y2")
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2


class MockEnv:

    def __init__(self, world_size, walls_raw,
                 max_speed=MAX_SPEED, robot_radius=ROBOT_RADIUS):
        self.world_size  = float(world_size)
        self.max_speed   = float(max_speed)
        self.robot_radius = float(robot_radius)
        self.walls = [_WallSeg(**w) for w in walls_raw]
        # Set per-step in replay
        self.positions: np.ndarray = None
        self.goals:     np.ndarray = None
        self.n_agents:  int        = 0

    def _wall_endpoints(self):
        if not self.walls:
            empty = np.zeros((0, 2), dtype=float)
            return empty, empty
        a = np.array([[w.x1, w.y1] for w in self.walls], dtype=float)
        b = np.array([[w.x2, w.y2] for w in self.walls], dtype=float)
        return a, b

    @staticmethod
    def _batched_ray_wall_distances_per_agent(*args, **kwargs):
        return MultiRobotEnv._batched_ray_wall_distances_per_agent(*args, **kwargs)


def load_policy(checkpoint_path: str, device="cpu") -> GATDeconflictionPolicy:
    """Load GATDeconflictionPolicy from checkpoint and set to eval mode."""
    ckpt_path = _ROOT / checkpoint_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    policy = GATDeconflictionPolicy(
        node_dim=12, edge_dim=9, hidden_dim=128, n_heads=4
    ).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("policy_state_dict", "policy_state", "state_dict"):
            if key in ckpt:
                policy.load_state_dict(ckpt[key], strict=True)
                break
        else:
            policy.load_state_dict(ckpt, strict=True)
    else:
        policy.load_state_dict(ckpt, strict=True)

    policy.eval()
    print(f"[load] Policy loaded from {ckpt_path.name}")
    return policy


def load_episode(json_path: str, npz_path: str = None) -> dict:

    with open(json_path) as f:
        meta = json.load(f)

    n_agents    = int(meta["n_agents"])
    world_size  = float(meta["world_size"])
    walls_raw   = meta["walls"]

    if npz_path and Path(npz_path).exists():
        d = np.load(npz_path)
        positions     = d["positions"].astype(np.float32)       # [T, N, 2]
        velocities    = d["corrected_velocities"].astype(np.float32)
        harmonic_vels = d["astar_velocities"].astype(np.float32)
        active        = d["active"].astype(bool)
        goals_seq     = d["goals"].astype(np.float32)           # [T, N, 2]
        T             = positions.shape[0]
        # goals don't change; use frame 0
        goals = goals_seq[0]
        print(f"[load] Episode from NPZ: T={T}, N={n_agents}")
    else:
        raise FileNotFoundError(
            f"NPZ not found at {npz_path}. Expected fields: "
            "positions, corrected_velocities, astar_velocities, active, goals"
        )

    return {
        "positions":     positions,
        "velocities":    velocities,
        "harmonic_vels": harmonic_vels,
        "goals":         goals,
        "active":        active,
        "n_agents":      n_agents,
        "n_timesteps":   T,
        "world_size":    world_size,
        "walls_raw":     walls_raw,
    }


# =============================================================================
# Phase 3: Activation collector
# =============================================================================

class ActivationCollector:


    def __init__(self, policy: GATDeconflictionPolicy):
        self.policy = policy
        self._handles = []
        self._reset_buffers()
        self._register_hooks()

    def _reset_buffers(self):
        self.node_emb_l1     = []   # [N, H] per step
        self.node_emb_l2     = []
        self.gru_outputs     = []   # [N, H] per step
        self.priority_raw    = []   # [N, 1] per step
        self.restrict_l1     = []   # [E, d*H] per step (only when E > 0)
        self.restrict_l2     = []
        self.sheaf_losses_l1 = []   # scalar per step
        self.sheaf_losses_l2 = []
        self.edge_ttc_l1     = []   # [E] TTC per edge when restrict_l1 fires
        self.edge_ttc_l2     = []
        # For message reconstruction: input h and src indices per step (E > 0 only)
        self._h_input_l1     = []   # [N, H] input to sheaf_layer1
        self._src_l1         = []   # [E] src indices
        self._h_input_l2     = []   # [N, H] input to sheaf_layer2
        self._src_l2         = []
        # per-step metadata (set from outside during replay)
        self._current_edge_attr: np.ndarray = None

    def _register_hooks(self):
        def sheaf1_hook(module, inp, output):
            h_new, loss = output
            self.node_emb_l1.append(h_new.detach().cpu().numpy().copy())
            self.sheaf_losses_l1.append(loss.item())
            # Capture input h and src for message reconstruction (only when E > 0)
            h_in     = inp[0]           # [N, H]
            edge_idx = inp[1]           # [2, E]
            if edge_idx.shape[1] > 0:
                self._h_input_l1.append(h_in.detach().cpu().numpy().copy())
                self._src_l1.append(edge_idx[0].cpu().numpy().copy())

        def restrict1_hook(module, inp, output):
            self.restrict_l1.append(output.detach().cpu().numpy().copy())
            if self._current_edge_attr is not None:
                self.edge_ttc_l1.append(
                    self._current_edge_attr[:, 7].copy() * 10.0
                )

        def sheaf2_hook(module, inp, output):
            h_new, loss = output
            self.node_emb_l2.append(h_new.detach().cpu().numpy().copy())
            self.sheaf_losses_l2.append(loss.item())
            h_in     = inp[0]
            edge_idx = inp[1]
            if edge_idx.shape[1] > 0:
                self._h_input_l2.append(h_in.detach().cpu().numpy().copy())
                self._src_l2.append(edge_idx[0].cpu().numpy().copy())

        def restrict2_hook(module, inp, output):
            self.restrict_l2.append(output.detach().cpu().numpy().copy())
            if self._current_edge_attr is not None:
                self.edge_ttc_l2.append(
                    self._current_edge_attr[:, 7].copy() * 10.0
                )

        def gru_hook(module, inp, output):
            # GRUCell returns just the new hidden state tensor [N, H]
            self.gru_outputs.append(output.detach().cpu().numpy().copy())

        def priority_hook(module, inp, output):
            self.priority_raw.append(output.detach().cpu().numpy().copy())

        self._handles = [
            self.policy.sheaf_layer1.register_forward_hook(sheaf1_hook),
            self.policy.sheaf_layer1.restriction_mlp.register_forward_hook(restrict1_hook),
            self.policy.sheaf_layer2.register_forward_hook(sheaf2_hook),
            self.policy.sheaf_layer2.restriction_mlp.register_forward_hook(restrict2_hook),
            self.policy.gru.register_forward_hook(gru_hook),
            self.policy.priority_head.register_forward_hook(priority_hook),
        ]

    def finalize(self):
        """Stack all per-step lists into flat numpy arrays."""
        self.node_emb_l1_arr  = np.concatenate(self.node_emb_l1,  axis=0)  # [T*N, H]
        self.node_emb_l2_arr  = np.concatenate(self.node_emb_l2,  axis=0)
        self.gru_outputs_arr  = np.concatenate(self.gru_outputs,  axis=0)
        self.priority_arr     = np.concatenate(self.priority_raw, axis=0).squeeze(-1)  # [T*N]

        if self.restrict_l1:
            self.restrict_l1_arr  = np.concatenate(self.restrict_l1, axis=0)   # [total_E, d*H]
            self.edge_ttc_l1_arr  = np.concatenate(self.edge_ttc_l1, axis=0)
        else:
            self.restrict_l1_arr  = np.zeros((0, 1))
            self.edge_ttc_l1_arr  = np.zeros(0)

        if self.restrict_l2:
            self.restrict_l2_arr  = np.concatenate(self.restrict_l2, axis=0)
            self.edge_ttc_l2_arr  = np.concatenate(self.edge_ttc_l2, axis=0)
        else:
            self.restrict_l2_arr  = np.zeros((0, 1))
            self.edge_ttc_l2_arr  = np.zeros(0)

        # Compute stalk-space messages: msg[e] = F[e] @ h_src[e]  →  [E, stalk_dim=64]
        H = self.gru_outputs_arr.shape[1]  # 128
        d = H // 2                          # 64
        for layer_idx, (h_inputs, srcs, Fflats, attr) in enumerate([
            (self._h_input_l1, self._src_l1, self.restrict_l1, "messages_l1"),
            (self._h_input_l2, self._src_l2, self.restrict_l2, "messages_l2"),
        ]):
            if h_inputs:
                msgs = []
                for h_in, src, F_flat in zip(h_inputs, srcs, Fflats):
                    E_step = F_flat.shape[0]
                    F_mats = F_flat.reshape(E_step, d, H)          # [E, 64, 128]
                    h_src  = h_in[src]                              # [E, 128]
                    msg    = np.einsum('edh,eh->ed', F_mats, h_src) # [E, 64]
                    msgs.append(msg)
                setattr(self, attr + "_arr", np.concatenate(msgs, axis=0))
            else:
                setattr(self, attr + "_arr", np.zeros((0, d)))

        print(f"[collect] node_emb_l1:  {self.node_emb_l1_arr.shape}")
        print(f"[collect] node_emb_l2:  {self.node_emb_l2_arr.shape}")
        print(f"[collect] gru_outputs:  {self.gru_outputs_arr.shape}")
        print(f"[collect] priority:     {self.priority_arr.shape}")
        print(f"[collect] restrict_l1:  {self.restrict_l1_arr.shape}")
        print(f"[collect] restrict_l2:  {self.restrict_l2_arr.shape}")
        print(f"[collect] messages_l1:  {self.messages_l1_arr.shape}")
        print(f"[collect] messages_l2:  {self.messages_l2_arr.shape}")

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# =============================================================================
# Phase 4: Episode replay
# =============================================================================

def replay_episode(policy, episode_data: dict, collector: ActivationCollector,
                   device="cpu") -> dict:

    positions     = episode_data["positions"]      # [T, N, 2]
    velocities    = episode_data["velocities"]     # [T, N, 2]
    harmonic_vels = episode_data["harmonic_vels"]  # [T, N, 2]
    active        = episode_data["active"]         # [T, N]
    goals         = episode_data["goals"]          # [N, 2]
    T, N          = positions.shape[:2]
    world_size    = episode_data["world_size"]
    walls_raw     = episode_data["walls_raw"]

    env = MockEnv(world_size=world_size, walls_raw=walls_raw)

    hidden_states = policy.init_hidden(N, device=torch.device(device))

    # Per-agent metadata
    n_neighbors_all     = np.zeros((T, N), dtype=np.float32)
    ttc_min_all         = np.full((T, N), 10.0, dtype=np.float32)
    dist_to_goal_all    = np.zeros((T, N), dtype=np.float32)
    component_size_all  = np.zeros((T, N), dtype=np.float32)
    timestep_all        = np.zeros((T, N), dtype=np.float32)

    # Track time_since_progress
    prev_dist = np.linalg.norm(positions[0] - goals[None, :, :].squeeze(0)
                               if goals.ndim == 2 else positions[0] - goals,
                               axis=1).astype(np.float32)
    time_since_progress = np.zeros(N, dtype=np.int32)

    print(f"[replay] Replaying {T} timesteps with N={N} agents …")

    for t in range(T):
        pos_t  = positions[t].astype(np.float32)       # [N, 2]
        vel_t  = velocities[t].astype(np.float32)      # [N, 2]
        hv_t   = harmonic_vels[t].astype(np.float32)   # [N, 2]
        act_t  = active[t].astype(bool)                # [N]

        # Headings from actual velocities (fall back to harmonic if zero)
        speeds = np.linalg.norm(vel_t, axis=1)
        headings = np.where(
            speeds > 1e-6,
            np.arctan2(vel_t[:, 1], vel_t[:, 0]),
            np.arctan2(hv_t[:, 1], hv_t[:, 0]),
        ).astype(np.float32)

        # Update time_since_progress
        curr_dist = np.linalg.norm(pos_t - goals, axis=1).astype(np.float32)
        improved  = curr_dist < prev_dist - 1e-6
        time_since_progress = np.where(improved, 0, time_since_progress + 1)
        prev_dist = curr_dist.copy()

        # Build graph
        env.positions = pos_t
        env.goals     = np.tile(goals, (1, 1)) if goals.ndim == 2 else goals
        env.n_agents  = N

        graph = build_graph(
            env=env,
            harmonic_velocities=hv_t,
            agent_velocities=vel_t,
            agent_headings=headings,
            time_since_progress=time_since_progress,
            active_mask=act_t,
        )

        # Connected components for component_sizes feature
        edge_index_np = (
            graph.edge_index.cpu().numpy()
            if graph.edge_index.numel() > 0
            else np.zeros((2, 0), dtype=np.int64)
        )
        _, comp_map = compute_connected_components(
            edge_index_np, N,
            at_goal=np.zeros(N, dtype=bool),
            active_mask=act_t,
        )
        comp_sizes = np.zeros(N, dtype=np.int64)
        if comp_map:
            for members in comp_map.values():
                sz = len(members)
                for idx in members:
                    comp_sizes[idx] = sz

        graph.component_sizes = torch.tensor(comp_sizes, dtype=torch.long,
                                             device=device)
        graph = graph.to(device)

        # Share edge_attr with hooks for TTC capture
        edge_attr_np = graph.edge_attr.cpu().numpy() if graph.edge_attr.numel() > 0 \
                       else np.zeros((0, 9), dtype=np.float32)
        collector._current_edge_attr = edge_attr_np

        active_tensor = torch.from_numpy(act_t.astype(np.float32)).to(device)

        with torch.no_grad():
            # Single forward pass — hooks fire exactly once per step
            _, _, hidden_states, _, _ = policy.forward(graph, hidden_states, active_tensor)

        # Per-agent metadata (vectorised)
        E = edge_index_np.shape[1]
        n_neighbors_all[t] = 0
        if E > 0:
            src = edge_index_np[0]
            # n_neighbors: count how many edges leave each src node
            counts = np.bincount(src, minlength=N).astype(np.float32)
            n_neighbors_all[t] = counts
            # min TTC per agent: edge_attr[:,7] is TTC/10
            ttc_vals = edge_attr_np[:, 7] * 10.0   # [E] raw TTC seconds
            np.minimum.at(ttc_min_all[t], src, ttc_vals)

        dist_to_goal_all[t]   = curr_dist
        component_size_all[t] = comp_sizes.astype(np.float32)
        timestep_all[t]       = float(t)

        if (t + 1) % 50 == 0:
            print(f"  … step {t+1}/{T}")

    collector.finalize()
    collector._current_edge_attr = None

    # Flatten T×N → T*N
    meta = {
        "n_neighbors":    n_neighbors_all.ravel(),
        "ttc_min":        ttc_min_all.ravel(),
        "dist_to_goal":   dist_to_goal_all.ravel(),
        "component_size": component_size_all.ravel(),
        "timestep":       timestep_all.ravel(),
        "active":         active.ravel().astype(np.float32),
    }
    return meta


# =============================================================================
# Phase 5: Dimensionality reduction
# =============================================================================

def reduce_embeddings(embeddings: np.ndarray, method: str = "umap",
                      n_pca_components: int = 2):
    """
    Reduce [M, D] → [M, 2].  Returns (reduced [M,2], diagnostics dict).
    """
    # Subsample for UMAP if very large
    M = embeddings.shape[0]
    diagnostics = {}

    if method == "pca" or not UMAP_AVAILABLE:
        pca = PCA(n_components=min(PCA_N_COMPONENTS, embeddings.shape[1]),
                  random_state=RANDOM_STATE)
        pca.fit(embeddings)
        ev = pca.explained_variance_ratio_
        diagnostics["explained_variance"] = ev
        reduced = pca.transform(embeddings)[:, :2]
        print(f"  PCA PC1={ev[0]:.3f}  PC1+PC2={ev[:2].sum():.3f}")
        return reduced, diagnostics

    # UMAP
    reducer = umap_lib.UMAP(
        n_neighbors=min(UMAP_N_NEIGHBORS, M - 1),
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
        n_components=2,
        low_memory=True,
    )
    reduced = reducer.fit_transform(embeddings)
    if _SKLEARN:
        try:
            from sklearn.manifold import trustworthiness
            tw = trustworthiness(embeddings, reduced, n_neighbors=10)
            diagnostics["trustworthiness"] = tw
            print(f"  UMAP trustworthiness={tw:.3f}")
        except Exception:
            pass
    return reduced, diagnostics


def pca_full(embeddings: np.ndarray):
    """Return full PCA model on embeddings (up to 10 components)."""
    k = min(PCA_N_COMPONENTS, embeddings.shape[1], embeddings.shape[0])
    pca = PCA(n_components=k, random_state=RANDOM_STATE)
    pca.fit(embeddings)
    return pca


# =============================================================================
# Phase 6: Visualization helpers
# =============================================================================

def _scatter(ax, xy, c, title, cmap="viridis", vmin=None, vmax=None,
             cbar_label=""):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap=cmap, s=SCATTER_POINT_SIZE,
                    alpha=SCATTER_ALPHA, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(sc, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)


def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {path}")


def fig1_node_embedding_space(gru_pca2, gru_umap2, meta):
    """4×2 grid: PCA & UMAP of gru_outputs colored by 4 features."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Figure 1 — GRU Output Embedding Space", fontsize=12)

    rows = [("PCA",  gru_pca2),
            ("UMAP", gru_umap2)]
    cols = [
        ("Priority score",  meta["priority"],   "RdBu_r",  None,  None,  "priority"),
        ("N neighbors",     meta["n_neighbors"],"plasma",  0,     None,  "count"),
        ("Min TTC (s)",     meta["ttc_min"],    "RdYlGn",  0,     10.0,  "seconds"),
        ("Dist to goal",    meta["dist_to_goal"],"Blues",  0,     None,  "world units"),
    ]

    for r, (rlabel, xy) in enumerate(rows):
        for c, (clabel, color, cmap, vmin, vmax, cbar) in enumerate(cols):
            if xy is None:
                axes[r, c].axis("off")
                continue
            _scatter(axes[r, c], xy, color, f"{rlabel}: {clabel}",
                     cmap=cmap, vmin=vmin, vmax=vmax, cbar_label=cbar)

    fig.tight_layout()
    _save(fig, "fig1_node_embedding_space.png")


def fig2_sheaf_layer_comparison(l1_pca, l2_pca, l1_umap, l2_umap, priority):
    """2×2: PCA/UMAP before and after sheaf processing colored by priority."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Figure 2 — Sheaf Layer Comparison (priority coloring)", fontsize=12)

    pairs = [
        (axes[0, 0], l1_pca,  "PCA  — after sheaf_layer1"),
        (axes[0, 1], l2_pca,  "PCA  — after sheaf_layer2"),
        (axes[1, 0], l1_umap, "UMAP — after sheaf_layer1"),
        (axes[1, 1], l2_umap, "UMAP — after sheaf_layer2"),
    ]
    for ax, xy, title in pairs:
        if xy is None:
            ax.axis("off")
            continue
        _scatter(ax, xy, priority, title, cmap="RdBu_r", cbar_label="priority mean")

    fig.text(0.5, 0.01,
             "Left: after sheaf_layer1.  Right: after sheaf_layer2.\n"
             "Increasing structure between layers suggests sheaf convolution is "
             "refining representations.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, "fig2_sheaf_layer_comparison.png")


def fig3_restriction_map_analysis(r1_arr, r1_ttc, r2_arr, r2_ttc):
    """1×2: PCA of restriction map outputs colored by min TTC."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 3 — Restriction Map Analysis (TTC coloring)", fontsize=12)

    for ax, r_arr, r_ttc, title in [
        (axes[0], r1_arr, r1_ttc, "sheaf_layer1 restriction maps"),
        (axes[1], r2_arr, r2_ttc, "sheaf_layer2 restriction maps"),
    ]:
        if r_arr.shape[0] < 3:
            ax.set_title(f"{title}\n(no edges collected)", fontsize=8)
            ax.axis("off")
            continue
        # PCA to 2D on possibly very wide [E, d*H] matrix
        k = min(2, r_arr.shape[1], r_arr.shape[0])
        pca = PCA(n_components=k, random_state=RANDOM_STATE)
        xy = pca.fit_transform(r_arr)
        if xy.shape[1] < 2:
            xy = np.hstack([xy, np.zeros((len(xy), 1))])
        _scatter(ax, xy, r_ttc, f"PCA — {title}",
                 cmap="RdYlGn", vmin=0, vmax=10, cbar_label="TTC (s)")

    fig.tight_layout()
    _save(fig, "fig3_restriction_map_analysis.png")


def fig4_priority_distribution(priority, ttc_min):
    """Left: histogram of priorities.  Right: scatter priority vs min TTC."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 4 — Priority Score Distribution", fontsize=12)

    axes[0].hist(priority, bins=60, color="steelblue", edgecolor="none", alpha=0.8)
    axes[0].axvline(0, color="red", lw=1, linestyle="--")
    axes[0].set_xlabel("Priority mean")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Priority score histogram (all timesteps)")

    sc = axes[1].scatter(ttc_min, priority,
                         s=SCATTER_POINT_SIZE, alpha=SCATTER_ALPHA,
                         c=ttc_min, cmap="RdYlGn", vmin=0, vmax=10)
    plt.colorbar(sc, ax=axes[1], label="TTC (s)")
    axes[1].set_xlabel("Min TTC (s)")
    axes[1].set_ylabel("Priority score")
    axes[1].set_title("Priority vs Min TTC\n(positive r → high-priority = more time before collision)")

    fig.tight_layout()
    _save(fig, "fig4_priority_distribution.png")


def fig5_sheaf_loss(losses_l1, losses_l2):
    """Sheaf inconsistency loss per step."""
    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.arange(len(losses_l1))
    ax.plot(t, losses_l1, label="sheaf_layer1", alpha=0.8)
    ax.plot(t, losses_l2, label="sheaf_layer2", alpha=0.8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Sheaf loss (inconsistency)")
    ax.set_title("Figure 5 — Sheaf Loss Over Episode\n"
                 "(spikes = congested interactions; low = free navigation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig5_sheaf_loss.png")


def fig6_pca_variance(pca_model):
    """Bar chart of explained variance ratio (first 10 PCs) of gru_outputs."""
    ev = pca_model.explained_variance_ratio_
    k  = len(ev)
    pct12 = ev[:2].sum() * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, k + 1), ev * 100, color="steelblue", edgecolor="none")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(
        f"Figure 6 — PCA Explained Variance (GRU outputs)\n"
        f"PC1+PC2 = {pct12:.1f}%  "
        f"{'→ structured representation' if pct12 > 15 else '→ likely unstructured (< 15%)'}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig6_pca_variance.png")


def fig7_messages_3d(messages_arr, ttc_arr, output_dir):
    """
    Project stalk-space messages [total_E, 64] to R^2 with UMAP (or PCA fallback)
    and render as 2D scatter plots colored by TTC and message norm.

    Each point is one message passed along one edge at one timestep.
    """
    if messages_arr.shape[0] < 4:
        print("[fig7] Not enough edges to plot messages.")
        return

    print(f"[fig7] Reducing {messages_arr.shape} messages to R^2 …")

    if UMAP_AVAILABLE:
        reducer = umap_lib.UMAP(
            n_neighbors=min(UMAP_N_NEIGHBORS, messages_arr.shape[0] - 1),
            min_dist=UMAP_MIN_DIST,
            metric=UMAP_METRIC,
            random_state=RANDOM_STATE,
            n_components=2,
            low_memory=True,
        )
        xy = reducer.fit_transform(messages_arr)
        method_label = "UMAP"
    else:
        pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
        xy   = pca2.fit_transform(messages_arr)
        ev   = pca2.explained_variance_ratio_
        method_label = f"PCA (PC1+PC2={ev.sum()*100:.1f}%)"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    titles_colors = [
        ("TTC (s)",      ttc_arr,                               "RdYlGn", 0,    10.0),
        ("Message norm", np.linalg.norm(messages_arr, axis=1),  "plasma", None, None),
    ]
    for ax, (label, c_vals, cmap, vmin, vmax) in zip(axes, titles_colors):
        sc = ax.scatter(xy[:, 0], xy[:, 1],
                        c=c_vals, cmap=cmap, s=SCATTER_POINT_SIZE,
                        alpha=SCATTER_ALPHA, vmin=vmin, vmax=vmax,
                        rasterized=True)
        plt.colorbar(sc, ax=ax, label=label)
        ax.set_title(f"Figure 7 — Sheaf Messages in R²\n({method_label})\nColored by {label}",
                     fontsize=9)
        ax.set_xlabel("dim 1", fontsize=8)
        ax.set_ylabel("dim 2", fontsize=8)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    _save(fig, "fig7_messages_3d.png")


def combined_figure(output_dir: str):
    """Combine all saved PNGs into one overview figure."""
    names = [
        "fig1_node_embedding_space.png",
        "fig2_sheaf_layer_comparison.png",
        "fig3_restriction_map_analysis.png",
        "fig4_priority_distribution.png",
        "fig5_sheaf_loss.png",
        "fig6_pca_variance.png",
        "fig7_messages_3d.png",
    ]
    images = []
    for n in names:
        p = os.path.join(output_dir, n)
        if os.path.exists(p):
            import matplotlib.image as mpimg
            images.append((n, mpimg.imread(p)))

    if not images:
        return
    fig, axes = plt.subplots(4, 2, figsize=(20, 36))
    for ax, (name, img) in zip(axes.ravel(), images):
        ax.imshow(img)
        ax.set_title(name.replace(".png", ""), fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(output_dir, "combined_overview.png")
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out}")


# =============================================================================
# Phase 7: Interpretation report
# =============================================================================

def interpretation_report(collector, meta, pca_model, output_dir):
    """Print structured analysis report and save as interpretation.txt."""
    ev   = pca_model.explained_variance_ratio_
    pc1  = float(ev[0])
    pc12 = float(ev[:2].sum())

    p    = meta["priority"]   # active-only priority scores (passed via meta_active)
    pmean, pstd, pmin, pmax = p.mean(), p.std(), p.min(), p.max()

    l1_mean = float(np.mean(collector.sheaf_losses_l1))
    l2_mean = float(np.mean(collector.sheaf_losses_l2))

    ttc  = meta["ttc_min"]
    dtg  = meta["dist_to_goal"]
    r_ttc = r_dtg = 0.0
    try:
        from scipy.stats import pearsonr
        valid_ttc = (ttc < 9.99)
        if valid_ttc.sum() > 10:
            r_ttc, _ = pearsonr(p[valid_ttc], ttc[valid_ttc])
        if dtg.std() > 1e-6:
            r_dtg, _ = pearsonr(p, dtg)
    except ImportError:
        pass

    # Struct flag: based on embedding PCA structure only, not priority magnitude
    emb_structured  = pc12 > 0.15
    prio_spread     = pmax - pmin
    prio_meaningful = prio_spread > 0.3   # range at least 0.3 across the episode

    def _corr_label(r):
        if r > 0.15:   return f"positive (r={r:.3f})"
        if r < -0.15:  return f"negative (r={r:.3f})"
        return f"weak (r={r:.3f})"

    report = f"""
================================================
SHEAF EMBEDDING ANALYSIS REPORT
Checkpoint: {CHECKPOINT_PATH}
Episode:    {Path(EPISODE_JSON).name}
================================================

PCA DIAGNOSTICS (GRU outputs)
  Explained variance (PC1):        {pc1:.3f}
  Explained variance (PC1+PC2):    {pc12:.3f}
  Interpretation: {"Structured — embeddings organise along dominant axes" if emb_structured else "Unstructured — embeddings are diffuse"}

UMAP DIAGNOSTICS
  Available: {UMAP_AVAILABLE}{"" if UMAP_AVAILABLE else "  ->  pip install umap-learn"}

PRIORITY SCORE STATISTICS
  Mean:    {pmean:.4f}
  Std:     {pstd:.4f}
  Min:     {pmin:.4f}
  Max:     {pmax:.4f}
  Range:   {prio_spread:.4f}  (> 0.3 = agents assigned meaningfully different priorities)

SHEAF CONSISTENCY
  Mean sheaf loss layer 1:  {l1_mean:.4f}
  Mean sheaf loss layer 2:  {l2_mean:.4f}
  Note: layer 2 loss > layer 1 is expected — layer 2 resolves deeper inconsistencies
        that layer 1 did not fully reconcile.

PRIORITY CORRELATIONS
  vs min TTC:       {_corr_label(r_ttc)}
  vs dist_to_goal:  {_corr_label(r_dtg)}
  Note: a positive r(priority, dist_to_goal) means agents farther from their goal
        receive higher priority — consistent with urgency-based deconfliction.

OVERALL ASSESSMENT
  Embedding structure: {"[OK] Structured" if emb_structured else "[--] Unstructured"}
  Priority range:      {"[OK] Differentiated" if prio_meaningful else "[--] Near-uniform (range < 0.3)"}
  {"The model has learned a structured priority signal — embeddings are organised and agents receive meaningfully different priorities." if emb_structured and prio_meaningful else "Embeddings are structured but priority scores are compressed — the priority head may still be near its zero initialisation or the reward signal for priority differentiation is weak." if emb_structured else "Embeddings and priorities both appear unstructured — policy has not converged to a meaningful representation."}

Figures saved to: {output_dir}/
================================================
""".strip()

    print("\n" + report + "\n")
    out = os.path.join(output_dir, "interpretation.txt")
    Path(out).write_text(report + "\n")
    print(f"[save] {out}")


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cpu"

    # ── Load policy & episode ─────────────────────────────────────────────────
    policy  = load_policy(CHECKPOINT_PATH, device=device)
    episode = load_episode(EPISODE_JSON, EPISODE_NPZ)

    # ── Replay & collect activations ──────────────────────────────────────────
    collector = ActivationCollector(policy)
    try:
        meta_raw = replay_episode(policy, episode, collector, device=device)
    finally:
        collector.remove_hooks()

    # Attach priority scores from collector to meta dict
    meta_raw["priority"] = collector.priority_arr

    # ── Filter to active agents only ──────────────────────────────────────────
    # Inactive agents have their hidden states zeroed each step, which creates
    # a dense cluster of zero vectors.  Running PCA/UMAP on all points then
    # slicing is NOT sufficient — the zero vectors distort the manifold layout
    # for nearby active agents too.  All reductions run on active points only.
    act = meta_raw["active"].astype(bool)
    meta_active = {k: v[act] for k, v in meta_raw.items()}

    gru_act = collector.gru_outputs_arr[act]
    l1_act  = collector.node_emb_l1_arr[act]
    l2_act  = collector.node_emb_l2_arr[act]

    # ── Dimensionality reduction (active agents only) ─────────────────────────
    print("\n[reduce] Running PCA …")
    gru_pca2_a, _  = reduce_embeddings(gru_act, method="pca")
    l1_pca2_a, _   = reduce_embeddings(l1_act,  method="pca")
    l2_pca2_a, _   = reduce_embeddings(l2_act,  method="pca")
    pca_full_model = pca_full(gru_act)

    gru_umap2_a = l1_umap2_a = l2_umap2_a = None
    if UMAP_AVAILABLE:
        print("[reduce] Running UMAP on active agents only (this may take a minute) …")
        gru_umap2_a, _ = reduce_embeddings(gru_act, method="umap")
        l1_umap2_a, _  = reduce_embeddings(l1_act,  method="umap")
        l2_umap2_a, _  = reduce_embeddings(l2_act,  method="umap")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n[plot] Generating figures …")
    fig1_node_embedding_space(gru_pca2_a, gru_umap2_a, meta_active)
    fig2_sheaf_layer_comparison(l1_pca2_a, l2_pca2_a, l1_umap2_a, l2_umap2_a,
                                collector.priority_arr[act])
    fig3_restriction_map_analysis(collector.restrict_l1_arr,
                                  collector.edge_ttc_l1_arr,
                                  collector.restrict_l2_arr,
                                  collector.edge_ttc_l2_arr)
    fig4_priority_distribution(collector.priority_arr[act], meta_active["ttc_min"])
    fig5_sheaf_loss(collector.sheaf_losses_l1, collector.sheaf_losses_l2)
    fig6_pca_variance(pca_full_model)
    fig7_messages_3d(collector.messages_l1_arr, collector.edge_ttc_l1_arr, OUTPUT_DIR)
    combined_figure(OUTPUT_DIR)

    # ── Report ────────────────────────────────────────────────────────────────
    interpretation_report(collector, meta_active, pca_full_model, OUTPUT_DIR)


if __name__ == "__main__":
    main()
