import numpy as np
import matplotlib.pyplot as plt

EP_PATH = r"C:\Users\Logan\Desktop\Python Programs\multi_robot_nav\datasets\il_dataset\episodes_walls+obstacles\episode_000001.npz"  # <-- change this

def load_episode(ep_path: str):
    # allow_pickle True is necessary for edge_index / edge_attr object arrays
    data = np.load(ep_path, allow_pickle=True)
    keys = set(data.files)

    # Resolve keys with fallbacks
    walls = data["walls_xyxy"] if "walls_xyxy" in keys else None

    if "positions_all" in keys:
        positions_all = data["positions_all"]
    elif "positions" in keys:
        positions_all = data["positions"]
    else:
        # reconstruct from obs/next if present
        positions_all = np.concatenate([data["obs_positions"], data["next_positions"][-1:]], axis=0)

    if "lidar_all" in keys:
        lidar_all = data["lidar_all"]
    elif "lidar_ranges" in keys:
        lidar_all = data["lidar_ranges"]
    elif "obs_lidar_ranges" in keys:
        lidar_all = np.concatenate([data["obs_lidar_ranges"], data["next_lidar_ranges"][-1:]], axis=0)
    else:
        lidar_all = None

    goals = data["goals"] if "goals" in keys else None

    edge_index = data["edge_index"] if "edge_index" in keys else None
    edge_attr = data["edge_attr"] if "edge_attr" in keys else None

    return {
        "walls": walls,
        "positions_all": positions_all,
        "lidar_all": lidar_all,
        "goals": goals,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "keys": data.files,
    }

def plot_walls(ax, walls_xyxy):
    if walls_xyxy is None:
        return
    for (x1, y1, x2, y2) in walls_xyxy:
        ax.plot([x1, x2], [y1, y2], linewidth=2)

def plot_trajectories(ax, positions_all, goals=None, max_agents=10):
    T1, N, _ = positions_all.shape
    Nplot = min(N, max_agents)

    for i in range(Nplot):
        ax.plot(positions_all[:, i, 0], positions_all[:, i, 1], alpha=0.7)
        ax.scatter(positions_all[0, i, 0], positions_all[0, i, 1], marker="o")   # start
        ax.scatter(positions_all[-1, i, 0], positions_all[-1, i, 1], marker="x") # end
        if goals is not None:
            ax.scatter(goals[i, 0], goals[i, 1], marker="*", s=120)

def overlay_lidar(ax, pos_xy, ranges, max_range, n_rays, alpha=0.25):
    # Rays assumed evenly spaced [0, 2pi)
    angles = np.linspace(0.0, 2*np.pi, n_rays, endpoint=False)
    # Clip for drawing
    rr = np.clip(ranges, 0.0, max_range)
    xs = pos_xy[0] + rr * np.cos(angles)
    ys = pos_xy[1] + rr * np.sin(angles)
    for x, y in zip(xs, ys):
        ax.plot([pos_xy[0], x], [pos_xy[1], y], alpha=alpha)

def overlay_graph_edges(ax, positions_t, edge_index_t, max_edges=400, alpha=0.25):
    if edge_index_t is None or edge_index_t.size == 0:
        return
    E = edge_index_t.shape[1]
    if E > max_edges:
        # downsample for clarity
        idx = np.random.choice(E, size=max_edges, replace=False)
        edge_index_t = edge_index_t[:, idx]

    src = edge_index_t[0]
    dst = edge_index_t[1]
    for i, j in zip(src, dst):
        pi = positions_t[i]
        pj = positions_t[j]
        ax.plot([pi[0], pj[0]], [pi[1], pj[1]], alpha=alpha)

def main():
    ep = load_episode(EP_PATH)
    print("Loaded keys:", ep["keys"])
    pos_all = ep["positions_all"]
    walls = ep["walls"]
    goals = ep["goals"]
    lidar_all = ep["lidar_all"]
    edge_index = ep["edge_index"]

    T1, N, _ = pos_all.shape
    print(f"Trajectory: (T+1, N, 2) = {pos_all.shape}")

    # --- Plot 1: full trajectories on map
    fig, ax = plt.subplots()
    plot_walls(ax, walls)
    plot_trajectories(ax, pos_all, goals=goals, max_agents=20)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Walls + agent trajectories (start=o, end=x, goal=*)")
    plt.show()

    # --- Plot 2: zoom on one frame: walls + positions + lidar + graph edges
    t = min(10, T1 - 1)  # pick a small timestep to inspect
    agent_i = 0

    fig, ax = plt.subplots()
    plot_walls(ax, walls)
    ax.scatter(pos_all[t, :, 0], pos_all[t, :, 1], s=40)

    # overlay graph edges for this timestep (edge_index stored per t aligned to obs_t)
    if edge_index is not None and len(edge_index) > 0:
        # If you stored edge_index aligned with obs_t, its length is T = T1-1
        if len(edge_index) == T1 - 1 and t < T1 - 1:
            overlay_graph_edges(ax, pos_all[t], edge_index[t], alpha=0.2)

    # overlay lidar for one agent if available
    if lidar_all is not None:
        n_rays = lidar_all.shape[-1]
        max_range_guess = float(np.nanmax(lidar_all))  # just for visualization if you didn't store max_range
        overlay_lidar(ax, pos_all[t, agent_i], lidar_all[t, agent_i], max_range=max_range_guess, n_rays=n_rays, alpha=0.15)
        ax.scatter([pos_all[t, agent_i, 0]], [pos_all[t, agent_i, 1]], s=120)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Frame t={t}: positions + (optional) graph edges + (optional) lidar for agent {agent_i}")
    plt.show()

if __name__ == "__main__":
    main()
