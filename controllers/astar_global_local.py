from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import deque

import heapq
import numpy as np


@dataclass
class AStarConfig:
    grid_size: int = 128
    obstacle_inflate_radius: float = 0.01 # world units
    waypoint_tolerance: float = 0.3       # how close to a waypoint before advancing


@dataclass
class LocalOptimizerConfig:
    k_nav: float = 3.0           # weight on following the global path
    k_coll: float = 7.0          # collision penalty strength
    k_wall: float = 9.0          # wall / obstacle penalty strength
    safe_distance_factor: float = 3.0  # * robot_radius
    wall_clearance_factor: float = 1.2  # * robot_radius
    max_speed: float = 1.0
    dt: float = 0.1
    max_iters: int = 3           # gradient refinement iterations

    # Deadlock breaker: pass around agents that are stopped at their goal
    k_pass_goal_block: float = 0.35              # strength of lateral "go-around" (0 disables)
    goal_block_distance_factor: float = 3.0      # * robot_radius: how close a stopped agent must be to trigger
    stuck_kick_frac: float = 0.45               # fraction of max_speed used for orthogonal kick when stuck
    goal_block_front_angle_deg: float = 60.0     # degrees: "in front" cone half-angle


class AStarGlobalLocalController:


    def __init__(
        self,
        astar_config: Optional[AStarConfig] = None,
        opt_config: Optional[LocalOptimizerConfig] = None,
    ):
        self.astar_cfg = astar_config or AStarConfig()
        self.opt_cfg = opt_config or LocalOptimizerConfig()

        # Caches for the current episode
        self._grid: Optional[np.ndarray] = None          # (H, W) bool: True = obstacle
        self._grid_x: Optional[np.ndarray] = None
        self._grid_y: Optional[np.ndarray] = None
        self._paths_world: Dict[int, List[np.ndarray]] = {}
        self._waypoint_indices: Dict[int, int] = {}
        self._episode_id: int = -1
        self._replan_attempted: Dict[int, bool] = {}
        self._last_positions: Dict[int, np.ndarray] = {}
        self._stuck_counts: Dict[int, int] = {}
        self._best_goal_dist: Dict[int, float] = {}
        self.debug = False
        self._dbg_every = 10

        # When an agent replans using temporary obstacles (inactive agents),
        # we keep a short-lived flag so waypoint skipping/LOS checks don't
        # immediately "cut through" those temporary obstacles.
        self._avoid_inactive_until: Dict[int, int] = {}


    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def __call__(self, env) -> np.ndarray:

        # Detect new episode via env.t == 0 (since reset sets t=0)
        if env.t == 0 or self._episode_id != id(env):
            self._episode_id = id(env)
            self._build_grid(env)
            self._plan_paths_for_all_agents(env, skip_smoothing=False)
            self._replan_attempted = {i: False for i in range(env.n_agents)}  # track empty-path retries
            self._last_positions = {i: np.asarray(env.positions[i], dtype=float).copy() for i in range(env.n_agents)}
            self._stuck_counts = {i: 0 for i in range(env.n_agents)}
            self._best_goal_dist = {
                i: float(np.linalg.norm(np.asarray(env.positions[i], dtype=float) - np.asarray(env.goals[i], dtype=float)))
                for i in range(env.n_agents)
            }
            self._pass_side = {}  # (i,j)->(+1/-1) to keep consistent passing side around stopped agents
            self._avoid_inactive_until = {i: -1 for i in range(env.n_agents)}


        positions = np.asarray(env.positions, dtype=float)
        goals = np.asarray(env.goals, dtype=float)

        stuck_mask = np.zeros(env.n_agents, dtype=bool)
        N = env.n_agents

        # Agents that have reached their goals are treated as inactive/frozen.
        # We also use this mask when an agent is "stuck" to temporarily treat
        # these inactive agents as obstacles for replanning.
        inactive_mask = np.linalg.norm(positions - goals, axis=1) <= env.goal_tolerance

        u_nom = np.zeros_like(positions)

        for i in range(N):
            pos_i = positions[i]
            goal_i = goals[i]
            path = self._paths_world.get(i, None)

            # Replan conditions:
            #   - path missing/empty (once), or
            #   - agent appears stuck (very low motion) for >2 steps (t>1) and not at goal
            goal_dist = float(np.linalg.norm(pos_i - goal_i))
            if goal_dist <= env.goal_tolerance:
                u_nom[i] = np.zeros(2)
                self._paths_world[i] = []
                self._waypoint_indices[i] = 0
                self._replan_attempted[i] = True
                self._stuck_counts[i] = 0
                self._best_goal_dist[i] = goal_dist
                self._last_positions[i] = pos_i.copy()
                continue
            stuck = False

            if env.t <= 1:
                # Initialize on the first step(s)
                self._best_goal_dist[i] = goal_dist
                self._stuck_counts[i] = 0
            else:
                best = self._best_goal_dist.get(i, goal_dist)

                progress_eps = 0.02   # must improve goal distance by at least this much to count as progress
                patience = 20         # steps without progress before we call it "stuck"

                if goal_dist < best - progress_eps:
                    self._best_goal_dist[i] = goal_dist
                    self._stuck_counts[i] = 0
                else:
                    self._stuck_counts[i] = self._stuck_counts.get(i, 0) + 1

                stuck = (self._stuck_counts[i] >= patience) and (goal_dist >= env.goal_tolerance)

            stuck_mask[i] = stuck

            need_replan = ((path is None or len(path) == 0) and not self._replan_attempted.get(i, False)) or stuck

            if getattr(self, "debug", False) and (env.t % self._dbg_every == 0):
                print(f"[t={env.t:04d}] agent {i}: need_replan={need_replan} stuck={stuck} "
                    f"goal_dist={goal_dist:.3f} path_len={(0 if path is None else len(path))}")
            
            if need_replan:
                path = []
                # Try normal inflate, unsmoothed first then smoothed; if still empty, shrink inflate and retry once.
                for scale in (1.0, 0.7):
                    self._build_grid(env, inflate_scale=scale)

                    # If we're stuck, inject inactive agents (at goal) into a temporary grid
                    # so replanning routes around them.
                    grid_for_plan = self._grid
                    if stuck:
                        grid_for_plan = self._grid.copy()
                        self._add_inactive_agents_to_grid(
                            grid_for_plan,
                            env,
                            inactive_mask=inactive_mask,
                            exclude_agent=i,
                            inflate_scale=scale,
                        )
                        # For a short window after a stuck-triggered replan, prevent
                        # waypoint skipping/LOS heuristics from cutting through the
                        # temporary "inactive-agent" obstacles.
                        self._avoid_inactive_until[i] = int(env.t) + 50

                    def _bfs_reachable(grid: np.ndarray, s: tuple[int, int], g: tuple[int, int]) -> tuple[bool, int]:
                        H, W = grid.shape
                        if grid[s] or grid[g]:
                            return (False, 0)

                        q = deque([s])
                        seen = np.zeros((H, W), dtype=np.uint8)
                        seen[s] = 1
                        count = 1

                        # 8-neighborhood; include diagonal corner-cutting guard to match a "physically plausible" reachability
                        nbrs = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))

                        while q:
                            i, j = q.popleft()
                            if (i, j) == g:
                                return (True, count)

                            for di, dj in nbrs:
                                ni, nj = i + di, j + dj
                                if not (0 <= ni < H and 0 <= nj < W):
                                    continue
                                if seen[ni, nj] or grid[ni, nj]:
                                    continue

                                # prevent diagonal corner-cutting through blocked orthogonal neighbors
                                if di != 0 and dj != 0:
                                    if grid[i + di, j] or grid[i, j + dj]:
                                        continue

                                seen[ni, nj] = 1
                                q.append((ni, nj))
                                count += 1

                        return (False, count)

                    if getattr(self, "debug", False):
                        H, W = self._grid.shape
                        s = self._world_to_grid(pos_i, env.world_size, H, W)
                        g = self._world_to_grid(goal_i, env.world_size, H, W)
                        ok, comp = _bfs_reachable(self._grid, s, g)
                        print(f"    BFS reachable={ok} comp_size={comp} start={s} goal={g} "
                            f"grid[start]={self._grid[s]} grid[goal]={self._grid[g]}")

                    if getattr(self, "debug", False):
                        si, sj = self._world_to_grid(pos_i, env.world_size, self._grid.shape[0], self._grid.shape[1])
                        gi, gj = self._world_to_grid(goal_i, env.world_size, self._grid.shape[0], self._grid.shape[1])
                        print(f"    scale={scale:.2f} grid[start]={self._grid[si, sj]} grid[goal]={self._grid[gi, gj]} "
                            f"start_ij=({si},{sj}) goal_ij=({gi},{gj})")

                    # Plan ONLY for the agent that needs replanning.
                    # - If we injected dynamic obstacles (inactive agents), we skip smoothing so we
                    #   don't LOS-smooth through those temporary obstacles.
                    self._plan_path_for_agent(
                        env,
                        agent_idx=i,
                        grid=grid_for_plan,
                        skip_smoothing=True,
                        inflate_scale=scale,
                    )
                    path = self._paths_world.get(i, [])
                    if (path is None or len(path) == 0) and (not stuck):
                        # If not stuck (normal replan), allow LOS smoothing on the base grid.
                        self._plan_path_for_agent(
                            env,
                            agent_idx=i,
                            grid=grid_for_plan,
                            skip_smoothing=False,
                            inflate_scale=scale,
                        )
                        path = self._paths_world.get(i, [])
                    if path and len(path) > 0:
                        break

                if getattr(self, "debug", False):
                    pl = 0 if path is None else len(path)
                    print(f"    replan result agent {i}: path_len={pl} replan_attempted={self._replan_attempted.get(i, False)}")
                    if path is not None and pl >= 2:
                        print(f"    start={path[0]} end={path[-1]} first_step_len={np.linalg.norm(path[1]-path[0]):.3f}")

                # Avoid repeated replans on empty path; allow future stuck-based replans
                if path is None or len(path) == 0:
                    self._replan_attempted[i] = True
                self._stuck_counts[i] = 0
                self._best_goal_dist[i] = goal_dist

            if path is None or len(path) == 0:
                dir_vec = goal_i - pos_i
                norm = np.linalg.norm(dir_vec)
                if norm > 1e-6:
                    v = self.opt_cfg.k_nav * dir_vec / norm
                    speed = np.linalg.norm(v)
                    if speed > 1e-6:
                        v = v * (self.opt_cfg.max_speed / speed)
                    u_nom[i] = v

                # Simple stuck escape: add an orthogonal kick relative to the goal direction.
                # This is intentionally lightweight (no replanning) and only triggers once the
                # "stuck" patience criterion is met.
                if stuck:
                    dir_g = goal_i - pos_i
                    ng = np.linalg.norm(dir_g)
                    if ng > 1e-6:
                        ghat = dir_g / ng
                        ortho = np.array([-ghat[1], ghat[0]], dtype=float)
                        # Alternate side deterministically to avoid persistent bias.
                        sign = 1.0 if ((i + self._stuck_counts.get(i, 0)) % 2 == 0) else -1.0
                        kick_speed = float(self.opt_cfg.max_speed * self.opt_cfg.stuck_kick_frac)
                        u_nom[i] = u_nom[i] + sign * kick_speed * ortho
                        sp = np.linalg.norm(u_nom[i])
                        if sp > self.opt_cfg.max_speed and sp > 1e-9:
                            u_nom[i] = u_nom[i] * (self.opt_cfg.max_speed / sp)
                else:
                    u_nom[i] = np.zeros(2)
                self._last_positions[i] = pos_i.copy()
                continue

            # Advance waypoint index when we are close enough
            wp_idx = self._waypoint_indices.get(i, 0)
            wp_idx = min(wp_idx, len(path) - 1)
            cell = (2.0 * env.world_size) / max(self.astar_cfg.grid_size - 1, 1)
            tol = min(self.astar_cfg.waypoint_tolerance, 0.75 * cell)

            # If this agent recently replanned around inactive agents, make sure
            # we don't immediately skip waypoints along a straight line that
            # passes through those (temporary) dynamic obstacles.
            avoid_inactive = int(env.t) <= int(self._avoid_inactive_until.get(i, -1))

            while wp_idx < len(path) - 1:
                if np.linalg.norm(pos_i - path[wp_idx]) < tol:
                    # Require that we can see the NEXT waypoint from our current position.
                    seg_len = float(np.linalg.norm(path[wp_idx + 1] - pos_i))
                    n_samples = max(15, int(np.ceil(seg_len / max(0.5 * cell, 1e-6))) + 1)

                    if (self._segment_is_free_with_inactive(pos_i, path[wp_idx + 1], env,
                                                           inactive_mask=inactive_mask,
                                                           exclude_agent=i,
                                                           n_samples=n_samples)
                        if avoid_inactive else
                        self._segment_is_free(pos_i, path[wp_idx + 1], env, n_samples=n_samples)):
                        wp_idx += 1
                    else:
                        break
                else:
                    break

            self._waypoint_indices[i] = wp_idx
            target_wp = path[wp_idx]

            if getattr(self, "debug", False) and (env.t % self._dbg_every == 0):
                # Is the chosen waypoint actually visible?
                los_ok = None
                if wp_idx < len(path) - 1:
                    seg_len = float(np.linalg.norm(path[wp_idx + 1] - pos_i))
                    n_samples = max(15, int(np.ceil(seg_len / max(0.5 * cell, 1e-6))) + 1)
                    if avoid_inactive:
                        los_ok = self._segment_is_free_with_inactive(
                            pos_i,
                            path[wp_idx + 1],
                            env,
                            inactive_mask=inactive_mask,
                            exclude_agent=i,
                            n_samples=n_samples,
                        )
                    else:
                        los_ok = self._segment_is_free(pos_i, path[wp_idx + 1], env, n_samples=n_samples)

                print(f"[t={env.t:04d}] agent {i}: wp_idx={wp_idx}/{len(path)-1} "
                    f"dist_to_wp={np.linalg.norm(pos_i-target_wp):.3f} tol={tol:.3f} "
                    f"los_to_next={los_ok}")


            # Nominal tracking toward the current waypoint (this is what makes A* matter)
            dir_vec = target_wp - pos_i
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-6:
                v = self.opt_cfg.k_nav * dir_vec / norm
                speed = np.linalg.norm(v)
                if speed > 1e-6:
                    v = v * (self.opt_cfg.max_speed / speed)
                u_nom[i] = v

                # Simple stuck escape: add an orthogonal kick relative to the goal direction.
                # This is intentionally lightweight (no replanning) and only triggers once the
                # "stuck" patience criterion is met.
                if stuck:
                    dir_g = goal_i - pos_i
                    ng = np.linalg.norm(dir_g)
                    if ng > 1e-6:
                        ghat = dir_g / ng
                        ortho = np.array([-ghat[1], ghat[0]], dtype=float)
                        # Alternate side deterministically to avoid persistent bias.
                        sign = 1.0 if ((i + self._stuck_counts.get(i, 0)) % 2 == 0) else -1.0
                        kick_speed = float(self.opt_cfg.max_speed * self.opt_cfg.stuck_kick_frac)
                        u_nom[i] = u_nom[i] + sign * kick_speed * ortho
                        sp = np.linalg.norm(u_nom[i])
                        if sp > self.opt_cfg.max_speed and sp > 1e-9:
                            u_nom[i] = u_nom[i] * (self.opt_cfg.max_speed / sp)
                if getattr(self, "debug", False) and (env.t % self._dbg_every == 0):
                    print(f"    u_nom[{i}]={u_nom[i]} |u_nom|={np.linalg.norm(u_nom[i]):.3f} "
                        f"toward_wp_unit={(target_wp-pos_i)/max(np.linalg.norm(target_wp-pos_i),1e-9)}")

            else:
                u_nom[i] = np.zeros(2)

            # Track last position for stuck detection
            self._last_positions[i] = pos_i.copy()

        # Local optimization 

        active = ~inactive_mask

        # --- Deadlock breaker: go around agents that are stopped at their goal ---
        if self.opt_cfg.k_pass_goal_block > 0.0:
            inactive = ~active
            max_speed = self.opt_cfg.max_speed if self.opt_cfg.max_speed is not None else env.max_speed

            block_dist = self.opt_cfg.goal_block_distance_factor * env.robot_radius
            front_cos = float(np.cos(np.deg2rad(self.opt_cfg.goal_block_front_angle_deg)))

            # lazily create memory if loading older checkpoints
            if not hasattr(self, "_pass_side") or self._pass_side is None:
                self._pass_side = {}

            for i in range(N):
                if not active[i]:
                    continue

                v = u_nom[i]
                sp = float(np.linalg.norm(v))
                if sp < 1e-6:
                    # fallback to goal direction
                    dg = goals[i] - positions[i]
                    ng = np.linalg.norm(dg)
                    if ng < 1e-6:
                        continue
                    gdir = dg / ng
                else:
                    gdir = v / sp

                # Find the nearest inactive agent that is in front and within block_dist
                best_j = -1
                best_d = 1e9
                best_r = None

                for j in range(N):
                    if not inactive[j]:
                        continue

                    r = positions[j] - positions[i]
                    d = float(np.linalg.norm(r))
                    if d < 1e-6 or d >= block_dist or d >= best_d:
                        continue

                    rhat = r / d
                    if float(np.dot(gdir, rhat)) > front_cos:
                        best_j = j
                        best_d = d
                        best_r = r

                if best_j >= 0:
                    rhat = best_r / best_d
                    perp = np.array([-rhat[1], rhat[0]], dtype=float)

                    key = (i, best_j)
                    side = self._pass_side.get(key)
                    if side is None:
                        # deterministic tie-break; avoids left/right jitter when perfectly collinear
                        side = 1.0 if (i < best_j) else -1.0
                        self._pass_side[key] = side

                    # Stronger lateral push when closer; capped so we don't blow past max speed
                    mag = self.opt_cfg.k_pass_goal_block * (1.0 - best_d / block_dist) * max_speed
                    v_new = v + side * mag * perp

                    s = float(np.linalg.norm(v_new))
                    if s > max_speed:
                        v_new *= (max_speed / (s + 1e-9))

                    u_nom[i] = v_new
        # --- end deadlock breaker ---


        actions = self._local_optimize(env, u_nom, active_mask = active)

        # If an agent is stuck, apply an orthogonal kick AFTER local optimization.
        # This prevents the optimizer's collision gradients from "erasing" the kick.
        if np.any(stuck_mask):
            max_speed = float(getattr(env, "max_speed", 1.0))
            kick_speed = float(self.opt_cfg.stuck_kick_frac) * max_speed
            for i in range(env.n_agents):
                if not stuck_mask[i]:
                    continue
                gvec = goals[i] - positions[i]
                ng = np.linalg.norm(gvec)
                if ng < 1e-9:
                    continue
                ghat = gvec / ng
                ortho = np.array([-ghat[1], ghat[0]], dtype=float)
                # Deterministic alternating side based on how long we've been stuck
                sign = -1.0 if (self._stuck_counts.get(i, 0) % 2 == 0) else 1.0
                actions[i] = actions[i] + sign * kick_speed * ortho
                s = np.linalg.norm(actions[i])
                if s > max_speed:
                    actions[i] = actions[i] * (max_speed / (s + 1e-9))
        if getattr(self, "debug", False) and (env.t % self._dbg_every == 0):
            # Show worst-case deviation from nominal
            diffs = np.linalg.norm(actions - u_nom, axis=1)
            k = int(np.argmax(diffs))
            print(f"[t={env.t:04d}] worst action deviation: agent {k}, "
                f"|u_nom|={np.linalg.norm(u_nom[k]):.3f} |act|={np.linalg.norm(actions[k]):.3f} "
                f"|act-u_nom|={diffs[k]:.3f} act={actions[k]} u_nom={u_nom[k]}")



        return actions

    # ------------------------------------------------------------------
    # Grid construction & A*
    # ------------------------------------------------------------------

    def _build_grid(self, env, inflate_scale: float = 1.0):
        """Build occupancy grid from env.walls with inflated obstacles."""
        world_size = env.world_size
        H = W = self.astar_cfg.grid_size

        xs = np.linspace(-world_size, world_size, W)
        ys = np.linspace(-world_size, world_size, H)
        grid_x, grid_y = np.meshgrid(xs, ys)  # (H, W)

        grid = np.zeros((H, W), dtype=bool)
        walls = getattr(env, "walls", [])
        has_env_dist = hasattr(env, "_distance_point_to_segment")
        
        cell = (2.0 * env.world_size) / max(self.astar_cfg.grid_size - 1, 1)
        inflate = self._compute_inflate(env, inflate_scale)


        for w in walls:
            a = np.array([w.x1, w.y1], dtype=float)
            b = np.array([w.x2, w.y2], dtype=float)

            # Restrict to bounding box around the wall
            xmin = min(a[0], b[0]) - inflate
            xmax = max(a[0], b[0]) + inflate
            ymin = min(a[1], b[1]) - inflate
            ymax = max(a[1], b[1]) + inflate

            j_min = np.searchsorted(xs, xmin, side="left")
            j_max = np.searchsorted(xs, xmax, side="right")
            i_min = np.searchsorted(ys, ymin, side="left")
            i_max = np.searchsorted(ys, ymax, side="right")

            j_min = max(0, j_min)
            j_max = min(W, j_max)
            i_min = max(0, i_min)
            i_max = min(H, i_max)

            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    p = np.array([grid_x[i, j], grid_y[i, j]], dtype=float)
                    if has_env_dist:
                        d = env._distance_point_to_segment(p, a, b)
                    else:
                        d = self._distance_point_to_segment_local(p, a, b)
                    if d <= inflate:
                        grid[i, j] = True

        # Mark the outer frame as obstacles to keep agents inside the world
        grid[0, :] = True
        grid[-1, :] = True
        grid[:, 0] = True
        grid[:, -1] = True

        self._grid = grid
        self._grid_x = grid_x
        self._grid_y = grid_y


    def _plan_paths_for_all_agents(self, env, skip_smoothing: bool = False, inflate_scale: float = 1.0):
        """Run A* for each agent and store (optionally smoothed) paths in world coords."""
        assert self._grid is not None
        grid = self._grid
        H, W = grid.shape

        self._paths_world.clear()
        self._waypoint_indices.clear()
        self._replan_attempted = {i: False for i in range(env.n_agents)}

        positions = np.asarray(env.positions, dtype=float)
        goals = np.asarray(env.goals, dtype=float)

        for i in range(env.n_agents):
            start_ij = self._world_to_grid(positions[i], env.world_size, H, W)
            goal_ij = self._world_to_grid(goals[i], env.world_size, H, W)

            # If start/goal landed inside inflated obstacles, project to nearest free cell
            if grid[start_ij]:
                start_ij = self._nearest_free_cell(grid, start_ij)
                if start_ij is None:
                    self._paths_world[i] = []
                    self._waypoint_indices[i] = 0
                    continue
            if grid[goal_ij]:
                goal_ij = self._nearest_free_cell(grid, goal_ij)
                if goal_ij is None:
                    self._paths_world[i] = []
                    self._waypoint_indices[i] = 0
                    continue

            path_cells = self._astar(grid, start_ij, goal_ij)
            if not path_cells:
                # No path found; leave empty; __call__ falls back to straight-line
                self._paths_world[i] = []
                self._waypoint_indices[i] = 0
                continue

            # Grid -> world coords
            path_world = [
                self._grid_cell_to_world(ij, env.world_size, H, W)
                for ij in path_cells
            ]

            if skip_smoothing:
                self._paths_world[i] = path_world
            else:
                # Path smoothing via line-of-sight segments; skip if clearance is tight
                inflate = self._compute_inflate(env, inflate_scale)
                if self._min_dist_path_to_walls(path_world, env) <= inflate:
                    self._paths_world[i] = path_world
                else:
                    path_smooth = self._smooth_path_los(path_world, env)
                    self._paths_world[i] = path_smooth
            self._waypoint_indices[i] = 0


    def _plan_path_for_agent(
        self,
        env,
        agent_idx: int,
        grid: Optional[np.ndarray] = None,
        skip_smoothing: bool = False,
        inflate_scale: float = 1.0,
    ) -> None:
        """Plan (or re-plan) a path for a single agent using the provided grid.

        This is used by the "stuck" logic so we can replan only for agent i,
        optionally on a temporary grid that includes dynamic obstacles.
        """
        if grid is None:
            if self._grid is None:
                self._build_grid(env, inflate_scale=inflate_scale)
            grid = self._grid
        assert grid is not None

        H, W = grid.shape
        positions = np.asarray(env.positions, dtype=float)
        goals = np.asarray(env.goals, dtype=float)

        i = int(agent_idx)
        start_ij = self._world_to_grid(positions[i], env.world_size, H, W)
        goal_ij = self._world_to_grid(goals[i], env.world_size, H, W)

        # If start/goal landed inside obstacles, project to nearest free cell
        if grid[start_ij]:
            start_ij = self._nearest_free_cell(grid, start_ij)
            if start_ij is None:
                self._paths_world[i] = []
                self._waypoint_indices[i] = 0
                return
        if grid[goal_ij]:
            goal_ij = self._nearest_free_cell(grid, goal_ij)
            if goal_ij is None:
                self._paths_world[i] = []
                self._waypoint_indices[i] = 0
                return

        path_cells = self._astar(grid, start_ij, goal_ij)
        if not path_cells:
            self._paths_world[i] = []
            self._waypoint_indices[i] = 0
            return

        path_world = [
            self._grid_cell_to_world(ij, env.world_size, H, W)
            for ij in path_cells
        ]

        if skip_smoothing:
            self._paths_world[i] = path_world
        else:
            inflate = self._compute_inflate(env, inflate_scale)
            if self._min_dist_path_to_walls(path_world, env) <= inflate:
                self._paths_world[i] = path_world
            else:
                self._paths_world[i] = self._smooth_path_los(path_world, env)

        self._waypoint_indices[i] = 0


    def _add_inactive_agents_to_grid(
        self,
        grid: np.ndarray,
        env,
        inactive_mask: np.ndarray,
        exclude_agent: Optional[int] = None,
        inflate_scale: float = 1.0,
    ) -> None:

        H, W = grid.shape
        world_size = env.world_size
        xs = np.linspace(-world_size, world_size, W)
        ys = np.linspace(-world_size, world_size, H)

        cell = (2.0 * world_size) / max(self.astar_cfg.grid_size - 1, 1)
        # Use a conservative radius so A* will route around a parked robot.
        # (distance between robot centers must exceed ~2*robot_radius to avoid collision)
        r = max(
            2.5 * env.robot_radius,
            self.opt_cfg.goal_block_distance_factor * env.robot_radius,
        ) + 0.5 * cell

        positions = np.asarray(env.positions, dtype=float)
        for j in range(env.n_agents):
            if not inactive_mask[j]:
                continue
            if exclude_agent is not None and j == int(exclude_agent):
                continue

            cx, cy = float(positions[j, 0]), float(positions[j, 1])
            xmin, xmax = cx - r, cx + r
            ymin, ymax = cy - r, cy + r

            j_min = max(0, int(np.searchsorted(xs, xmin, side="left")))
            j_max = min(W, int(np.searchsorted(xs, xmax, side="right")))
            i_min = max(0, int(np.searchsorted(ys, ymin, side="left")))
            i_max = min(H, int(np.searchsorted(ys, ymax, side="right")))

            r2 = r * r
            for ii in range(i_min, i_max):
                y = ys[ii]
                dy2 = (y - cy) * (y - cy)
                # quick reject by y
                if dy2 > r2:
                    continue
                for jj in range(j_min, j_max):
                    x = xs[jj]
                    dx = x - cx
                    if dx * dx + dy2 <= r2:
                        grid[ii, jj] = True

    def _astar(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Standard 8-connected A* on a 2D grid; grid[i,j] == True means obstacle."""
        H, W = grid.shape
        si, sj = start
        gi, gj = goal

        if grid[si, sj] or grid[gi, gj]:
            return []

        # 8-connected neighbors
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

        def h(i, j):
            return np.hypot(i - gi, j - gj)

        open_heap = []
        heapq.heappush(open_heap, (h(si, sj), 0.0, (si, sj)))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score = {(si, sj): 0.0}
        visited = set()

        while open_heap:
            f, g, (i, j) = heapq.heappop(open_heap)
            if (i, j) in visited:
                continue
            visited.add((i, j))

            if (i, j) == (gi, gj):
                # Reconstruct path
                path = [(i, j)]
                while (i, j) in came_from:
                    (i, j) = came_from[(i, j)]
                    path.append((i, j))
                path.reverse()
                return path

            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                if not (0 <= ni < H and 0 <= nj < W):
                    continue
                if grid[ni, nj]:
                    continue
                # prevent diagonal corner-cutting
                if di != 0 and dj != 0:
                    if grid[i + di, j] or grid[i, j + dj]:
                        continue


                step_cost = np.hypot(di, dj)
                tentative_g = g + step_cost

                if (ni, nj) not in g_score or tentative_g < g_score[(ni, nj)]:
                    g_score[(ni, nj)] = tentative_g
                    came_from[(ni, nj)] = (i, j)
                    f_new = tentative_g + h(ni, nj)
                    heapq.heappush(open_heap, (f_new, tentative_g, (ni, nj)))

        return []  # no path found

    # ------------------------------------------------------------------
    # Path smoothing by line-of-sight
    # ------------------------------------------------------------------

    def _smooth_path_los(
        self,
        path_world: List[np.ndarray],
        env,
        n_samples: int = 10,
    ) -> List[np.ndarray]:
        """
        Line-of-sight smoothing: keep endpoints of maximal straight segments
        that don't intersect inflated obstacles.
        """
        if len(path_world) <= 2:
            return path_world

        smoothed: List[np.ndarray] = []
        i = 0
        while i < len(path_world) - 1:
            smoothed.append(path_world[i])
            j = len(path_world) - 1
            found = False
            while j > i + 1:
                if self._segment_is_free(path_world[i], path_world[j], env, n_samples):
                    i = j
                    found = True
                    break
                j -= 1
            if not found:
                i += 1
        smoothed.append(path_world[-1])
        return smoothed

    def _segment_is_free(
        self,
        p: np.ndarray,
        q: np.ndarray,
        env,
        n_samples: int,
    ) -> bool:
        """
        Check if segment pq is free of collisions with walls, inflated by
        obstacle_inflate_radius.
        """
        walls = getattr(env, "walls", [])
        if not walls:
            return True

        has_env_dist = hasattr(env, "_distance_point_to_segment")
        cell = (2.0 * env.world_size) / max(self.astar_cfg.grid_size - 1, 1)
        buffer = 0.5 * np.sqrt(2.0) * cell
        inflate = env.robot_radius + self.astar_cfg.obstacle_inflate_radius + buffer

        seg_len = float(np.linalg.norm(q - p))

        # choose step small enough to not "jump over" an inflated wall
        step = min(0.5 * cell, 0.25 * max(inflate, 1e-6))
        n = max(n_samples, int(np.ceil(seg_len / max(step, 1e-6))) + 1)

        for t in np.linspace(0.0, 1.0, n):
            r = (1.0 - t) * p + t * q
            for w in walls:
                a = np.array([w.x1, w.y1], dtype=float)
                b = np.array([w.x2, w.y2], dtype=float)
                d = env._distance_point_to_segment(r, a, b) if has_env_dist else \
                    self._distance_point_to_segment_local(r, a, b)
                if d <= inflate:
                    return False
        return True

    def _segment_is_free_with_inactive(
        self,
        p: np.ndarray,
        q: np.ndarray,
        env,
        inactive_mask: np.ndarray,
        exclude_agent: Optional[int],
        n_samples: int,
    ) -> bool:
        """LOS check against walls *and* inactive (goal-reached) agents.

        This is used after a stuck-triggered replan where we treated inactive agents
        as temporary A* obstacles. Without this, the waypoint-advancing heuristic
        can "skip" through those temporary obstacles because it only checks walls.
        """
        if not self._segment_is_free(p, q, env, n_samples=n_samples):
            return False

        if inactive_mask is None or not np.any(inactive_mask):
            return True

        positions = np.asarray(env.positions, dtype=float)
        world_size = env.world_size
        cell = (2.0 * world_size) / max(self.astar_cfg.grid_size - 1, 1)

        # Forbidden region for the *center* of the moving robot near a parked robot.
        # Centers must be separated by ~2*robot_radius to avoid overlap; add a small buffer.
        r = max(2.1 * env.robot_radius, self.opt_cfg.goal_block_distance_factor * env.robot_radius) + 0.25 * cell
        r2 = r * r

        seg_len = float(np.linalg.norm(q - p))
        step = min(0.5 * cell, 0.25 * max(r, 1e-6))
        n = max(n_samples, int(np.ceil(seg_len / max(step, 1e-6))) + 1)

        for t in np.linspace(0.0, 1.0, n):
            x = (1.0 - t) * p + t * q
            for j in range(env.n_agents):
                if not inactive_mask[j]:
                    continue
                if exclude_agent is not None and j == int(exclude_agent):
                    continue
                d = x - positions[j]
                if float(d[0] * d[0] + d[1] * d[1]) <= r2:
                    return False
        return True

    # ------------------------------------------------------------------
    # Helpers: world/grid conversion & distance
    # ------------------------------------------------------------------

    @staticmethod
    def _world_to_grid(
        pos: np.ndarray,
        world_size: float,
        H: int,
        W: int,
    ) -> Tuple[int, int]:
        x, y = float(pos[0]), float(pos[1])
        u = (x + world_size) / (2 * world_size + 1e-9)
        v = (y + world_size) / (2 * world_size + 1e-9)
        j = int(np.clip(round(u * (W - 1)), 0, W - 1))
        i = int(np.clip(round(v * (H - 1)), 0, H - 1))
        return i, j

    @staticmethod
    def _grid_cell_to_world(
        ij: Tuple[int, int],
        world_size: float,
        H: int,
        W: int,
    ) -> np.ndarray:
        i, j = ij
        x = -world_size + (2 * world_size) * (j / max(W - 1, 1))
        y = -world_size + (2 * world_size) * (i / max(H - 1, 1))
        return np.array([x, y], dtype=float)

    @staticmethod
    def _distance_point_to_segment_local(
        p: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            return float(np.linalg.norm(p - a))
        t = float(np.dot(p - a, ab) / denom)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def _compute_inflate(self, env, inflate_scale: float = 1.0) -> float:
        cell = (2.0 * env.world_size) / max(self.astar_cfg.grid_size - 1, 1)
        buffer = 0.5 * np.sqrt(2.0) * cell
        return env.robot_radius + inflate_scale * (self.astar_cfg.obstacle_inflate_radius + buffer)

    def _min_dist_path_to_walls(self, path_pts: List[np.ndarray], env) -> float:
        walls = getattr(env, "walls", [])
        if not walls or not path_pts:
            return np.inf
        has_env_dist = hasattr(env, "_distance_point_to_segment")
        dmin = np.inf
        for p in path_pts:
            for w in walls:
                a = np.array([w.x1, w.y1], dtype=float)
                b = np.array([w.x2, w.y2], dtype=float)
                d = env._distance_point_to_segment(p, a, b) if has_env_dist else self._distance_point_to_segment_local(p, a, b)
                if d < dmin:
                    dmin = d
        return float(dmin)
    
    @staticmethod
    def _nearest_free_cell(
        grid: np.ndarray,
        ij: Tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        """
        Find the nearest non-obstacle cell to ij (inclusive).
        Returns None if the grid is fully occupied.
        """
        if not grid[ij]:
            return ij

        free_i, free_j = np.where(~grid)
        if free_i.size == 0:
            return None

        di = free_i - ij[0]
        dj = free_j - ij[1]
        dist2 = di * di + dj * dj
        idx = np.argmin(dist2)
        return int(free_i[idx]), int(free_j[idx])

    # ------------------------------------------------------------------
    # Local optimizer (collision avoidance)
    # ------------------------------------------------------------------

    def _local_optimize(self, env, u_nom: np.ndarray, active_mask=None) -> np.ndarray:
        """
        One- or few-step gradient-like refinement of nominal velocities to
        reduce agent-agent collisions while staying close to u_nom.

        J_i(u_i) ≈ 0.5 ||u_i - u_nom_i||^2
                + sum_j 0.5 * k_coll * max(0, d_safe - ||x_i' - x_j'||)^2

        with x_i' = x_i + dt * u_i, x_j' = x_j + dt * u_j.
        """

        if active_mask is None:
            active_mask = np.ones(env.n_agents, dtype=bool)

        positions = np.asarray(env.positions, dtype=float)
        N = env.n_agents
        u = u_nom.copy()

        # Active mask: True => we optimize u[i]; False => agent i is "frozen" (u[i]=0)
        if active_mask is None:
            active_mask = np.ones(N, dtype=bool)
        else:
            active_mask = np.asarray(active_mask, dtype=bool)
            if active_mask.shape != (N,):
                raise ValueError(f"active_mask must have shape ({N},), got {active_mask.shape}")

        # Freeze inactive agents (treat as static obstacles)
        u[~active_mask] = 0.0
        u_nom = u_nom.copy()
        u_nom[~active_mask] = 0.0

        dt = self.opt_cfg.dt if self.opt_cfg.dt is not None else env.dt
        max_speed = self.opt_cfg.max_speed if self.opt_cfg.max_speed is not None else env.max_speed
        robot_radius = env.robot_radius
        d_safe = self.opt_cfg.safe_distance_factor * robot_radius
        wall_clearance = self.opt_cfg.wall_clearance_factor * robot_radius

        # Early out: skip optimization if disabled
        if (d_safe <= 0.0 and wall_clearance <= 0.0) or self.opt_cfg.max_iters <= 0:
            return u

        for _ in range(self.opt_cfg.max_iters):

            for i in range(N):

                if not active_mask[i]:
                    continue

                # Start with gradient from path-tracking term
                grad = u[i] - u_nom[i]

                # Add collision gradients
                for j in range(N):
                    if i == j:
                        continue

                    # Predicted next positions
                    p_i_next = positions[i] + dt * u[i]
                    p_j_next = positions[j] + dt * u[j]
                    diff = p_i_next - p_j_next
                    dist = np.linalg.norm(diff)

                    if dist < d_safe and dist > 1e-6:
                        # Penalty: 0.5 * k_coll * (d_safe - dist)^2
                        # grad_J wrt u_i points roughly from j to i (to push away):
                        direction = diff / dist  # from j to i
                        grad += -self.opt_cfg.k_coll * (d_safe - dist) * direction 

                # Add wall / obstacle repulsion based on predicted next position
                if wall_clearance > 0.0:
                    p_i_next = positions[i] + dt * u[i]
                    for w in getattr(env, "walls", []):
                        a = np.array([w.x1, w.y1], dtype=float)
                        b = np.array([w.x2, w.y2], dtype=float)
                        proj, dist = self._project_to_segment(p_i_next, a, b)
                        if dist < wall_clearance and dist > 1e-6:
                            direction = (p_i_next - proj) / dist
                            grad += -self.opt_cfg.k_wall * (wall_clearance - dist) * direction  

                # Gradient descent step (step size = 1.0 for now)
                alpha = 1.0
                u[i] = u[i] - alpha * grad

                # Clip to max_speed
                speed = np.linalg.norm(u[i])
                if speed > max_speed:
                    u[i] = u[i] * (max_speed / (speed + 1e-9))
        u[~active_mask] = 0.0

        return u

    @staticmethod
    def _project_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Project point p onto segment ab and return (projection, distance).
        """
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            proj = a
        else:
            t = float(np.dot(p - a, ab) / denom)
            t = max(0.0, min(1.0, t))
            proj = a + t * ab
        diff = p - proj
        return proj, float(np.linalg.norm(diff))
