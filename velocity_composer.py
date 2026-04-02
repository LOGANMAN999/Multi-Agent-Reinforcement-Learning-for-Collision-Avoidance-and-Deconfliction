import math
import numpy as np


def compose_velocity(
    a_star_vel: np.ndarray,  # (2,) A* preferred velocity
    delta_speed: float,       # RL speed correction (additive)
    delta_heading: float,     # RL heading correction (additive, radians)
    agent_heading: float,     # current agent heading (radians), used as fallback
) -> np.ndarray:
    a_star_vel = np.asarray(a_star_vel, dtype=float)
    a_star_speed = float(np.linalg.norm(a_star_vel))

    if a_star_speed < 1e-6:
        # Agent at goal or A* has stopped; use current heading as base
        base_heading = float(agent_heading)
        base_speed = 0.0
    else:
        base_heading = math.atan2(float(a_star_vel[1]), float(a_star_vel[0]))
        base_speed = a_star_speed

    final_speed = base_speed + float(delta_speed)
    final_heading = base_heading + float(delta_heading)

    v_final = np.array([
        final_speed * math.cos(final_heading),
        final_speed * math.sin(final_heading),
    ], dtype=float)

    return v_final


def batch_compose_velocity(
    a_star_vels: np.ndarray,     # (N, 2)
    delta_speeds: np.ndarray,    # (N,)
    delta_headings: np.ndarray,  # (N,)
    agent_headings: np.ndarray,  # (N,)
) -> np.ndarray:

    a_star_vels = np.asarray(a_star_vels, dtype=float)
    delta_speeds = np.asarray(delta_speeds, dtype=float)
    delta_headings = np.asarray(delta_headings, dtype=float)
    agent_headings = np.asarray(agent_headings, dtype=float)

    N = a_star_vels.shape[0]
    v_finals = np.zeros((N, 2), dtype=float)

    for i in range(N):
        v_finals[i] = compose_velocity(
            a_star_vels[i],
            delta_speeds[i],
            delta_headings[i],
            agent_headings[i],
        )

    return v_finals
