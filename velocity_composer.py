"""
# Phase 0 Header
# File: src/RL_stack/velocity_composer.py
# Dependencies: numpy, math
# Responsibility: Combine A* base velocity with RL speed/heading corrections
#   to produce a final velocity command in world-frame coordinates.
# Does NOT: interact with the environment, apply safety filtering, or learn.
"""

import math
import numpy as np


def compose_velocity(
    a_star_vel: np.ndarray,  # (2,) A* preferred velocity
    delta_speed: float,       # RL speed correction (additive)
    delta_heading: float,     # RL heading correction (additive, radians)
    agent_heading: float,     # current agent heading (radians), used as fallback
) -> np.ndarray:
    """
    Combine A* base velocity with RL corrections in speed/heading space.

    Rotational equivariance argument:
      Applying delta_heading in heading space means the same learned correction
      (e.g., "turn left 10 degrees") applies identically regardless of which
      direction the agent is heading. If we instead applied delta in world-frame
      xy coordinates, a learned correction [0.1, 0] (rightward in world frame)
      would mean "turn right" for a north-facing agent but "turn left" for a
      south-facing agent — not rotationally equivariant. Working in speed/heading
      space ensures the RL policy learns directional corrections that are
      independent of absolute orientation.

    Math:
      a_star_speed = ||a_star_vel||
      a_star_heading = atan2(a_star_vel[1], a_star_vel[0])
        (fallback: agent_heading when a_star_speed ≈ 0)
      final_speed = a_star_speed + delta_speed
      final_heading = a_star_heading + delta_heading
      v_final = [final_speed * cos(final_heading), final_speed * sin(final_heading)]

    Edge case:
      If a_star_speed ≈ 0 (agent at goal or A* has stopped it), we use
      agent_heading as the heading base so delta_heading still has a meaningful
      reference frame and the agent can still be steered by RL corrections.

    Args:
        a_star_vel:    (2,) preferred velocity from A* controller (world frame).
        delta_speed:   scalar speed correction from RL policy.
        delta_heading: scalar heading correction from RL policy (radians).
        agent_heading: current agent heading (radians), used when A* vel is ~0.

    Returns:
        v_final: (2,) final velocity in world frame.
    """
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
    """
    Vectorized compose_velocity over N agents.

    Args:
        a_star_vels:    (N, 2) A* preferred velocities.
        delta_speeds:   (N,) RL speed corrections.
        delta_headings: (N,) RL heading corrections (radians).
        agent_headings: (N,) current agent headings (radians).

    Returns:
        v_finals: (N, 2) final velocity commands in world frame.
    """
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
