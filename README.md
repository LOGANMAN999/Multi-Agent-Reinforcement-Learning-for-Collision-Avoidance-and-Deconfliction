# Multi-Agent Collision Avoidance (A* Expert → GNN‑GRU Imitation → PPO Fine‑Tuning)

A research sandbox for **multi-robot navigation and collision avoidance** in 2D worlds with walls/obstacles. The project implements an **expert planner/controller** and trains a **graph neural network with recurrence** to imitate (and optionally improve via PPO).

**Pipeline**
1. **Expert demonstrations**: global planning via **A\*** on an occupancy grid + a **local collision-avoidance controller**.  
2. **Dataset generation**: rollout the expert and log **graph-structured observations** and expert actions.  
3. **Student policy (IL)**: a **message-passing GNN + per-agent GRU** to output per-agent velocity actions.  
4. **(Optional) PPO fine-tuning**: initialize from the supervised student and refine online with recurrent PPO.

---

## What’s inside

- `sim_env.py` — 2D multi-agent environment + LiDAR-style sensing + wall intersection utilities  
- `astar_global_local.py` — expert A* global planner + local controller  
- `map_generation.py` — random map/spec generation  
- `dataset_generator.py` — generates expert rollouts into `.npz` episodes  
- `build_student_dataset_gru.py` — converts episodes → normalized training shards (sequence mode for GRU)  
- `gnn_student_model_gru.py` — student GNN‑GRU model  
- `train_student_gnn_gru.py` — supervised imitation training  
- `rl_*_gru.py`, `train_student_rl_ppo_gru.py`, `rl_eval.py` — recurrent PPO fine-tuning + evaluation



---

