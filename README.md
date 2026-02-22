# Multi-Agent Collision Avoidance (A* Expert → GNN‑GRU Imitation → PPO Fine‑Tuning)

A research sandbox for **multi-robot navigation and collision avoidance** in 2D grid worlds with walls/obstacles. The project implements an **expert planner/controller** and trains a **graph neural network with recurrence** to imitate (and optionally improve via PPO).

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

> Note: filenames reflect the current snapshot of the codebase. If you reorganize folders, update these paths.

---

## Setup

### Requirements
- Python **3.10+**
- `numpy`, `matplotlib`
- **PyTorch**
- **PyTorch Geometric** (for GNN training)

Install basics:
```bash
pip install numpy matplotlib torch
```

Install PyTorch Geometric following the official instructions for your specific PyTorch/CUDA combo.

---

## Quickstart

### 1) Generate expert demonstrations
This produces `.npz` episode files containing state, geometry, graph info, and expert actions.

```bash
python dataset_generator.py \
  --out_dir datasets/il_dataset \
  --n_episodes 1000 \
  --max_steps 300 \
  --seed 0
```

### 2) Build the student training dataset (GRU sequences)
Creates train/val/test shards plus normalization stats (`stats.json`).

```bash
python build_student_dataset_gru.py \
  --dataset_root datasets/il_dataset \
  --out_dir datasets/il_dataset/processed_student_gru_v1 \
  --sequence_mode \
  --seq_len 32 \
  --stride 16
```

### 3) Train the student policy (imitation learning)
```bash
python train_student_gnn_gru.py \
  --data_dir datasets/il_dataset/processed_student_gru_v1 \
  --epochs 40 \
  --batch_size 32 \
  --lr 3e-4
```

### 4) Evaluate
If using the included eval harness:
```bash
python rl_eval.py \
  --ckpt datasets/il_dataset/processed_student_gru_v1/checkpoints/best.pt \
  --stats_json datasets/il_dataset/processed_student_gru_v1/stats.json \
  --episodes 50 \
  --max_steps 400 \
  --out_dir runs/eval_student_seed0
```

---

## Optional: PPO fine-tuning (recurrent)
Initialize from the supervised checkpoint and run PPO updates.

```bash
python train_student_rl_ppo_gru.py \
  --supervised_ckpt datasets/il_dataset/processed_student_gru_v1/checkpoints/best.pt \
  --stats_json datasets/il_dataset/processed_student_gru_v1/stats.json \
  --run_dir runs/rl_ppo_gru_v1 \
  --updates 200 \
  --horizon 256 \
  --seq_len 32 \
  --batch_size 4 \
  --n_agents 8
```

Then evaluate:
```bash
python rl_eval.py \
  --ckpt runs/rl_ppo_gru_v1/checkpoints/best_rl.pt \
  --stats_json datasets/il_dataset/processed_student_gru_v1/stats.json \
  --episodes 50 \
  --out_dir runs/rl_ppo_gru_v1/eval_seed0
```

---

## Observation model (high level)

Each timestep is a **graph**:
- **Nodes** = agents (per-agent features include goal-relative quantities, normalized LiDAR, and an activity mask).
- **Edges** = neighbor relations within a radius (edge features include relative displacement, distance, and occlusion flags).

A GRU maintains **per-agent memory**, improving performance under partial observability and multi-agent interaction.

---

## Metrics to watch
Typical evaluation reports include:
- **Success rate** (all agents reach goals)
- **Wall collisions**
- **Inter-agent collisions**
- **Episode length / time-to-goal**

---

## Roadmap (suggested)
- Curriculum over obstacle density / agent count
- Vectorized environments for faster PPO
- More robust deadlock resolution (expert + learned)
- Standardized benchmark suite and plots

---

## License
Choose a license before publishing (MIT/BSD-3/Apache-2.0 are common for research code). Add a `LICENSE` file when ready.

## Citation
If you use this code in academic work, add a short citation block here (paper/arXiv link) once available.
