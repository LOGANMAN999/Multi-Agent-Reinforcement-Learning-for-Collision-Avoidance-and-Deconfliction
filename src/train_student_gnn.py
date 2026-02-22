"""Train the StudentGNN on processed PyG shards.

Assumes the dataset was produced under:
  datasets/il_dataset/processed_student_v1/
    train/shard_*.pt
    val/shard_*.pt
    stats.json

Each shard file is a torch-saved Python list of torch_geometric.data.Data objects.
Each Data must have: x, edge_index, edge_attr, y, mask.

This script:
  - builds the 3-layer gated MPNN (StudentGNN)
  - trains with masked node-level regression loss
  - logs train/val loss per epoch to CSV
  - prints losses to terminal after each epoch

Usage:
  python train_student_gnn.py --data_dir datasets/il_dataset/processed_student_v1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Tuple
from collections import defaultdict

import numpy as np
import torch

# PyG Batch helper
from torch_geometric.loader import DataLoader
try:
    from torch_geometric.data import Batch
except Exception as e:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required. Install PyTorch Geometric and dependencies."
    ) from e

# Local model import (works if this file is alongside gnn_student_model.py)
_THIS_DIR = Path(__file__).resolve().parent
import sys
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from gnn_student_model import StudentGNN, load_cfg_from_stats  # noqa: E402


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_shards(split_dir: Path) -> List[Path]:
    shards = sorted(split_dir.glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.pt files found in {split_dir}")
    return shards


def load_stats(stats_path: Path) -> dict:
    with stats_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_feature_dims(stats: dict) -> Tuple[int, int]:
    # Prefer stored means if present
    if "x_mean" in stats and "edge_mean" in stats:
        node_dim = len(stats["x_mean"])
        edge_dim = len(stats["edge_mean"])
        return int(node_dim), int(edge_dim)

    # Fallback: sometimes names differ
    for xk in ("node_mean", "x_mu"):
        for ek in ("edge_mean", "edge_mu"):
            if xk in stats and ek in stats:
                return int(len(stats[xk])), int(len(stats[ek]))

    raise KeyError(
        "Could not infer node/edge feature dims from stats.json. "
        "Expected keys like 'x_mean' and 'edge_mean'."
    )


def batch_iter(
    shard_paths: List[Path],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> Iterable[Batch]:
    """Yield PyG Batches by streaming shard files.

    This avoids loading the entire dataset into memory.
    """
    rng = np.random.default_rng(seed)

    shard_order = list(range(len(shard_paths)))
    if shuffle:
        rng.shuffle(shard_order)

    buf = []
    for idx in shard_order:
        shard = shard_paths[idx]
        data_list = torch.load(shard, map_location="cpu")

        # data_list is a Python list[Data]
        if shuffle:
            perm = rng.permutation(len(data_list))
            data_list = [data_list[i] for i in perm]

        for d in data_list:
            buf.append(d)
            if len(buf) >= batch_size:
                yield Batch.from_data_list(buf)
                buf = []

    if buf:
        yield Batch.from_data_list(buf)


def loss_sum_and_count(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss: str,
    huber_delta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (loss_sum, count) where count is #active nodes.

    We compute a per-node loss and then sum over active nodes so epoch averages are
    properly weighted even when batches contain varying numbers of agents.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()

    if mask.numel() == 0 or mask.sum().item() == 0:
        z = pred.new_tensor(0.0)
        c = pred.new_tensor(0.0)
        return z, c

    # Per-node loss: average over action dimensions
    if loss == "mse":
        per_node = ((pred - target) ** 2).mean(dim=-1)
    elif loss == "huber":
        per_elem = torch.nn.functional.huber_loss(pred, target, delta=huber_delta, reduction="none")
        per_node = per_elem.mean(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss}")

    per_node_active = per_node[mask]
    loss_sum = per_node_active.sum()
    count = pred.new_tensor(float(per_node_active.numel()))
    return loss_sum, count


def masked_sse_and_count(pred, y, mask):
    """
    pred: (num_nodes, 2)
    y:    (num_nodes, 2)
    mask: (num_nodes,) bool or 0/1

    Returns:
      sse: sum of squared error over active nodes and both dims
      count: number of active scalars (= active_nodes * 2)
    """
    if mask is None:
        err = pred - y
        sse = (err * err).sum()
        count = y.numel()
        return sse, count

    m = mask.bool()
    if m.sum() == 0:
        return pred.new_tensor(0.0), pred.new_tensor(0.0)

    err = pred[m] - y[m]          # (active_nodes, 2)
    sse = (err * err).sum()
    count = err.numel()           # active_nodes * 2
    return sse, count

@torch.no_grad()
def per_episode_mse_distribution(model, split_dir, batch_size=64, device="cuda"):
    """
    Returns:
      mse_by_ep: dict episode_str -> MSE (mean over active nodes and action dims)
      all_mse:   list of per-episode MSE values
      overall_mse: node-weighted overall MSE across the entire split

    Expects each Data to have either:
      - data.meta = {"episode": str, "t": int}   (PyG batches into batch.meta["episode"])
        OR
      - data.episode = str
    """
    model.eval()
    split_dir = Path(split_dir)
    shard_paths = sorted(split_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.pt found in {split_dir}")

    sse_by_ep = defaultdict(float)
    cnt_by_ep = defaultdict(float)
    total_sse = 0.0
    total_cnt = 0.0

    def _get_episode_key(batch, g: int):
        # Preferred: explicit attribute
        if hasattr(batch, "episode"):
            try:
                return batch.episode[g]
            except Exception:
                pass

        # Next: meta dict batched by key
        if hasattr(batch, "meta") and isinstance(batch.meta, dict):
            if "episode" in batch.meta:
                return batch.meta["episode"][g]

        # Fallback
        return "UNKNOWN_EPISODE"

    for sp in shard_paths:
        data_list = torch.load(sp)  # list[Data]
        loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)  # (num_nodes_total, 2)

            # batch.ptr gives node slice boundaries per graph in the batch:
            # nodes for graph g are [ptr[g], ptr[g+1])
            ptr = batch.ptr
            num_graphs = int(batch.num_graphs)

            for g in range(num_graphs):
                start = int(ptr[g].item()) if torch.is_tensor(ptr[g]) else int(ptr[g])
                end = int(ptr[g + 1].item()) if torch.is_tensor(ptr[g + 1]) else int(ptr[g + 1])
                if end <= start:
                    continue

                pred_g = pred[start:end]
                y_g = batch.y[start:end]
                mask_g = batch.mask[start:end] if hasattr(batch, "mask") else None

                sse, cnt = masked_sse_and_count(pred_g, y_g, mask_g)

                # cnt may be tensor or python number
                cnt_val = float(cnt.item()) if torch.is_tensor(cnt) else float(cnt)
                if cnt_val <= 0.0:
                    continue

                sse_val = float(sse.item()) if torch.is_tensor(sse) else float(sse)
                ep = _get_episode_key(batch, g)

                sse_by_ep[ep] += sse_val
                cnt_by_ep[ep] += cnt_val
                total_sse += sse_val
                total_cnt += cnt_val

    mse_by_ep = {ep: (sse_by_ep[ep] / max(cnt_by_ep[ep], 1e-12)) for ep in sse_by_ep}
    all_mse = list(mse_by_ep.values())
    overall_mse = total_sse / max(total_cnt, 1e-12)
    return mse_by_ep, all_mse, overall_mse

def summarize(name, arr):
    arr = np.array(arr, dtype=float)
    print(f"{name}: n_eps={len(arr)} mean={arr.mean():.6f} median={np.median(arr):.6f} "
          f"p90={np.quantile(arr,0.9):.6f} max={arr.max():.6f}")


@torch.no_grad()
def zero_predictor_mse(split_dir, batch_size=64, device="cuda"):
    """
    Computes MSE of predicting 0 for y, masked by active nodes.
    """
    split_dir = Path(split_dir)
    shard_paths = sorted(split_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.pt found in {split_dir}")

    total_sse = 0.0
    total_cnt = 0.0

    for sp in shard_paths:
        data_list = torch.load(sp)
        loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
        for batch in loader:
            batch = batch.to(device)
            y = batch.y
            pred = torch.zeros_like(y)

            mask = batch.mask if hasattr(batch, "mask") else None
            sse, cnt = masked_sse_and_count(pred, y, mask)
            total_sse += float(sse.item())
            total_cnt += float(cnt)

    return total_sse / max(total_cnt, 1.0)



def append_csv_row(csv_path: Path, fieldnames: List[str], row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        type=str,
        default="datasets/il_dataset/processed_student_v1",
        help="Processed dataset directory containing train/ val/ stats.json",
    )
    ap.add_argument("--epochs", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"])
    ap.add_argument("--huber_delta", type=float, default=1.0)
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)
    ap.add_argument(
        "--log_csv",
        type=str,
        default=None,
        help="CSV path for epoch logs (default: <data_dir>/training_log.csv)",
    )

    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    stats_path = data_dir / "stats.json"

    ckpt_dir = Path(data_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_epoch = -1

    if not stats_path.exists():
        raise FileNotFoundError(f"stats.json not found at {stats_path}")

    stats = load_stats(stats_path)
    node_dim, edge_dim = infer_feature_dims(stats)

    # Build model
    cfg = load_cfg_from_stats(stats_path, node_dim=node_dim, edge_dim=edge_dim, hidden_dim=128)
    model = StudentGNN(cfg)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("bias") or ("norm" in name.lower()) or ("layernorm" in name.lower()):
            no_decay.append(p)
        else:
            decay.append(p)

    """
    optimizer = torch.optim.Adam(
        [
            {"params": decay, "weight_decay": 1e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    """


    # Shards
    train_shards = discover_shards(train_dir)
    val_shards = discover_shards(val_dir)

    # Logging
    log_csv = Path(args.log_csv) if args.log_csv else (data_dir / "training_log.csv")
    fieldnames = ["epoch", "train_loss", "val_loss", "lr"]

    set_seed(args.seed)

    print(f"Data dir: {data_dir}")
    print(f"Train shards: {len(train_shards)} | Val shards: {len(val_shards)}")
    print(f"Node dim: {node_dim} | Edge dim: {edge_dim}")
    print(f"Device: {device}")

    # ---- Optional: baseline once before training ----
    train_zero = zero_predictor_mse(data_dir / "train", batch_size=args.batch_size, device=device)
    val_zero   = zero_predictor_mse(data_dir / "val",   batch_size=args.batch_size, device=device)
    print(f"[baseline] zero-predictor MSE train={train_zero:.6f} val={val_zero:.6f}")


    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss_sum = 0.0
        train_count = 0.0

        for batch in batch_iter(train_shards, args.batch_size, shuffle=True, seed=args.seed + epoch * 1000):
            batch = batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss_sum_t, count_t = loss_sum_and_count(
                pred,
                batch.y,
                batch.mask,
                loss=args.loss,
                huber_delta=args.huber_delta,
            )

            # If a batch has no active nodes, skip
            if count_t.item() == 0:
                continue

            loss = loss_sum_t / count_t
            if not torch.isfinite(loss):
                continue

            loss.backward()
            if args.clip_grad_norm and args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            train_loss_sum += float(loss_sum_t.item())
            train_count += float(count_t.item())

        train_loss = train_loss_sum / max(train_count, 1e-9)

        # ---- Val ----
        model.eval()
        val_loss_sum = 0.0
        val_count = 0.0
        with torch.no_grad():
            for batch in batch_iter(val_shards, args.batch_size, shuffle=False, seed=args.seed + 999):
                batch = batch.to(device)
                pred = model(batch)
                loss_sum_t, count_t = loss_sum_and_count(
                    pred,
                    batch.y,
                    batch.mask,
                    loss=args.loss,
                    huber_delta=args.huber_delta,
                )

                if count_t.item() == 0:
                    continue

                val_loss_sum += float(loss_sum_t.item())
                val_count += float(count_t.item())

        val_loss = val_loss_sum / max(val_count, 1e-9)

        lr = optimizer.param_groups[0]["lr"]

        # Print to terminal
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={lr:.2e}")

        # ---- Optional: per-episode distributions every K epochs ----
        K = 10
        if (epoch % K == 0) or (epoch == args.epochs):
            mse_by_ep_val, all_val, overall_val = per_episode_mse_distribution(
                model, data_dir / "val", batch_size=args.batch_size, device=device
            )
            mse_by_ep_tr, all_tr, overall_tr = per_episode_mse_distribution(
                model, data_dir / "train", batch_size=args.batch_size, device=device
            )

            summarize("TRAIN per-episode MSE", all_tr)
            summarize("VAL   per-episode MSE", all_val)
            print(f"[dist] overall train MSE (node-weighted) = {overall_tr:.6f}")
            print(f"[dist] overall val   MSE (node-weighted) = {overall_val:.6f}")


        # Always save "last"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "cfg": cfg.__dict__,  # if cfg is a dataclass/config object
            },
            ckpt_dir / "last.pt",
        )

        # Save best by val
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "cfg": cfg.__dict__,
                },
                ckpt_dir / "best.pt",
            )
            print(f"[checkpoint] New best val {best_val:.6f} at epoch {best_epoch}")


        # Log to CSV
        append_csv_row(
            log_csv,
            fieldnames,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
            },
        )

    print("\n=== Final per-episode distribution summary (best checkpoint may differ) ===")
    mse_by_ep_val, all_val, overall_val = per_episode_mse_distribution(
        model, data_dir / "val", batch_size=args.batch_size, device=device
    )
    summarize("VAL per-episode MSE", all_val)
    print(f"[final] overall val MSE (node-weighted) = {overall_val:.6f}")

    print(f"Done. Wrote logs to: {log_csv}")


if __name__ == "__main__":
    main()
