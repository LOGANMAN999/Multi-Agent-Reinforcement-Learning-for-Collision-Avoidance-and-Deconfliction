"""Train StudentGNNGRU on fixed-length sequence shards.

This script is meant to be used with datasets produced by:
  python build_student_dataset_gru.py --sequence_mode ...

Dataset format:
  Each shard is a list of dict samples with keys:
    x: (L,N,Dx) float  (already normalized)
    y: (L,N,2)  float  (expert actions normalized by max_speed)
    mask: (L,N) bool   (active agents)
    edge_index: list[L] of (2,E) long tensors
    edge_attr:  list[L] of (E,De) float tensors

We batch by stacking x/y/mask to shape (B,L,N,*) and transposing the edge lists
to the model's expected structure: edge_index[t][b].
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


# Local model import (works if this file is alongside gnn_student_model_gru.py)
_THIS_DIR = Path(__file__).resolve().parent
import sys

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from gnn_student_model_gru import StudentGNNGRU, StudentGNNGRUConfig  # noqa: E402


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
    node_dim = len(stats["node_mean"]) if "node_mean" in stats else len(stats["x_mean"])
    edge_dim = len(stats["edge_mean"]) if "edge_mean" in stats else len(stats["edge_mean"])
    return int(node_dim), int(edge_dim)


def batch_iter_seq(
    shard_paths: List[Path],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> Iterable[List[Dict]]:
    """Yield lists of dict samples by streaming shard files."""
    rng = np.random.default_rng(seed)
    shard_order = list(range(len(shard_paths)))
    if shuffle:
        rng.shuffle(shard_order)

    buf: List[Dict] = []
    for idx in shard_order:
        shard = shard_paths[idx]
        samples: List[Dict] = torch.load(shard, map_location="cpu")
        if shuffle:
            perm = rng.permutation(len(samples))
            samples = [samples[i] for i in perm]

        for s in samples:
            buf.append(s)
            if len(buf) >= batch_size:
                yield buf
                buf = []

    if buf:
        yield buf


def collate_seq(samples: Sequence[Dict]) -> Dict:
    """Collate a list of sequence samples into a batch dict."""
    # x,y,mask: (L,N,*) -> (B,L,N,*)
    x = torch.stack([s["x"] for s in samples], dim=0)
    y = torch.stack([s["y"] for s in samples], dim=0)
    mask = torch.stack([s["mask"] for s in samples], dim=0)

    B, L, N, _ = x.shape

    # edge_index/edge_attr come as list[L] per sample; transpose to list[L] of list[B]
    edge_index: List[List[torch.Tensor]] = [[None for _ in range(B)] for _ in range(L)]
    edge_attr: List[List[torch.Tensor]] = [[None for _ in range(B)] for _ in range(L)]
    for b, s in enumerate(samples):
        ei_list = s["edge_index"]
        ea_list = s["edge_attr"]
        if len(ei_list) != L or len(ea_list) != L:
            raise ValueError("Inconsistent sequence length in shard")
        for t in range(L):
            edge_index[t][b] = ei_list[t]
            edge_attr[t][b] = ea_list[t]

    return {"x": x, "y": y, "mask": mask, "edge_index": edge_index, "edge_attr": edge_attr}


def loss_sum_and_count(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Masked MSE: returns (loss_sum_over_active_nodes, active_node_count)."""
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.numel() == 0 or mask.sum() == 0:
        z = pred.new_tensor(0.0)
        c = pred.new_tensor(0.0)
        return z, c

    per_node = ((pred - target) ** 2).mean(dim=-1)  # (...,N)
    per_node_active = per_node[mask]
    loss_sum = per_node_active.sum()
    count = pred.new_tensor(float(per_node_active.numel()))
    return loss_sum, count


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
    ap.add_argument("--data_dir", type=str, default="datasets/il_dataset/processed_student_v1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--gnn_hidden_dim", type=int, default=128)
    ap.add_argument("--gru_hidden_dim", type=int, default=128)
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)
    ap.add_argument("--log_csv", type=str, default=None)
    ap.add_argument("--ckpt_name", type=str, default="best.pt")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    stats_path = data_dir / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"stats.json not found at {stats_path}")

    stats = load_stats(stats_path)
    node_dim, edge_dim = infer_feature_dims(stats)

    # IMPORTANT: y in the dataset is actions/max_speed, so the target range is ~[-1,1].
    # We therefore set cfg.max_speed=1.0 so the model outputs normalized actions.
    cfg = StudentGNNGRUConfig(
        node_dim=node_dim,
        edge_dim=edge_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        num_layers=3,
        gru_hidden_dim=args.gru_hidden_dim,
        action_dim=2,
        dropout=0.0,
        max_speed=1.0,
    )
    model = StudentGNNGRU(cfg)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_shards = discover_shards(train_dir)
    val_shards = discover_shards(val_dir)

    ckpt_dir = data_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_csv = Path(args.log_csv) if args.log_csv else (data_dir / "training_log_gru.csv")
    fieldnames = ["epoch", "train_loss", "val_loss", "lr"]

    set_seed(args.seed)
    best_val = float("inf")
    best_epoch = -1

    print(f"Data dir: {data_dir}")
    print(f"Train shards: {len(train_shards)} | Val shards: {len(val_shards)}")
    print(f"Node dim: {node_dim} | Edge dim: {edge_dim}")
    print(f"Device: {device}")

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        tr_sum = 0.0
        tr_cnt = 0.0
        for samples in batch_iter_seq(train_shards, args.batch_size, shuffle=True, seed=args.seed + epoch * 1000):
            batch = collate_seq(samples)
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred, _ = model.forward_sequence(
                x,
                batch["edge_index"],
                batch["edge_attr"],
                mask=mask,
                h0=None,
            )

            # Flatten for masked loss
            loss_sum, cnt = loss_sum_and_count(
                pred.reshape(-1, 2),
                y.reshape(-1, 2),
                mask.reshape(-1),
            )
            if cnt.item() == 0:
                continue
            loss = loss_sum / cnt
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if args.clip_grad_norm and args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            tr_sum += float(loss_sum.item())
            tr_cnt += float(cnt.item())

        train_loss = tr_sum / max(tr_cnt, 1e-9)

        # ---- Val ----
        model.eval()
        va_sum = 0.0
        va_cnt = 0.0
        with torch.no_grad():
            for samples in batch_iter_seq(val_shards, args.batch_size, shuffle=False, seed=args.seed + 999):
                batch = collate_seq(samples)
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                mask = batch["mask"].to(device)
                pred, _ = model.forward_sequence(
                    x,
                    batch["edge_index"],
                    batch["edge_attr"],
                    mask=mask,
                    h0=None,
                )
                loss_sum, cnt = loss_sum_and_count(
                    pred.reshape(-1, 2),
                    y.reshape(-1, 2),
                    mask.reshape(-1),
                )
                if cnt.item() == 0:
                    continue
                va_sum += float(loss_sum.item())
                va_cnt += float(cnt.item())

        val_loss = va_sum / max(va_cnt, 1e-9)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | lr={lr:.2e}")

        # Save last
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "cfg": cfg.__dict__,
            },
            ckpt_dir / "last_gru.pt",
        )

        # Save best
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
                ckpt_dir / args.ckpt_name,
            )
            print(f"[checkpoint] New best val {best_val:.6f} at epoch {best_epoch}")

        append_csv_row(log_csv, fieldnames, {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})

    print(f"Done. Best val {best_val:.6f} at epoch {best_epoch}. Logs: {log_csv}")


if __name__ == "__main__":
    main()
