import json
import argparse
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def load_counts(jsonl_path: str):
    # total counters
    agents_tot = Counter()
    walls_tot = Counter()
    obs_tot = Counter()

    # success counters
    agents_succ = Counter()
    walls_succ = Counter()
    obs_succ = Counter()

    total_rows = 0
    success_rows = 0
    bad_lines = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            if not all(k in d for k in ("n_agents", "n_walls", "n_obstacles", "success")):
                bad_lines += 1
                continue

            a = int(d["n_agents"])
            w = int(d["n_walls"])
            o = int(d["n_obstacles"])
            s = bool(d["success"])

            total_rows += 1
            agents_tot[a] += 1
            walls_tot[w] += 1
            obs_tot[o] += 1

            if s:
                success_rows += 1
                agents_succ[a] += 1
                walls_succ[w] += 1
                obs_succ[o] += 1

    return (agents_tot, agents_succ,
            walls_tot, walls_succ,
            obs_tot, obs_succ,
            total_rows, success_rows, bad_lines)


def overlay_bars(ax, total_counter: Counter, success_counter: Counter, xlabel: str, title: str,
                 total_width=0.85, success_width=0.55):
    if not total_counter:
        ax.set_title(title + " (no data)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("episodes")
        return

    xs = np.array(sorted(total_counter.keys()), dtype=int)
    total = np.array([total_counter[x] for x in xs], dtype=int)
    succ = np.array([success_counter.get(x, 0) for x in xs], dtype=int)

    # Total bars (red) behind
    ax.bar(xs, total, width=total_width, color="red", alpha=0.35, label="total", zorder=1)
    # Success bars (blue) on top
    ax.bar(xs, succ, width=success_width, color="blue", alpha=0.90, label="success", zorder=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("episodes")
    ax.set_title(title)
    ax.set_xticks(xs)  # remove if too cluttered
    ax.grid(True, axis="y", alpha=0.25, zorder=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "jsonl_path",
        nargs="?",
        default="datasets/il_dataset/logs/run_log_walls+obstacles.jsonl",
        help="Path to the run_log_*.jsonl file",
    )
    args = ap.parse_args()

    (agents_tot, agents_succ,
     walls_tot, walls_succ,
     obs_tot, obs_succ,
     total_rows, success_rows, bad_lines) = load_counts(args.jsonl_path)

    print(
        f"Loaded {total_rows} rows (successes={success_rows}); "
        f"skipped {bad_lines} malformed/missing-key lines."
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), constrained_layout=True)

    overlay_bars(axes[0], agents_tot, agents_succ, "n_agents",
                 "Episodes by # agents (total vs success)")
    overlay_bars(axes[1], walls_tot, walls_succ, "n_walls",
                 "Episodes by # walls (total vs success)")
    overlay_bars(axes[2], obs_tot, obs_succ, "n_obstacles",
                 "Episodes by # obstacles (total vs success)")

    # One shared legend (avoid repeating on each subplot)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    plt.show()


if __name__ == "__main__":
    main()
