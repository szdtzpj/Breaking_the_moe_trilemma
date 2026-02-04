"""
Plot the evolution of group_maps across multiple cluster events as a heatmap.
Usage:
  python -m scripts.visualize_group_evolution --log logs/training_log.jsonl --out plots/group_evolution.png
Explanation:
  Rows = cluster event order
  Columns = expert id (globally deduplicated and sorted)
  Cells = group id ( -1 if none)
"""
import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
from deguc.core.logging import SimpleLogger

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    events = SimpleLogger.load_events(args.log)
    clusters = [e for e in events if e.get("event") == "cluster" and "group_map" in e]
    if not clusters:
        print("No cluster events or missing group_map in the log.")
        return
    # Sort by step
    clusters.sort(key=lambda x: x["step"])
    # Collect all experts
    all_experts = sorted(set(e for c in clusters for exps in c["group_map"].values() for e in exps))
    exp_to_col = {e: i for i, e in enumerate(all_experts)}

    mat = np.full((len(clusters), len(all_experts)), -1, dtype=int)
    steps = []
    for r, c in enumerate(clusters):
        steps.append(c["step"])
        gmap = c["group_map"]
        for g, exps in gmap.items():
            for e in exps:
                mat[r, exp_to_col[e]] = g

    plt.figure(figsize=(max(6, len(all_experts)*0.3), max(4, len(clusters)*0.4)))
    im = plt.imshow(mat, aspect='auto', interpolation='nearest', cmap='tab20')
    plt.colorbar(im, shrink=0.6, label="Group ID")
    plt.yticks(range(len(steps)), [f"{s}" for s in steps])
    plt.xlabel("Expert ID (sorted)")
    plt.ylabel("Cluster Event Step")
    plt.title("Group Map Evolution")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out)
    plt.close()
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
