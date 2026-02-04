"""
Read a single group_map JSON snapshot (located in output_dir/group_maps/step_x.json) and plot the matrix. Use:
  python -m scripts.visualize_single_groupmap --snapshot outputs_8g_cls/group_maps/step_2000.json --out plots/groupmap_step2000.png
"""
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, help="Path to step_x.json file")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.snapshot, "r", encoding="utf-8") as f:
        data = json.load(f)
    group_map = data["group_map"]
    all_experts = sorted(set(e for exps in group_map.values() for e in exps))
    exp_to_col = {e: i for i, e in enumerate(all_experts)}

    rows = len(group_map)
    cols = len(all_experts)
    mat = np.full((rows, cols), -1, dtype=int)
    for ri, (g, exps) in enumerate(sorted(group_map.items(), key=lambda x: x[0])):
        for e in exps:
            mat[ri, exp_to_col[e]] = g

    plt.figure(figsize=(max(6, cols*0.3), max(3, rows*0.4)))
    im = plt.imshow(mat, aspect='auto', interpolation='nearest', cmap='tab20')
    plt.colorbar(im, shrink=0.6, label="Group ID")
    plt.title(f"Group Map Snapshot step {data['step']}")
    plt.xlabel("Expert ID (sorted)")
    plt.ylabel("Group Row (sorted by group id)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out)
    plt.close()
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
