"""
Analyze training logs and plot curves.
Usage:
    python -m scripts.analyze_logs --log logs/training_log.jsonl --out plots
"""
import argparse, os, math
from deguc.core.logging import SimpleLogger
import matplotlib.pyplot as plt
import numpy as np

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def moving_avg(x, k=20):
    if len(x) < k:
        return x
    import numpy as np
    return np.convolve(x, np.ones(k)/k, mode='valid')

def plot_curve(xs, ys, title, ylabel, save_path, ma=0):
    plt.figure(figsize=(7,4))
    plt.plot(xs, ys, label="raw", alpha=0.5)
    if ma > 1 and len(ys) >= ma:
        ma_y = moving_avg(ys, ma)
        ma_x = xs[len(xs)-len(ma_y):]
        plt.plot(ma_x, ma_y, label=f"MA({ma})", linewidth=2)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to training_log.jsonl")
    ap.add_argument("--out", default="plots", help="Output directory")
    ap.add_argument("--ma", type=int, default=20, help="Moving average window (optional)")
    args = ap.parse_args()

    ensure_dir(args.out)
    events = SimpleLogger.load_events(args.log)
    train_events = [e for e in events if e.get("event") == "train"]
    val_events = [e for e in events if e.get("event") == "validation"]
    cluster_events = [e for e in events if e.get("event") == "cluster"]

    if not train_events:
        print("No train events found. Please check if the log path is correct.")
        return

    steps = [e["step"] for e in train_events]
    losses = [e["loss"] for e in train_events]
    balances = [e.get("balance") for e in train_events]
    groups = [e.get("groups") for e in train_events]
    offloaded = [e.get("offloaded") for e in train_events]
    reloaded = [e.get("reloaded") for e in train_events]

    # May be classification task
    accs = [e.get("batch_acc") for e in train_events if "batch_acc" in e]
    # May be language modeling task
    ppls = [e.get("batch_ppl") for e in train_events if "batch_ppl" in e]

    plot_curve(steps, losses, "Training Loss", "loss",
               os.path.join(args.out, "train_loss.png"), args.ma)
    if balances[0] is not None:
        plot_curve(steps, balances, "Balance Loss", "balance",
                   os.path.join(args.out, "balance_loss.png"), args.ma)
    plot_curve(steps, groups, "Groups Over Time", "groups",
               os.path.join(args.out, "groups.png"), 1)
    plot_curve(steps, offloaded, "Offloaded Experts", "offloaded",
               os.path.join(args.out, "offloaded.png"), 1)
    plot_curve(steps, reloaded, "Reloaded Experts", "reloaded",
               os.path.join(args.out, "reloaded.png"), 1)

    if accs:
        # Align steps for accuracy: scan again
        acc_steps = [e["step"] for e in train_events if "batch_acc" in e]
        plot_curve(acc_steps, accs, "Batch Accuracy", "acc",
                   os.path.join(args.out, "batch_acc.png"), args.ma)
    if ppls:
        ppl_steps = [e["step"] for e in train_events if "batch_ppl" in e]
        plot_curve(ppl_steps, ppls, "Batch Perplexity", "ppl",
                   os.path.join(args.out, "batch_ppl.png"), args.ma)

    # Validation curves
    if val_events:
        v_steps = [e["step"] for e in val_events]
        if "val_loss" in val_events[0]:
            v_loss = [e["val_loss"] for e in val_events]
            plot_curve(v_steps, v_loss, "Validation Loss", "val_loss",
                       os.path.join(args.out, "val_loss.png"), 1)
        if "val_acc" in val_events[0]:
            v_acc = [e["val_acc"] for e in val_events]
            plot_curve(v_steps, v_acc, "Validation Accuracy", "val_acc",
                       os.path.join(args.out, "val_acc.png"), 1)
        if "val_ppl" in val_events[0]:
            v_ppl = [e["val_ppl"] for e in val_events]
            plot_curve(v_steps, v_ppl, "Validation Perplexity", "val_ppl",
                       os.path.join(args.out, "val_ppl.png"), 1)

    # Group stability
    if cluster_events:
        c_steps = [e["step"] for e in cluster_events]
        stabs = [e.get("stability") for e in cluster_events]
        plot_curve(c_steps, stabs, "Group Stability", "stability",
                   os.path.join(args.out, "group_stability.png"), 1)
    print("Completed. Plots have been saved to", args.out)

if __name__ == "__main__":
    main()
