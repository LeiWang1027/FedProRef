"""
FedProRef: Results Visualization
Parse log files and generate comparison plots.
"""
import os
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(path):
    """Parse a log file. Returns list of dicts with round, acc, ece, nll."""
    records = []
    with open(path, "r") as f:
        for line in f:
            m = re.search(r"Round\s+(\d+)\s*\|\s*acc:\s*([\d.]+)\s*\|\s*ece:\s*([\d.]+)\s*\|\s*nll:\s*([\d.]+)", line)
            if m:
                records.append({
                    "round": int(m.group(1)),
                    "acc": float(m.group(2)),
                    "ece": float(m.group(3)),
                    "nll": float(m.group(4)),
                })
    return records


def find_logs(log_dir, pattern=""):
    """Find log files matching pattern."""
    logs = {}
    for fname in sorted(os.listdir(log_dir)):
        if fname.endswith(".log") and pattern in fname:
            name = fname.replace(".log", "")
            # Clean up name: remove timestamp suffix
            parts = name.rsplit("_", 2)
            if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
                label = "_".join(parts[:-2])
            else:
                label = name
            logs[label] = os.path.join(log_dir, fname)
    return logs


def plot_comparison(logs_dict, metric="acc", output_path="comparison.png",
                    title="Method Comparison"):
    """Plot metric curves for multiple methods."""
    plt.figure(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (label, path) in enumerate(logs_dict.items()):
        records = parse_log(path)
        if not records:
            continue
        rounds = [r["round"] for r in records]
        values = [r[metric] for r in records]
        plt.plot(rounds, values, label=label, color=colors[i % 10], linewidth=1.5)

    plt.xlabel("Communication Round")
    ylabel_map = {"acc": "Top-1 Accuracy (%)", "ece": "ECE", "nll": "NLL"}
    plt.ylabel(ylabel_map.get(metric, metric))
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def print_summary(logs_dict):
    """Print best results for each method."""
    print(f"\n{'Method':<40s} {'Best Acc':>10s} {'@ Round':>8s} {'Best ECE':>10s} {'Best NLL':>10s}")
    print("-" * 80)
    for label, path in logs_dict.items():
        records = parse_log(path)
        if not records:
            continue
        best_acc_rec = max(records, key=lambda r: r["acc"])
        best_ece_rec = min(records, key=lambda r: r["ece"])
        best_nll_rec = min(records, key=lambda r: r["nll"])
        print(f"{label:<40s} "
              f"{best_acc_rec['acc']:>9.2f}% "
              f"{best_acc_rec['round']:>7d} "
              f"{best_ece_rec['ece']:>10.4f} "
              f"{best_nll_rec['nll']:>10.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--pattern", default="", help="Filter log files by pattern")
    parser.add_argument("--output_dir", default="./plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logs = find_logs(args.log_dir, args.pattern)

    if not logs:
        print(f"No log files found in {args.log_dir}")
        return

    print(f"Found {len(logs)} log files:")
    for k in logs:
        print(f"  - {k}")

    # Plot all three metrics
    for metric in ["acc", "ece", "nll"]:
        out_path = os.path.join(args.output_dir, f"comparison_{metric}.png")
        plot_comparison(logs, metric, out_path,
                        title=f"FedProRef: {metric.upper()} Comparison")

    # Print summary table
    print_summary(logs)


if __name__ == "__main__":
    main()
