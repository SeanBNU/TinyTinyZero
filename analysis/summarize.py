"""Read a preliminary JSONL and print + plot the failure-mode breakdown.

Usage:  python analysis/summarize.py [--in results/preliminary_0p5b.jsonl]
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(ROOT / "results" / "preliminary_0p5b.jsonl"))
    ap.add_argument("--chart", default=str(ROOT / "results" / "failure_modes.png"))
    ap.add_argument("--txt", default=str(ROOT / "results" / "summary.txt"))
    args = ap.parse_args()

    records = [json.loads(line) for line in Path(args.inp).read_text().splitlines() if line.strip()]
    n = len(records)
    n_success = sum(1 for r in records if r["reward"] == 1.0)
    reasons = Counter(r["reason"] for r in records if r["reward"] == 0.0)

    lines = []
    lines.append(f"Qwen2.5-0.5B-Instruct on countdown, N = {n} trials")
    lines.append("=" * 60)
    lines.append(f"  successes:          {n_success}/{n}  ({100*n_success/n:.1f}%)")
    lines.append(f"  zero-reward trials: {n - n_success}/{n}")
    lines.append("")
    lines.append("Failure-mode breakdown among zero-reward trials:")
    for reason, count in reasons.most_common():
        pct = 100 * count / max(1, n - n_success)
        bar = "#" * int(pct / 2)
        lines.append(f"  {reason:<22} {count:>3}  {pct:>5.1f}%  {bar}")
    report = "\n".join(lines)
    print(report)

    Path(args.txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.txt).write_text(report + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["correct"] + list(reasons.keys())
        counts = [n_success] + [reasons[k] for k in reasons]
        colors = ["#2a9d5f"] + ["#c4514c"] * len(reasons)
        fig, ax = plt.subplots(figsize=(8, 4.2))
        bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_ylabel("Trials")
        ax.set_title(f"Qwen2.5-0.5B-Instruct on countdown (N = {n})")
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(c),
                    ha="center", va="bottom", fontsize=10)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(args.chart, dpi=140)
        print(f"\nchart: {args.chart}")
    except ImportError:
        print("\n(matplotlib not installed; skipping chart)")


if __name__ == "__main__":
    main()
