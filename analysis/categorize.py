"""Apply the four-bin behavior rubric to a JSONL of completions.

Uses keyword-based heuristics for a first-pass categorization. Human review
remains authoritative; see analysis/rubric.md.

Usage:  python analysis/categorize.py --in results/preliminary_0p5b.jsonl
"""
from __future__ import annotations
import argparse, json, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BACKTRACK_MARKERS = [
    r"\binstead\b", r"\bgo back\b", r"\brecheck\b", r"\bwait\b",
    r"\bactually\b", r"\bhmm\b", r"\bno,\s", r"\blet me try\b",
]
VERIFY_MARKERS = [
    r"\bcheck\b", r"\btoo high\b", r"\btoo low\b", r"\bnot right\b",
    r"\bclose\b", r"\boff by\b", r"\bone off\b",
]


def categorize(record: dict) -> int:
    """Return 1..4 per the rubric. Backtracking dominates verification dominates structured."""
    text = (record.get("completion") or "").lower()
    reason = record.get("reason", "")
    # Rubric: unparseable or illegal-numbers completions -> Bin 1 regardless of surface text.
    if reason in ("no_answer_line", "numbers_not_in_set", "illegal_chars"):
        return 1
    n_backtrack = sum(1 for pat in BACKTRACK_MARKERS if re.search(pat, text))
    n_verify = sum(1 for pat in VERIFY_MARKERS if re.search(pat, text))
    if n_backtrack >= 2 or (n_backtrack >= 1 and n_verify >= 1):
        return 4
    if n_verify >= 1:
        return 3
    return 2  # valid expression attempt, no verification markers


BIN_NAMES = {
    1: "random guessing",
    2: "structured attempt",
    3: "self-verification",
    4: "backtracking",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(ROOT / "results" / "preliminary_0p5b.jsonl"))
    args = ap.parse_args()

    records = [json.loads(line) for line in Path(args.inp).read_text().splitlines() if line.strip()]
    counts = Counter(categorize(r) for r in records)
    n = len(records)
    print(f"Behavior rubric applied to {n} completions from {Path(args.inp).name}:")
    print("-" * 54)
    for b in (1, 2, 3, 4):
        c = counts.get(b, 0)
        pct = 100 * c / max(1, n)
        bar = "#" * int(pct / 2)
        print(f"  bin {b} {BIN_NAMES[b]:<22} {c:>3}  {pct:>5.1f}%  {bar}")
    print("\nHeuristic first pass. Final counts for the paper come from hand-coded review.")


if __name__ == "__main__":
    main()
