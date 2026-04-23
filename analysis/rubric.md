# Behavior categorization rubric

Four-bin qualitative coding scheme for Countdown completions, applied to samples drawn from a fixed held-out analysis set at each (model size, training checkpoint).

## Bin 1 — Random guessing

No apparent strategy. The model emits a short arithmetic expression with little or no reasoning text. Any correctness is coincidental.

> Using [4, 8, 15, 16] to reach 23. 4 + 15 = 19. Answer: 4 + 15

## Bin 2 — Structured attempt (no verification)

The model tries valid operations on the given numbers and commits to an answer without checking whether the expression evaluates to the target.

> Using [4, 8, 15, 16] to make 23. 16 + 8 - 4 + 15 = 35. That might work. Answer: 16 + 8 - 4 + 15

## Bin 3 — Self-verification

The model computes an intermediate result, explicitly checks it against the target, and recognizes when the check fails, but does not necessarily explore alternatives.

> [4, 8, 15, 16] to 23. Let me try 16 + 8 = 24, close. 24 - 4 = 20. Not right. What about 15 + 8 = 23? Yes. Answer: 15 + 8

## Bin 4 — Backtracking

The model abandons a candidate path mid-reasoning and explores an alternative. This is the signature "aha" behavior.

> Given [4, 8, 15, 16], target 23. Try 16 + 4 = 20, plus 8 = 28, too high. Subtract instead: 16 + 8 = 24, minus 4 = 20, still not 23. Try 15 + 8 = 23. That uses two numbers and equals the target. Answer: 15 + 8

## Coding procedure

- Each completion is assigned the highest-numbered applicable bin (e.g., any completion that backtracks counts as Bin 4 even if it also verifies).
- Completions that produce no parseable Answer line or use numbers outside the provided set are coded as Bin 1 regardless of surface structure.
- For each (model size, checkpoint) pair, 50 completions are drawn from the fixed held-out set and categorized.
- A second rater independently categorizes 20% of the samples to compute Cohen's kappa.

## Automated preliminary coding

`categorize.py` applies a keyword-based heuristic for an initial pass (looking for markers like `let me try`, `too high`, `not right`, `wait`, `instead`). Human review remains authoritative; the heuristic is a triage aid for large samples and a sanity check on the manual coding.
