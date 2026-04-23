# TinyTinyZero

Final project for CS 4180 (Reinforcement Learning, Prof. Amato), Northeastern, Spring 2026. Sean Blundin.

Replication and extension of TinyZero (Pan et al., 2025), which reproduces DeepSeek R1-Zero's emergent reasoning on Qwen2.5-3B using GRPO on the Countdown task. This project runs the identical pipeline on Qwen2.5-0.5B, 1.5B, and 3B to characterize where the reasoning emerges and what it looks like at each scale.

## Contents

- `PROJECT-PROPOSAL.md` — proposal submitted Feb 28.
- `demo/` — Flask app that runs Qwen2.5-0.5B locally and walks through one GRPO step.
- `analysis/` — scripts that produced the preliminary-findings data in the paper (Section 5.2) and the four-bin behavior rubric (Section 4.3).
- `results/` — JSONL output + chart + text summary from running the analysis scripts. See `results/README.md`.

## Running the demo

```
cd demo
./run.sh
```

Opens on `http://127.0.0.1:5055`. Needs Python 3.9+ and about 1 GB of disk for the Qwen2.5-0.5B weights (downloaded on first load from HuggingFace). Uses the MPS backend on Apple Silicon, CUDA otherwise, CPU as a fallback.
