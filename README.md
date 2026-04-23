# TinyTinyZero

Final project for CS 4180 (Reinforcement Learning, Prof. Amato), Northeastern, Spring 2026. Sean Blundin.

Replication and extension of TinyZero (Pan et al., 2025), which reproduces DeepSeek R1-Zero's emergent reasoning on Qwen2.5-3B using GRPO on the Countdown task. This project runs the identical pipeline on Qwen2.5-0.5B, 1.5B, and 3B to characterize where the reasoning emerges and what it looks like at each scale.

## Contents

- `PROJECT-PROPOSAL.md` — proposal submitted Feb 28.
- `report/paper.html` — the final paper (AAAI-structured). Open in a browser and print to PDF.
- `report/status_summary.md` — 1-page status handout from the Apr 16 poster session.
- `poster/` — the poster panels (8 letter-size pages), open in a browser and print.
- `demo/` — a Flask app that runs Qwen2.5-0.5B locally and walks through one GRPO step.

## Running the demo

```
cd demo
./run.sh
```

Opens on `http://127.0.0.1:5055`. Needs Python 3.9+ and about 1 GB of disk for the Qwen2.5-0.5B weights (downloaded on first load from HuggingFace). Uses the MPS backend on Apple Silicon, CUDA otherwise, CPU as a fallback.
