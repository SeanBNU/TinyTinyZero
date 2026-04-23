"""Run N countdown trials through Qwen2.5-0.5B-Instruct and save each to a JSONL.
Produces the Section 5.2 preliminary-findings dataset.

    python analysis/collect_preliminary.py --n 30
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))
from app import (
    COUNTDOWN_PROMPT_TEMPLATE,
    countdown_reward,
    make_countdown_problem,
    _load_model,
    _model_state,
    generate_once,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "results" / "preliminary_0p5b.jsonl"))
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen2.5-0.5B-Instruct...")
    _load_model()
    print(f"  loaded on {_model_state['device']}")

    t_start = time.time()
    with out_path.open("w") as f:
        for i in range(args.n):
            problem = make_countdown_problem(seed=1000 + i)
            prompt = COUNTDOWN_PROMPT_TEMPLATE.format(
                numbers=problem["numbers"], target=problem["target"]
            )
            t0 = time.time()
            completion = generate_once(
                prompt,
                temperature=args.temperature,
                seed=1000 + i,
                max_new_tokens=args.max_new,
            )
            dt = time.time() - t0
            reward = countdown_reward(completion, problem["numbers"], problem["target"])
            record = {
                "trial": i,
                "numbers": problem["numbers"],
                "target": problem["target"],
                "completion": completion,
                "reward": reward["reward"],
                "reason": reward["reason"],
                "answer": reward["answer"],
                "value": reward["value"],
                "lenient_value": reward.get("lenient_value"),
                "gen_seconds": round(dt, 2),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            flag = "OK" if reward["reward"] == 1.0 else "--"
            print(f"  trial {i:>2} [{flag}] r={reward['reward']} reason={reward['reason']:<20} ({dt:.1f}s)")

    total = time.time() - t_start
    print(f"\nWrote {args.n} trials to {out_path} in {total:.0f}s")


if __name__ == "__main__":
    main()
