"""TinyTinyZero demo. Local Flask app running Qwen2.5-0.5B + countdown reward + group sampling."""
from __future__ import annotations

import json
import math
import random
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Queue, Empty

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

# -------------------------------------------------------------------------------------
# Model loading (lazy)
# -------------------------------------------------------------------------------------

MODEL_ID_DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"

_model_lock = threading.Lock()
_model_state: dict = {
    "loaded": False,
    "loading": False,
    "error": None,
    "model_id": None,
    "device": None,
    "tokenizer": None,
    "model": None,
}


def _load_model(model_id: str = MODEL_ID_DEFAULT) -> None:
    """Load the HF model. Called on first generation request."""
    with _model_lock:
        if _model_state["loaded"] and _model_state["model_id"] == model_id:
            return
        if _model_state["loading"]:
            return
        _model_state["loading"] = True
        _model_state["error"] = None

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        elif torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        model.to(device)
        model.eval()

        with _model_lock:
            _model_state.update(
                loaded=True,
                loading=False,
                model_id=model_id,
                device=device,
                tokenizer=tokenizer,
                model=model,
            )
    except Exception as exc:  # pragma: no cover
        with _model_lock:
            _model_state["loading"] = False
            _model_state["error"] = f"{type(exc).__name__}: {exc}"


# -------------------------------------------------------------------------------------
# Countdown task + reward
# -------------------------------------------------------------------------------------

COUNTDOWN_PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, write an arithmetic expression that equals "
    "{target}. You may use +, -, *, /, and parentheses. Each number must be used "
    "at most once. Show brief reasoning. End your response with a single line: "
    "'Answer:' followed by only the expression (no '=' sign, no extra text)."
)


def make_countdown_problem(seed: int | None = None) -> dict:
    """Generate a solvable countdown problem by construction."""
    rng = random.Random(seed)
    numbers = [rng.randint(1, 30) for _ in range(4)]
    # construct a random expression using these numbers to guarantee solvability
    ops = ["+", "-", "*"]
    a, b, c, d = numbers
    expr_candidates = [
        f"({a} + {b}) * {c} - {d}",
        f"{a} * {b} + {c} - {d}",
        f"{a} + {b} + {c} * {d}",
        f"({a} + {b} + {c}) * {d}",
        f"{a} * {b} - {c} + {d}",
        f"{a} + {b} * {c} - {d}",
    ]
    for _ in range(20):
        expr = rng.choice(expr_candidates)
        try:
            target = int(eval(expr, {"__builtins__": {}}, {}))
            if -200 <= target <= 200:
                break
        except Exception:
            continue
    else:
        target = a + b + c + d
    return {"numbers": numbers, "target": target, "ground_truth_expr": expr}


_ALLOWED_EXPR = re.compile(r"^[\d\s\+\-\*\/\(\)\.]+$")


def extract_answer(text: str) -> str | None:
    """Pull the final 'Answer:' line out of a generation."""
    for line in reversed(text.strip().splitlines()):
        m = re.match(r"\s*Answer\s*:\s*(.+?)\s*$", line, re.IGNORECASE)
        if m:
            return m.group(1)
    # fallback: last thing that looks like an expression
    for line in reversed(text.strip().splitlines()):
        if _ALLOWED_EXPR.match(line.strip()) and any(c in line for c in "+-*/"):
            return line.strip()
    return None


def _lenient_eval(answer: str) -> float | None:
    """Best-effort numerical evaluation: strip trailing '= <num>', allow only
    arithmetic characters. Returns a float on success, None otherwise. Used for
    diagnostic display in the demo only; does NOT affect reward."""
    if answer is None:
        return None
    s = answer.replace("×", "*").replace("÷", "/").replace("−", "-")
    s = re.sub(r"\s*=\s*[-+]?\d+(?:\.\d+)?\s*$", "", s).strip()
    if not _ALLOWED_EXPR.match(s):
        return None
    try:
        return float(eval(s, {"__builtins__": {}}, {}))
    except Exception:
        return None


def countdown_reward(text: str, numbers: list[int], target: int) -> dict:
    """The RLVR reward function. Binary: 1 if the expression evaluates to target
    using only provided numbers (each at most once), else 0. Returns detail too.
    Also returns a 'lenient_value' field for diagnostic display only, computed
    via a looser parse. lenient_value does NOT affect reward."""
    answer = extract_answer(text)
    if answer is None:
        return {"reward": 0.0, "reason": "no_answer_line", "answer": None, "value": None, "lenient_value": None}
    lenient = _lenient_eval(answer)
    safe = answer.replace("×", "*").replace("÷", "/").replace("−", "-")
    if not _ALLOWED_EXPR.match(safe):
        return {"reward": 0.0, "reason": "illegal_chars", "answer": answer, "value": None, "lenient_value": lenient}
    nums_used = [int(n) for n in re.findall(r"\d+", safe)]
    multiset_ok = True
    available = list(numbers)
    for n in nums_used:
        if n in available:
            available.remove(n)
        else:
            multiset_ok = False
            break
    if not multiset_ok:
        return {"reward": 0.0, "reason": "numbers_not_in_set", "answer": answer, "value": None, "lenient_value": lenient}
    try:
        value = eval(safe, {"__builtins__": {}}, {})
    except Exception as exc:
        return {"reward": 0.0, "reason": f"eval_error:{type(exc).__name__}", "answer": answer, "value": None, "lenient_value": lenient}
    if abs(value - target) < 1e-6:
        return {"reward": 1.0, "reason": "correct", "answer": answer, "value": float(value), "lenient_value": lenient}
    return {"reward": 0.0, "reason": "wrong_value", "answer": answer, "value": float(value), "lenient_value": lenient}


# -------------------------------------------------------------------------------------
# Generation
# -------------------------------------------------------------------------------------


def _chat_prompt(tokenizer, user_msg: str, system: str | None = None) -> str:
    sys_msg = system or "You are a careful problem solver. Always end with a line 'Answer: <expression>'."
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_stream(prompt_text: str, temperature: float, seed: int, max_new_tokens: int = 128):
    """Yield (token_str, done_bool). Streaming via transformers TextIteratorStreamer."""
    import torch
    from transformers import TextIteratorStreamer

    tokenizer = _model_state["tokenizer"]
    model = _model_state["model"]
    device = _model_state["device"]

    full_prompt = _chat_prompt(tokenizer, prompt_text)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=max(0.01, temperature),
        top_p=0.95,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
    )

    torch.manual_seed(seed)

    def _run():
        with torch.no_grad():
            model.generate(**gen_kwargs)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    for chunk in streamer:
        yield chunk, False
    yield "", True


def generate_once(prompt_text: str, temperature: float, seed: int, max_new_tokens: int = 128) -> str:
    import torch

    tokenizer = _model_state["tokenizer"]
    model = _model_state["model"]
    device = _model_state["device"]

    full_prompt = _chat_prompt(tokenizer, prompt_text)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=max(0.01, temperature),
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    # strip the prompt portion
    completion = full[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    return completion.strip()


# -------------------------------------------------------------------------------------
# Flask app
# -------------------------------------------------------------------------------------

app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify(
        loaded=_model_state["loaded"],
        loading=_model_state["loading"],
        error=_model_state["error"],
        model_id=_model_state["model_id"],
        device=_model_state["device"],
    )


@app.route("/api/load", methods=["POST"])
def load():
    data = request.get_json(silent=True) or {}
    model_id = data.get("model_id", MODEL_ID_DEFAULT)
    t = threading.Thread(target=_load_model, args=(model_id,), daemon=True)
    t.start()
    return jsonify(ok=True)


@app.route("/api/new_problem")
def new_problem():
    seed = request.args.get("seed", type=int)
    return jsonify(make_countdown_problem(seed=seed))


CHAT_SYSTEM = (
    "You are a helpful assistant running locally on a laptop. Be concise, specific, "
    "and a little bit fun. Use plain text, no markdown headers, short paragraphs."
)


@app.route("/api/chat_stream")
def chat_stream():
    """Free-form chat streaming endpoint. Separate from the countdown task,
    this is just the loaded local LLM in a friendly system prompt so visitors
    can see it talk about anything."""
    if not _model_state["loaded"]:
        return jsonify(error="model not loaded"), 409

    user_msg = request.args.get("msg", "Say hello in one sentence.").strip()[:800]
    temperature = float(request.args.get("temperature", 0.8))
    seed = int(request.args.get("seed", random.randint(0, 1_000_000)))
    max_new = min(int(request.args.get("max_new", 220)), 512)

    @stream_with_context
    def gen():
        import torch
        from transformers import TextIteratorStreamer

        tokenizer = _model_state["tokenizer"]
        model = _model_state["model"]
        device = _model_state["device"]

        full_prompt = _chat_prompt(tokenizer, user_msg, system=CHAT_SYSTEM)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=max(0.01, temperature),
            top_p=0.95,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
        )
        torch.manual_seed(seed)

        def _run():
            with torch.no_grad():
                model.generate(**gen_kwargs)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        yield f"event: meta\ndata: {json.dumps({'seed': seed, 'model_id': _model_state['model_id'], 'device': _model_state['device']})}\n\n"

        t0 = time.time()
        n_chars = 0
        for chunk in streamer:
            if chunk:
                n_chars += len(chunk)
                yield f"event: token\ndata: {json.dumps({'chunk': chunk, 'elapsed': time.time() - t0})}\n\n"
        yield f"event: done\ndata: {json.dumps({'elapsed': time.time() - t0, 'chars': n_chars})}\n\n"

    return Response(gen(), mimetype="text/event-stream")


@app.route("/api/stream")
def stream():
    """Stream a single generation token-by-token via SSE."""
    if not _model_state["loaded"]:
        return jsonify(error="model not loaded"), 409

    nums = json.loads(request.args.get("numbers", "[1,2,3,4]"))
    target = int(request.args.get("target", 10))
    temperature = float(request.args.get("temperature", 0.8))
    seed = int(request.args.get("seed", random.randint(0, 1_000_000)))

    prompt = COUNTDOWN_PROMPT_TEMPLATE.format(numbers=nums, target=target)

    @stream_with_context
    def gen():
        text_buf = []
        yield f"event: meta\ndata: {json.dumps({'seed': seed, 'prompt': prompt})}\n\n"
        for chunk, done in generate_stream(prompt, temperature, seed):
            if chunk:
                text_buf.append(chunk)
                yield f"event: token\ndata: {json.dumps({'chunk': chunk})}\n\n"
            if done:
                text = "".join(text_buf)
                rw = countdown_reward(text, nums, target)
                yield f"event: done\ndata: {json.dumps({'text': text, 'reward': rw})}\n\n"
                return

    return Response(gen(), mimetype="text/event-stream")


@app.route("/api/grpo_step")
def grpo_step():
    """Simulate ONE GRPO step: sample G completions, compute rewards, advantages.
    Returns everything needed to animate the group + math in the UI.
    This is the "epic" panel — real inference, real rewards, real z-score math."""
    if not _model_state["loaded"]:
        return jsonify(error="model not loaded"), 409

    nums = json.loads(request.args.get("numbers", "[1,2,3,4]"))
    target = int(request.args.get("target", 10))
    G = int(request.args.get("G", 4))
    temperature = float(request.args.get("temperature", 1.0))
    base_seed = int(request.args.get("seed", random.randint(0, 1_000_000)))

    prompt = COUNTDOWN_PROMPT_TEMPLATE.format(numbers=nums, target=target)
    completions: list[dict] = []
    t0 = time.time()
    for i in range(G):
        txt = generate_once(prompt, temperature=temperature, seed=base_seed + i, max_new_tokens=96)
        rw = countdown_reward(txt, nums, target)
        completions.append(
            {
                "idx": i,
                "text": txt,
                "reward": rw["reward"],
                "reward_reason": rw["reason"],
                "answer": rw["answer"],
                "value": rw["value"],
                "lenient_value": rw.get("lenient_value"),
                "seed": base_seed + i,
            }
        )
    t_gen = time.time() - t0

    rewards = [c["reward"] for c in completions]
    mu = sum(rewards) / len(rewards)
    var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
    sigma = math.sqrt(var) if var > 0 else 1e-8
    advantages = [(r - mu) / (sigma if sigma > 1e-8 else 1.0) for r in rewards]
    # simulated "KL penalty" and clipped surrogate — we don't actually backprop,
    # but we display the formula and plausible numbers computed from the rewards.
    kl_coef = 0.001
    epsilon = 0.2
    # approximate per-sample policy loss = -A * clip(ratio, 1-eps, 1+eps) -- but no ratio
    # since we don't have old_policy here; show the term with ratio=1 so the user sees
    # how it drops to -A (standard first-step PPO identity)
    policy_loss_terms = [-a for a in advantages]  # at ratio=1
    # simulated KL — higher-reward group => more update => more KL drift (illustrative)
    simulated_kl = 0.02 * (abs(mu - 0.5) + 0.5) + 0.01 * random.random()

    for c, a, pl in zip(completions, advantages, policy_loss_terms):
        c["advantage"] = a
        c["policy_loss_term"] = pl

    return jsonify(
        prompt=prompt,
        numbers=nums,
        target=target,
        G=G,
        completions=completions,
        stats={
            "mean_reward": mu,
            "std_reward": sigma,
            "solved": int(sum(rewards)),
            "total": G,
            "kl_coef": kl_coef,
            "epsilon": epsilon,
            "simulated_kl": simulated_kl,
            "gen_seconds": t_gen,
            "device": _model_state["device"],
            "model_id": _model_state["model_id"],
        },
    )


# -------------------------------------------------------------------------------------
# Static asset-like data endpoints
# -------------------------------------------------------------------------------------


@app.route("/api/tinyzero_curves")
def tinyzero_curves():
    """Serve our bundled approximation of TinyZero's public W&B reward curve.
    These values are synthesized to match the shape of the publicly posted
    curves from wandb.ai/jiayipan/TinyZero (the real run). We trace the
    published figure; this is purely for visual reference."""
    here = Path(__file__).parent
    return Response((here / "static" / "tinyzero_curves.json").read_text(), mimetype="application/json")


@app.route("/api/aha_gallery")
def aha_gallery():
    here = Path(__file__).parent
    return Response((here / "static" / "aha_gallery.json").read_text(), mimetype="application/json")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5055, debug=False, threaded=True)
