# Project Proposal: Replicating and Extending DeepSeek R1-Zero's Emergent Reasoning at Small Scale

**Sean Blundin**
Northeastern University
Professor Chris Amato
CS 4180/5180: Reinforcement Learning, Spring 2026
February 28, 2026

*Apologies for the late submission. I wanted to thoroughly research the landscape before committing to an ambitious project.*

## Executive Summary

TinyZero [4] reproduced DeepSeek R1-Zero's emergent reasoning by training Qwen2.5-3B with GRPO on a simple countdown task. This project replicates that result, then runs the identical setup on Qwen2.5-1.5B and Qwen2.5-0.5B to investigate at what model size self-verification and search behaviors emerge.

## 1. Problem Description

DeepSeek R1 (January 2025) demonstrated that LLMs can develop sophisticated reasoning behaviors (self-verification, backtracking, search) through pure reinforcement learning, without supervised fine-tuning on human-written reasoning traces [1]. This paradigm, Reinforcement Learning with Verifiable Rewards (RLVR), has since become central to LLM post-training, adopted by OpenAI's o1/o3, Anthropic's Claude, and Qwen 3 [3, 5].

Shortly after, Pan et al. [4] released TinyZero, a minimal reproduction of R1-Zero using Qwen2.5-3B on a countdown game (use given numbers and arithmetic to reach a target). For under $30 of compute, the model developed emergent self-verification and search behaviors through RL alone, using GRPO (Group Relative Policy Optimization) [6] on the veRL framework.

This project aims to (1) replicate TinyZero's core result on Qwen2.5-3B and (2) run the identical training on Qwen2.5-1.5B and Qwen2.5-0.5B to investigate at what scale emergent reasoning appears.

**Formal setup.** The model receives a list of integers and a target, and must produce an arithmetic expression evaluating to the target. In RL terms: the state is the input prompt; the action space is all possible token sequences; the reward is binary (1 if the expression evaluates correctly, 0 otherwise). The reward is computed by a Python expression evaluator, fully automatic with no human judgment required.

**Why this is interesting.** The central question is whether R1-Zero's "aha moment," where the model spontaneously begins verifying its own answers, is a property of scale or emerges even in very small models. TinyZero showed it works at 3B; this project probes the lower boundary. This connects to a live debate: Yue et al. [7] argued RLVR does not elicit fundamentally new reasoning but merely improves sampling efficiency of existing capabilities. Understanding how this plays out across model sizes is directly relevant.

## 2. Algorithms

**Primary algorithm: GRPO [6].** GRPO is a PPO variant that eliminates the critic (value) network. Instead of estimating advantages via a learned value function, GRPO samples a group of G responses per prompt, computes rewards, and calculates advantages as z-scores: A_i = (r_i - μ) / σ. This removes roughly half the memory overhead of PPO while retaining the clipped surrogate objective. A KL penalty against a reference policy prevents excessive drift. I will follow TinyZero's exact setup: veRL framework, GRPO, binary outcome rewards, and the Qwen2.5 family (0.5B, 1.5B, 3B).

**Baselines.** (1) Untrained base models (zero-shot), establishing the performance floor. (2) Chain-of-thought prompting without RL training, to isolate the effect of RL from formatting.

**Why GRPO.** It is the algorithm used in both DeepSeek R1 and TinyZero. It is suited to this setting because it works with binary verifiable rewards (no learned reward model), is memory-efficient (critical for running multiple model sizes), and is well-supported in veRL and TRL with extensive open-source documentation.

## 3. Expected Results

**(a) Core replication (Qwen2.5-3B).** I expect to reproduce TinyZero's finding: the 3B model learns to solve countdown tasks with increasing accuracy, and reasoning traces show emergent self-verification. I will compare reward curves to TinyZero's public Weights & Biases logs.

**(b) Scaling analysis (0.5B, 1.5B, 3B).** All three sizes trained under identical hyperparameters, comparing: (i) held-out accuracy; (ii) training reward curves; (iii) qualitative analysis of reasoning traces. Pan et al. noted 0.5B showed only "basic guessing" while 1.5B+ showed reasoning, but did not analyze this in depth. I will categorize behaviors at each scale (random guessing, partial arithmetic, self-verification, backtracking) with concrete examples.

**(c) Risks.** Primary risks are compute availability and RL training instability (reward collapse, KL explosion [2]). Mitigation: start with 0.5B to debug the pipeline quickly; use TinyZero's exact hyperparameters and logs as reference; the countdown task completes individual runs in hours. Fallback: if the full scaling analysis is infeasible, the core 3B replication with detailed qualitative analysis is itself a substantive contribution.

## Note on Background and Preparation

I want to be transparent that this project is a stretch for me, and that is intentional. I have worked with LLMs since GPT-2 (2020) but have not previously implemented RL training pipelines for language models. I enrolled in this course specifically to understand how RL has been incorporated into modern LLMs, a development I had lost touch with. In preparation, I have skimmed the key papers in this space [1, 5, 6, 7] and looked more deeply into the TinyZero codebase, experiment logs, and the veRL documentation [4]. I am confident I can execute the replication, and I am looking forward to the challenge.

## References

1. Guo, D., et al. (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948. Published in *Nature*, Sept. 2025.
2. Hu, J., Liu, M., Diao, S., Lu, X., and Dong, Y. (2025). "Scaling LLM RL with Prolonged Training Using ProRL v2." NVIDIA Technical Blog, Aug. 2025.
3. Karpathy, A. (2025). "2025 LLM Year in Review." karpathy.bearblog.dev, Dec. 2025.
4. Pan, J., Zhang, J., Wang, X., Yuan, L., Peng, H., and Suhr, A. (2025). "TinyZero." github.com/Jiayi-Pan/TinyZero.
5. Raschka, S. (2025). "The State of Reinforcement Learning for LLM Reasoning." magazine.sebastianraschka.com, Apr. 2025.
6. Shao, Z., et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300.
7. Yue, Y., et al. (2025). "Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" arXiv:2504.13837.
8. Ye, Y., et al. (2025). "LIMO: Less is More for Reasoning." arXiv:2502.03387. COLM 2025.
