---
title: MILESTONE 8 — Chain-consistency + retrieval-grounded reward extensions (M5.1 ceiling-buster)
tags: [milestone, training, reward-design, m8, m5.1, grpo, f1-ceiling]
source: internal
created: 2026-05-16
updated: 2026-05-16
---

# Milestone 8: Chain-consistency and retrieval-grounded reward for Qwen3.5-0.8B GRPO

## Context

[M5.1](../milestone_5/MILESTONE_5.md) (Qwen3.5-0.8B GRPO on the ReSearch-paper recipe, MuSiQue-only, F1-only reward) is the in-flight production run on Spheron H200 dedicated tier. Live trajectory in [`../report/RESULTS_M5_1_H200.md`](../report/RESULTS_M5_1_H200.md) shows the reward window-mean climbed cleanly through cadences 4-6 (0.171 → 0.224 over 30 steps) then **plateaued in the 0.22-0.24 band** across cadences 6-9. The cadence-9 window mean of 0.228 is the run high; the per-cadence reward gain has decelerated to **≤ +0.005 absolute with σ ≈ ±0.012**.

The plateau is *not* model-capability-bound. Cadence-9 step 84 generalised the "country → chief justice" chain from Ghana (seen 3× across cadences 5-8) to **Nigeria → Walter Samuel Nkanu Onnoghen** in 2 well-aimed tool calls; the planned-multi-hop count grew to **153 rollouts** in cadence 9 (new high vs cadence 8's 132); 4-hop+ successes have been stable at 20-32 per 10-step window since cadence 4. **The chain-reasoning capability is mature and stable.**

The plateau *is* reward-design-bound — specifically the F1-only reward's inability to distinguish chain-correct rollouts from chain-broken rollouts that happened to land on a gold token. Trace-level evidence in [`../report/RESULTS_M5_1_H200.md` §9.5](../report/RESULTS_M5_1_H200.md) compares cadence-9 step 91 idx 241 (Kotri railway: clean 3-call chain, reward 1.0) with cadence-9 step 93 idx 10 (Fox Island: 3-call chain with **silent entity flip USA → UK**, reward 1.0). Both get reward 1.0; GRPO's group-relative advantage estimator cannot tell them apart. Once both modes coexist in policy, gradient pressure stops preferring one over the other and the scalar reward asymptotes.

[Phase-1 finding #4](../report/RESULTS_m0_b.md) predicted this for the *partial-credit* variant of the F1 reward: "the paper's partial-credit reward creates a 0.1 floor that masks the tool-use signal". M5.1 already dropped the partial-credit floor (one of the two intentional divergences from the ReSearch paper); the remaining ceiling is the **chain-quality blindness** of token-overlap F1 itself. M8 addresses exactly this.

## Goal

Two reward extensions, both pure functions of the rollout string, both wired into the same hook point as the M5.1 F1 reward:

| Phase | Goal | Deliverables |
|---|---|---|
| **M8.1** | Add a **chain-consistency penalty** that detects silent entity flips between adjacent `<think>` blocks (where the new entity is not justified by an intervening `<tool_response>`). Composed multiplicatively with F1: `reward = f1 * (1 - chain_inconsistency_penalty)`. | `training_m8_1/` scaffold copied from `training_m5_1/`; edit to `src/rewards/search_r1.py` (~30 LoC + 4 unit tests); 50-step smoke + comparison vs M5.1 cadence-1 window. |
| **M8.2** | Add a **retrieval-grounded factor** that multiplies F1 by the fraction of answer tokens appearing in the concatenated retrieval payload. Composed with M8.1: `reward = f1 * (1 - chain_inconsistency) * max(0.3, retrieval_grounded)`. | `training_m8_2/` scaffold; +20 LoC in `src/rewards/search_r1.py`; 50-step smoke; full 311-step training run; eval via `evaluation_qwen35/` at the same protocol as M5.1's checkpoints. |

After M8.2, running `training_m8_2/scripts/run.sh` produces a training run that is identical to the M5.1 recipe **modulo**:
- Reward: composed F1 × (1 − chain-inconsistency penalty) × max(0.3, retrieval-grounded factor) (M5.1: F1 only).
- Everything else (model, dataset, hyperparameters, retriever, prompt template, tool tags, GRPO config) unchanged — so the only causal variable in the M8.2-vs-M5.1 comparison is the reward shape.

## Why this is a milestone (and not an M5.2 entry)

M5.2 was provisioned in [M5 §"Parallel experiments"](../milestone_5/MILESTONE_5.md) as the natural successor to M5.1, with candidate ablations listed as F1→EM, dataset swaps, or GRPO→MC-GRPO. M8 is **a different category of change**: it modifies the *scalar contract between rollouts and the optimiser*, not a hyperparameter or a data choice. Three reasons it earns its own milestone:

1. **The plateau diagnosis is the core M5.1 finding.** Without M5.1's 90-step trajectory and the trace-level evidence in §9.5, we'd be guessing at why the reward saturates. M5.1 produced the data; M8 acts on it. That's a milestone-grade dependency.
2. **The thesis story changes.** M5.1's chapter ends with "F1-only reward plateaus at 0.22-0.24, but chain reasoning continues to improve". M8's chapter ends with either "shaped reward breaks past the ceiling to 0.27-0.32" (positive result) or "shaped reward fails to lift the ceiling, suggesting the bound is model-prior-bound" (negative result; still load-bearing). Both outcomes are thesis-quality findings; both deserve their own narrative scope.
3. **The implementation surface is tiny but the design space isn't.** ~50 LoC of code; many design choices (penalty curve shape, grounding granularity, composition order, edge cases). M8 is the right place to argue those choices on the record, not bury them in an M5 sub-section.

## The F1-reward ceiling (one-page recap)

Detailed in [`../report/RESULTS_M5_1_H200.md` §9.5](../report/RESULTS_M5_1_H200.md). The short version:

```
reward(rollout) = f1(extract_solution(rollout), gold_answer)
```

is implemented at [`../../training_m5_1/src/rewards/search_r1.py:113`](../../training_m5_1/src/rewards/search_r1.py#L113). The `solution_str` argument is the full rollout text with `<think>`, `<tool_call>`, `<tool_response>`, `<answer>` blocks. The reward function only consults `extract_solution(...)` (which returns the first `<answer>…</answer>` content) and the gold answer list. **Everything else in the rollout is invisible to the optimiser.**

GRPO's advantage estimator is `A_i = r_i - mean(r_group)` with optional normalisation. In a group of 5 rollouts on the same prompt, suppose:
- 2 land at F1 = 1.0 via clean chain reasoning,
- 1 lands at F1 = 1.0 via silent entity flip that happened to align with the gold token,
- 2 land at F1 = 0.0 (wrong answer).

The optimiser sees `[1.0, 1.0, 1.0, 0.0, 0.0]` → advantage `[+0.4, +0.4, +0.4, -0.6, -0.6]`. The chain-broken rollout gets the *same positive advantage* as the clean rollouts. Gradient ascent pushes the policy toward all three high-reward shapes equally. After enough steps, the policy spreads probability mass over both modes; the scalar reward can't push higher because chain-broken-but-token-aligned is now a stable equilibrium of the optimisation.

Cadence-9 trace evidence in the RESULTS doc establishes this is happening empirically, not just theoretically. Hand-sampled hit rate of chain-broken-but-reward-1.0 rollouts: **~10-15 % of perfect-reward rollouts per cadence**.

## Reward design A: chain-consistency penalty (M8.1)

**Intuition.** Penalise the silent-switch pattern. If two adjacent `<think>` blocks reference different bridge entities and no `<tool_response>` between them mentions the new entity, the policy is reward-hacking via reasoning incoherence.

**Algorithm**:

1. Parse `solution_str` into ordered `<think>` blocks and `<tool_response>` blocks.
2. For each `<think>` block, extract the *current bridge entity* — the last named entity matching a heuristic regex (or NER as a sharpening; see "Sharpening" below).
3. For each adjacent pair `(think_i, think_{i+1})`:
   - If the bridge entities differ AND
   - No intervening `<tool_response>` mentions the new entity,
   - Count it as a silent flip.
4. Penalty = `min(0.5, 0.2 × n_flips)`. Capped so reward stays ≥ 0.5 × F1.

**Implementation**:

```python
# training_m8_1/src/rewards/search_r1.py (additions on top of M5.1)

import re

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
_TOOL_RE  = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)
_BRIDGE_RE = re.compile(
    r"(?:country|city|state|nation|place|location)\s+(?:is|=|:|containing|of)?\s*[\"']?([A-Z][A-Za-z ]{2,40})[\"']?",
    re.I,
)


def chain_inconsistency_penalty(rollout: str) -> float:
    """0.0 = clean chain; up to 0.5 = silent entity-flip detected.

    Detects when the model's <think> block i references entity X and block
    i+1 references entity Y, with no intervening <tool_response> mentioning
    Y. The penalty is capped so reward stays >= 0.5 * f1 even on the worst
    rollout; we want a gradient signal, not a zeroing.
    """
    thinks = _THINK_RE.findall(rollout)
    tools = _TOOL_RE.findall(rollout)
    if len(thinks) < 2:
        return 0.0
    bridges = []
    for t in thinks:
        cands = _BRIDGE_RE.findall(t)
        bridges.append(cands[-1].strip() if cands else None)
    flips = 0
    for i in range(1, len(bridges)):
        prev, curr = bridges[i - 1], bridges[i]
        if prev is None or curr is None or prev == curr:
            continue
        intervening = tools[i - 1:i] if i - 1 < len(tools) else []
        if not any(curr.lower() in tr.lower() for tr in intervening):
            flips += 1
    return min(0.5, 0.2 * flips)
```

Then in `compute_search_r1_reward`:

```python
def compute_search_r1_reward(solution_str: str, golden_answers) -> dict:
    answer = extract_solution(solution_str)
    if answer is None:
        return {"reward": 0.0, "extracted_answer": None, "f1": 0.0, "em": 0.0,
                "chain_inconsistency": 0.0}
    f1 = f1_check(answer, golden_answers)
    em = em_check(answer, golden_answers)
    penalty = chain_inconsistency_penalty(solution_str)
    return {
        "reward": float(f1 * (1.0 - penalty)),
        "extracted_answer": answer,
        "f1": float(f1),
        "em": float(em),
        "chain_inconsistency": float(penalty),
    }
```

**Hook point**: import in [`training_m8_1/src/environments/search_r1_env.py:56-60`](../../training_m5_1/src/environments/search_r1_env.py#L56-L60) (copy from M5.1, no env logic changes).

**Per-rollout overhead**: < 1 ms (regex-only). Negligible vs the 400-600 s step wall.

**Sharpening (post-M8.1, optional)**: swap `_BRIDGE_RE` for spaCy `en_core_web_sm` NER on each `<think>` block. Adds ~50 MB to the Ray worker venv and ~5 ms / rollout; still negligible. Reduces false negatives where the bridge entity is named in a less-canonical sentence structure than the regex catches.

## Reward design B: retrieval-grounded scoring (M8.2)

**Intuition.** A correct answer must appear in at least one retrieved chunk, or be derivable from one. If the answer tokens are nowhere in the retrieval payload, the model is hallucinating from prior — which violates the retrieval-augmented contract the recipe is supposed to teach.

**Algorithm**:

1. Collect all `<tool_response>` payloads from the rollout.
2. Tokenise the predicted answer with the same `normalize_answer` used by F1.
3. Count what fraction of answer tokens appears in the concatenated retrieval text.
4. If no tool calls at all: return 0.5 (retrieval-bypass — penalise but don't zero out; the model still gave the right token from prior).
5. Floor the factor at 0.3 so partial-grounding doesn't crush the gradient signal in a group.

**Implementation**:

```python
def retrieval_grounded_factor(rollout: str, predicted_answer: str) -> float:
    """1.0 = answer fully grounded in retrieved chunks; 0.5 = no tools used;
    0.3 floor when answer tokens are mostly absent from retrieval payload."""
    if not predicted_answer:
        return 0.0
    tools = _TOOL_RE.findall(rollout)
    if not tools:
        return 0.5  # retrieval-bypass
    retrieved = " ".join(tools).lower()
    answer_tokens = normalize_answer(predicted_answer).split()
    if not answer_tokens:
        return 0.0
    grounded = sum(1 for tok in answer_tokens if tok in retrieved)
    return grounded / len(answer_tokens)
```

Final composed reward (M8.2):

```python
def compute_search_r1_reward(solution_str: str, golden_answers) -> dict:
    answer = extract_solution(solution_str)
    if answer is None:
        return {"reward": 0.0, "extracted_answer": None, "f1": 0.0, "em": 0.0,
                "chain_inconsistency": 0.0, "retrieval_grounded": 0.0}
    f1 = f1_check(answer, golden_answers)
    em = em_check(answer, golden_answers)
    penalty = chain_inconsistency_penalty(solution_str)
    grounded = retrieval_grounded_factor(solution_str, answer)
    reward = f1 * (1.0 - penalty) * max(0.3, grounded)
    return {
        "reward": float(reward),
        "extracted_answer": answer,
        "f1": float(f1),
        "em": float(em),
        "chain_inconsistency": float(penalty),
        "retrieval_grounded": float(grounded),
    }
```

**Per-rollout overhead**: < 1 ms (string-in-string membership check).

**Sharpening (post-M8.2, optional)**: replace plain-token-in-text with an embedding-similarity check against retrieved chunks via the e5-base-v2 retriever already running in the cluster. Credits paraphrased answers (model says "America"; retrieval gave "United States"). One local embedding call per rollout (~50 ms with the warmed retriever) — still trivial at 400-600 s steps.

## Effect on the M5.1 cadence-9 reference traces

Concrete numbers from [`../report/RESULTS_M5_1_H200.md` §9.5](../report/RESULTS_M5_1_H200.md):

| Trace | F1 | chain_inconsistency | retrieval_grounded | reward M5.1 | **reward M8.2** | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Step 91 idx 241 — Kotri railway (clean) | 1.0 | 0.0 | 1.0 | 1.000 | **1.000** | 0 |
| Step 93 idx 10 — Fox Island (silent flip) | 1.0 | 0.2 | 1.0 | 1.000 | **0.800** | **−0.200** |

A **0.2 advantage gap** between chain-clean and chain-broken siblings in the same GRPO group is what the optimiser needs to start preferring clean chains. The penalty is *gradient-positive on the clean siblings* and *gradient-negative on the broken one*, not zero on both.

## Expected effect on the curve (predicted)

The M5.1 cadence-9 plateau at **0.22-0.24 window mean** is F1-only-reward-bound. Under M8.2:

1. **Many cadence-5+ rollouts currently scoring reward = 1.0 would drop to 0.7-0.9** because their chains contain at least one silent flip. Hand-sampling estimates ~10-15 % of perfect-F1 rollouts are chain-broken; under M8.2, their reward drops by ~0.1-0.2 each.
2. **Window mean will dip briefly in the first 5-10 steps** of a fresh M8.2 run as the policy re-explores under stricter reward.
3. **The asymptote should be higher**: the policy can no longer exploit chain-broken shortcuts. Predicted M8.2 ceiling: **0.27-0.32 window mean**, vs M5.1's 0.22-0.24.
4. **`<think>` block coherence should improve** (entity references stay stable across blocks unless retrieval changes them); 4-hop+ generalisation rate should hold or improve.

**Success criterion**: M8.2 window mean at step ≥ 80 exceeds **0.25** with chain_inconsistency mean < 0.05 across the population. Either condition failing means the chain-quality regularisation isn't doing what we expected and the structural plateau hypothesis needs revisiting.

**Negative-result criterion**: M8.2 window mean plateaus at < 0.24 with chain_inconsistency mean still > 0.05. That would suggest the reward ceiling is model-prior-bound, not reward-shape-bound, and the next move is to swap to a larger base model (Qwen3.5-2B) rather than further reward shaping.

Both outcomes are publishable. The negative result is arguably more valuable for the thesis chapter because it constrains future-work claims.

## Folder layout — `training_m8_1/`, `training_m8_2/`

Per the M5 [per-experiment folder convention](../milestone_5/MILESTONE_5.md#folder-layout--training_m5_1-training_m5_2-), M8.1 and M8.2 each live in fully self-contained directories at the repo root:

```
training_m8_1/
  nemo_rl/                  # vendored copy from training_m5_1/ at create-time
  src/
    rewards/
      search_r1.py          # F1 + chain_inconsistency_penalty (only file that differs from M5.1)
    environments/
      search_r1_env.py      # unchanged from M5.1
    ...                     # everything else copied from M5.1
  configs/
    m8_1_chain_consistency.yaml  # copy of m5_1_research_paper.yaml; only the experiment_name changes
  scripts/
    setup.sh
    run.sh
    smoke.sh
  tests/
    test_chain_inconsistency.py  # 4 unit tests below
  README.md

training_m8_2/
  ...                       # same as m8_1, but src/rewards/search_r1.py adds retrieval_grounded_factor
```

The wider repo layout (`data/`, `corpus/`, `indexes/`, `models/`, `eval/`) is shared from the root, read-only.

## Unit tests (M8.1)

`training_m8_1/tests/test_chain_inconsistency.py`:

```python
from training_m8_1.src.rewards.search_r1 import (
    chain_inconsistency_penalty, compute_search_r1_reward,
)


def test_clean_chain_no_penalty():
    """Cadence-9 step 91 Kotri pattern: bridge resolved + verified."""
    rollout = """
    <think>I need to find Aqeel Khan's home city.</think>
    <tool_call>...</tool_call>
    <tool_response>Aqeel Khan (born 30 January 1980, in Karachi)</tool_response>
    <think>Aqeel Khan's home city is Karachi. Now I need the Karachi-Kotri railway.</think>
    <tool_call>...</tool_call>
    <tool_response>Karachi-Kotri railway opened 13 May 1861.</tool_response>
    <answer> 13 May 1861 </answer>
    """
    assert chain_inconsistency_penalty(rollout) == 0.0


def test_silent_flip_penalised():
    """Cadence-9 step 93 Fox Island pattern: USA -> UK with no justification."""
    rollout = """
    <think>Country containing Fox Island: USA.</think>
    <tool_call>...</tool_call>
    <tool_response>First Pan-African Conference London 1900.</tool_response>
    <think>Country containing Fox Island: United Kingdom. Now I need...</think>
    <answer> United Kingdom </answer>
    """
    assert chain_inconsistency_penalty(rollout) >= 0.2


def test_justified_switch_not_penalised():
    """If retrieval surfaces the new entity, switching to it is correct behaviour."""
    rollout = """
    <think>Possible bridge: Manchester.</think>
    <tool_call>...</tool_call>
    <tool_response>Actually, the event was held in Birmingham, not Manchester.</tool_response>
    <think>Correcting: the country is Birmingham (city).</think>
    <answer> Birmingham </answer>
    """
    assert chain_inconsistency_penalty(rollout) == 0.0


def test_compose_with_f1():
    """End-to-end: a perfect-F1 silent-flip rollout gets reward 0.8, not 1.0."""
    rollout = """
    <think>Country: USA.</think>
    <tool_call>...</tool_call>
    <tool_response>London ...</tool_response>
    <think>Country: United Kingdom.</think>
    <answer> United Kingdom </answer>
    """
    out = compute_search_r1_reward(rollout, ["United Kingdom"])
    assert out["f1"] == 1.0
    assert 0.7 <= out["reward"] <= 0.9
    assert out["chain_inconsistency"] >= 0.2
```

Add equivalent retrieval-grounded tests in `training_m8_2/tests/test_retrieval_grounded.py` (fully-grounded, partially-grounded, ungrounded, no-tools-bypass).

## Smoke + full-run plan

| Phase | Steps | Hardware | Wall | Cost | Acceptance gate |
|---|---:|---|---:|---:|---|
| M8.1 unit tests | 0 | local | < 1 min | $0 | All 4 tests pass; F1=1.0 silent-flip rollout returns reward 0.7-0.9 |
| M8.1 smoke (50 steps) | 50 | 1× A100 / 4090 / H200 | 4-6 h | $5-25 | Reward window-mean within 20 % of M5.1's first-50-step trajectory; chain_inconsistency mean = 0.05-0.15 |
| M8.2 unit tests | 0 | local | < 1 min | $0 | retrieval_grounded == 1.0 for grounded rollouts; 0.5 for no-tool rollouts; < 1.0 for hallucinated answers |
| M8.2 smoke (50 steps) | 50 | 1× A100 / H200 | 4-6 h | $5-25 | retrieval_grounded mean = 0.8-0.95; reward window-mean within 20 % of M8.1 smoke |
| **M8.2 full run** | 311 (1 epoch) | 1× H200 dedicated $4.70/h | ~37 h | **~$175** | Window mean at step ≥ 80 > 0.25; chain_inconsistency mean < 0.05; eval on `evaluation_qwen35/` matches or beats M5.1's step_90 baseline on average EM |

**Total M8 budget**: ~$200 (smoke + full run + eval) on top of the M5.1 baseline cost.

## Eval protocol (carry from M4 / M5.1)

Every M8 checkpoint (step_10, step_20, …) is published to a new HF repo `pantomiman/qwen3.5-0.8b-grpo-musique-h200-m8-2-seed42` (parallel to M5.1's `…-h200-a4-seed42`). Evaluation pipeline is unchanged: [`evaluation_qwen35/`](../../evaluation_qwen35/) with the `qwen35_native` prompt mode, IVF-SQ8 retriever, 5-search-turn budget. Direct comparison against M5.1's step_90 baseline on the same 7-benchmark suite (NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle).

The thesis-defensible numerical claim, if M8.2 succeeds, is:

> *Composed chain-consistency + retrieval-grounded reward lifts the GRPO training-reward ceiling on Qwen3.5-0.8B from ≈ 0.22 to ≈ 0.28-0.32 (window-mean F1) without changing the model, dataset, or pipeline. Downstream eval EM improves by Δ on the 7-benchmark suite.*

If M8.2 produces a *negative* result (no lift past 0.24), the claim becomes:

> *Reward shaping at the chain-coherence and retrieval-grounding level is insufficient to lift the F1 ceiling at 0.8B scale; the bound is plausibly model-prior-bound, suggesting the recipe-search frontier moves to base-model scaling next.*

Both versions are publishable. The negative result is the more constrained claim.

## Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Regex-based bridge extraction misses real flips (false negatives) | Medium | Sharpen to NER post-smoke if `chain_inconsistency` mean stays near 0; the smoke output tells us empirically |
| 2 | Penalty too aggressive — kills useful exploration | Medium | Floor at 0.5 × F1 (penalty cap 0.5); revisit if smoke shows reward collapsing |
| 3 | Retrieval-grounded factor too literal — paraphrased correct answers get penalised | Medium | Embedding-similarity sharpening (post-M8.2) if mean factor stays low for clearly-correct rollouts |
| 4 | The structural plateau is model-prior-bound, not reward-bound — M8.2 doesn't lift | Low for negative-result-utility, high for positive-result | This is the test; the negative result is publishable and constrains future-work direction |
| 5 | New reward shape changes the entropy / KL trajectory and destabilises GRPO | Low | KL coefficient and learning rate unchanged from M5.1; smoke detects instability at 50 steps |

## What's left

| # | Task | Owner | Blocked on |
|---|---|---|---|
| 1 | Wait for M5.1 to complete (1 epoch, step 311 ETA ~22:00 UTC May 17) | — | nothing (autonomous) |
| 2 | Scaffold `training_m8_1/` (copy `training_m5_1/`) | — | (1) |
| 3 | Implement `chain_inconsistency_penalty` + 4 unit tests | — | (2) |
| 4 | M8.1 smoke (50 steps, 1× H200) | — | (3) |
| 5 | Scaffold `training_m8_2/` (copy `training_m8_1/`) | — | (4) |
| 6 | Implement `retrieval_grounded_factor` + 4 unit tests | — | (5) |
| 7 | M8.2 smoke | — | (6) |
| 8 | M8.2 full training run (311 steps) | — | (7) |
| 9 | Eval M8.2 checkpoints on 7-benchmark suite via `evaluation_qwen35/` | — | (8) |
| 10 | Write up M8 chapter; commit `RESULTS_m8.md` next to `RESULTS_M5_1_H200.md` | — | (9) |

## Pointers

- M5.1 plateau evidence (the dependency this milestone acts on): [`../report/RESULTS_M5_1_H200.md` §9.5](../report/RESULTS_M5_1_H200.md) — F1-reward ceiling diagnosis with Fox Island vs Kotri trace comparison.
- M5.1 reward implementation (the file to copy + edit): [`../../training_m5_1/src/rewards/search_r1.py`](../../training_m5_1/src/rewards/search_r1.py).
- M5.1 env (no changes for M8, just import a different reward): [`../../training_m5_1/src/environments/search_r1_env.py`](../../training_m5_1/src/environments/search_r1_env.py).
- M5 per-experiment folder convention: [`../milestone_5/MILESTONE_5.md` §"Folder layout"](../milestone_5/MILESTONE_5.md).
- Phase-1 finding #4 (the partial-credit-floor precedent for F1-reward ceilings): [`../report/RESULTS_m0_b.md`](../report/RESULTS_m0_b.md) (Qwen3-0.6B v1 ablation block).
- ReSearch paper (still the algorithmic source of truth): [arXiv:2503.19470](https://arxiv.org/abs/2503.19470).
- Eval pipeline carry-over: [`../milestone_4/MILESTONE_4.md`](../milestone_4/MILESTONE_4.md), [`../../evaluation_qwen35/`](../../evaluation_qwen35/).
- M6 publication-framing brief (the wider story M8 sits inside): [`../milestone_6/MILESTONE_6.md`](../milestone_6/MILESTONE_6.md).
