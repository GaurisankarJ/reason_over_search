# MILESTONE 7.4 — F1+format reward (Search-R1-style) on base model

**Status: open 2026-05-13. Predecessor: M7.3 (hard-imperative prompt) held tool-use at 35-42% across 8 steps but reward stayed flat because trajectories didn't complete — F1-only sees zeros. M7.4 changes the reward, keeps the M7.1 prompt.**

## Hypothesis

**M7.1's reward (F1-only) is what caused the tool-use collapse, not the prompt.** Search-R1 (Wei et al., 2025) and ReSearch both use a format-gated reward with a 0.1 floor for non-empty answers — and they retain tool-use behavior. M5.5 (on `experiment_1_alice` branch) ports this design as a clean ablation for the hybrid model. M7.4 ports it to base.

By switching the reward from F1-only to F1+format, two things should happen:
1. **Tool-use survives** because every well-structured rollout (including those that emit `<tool_call>`) gets the 0.1 floor when it can't match the gold answer. The model isn't punished for trying to use the tool.
2. **GRPO actually learns** because the gradient on every valid rollout is non-zero (vs M7.3 where 60-70% of rollouts had no closing `<answer>` and produced no F1 signal).

## What changes vs M7.1

| | M7.1 | M7.4 |
|---|---|---|
| Reward | F1-only on `<answer>X</answer>`, no floor, no format gate | **F1 + format gate + 0.1 floor** |
| Prompt | M4.3 base lock (`m7_qwen35_base_user.txt`) | **identical** |
| Arm | `qwen_native_no_system` | identical |
| Model | `Qwen3.5-0.8B-Base` (fresh) | identical |
| Hyperparams | G=5, batch=320, seq=8192, lr=1e-6, KL=0.001 | identical |
| Data | MuSiQue train.parquet | identical |
| Steps | 100 (probe) → 622 (full) | 100 (probe) → 622 (full); same shape |

**Single-variable ablation:** only the reward function changes. The eval template stays at `qwen35_minimal_no_system` (M4.3 base lock, byte-equal to training prompt).

## The new reward (verbatim from `training_m7_1/src/rewards/search_r1.py`)

```
reward(rollout) =
    0.0       if format INVALID
    0.1       if format OK but no <answer>  (impossible under our gate)
    0.1       if format OK with <answer> but F1 == 0
    f1        if format OK with <answer> AND F1 > 0
```

**Format validity** (5 rules, walker on the joined rollout text):
1. `<think>` open-count == close-count AND ≥ 1 block present
2. `<tool_call>` open-count == close-count
3. Each `<tool_call>` body contains a valid Qwen3.5 nested-XML `<function=NAME>...<parameter=ARG>VAL</parameter>...</function>` block
4. `<answer>` open-count == close-count AND ≥ 1 closed block
5. Rollout (stripped) **ends with `</answer>`** — terminal answer required

Tool-call emission is NOT required by the format gate. A model that thinks then answers directly with no tool call still gets F1 reward (or the 0.1 floor) as long as it produces `<think>...</think><answer>X</answer>` cleanly. What's punished is *broken* rollouts (truncated mid-thought, mismatched tags, JSON inside `<tool_call>` instead of nested-XML).

## Decision criteria at step 100

| outcome | what it shows | next step |
|---|---|---|
| **A. Tool-use ≥ 10% AND reward_mean ≥ 0.05** | Format gate + 0.1 floor is the missing piece. M7.1 collapse fixable by reward redesign alone, NOT by prompt-engineering. | Continue to step 622 (`--mode extend` on this ckpt dir). |
| **B. Tool-use ≥ 10% BUT reward stuck below 0.03** | Tool use survives but model isn't learning to actually answer correctly. Format gate prevents the easy bypass but the base model lacks the multi-hop reasoning capability under these conditions. | Stop at 100. Document as null result. |
| **C. Tool-use < 5%** | F1+format reward STILL allows the model to bypass tools (because emit `<think>...</think><answer>X</answer>` without `<tool_call>` is valid). Format gate ≠ tool-use gate. | Stop at 100. Conclude: explicit tool-use term needed in reward (e.g., reward only positive if `<tool_call>` was emitted). |

## Risks

1. **Format gate hurts EARLY learning** because step-1 base trajectories are often malformed (mid-truncated, missing tags). Many will get reward=0.0 in early steps, slowing learning vs M7.1 which gave partial F1 even on broken rollouts. Mitigation: watch step 5-10 carefully; if reward is flat 0.0, this is a sign the format gate is too strict (we'd loosen it).
2. **The 0.1 floor IS additive in M7.1's parametric-only regime**: at M7.1 ~$f1 \approx 0.16$, adding the 0.1 floor for failed answers might just inflate reward without changing behavior. To distinguish learning from inflation, compare M7.4 step-50 reward to M7.1 step-50 reward AT EQUAL TOOL-USE RATE — and look at multi-hop dataset performance (musique, 2wiki, hotpotqa) which is where tool use should matter.

## Phase plan

Mirror M7.1's cadence:

| phase | target | wall (est) | $ |
|---|---|---:|---:|
| M7.4.0 | port reward + smoke test | 5 min | $0 |
| M7.4.1 | short100 (100 GRPO steps) | ~3-5 h | $3-5 |
| M7.4.2 | extend if go-criterion passes | ~10 h | $8-10 |
| M7.4.3 | M7.2-style Plan A eval on step_100 + step_622 | 2.5 h | $2 |

Total budget: ~$15-20.

## Pointers

- M5.5 source (experiment_1_alice): `git show origin/experiment_1_alice:training_m5_5/src/rewards/search_r1.py`
- M7.1 closing finding (the motivation): [`MILESTONE_7.md` §"M7.1 closing finding"](MILESTONE_7.md)
- M7.3 prompt-only attempt (8-step result; tool-use held but reward died): [`MILESTONE_7_3.md`](MILESTONE_7_3.md)
- New reward (now in place at training_m7_1/): [`training_m7_1/src/rewards/search_r1.py`](../../training_m7_1/src/rewards/search_r1.py)
- New config: [`training_m7_1/configs/m7_4_short100.yaml`](../../training_m7_1/configs/m7_4_short100.yaml)
- Launcher: `bash training_m7_1/scripts/run.sh --mode m74_short100`
