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

## M7.4 attempt 1 (2026-05-13, 16 steps, killed) — bootstrap failure + porting bug

Launched 2026-05-13 09:20 UTC. Killed at step 16 after the reward-bucket data showed an unfixable signal-density problem under the original walker.

### What we saw

| step | rew_mean | % format-INVALID | floor hits / 320 |
|---:|---:|---:|---:|
| 1 | 0.0009 | 99.1% | 3 |
| 2-5 | 0.0000 | **100.0%** | **0** |
| 6 | 0.0003 | 99.7% | 1 |
| 7 | 0.0006 | 99.4% | 2 |
| 8 | 0.0003 | 99.7% | 1 |
| 9 | 0.0009 | 99.1% | 3 |
| 10 | 0.0003 | 99.7% | 1 |
| 11-16 | 0.0003-0.0009 | 99-100% | 1-3 |

Across 5120 trajectories (16 × 320), only **15** got any non-zero reward — all at the 0.1 floor, **zero** F1>0. Tool-call emission rate steady at 10-14% (M7.1 baseline; no learning). The reward landscape was 99.8% flat zero — GRPO had no gradient direction.

### Diagnosis (the porting bug)

Step-14 breakdown of the 320 rollouts:

- 72% (229) — no `<answer>` block at all (truncated mid-thought)
- 27% (85) — **HAS `<answer>` but format gate rejects it**  ← the gap
- 2% (6) — passes format gate

Of the 85 rejected-with-answer rollouts:

| failure reason | count |
|---|---:|
| **"no `<think>` block present"** | **43** |
| **"think tags unbalanced: open=0 close=1"** | **22** |
| "rollout does not terminate on `</answer>`" | 6 |
| other think imbalance | 8 |
| `<answer>` imbalance | 6 |

**86% (73/85) of rejections were `<think>` accounting bugs.** Root cause: the Qwen3.5 chat template with `enable_thinking=True` emits `<think>\n` as the **generation prompt prefix** (in the prompt, not the assistant turn). The env builds `solution_str` via

```python
solution_str = "".join(m["content"] for m in log if m["role"] != "user")
```

which skips user turns — and the `<think>\n` opener lives inside the user-prompt rendering. So when a model generates a perfect `"thinking content </think> <answer>X</answer>"`, the format walker sees `<think>` count = 0, `</think>` count = 1, and rejects with "think tags unbalanced". The M5.5 walker was designed for an upstream context where `<think>` is emitted by the model itself; in our `enable_thinking=True` setup it's implicit in the prompt prefix, creating a systematic blind spot.

### Fix (commit `40c9ec3`)

One-line change in `is_valid_format()`:

```python
# was:
text = solution_str
# now:
text = "<think>\n" + solution_str
```

Prepending the implicit opener makes the balance walker see what the model actually generated within (effectively starting from `<think>` like the model's POV).

### Fix impact on real attempt-1 step-14 data

| metric | before fix | after fix |
|---|---:|---:|
| format-valid rate | 1.9% | **7.2%** (~4×) |
| "no `<think>` block" failures | 43 | **0** |
| "think open=0 close=1" failures | 22 | **0** |

Remaining 92.8% format-invalid is genuine model behavior (didn't close `</think>`, ran out of tokens before `<answer>`, etc.) — not a porting bug. **7.2% non-zero density** vs M7.1's step-1 **4.1% F1-positive rate** — almost 2× more rollouts contributing gradient, plus the 0.1 floor adding pressure toward format-valid behavior.

All 4 reward-logic smoke cases still pass:
- `garbage` → 0.0  fmt=False
- `valid+F1=0` → 0.1  fmt=True  (the floor)
- `valid+perfect` → 1.0  fmt=True
- `valid+partial(NY City vs NY)` → 0.8  fmt=True

## M7.4 attempt 2 (2026-05-13 11:01 UTC, killed ~10 min in)

Relaunched with the single-prepend fix at `40c9ec3`. User flagged that multi-turn rollouts have N+1 assistant turns separated by N closed `</tool_response>` blocks — each one gets its own implicit `<think>\n` opener from the chat template. Single-prepend handled only the first turn. Killed before it could complete a step.

### Multi-turn refinement (commit `a964b4c`)

```python
num_implicit_thinks = 1 + solution_str.count("</tool_response>")
text = ("<think>\n" * num_implicit_thinks) + solution_str
```

Empirical impact on attempt-1 step-14 data:

| state | format-valid rate |
|---|---:|
| before any fix | 1.9% |
| single-prepend | 7.2% ← permissive on multi-turn |
| **multi-turn fix** | **5.9%** ← correct |

The drop from 7.2% to 5.9% is the truth: the single-prepend was being permissive on multi-turn rollouts. The 5.9% reflects genuine model behavior — base models don't reliably close `</think>` between assistant turns, so multi-turn rollouts mostly fail the strict pairing rule.

## M7.4 attempt 3 (2026-05-13 12:47 UTC, running)

Relaunched with the multi-turn fix at `a964b4c`. Run name: `qwen3.5-0.8b-base-musique-m7_m74_short100-seed42-20260513T1247Z`.

### Trajectory through step 17 (~50 min wall)

| step | rew_mean | %fmt_valid | %tool_call | F1>0 hits | floor (0.1) | wall (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0169 | 12.8% | 13.4% | 2 | 39 | 325 |
| 5 | 0.0348 | 25.6% | 10.3% | 9 | 73 | 286 |
| 8 | 0.0654 | 55.0% | 5.6% | 28 | 148 | 194 |
| 10 | 0.0996 | 82.2% | **2.2%** | 32 | 231 | 132 |
| 13 | 0.1273 | 92.8% | 0.6% | 43 | 254 | 109 |
| 15 | 0.1475 | 95.9% | 0.3% | 59 | 248 | 93 |
| 17 | 0.1235 | 94.7% | **0.0%** | 35 | 268 | 94 |

### The dual finding

**What worked:**
- Format-valid rate climbed 12.8% → 95% in 17 steps. The bootstrap problem is solved by the fixes.
- Reward grew 0.017 → 0.15 — well above M7.1's full 100-step ceiling (0.087) in 1/6 the steps.
- F1>0 hits 2 → 59 per step (~18% of trajectories with non-trivial F1).
- Pace dropped 325s → 94s (rollouts shorter as model commits to single-turn).

**What broke:**
- **Tool-call rate collapsed 13.4% → 0.0% in 17 steps. Faster collapse than M7.1.**

### Mechanism (the surprise)

The strict format gate **accelerates** tool-use collapse on base models. Every multi-turn rollout (the ones with tool calls) systematically fails the format gate because the base model doesn't close `</think>` between assistant turns — it mixes thinking with tool calls without closing each block. Under this reward landscape:

- **Single-turn** (skip tool, just `<think>...</think><answer>X</answer>`) → easy format-valid → 0.1 floor or F1
- **Multi-turn** (use tool) → almost guaranteed format failure → 0 reward

GRPO learned the dominant strategy: skip the tool to satisfy the format gate. Inverse of M7.1's failure mode (where skipping was reward-equivalent because no incentive to use tool), same end behavior (0% tool use).

### Paper-worthy finding (now a clean trichotomy)

| reward | tool collapse mechanism | outcome |
|---|---|---|
| **M7.1 — F1-only** | no incentive to use tool; parametric path equally rewarded | tool 14% → 0% by step 50 |
| **M7.4 — F1 + strict format** | format gate punishes multi-turn rollouts (base model doesn't close `</think>` per turn); single-turn path easier | tool 13% → 0% by step 17 |
| **(hypothesized) M7.5 — F1 + loose format + explicit tool-use term** | requires only ≥1 closed `</think>`; bonus for `<tool_call>` regardless of nesting | TBD |

**Conclusion:** neither extreme of the spectrum works for base-model GRPO on this task. The reward must either (a) loosen format requirements about per-turn `</think>` accounting, or (b) carry an explicit tool-use reward term (independent of the format gate). The M5.5 design from `experiment_1_alice` works for hybrid post-trained models because they already follow per-turn `<think>...</think>` convention; it does NOT transfer to base.

### Cost ledger M7.4

| segment | cost |
|---|---:|
| Attempt 1 (16 steps, dead, buggy walker) | ~$1.60 |
| Diagnosis + fixes (no GPU) | $0 |
| Attempt 2 (10 min, killed before step 1) | ~$0.15 |
| Attempt 3 (running, 17 steps in 50 min) | ~$0.70 |
| **M7.4 total so far** | **~$2.45** |

### Backup status

Trajectories from all attempts uploaded to HF Hub dataset repo for VM-crash insurance:
- `m7_4_attempt1_buggy_walker_steps_1_15.tar.gz`
- `m7_4_attempt3_fixed_walker_through_step_17.tar.gz`
- See: `huggingface.co/datasets/pantomiman/qwen3.5-0.8b-base-grpo-musique-trajectories`

## Pointers

- M5.5 source (experiment_1_alice): `git show origin/experiment_1_alice:training_m5_5/src/rewards/search_r1.py`
- M7.1 closing finding (the motivation): [`MILESTONE_7.md` §"M7.1 closing finding"](MILESTONE_7.md)
- M7.3 prompt-only attempt (8-step result; tool-use held but reward died): [`MILESTONE_7_3.md`](MILESTONE_7_3.md)
- Reward (with M7.4 fix): [`training_m7_1/src/rewards/search_r1.py`](../../training_m7_1/src/rewards/search_r1.py)
- Config: [`training_m7_1/configs/m7_4_short100.yaml`](../../training_m7_1/configs/m7_4_short100.yaml)
- Launcher: `bash training_m7_1/scripts/run.sh --mode m74_short100`
- Watcher: `bash training_m7_1/scripts/watch_m7_4.sh <train_pid>`
