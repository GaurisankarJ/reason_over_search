# MILESTONE 7.3 — In-context 3-hop demonstration to prevent tool-call collapse

**Status: planning (2026-05-13). Predecessor M7.1 closed with the finding that F1-only GRPO induces complete tool-call collapse; see [`MILESTONE_7.md` §"M7.1 closing finding"](MILESTONE_7.md).**

## Hypothesis

The M7.1 base model **knows how to use the search tool** — at step 1 (untrained), 14.4% of trajectories emit `<tool_call>` blocks correctly. GRPO with F1-only reward then trains this behavior out because direct parametric answering yields the same `<answer>X</answer>`-only reward at zero retrieval cost.

A worked multi-hop demonstration inlined into the user prompt may shift the prior strongly enough that the demonstrated pattern (think → tool_call → tool_response → think → answer) survives GRPO updates. Mechanism: the model's first 100-step gradient is dominated by short, direct rollouts; if the prompt itself walks through a 3-hop trajectory, the model's "natural next-token" prior for a MuSiQue question is now "emit `<tool_call>`" rather than "emit `<answer>`".

**This is a single-variable ablation vs M7.1.** Same reward (F1-only, no floor). Same arm (`qwen_native_no_system`). Same hyperparameters (lr=1e-6, KL=0.001, G=5, batch=320 traj/step, seq=8192). Same model (Qwen3.5-0.8B-Base). **Only the prompt changes.**

If M7.3 preserves tool-use at training step 100, we learn: prompt prior > F1 gradient. If M7.3 collapses too, we learn: the reward design itself is the necessary intervention (Search-R1's 0.1 floor + format gate is load-bearing).

## What changes vs M7.1

| | M7.1 | M7.3 |
|---|---|---|
| Prompt | bare instruction + Question | bare instruction + **3-hop worked example** + Question |
| Reward | F1 on `<answer>X</answer>`, no floor | identical |
| Arm | `qwen_native_no_system` | identical |
| Training data | MuSiQue train.parquet | identical |
| Model init | `Qwen/Qwen3.5-0.8B-Base` (fresh) | `Qwen/Qwen3.5-0.8B-Base` (fresh — see "Resume policy" below) |
| Steps | 100 (probe) → 622 (full) | 100 (probe) → 622 (full); same shape |

**Resume policy:** start from the **untrained base model**, NOT from `m7_short100/seed42/step_100`. Resuming from the M7.1 ckpt would conflate "did the demo prevent collapse" with "did the demo undo prior hacking." Fresh start = clean ablation.

## Prompt change (proposed)

Current prompt (M7.1 lock, `training_m7_1/src/prompts/m7_qwen35_base_user.txt`):

```
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call the search tool by writing:
<tool_call>
<function=search>
<parameter=query>
your query
</parameter>
</function>
</tool_call>
The result will be returned inside <tool_response> and </tool_response>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {}
```

M7.3 prompt (proposed — under user review):

```
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call the search tool by writing:
<tool_call>
<function=search>
<parameter=query>
your query
</parameter>
</function>
</tool_call>
The result will be returned inside <tool_response> and </tool_response>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations.

Here is a worked example.

Question: In which year did the country where the founder of the company that produces the iPhone was born first land a spacecraft on the Moon?
<think>
I need to find: (1) who founded the iPhone's company, (2) where they were born, (3) when that country first landed a spacecraft on the Moon.
</think>
<tool_call>
<function=search>
<parameter=query>
Who founded Apple Inc.
</parameter>
</function>
</tool_call>
<tool_response>
Apple Inc. was co-founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.
</tool_response>
<think>
Steve Jobs co-founded Apple. I should check where he was born.
</think>
<tool_call>
<function=search>
<parameter=query>
Where was Steve Jobs born
</parameter>
</function>
</tool_call>
<tool_response>
Steve Jobs was born in San Francisco, California, United States, on February 24, 1955.
</tool_response>
<think>
Jobs was born in the United States. I need to find when the United States first landed a spacecraft on the Moon.
</think>
<tool_call>
<function=search>
<parameter=query>
First United States spacecraft to land on the Moon
</parameter>
</function>
</tool_call>
<tool_response>
The United States first soft-landed an uncrewed spacecraft on the Moon with Surveyor 1 on June 2, 1966. The first crewed landing was Apollo 11 on July 20, 1969.
</tool_response>
<think>
The first U.S. spacecraft landing on the Moon was Surveyor 1 in 1966.
</think>
<answer> 1966 </answer>

Now answer this question.
Question: {}
```

**Properties of the demo:**
- **True 3-hop**: bridging through Apple → Jobs → US → Surveyor 1.
- **Shows the exact format** the model is expected to emit (think + tool_call + tool_response + think + answer).
- **Demonstrates clarification** in `<think>` blocks (model articulates next step before each retrieval).
- **Demonstrates final synthesis** (closing think block before answer).
- **Length**: ~400 tokens. Combined with the original ~150-token instruction = ~550-token prompt. Leaves 7600+ tokens for the model's actual rollout — fine.
- **Render parity must be re-verified** before launch (same byte-equal check we did for M7.1, but on the longer template).

## Eval template parity

The eval template `QWEN35_MINIMAL_NO_SYSTEM_TEMPLATE` in [`evaluation_qwen35/flashrag/search_r1/templates.py`](../../evaluation_qwen35/flashrag/search_r1/templates.py) must also be updated to include the same demonstration, so the trained model sees the same context at eval time. The training prompt and eval template must render byte-equal (as verified for M7.1; [`MILESTONE_7.md` §"Verification results (M7.0.5)"](MILESTONE_7.md)).

If we want to disentangle "demo at train time" vs "demo at eval time" effects, that's a separate ablation (M7.4?) — not in scope for M7.3.

## Phase plan

Mirror the M7.1 cadence:

| phase | target | wall | $ | decision |
|---|---|---:|---:|---|
| M7.3.0 — prompt change + render-parity verify | byte-equal training vs eval render at new prompt length | 5 min | ~$0 | proceed only on byte-equal |
| M7.3.1 — short100 probe | 100 GRPO steps × 320 traj | ~3 h | ~$5 | check tool-use rate at step 25, 50, 100 |
| M7.3.GO | tool-use rate ≥ 5% at step 100 → continue to extend | — | — | else → declare prompt-only fix insufficient, write up as null result |
| M7.3.2 — extend | step 100 → 622 (or earlier kill at step 134-style if pattern is clear) | ~6 h | ~$10 | full or early-stop on signal stability |
| M7.3.3 — eval | Plan A 7-dataset eval on trained ckpt | ~2.5 h | ~$3 | reuse [`training_m7_1/scripts/eval_m7_2.sh`](../../training_m7_1/scripts/eval_m7_2.sh) |

**Total budget: ~$15-20, ~12 h wall.**

## Success criteria

A: **Tool-use survives** (% trajectories emitting `<tool_call>` ≥ 10% at step 100). Then continue to step 622, eval on Plan A, compare to M7.1's tool-bypass results. Headline: "demo prior dominates F1 gradient at this scale."

B: **Tool-use collapses again** (% `<tool_call>` ≤ 1% at step 100, same as M7.1 collapse rate). Then stop at step 100, write up "demo prior insufficient — reward design needed." Headline: "Search-R1's reward floor is load-bearing."

Either outcome is publishable. There's no "wasted run" in this design — the answer to "does demo prevent collapse" is itself the M7.3 contribution.

## Risks

1. **Demo content quality.** A poorly-chosen demo (factually wrong, awkward bridge, format mismatch) trains the model to emit gibberish. Mitigation: review demo content with user before launch.
2. **Prompt-length tax on compute.** +400 tokens × 320 trajectories × generation step adds ~3-5% wall-clock. Negligible vs the M7.1 baseline.
3. **Render-parity drift.** If `apply_chat_template` doesn't render the new template byte-identically between training and eval pipelines, the trained model sees a different context at eval than at training. Mitigation: rerun the M7.0.5 verification before launching.
4. **Demo memorization.** The model might just learn to repeat the demo's exact form (verbatim copying of Apple/Jobs/Moon) on every MuSiQue question. Mitigation: read sample trajectories at step 25 — if the model is reproducing the demo's entity names instead of substituting the actual question's entities, the demo is doing the wrong thing and we kill early.

## Pointers

- M7.1 closing finding: [`MILESTONE_7.md` §"M7.1 closing finding"](MILESTONE_7.md)
- Current M7.1 prompt (to edit): [`training_m7_1/src/prompts/m7_qwen35_base_user.txt`](../../training_m7_1/src/prompts/m7_qwen35_base_user.txt)
- Eval template (to mirror): [`evaluation_qwen35/flashrag/search_r1/templates.py`](../../evaluation_qwen35/flashrag/search_r1/templates.py) (`QWEN35_MINIMAL_NO_SYSTEM_TEMPLATE`)
- Training config (reuse): [`training_m7_1/configs/m7_1_short100.yaml`](../../training_m7_1/configs/m7_1_short100.yaml) (clone to `m7_3_short100.yaml`)
- Launcher (clone): [`training_m7_1/scripts/run.sh`](../../training_m7_1/scripts/run.sh) (`--mode m73_short100`)
- Eval driver (reuse as-is): [`training_m7_1/scripts/eval_m7_2.sh`](../../training_m7_1/scripts/eval_m7_2.sh)
- Watchers (clone, retarget PID): [`training_m7_1/scripts/watch_extend.sh`](../../training_m7_1/scripts/watch_extend.sh), [`training_m7_1/scripts/auto_eval_after_extend.sh`](../../training_m7_1/scripts/auto_eval_after_extend.sh)
