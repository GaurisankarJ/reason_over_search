# Plan B vs. Search-R1 paper — gap analysis

Side-by-side of [RESULTS_PLAN_B.md](RESULTS_PLAN_B.md) (1 seed × 7 datasets × 2 variants, factoid/multihop subsampled to 1 k, Bamboogle/MuSiQue full) against the GRPO numbers from Search-R1 v5 (Appendix F / Table 3). Paper targets and the 10 fixes already applied are tracked in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Per-dataset EM, ours vs paper

| Dataset | Variant | Plan B EM | Paper EM | Δ (pp) |
|---|---|---:|---:|---:|
| NQ              | base     | 0.316 | 0.421 | **−10.5** |
| NQ              | instruct | 0.399 | 0.397 | +0.2 |
| TriviaQA        | base     | 0.421 | 0.583 | **−16.2** |
| TriviaQA        | instruct | 0.539 | 0.565 | −2.6 |
| PopQA           | base     | 0.309 | 0.413 | **−10.4** |
| PopQA           | instruct | 0.412 | 0.391 | +2.1 |
| HotpotQA        | base     | 0.201 | 0.297 | **−9.6** |
| HotpotQA        | instruct | 0.354 | 0.331 | +2.3 |
| 2WikiMultiHopQA | base     | 0.207 | 0.274 | **−6.7** |
| 2WikiMultiHopQA | instruct | 0.353 | 0.310 | +4.3 |
| MuSiQue         | base     | 0.034 | 0.066 | −3.2 |
| MuSiQue         | instruct | 0.149 | 0.124 | +2.5 |
| Bamboogle       | base     | 0.112 | 0.128 | −1.6 |
| Bamboogle       | instruct | 0.360 | 0.232 | **+12.8** |

Averages:

| Variant  | Plan B | Paper | Δ |
|---       |---:    |---:   |---:|
| base     | 0.229  | 0.312 | **−8.3 pp** |
| instruct | 0.367  | 0.336 | +3.1 pp |

## Headline pattern

The two variants behave very differently against the paper:

- **Base** is below paper on **all 7 datasets** (−1.6 to −16.2 pp). The bias is one-sided and systematic — far beyond the ~1.5 pp subsample SE on 1 k-row factoid samples. Something is wrong with the base run, not just noise.
- **Instruct** is within ±5 pp on 6 of 7 datasets and overshoots Bamboogle by +12.8 pp. The +3.1 pp average lift is consistent with the Bamboogle smoke observation in [REPRODUCIBILITY.md](REPRODUCIBILITY.md#smoke-validation): instruct cleanly closes `</answer>`, while base length-truncates 17 % of the time and loses those examples.

So the instruct→paper gap is plausibly noise + a small systematic edge from cleaner stop behaviour. The base→paper gap is not. Plan A will not fix the base variant — it will just give us tighter error bars on a wrong number.

## What is the same as the paper

Confirmed by audit of the codebase against the official `PeterGriffinJin/Search-R1`:

| Knob | Ours | Paper |
|---|---|---|
| Models | `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-(it-)em-grpo` | same checkpoints |
| Underlying base | `Qwen/Qwen2.5-3B` | same |
| Retriever encoder | E5-base-v2 | same |
| Corpus | wiki18_100w | same |
| Index | FAISS Flat IP, 768-d | same |
| Top-k | 3 | 3 |
| Max search turns | 4 | 4 |
| Per-step token limit | 500 | 500 |
| Retrieval query max len | 256 | 256 |
| Observation truncation | 500 tokens | 500 tokens |
| Passage format | `Doc i(Title: …) <text>` | byte-identical |
| `<information>` whitespace | `\n\n` outside, none inside | byte-identical |
| Prompt template | `SEARCH_R1_TEMPLATE` (`flashrag/search_r1/templates.py`) | byte-identical to `make_prefix(template_type='base')` |
| Chat template (instruct) | `tokenizer.apply_chat_template()` | same |
| Stop tokens | `</search>`, `</answer>`, `<\|im_end\|>`, `<\|endoftext\|>` | same |
| `<search>` regex | first-match | same |
| Invalid-search corrective text | matches official wording | same |
| EM metric | SQuAD-canonical normalize → exact equality (`flashrag/search_r1/answer_utils.py`) | same |
| F1, ACC | token-level F1, sub-EM | same |

All 10 audit divergences listed in [REPRODUCIBILITY.md](REPRODUCIBILITY.md#divergences-fixed) have been applied.

## What is *not* the same / could deviate

Ranked by how much it could explain the −8 pp base-variant gap.

### High-suspicion (likely material)

1. **`apply_chat=False` on base** — the load-bearing miss. [PAPER_VS_OURS_AUDIT.md D1](PAPER_VS_OURS_AUDIT.md#d1-in-detail-the-load-bearing-one) shows verl's `rl_dataset.py:128` applies the chat template unconditionally when the tokenizer has one, and Qwen2.5-3B base ships with a chat template. The paper's reported base GRPO numbers therefore use chat-wrapped prompts. Our `scripts/run_one.sh:35` hard-codes `apply_chat=False` for base. The Bamboogle probe below already confirmed +1.6 pp lift and 84 %→100 % close-rate from flipping this. **In-flight as the next sweep on NQ-1k + TriviaQA-1k.**

   *(Note: an earlier version of this section listed `temperature=1.0, top_p=0.95` as suspect #1, citing the upstream `ppo_trainer.yaml` rollout block. That was incorrect — see [PAPER_VS_OURS_AUDIT.md D3](PAPER_VS_OURS_AUDIT.md): verl's `_validate()` at `ray_trainer.py:478,508` hard-codes `do_sample=False` and `vllm_rollout.py:162-171` overrides to `temperature=0, top_p=1.0`. The paper eval is greedy. Our `temperature: 0.0` is correct. Post-mortem: [docs/archive/TEMPERATURE_HYPOTHESIS_WRONG.md](archive/TEMPERATURE_HYPOTHESIS_WRONG.md).)*
2. **Base-variant trace hygiene.** Per the smoke profile: instruct hits `</answer>` 125/125, base length-truncates 21/125 on Bamboogle. Plan B does not record `format_valid` / `length_truncated` rates per-dataset — we are blind to whether base is silently failing on NQ/TriviaQA/PopQA/HotpotQA the same way. If `step_limit=500` is biting on factoid prompts, the base variant is being undercounted. Action: pull the `search_r1_format_valid` / `search_r1_extracted_answer` fields out of the existing run JSONs and tabulate truncation rate per (dataset, variant).
3. **Single seed at greedy.** Greedy decoding makes seed irrelevant in principle, but tokenizer/index nondeterminism + KV-cache effects at temp=0 can still produce ~1–2 pp run-to-run drift on n=1 k. With 5 seeds (Plan A) or 3 seeds (Plan C-lite) the per-dataset SE drops by √5 ≈ 2.2× and ±10 pp gaps stay ±10 pp.

### Medium-suspicion

4. **1 k subsample SE on multi-hop.** ~1.5 pp on factoid; bigger (~2.5–3 pp) on 2Wiki/HotpotQA where answer entropy is higher. Doesn't explain the base bias direction but adds noise to per-dataset deltas.
5. **SGLang launch flags.** Currently launching with `--disable-radix-cache --disable-overlap` (per [EVAL_OPS.md:89](EVAL_OPS.md)). These should not change EM at temp=0, but worth confirming with an A/B because they are a divergence from the upstream SGLang defaults.
6. **Bamboogle instruct overshoot (+12.8 pp).** Already analysed in [REPRODUCIBILITY.md:60-63](REPRODUCIBILITY.md). On n=125 with single seed this is ~3.4 σ out — not within noise, but the explanation (cleaner `</answer>` stop on instruct) is consistent with the same mechanism that drags base down. Same root cause, opposite signs across variants.

### Low-suspicion (already verified or improbable)

7. Index type — using Flat IP, recall=100 %; the IVF-SQ8 alternative loses <1 % recall and would only narrow the gap, not widen it.
8. Tokenizer / chat template handling — already verified byte-identical against upstream for the prompt template, and instruct numbers track the paper closely, which exercises the chat-template path.
9. Paper-version drift — Search-R1 has 5 arxiv revisions; we are comparing to v5 / Table 3. Earlier revisions report slightly different numbers, but not by 8 pp.

## Probe: base + `apply_chat=True` on Bamboogle (2026-04-28)

Hypothesis: the base GRPO checkpoint also benefits from the chat scaffold the instruct variant uses, even though `run_one.sh:35` hard-codes `apply_chat=False` for base. The base model's tokenizer config ([`search_r1_base_model/tokenizer_config.json`](../evaluation_search_r1/search_r1_base_model/tokenizer_config.json)) ships with a Qwen2.5 chat template (`"You are a helpful assistant."`), so `apply_chat=True` is well-defined for it.

Re-ran Bamboogle (test split, n=125, seed=1) with `--apply_chat True --generator_model search_r1_base_model`, save_note `search_r1_base_applychat_seed1`. Everything else unchanged.

| Run | EM | F1 | ACC | closes `</answer>` | mean turns | paper EM |
|---|---:|---:|---:|---:|---:|---:|
| Plan B base, no chat        | 0.112 | 0.172 | 0.120 | **105/125 (84 %)** | 0.95 | 0.128 |
| **Base + `apply_chat=True`** | **0.128** | **0.207** | **0.136** | **125/125 (100 %)** | 1.01 | 0.128 |
| Plan B instruct (chat)      | 0.360 | 0.451 | 0.376 | 125/125 (100 %)    | 1.94 | 0.232 |

What this shows:
- **Trace hygiene is fully fixed.** 20 of 125 samples in the no-chat base run never produced an `</answer>` close — they length-truncated mid-trace and lost their EM credit. With chat scaffolding the rate is 0/125.
- **EM hits the paper number on the nose** (0.128). The +1.6 pp lift on Bamboogle is small (within noise on n=125), but it is in the right direction and the mechanism (truncation rate) clearly improved.
- **Mean turns barely moved** (0.95 → 1.01), well below instruct's 1.94. Chat scaffolding fixes *how the trace ends*, not how aggressively the base model uses search. So the lift on multi-hop datasets (HotpotQA, 2Wiki) — which need >1 turn — may be smaller than on factoid; conversely, factoid datasets that need ~1 turn are exactly where the truncation losses dominated and where the lift should be largest.

Caveat: 16 % truncation rate is dataset-dependent. NQ/TriviaQA prompts are shorter than Bamboogle's reasoning-heavy ones, so the without-chat base could either be cleaner there (smaller lift) or just as broken (similar lift). Need to measure.

## Recommended next steps before Plan A

Plan A is 17 days of compute. Don't burn it on a setup that has a one-sided 8 pp gap on one variant.

Ordered by cost / information ratio:

1. **(in flight — highest priority, confirmed lift on Bamboogle) Re-run base variant on NQ-1k and TriviaQA-1k with `apply_chat=True`.** Bamboogle probe above closed the gap to paper exactly via this single flag flip and pushed format-validity 84 %→100 %. NQ/TriviaQA are where the −10 to −16 pp gaps live; if the same mechanism is at play, this is the fix. Modify [`scripts/run_one.sh:35`](../scripts/run_one.sh#L35) to set `apply_chat=True` for base, or invoke `run_eval.py` directly with `--apply_chat True --generator_model search_r1_base_model` and a save_note like `search_r1_base_applychat_seedN`.
2. **(5 min) Add the missing prompt sentence** `For example, <answer> Beijing </answer>.` back to [`flashrag/search_r1/templates.py`](../evaluation_search_r1/flashrag/search_r1/templates.py) — see [PAPER_VS_OURS_AUDIT.md](PAPER_VS_OURS_AUDIT.md#prompt-template-micro-divergence-negligible). Free; LOW severity.
3. **(5 min) Remove the `add_special_tokens` block** in [`flashrag/pipeline/active_pipeline.py:37-42`](../evaluation_search_r1/flashrag/pipeline/active_pipeline.py#L37-L42) — see [PAPER_VS_OURS_AUDIT.md D8](PAPER_VS_OURS_AUDIT.md#d8-special-token-additions). Smoke-test that `</search>` still matches as a stop string.
4. **(30 min, free) Pull format-validity and length-truncation rate from the rest of the existing Plan B base JSONs**, per dataset. Use `'</answer>' in final_response` as the close-rate signal. Anywhere the rate is <90 % is a candidate for the apply_chat fix above.
5. **(2 h) Re-run base variant on NQ-1k with `step_limit` raised** (500 → 1024) and check whether truncation rate drops and EM rises. Rules in/out the per-step cap as the cause. Only run if (1) leaves a residual gap.
6. **(4 h) One-seed full-NQ base run** (Plan C slice for one dataset, ~3 h on a 4090 with current flags). If full-NQ base EM is also ~0.32 (vs paper 0.421), the gap is not subsample noise and is reproducible at scale — Plan A on base would just confirm the same wrong number.
7. **(1 day) Add a 2nd and 3rd seed to Plan B** to bracket per-dataset variance before committing to 5-seed Plan A. Cheap insurance.
8. **Only after the above resolves the base gap to within ~3 pp on at least one dataset:** launch Plan A. Otherwise launch Plan C (3.4 days, full data, 1 seed) on the *fixed* config and decide based on those numbers whether the 5-seed extra precision is worth 17 days.

Do **not** change `temperature` or `top_p`. Paper eval is greedy ([PAPER_VS_OURS_AUDIT.md D3](PAPER_VS_OURS_AUDIT.md)). Our config is correct on this axis.

## Concrete next action (after the in-flight apply_chat sweep)

```bash
# 1. Apply audit fix #2: edit templates.py to restore "For example, <answer> Beijing </answer>."
# 2. Apply audit fix #3: remove add_special_tokens block in active_pipeline.py:37-42
# 3. Re-run base on NQ-1k + TriviaQA-1k with apply_chat=True; compare to paper.
scripts/run_one.sh base nq 1        # save_note: search_r1_base_applychat_seed1
scripts/run_one.sh base triviaqa 1
```

If base EM closes to within ~3 pp of paper on NQ-1k, the apply_chat flip is the fix. Else fall through to step_limit (#5).
