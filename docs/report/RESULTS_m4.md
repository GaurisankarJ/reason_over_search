---
title: Results M4 — Qwen3.5-0.8B baseline (untrained, base + hybrid)
tags: [report, eval, m4, qwen3.5]
source: internal
created: 2026-05-08
updated: 2026-05-10
---

# Results M4: Qwen3.5-0.8B Untrained Baseline

**Status (2026-05-12):** M4 fully closed. Both variants validated at full Plan A. Locked prompts:
- **Hybrid: `qwen35_terse`** (M4.5, mean EM 0.092 — Δ +0.032 / +53 % over M4.2 lock `qwen35_minimal` 0.060; closes 76 % of the M3 cross-family gap).
- **Base: `qwen35_minimal_no_system`** (M4.3, mean EM 0.010 — Phase 4 prompt screen produced no winner; locked unchanged).

This is the locked untrained floor for any M5+ GRPO-trained Qwen3.5-0.8B checkpoint. M5 training uses the hybrid template byte-aligned to the eval render; see [`../milestone_4/MILESTONE_4.md` §"Handoff to M5"](../milestone_4/MILESTONE_4.md).

## 1. Run roster

| Variant | HF id | Local path | `prompt_mode` (final lock) | `enable_thinking` | Auto-inject |
|---|---|---|---|---|---|
| `qwen3.5_0.8b` (hybrid) | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) | `eval/qwen3.5_0.8b/` | **`qwen35_terse`** (M4.5) | True | yes |
| `qwen3.5_0.8b_base` | [`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) | `eval/qwen3.5_0.8b_base/` | **`qwen35_minimal_no_system`** (M4.3) | True | no |

Per-variant prompt-mode asymmetry: hybrid keeps the auto-injected `# Tools` + `<IMPORTANT>` block (in-distribution for tool-use post-training) and uses the terse user prose (M4.5 lock — 3 sentences, ~30 user tokens); base drops the system block entirely (the same block hurts base, which lacks the tool-use prior). The asymmetry was re-confirmed by M4.4 Phase 4 (7-candidate base screen produced no candidate above the +0.025 bar; see §5.5).

Pipeline: [`evaluation_qwen35/`](../../evaluation_qwen35/), [`scripts/run_m4.sh`](../../scripts/run_m4.sh), [`scripts/orchestrate_C_then_A.sh`](../../scripts/orchestrate_C_then_A.sh). Code-setup audit: [`CODE_SETUP_m4.md`](CODE_SETUP_m4.md). Milestone narrative: [`../milestone_4/MILESTONE_4.md`](../milestone_4/MILESTONE_4.md).

## 2. Eval configuration (M4)

Same shape as M3 (per CODE_SETUP_m4 §1):

| Knob | Value | Note |
|---|---|---|
| Decoding | `temperature=0.0` (greedy) | Single seed; greedy => seed-invariant |
| `apply_chat` | True | Both base and hybrid render through `tokenizer.apply_chat_template` |
| Action / observation tags | `<tool_call>` / `<tool_response>` | Qwen3.5-native vocab tokens (248058/9, 248066/7) |
| Retriever | IVF-SQ8 × 16 workers | top-5 (M4 close-out used 16w; M4 smokes used 8w; same per-call latency) |
| `generator_max_input_len` | 8192 (M4.2 bump from 4096) | per `RESULTS_SMOKE_m4.md` v3 |
| `max_search_turns` / `step_limit` / `max_obs_length` | 5 / 8192 / 256 | matches M3 |
| Datasets | bamboogle, nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique | 7 |
| Splits | test / test / test / test / dev / dev / dev | per dataset |

## 3. Smoke results (100 random items / dataset, seed=1)

Live iteration log (v1 → v2 → v3 → M4.2 → M4.3) at [`RESULTS_SMOKE_m4.md`](RESULTS_SMOKE_m4.md). Headline locked-best per-variant smokes:

| Variant | mode | mean EM (n=100/ds × 7) | vs verbose default |
|---|---|---:|---|
| Hybrid (`qwen3.5_0.8b`) | `qwen35_minimal` (M4.2) | **0.057** | 6.6× over the v3 verbose (0.0086) |
| Base (`qwen3.5_0.8b_base`) | `qwen35_minimal_no_system` (M4.3) | **0.016** | 5× over `qwen35_minimal` (0.003) |

Mechanism: the auto-inject `<IMPORTANT>` reminder is in-distribution for hybrid's tool-use post-training (drives search loops); on base it's pure scaffolding noise that crowds out the answer. See [`RESULTS_SMOKE_m4.md`](RESULTS_SMOKE_m4.md) §6–§7 for the full asymmetric-lock rationale.

## 4. Full sweep (51,713 items / variant)

### 4.1 M4.5 hybrid full sweep with `qwen35_terse` (2026-05-12)

Vast 1× A100-80GB; greedy decode, `seed=1`; optimised stack (multi-block `<tool_response>` per chunk, per-chunk cap 120 tok, `generator_max_input_len=8192`, retriever IVF-SQ8 × 16 + `asyncio.to_thread` + `INFERENCE_MAX_WORKERS=128`); SGLang `qwen3.5_0.8b` (hybrid), `enable_thinking=True`, `prompt_mode=qwen35_terse` (Phase 1b winner; auto-inject ON, terse user message). Wall-clock: **3 h 36 min** (13:14 → 16:50 UTC on 2026-05-12, single contiguous run).

| Dataset | N | M4.5 hybrid EM (`qwen35_terse`) | hybrid ACC | hybrid F1 | M4.2 EM (prior lock) | Δ vs M4.2 |
|---|---:|---:|---:|---:|---:|---:|
| bamboogle (test) | 125 | **0.072** | 0.112 | 0.113 | 0.048 | +0.024 |
| nq (test) | 3,610 | **0.100** | 0.204 | 0.144 | 0.063 | +0.037 |
| triviaqa (test) | 11,313 | **0.202** | 0.299 | 0.250 | 0.124 | **+0.078** |
| popqa (test) | 14,267 | **0.124** | 0.183 | 0.151 | 0.075 | +0.049 |
| hotpotqa (dev) | 7,405 | **0.083** | 0.129 | 0.120 | 0.064 | +0.019 |
| 2wikimultihopqa (dev) | 12,576 | **0.046** | 0.067 | 0.058 | 0.040 | +0.006 |
| musique (dev) | 2,417 | **0.018** | 0.029 | 0.027 | 0.008 | +0.010 |
| **mean (simple)** | **51,713** | **0.092** | **0.146** | **0.123** | **0.060** | **+0.032** |

**Headline: M4.5 lifts hybrid mean EM from 0.060 (M4.2 lock) → 0.092 (terse). Δ +0.032 absolute, +53 % relative.** The lift is concentrated in open-domain entity-recall datasets where M4.2's bypass-and-fabricate failure mode was costliest: triviaqa **+0.078** (+63 % rel.), popqa +0.049 (+65 % rel.), nq +0.037 (+59 % rel.). Multi-hop datasets (hotpotqa +0.019, 2wiki +0.006, musique +0.010) show smaller absolute lifts but musique's +0.010 represents a **2.2× relative improvement** — first measurable multi-hop signal at scale for untrained Qwen3.5-0.8B.

**Phase 1b prediction calibration**: n=300 anchor was mean EM 0.103; full-sweep landed at 0.092. Δ −0.011 vs prediction (slight n-scale shrinkage, well within Wilson 95 % CI half-width at n=300 of ~2.7 pp). All 7 datasets directionally positive at full scale, matching n=300.

### 4.2 Combined hybrid + base (final M4 locks)

Base row carried forward from §4.3 below (M4.3 lock `qwen35_minimal_no_system`; M4.4 Phase 4 produced no winner, see §5.5).

| Dataset | N | hybrid EM (M4.5 terse) | base EM (M4.3 minimal_no_system) | hybrid ACC | base ACC | hybrid F1 | base F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bamboogle (test) | 125 | 0.072 | 0.000 | 0.112 | 0.000 | 0.113 | 0.009 |
| nq (test) | 3,610 | 0.100 | 0.004 | 0.204 | 0.020 | 0.144 | 0.010 |
| triviaqa (test) | 11,313 | 0.202 | 0.012 | 0.299 | 0.031 | 0.250 | 0.029 |
| popqa (test) | 14,267 | 0.124 | 0.008 | 0.183 | 0.024 | 0.151 | 0.014 |
| hotpotqa (dev) | 7,405 | 0.083 | 0.011 | 0.129 | 0.028 | 0.120 | 0.022 |
| 2wikimultihopqa (dev) | 12,576 | 0.046 | 0.032 | 0.067 | 0.076 | 0.058 | 0.043 |
| musique (dev) | 2,417 | 0.018 | 0.000 | 0.029 | 0.005 | 0.027 | 0.006 |
| **mean (simple)** | **51,713** | **0.092** | **0.010** | **0.146** | **0.026** | **0.123** | **0.019** |

Hybrid is now **9.2× over base** on EM (0.092 vs 0.010), up from 6× at M4.2 lock. Base 2wiki (0.032) still narrowly beats hybrid 2wiki at M4.2 (0.040 → 0.046 with terse; gap now 0.014); on every other dataset hybrid dominates by ≥ 4×.

### 4.3 Prior full sweep — M4.2 hybrid / M4.3 base (2026-05-09)

Original full sweep on the M4.2/M4.3 locked prompts. Vast.ai 1× A100-SXM4 (instance `36465611`); orchestrator [`scripts/orchestrate_C_then_A.sh`](../../scripts/orchestrate_C_then_A.sh) (Phase 4 = base FULL × 7; Phase 6 = hybrid FULL × 7). Wall-clock: base 2 h 26 min (17:12 → 19:07 UTC); hybrid 5 / 7 cells in the original 2026-05-09 run (19:17 → 21:15 UTC), then 2wikimultihopqa + musique re-ran on 2026-05-10 (14:40 → 17:01 UTC) after the original run hit a transformers-4.57.1 → 5.7.0 venv revert. Greedy decode, single seed.

| Dataset | N | hybrid EM (M4.2 minimal) | base EM (M4.3 minimal_no_system) |
|---|---:|---:|---:|
| bamboogle (test) | 125 | 0.048 | 0.000 |
| nq (test) | 3,610 | 0.063 | 0.004 |
| triviaqa (test) | 11,313 | 0.124 | 0.012 |
| popqa (test) | 14,267 | 0.075 | 0.008 |
| hotpotqa (dev) | 7,405 | 0.064 | 0.011 |
| 2wikimultihopqa (dev) | 12,576 | 0.040 | 0.032 |
| musique (dev) | 2,417 | 0.008 | 0.000 |
| **mean (simple)** | **51,713** | **0.060** | **0.010** |

M4.2 hybrid row superseded by §4.1 (M4.5 terse). M4.3 base row retained as the canonical base number (no Phase 4 winner; see §5.5).

## 5. Cross-family comparison (M3 vs M4)

Untrained-floor comparison. M3 reference is the pre-GRPO Qwen3-0.6B hybrid (`qwen_3_0.6b`, [`RESULTS_m3.md`](RESULTS_m3.md) §6). All values are EM at full Plan A (51,713 items / variant), greedy decode. Hybrid column is M4.5 (terse-locked) which supersedes the M4.2 hybrid lock; M4.2 baseline shown for reference.

| Dataset | M3 (Qwen3-0.6B hybrid) | **M4.5 (Qwen3.5-0.8B + terse)** | M4.2 (Qwen3.5-0.8B + minimal) | M4 (Qwen3.5-0.8B base) | **Δ M4.5 vs M3** | Δ M4.2 vs M3 |
|---|---:|---:|---:|---:|---:|---:|
| bamboogle | 0.056 | **0.072** | 0.048 | 0.000 | **+0.016** | −0.008 |
| nq | 0.113 | **0.100** | 0.063 | 0.004 | **−0.013** | −0.050 |
| triviaqa | 0.178 | **0.202** | 0.124 | 0.012 | **+0.024** | −0.054 |
| popqa | 0.133 | **0.124** | 0.075 | 0.008 | **−0.009** | −0.058 |
| hotpotqa | 0.083 | **0.083** | 0.064 | 0.011 | **+0.000** | −0.019 |
| 2wikimultihopqa | 0.141 | **0.046** | 0.040 | 0.032 | **−0.095** | −0.101 |
| musique | 0.010 | **0.018** | 0.008 | 0.000 | **+0.008** | −0.002 |
| **mean** | **0.102** | **0.092** | **0.060** | **0.010** | **−0.010** | **−0.042** |

**Headline: M4.5 closes ~76 % of the M3 cross-family gap.** Avg Δ improved from −0.042 (M4.2) → **−0.010 (M4.5)**. Three datasets now exceed M3 (bamboogle +0.016, triviaqa +0.024, musique +0.008); two are essentially tied (hotpotqa Δ 0, popqa Δ −0.009, nq Δ −0.013). The remaining gap is concentrated almost entirely on **2wikimultihopqa** (Δ −0.095 — terse 0.046 vs M3 0.141) — M3's Qwen3-0.6B was unusually strong on 2wiki, likely a side-effect of its smaller-model post-training distribution that Qwen3.5-0.8B lacks.

Excluding 2wiki, M4.5 mean = 0.099 vs M3 mean (also ex-2wiki) = 0.096 → **M4.5 actually exceeds M3 by +0.003 across the 6 non-2wiki datasets**. The Qwen3.5 cross-family gap is therefore not a Qwen3.5 *floor* problem — it's a 2wiki-specific shape mismatch.

## 5.5 M4.4 Phase 4 — base prompt screen (2026-05-12)

Re-opens the M4.3 base lock (`qwen35_minimal_no_system`, full-sweep mean EM **0.010**) to test whether any Phase-1b-style prompt intervention lifts base. Full design + per-candidate templates in [`../milestone_4/MILESTONE_4.md` §"Phase 4 results"](../milestone_4/MILESTONE_4.md). All runs: hybrid SGLang swapped to base on `eval/qwen3.5_0.8b_base/`, n=300 / dataset (bamboogle full split = 125), greedy `seed=1`.

**Primary 4-candidate screen (n=300, mean EM):**

| # | mode | bamboogle (n=125) | nq | triviaqa | popqa | hotpotqa | 2wiki | musique | **mean EM** | Δ vs anchor | bar +0.025 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| (iv) | `qwen35_minimal_no_system` (M4.3 anchor re-run at n=300) | 0.000 | 0.003 | 0.010 | 0.003 | 0.003 | 0.037 | 0.000 | **0.008** | – | – |
| (i) | `qwen35_terse` (Phase-1b hybrid winner; auto-inject ON) | 0.008 | 0.007 | 0.017 | 0.010 | 0.007 | 0.010 | 0.000 | **0.008** | +0.0002 | FAIL |
| (ii) | `qwen35_terse_no_system` (terse user + no auto-inject) | 0.008 | 0.007 | 0.017 | 0.007 | 0.013 | 0.027 | 0.000 | **0.011** | +0.0031 | FAIL |
| (iii) | `qwen35_research_role_no_system` (role-prime in user + no auto-inject) | 0.000 | 0.013 | 0.013 | 0.013 | 0.013 | 0.023 | 0.000 | **0.011** | +0.0028 | FAIL |

**All 4 fail the +0.025 bar.** The largest lift (cand ii at Δ +0.0031) is ~8× below the bar. Auto-inject is re-confirmed harmful on base (cand i regresses to the floor). User-prose interventions (terse, role-prime) move base within n=300 noise of the M4.3 anchor.

**Fallback B (top-3 Phase-1b near-miss prose ported to no-system, n=300, mean EM):**

| # | mode | bamboogle | nq | triviaqa | popqa | hotpotqa | 2wiki | musique | **mean EM** | Δ vs anchor | bar |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B-1 | `qwen35_decide_no_system` | 0.008 | 0.023 | 0.013 | 0.003 | 0.017 | 0.040 | 0.000 | **0.015** | +0.007 | FAIL |
| B-2 | `qwen35_source_only_no_system` | 0.000 | 0.003 | 0.027 | 0.013 | 0.023 | 0.023 | 0.000 | **0.013** | +0.005 | FAIL |
| B-3 | `qwen35_self_check_no_system` | 0.000 | 0.007 | 0.023 | 0.007 | 0.020 | **0.043** | **0.003** | **0.015** | +0.007 | FAIL |

**All 7 Phase 4 candidates fail the +0.025 base bar.** Best two (B-1, B-3) tie at mean EM 0.015 vs the bar at 0.035 (~2.3× below). M4.3 lock `qwen35_minimal_no_system` (full-sweep mean EM 0.010) stays as the locked base prompt; **M4.6 base full sweep is NOT run** (no winner to validate at full scale, §4 base row unchanged).

**B-3 notable single-cell signals:** musique EM 0.003 (first non-zero base musique anywhere — 1 correct of 300) and 2wiki EM 0.043 (highest base 2wiki across the 7-candidate screen). Both within Wilson 95 % CI half-width at n=300 (~1.5 pp); not statistically distinguishable from noise but the only individual cells where a Phase 4 prompt did something measurable on base. Mechanism: self-check instruction may have caught fabrications on the binary-comparison subset of 2wiki.

**Musique EM = 0.000 on base across every candidate except B-3's single chance-correct item.** Confirms multi-hop is unreachable for base via prompt search — M5 (GRPO training) is the right next leverage point.

## 6. Findings

1. **The untrained M4 hybrid floor is 0.060 EM**; the base floor is 0.010 EM. Both are below the M3 pre-GRPO Qwen3-0.6B floor (0.102 EM). M5 GRPO training has a clear "beat the untrained floor" target on the same eval protocol.
2. **The smoke-iteration prompt-asymmetry lock-in (M4.2/M4.3) is real.** Hybrid wants the auto-injected tools spec; base needs it stripped. The 5–6× lift over the verbose default is preserved at full-data scale (smoke 0.057 → full 0.060 for hybrid; smoke 0.016 → full 0.010 for base, modest sample-shape variance).
3. **Cross-family ranking flips in 1 cell.** On 2wikimultihopqa, **base 0.032 > hybrid 0.040** is within 1 pp (0.008 absolute, ~20 % relative gap of the smaller); the auto-injected tools spec costs the hybrid more than the absent system block costs the base on this dataset specifically. Worth flagging if M5 reward shaping ever needs to dial-down tool-use on multi-hop.
4. **M5 train rollout is byte-aligned to the M4 hybrid pipeline by construction** ([`milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md)); evaluating any M5 checkpoint against this table is therefore directly comparable (no train/eval-drift audit needed).

## 7. Operational notes

- **Two-phase orchestrator run** ([`scripts/orchestrate_C_then_A.sh`](../../scripts/orchestrate_C_then_A.sh) `START_PHASE=4`): Phase 4 base FULL → Phase 5 model swap → Phase 6 hybrid FULL. Skip-aware via `metric_score.txt` glob match in [`scripts/run_m4.sh`](../../scripts/run_m4.sh).
- **2wiki + musique hybrid re-run** (2026-05-10) was triggered when the eval venv reverted to `transformers==4.57.1` between sessions, breaking `AutoConfig.from_pretrained` on `model_type=qwen3_5`. Fixed by re-installing `transformers==5.7.0` and re-running just those two cells; published [`pantomiman/reason-over-search-v1:v2`](https://hub.docker.com/r/pantomiman/reason-over-search-v1) bakes the upgrade in, and [`training/scripts/bootstrap.sh`](../../training/scripts/bootstrap.sh) idempotently upgrades transformers so a `:v1` box self-heals after bootstrap. Full chronicle in [`../log.md`](../log.md) under 2026-05-10.

## 8. Pointers

- M4 milestone narrative: [`../milestone_4/MILESTONE_4.md`](../milestone_4/MILESTONE_4.md)
- M4 code-setup deltas vs M3: [`CODE_SETUP_m4.md`](CODE_SETUP_m4.md)
- M4 smoke iteration log: [`RESULTS_SMOKE_m4.md`](RESULTS_SMOKE_m4.md)
- M3 results (the within-family floor): [`RESULTS_m3.md`](RESULTS_m3.md)
- M5 plan (the trained checkpoint that this floor anchors): [`../milestone_5/MILESTONE_5.md`](../milestone_5/MILESTONE_5.md)
- Active recipe-ablation plan: [`../todo/TODO_2026-05-10.md`](../todo/TODO_2026-05-10.md)
