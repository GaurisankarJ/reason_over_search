---
title: Code Setup v3 (M4 Qwen3.5-0.8B baseline evaluation pipeline)
tags: [report, eval, m4]
source: internal
created: 2026-05-08
updated: 2026-05-08
---

# Code Setup v3 — M4 Qwen3.5-0.8B Evaluation Pipeline: What Changed vs the M3 Pipeline

**Date**: 2026-05-08
**Scope**: Documents what changed from the M3 evaluation pipeline ([`evaluation_research/`](../../evaluation_research/), Qwen3-0.6B with Search-R1 invented `<search>` / `<result>` tags, audited in [`CODE_SETUP_v2.md`](CODE_SETUP_v2.md)) to the M4 pipeline ([`evaluation_qwen35/`](../../evaluation_qwen35/)) for the **Qwen3.5-0.8B base + hybrid baselines** with the family-native `<tool_call>` / `<tool_response>` tag scheme.
**Cluster (M4 eval)**: ALICE 1× A100-80GB (`gpu-short` + `gpu-a100-80g`).
**Source paths**: [`evaluation_qwen35/`](../../evaluation_qwen35/) (overlay copy of `evaluation_research/` with the M4 deltas), [`scripts/run_m4.sh`](../../scripts/run_m4.sh), [`scripts/sbatch_m4.sh`](../../scripts/sbatch_m4.sh), [`scripts/m4_download_models.sh`](../../scripts/m4_download_models.sh).

---

## 1. Headline Diff vs the M3 Pipeline (`evaluation_research/`)

| Dimension | M3 (`evaluation_research/`) | M4 (this doc, `evaluation_qwen35/`) |
|---|---|---|
| **Model family** | Qwen3-0.6B (hybrid) | **Qwen3.5-0.8B** (base + hybrid; both untrained) |
| **HF repos** | `Qwen/Qwen3-0.6B`, `pantomiman/Qwen3-0.6B-v0`, `pantomiman/Qwen3-0.6B-v0.1` | **`Qwen/Qwen3.5-0.8B`** + **`Qwen/Qwen3.5-0.8B-Base`** |
| **Action tag** | `<search>` / `</search>` (Search-R1 invention; not in pre-training) | **`<tool_call>` / `</tool_call>`** (Qwen3.5-native vocab tokens 248058 / 248059; in distribution from post-training) |
| **Observation tag** | `<result>` / `</result>` (M3 verl-legacy training rollout convention) | **`<tool_response>` / `</tool_response>`** (Qwen3.5-native vocab tokens 248066 / 248067) |
| **Prompt template (M3 `p1_basic_w_ex` ↔ M4 verbatim with renamed tags)** | `QWEN3_0_6B_TEMPLATE` | **`QWEN35_0_8B_TEMPLATE`** (same prose; only `<search>` ↔ `<tool_call>`, `<result>` ↔ `<tool_response>`) |
| **prompt_mode** | `qwen3` / `qwen3_p1_basic_w_ex` / `qwen3_p3_decide_no_ex` | **`qwen35`** / **`qwen35_p1_basic_w_ex`** |
| **Action stop tokens** | `[</search>, </answer>, <\|im_end\|>, <\|endoftext\|>]` | `[</tool_call>, </answer>, <\|im_end\|>, <\|endoftext\|>]` |
| **Result wrapper** | `" <result>\n{X}\n</result>"` (leading space) | `" <tool_response>\n{X}\n</tool_response>"` (leading space, parallel form) |
| **`enable_thinking`** | True (Qwen3 hybrid) | True for hybrid; False for base (chat template's auto-injected `<think>\n\n</think>` is harmless on a base model) |
| **Per-mode budgets** | `max_search_turns=5`, `step_limit=8192`, `max_obs_length=256`, `retrieval_topk=5`, `generator_max_input_len=4096` | identical (M3 and M4 share the same budget shape for cross-family comparability) |
| **Retrieval text format** | raw `{contents}\n\n` joined and stripped | identical |
| **Retriever** | IVF-SQ8 × 8 workers (default); flat IP × 2 (paper-fidelity opt-in) | identical (project-root `corpus/`, `indexes/`, `models/`) |
| **SGLang launch flags** | `--context-length 8192 --dtype bfloat16 --trust-remote-code` | identical |
| **Eval venv** | `/home/s4374886/.conda/envs/evaluation_search_r1` (with `evaluation_research/` editable-installed) | **same conda env**; `evaluation_qwen35/flashrag/` resolves via cwd-precedence (script-dir is sys.path[0] when run via `cd evaluation_qwen35 && python run_eval.py`); no editable install needed |
| **Quick-eval (100 items)** | Not wired (see CODE_SETUP_v2 §10: `sample_num` was unused) | **`--test_sample_num 100 --random_sample True --seed 1`** plumbed through `run_eval.py` to FlashRAG's `Dataset.sample_num` (subsample at load time); `random.seed(config["seed"])` set in `run_eval.py` for determinism |

---

## 2. What's Unchanged (M3 14-fix audit holds verbatim)

The 14 alignment fixes catalogued in [`CODE_SETUP_v2.md`](CODE_SETUP_v2.md) §3 transfer to M4 unchanged:

- Project-root `corpus/`, `indexes/`, `models/` symlinks (#1)
- `pip install ninja` in the conda env (#2)
- `module load CUDA/12.4.0` + `source /etc/profile.d/lmod.sh` (#3)
- Single-brace template in `QWEN35_0_8B_TEMPLATE` (#4 carries forward)
- `extract_solution` unwraps `\boxed{X}` from `<answer>...</answer>` content (#5; reused unchanged)
- Leading-space `" <tool_response>\n{X}\n</tool_response>"` (#6; same intent as M3's leading-space `<result>`)
- Raw `{contents}\n\n` retrieval text (#7; same)
- top-5 retrieval (#8; same)
- per-mode budgets (#9; same shape)
- `generator_max_input_len: 4096` (#10; same)
- `enable_thinking=True` for hybrid (#11; base uses False)
- `SLURM_SUBMIT_DIR` for `REPO_ROOT` (#12; same)
- Retriever 1200 s readiness wait (#13; same as M3.1)
- HF arrow cache hard-link (#14; n/a if cache already warm from M3.1)

---

## 3. New M4-Specific Fixes (3 net-new deltas)

### 3.1 `get_generator` short-circuits on `framework=sgl_remote` before VL detection

**Symptom:** Qwen3.5 ships with `vision_config` in its `config.json` (the family is the VL-co-pretrained one — `Qwen3_5ForConditionalGeneration`). FlashRAG's [`flashrag/utils/utils.py:50`](../../evaluation_qwen35/flashrag/utils/utils.py) checked `if all(["vision" not in key for key in model_config.keys()])` and routed to `HFMultiModalGenerator` whenever any `vision_*` config key was present. With M4 we want SGLang to do the inference (`framework=sgl_remote`); the in-process VL check should never run for the remote-HTTP path.

**Fix:** moved the `framework == "sgl_remote"` branch *before* the VL detection. The remote path returns `SGLRemoteGenerator(config, **params)` and skips the local config inspection altogether. Same fix benefits any future VL-flavoured model served via SGLang remote.

```python
# evaluation_qwen35/flashrag/utils/utils.py
if config['framework'] == 'openai':
    return getattr(...,"OpenaiGenerator")(config, **params)
# NEW: short-circuit before VL detection
if config["framework"] == "sgl_remote":
    return getattr(...,"SGLRemoteGenerator")(config, **params)
# (then the original VL / vLLM / hf / fschat dispatch)
```

Caught while inspecting the freshly-downloaded `Qwen/Qwen3.5-0.8B/config.json` on ALICE (commit `2be8e0a`).

### 3.2 `<tool_call>` flat-form parser + family-based dispatch in `active_pipeline.py`

The flat form (`<tool_call>X</tool_call>`) is a deliberate choice over Qwen3.5's nested XML form (`<tool_call><function=name><parameter=arg>X</parameter></function></tool_call>`; see [`docs/training/CHAT_TEMPLATE.md`](../training/CHAT_TEMPLATE.md) §2). The flat form keeps the prompt prose literal-identical to M3's `p1_basic_w_ex` (one-line query, one-line response), and avoids the auto-injected tools schema preamble that would otherwise prepend ~600 tokens of tool description on every prompt.

New parser ([`evaluation_qwen35/flashrag/search_r1/parser.py`](../../evaluation_qwen35/flashrag/search_r1/parser.py)):

```python
def extract_tool_call_query(text):
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not match: return "", "no_tool_call_tag"
    query = match.group(1).strip()
    return (query, "valid_tool_call_tag") if query else ("", "empty_tool_call_query")
```

Family-based dispatch ([`evaluation_qwen35/flashrag/pipeline/active_pipeline.py`](../../evaluation_qwen35/flashrag/pipeline/active_pipeline.py)):

```python
def _is_qwen3(mode):  return mode == 'qwen3'  or mode.startswith('qwen3_')   # NOT qwen35
def _is_qwen35(mode): return mode == 'qwen35' or mode.startswith('qwen35_')
def _is_qwen_family(mode): return _is_qwen3(mode) or _is_qwen35(mode)
```

The pipeline picks `action_stop`, the parser, and the result-wrapper template by family (qwen3 / qwen35 / search_r1) — single switch swaps the four behaviours atomically. Same per-mode budgets / retrieval format / `enable_thinking` apply to qwen3 + qwen35 (the families share the 4096-budget / top-5 / 256-obs / 5-turn shape so cross-family comparison is apples-to-apples).

### 3.3 Quick-eval plumbing (`--test_sample_num` / `--random_sample` / `--seed`)

CODE_SETUP_v2 §10 noted that `sample_num` was *not respected* by the M3 search_r1 path. M4 wires the FlashRAG `test_sample_num` / `random_sample` / `seed` config keys through CLI flags so the smoke run can hit 100 random items per dataset deterministically:

```bash
--test_sample_num 100 --random_sample True --seed 1
```

Implementation ([`evaluation_qwen35/run_eval.py`](../../evaluation_qwen35/run_eval.py)):

1. Three CLI flags added; if `--test_sample_num` is set, `random_sample` and `seed` are also threaded into `config_dict` (and propagated to `Config`).
2. Before `get_dataset(config)`, call `random.seed(config["seed"])` so FlashRAG's `Dataset._load_data` uses a deterministic shuffle when `random_sample=True` (FlashRAG itself uses the global `random` module without seeding it).
3. `save_note` includes `_n100` so quick-eval results don't collide with full-sweep results in `results/<dataset>/`.

Full sweep recovers the original behaviour by simply omitting all three flags.

---

## 4. Worktree-on-ALICE workflow (operational note, not a code fix)

ALICE's main checkout (`/zfsstore/user/s4374886/omega/reason_over_search/`) carries an active session for the M3.1 closure work on branch `research_v1`. To run the M4 smoke without disturbing that session, we created a **second git worktree** for `research_v2`:

```bash
# On ALICE:
MAIN=/zfsstore/user/s4374886/omega/reason_over_search
WT=/zfsstore/user/s4374886/omega/reason_over_search-m4
cd $MAIN
# LFS hooks are configured in the main repo but git-lfs binary is not on PATH on ALICE;
# disable the LFS filter for the one-shot worktree-add operation.
git -c core.hooksPath=/dev/null \
    -c filter.lfs.smudge= -c filter.lfs.process= -c filter.lfs.required=false \
    worktree add $WT research_v2
cd $WT
# Symlink the heavy / gitignored artifacts that live in the main checkout:
for d in eval corpus indexes models data data_subsample logs; do
  [[ -e $MAIN/$d ]] && [[ ! -e $WT/$d ]] && ln -sfn $MAIN/$d $d
done
# Now sbatch from the worktree — SLURM_SUBMIT_DIR resolves to $WT;
# scripts/sbatch_m4.sh + run_m4.sh exist here; eval/ etc. resolve through symlinks.
sbatch --time=01:00:00 scripts/sbatch_m4.sh qwen3.5_0.8b 100
sbatch --time=01:00:00 scripts/sbatch_m4.sh qwen3.5_0.8b_base 100
```

The worktree shares `.git/objects/` with the main checkout (no extra disk for the git history). Branches are isolated (the main checkout stays on `research_v1`; the worktree stays on `research_v2`). When the M3.1 session is done and merges, we can `git worktree remove $WT` and run M4 from the main checkout.

---

## 5. Variant dispatch + run recipe

| Variant | Path | `enable_thinking` | `prompt_mode` |
|---|---|---|---|
| `qwen3.5_0.8b` (hybrid) | `eval/qwen3.5_0.8b/` | True | `qwen35` |
| `qwen3.5_0.8b_base` | `eval/qwen3.5_0.8b_base/` | False | `qwen35` |

Quick eval (100 random items / dataset, seed 1):
```bash
sbatch --time=01:00:00 scripts/sbatch_m4.sh qwen3.5_0.8b 100
sbatch --time=01:00:00 scripts/sbatch_m4.sh qwen3.5_0.8b_base 100
```

Full sweep (51,713 items / variant):
```bash
sbatch scripts/sbatch_m4.sh qwen3.5_0.8b
sbatch scripts/sbatch_m4.sh qwen3.5_0.8b_base
```

Single-(variant, dataset, seed): [`scripts/run_m4.sh`](../../scripts/run_m4.sh) with same args + optional 5th positional `test_sample_num`.

---

## 6. Hardware sizing (unchanged from M3)

Same 8-worker IVF-SQ8 retriever (~134 GB RAM peak) + SGLang on 1× A100-80GB. CPU `--cpus-per-task=40`, `--mem=160g`. Smoke wall-clock target: ≤ 25 min total for both variants × 7 datasets × 100 items at 32 INFERENCE_MAX_WORKERS on a warm pipeline. Full sweep target: ~150 min / variant (M3 reference).

---

## 7. Pointers

- M3 alignment audit (the 14 fixes M4 inherits unchanged): [`CODE_SETUP_v2.md`](CODE_SETUP_v2.md) §3
- M4 milestone narrative: [`../milestone_four/MILESTONE_4.md`](../milestone_four/MILESTONE_4.md)
- M4 results table (when populated): `RESULTS_v3.md` (planned)
- Qwen3.5 chat template (verbatim): [`../training/CHAT_TEMPLATE.md`](../training/CHAT_TEMPLATE.md) §2
- Active recipe-ablation plan: [`../TODO_2026-05-04.md`](../TODO_2026-05-04.md)
