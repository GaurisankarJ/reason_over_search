# Search-R1 Reproducibility Audit

End-to-end audit of the local evaluation setup against the official Search-R1 implementation
(`PeterGriffinJin/Search-R1`). Cross-referenced files: `infer.py`,
`search_r1/llm_agent/generation.py`, `search_r1/search/retrieval_server.py`,
`scripts/data_process/{nq_search.py, qa_search_test_merge.py}`, `retrieval_launch.sh`,
`scripts/nq_hotpotqa/evaluate.sh`.

---

## 1. Datasets — correct source, schema, splits

| | Source | Format | Splits used | Status |
|---|---|---|---|---|
| Local | `data/<name>/{train,dev,test}.jsonl` | `{id, question, golden_answers, metadata}` | test if exists else dev | matches official |
| Official | `RUC-NLPIR/FlashRAG_datasets` (per `nq_search.py`, `qa_search_test_merge.py`) | identical | test if exists else dev | — |

Local row counts match the FlashRAG release exactly:

- Bamboogle test: 125
- HotpotQA dev: 7,405
- 2WikiMultiHopQA dev: 12,576
- MuSiQue dev: 2,417

**Missing locally:** `nq/test.jsonl` (3,610), `triviaqa/test.jsonl` (11,313),
`popqa/test.jsonl` (14,267) — same provenance, just need to download.

**Note (medium severity):** the official preprocessing in `qa_search_test_merge.py` does:

```python
example['question'] = example['question'].strip()
if example['question'][-1] != '?':
    example['question'] += '?'
```

The local pipeline (`SearchR1Pipeline.run_item`) consumes `item.question` as-is.
Some FlashRAG records don't end with `?`. Fixable in 2 lines; small but real
distribution shift.

---

## 2. Retriever — same model, corpus, index; different top-k and query length

| Component | Local | Official | Match? |
|---|---|---|---|
| Encoder | `intfloat/e5-base-v2`, mean pool, fp16 | same | yes |
| `query: ` / `passage: ` prefix | applied (`flashrag/retriever/utils.py:set_default_instruction`) | applied (`retrieval_server.py:Encoder.encode`) | yes |
| Corpus | `wiki18_100w.jsonl` (= wiki-18 from `PeterJinGo/wiki-18-corpus`) | `wiki-18.jsonl` (same hosted release) | yes |
| FAISS index | `wiki18_100w_e5_flat_inner.index` from `PeterJinGo/wiki-18-e5-index` (the paper's own release) | `e5_Flat.index` (same release) | yes |
| `retrieval_query_max_length` | **128** | **256** | divergence (medium) |
| `retrieval_topk` | **5** (FlashRAG default in `basic_config.yaml`) | **3** (`retrieval_launch.sh`, `evaluate.sh`, `infer.py`) | divergence (high) |
| Server API | `/search`, `/batch_search` returning `[{id, contents}]` | `/retrieve` returning `{result: [[{document: {contents}, score}]]}` | different shape (local code adapts) |

**top-k = 5 vs 3 is a real reproducibility gap.** The model was trained with
top-3 contexts. Top-5 changes the conditioning distribution and the EM ceiling
per turn.

---

## 3. Prompt template — identical text, instruct path matches

The `SEARCH_R1_TEMPLATE` in `flashrag/search_r1/templates.py` is **byte-identical**
to `make_prefix(template_type='base')` in the official `nq_search.py` and
`qa_search_test_merge.py`. The official only ever exposes `template_type='base'`;
both base and instruct training use the same prefix wrapped as a `{"role": "user"}`
message.

`SearchR1Pipeline.run_item` does:

- `apply_chat=False` (base): raw template — matches official.
- `apply_chat=True` (instruct): renders through Qwen chat template — matches
  `infer.py`'s `if tokenizer.chat_template` branch.
- `enable_thinking` flag: Qwen3-only feature, default `False`; not used by Search-R1.

**Unused / non-canonical:** `SEARCH_R1_TEMPLATE_SYS` (the "You are a helpful
assistant…" system prompt) is defined but the pipeline never uses it. Good — it
would be OOD if turned on.

---

## 4. Multi-turn loop — several behavioral divergences

| Knob | Local | Official (`evaluate.sh` / `generation.py` / `infer.py`) | Severity |
|---|---|---|---|
| `max_search_turns` | **8** | **4** (`max_turns=4` in `evaluate.sh`) | medium — extra turns shift the policy off training distribution |
| Per-step max new tokens | `step_limit = 512` | `max_response_length = 500` | minor |
| Observation truncation | none | `max_obs_length = 500` tokens, applied to every `<information>` block | medium — long passages can push later turns OOD or overflow |
| Prompt budget | `generator_max_input_len = 4096` | `max_prompt_length = 4096`, `max_start_length = 2048` | match |
| Stop tokens (per step) | `['</search>', '</answer>', '<\|im_end\|>', '<\|endoftext\|>']` | `</search>` variants + EOS; then post-step `split('</search>')[0]+'</search>'` else `split('</answer>')[0]+'</answer>'` | effectively equivalent |
| Sampling temperature | SGLang server default (1.0 unless overridden) | `temperature=1` in `evaluate.sh` rollout, `0.7` in `infer.py` demo | match for eval path (both 1.0) |
| Search-tag extraction | regex with `[-1]` (last `<search>...</search>` in step) | regex first match | minor (rare with stop tokens) |
| Empty / invalid `<search>` content | inserts `'nothing to search'` text | inserts hard-coded **corrective message** ("My previous action is invalid…") | low–medium — only fires on malformed outputs but is OOD when it does |

---

## 5. Retrieved-passage formatting — highest-severity divergence

This is the biggest concrete mismatch. The Search-R1 models are trained on a
very specific information-block format. The local pipeline does not produce it.

**Official** (`generation.py:_passages2string` + `execute_predictions`):

```
\n\n<information>Doc 1(Title: <title>) <text>
Doc 2(Title: <title>) <text>
Doc 3(Title: <title>) <text>
</information>\n\n
```

**Local** (`active_pipeline.py:138-142` + retriever returning raw `contents`):

```
\n<information>
<title1>
<text1>

<title2>
<text2>

...
</information>
```

Two distinct issues bundled here:

1. **No `Doc i(Title: …) …` wrapper.** Each passage is the raw `contents` field
   (`Title\nText`). The model never saw passages without `Doc N(Title: X)`
   framing during RL training.
2. **Different whitespace around `<information>`.** Official:
   `\n\n<info>X</info>\n\n`. Local: `\n<info>\nX\n</info>` (newlines *inside*
   the tags, single newline outside).

Both are likely to shave several points of EM, especially for the base-model
variant where the RL policy is more brittle.

---

## 6. Answer parsing & EM — match

- `normalize_answer` in `flashrag/search_r1/answer_utils.py` is the canonical
  Rajpurkar/SQuAD normalization. Same function in Search-R1's
  `verl/utils/reward_score/qa_em.py` upstream.
- `extract_solution` takes the **last** `<answer>…</answer>` match — same as
  official.
- Boxed/`<answer>` fallback in `_extract_pred_fallback` is more permissive than
  official; that only helps borderline cases.

The `search_r1_reward` and the format-validity checks in `reward.py` are local
additions for diagnostics; they don't affect EM.

---

## 7. Models & SGLang — checkpoints match, server flags fine

- `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo` (base) and
  `…-it-em-grpo` (instruct) — these are the released GRPO checkpoints used in
  the paper's 3B results.
- SGLang launch:
  `--context-length 8192 --dtype bfloat16 --disable-overlap --disable-radix-cache`.
  Disabling overlap/radix-cache is conservative for determinism — fine. The
  paper used vLLM, not SGLang, but both honor the same logits given identical
  sampling params.

---

## 8. Severity-ranked divergence list (what's likely costing EM)

| # | Divergence | Severity | Fix complexity |
|---|---|---|---|
| 1 | Information block not formatted as `Doc i(Title: …) …` | **High** | small |
| 2 | `retrieval_topk` 5 vs 3 | **High** | one-line config |
| 3 | `<information>` wrapper whitespace differs | **Medium-High** | small |
| 4 | `max_search_turns` 8 vs 4 | Medium | one-line |
| 5 | No `max_obs_length=500` truncation on retrieval text | Medium | small |
| 6 | Question normalization (`.strip()` + trailing `?`) missing | Low-Medium | 2 lines |
| 7 | `retrieval_query_max_length` 128 vs 256 | Low-Medium | one-line |
| 8 | Invalid-search fallback text differs | Low | small |
| 9 | Last-vs-first `<search>` regex match | Low | one-line |
| 10 | Per-step token limit 512 vs 500 | Negligible | — |

Sampling temperature, encoder, index, corpus, prompt template, and EM logic all
match. The remaining gaps are concentrated in (a) the retrieval-context surface
format the model sees and (b) decode-loop budgets — both of which are
training-time-fixed conventions in Search-R1 that should be mirrored.

---

## Bottom line

The local setup can reproduce **close to** the paper numbers as configured
today, but not within the typical ±1 EM noise band — the `Doc i(Title: …) …`
formatting and `top_k=3` are the two issues most likely to keep results
systematically below the published 3B Qwen2.5 numbers. The retriever, index,
encoder, models, prompt text, and EM scorer are all the right ones; the gaps
are in the orchestration layer (`SearchR1Pipeline`) and the FlashRAG defaults.

### Recommended next steps

1. Apply fixes 1–6 from the table above.
2. Run a smoke test on Bamboogle (125 examples, fastest dataset) before vs.
   after the changes to quantify the delta.
3. Re-run the full benchmark sweep (all 7 datasets × base/instruct × 3 seeds)
   once the smoke test confirms the gap closes.
