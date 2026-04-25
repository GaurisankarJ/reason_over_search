# Where the time is actually going

## Profile from the smoke run (Bamboogle base, 125 examples, 6m 03s wall clock)

| Quantity | Value |
|---|---|
| Wall-clock per example (8 concurrent workers) | **2.85 s** |
| Effective single-worker time | ~22.8 s |
| `prompt_tokens` (last turn) | mean 569, p95 813, max 1,163 |
| `completion_tokens` (last turn) | mean 99, median 11, p95 500 |
| Search turns per example | mean 1.03, max 2 |
| LLM calls per example | ~2 (initial → `</search>`, continuation → `</answer>`) |
| Retrieval calls per example | ~1 |
| GPU utilization while idle | 0% (model resident, 22 GB used of 24 GB) |
| Eval-side concurrency | `INFERENCE_MAX_WORKERS = 8` |

## Where the per-example time goes

Roughly per example, on a 4090 with the current SGLang flags:

```
GPU  ──  prefill ~600–1500 token prompt × 2 calls       ~0.4–0.6 s
GPU  ──  decode ~50–150 tokens × 2 calls (~80 tok/s)    ~0.8–1.6 s
CPU  ──  FAISS Flat IP over 21M × 768-dim (1 worker)    ~0.1–0.3 s
HTTP/coord overhead (gen ↔ retrieve ↔ gen)              ~0.2–0.4 s
                                                  ─────  ~1.5–2.9 s
```

That matches the observed ~2.85 s.

## Bottleneck: **the GPU**, distantly followed by the single-worker retriever

### 1. GPU (dominant — ~70–80% of wall clock)

Two things stack against us:

- **Decode is memory-bandwidth-bound, not compute-bound.** A 3B model in bf16
  on a 4090 ceilings at ~80–120 tok/s single stream and only ~3–5× that with
  batching, because every token needs to read the full KV cache from HBM.
  Adding more concurrent examples doesn't scale linearly.
- **The README launches SGLang with `--disable-radix-cache --disable-overlap`.**
  Those flags turn off two big optimizations:
  - **Radix cache** = prefix caching across turns. Right now turn 2's prompt
    is "turn-1 prompt + response + retrieval block", and SGLang re-prefills
    the entire ~1 K tokens from scratch every turn instead of reusing the
    already-encoded prefix. That's ~50% wasted prefill work in a multi-turn
    run.
  - **Overlap** = pipelining the next forward pass with detokenization.
    Disabling it costs ~10–20% throughput.

  These flags exist because the original Search-R1 setup hit determinism /
  stability issues with them on. Worth a controlled experiment to see if our
  pipeline is OK with them re-enabled.

### 2. Retriever (~5–10% of wall clock today; would matter on the big datasets)

The FAISS index is `IndexFlat` (exhaustive 21 M × 768-dim inner-product
search) on CPU, with 1 worker. Each query is ~100–300 ms of pure CPU. With 8
concurrent eval threads all hitting `/search`, queries serialize behind one
worker. On PopQA (14,267 examples) this adds up to ~30–60 minutes of pure
retrieval queueing per run.

### 3. Everything else (≤10%)

HTTP round-trips, tokenizer overhead, Python orchestration. Negligible.

## What would actually move the needle

Ranked by expected speedup vs effort, no model/data changes:

| Change | Est. speedup | Effort | Risk |
|---|---|---|---|
| Drop `--disable-radix-cache` and `--disable-overlap` from SGLang | **1.5–2×** | trivial (flag change) | low — needs a quick A/B to confirm EM unchanged |
| Bump `INFERENCE_MAX_WORKERS` 8 → 16 or 32 (4090 has spare KV slack at 22 GB used) | 1.2–1.5× | one-line | medium — too high causes KV thrashing |
| Run retriever with `--num_retriever 4` (parallel FAISS workers) | 1.0–1.1× today, **1.3–1.5× on PopQA/2Wiki** | trivial | low — FAISS Flat is read-only, fully parallel-safe |
| Swap FAISS Flat → `IVF65536,SQ8` (already mentioned in `local_retriever/INDEXING.md`) | **5–10×** retrieval, **1.05–1.10×** end-to-end | medium (rebuild index ~hours) | low — ~1% recall hit |
| Move to A100 80 GB / H100 | 2–3× | hardware swap | none |

Realistic sweet spot for **this box**: re-enable radix-cache + overlap, bump
workers to 16, run retriever with 4 workers. That likely takes Plan B from
~12–18 h down to ~6–10 h, Plan C from ~3.4 days to ~1.5–2 days, and Plan A
from ~17 days to ~7–10 days. None of that changes the fundamental shape:
**the GPU's decode bandwidth is what gates throughput on a 3B model in this
multi-turn loop.**

## Suggested next step

Run a quick A/B on Bamboogle × 1, before/after enabling radix-cache + overlap
(plus retriever `--num_retriever 4`), to confirm:

1. EM/F1 land within noise of the current 0.088 / 0.155 — i.e., no
   determinism / correctness regression.
2. Wall-clock drops by the projected ~1.5–2×.

If both hold, re-launch the chosen sweep plan with the faster configuration
and proportionally shorter ETA.
