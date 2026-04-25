# What (if anything) went wrong with our setup?

Honest read: **probably not much, and the framing of "wrong" needs unpacking
first.** Then a list of suspects in order of likelihood.

## Reframe: is anything actually wrong?

We have **two data points from one seed each on n=125**:

|                       | Δ vs paper | σ band |
|---                    |---         |---     |
| Base Bamboogle        | −4.0 pp    | ~1.3 σ — **inside noise** |
| Instruct Bamboogle    | +12.8 pp   | ~3.4 σ — outside noise, but **in the model's favour** |

The base run is unremarkable — well within sampling noise on a 125-example
benchmark. Only the instruct overshoot is statistically interesting, and it's
an *overshoot*, not a regression. If something were misconfigured to hurt
eval, both would skew the same way (or instruct, with its more brittle
multi-turn behaviour, would skew worse). They don't. That's evidence the
pipeline isn't systemically broken.

That said, +12.8 pp on a benchmark this small with one seed could collapse to
~+3 pp once we average 3 seeds and that would be unremarkable too. So step
one is "more seeds before drawing conclusions." Step two is the diagnostic
list below.

## Most likely explanations, ranked

### 1. Single-seed sampling variance on a 125-example benchmark *(most likely)*

SE ≈ 3.8 pp, p(observe |Δ| ≥ 12.8 pp) ≈ 0.001 under Gaussian — but
per-question outcomes are correlated and Bamboogle has a long tail of
"trivially answerable with one good search". A few extra lucky retrievals
get amplified. Mitigation: 3+ seeds on Bamboogle and any benchmark where we
see a >2σ gap.

### 2. Stricter end-of-trace handling than the paper's inference loop

We stop generation hard on `</answer>`. The paper's `_postprocess_responses`
does `split('</answer>')[0] + '</answer>'` after the fact. Same effect when
the model emits `</answer>` cleanly. **But** the paper's training-time
rollout doesn't re-stop on `</answer>` — the model continues to EOS, which
can include irrelevant trailing tokens. If their evaluation uses the same
training rollout pipeline rather than the inference-time post-processor, our
cleaner stop could marginally help EM by keeping the answer tag tight. This
would help instruct (which reaches `</answer>` reliably) more than base
(which hits length truncation 17 % of the time). That asymmetry **fits the
observation**.

### 3. Live retriever still using `retrieval_query_max_length=128`

The retriever process was started before we changed the config to 256, and
we haven't restarted it. So queries longer than 128 tokens are still being
truncated server-side. For Bamboogle most queries are short (<30 tokens), so
this is unlikely to matter much here, but it's a real config drift to clean
up. Both variants are affected equally though, so it doesn't explain the
asymmetry.

### 4. SGLang `--disable-radix-cache` / `--disable-overlap` were dropped on the instruct restart

The `manage_sglang.sh` edit (presumably yours) removed those flags before
the instruct run. Re-enabling radix-cache changes the order of FP operations
across turns (KV cache is reused instead of re-prefilled), which can shift
logits by tiny amounts. Over a 125-example sample this is mostly noise, but
it makes our base run (with the original flags) and our instruct run
(without them) **not directly comparable** as a controlled A/B. Mitigation:
launch both variants under the same SGLang flags before declaring deltas.

### 5. Paper version drift

The paper's been through 5 HTML versions on arxiv. The `0.232` number for
instruct Bamboogle GRPO is from v5. Earlier versions might report different
numbers, and the released checkpoints might correspond to a different
version's numbers than v5's table. Mitigation: pull the README on the
upstream GitHub for any reproduction instructions and check whether they
reference a specific paper version.

### 6. Lighter, less-likely suspects

- **`extract_solution` last-match vs paper's first-match.** I left this as
  last-match in `answer_utils.py` (the parser fix in `parser.py` was for
  `<search>`). With stop-on-`</answer>`, there's exactly one `</answer>` per
  trace — last and first match are the same trace. Not an issue here.
- **`apply_chat_template`'s default system prompt.** Qwen2.5's tokenizer
  prepends `"You are Qwen, created by Alibaba Cloud. You are a helpful
  assistant."` automatically. The training rollout for the GRPO instruct
  checkpoint may or may not have included this exact system prompt. If it
  didn't, we're slightly out of distribution — could go either way for EM.
- **Loss-masked retrieved tokens at training, no analog at inference.**
  Doesn't affect generation; mentioned only to rule out.

## What I'd actually do to disambiguate

In order of cost:

1. **Re-run base Bamboogle once** under the *current* SGLang flags (no
   `--disable-radix-cache`, no `--disable-overlap`). If the base score moves
   materially, then suspect #4 was confounding the comparison.

2. **Run 3 seeds of Bamboogle instruct.** If the mean lands near 0.30, the
   +12.8 pp was lucky. If it stays at 0.36, something is genuinely making us
   better than paper, and we'd want to find it before publication.

3. **Run NQ base × 1.** NQ has 3,610 examples → 1-σ ≈ 0.8 pp. Any systematic
   bias becomes visible there. Paper number is 0.421 GRPO base. If we land
   within 0.405–0.435, the pipeline is reproducing within noise across the
   board. If we land outside, that's where to debug.

4. **Restart the retriever** with `retrieval_query_max_length=256` so we
   close that config-vs-running-process drift before any sweep.

## Bottom line

Most of the gap is probably noise + a small systematic edge from how cleanly
we close out traces, and we won't know for sure until we have 3 seeds and at
least one big-N benchmark in the data.
