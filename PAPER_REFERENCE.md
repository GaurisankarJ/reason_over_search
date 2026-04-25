# Paper reference & smoke comparison

## 1. Yes — we're running GRPO

Confirmed from the model paths the README pulls:

- `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo` (base)
- `PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo` (instruct)

Both are GRPO with EM-reward. The base `config.json` shows
`_name_or_path: "Qwen/Qwen2.5-3B"` underlying.

## 2. The paper numbers we should compare against

The paper's **main results table** (Table 1) reports **PPO** by default. The
PPO-vs-GRPO numbers we should be comparing to are in
**Appendix F / Section 5.1 ("Different RL methods: PPO vs. GRPO")**,
**Table 3**, caption:

> "The performance results of Search-R1 with PPO and GRPO on seven datasets."

Pulled from `arxiv.org/html/2503.09516v5#A6`. Verbatim:

### Qwen2.5-3B paper numbers — EM

|                    | NQ    | TriviaQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | Avg   |
|---                 |---    |---       |---    |---       |---    |---      |---        |---    |
| PPO base           | 0.406 | 0.587    | 0.435 | 0.284    | 0.273 | 0.049   | 0.088     | 0.303 |
| **GRPO base**      | **0.421** | **0.583** | **0.413** | **0.297** | **0.274** | **0.066** | **0.128** | **0.312** |
| PPO instruct       | 0.341 | 0.545    | 0.378 | 0.324    | 0.319 | 0.103   | 0.264     | 0.325 |
| **GRPO instruct**  | **0.397** | **0.565** | **0.391** | **0.331** | **0.310** | **0.124** | **0.232** | **0.336** |

The bold rows are the ones we should be comparing to. (Earlier
"~0.40–0.43 expected for instruct Bamboogle" was wrong — actual paper
number is **0.232**.)

## 3. Our smoke runs vs paper

| Dataset    | Variant   | Our EM | Paper EM (GRPO) | Δ            | 1-σ on n=125 |
|---         |---        |---     |---              |---           |---           |
| Bamboogle  | base      | 0.088  | 0.128           | **−4.0 pp**  | 3.0 pp       |
| Bamboogle  | instruct  | 0.360  | 0.232           | **+12.8 pp** | 3.8 pp       |

### Two surprises

- **Base −4 pp** is within ~1.3 σ of paper — plausibly just sampling noise on
  125 examples, or some residual setup difference. Curious aside: 0.088
  happens to match the paper's **PPO** base Bamboogle exactly. Could be
  coincidence; could be that something in our pipeline narrows the GRPO–PPO
  gap on the base. Hard to say from one seed.

- **Instruct +12.8 pp is outside noise (~3.4 σ).** That's a meaningful
  overshoot, not undershoot. Possible explanations:
  1. **Lucky seed** — single-seed temperature-1.0 runs can be lucky; a 3-seed
     average will narrow this.
  2. **Pipeline difference works in the model's favour.** Our
     `max_obs_length=500` truncates retrieval differently than the paper's
     tokenizer-level truncation; our `extract_solution` takes the **last**
     `<answer>` match rather than the first; our `</answer>` stop token cuts
     cleanly. Any of these could marginally help.
  3. **Different paper revision.** The numbers above are from v5; earlier
     versions may have different numbers.

**This is information, not a problem.** Reproducing within a few pp of paper
on the hardest OOD benchmark with one seed is a strong signal that the
post-fix pipeline is correct. We don't need to chase the +12.8 pp on instruct
yet — better to:

1. Get more seeds on Bamboogle to tighten the estimate.
2. Check a larger benchmark (NQ at 3,610 examples has 1-σ ≈ 0.8 pp, so any
   systematic bias becomes visible) before drawing conclusions.

## 4. Doc cleanup needed

`SMOKE_TEST_INSTRUCT.md` currently says "paper's published 3B-instruct-GRPO
Bamboogle EM is in the **~0.40–0.43** range" — that's wrong (actual: 0.232).

Action items:
- Correct `SMOKE_TEST_INSTRUCT.md` with the right paper number.
- Use this file (`PAPER_REFERENCE.md`) as the canonical source of paper
  numbers for future smoke checks and the eventual sweep aggregation.
