# Search-R1 evaluation results

_Source: `/workspace/reason_over_search/evaluation_search_r1/results` (8 runs)_

## Per-seed scores

### EM

| Dataset | Variant | seed=1 | mean | std | n |
|---|---|---|---|---|---|
| bamboogle | base | 0.088 | 0.088 | — | 1 |
| bamboogle | instruct | — | — | — | 0 |
| nq | base | 0.390 | 0.390 | — | 1 |
| nq | instruct | 0.399 | 0.399 | — | 1 |
| triviaqa | base | — | — | — | 0 |
| triviaqa | instruct | 0.539 | 0.539 | — | 1 |
| popqa | base | — | — | — | 0 |
| popqa | instruct | 0.412 | 0.412 | — | 1 |
| musique | base | — | — | — | 0 |
| musique | instruct | 0.149 | 0.149 | — | 1 |
| 2wikimultihopqa | base | — | — | — | 0 |
| 2wikimultihopqa | instruct | 0.353 | 0.353 | — | 1 |
| hotpotqa | base | — | — | — | 0 |
| hotpotqa | instruct | 0.354 | 0.354 | — | 1 |

### F1

| Dataset | Variant | seed=1 | mean | std | n |
|---|---|---|---|---|---|
| bamboogle | base | 0.172 | 0.172 | — | 1 |
| bamboogle | instruct | — | — | — | 0 |
| nq | base | 0.474 | 0.474 | — | 1 |
| nq | instruct | 0.481 | 0.481 | — | 1 |
| triviaqa | base | — | — | — | 0 |
| triviaqa | instruct | 0.628 | 0.628 | — | 1 |
| popqa | base | — | — | — | 0 |
| popqa | instruct | 0.460 | 0.460 | — | 1 |
| musique | base | — | — | — | 0 |
| musique | instruct | 0.221 | 0.221 | — | 1 |
| 2wikimultihopqa | base | — | — | — | 0 |
| 2wikimultihopqa | instruct | 0.423 | 0.423 | — | 1 |
| hotpotqa | base | — | — | — | 0 |
| hotpotqa | instruct | 0.459 | 0.459 | — | 1 |

### ACC

| Dataset | Variant | seed=1 | mean | std | n |
|---|---|---|---|---|---|
| bamboogle | base | 0.096 | 0.096 | — | 1 |
| bamboogle | instruct | — | — | — | 0 |
| nq | base | 0.428 | 0.428 | — | 1 |
| nq | instruct | 0.439 | 0.439 | — | 1 |
| triviaqa | base | — | — | — | 0 |
| triviaqa | instruct | 0.616 | 0.616 | — | 1 |
| popqa | base | — | — | — | 0 |
| popqa | instruct | 0.452 | 0.452 | — | 1 |
| musique | base | — | — | — | 0 |
| musique | instruct | 0.173 | 0.173 | — | 1 |
| 2wikimultihopqa | base | — | — | — | 0 |
| 2wikimultihopqa | instruct | 0.395 | 0.395 | — | 1 |
| hotpotqa | base | — | — | — | 0 |
| hotpotqa | instruct | 0.389 | 0.389 | — | 1 |

## Grand averages

### Grand average EM across all runs

| Variant | mean | n_runs |
|---|---|---|
| base | 0.239 | 2 |
| instruct | 0.368 | 6 |

### Grand average F1 across all runs

| Variant | mean | n_runs |
|---|---|---|
| base | 0.323 | 2 |
| instruct | 0.445 | 6 |

### Grand average ACC across all runs

| Variant | mean | n_runs |
|---|---|---|
| base | 0.262 | 2 |
| instruct | 0.411 | 6 |

## Trace health

Close-rate = fraction of examples whose `final_response` contains `</answer>`. 
Length-truncated = fraction whose SGLang `stop_reason` was anything other than `stop`/`eos`/`stop_str` (typically the per-step token cap firing).

### Close-rate

| Dataset | Variant | seed=1 | mean | n_examples |
|---|---|---|---|---|
| bamboogle | base | 100.0% | 100.0% | 125 |
| nq | base | 99.9% | 99.9% | 1000 |
| nq | instruct | 98.3% | 98.3% | 1000 |
| triviaqa | instruct | 98.9% | 98.9% | 1000 |
| popqa | instruct | 98.2% | 98.2% | 1000 |
| musique | instruct | 91.7% | 91.7% | 2417 |
| 2wikimultihopqa | instruct | 95.2% | 95.2% | 1000 |
| hotpotqa | instruct | 98.0% | 98.0% | 1000 |

### Length-truncation rate

| Dataset | Variant | seed=1 | mean | n_examples |
|---|---|---|---|---|
| bamboogle | base | 0.0% | 0.0% | 125 |
| nq | base | 0.1% | 0.1% | 1000 |
| nq | instruct | 0.2% | 0.2% | 1000 |
| triviaqa | instruct | 0.1% | 0.1% | 1000 |
| popqa | instruct | 0.0% | 0.0% | 1000 |
| musique | instruct | 0.2% | 0.2% | 2417 |
| 2wikimultihopqa | instruct | 0.0% | 0.0% | 1000 |
| hotpotqa | instruct | 0.2% | 0.2% | 1000 |

### Mean completion tokens (whole trace, summed over turns)

| Dataset | Variant | seed=1 | mean | n_examples |
|---|---|---|---|---|
| bamboogle | base | 10 | 10 | 125 |
| nq | base | 10 | 10 | 1000 |
| nq | instruct | 44 | 44 | 1000 |
| triviaqa | instruct | 45 | 45 | 1000 |
| popqa | instruct | 41 | 41 | 1000 |
| musique | instruct | 55 | 55 | 2417 |
| 2wikimultihopqa | instruct | 65 | 65 | 1000 |
| hotpotqa | instruct | 55 | 55 | 1000 |
