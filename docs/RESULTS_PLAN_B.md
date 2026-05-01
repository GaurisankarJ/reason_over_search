# Search-R1 evaluation results

_Source: `/workspace/reason_over_search/evaluation_search_r1/results/_archive_v1` (14 runs)_

## Per-seed scores

### EM

| Dataset | Variant | seed=1 | mean | std | n |
|---|---|---|---|---|---|
| bamboogle | base | 0.088 | 0.088 | — | 1 |
| bamboogle | instruct | 0.344 | 0.344 | — | 1 |
| nq | base | 0.390 | 0.390 | — | 1 |
| nq | instruct | 0.402 | 0.402 | — | 1 |
| triviaqa | base | 0.583 | 0.583 | — | 1 |
| triviaqa | instruct | 0.531 | 0.531 | — | 1 |
| popqa | base | 0.424 | 0.424 | — | 1 |
| popqa | instruct | 0.413 | 0.413 | — | 1 |
| musique | base | 0.055 | 0.055 | — | 1 |
| musique | instruct | 0.141 | 0.141 | — | 1 |
| 2wikimultihopqa | base | 0.239 | 0.239 | — | 1 |
| 2wikimultihopqa | instruct | 0.350 | 0.350 | — | 1 |
| hotpotqa | base | 0.263 | 0.263 | — | 1 |
| hotpotqa | instruct | 0.346 | 0.346 | — | 1 |

### F1

| Dataset | Variant | seed=1 | mean | std | n |
|---|---|---|---|---|---|
| bamboogle | base | 0.172 | 0.172 | — | 1 |
| bamboogle | instruct | 0.453 | 0.453 | — | 1 |
| nq | base | 0.474 | 0.474 | — | 1 |
| nq | instruct | 0.487 | 0.487 | — | 1 |
| triviaqa | base | 0.657 | 0.657 | — | 1 |
| triviaqa | instruct | 0.620 | 0.620 | — | 1 |
| popqa | base | 0.458 | 0.458 | — | 1 |
| popqa | instruct | 0.458 | 0.458 | — | 1 |
| musique | base | 0.123 | 0.123 | — | 1 |
| musique | instruct | 0.216 | 0.216 | — | 1 |
| 2wikimultihopqa | base | 0.306 | 0.306 | — | 1 |
| 2wikimultihopqa | instruct | 0.422 | 0.422 | — | 1 |
| hotpotqa | base | 0.365 | 0.365 | — | 1 |
| hotpotqa | instruct | 0.458 | 0.458 | — | 1 |

### ACC

| Dataset | Variant | seed=1 | mean | std | n |
|---|---|---|---|---|---|
| bamboogle | base | 0.096 | 0.096 | — | 1 |
| bamboogle | instruct | 0.376 | 0.376 | — | 1 |
| nq | base | 0.428 | 0.428 | — | 1 |
| nq | instruct | 0.450 | 0.450 | — | 1 |
| triviaqa | base | 0.638 | 0.638 | — | 1 |
| triviaqa | instruct | 0.604 | 0.604 | — | 1 |
| popqa | base | 0.441 | 0.441 | — | 1 |
| popqa | instruct | 0.448 | 0.448 | — | 1 |
| musique | base | 0.064 | 0.064 | — | 1 |
| musique | instruct | 0.167 | 0.167 | — | 1 |
| 2wikimultihopqa | base | 0.252 | 0.252 | — | 1 |
| 2wikimultihopqa | instruct | 0.390 | 0.390 | — | 1 |
| hotpotqa | base | 0.284 | 0.284 | — | 1 |
| hotpotqa | instruct | 0.383 | 0.383 | — | 1 |

## Grand averages

### Grand average EM across all runs

| Variant | mean | n_runs |
|---|---|---|
| base | 0.292 | 7 |
| instruct | 0.361 | 7 |

### Grand average F1 across all runs

| Variant | mean | n_runs |
|---|---|---|
| base | 0.365 | 7 |
| instruct | 0.445 | 7 |

### Grand average ACC across all runs

| Variant | mean | n_runs |
|---|---|---|
| base | 0.315 | 7 |
| instruct | 0.403 | 7 |

## Trace health

Close-rate = fraction of examples whose `final_response` contains `</answer>`. 
Length-truncated = fraction whose SGLang `stop_reason` was anything other than `stop`/`eos`/`stop_str` (typically the per-step token cap firing).

### Close-rate

| Dataset | Variant | seed=1 | mean | n_examples |
|---|---|---|---|---|
| bamboogle | base | 100.0% | 100.0% | 125 |
| bamboogle | instruct | 100.0% | 100.0% | 125 |
| nq | base | 99.9% | 99.9% | 1000 |
| nq | instruct | 99.0% | 99.0% | 1000 |
| triviaqa | base | 99.9% | 99.9% | 1000 |
| triviaqa | instruct | 99.1% | 99.1% | 1000 |
| popqa | base | 100.0% | 100.0% | 1000 |
| popqa | instruct | 97.5% | 97.5% | 1000 |
| musique | base | 100.0% | 100.0% | 2417 |
| musique | instruct | 91.4% | 91.4% | 2417 |
| 2wikimultihopqa | base | 99.6% | 99.6% | 1000 |
| 2wikimultihopqa | instruct | 94.2% | 94.2% | 1000 |
| hotpotqa | base | 100.0% | 100.0% | 1000 |
| hotpotqa | instruct | 98.1% | 98.1% | 1000 |

### Length-truncation rate

| Dataset | Variant | seed=1 | mean | n_examples |
|---|---|---|---|---|
| bamboogle | base | 0.0% | 0.0% | 125 |
| bamboogle | instruct | 0.0% | 0.0% | 125 |
| nq | base | 0.1% | 0.1% | 1000 |
| nq | instruct | 0.1% | 0.1% | 1000 |
| triviaqa | base | 0.1% | 0.1% | 1000 |
| triviaqa | instruct | 0.3% | 0.3% | 1000 |
| popqa | base | 0.0% | 0.0% | 1000 |
| popqa | instruct | 0.0% | 0.0% | 1000 |
| musique | base | 0.0% | 0.0% | 2417 |
| musique | instruct | 0.3% | 0.3% | 2417 |
| 2wikimultihopqa | base | 0.4% | 0.4% | 1000 |
| 2wikimultihopqa | instruct | 0.2% | 0.2% | 1000 |
| hotpotqa | base | 0.0% | 0.0% | 1000 |
| hotpotqa | instruct | 0.3% | 0.3% | 1000 |

### Mean completion tokens (whole trace, summed over turns)

| Dataset | Variant | seed=1 | mean | n_examples |
|---|---|---|---|---|
| bamboogle | base | 10 | 10 | 125 |
| bamboogle | instruct | 44 | 44 | 125 |
| nq | base | 10 | 10 | 1000 |
| nq | instruct | 45 | 45 | 1000 |
| triviaqa | base | 9 | 9 | 1000 |
| triviaqa | instruct | 45 | 45 | 1000 |
| popqa | base | 9 | 9 | 1000 |
| popqa | instruct | 41 | 41 | 1000 |
| musique | base | 10 | 10 | 2417 |
| musique | instruct | 56 | 56 | 2417 |
| 2wikimultihopqa | base | 12 | 12 | 1000 |
| 2wikimultihopqa | instruct | 65 | 65 | 1000 |
| hotpotqa | base | 9 | 9 | 1000 |
| hotpotqa | instruct | 55 | 55 | 1000 |
