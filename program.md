# experiment_ros — autoresearch on Reason-Over-Search

This is an experiment to have the LLM do its own research on the **Search-R1 inference pipeline**: iterate on prompts, parsing, and pipeline orchestration to push EM (and F1) on the published Search-R1 3B benchmarks.

The subject of this experiment is the sibling repository at [`/Users/sandheepp/broadsword/reason_over_search/`](../../reason_over_search/). All git commits, branches, and edits described below happen **inside that repo**, not inside `autoresearch/`. This directory is just the playbook.

Background reading (skim before starting):
- [reason_over_search/README.md](../../reason_over_search/README.md) — milestone framing.
- [reason_over_search/evaluation_search_r1/README.md](../../reason_over_search/evaluation_search_r1/README.md) — eval invocation.
- [reason_over_search/evaluation_search_r1/REPRODUCIBILITY.md](../../reason_over_search/evaluation_search_r1/REPRODUCIBILITY.md) — paper targets, divergences fixed.
- [reason_over_search/evaluation_search_r1/EVAL_OPS.md](../../reason_over_search/evaluation_search_r1/EVAL_OPS.md) — sweep plans, where the wall-clock goes, speedup ranking.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr27`). The branch `experiment_ros/<tag>` must not already exist in `reason_over_search/` — this is a fresh run.
2. **Create the branch** in the *reason_over_search* repo:
   ```bash
   cd /Users/sandheepp/broadsword/reason_over_search
   git checkout -b experiment_ros/<tag>
   ```
3. **Read the in-scope files**. The editable surface is the inference pipeline only — these are the files you may modify:
   - [`evaluation_search_r1/flashrag/pipeline/active_pipeline.py`](../../reason_over_search/evaluation_search_r1/flashrag/pipeline/active_pipeline.py) — the search↔generate loop, turn budget, observation handling.
   - [`evaluation_search_r1/flashrag/search_r1/parser.py`](../../reason_over_search/evaluation_search_r1/flashrag/search_r1/parser.py) — `<search>` / `<answer>` parsing, observation truncation.
   - [`evaluation_search_r1/flashrag/search_r1/templates.py`](../../reason_over_search/evaluation_search_r1/flashrag/search_r1/templates.py) — prompt templates (base + instruct).
   - [`evaluation_search_r1/flashrag/config/basic_config.yaml`](../../reason_over_search/evaluation_search_r1/flashrag/config/basic_config.yaml) — `retrieval_topk`, `max_search_turns`, retrieval/generation lengths, sampling params.

   And read these for context (do **not** modify):
   - [`evaluation_search_r1/run_eval.py`](../../reason_over_search/evaluation_search_r1/run_eval.py) — eval entrypoint and metric pipe.
   - [`evaluation_search_r1/flashrag/search_r1/answer_utils.py`](../../reason_over_search/evaluation_search_r1/flashrag/search_r1/answer_utils.py) — the EM scorer (ground truth metric).
   - [`scripts/run_one.sh`](../../reason_over_search/scripts/run_one.sh) — one-run runner you'll call.
4. **Verify runtime prereqs are up**. The eval needs three things alive on the box:
   - **Retriever** on `127.0.0.1:3005`. Sanity check: `curl -sS http://127.0.0.1:3005/health` returns `"healthy"`. If not, tell the human to start it per [`local_retriever/README.md`](../../reason_over_search/local_retriever/README.md).
   - **SGLang** on `127.0.0.1:3000` serving the **instruct** variant (the fast-iter target — see below). Sanity check: `curl -sS http://127.0.0.1:3000/get_model_info | grep -o instruct` should match. If not: `scripts/manage_sglang.sh switch instruct`.
   - **Eval venv** at `/venv/evaluation_search_r1`. Sanity check: `/venv/evaluation_search_r1/bin/python -c "import flashrag"` exits 0.
5. **Initialize results.tsv** in `reason_over_search/` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment is **one Bamboogle (test, 125 examples) eval run on the instruct variant**, kicked off with:

```bash
cd /Users/sandheepp/broadsword/reason_over_search
scripts/run_one.sh instruct bamboogle <seed> > run.log 2>&1
```

Pick a fixed seed (e.g. `1`) and reuse it for every run so the labels in `results/` don't pile up — but note that SGLang ignores FlashRAG's `seed` (sampling is `temp=1.0` non-deterministic), so single-run noise is real (~3 pp at n=125). Treat improvements <2 pp EM with skepticism; confirm them with a second run on a different seed before declaring victory.

**Why Bamboogle/instruct as the fast loop**: 125 examples × ~2.85 s/example ≈ 6 min wall-clock — same iteration tempo as the original `train.py` 5-min budget. Bamboogle is multi-hop so it stresses the search loop; instruct never length-truncates (clean signal); it's where the smoke test in `REPRODUCIBILITY.md` ran.

**What you CAN do** — modify any of the four files listed in Setup step 3:
- Tweak prompt wording, few-shot exemplars, system prompt.
- Adjust `retrieval_topk`, `max_search_turns`, `retrieval_query_max_length`, observation truncation length, per-step token cap.
- Change `<information>` / `<search>` / `<answer>` formatting and parsing (regex anchoring, whitespace, doc rendering).
- Re-orchestrate the active pipeline: re-ranking, query rewriting before retrieve, dedup of identical search queries across turns, early-stop heuristics, fallback when `<search>` is malformed.
- Change sampling params (temperature, top_p, repetition penalty) via the yaml.

**What you CANNOT do**:
- Modify model checkpoints (`search_r1_base_model/`, `search_r1_instruct_model/`). They are the frozen GRPO targets we are evaluating.
- Modify the FAISS index, retriever code (`local_retriever/`), or the corpus. Retrieval recall is fixed.
- Modify `flashrag/search_r1/answer_utils.py` (the EM/F1 scorer is the ground truth — changing it is gaming the metric).
- Install new packages or add dependencies. Use only what's in `requirements.txt`.
- Modify `prepare.py`-equivalent fixed scaffolding: `run_eval.py`, the `scripts/`, the dataset jsonl files, the SGLang launch flags.

**The goal is simple: get the highest EM on Bamboogle instruct.** F1 is the tiebreaker when EM ties. Since each run is ~6 min, you don't need to worry about runtime — it's roughly fixed. Everything inside the editable surface is fair game: prompts, parsing, turn budget, retrieval params, pipeline structure.

**Generalization sanity check**: every ~10 kept changes, run a confirmation sweep on a second dataset (e.g. `scripts/run_one.sh instruct 2wikimultihopqa 1` — multi-hop, dev split, ~12k examples → subsample if too slow, or use the smaller MuSiQue dev). If a change helps Bamboogle but tanks 2Wiki by >3 pp, treat it as overfitting to Bamboogle and discard. The 7-dataset Plan B sweep ([`scripts/sweep_b_reduced.sh`](../../reason_over_search/scripts/sweep_b_reduced.sh), ~12–18 h) is the final-quality confirmation, not part of the inner loop.

**Simplicity criterion**: All else being equal, simpler is better. A small EM bump that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. A 0.005 EM improvement that adds 30 lines of regex hacks? Probably not worth it. A 0.005 EM improvement from deleting code? Definitely keep. Equal EM but much simpler? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run `scripts/run_one.sh instruct bamboogle 1` against unmodified `experiment_ros/<tag>` (which forks from current master). Expect EM around `0.36` per the smoke test in `REPRODUCIBILITY.md`.

## Output format

`run_one.sh` writes results to `evaluation_search_r1/results/bamboogle/bamboogle_<YYYY>_<MM>_<DD>_<HH>_<MM>_search_r1_instruct_seed<N>/`. The key file is `metric_score.txt`:

```
em: 0.36
acc: 0.392
f1: 0.4609015873015873
```

Extract the metrics:

```bash
# Find the most recent result dir for this run and pull em/f1
LATEST=$(ls -dt evaluation_search_r1/results/bamboogle/bamboogle_*_search_r1_instruct_seed1 | head -1)
grep -E "^(em|f1):" "$LATEST/metric_score.txt"
```

If `metric_score.txt` doesn't exist, the run crashed. Run `tail -n 80 run.log` to read the Python stack trace.

## Logging results

When an experiment is done, log it to `results.tsv` in `reason_over_search/` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	em	f1	status	description
```

1. git commit hash (short, 7 chars) — from the `reason_over_search/` repo.
2. EM achieved on Bamboogle instruct (e.g. `0.360`) — use `0.000` for crashes.
3. F1 achieved (e.g. `0.461`, round to .3f) — use `0.000` for crashes.
4. status: `keep`, `discard`, or `crash`.
5. short text description of what this experiment tried.

Example:

```
commit	em	f1	status	description
a1b2c3d	0.360	0.461	keep	baseline (instruct, bamboogle, seed 1)
b2c3d4e	0.376	0.470	keep	add explicit "If unsure, search again" to system prompt
c3d4e5f	0.344	0.448	discard	bump max_search_turns 4 -> 6
d4e5f6g	0.000	0.000	crash	regex change broke <search> parsing
```

Do **not** commit `results.tsv` — leave it untracked.

## The experiment loop

The experiment runs on a dedicated branch in the `reason_over_search/` repo (e.g. `experiment_ros/apr27` or `experiment_ros/apr27-gpu0`).

LOOP FOREVER:

1. Look at the git state in `reason_over_search/`: the current branch/commit you're on.
2. Pick an experimental idea. Hack one of the four in-scope files directly.
3. `git commit` (in `reason_over_search/`).
4. Run the experiment:
   ```bash
   cd /Users/sandheepp/broadsword/reason_over_search
   scripts/run_one.sh instruct bamboogle 1 > run.log 2>&1
   ```
   Redirect everything — do NOT use `tee` or let output flood your context.
5. Read out the results:
   ```bash
   LATEST=$(ls -dt evaluation_search_r1/results/bamboogle/bamboogle_*_search_r1_instruct_seed1 | head -1)
   grep -E "^(em|f1):" "$LATEST/metric_score.txt"
   ```
6. If the grep output is empty, the run crashed. `tail -n 80 run.log`, attempt a fix. If you can't get things to work after a few attempts, give up and log `crash`.
7. Record the result in `results.tsv` (do not commit it).
8. If EM improved (higher), or EM tied and F1 improved, you "advance" the branch — keep the commit.
9. If EM is worse (or EM tied and F1 didn't improve), `git reset --hard HEAD~1` back to where you started.

The idea: you are a completely autonomous researcher. If a change helps, keep. If not, discard. Advance the branch so you can iterate.

**Resume hazard**: `run_one.sh` is *resume-aware* — if it sees a `metric_score.txt` for `(instruct, bamboogle, seed=1)` it will **skip the run and exit 0** without producing fresh metrics. Because every experiment uses the same `(variant, dataset, seed)` triple, you must clear the previous run's output before each iteration:

```bash
rm -rf evaluation_search_r1/results/bamboogle/bamboogle_*_search_r1_instruct_seed1
```

Do this at the top of every loop iteration, *before* step 4. Otherwise step 5 will return stale numbers from the previous commit and you will draw wrong conclusions.

**Timeout**: Each run takes ~6 min on Bamboogle instruct (+ small startup overhead). If a run exceeds 15 minutes, kill it (`pkill -f run_eval.py`) and treat it as a failure (discard and revert). Common runaway causes: a parser change that prevents `<answer>` from ever closing → model hits the per-step token cap every turn → the loop runs all `max_search_turns` for every example.

**Crashes**: If a run crashes (a regex bug, a yaml typo, an assertion in `active_pipeline.py`, an OOM in SGLang), use your judgment: if it's a dumb fix (typo, missing import, off-by-one), fix it and re-run. If the idea itself is fundamentally broken, skip it, log `crash`, and move on.

**Watch for SGLang/retriever drift**: between runs, the runtime prereqs should stay up. Periodically (~every 10 runs) re-check:
```bash
curl -sS http://127.0.0.1:3005/health
curl -sS http://127.0.0.1:3000/get_model_info | grep -q instruct && echo "sglang ok"
```
If either is down, restart per [`EVAL_OPS.md`](../../reason_over_search/evaluation_search_r1/EVAL_OPS.md) and continue.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or away from the computer, and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read [REPRODUCIBILITY.md](../../reason_over_search/evaluation_search_r1/REPRODUCIBILITY.md) for the 10 already-applied fixes (the *11th* fix is yours to find), re-read the Search-R1 paper sections referenced in `evaluation_search_r1/README.md`, try combinations of previous near-misses, try more radical pipeline restructures (re-ranking, query expansion, multi-query retrieval, self-consistency over multiple sampled traces). The loop runs until the human interrupts you, period.

As an example use case: at ~6 min/run you can run ~10/hour, so ~80 over the duration of average human sleep. The user wakes up to a `results.tsv` of attempted ideas with EM/F1 deltas — all completed by you while they slept.
