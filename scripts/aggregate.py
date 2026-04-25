#!/usr/bin/env python3
"""Aggregate metric_score.txt files across runs and write a markdown report.

Walks `evaluation_search_r1/results/<dataset>/<dataset>_<timestamp>_<save_note>/metric_score.txt`,
groups by (dataset, variant) using the `save_note` convention
`search_r1_<variant>_seed<N>`, and reports per-seed scores plus mean over seeds.

Usage:
  scripts/aggregate.py [--output RESULTS.md] [--results-dir DIR]
"""
from __future__ import annotations
import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "evaluation_search_r1" / "results"

DATASETS = ["bamboogle", "nq", "triviaqa", "popqa", "musique", "2wikimultihopqa", "hotpotqa"]
METRICS = ["em", "f1", "acc"]
SAVE_NOTE_RE = re.compile(r"search_r1_(?P<variant>base|instruct)_(?:seed|run)(?P<seed>\d+)$")


def parse_metric_file(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            continue
    return out


def collect(results_dir: Path):
    """Return {(dataset, variant, seed): metrics}."""
    runs: dict[tuple[str, str, int], dict[str, float]] = {}
    for ds_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        ds = ds_dir.name
        if ds not in DATASETS:
            continue
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            metric_file = run_dir / "metric_score.txt"
            if not metric_file.exists():
                continue
            # Extract trailing save_note from the run_dir name.
            # Format: <dataset>_<YYYY>_<MM>_<DD>_<HH>_<MM>_<save_note>
            name = run_dir.name
            if not name.startswith(f"{ds}_"):
                continue
            tail = name[len(ds) + 1:]
            # The timestamp has 5 numeric parts; everything after is the save_note.
            parts = tail.split("_")
            if len(parts) <= 5:
                continue
            save_note = "_".join(parts[5:])
            m = SAVE_NOTE_RE.match(save_note)
            if not m:
                continue
            variant = m.group("variant")
            seed = int(m.group("seed"))
            metrics = parse_metric_file(metric_file)
            runs[(ds, variant, seed)] = metrics
    return runs


def fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def render(runs, metric: str) -> str:
    """Markdown table for one metric."""
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for (ds, variant, seed), m in runs.items():
        if metric in m:
            grouped[(ds, variant)].append((seed, m[metric]))

    seeds_present = sorted({seed for v in grouped.values() for seed, _ in v})
    if not seeds_present:
        return f"### {metric.upper()}\n\n_no runs found_\n"

    lines = [f"### {metric.upper()}", ""]
    header = ["Dataset", "Variant", *[f"seed={s}" for s in seeds_present], "mean", "std", "n"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for ds in DATASETS:
        for variant in ("base", "instruct"):
            row = [ds, variant]
            scores = dict(grouped.get((ds, variant), []))
            seed_vals = [scores.get(s) for s in seeds_present]
            row.extend(fmt(v) for v in seed_vals)
            present = [v for v in seed_vals if v is not None]
            row.append(fmt(statistics.mean(present)) if present else "—")
            row.append(fmt(statistics.stdev(present)) if len(present) > 1 else "—")
            row.append(str(len(present)))
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def grand_average(runs, metric: str) -> str:
    """Mean across (datasets × seeds) per variant."""
    by_variant: dict[str, list[float]] = defaultdict(list)
    for (ds, variant, seed), m in runs.items():
        if metric in m:
            by_variant[variant].append(m[metric])
    if not by_variant:
        return ""
    lines = [f"### Grand average {metric.upper()} across all runs", ""]
    lines.append("| Variant | mean | n_runs |")
    lines.append("|---|---|---|")
    for variant in ("base", "instruct"):
        vals = by_variant.get(variant, [])
        if vals:
            lines.append(f"| {variant} | {statistics.mean(vals):.3f} | {len(vals)} |")
        else:
            lines.append(f"| {variant} | — | 0 |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "RESULTS.md")
    args = ap.parse_args()

    runs = collect(args.results_dir)

    md = ["# Search-R1 evaluation results", ""]
    md.append(f"_Source: `{args.results_dir}` ({len(runs)} runs)_")
    md.append("")
    md.append("## Per-seed scores")
    md.append("")
    for metric in METRICS:
        md.append(render(runs, metric))
    md.append("## Grand averages")
    md.append("")
    for metric in METRICS:
        ga = grand_average(runs, metric)
        if ga:
            md.append(ga)

    args.output.write_text("\n".join(md))
    print(f"wrote {args.output} ({len(runs)} runs aggregated)")


if __name__ == "__main__":
    main()
