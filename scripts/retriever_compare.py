"""Compare two retriever benchmark JSONs side-by-side."""

from __future__ import annotations

import argparse
import json
import sys


def fmt(v: float | None, suffix: str = "") -> str:
    if v is None:
        return "-"
    return f"{v:,.2f}{suffix}"


def speedup(a: float, b: float) -> str:
    """How much faster is a vs b (a should be the smaller/faster latency)."""
    if not b or not a:
        return "-"
    return f"{b/a:.2f}x"


def render(cpu: dict, gpu: dict) -> str:
    rows: list[tuple[str, str, str, str]] = []

    cs = cpu["single"]["latency_ms"]
    gs = gpu["single"]["latency_ms"]
    cb = cpu["batch"]["request_latency_ms"]
    gb = gpu["batch"]["request_latency_ms"]
    cbi = cpu["batch"]["per_item_latency_ms"]
    gbi = gpu["batch"]["per_item_latency_ms"]

    rows.append(("--- single /search ---", "", "", ""))
    for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"):
        rows.append((k, fmt(cs.get(k)) + " ms", fmt(gs.get(k)) + " ms",
                     speedup(gs.get(k, 0), cs.get(k, 0))))
    rows.append(("qps", fmt(cpu["single"].get("qps")),
                 fmt(gpu["single"].get("qps")),
                 speedup(cpu["single"].get("qps") or 1e-9, gpu["single"].get("qps") or 1e-9)))

    rows.append(("--- batch /batch_search ---", "", "", ""))
    bs_cpu = cpu["batch"].get("size")
    bs_gpu = gpu["batch"].get("size")
    rows.append(("batch_size", str(bs_cpu), str(bs_gpu), ""))
    for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"):
        rows.append((f"req {k}", fmt(cb.get(k)) + " ms", fmt(gb.get(k)) + " ms",
                     speedup(gb.get(k, 0), cb.get(k, 0))))
    for k in ("mean_ms", "p50_ms", "p95_ms"):
        rows.append((f"per-item {k}", fmt(cbi.get(k)) + " ms", fmt(gbi.get(k)) + " ms",
                     speedup(gbi.get(k, 0), cbi.get(k, 0))))
    rows.append(("effective_qps", fmt(cpu["batch"].get("effective_qps")),
                 fmt(gpu["batch"].get("effective_qps")),
                 speedup(cpu["batch"].get("effective_qps") or 1e-9,
                         gpu["batch"].get("effective_qps") or 1e-9)))

    rows.append(("--- system (during run) ---", "", "", ""))
    cs_sys = cpu.get("system", {})
    gs_sys = gpu.get("system", {})
    def s(d: dict, key: str, sub: str) -> str:
        v = d.get(key, {})
        return fmt(v.get(sub)) if v else "-"
    rows.append(("proc CPU% mean (sum across threads)",
                 s(cs_sys, "proc_cpu_pct_sum", "mean"),
                 s(gs_sys, "proc_cpu_pct_sum", "mean"), ""))
    rows.append(("proc CPU% max",
                 s(cs_sys, "proc_cpu_pct_sum", "max"),
                 s(gs_sys, "proc_cpu_pct_sum", "max"), ""))
    rows.append(("proc RSS MiB mean",
                 s(cs_sys, "proc_rss_mib", "mean"),
                 s(gs_sys, "proc_rss_mib", "mean"), ""))
    rows.append(("proc RSS MiB max",
                 s(cs_sys, "proc_rss_mib", "max"),
                 s(gs_sys, "proc_rss_mib", "max"), ""))
    rows.append(("GPU util % mean",
                 s(cs_sys, "gpu_util_pct", "mean"),
                 s(gs_sys, "gpu_util_pct", "mean"), ""))
    rows.append(("GPU util % max",
                 s(cs_sys, "gpu_util_pct", "max"),
                 s(gs_sys, "gpu_util_pct", "max"), ""))
    rows.append(("VRAM MiB mean",
                 s(cs_sys, "vram_used_mib", "mean"),
                 s(gs_sys, "vram_used_mib", "mean"), ""))
    rows.append(("VRAM MiB max",
                 s(cs_sys, "vram_used_mib", "max"),
                 s(gs_sys, "vram_used_mib", "max"), ""))
    rows.append(("Power W mean",
                 s(cs_sys, "power_w", "mean"),
                 s(gs_sys, "power_w", "mean"), ""))

    col_widths = [
        max(len(r[i]) for r in rows + [("metric", "CPU IVF", "GPU IVF", "GPU vs CPU")])
        for i in range(4)
    ]
    header = ("metric", "CPU IVF", "GPU IVF", "GPU vs CPU")
    sep = "  "
    lines = [
        sep.join(h.ljust(w) for h, w in zip(header, col_widths)),
        sep.join("-" * w for w in col_widths),
    ]
    for row in rows:
        lines.append(sep.join(c.ljust(w) for c, w in zip(row, col_widths)))
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", required=True)
    p.add_argument("--gpu", required=True)
    args = p.parse_args()
    with open(args.cpu) as f:
        cpu = json.load(f)
    with open(args.gpu) as f:
        gpu = json.load(f)
    print(f"CPU run: {cpu['label']}  (pid {cpu.get('pid')})")
    print(f"GPU run: {gpu['label']}  (pid {gpu.get('pid')})")
    print()
    print(render(cpu, gpu))
    return 0


if __name__ == "__main__":
    sys.exit(main())
