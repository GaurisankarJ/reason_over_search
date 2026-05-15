---
title: COMPUTE_GRANTS — free compute access routes (SURF + EuroHPC)
tags: [compute, grants, surf, eurohpc, snellius, lumi, leonardo, marenostrum, h100, a100, funding]
source: internal
created: 2026-05-15
updated: 2026-05-15
---

# Free compute access routes — SURF + EuroHPC

> **Purpose**: catalogue the academic-grant routes Sanju (Leiden University) can use to fund continued M5.x training without burning the commercial $1000 USD budget. Sister doc to [`HARDWARE_COMPARISON.md`](HARDWARE_COMPARISON.md) (commercial provider comparison) and [`VAST_AI_PLAN_A.md`](VAST_AI_PLAN_A.md) (Vast cost analysis).
>
> **TL;DR**: two viable routes; **SURF Small Compute Applications (NWO)** is the fast, easy, big-allocation path for thesis follow-up work. **EuroHPC Development Access** is the EU-wide fast path with the next biggest available systems. Both require the supervisor to be the formal applicant (Master's students are not eligible PIs at either). Neither is fast enough to help the live M5.1-prod-a4 run before the 2026-06-10 experimentation deadline; both are excellent for post-thesis ablations (M5.5, M5.6, M5.2 sweeps, paradigm-review variants).

## 0. Why this doc exists

Phase-1 ran on ALICE (Leiden local HPC; retired going forward per [CLAUDE.md](../../.claude/CLAUDE.md)). Phase-2/M5 ran on rented Vast / Spheron / RunPod GPUs against a ~$1000 USD budget. That budget is shrinking; M5.5 + M5.6 + M5.2 ablations are queued behind the current M5.1 run. Even at the cheapest verified rates (Spheron H200 at $1.56/h, B200 at $2.25/h) the full ablation triad burns most of what is left. The grant routes here would cover the same work at zero marginal cost.

The two ecosystems:

| Ecosystem | Operator | Target systems | Speed | Allocation size | PI requirement |
|---|---|---|---|---|---|
| **SURF Small Compute (NWO)** | Dutch national | Snellius (Amsterdam) | **1 to 2 weeks** | up to 1,000,000 SBU (~5,200 H100-h or ~7,800 A100-h) | Tenured / tenure-track at NL university |
| **EuroHPC Development Access** | EU JU | LUMI, Leonardo, MareNostrum 5, MeluXina, Karolina, Vega, Discoverer, Deucalion, JUPITER | **2 to 3 weeks** | "small number of node-hours" (typical: 50k node-hours, varies by system) | Employment contract at academic / industry org in MS/AC, valid 3 months after allocation end |
| **EuroHPC Regular Access** | EU JU | same systems | **~4 months** from cut-off to access | Large (10k+ GPU-node-hours) | same as Development |
| **EuroHPC AI for Science** | EU JU | AI-focused subset (MareNostrum 5, LUMI) | Continuously open; cut-offs every 2 months | "Substantial allocation" for AI workloads | same as Development |

For Leiden Master's thesis follow-up at ~5k-10k GPU-h scale, **SURF Small Compute** dominates: faster turnaround, larger allocation, no EU consortium overhead. **EuroHPC Development** is the fallback if SURF's Snellius is busy or you need a system specifically not at SURF.

## 1. SURF Small Compute Applications (Snellius)

Authoritative sources:
- [SURF Small Compute Applications (NWO) page](https://www.surf.nl/en/small-compute-applications-nwo)
- [SURF wiki: Small Compute applications](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660193/Small+Compute+applications)
- [SURF wiki: Snellius partitions + accounting](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius+partitions+and+accounting)
- [SURF wiki: Obtaining an account on Snellius](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660190/Obtaining+an+account+on+Snellius)

### 1.1 Eligibility

The applicant must hold a **tenured or tenure-track appointment** at an NWO-eligible Dutch research org (or a temporary appointment extending past project end with institutional guarantee). Leiden University qualifies. Master's students cannot apply directly; the supervisor is listed as PI and the student as project member.

Leiden is confirmed as an SURF member institution: explicitly listed in the [Snellius European quantum computer consortium announcement](https://www.surf.nl/en/news/surf-hosts-european-quantum-computer) alongside Delft, Antwerp, Nikhef, eScience Center.

> **Local-name gotcha**: the building at Niels Bohrweg 1 on the Leiden campus is called "Snellius" (Faculty of Science) and is unrelated to the SURF Snellius supercomputer. Both are named after the 17th-century mathematician Willebrord Snellius. Don't conflate them when emailing IT.

### 1.2 Allocation size

Per the [Small Compute wiki](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660193/Small+Compute+applications):

| Service | Max per application |
|---|---|
| Snellius | **1,000,000 SBU** (Standard Billing Units), CPU or GPU combined |
| HPC Cloud | 50,000 CPU-core-hours OR 5,000 GPU-hours |
| Storage | 200 GB scratch |
| Support | 4 hours |
| Duration | 1 year, renewable |

Limit: **one application per service per calendar year per applicant**.

### 1.3 GPU-hour conversion (Snellius partitions)

From the [partitions wiki](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius+partitions+and+accounting):

| Partition | SBU/GPU-hour | 1M SBU = | Per-GPU VRAM | Per-node config | Fits M5.1 prod? |
|---|---:|---:|---:|---|---|
| `gpu_a100` | 128 | **7,812 A100-h** | **40 GiB** | 4 GPU, 72-core CPU, 480 GiB RAM | **No** (40 GiB too small for seq=8192) |
| `gpu_h100` | 192 | **5,208 H100-h** | **94 GiB** | 4 GPU, 64-core CPU, 720 GiB RAM | **Yes** |
| `gpu_mig` | varies | smaller per slice | A100 split into MIG | 8 slices/node | No (too small) |

**Target `gpu_h100`.** The A100 partition is more common but its 40 GiB cards repeat the OOM failure mode that already pushed M5.1 off ALICE A100-40GB (the prior CLAUDE.md anchor). 94 GiB H100 is the right size for the M5.1 production yaml as-is.

Per-job limits: max wall time **120 h (5 days)**; partition encourages full-node use; 4× H100 NVLink within node.

### 1.4 What 1M SBU on H100 actually funds for M5.x

5,208 H100-hours, calibrated against live H200 cadence (~10 min/step real wall on H200; H100 SXM is ~70-85% of H200 throughput for memory-bound RL workloads):

| Run | Steps | Wall (1× H100) | SBU used | % of 1M-SBU budget |
|---|---:|---:|---:|---:|
| M5.1 full 2-epoch | 622 | ~125-150 h | ~24-29k | ~3% |
| M5.5 (F1+format) | 622 | ~125-150 h | ~24-29k | ~3% |
| M5.6 (EM-only) | 622 | ~125-150 h | ~24-29k | ~3% |
| **Ablation triad back-to-back** | 1,866 | ~400 h | ~77k | **~8%** |
| M5.2 systems sweep (5 short runs × 50 steps) | ~250 | ~25 h | ~5k | ~0.5% |
| **Total post-thesis program** | — | ~425 h | ~82k | **~8% of budget** |

Roughly **4,800 H100-h of headroom** after the planned ablations. Plenty of room for paradigm-review variants ([`research/PARADIGM_REVIEW.md`](../research/PARADIGM_REVIEW.md)) or longer-schedule paper-faithful retries.

### 1.5 Process

1. Supervisor logs into the [SURF Servicedesk portal](https://servicedesk.surf.nl/).
2. Submits the **Small Compute Application** form (Snellius track).
3. **1 to 2 weeks** to account creation + access.
4. Student is added as project member; gets `ssh` access and SBU budget tied to the project ID.

### 1.6 Open questions for SURF (worth asking before relying on it)

These came up during the May 2026 docs review but didn't resolve:

1. **Is `gpu_h100` SXM5 or PCIe?** The wiki mentions 4 GPU per node but not the interconnect generation. TP=2/TP=4 multi-GPU scaling depends on it.
2. **Queue wait on `gpu_h100`?** SBU billing is for compute time, but a 6-12 hour queue wait per job degrades effective throughput on multi-day runs.
3. **Container vs. bare-metal user env?** The NeMo-RL stack needs a specific torch/vLLM/mamba-ssm combination; SURF's policy on user-provided Apptainer/Singularity vs. the system module stack matters.

## 2. EuroHPC Development Access

Authoritative sources:
- [Call landing page](https://www.eurohpc-ju.europa.eu/eurohpc-ju-call-proposals-development-access_en)
- [Application portal: access.eurohpc-ju.europa.eu/calls/42](https://access.eurohpc-ju.europa.eu/calls/42)
- [Supercomputers access policy + FAQ](https://www.eurohpc-ju.europa.eu/supercomputers/supercomputers-access-policy-and-faq_en)
- [LUMI access summary](https://lumi-supercomputer.eu/access-calls-for-ai-factories/)

### 2.1 Eligibility

PI must have an employment contract at an academic, research, or industrial org in an EU Member State or Horizon-2020-associated country, valid for at least 3 months after the end of the allocation period. Master's students apply through supervisor (same pattern as SURF). Project framing must be "code and algorithm development and optimisation" or "development of AI application methods"; pure scientific production runs go to Regular Access.

Our work fits the framing cleanly: NeMo-RL recipe development (M5.3 system-gain levers; M5.5/M5.6 reward-shape ablations; M5.2 paradigm-review variants) is exactly the "code and algorithm development" the call targets.

### 2.2 Available systems

From the [EuroHPC supercomputers list](https://www.eurohpc-ju.europa.eu/supercomputers/supercomputers-access-policy-and-faq_en):

| System | Country | GPU type | Per-GPU VRAM | Fits M5.1? |
|---|---|---|---:|---|
| **MareNostrum 5** | Spain (BSC) | H100 SXM | 80 GiB | **Yes** (target) |
| **Leonardo** | Italy (CINECA) | A100 SXM | 64 GiB | **Yes** (target) |
| **JUPITER** | Germany (FZJ) | H100 + GH200 | 96 GiB / 96 GiB | Yes (newest) |
| **MeluXina** | Luxembourg | A100 | 40 GiB | **No** |
| **Karolina** | Czech Republic | A100 | 40 GiB | **No** |
| **Vega** | Slovenia | A100 | 40 GiB | **No** |
| **LUMI** | Finland | AMD MI250X | 64 GiB | **No** (ROCm; NeMo-RL + Qwen3.5 hybrid uncharted) |
| **Discoverer** | Bulgaria | A100 | 40 GiB | **No** |
| **Deucalion** | Portugal | A100 + GH200 | 40 GiB / 96 GiB | partial |

**Target systems for M5.1 shape: Leonardo (A100-64GB), MareNostrum 5 (H100), or JUPITER (H100/GH200).** The 40 GiB A100 sites are the same OOM trap as Snellius `gpu_a100`. LUMI's AMD path is a no-go without a separate venv rebuild and Qwen3.5-hybrid validation.

### 2.3 Allocation size

Per the [call page](https://www.eurohpc-ju.europa.eu/eurohpc-ju-call-proposals-development-access_en):

> "Users will typically be allocated a small number of node hours; the allocation period is one year and is renewable up to two times."

Typical "small" is ~50,000 node-hours on the target system (per past calls; not pinned on the public page). On a 4× H100 node at MareNostrum 5 that is ~200,000 GPU-hours, **40× larger than the SURF Small Compute H100 budget**. Realistically not all of it is usable in a year, but the per-application ceiling is much higher than SURF.

### 2.4 Process

1. Register an account on the [EuroHPC Access Portal](https://access.eurohpc-ju.europa.eu/).
2. Verify institutional affiliation (supervisor's Leiden employment).
3. Submit proposal via the Development Access call form at [calls/42](https://access.eurohpc-ju.europa.eu/calls/42).
4. **2 to 3 weeks** to evaluation + access.
5. **Monthly cut-offs**: 1st of each month (Jan, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec 2026).
6. **First-come-first-served** within the monthly quota; early submission within each cut-off window improves odds.

### 2.5 Proposal contents

The full document requirements are in the call's "Information for Applicants" PDF (downloadable from the call page; not extracted to plain-text by the WebFetch tool). From related EuroHPC calls (Regular Access PDF, AI for Science PDF) the standard sections are:

- **Project title** (one line)
- **Abstract** (~300 words)
- **Scientific case** (~2 pages): the research problem, why HPC is needed, expected outcomes
- **Technical case** (~2 pages): the codebase (NeMo-RL fork), the workload shape (GPU-hours per run, memory profile, parallelism), benchmarking evidence (link to our M5.1 live data)
- **Resource request**: target system, node-hours, storage, software stack
- **Project team**: PI (supervisor), collaborators (you), affiliations
- **Risk assessment + ethical statement** (~0.5 page each)

For our case the technical case writes itself: M5.1 live data on H200 (live W&B), the [`PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md) mapping, and the M5.3 / M5.5 / M5.6 ablation plans already specify everything the form asks for.

## 3. EuroHPC Regular Access

The big track. Wrong timing for the thesis but the right vehicle for a follow-up publication-scale program.

| Field | Value | Source |
|---|---|---|
| Cut-offs | Continuous, with cut-off dates triggering evaluation | [call page](https://www.eurohpc-ju.europa.eu/eurohpc-ju-call-proposals-regular-access-mode_en) |
| Time to access | **~4 months** from cut-off | [LUMI Regular Access summary](https://lumi-supercomputer.eu/eurohpc-jus-regular-access-call-open-for-european-researchers/) |
| Allocation windows | Aug 2026 to Aug 2027 OR Jan 2027 to Jan 2028 | same |
| Allocation size | Large (typical: 100k+ GPU-node-hours) | [full call PDF](https://www.eurohpc-ju.europa.eu/document/download/004ebf96-38c2-41a9-ac59-0c252a0267da_en?filename=Regular+Access+-+Full+Call+Details-FINAL.pdf) |
| Tracks | Scientific Access / Industry Access / Public Administration | same |
| Eligibility | Same as Development (PI with 3-month-post-allocation contract) | same |

For Leiden Master's, this is the **post-publication path**: if M5.x results justify a paper, the Regular Access proposal would fund the scaling-up runs (3B model, full paper-faithful 256-batch shape, multi-seed) that the thesis budget cannot.

## 4. EuroHPC AI for Science / AI Factories

The newest track (opened 2024-2025), specifically targeting AI workloads rather than traditional HPC science.

| Field | Value | Source |
|---|---|---|
| Cut-offs 2026 | **30 Apr, 30 Jun, 31 Aug, 30 Oct, 11 Dec** | [LUMI AI Factory summary](https://lumi-ai-factory.eu/articles/eurohpc-ju-access-calls-for-ai-factories/) |
| Target | AI applications for science; LLMs + foundation models explicitly mentioned | [AI for Science call page](https://www.eurohpc-ju.europa.eu/eurohpc-ju-call-proposals-ai-science-and-collaborative-eu-projects_en) |
| Cost | **No cost** for eligible public/private orgs (publicly funded research) | same |
| Available systems | AI Factories subset (BSC AI Factory at MareNostrum 5, LUMI AI Factory) | [systems page](https://www.eurohpc-ju.europa.eu/ai-factories/ai-factories-systems_en) |
| Full call PDF | [EuroHPC AI Access call](https://eurohpc-ju.europa.eu/document/download/f67beb2a-6a97-4f02-b6d7-017c86957867_en?filename=EuroHPC+AI+Access+-+Full+Call+Details.pdf) | same |
| Eligibility | All scientific users (any funding source), public sector, industrial users in EU R&I projects | same |

**This is the highest-fit track for our work**: explicit focus on LLMs + foundation models + AI training workflows. The other EuroHPC tracks treat AI as a science-domain user; this one is built for it. The Q2 cut-off (30 June 2026) is the next realistic target date for a Leiden Master's thesis follow-up program.

## 5. Decision matrix for our workload

| Constraint | SURF Small Compute | EuroHPC Development | EuroHPC AI for Science | EuroHPC Regular |
|---|---|---|---|---|
| Time to access | **1-2 weeks** | 2-3 weeks | ~1 month (next cut-off 2026-06-30) | ~4 months |
| Allocation ceiling | ~5,200 H100-h | ~50k node-h (~200k GPU-h on 4× H100) | substantial | very large |
| PI eligibility | tenured NL | tenured EU | tenured EU | tenured EU |
| Workload framing | open | "code/algorithm dev" | **AI training explicit** | "scientific production" |
| GPU memory available | H100 94 GiB | H100 80 GiB (MareNostrum 5) or A100 64 GiB (Leonardo) | H100/MI300 at AI Factories | same as Development |
| Fits M5.1 prod yaml | yes (`gpu_h100`) | yes (target Leonardo or MN5) | yes | yes |
| Application effort | ~30-45 min for supervisor | ~1 day proposal | ~1 day proposal | ~3-5 days proposal |
| For thesis (deadline 2026-06-10) | **maybe** if applied this week | **no** (turnaround too close to deadline) | no | no |
| **For post-thesis M5.x triad** | **yes, easiest** | yes | yes (best framing) | overkill |
| For publication-scale follow-up | too small | too small | borderline | yes |

## 6. Recommended sequence

1. **This week (2026-05-15 to 2026-05-22)**: supervisor submits **SURF Small Compute Application** for `gpu_h100` partition, 1,000,000 SBU, 1-year duration. Project title: "Retrieval-augmented RL post-training of small language models — Master's thesis ablation runs". Justification leans on Phase-1 + M5.1 deliverables already documented.
2. **In parallel**: register an account on the [EuroHPC Access Portal](https://access.eurohpc-ju.europa.eu/), verify institutional affiliation. No proposal yet; just have the account ready.
3. **After thesis submission (post 2026-06-15)**: depending on what SURF granted:
   - If SURF granted full 1M SBU: run M5.5 + M5.6 + M5.2 sweep on `gpu_h100` over the summer (~8% of budget).
   - If SURF denied or partial: submit **EuroHPC Development Access** for the 2026-07-01 cut-off, targeting Leonardo A100-64GB or MareNostrum 5 H100. Access by ~mid-July.
4. **If M5.x results justify a publication push (Q4 2026)**: prepare a **Regular Access** proposal for the January 2027 allocation window OR an **AI for Science** proposal for the 2026-10-30 cut-off. Framing: extend to 3B + paper-faithful 256-batch shape + multi-seed.

## 7. What supervisor needs (concise)

Hand this list to supervisor so they don't have to chase details:

```
Form: SURF Servicedesk → Small Compute Applications → Snellius
URL: https://servicedesk.surf.nl/ (login with SURFconext / Leiden credentials)

Project title:
"Retrieval-augmented RL post-training of small LMs — Master's thesis ablation runs"

Resource request:
- Snellius, gpu_h100 partition
- 1,000,000 SBU (~5,200 H100 GPU-hours)
- 200 GB project storage
- 1 year, renewable

Project members:
- PI: [supervisor name + Leiden affiliation]
- Member: Sanju (Master's student)

Workload summary:
- Qwen3.5-0.8B GRPO with retrieval tool-use; NeMo-RL framework
- Per-run shape: ~5 days wall-clock, single-GPU H100
- Plan: 3 ablation runs (reward shape: F1, F1+format, EM) + systems sweeps
- Total: ~400-450 GPU-hours (~8% of allocation)

Justification (paste verbatim):
This project reproduces and extends two published methods (ReSearch, arXiv:2503.19470;
Search-R1, arXiv:2503.09516) for retrieval-augmented RL training of small language models.
A complete training pipeline has been validated on commercial cloud (Spheron H200) and
the paper-faithful production configuration is locked. The allocation will fund three
reward-shape ablations (F1-only, F1+format, EM-only) and one systems-efficiency sweep
that together close the open research questions for the Master's thesis follow-up
program. All experiments fit within a single 1M-SBU allocation with significant
headroom for extensions.

Hardware preference: gpu_h100 specifically (94 GiB per GPU). The gpu_a100 partition
(40 GiB) does not fit our memory footprint at the production sequence length.
```

## 8. Pointers

- Sister doc: commercial provider comparison: [`HARDWARE_COMPARISON.md`](HARDWARE_COMPARISON.md)
- Vast cost analysis: [`VAST_AI_PLAN_A.md`](VAST_AI_PLAN_A.md)
- Spheron H200 runbook: [`../spheron/SETUP_SPHERON.md`](../spheron/SETUP_SPHERON.md)
- Active TODO: [`../todo/TODO_2026-05-12.md`](../todo/TODO_2026-05-12.md)
- M5 paper mapping: [`../milestone_5/PAPER_VS_OURS_M5.md`](../milestone_5/PAPER_VS_OURS_M5.md)
- M5.3 ablation plan (systems): [`../milestone_5/MILESTONE_5_3.md`](../milestone_5/MILESTONE_5_3.md)
- M5.5 reward ablation (F1+format): [`../milestone_5/MILESTONE_5_5.md`](../milestone_5/MILESTONE_5_5.md)
- M5.6 reward ablation (EM-only): [`../milestone_5/MILESTONE_5_6.md`](../milestone_5/MILESTONE_5_6.md)

## 9. External sources (May 2026)

SURF:
- [SURF Small Compute Applications (NWO)](https://www.surf.nl/en/small-compute-applications-nwo)
- [SURF wiki: Small Compute applications](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660193/Small+Compute+applications)
- [SURF wiki: Snellius partitions and accounting](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660209/Snellius+partitions+and+accounting)
- [SURF wiki: Obtaining an account on Snellius](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660190/Obtaining+an+account+on+Snellius)
- [Snellius national supercomputer overview](https://www.surf.nl/en/services/snellius-the-national-supercomputer)
- [Leiden + Snellius consortium announcement](https://www.surf.nl/en/news/surf-hosts-european-quantum-computer)
- [NWO Computing Time on National Computing Facilities (round 2025)](https://www.nwo.nl/en/calls/computing-time-on-national-computing-facilities-round-2025)

EuroHPC:
- [Access calls landing](https://www.eurohpc-ju.europa.eu/access-our-supercomputers/eurohpc-access-calls_en)
- [Development Access call](https://www.eurohpc-ju.europa.eu/eurohpc-ju-call-proposals-development-access_en)
- [Benchmark Access call](https://www.eurohpc-ju.europa.eu/eurohpc-ju-call-proposals-benchmark-access_en)
- [Regular Access call](https://www.eurohpc-ju.europa.eu/eurohpc-ju-call-proposals-regular-access-mode_en)
- [Extreme Scale Access call](https://www.eurohpc-ju.europa.eu/eurohpc-ju-call-proposals-extreme-scale-access-mode_en)
- [AI for Science / Collaborative EU Projects call](https://www.eurohpc-ju.europa.eu/eurohpc-ju-call-proposals-ai-science-and-collaborative-eu-projects_en)
- [AI Factories Access Modes](https://www.eurohpc-ju.europa.eu/ai-factories/ai-factories-access-modes_en)
- [AI Factories Systems](https://www.eurohpc-ju.europa.eu/ai-factories/ai-factories-systems_en)
- [Supercomputers Access Policy + FAQ](https://www.eurohpc-ju.europa.eu/supercomputers/supercomputers-access-policy-and-faq_en)
- [Application portal](https://access.eurohpc-ju.europa.eu/)
- [Application portal: Development Access form (call 42)](https://access.eurohpc-ju.europa.eu/calls/42)
- [Regular Access full call PDF](https://www.eurohpc-ju.europa.eu/document/download/004ebf96-38c2-41a9-ac59-0c252a0267da_en?filename=Regular+Access+-+Full+Call+Details-FINAL.pdf)
- [AI Access full call PDF](https://eurohpc-ju.europa.eu/document/download/f67beb2a-6a97-4f02-b6d7-017c86957867_en?filename=EuroHPC+AI+Access+-+Full+Call+Details.pdf)
- [LUMI AI Factory access summary](https://lumi-ai-factory.eu/articles/eurohpc-ju-access-calls-for-ai-factories/)
- [LUMI Regular Access summary](https://lumi-supercomputer.eu/eurohpc-jus-regular-access-call-open-for-european-researchers/)
- [EuroCC Greece application guide](https://eurocc-greece.gr/how-to-apply-for-access-to-eurohpc-ju-supercomputers/)
