---
title: M6 Phase 1: 2026-05 competitive landscape table
tags: [milestone, m6, phase1, literature, landscape]
source: internal
created: 2026-05-16
updated: 2026-05-16
---

# Landscape Table (2026-05): GRPO-family × agent / tool / retrieval × small-LM / efficiency

> Phase 1 deliverable of [`MILESTONE_6.md`](MILESTONE_6.md). One row per paper that materially shapes positioning of the M6 picked pair. Seeds: (a) the six external entries in [`MILESTONE_6.md` §7](MILESTONE_6.md#sources-of-truth-in-order), (b) the 18 entries surfaced by [`LITERATURE_GAP_AUDIT_2026-05-16.md`](LITERATURE_GAP_AUDIT_2026-05-16.md), (c) the four foundational papers (Search-R1, ReSearch, R1-Searcher, E2H) the project descends from.
>
> Schema (columns): paper, year, venue, task, algorithm family, model + size, compute reported, reward shape, headline metric, one-sentence contribution. **No editorialising in the rows** (per the Phase-1 protocol); positioning analysis is in §3 below.

**Status**: frozen 2026-05-16. Revise only on a new must-engage paper.

---

## 1. Foundational papers (the project descends from these)

| # | Paper | Year | Venue | Task | Algorithm | Model + size | Compute | Reward | Headline metric | One-sentence contribution |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 | Search-R1 [arXiv:2503.09516](https://arxiv.org/abs/2503.09516) | 2025 | NeurIPS 2025 | 7-dataset open-domain QA + multi-hop | GRPO with `<search>…</search>` tool tags | Qwen2.5-3B base + instruct | 8× H100, ~5 days | EM-only on `<answer>X</answer>`, observation masking | EM avg 0.312 (base) / 0.336 (instruct) on 7 benchmarks | Introduces in-loop retrieval with masked-observation GRPO; reference recipe for retrieval-augmented RL. |
| F2 | ReSearch [arXiv:2503.19470](https://arxiv.org/abs/2503.19470) | 2025 | NeurIPS 2025 | Multi-hop QA (NQ, HotpotQA, 2Wiki, MuSiQue) | GRPO with `<search>`/`<result>` tags | Qwen2.5-7B / 32B base + instruct | Not stated (multi-GPU) | F1 on `<answer>X</answer>` + 0.1 partial-credit floor (format-OK but F1=0) | EM 0.40-0.47 avg on 4 benchmarks | Multi-hop reasoning emerges from GRPO + retrieval interleaving alone; the recipe M5.1 ports. |
| F3 | R1-Searcher [arXiv:2503.05592](https://arxiv.org/abs/2503.05592) | 2025 | Pre-print | Multi-hop QA | Two-stage curriculum (SFT cold-start → GRPO) | Qwen2.5-7B base | Not stated | Outcome + format | EM 0.40-0.50 on hot/2wiki | Base-model cold-start requires SFT before GRPO can take hold. |
| F4 | E2H curriculum [arXiv:2506.06632](https://arxiv.org/abs/2506.06632) | 2025 | Pre-print | Multi-hop QA | GRPO + easy-to-hard data scheduling | Qwen2.5-3B / 7B | Not stated | Outcome | +3-5 pp EM over flat data | Curriculum from NQ → HotpotQA → MuSiQue lifts the harder benchmarks. |

## 2. 2026 competitive landscape (the picked pair must position against these)

### 2a. Direct competitors (must engage in related work)

| # | Paper | Year | Venue | Task | Algorithm | Model + size | Compute | Reward | Headline metric | One-sentence contribution |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | Tree-GRPO [arXiv:2509.21240](https://arxiv.org/abs/2509.21240) | 2025 | ICLR 2026 | Multi-hop QA (Search-R1 setup) | Tree-search rollouts with prefix sharing | Qwen2.5-3B | Not stated | Outcome (same as Search-R1) | Matches Search-R1 at 1/4 rollout budget | Prefix-sharing across siblings cuts the rollout budget without losing reward. |
| C2 | DIVA-GRPO (informal cite via [Agentic RL Survey](https://arxiv.org/abs/2509.02547)) | 2025 | Pre-print | Math + multi-hop QA | Group-divergence-aware GRPO | 1.5B-7B | Not stated | Outcome | 2.55× step + 1.76× wall-clock reduction | Variance-aware group selection cuts wasted rollouts. |
| C3 | MC-GRPO [arXiv:2601.22582](https://arxiv.org/abs/2601.22582) | 2026 | Pre-print | Math (GSM8K, MATH) | GRPO with median-of-group baseline, G+1 rollouts | Qwen3-1.7B | Not stated | Outcome | 78.90 % → 83.54 % GSM8K at G=2 | Median baseline beats mean at small G; not yet applied to search-tool. |
| C4 | JustRL [arXiv:2512.16649](https://arxiv.org/abs/2512.16649) | 2025 | ICLR 2026 blogpost | Math reasoning | Plain GRPO (no KL, no curriculum, fixed hyperparameters) | DeepSeek-1.5B / Nemotron-1.5B | Half the compute of ProRL-V2 | EM-only | Matches or beats ProRL-V2's nine-stage pipeline | "Tricks may hurt": minimal recipe outperforms layered tricks. |
| C5 | How-to-Train-Your-Deep-Research-Agent [arXiv:2602.19526](https://arxiv.org/html/2602.19526v1) | 2026 | Pre-print | Multi-hop QA (Search-R1 setup) | GRPO, multiple reward variants | 3B+ | Not stated | EM, F1, F1 + lightweight action penalties | F1 + action-penalty beats EM and naive F1 | Direct EM vs F1 reward ablation in Search-R1; **partially scoops Candidate C**. |
| C6 | LiteResearcher [arXiv:2604.17931](https://arxiv.org/abs/2604.17931) | 2026 | Pre-print | Deep-research (GAIA, Xbench) | Lightweight agentic-RL framework | 4B | Not stated | Outcome | GAIA 71.3 % / Xbench 78.0 % | Closest published small-model agentic-RL search agent. |
| C7 | DGPO / Compact LMs Search Like Agents [arXiv:2508.20324](https://arxiv.org/abs/2508.20324) | 2025 (v4 Apr 2026) | Pre-print | Multi-hop QA (agentic RAG) | Distillation-guided GRPO | Qwen2.5-0.5B / 1B | Not stated | Outcome + distillation | Recovers search behaviour where pure RL collapses | 0.5-1B agentic search **fails under pure RL**; distillation guidance preserves search. Mechanistic support for Phase-1 Finding 3. |

### 2b. Dense-credit-assignment cluster (positioning shift: F1-only must be defended as minimalism)

| # | Paper | Year | Venue | Task | Algorithm | Model + size | Compute | Reward | Headline metric | One-sentence contribution |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | IGPO [arXiv:2510.14967](https://arxiv.org/abs/2510.14967) | 2025 | ICLR 2026 Poster | Multi-hop QA | GRPO + per-turn information-gain reward | Not stated | Not stated | Dense per-turn (Δ P(correct)) + outcome | Beats outcome-only GRPO | Per-turn marginal-probability gain as dense reward. |
| D2 | TIPS [arXiv:2603.22293](https://arxiv.org/html/2603.22293v1) | 2026 | Pre-print | 7-dataset QA | GRPO + teacher-log-likelihood dense reward | Qwen2.5-7B | Not stated | Dense per-turn + outcome | +11.8 % EM / +13.6 % F1 over PPO | Teacher-LM dense reward shaping outperforms outcome-only. |
| D3 | IG-Search [arXiv:2604.15148](https://arxiv.org/abs/2604.15148) | 2026 | Pre-print | HotpotQA / 2Wiki / MuSiQue / Bamboogle | GRPO + per-search-step IG via counterfactual | Not stated | Not stated | Dense per-search-step (IG) + outcome | Beats Search-R1 on the project's eval suite | Per-search-step information gain via random-doc counterfactual modulation of advantage. |
| D4 | Search-P1 [arXiv:2602.22576](https://arxiv.org/html/2602.22576v1) | 2026 | Pre-print (Tencent) | Multi-hop QA | GRPO + path-centric coverage reward | Qwen2.5-3B / 7B | Not stated | Dense path-centric + outcome | +7.7 avg accuracy over Search-R1 | Order-agnostic step coverage + dual-track scoring vs reference planners. |

### 2c. Additive citations (cheap upgrades to related-work or future-work mentions)

| # | Paper | Year | One-line relevance |
|---|---|---|---|
| A1 | AERO [arXiv:2602.14338](https://arxiv.org/abs/2602.14338) | 2026 | Adaptive rollout + Bayesian-posterior dead-zone avoidance; 48 % compute reduction at 1.5B. |
| A2 | BAPO [arXiv:2602.20722](https://arxiv.org/html/2602.20722v1) | 2026 | Off-policy GRPO with replay buffer of hard samples; lower-bound improvement guarantee. |
| A3 | Path Not Taken [arXiv:2511.08567](https://arxiv.org/abs/2511.08567) | 2025 | Three-Gate Theory: RLVR updates off principal SVD directions; mechanistic foundation for PEFT-for-RLVR. |
| A4 | Learning from Less [arXiv:2604.18381](https://arxiv.org/abs/2604.18381) | 2026 (MLSys 2026 oral) | 5× sample efficiency from mixed-complexity training; relevant to budget framing. |
| A5 | Rethinking E2H [arXiv:2603.27226](https://arxiv.org/html/2603.27226) | 2026 | **Negative result on E2H** for deductive reasoning; reframes Candidate D rationale. |
| A6 | NeMo-RL Speculative Decoding [arXiv:2604.26779](https://arxiv.org/html/2604.26779v1) | 2026 (NVIDIA) | EAGLE-3 + MTP inside NeMo-RL+vLLM; 1.8× rollout throughput at 8B. **Free win on M5.1 substrate**. |
| A7 | Balanced Aggregation [arXiv:2605.04077](https://arxiv.org/abs/2605.04077) | 2026 | Token-vs-sequence aggregation bias; validated on Qwen3-1.7B (project size class). |
| A8 | EqLen [arXiv:2604.17328](https://arxiv.org/html/2604.17328v1) | 2026 | Truncation-induced entropy collapse and learning-tax; informs small-rollout-regime characterisation. |
| A9 | Why GRPO Needs Normalization [arXiv:2601.23135](https://arxiv.org/abs/2601.23135) | 2026 | First rigorous theory for GRPO std-normalisation. |
| A10 | EBPO [arXiv:2602.05165](https://arxiv.org/html/2602.05165) | 2026 | Empirical-Bayes posterior-shrinkage GRPO extension. |
| A11 | REAL [arXiv:2602.05630](https://arxiv.org/abs/2602.05630) | 2026 | Verifiable rewards as classification labels; +6.7 % Pass@1 over DAPO at 1.5B. |
| A12 | Agentic RL Survey [arXiv:2509.02547](https://arxiv.org/abs/2509.02547) | 2025 (Apr 2026 update) | 500+ work synthesis; useful map for related-work coverage check. |

## 3. Positioning analysis (where the picked pair sits in this map)

Three axes that the M6 picked pair must take a defensible position on:

**Axis A: Reward shape**. The 2026 trend is dense / shaped reward (cluster D1-D4 + C5). Outcome-only (F1 on `<answer>X</answer>`) is the minority position. The M5.1 recipe lands on F1-only (ReSearch-paper recipe with the 0.1 floor dropped, per Phase-1 Finding 1). Defending F1-only requires either (i) a positive ablation showing it matches or beats the cluster at 0.8B, or (ii) framing it as deliberate minimalism with the 0.1-floor mechanism story as the wedge. The picked pair's reward-shape ablation supplies the evidence for (i) and inherits (ii) from Phase-1 Finding 1.

**Axis B: Rollout efficiency**. Tree-GRPO (C1), DIVA-GRPO (C2), AERO (A1), BAPO (A2) all attack wasted rollouts under different mechanisms. MC-GRPO (C3) attacks small G specifically. The project's M5.1 substrate (G=5, NeMo-RL, MuSiQue-only) sits in the small-G corner of this map. The honest position: "characterise the small-rollout regime for search-tool RL at sub-1B; defer altering it to a follow-up." This makes the picked pair a *characterisation* contribution, not an efficiency contribution, and earns the small-scale scope.

**Axis C: Scale**. Most cited papers are 1.5B-7B, with C7 DGPO and C6 LiteResearcher the only sub-2B reference points. DGPO argues 0.5-1B agentic RAG fails pure RL; M5.1's first 16 steps (shrink-and-improve regime, Phase-1 Finding 3) is *evidence against* the pure failure mode at 0.8B but not yet a positive convergence claim. The picked pair's results land in this gap.

## 4. Positioning paragraph (200 words; for the paper's related-work section)

> *Search-augmented RL at sub-1B is sparsely populated. Search-R1 (Jin et al., 2025) and ReSearch (Chen et al., 2025) define the reference recipe at 3B-32B; R1-Searcher (Song et al., 2025) shows base-model cold-start needs SFT below 7B. At 2026, the field bifurcates along three axes. Dense-reward shaping (IGPO, TIPS, IG-Search, Search-P1) replaces outcome-only F1 with per-turn or per-search-step signals; outcome-only is now a minority position. Rollout-efficiency methods (Tree-GRPO, DIVA-GRPO, AERO, BAPO, MC-GRPO) attack wasted rollouts under different mechanisms, with Tree-GRPO at ICLR 2026 the closest direct comparator on the Search-R1 setup. Reward-design ablations are rare; concurrent work (Wang et al., 2026, arXiv:2602.19526) studies EM vs F1 vs F1+penalties in Search-R1 at 3B+ and finds F1+action-penalties wins. At 0.5-1B specifically, DGPO (Liu et al., 2025) shows pure RL collapses search behaviour at this scale; LiteResearcher (Chen et al., 2026) is the closest published small-model search agent at 4B. We characterise the F1-only minimal-recipe regime at 0.8B on a single A100, with two ablations chosen to address the most adjacent prior work directly.*

## 5. Cross-references

- Audit that fed this table: [`LITERATURE_GAP_AUDIT_2026-05-16.md`](LITERATURE_GAP_AUDIT_2026-05-16.md).
- Project measurements the picked pair builds on: [`PHASE_1_SALVAGE.md`](PHASE_1_SALVAGE.md), [`DATA_AUDIT_PHASE_1.md`](DATA_AUDIT_PHASE_1.md).
- Picked pair (downstream): [`CANDIDATE_EXPERIMENTS.md`](CANDIDATE_EXPERIMENTS.md), [`PICKED_PAIR.md`](PICKED_PAIR.md), [`PUBLICATION_FRAMING.md`](PUBLICATION_FRAMING.md).
- Frozen 2026-05-04 literature arm: [`../research/SURVEY_FOCUSED.md`](../research/SURVEY_FOCUSED.md), [`../research/PARADIGM_REVIEW.md`](../research/PARADIGM_REVIEW.md).
