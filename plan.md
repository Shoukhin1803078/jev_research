# plan.md — When Does Context Compression Accelerate CPU RAG? (DECAF)

Master working plan for a research paper on reducing RAG latency on CPU-only hardware. This is the persistent, version-controlled plan; the supervisor-facing narrative lives in `proposals/draft/DECAF_CPU_RAG_Proposal.md`.

> **Revision note (v2).** Revised in response to `feedback.md`. The "first"/"no existing study" claims are removed (recent work: SARA ACL 2026; Perception Compressor NAACL 2025 Findings; BRIEF-Pro ACL 2026 Findings; HS-RAG IEEE 2026; CPU-only quantized-LLM benchmarks IEEE 2026). The contribution is reframed from "a new training-free adaptive compressor" to a **decode-centric characterization of when compression accelerates CPU RAG** + a CPU-aware controller + a break-even analysis + a reproducible benchmark. Three previously co-equal contributions are now hierarchically ordered (characterization + method primary; KV/prefetch transfer secondary).

---

## 0. Constraints and assumptions (fixed)

| Item | Value |
|---|---|
| Compute for experiments | **CPU only — no GPU** |
| Memory / cores | **16 GB RAM, 4–8 cores** |
| Models | **Qwen2.5-3B-Instruct** (primary) and **Qwen2.5-7B-Instruct** (main benchmark + subset) |
| Quantization | **Q4_K_M and Q8_0** core (Q5 optional, skipped initially) |
| Runtime | **llama.cpp** (llama-cpp-python) on CPU; Transformers-CPU cross-check |
| Training | **Fully training-free** — off-the-shelf models only |
| Sources | 22 papers in `papers/`, prior analysis `research_gap.md`, reviewer input `feedback.md` |

## 1. Thesis (reframed)

> Context compression is widely assumed to help RAG latency, but its CPU benefit is not automatic: the compressor costs CPU time, and whether it wins depends on context length, model size and quantization in a way that has not been characterized on CPU. On CPU, **decode dominates** — so the real question is **when reduction in context actually lowers CPU decoding cost enough to justify the compressor's own overhead.**

**Primary RQ:** When does reducing retrieved context actually translate into lower CPU decoding cost, and what compression strategy maximizes answer quality per unit of CPU time?

## 2. Why this shape (GPU → CPU inversion, per folder)

- **KV-cache reuse (8 papers):** all target prefill/TTFT; several state gains shrink when decode dominates; all need vLLM/Triton/CUDA. → not portable and aimed at the wrong phase.
- **Prefetch/pipelining (5):** hide retrieval latency behind generation; on CPU retrieval is ms vs seconds of decode → Amdahl ≈ 0 E2E. → secondary transferability study only.
- **Compression (4 + new literature):** the only axis that helps CPU (shrinks KV cache → cuts prefill and memory-bound decode). But the decode benefit is not isolated (ACC-RAG TTFT-only; CORE-RAG tokens-only; ECoRAG total-only; REFRAG isolates decode but needs 64×H100). New work (SARA, Perception Compressor, BRIEF-Pro) occupies training-free/adaptive compression generally, so DECAF must not claim that space.
- **Visual (3):** not CPU-viable at 3B–4B. Out of scope.
- **New CPU-side literature (SARA, HS-RAG, CPU quantized benchmarks):** show CPU-only RAG and training-free compression are each studied; the **interaction** with decode scaling, quantization and grounding is not.

## 3. Literature synthesis (CPU lens)

### 3.1 Positioning table
See Table 1 of the proposal for the full matrix. Key rows: existing KV-reuse (CacheBlend, TurboRAG, RAGCache, CacheTune, FusionRAG, SpecCache, CoinRAG, PCR) — GPU, prefill/TTFT, no decode measure; prefetch (Predictive Prefetching, TeleRAG, HedraRAG, RAGO, VoiceAgentRAG) — no CPU, no decode; compression (REFRAG, CORE-RAG, ACC-RAG, ECoRAG, SARA, Perception Compressor, BRIEF-Pro) — none CPU-only with decode attribution; CPU-side (HS-RAG, IEEE quantized benchmark) — CPU but no compression/grounding interaction; grounding study (CustomNLP4U 2026) — grounding drop under compression. **DECAF = the only row combining CPU-only + training-free + adaptive + decode measured + grounding measured.**

### 3.2 Verified references to cite (all confirmed via search)
- **SARA** — "Selective and Adaptive Retrieval-augmented Generation with Context Compression", ACL 2026 (2026.acl-long.661; arXiv 2507.05633).
- **Perception Compressor** — "A Training-Free Prompt Compression Framework", NAACL 2025 Findings (2025.findings-naacl.229; arXiv 2409.19272). *Key training-free adaptive baseline.*
- **BRIEF-Pro** — "Universal Context Compression with Short-to-Long Synthesis", ACL 2026 Findings.
- **HS-RAG** — "A Lightweight Baseline for Efficient RAG in CPU-Constrained Environments", IEEE 2026 (11634756). CPU-only, decode over long contexts.
- **Quantized LLM CPU benchmark** — "Performance Benchmarking of Quantized LLMs on CPU-only Systems", IEEE 2026 (11648848). CPU, 3B–7B, latency/memory.
- **Evidence-grounding study** — "Efficiency vs. Verifiability in Evidence-Aware RAG: Does Prompt Compression Preserve Citation Grounding?", CustomNLP4U 2026 (2026.customnlp4u-1.19). 2–4% correctness drop vs 40–50% grounding drop.

### 3.3 Safe gap wording
Use: *"Prior work has studied CPU-only RAG deployment and context compression separately, but the interaction between context compression, CPU decode-time scaling, quantization, and end-to-end latency remains insufficiently characterized under a controlled small-model setting."* Do **not** use "first" / "no existing study" / "no compression paper".

## 4. The system: DECAF

DECAF = **D**ecode-**C**entric **A**daptive evidence selection **F**or CPU-only RAG. Single-pass; decides **before** invoking the decoder (no iterative generation).

```
Query → Retrieve top-N → Sentence segmentation
      → Fine-grained evidence scoring (relevance, coverage, uncertainty reduction)
      → CPU-aware budget controller (estimated marginal CPU decode cost)
      → Adaptive evidence selection (single pass) → Reordering
      → Qwen2.5-3B/7B GGUF via llama.cpp → Answer + telemetry
```

**Core mechanism** (replaces the earlier three-gate design):
```
Score(s_i) = [ Relevance × Coverage × UncertaintyReduction ] / EstimatedCPUCost(s_i)
select while:  MarginalQualityGain(s_i) ≥ λ · MarginalLatencyCost(s_i)
```
- `EstimatedCPUCost(s_i)` derived from the measured decode-scaling curve → makes it CPU-aware.
- Single budget knob `λ`; decoder-free decision.
- **Central principle:** the compressor must be cheaper than the decoding it saves (`T_compress < ΔT_decode`).

Components: cross-encoder scorer (`bge-reranker-base`, ~110M); CPU-aware budget controller; relevance reordering; telemetry/measurement; decoder-free grounding evaluator.

## 5. Theoretical framework and measurement methodology

Latency: `T_total = T_retrieve + T_compress + T_prefill(L) + N_out·TPOT(L)`.

**Roofline is a hypothesis, not a law.** First-order `TPOT(L) ≈ (W + KV(L))/BW_mem`, but CPU inference is also shaped by cache hierarchy, SIMD/vectorization, quantization kernels, GQA/MQA, memory locality, NUMA, thread scheduling, BLAS choice, weight reuse, KV layout. Test empirically (E8) rather than assert.

| Quantity | Definition |
|---|---|
| Compression efficiency | `CE = ΔT_decode / ΔL` |
| Compression overhead | `O_c = T_compress / T_baseline` |
| Net speedup | `S_net = T_baseline / (T_compress + T_prefill^c + T_decode^c)` |
| Quality-adjusted efficiency | `QE = Quality / Latency` (Pareto) |
| Break-even length | `L* = min L : T_compress + T_baseline_decode > T_decode^full` |

## 6. Research questions and hypotheses

**RQs (4):**
1. How does retrieved-context length affect TTFT, TPOT and E2E latency in CPU-only quantized RAG?
2. When does compression produce **net** savings after overhead (what is `L*`)?
3. Can a training-free CPU-aware compressor beat fixed-ratio and existing prompt compression on the quality–latency–grounding Pareto?
4. How do model size and quantization alter the compression–quality–latency relationship?

KV-reuse/prefetch = **secondary analysis**, not an RQ.

**Hypotheses:**
- H1: As context length grows, decode's share of E2E CPU latency grows, and compression gives larger relative gains in TPOT than TTFT.
- H2: For each (model, quant) there is a break-even `L*` (net loss below, net win above).
- H3: A single-pass CPU-aware gate matches/beats fixed-ratio and Perception Compressor on the quality–latency–grounding Pareto.
- H4: Model size and quantization shift `L*` and the frontier.
- H5 (secondary): KV-reuse/prefetch give limited net E2E benefit when CPU decode dominates.

## 7. Experimental plan

### 7.1 Datasets (depth > breadth)
- Main: **HotpotQA (distractor)** and **2WikiMultihopQA** — ~**500 queries each**.
- Generalization: **long-context benchmark** (LongBench multi-doc QA; MuSiQue/NarrativeQA alternates) and **TriviaQA or NQ** — ~**200 queries each**.
- 50 calibration queries per model/quant (threshold selection only; no training). 3B on all; 7B on main + subset.
- Retrieval modes: provided-context (core) + small BM25/dense index (end-to-end baselines). Exclude full 21M DPR corpus.
- **New (feedback):** at least one long-context set is mandatory to measure `L*`; short contexts do not stress the phenomenon.

### 7.2 Setup
- Hardware: fixed CPU (record model, cores, RAM, threads), 16 GB.
- Models: Qwen2.5-3B-Instruct (all) + Qwen2.5-7B-Instruct (main + subset). **Quant: Q4_K_M + Q8_0** (Q5 optional).
- Runtime: llama.cpp (llama-cpp-python), `n_threads` = physical cores.

### 7.3 Baselines
- Retrieval: BM25 top-k; dense top-k; hybrid.
- Compression/selection: full context; fixed top-k; LLMLingua-2; **Perception Compressor** (training-free adaptive); DECAF. ACC-RAG/ECoRAG as reference points where compute permits.
- All on identical CPU hardware/datasets/metrics. GPU-trained (TurboRAG/REFRAG/CORE-RAG/CoinRAG) cited qualitatively with compute barrier stated.

### 7.4 Metrics
Latency: TTFT, **TPOT**, E2E, P50/P95, tokens/s. System: peak RSS, compression ratio, `O_c`, achieved memory BW, CPU utilization, energy (RAPL/powerstat where available). Quality: EM, F1, ROUGE-L. **Grounding (new): evidence recall/precision, answer-support rate, citation correctness.** Derived: CE, O_c, S_net, QE, `L*`. Composite: accuracy–latency Pareto.

### 7.5 Experiments (E1–E9; trimmed ~25–30%)
| ID | Experiment | Tests |
|---|---|---|
| E1 | Context-length scaling L∈{1K,2K,4K,8K,16K,32K}, full vs compressed | RQ1/H1 |
| E2 | Compression ratio (0.8/0.6/0.4/0.2) vs quality | RQ3 |
| E3 | Compressor overhead vs decode savings (`O_c`,`S_net`) | RQ2 |
| E4 | **Break-even `L*`** per model×quant | RQ2/H2 |
| E5 | Quantization interaction (Q4_K_M vs Q8_0) | RQ4 |
| E6 | Model-size interaction (3B vs 7B) | RQ4 |
| E7 | **Grounding preservation** under compression | RQ3 |
| E8 | CPU memory-bandwidth / roofline analysis | §5 |
| E9 | Generalization dataset | RQ3 |

### 7.6 Component analysis
Scorer off/on; CPU-aware cost term on/off (relevance-only); coverage term on/off; reordering on/off; `λ` sweep — quantify what CPU-awareness adds over plain relevance top-k.

### 7.7 Secondary analysis: KV-reuse & prefetch transferability (H5)
llama.cpp prefix/KV cache + async prefetch overlapped with decode; report net E2E and TTFT; secondary, not headline.

### 7.8 Statistical plan
Latency: paired bootstrap CI + median + P95 + effect size. Accuracy (exact-answer): McNemar. Continuous accuracy/grounding: permutation/paired bootstrap. Holm–Bonferroni across the core family. Report absolute + relative; Pareto is the primary story, not significance.

## 8. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Compressor cost erases gains | Single-pass decoder-free ~110M scorer; report `O_c` as a first-class result |
| Adaptive gate instability/expense | One gate + one knob `λ`; no iterative generation |
| 7B too slow on 4–8 cores | 3B primary; 7B main + subset |
| 32K points slow on CPU | High-`L` points on a small subset sufficient to fit `TPOT(L)` |
| No native grounding gold | Score against retained gold/supporting evidence |
| Roofline over-simplified | Treat as hypothesis; validate against measured bandwidth/cache/utilization |
| Novelty overlap (SARA/Perception Compressor/BRIEF-Pro) | Position as CPU decode-centric characterization + break-even, not "first training-free adaptive" |

## 9. Timeline
1. Harness + telemetry + methodology (3 w) → 2. E1 + E8 characterization (4 w) → 3. Baselines incl. Perception Compressor (3–4 w) → 4. DECAF (scorer + controller + reordering + grounding) (4 w) → 5. E2–E7, E9, `L*`, Pareto, component analysis (4–5 w) → 6. Secondary + writing (4 w).

## 10. Publication strategy (venue decided)
First version targeted at **efficient-NLP** (ACL/EMNLP efficiency/retrieval, main or Findings; NAACL), with methodology strong enough that systems reviewers respect it; **MLSys / EuroSys / ASPLOS / USENIX ATC** as alternative home if the characterization warrants a systems framing. Decide now, because the two framings imply different lead contributions.

## 11. Repo deliverables and layout
| File | Status | Purpose |
|---|---|---|
| `plan.md` | this file | master working plan |
| `proposals/draft/DECAF_CPU_RAG_Proposal.md` | updated | supervisor-facing proposal (24-section spine) |
| `research_gap.md` | existing | per-paper gap analysis (GPU-era assumptions) |
| `feedback.md` | existing | reviewer feedback driving v2 |
| `papers/` | existing | 22 source PDFs |

## 12. Verification checklist (v2)
1. No "first"/"no existing study"/"no compression paper" claims remain — replaced by "insufficiently characterized".
2. New literature cited and verified: SARA (ACL 2026), Perception Compressor (NAACL 2025 F), BRIEF-Pro (ACL 2026 F), HS-RAG (IEEE 2026), CPU quantized benchmark (IEEE 2026), grounding study (CustomNLP4U 2026).
3. Contribution hierarchy: characterization + CPU-aware method primary; KV/prefetch transfer secondary (not RQ5).
4. Roofline presented as a hypothesis with microarchitectural qualifications; E8 tests it empirically.
5. Measurement framework (CE, O_c, S_net, QE, `L*`) defined; break-even is a first-class experiment.
6. Single-pass CPU-aware controller replaces the three-gate design; `T_compress < ΔT_decode` stated as central principle.
7. Datasets add a long-context benchmark; depth > breadth (500×2 main, 200×2 generalization); quant Q4_K_M+Q8_0.
8. Metrics add **grounding** and **energy**; baselines add Perception Compressor + retrieval baselines.
9. Experiments trimmed to E1–E9; stats use McNemar/permutation + absolute & relative.
10. Venue framing decided explicitly.

## 13. Immediate next steps
1. Confirm CPU/cores/RAM + llama.cpp build.
2. Fix dataset subsets/splits (incl. the long-context set) and model hashes.
3. Build harness + measurement methodology (E1, E8).
4. Implement the single-pass CPU-aware controller.
5. Run E2–E7, E9 and the secondary transferability analysis.
