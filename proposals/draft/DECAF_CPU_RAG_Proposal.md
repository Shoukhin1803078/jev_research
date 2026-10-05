**RESEARCH PROJECT PROPOSAL**

**DECAF**

**Decode-Centric, Training-Free Adaptive Context Compression for Low-Latency Retrieval-Augmented Generation on CPU-Only Small Language Models**

*The first accuracy–latency characterization of RAG context compression on CPU-only, quantized 3B/7B models, isolating the decode-side (TPOT) benefit that the GPU-era literature never measures.*

Prepared by

**Tasnia Haque**

Research Project Proposal — Efficient Retrieval-Augmented Generation

AI / ML & Systems-for-ML Research

---

# Document Roadmap

This proposal is organized into twenty-one sections spanning motivation, literature positioning, theoretical framing, system design, experimental protocol, evaluation, risk management and publication planning. Sections 1–6 establish the problem, the literature gap and the research questions; Sections 7–10 specify the proposed framework and its technical modules; Sections 11–18 specify the experimental protocol, baselines, ablations, transfer study and statistical plan; Sections 19–21 address reproducibility, timeline, publication strategy and conclusion.

---

# 1. Executive Summary

Reducing latency in Retrieval-Augmented Generation (RAG) has become an active research area across 2025–2026, but the field's four main axes — (i) KV-cache reuse, (ii) retrieval prefetching/pipelining, (iii) context compression, and (iv) visual-document retrieval compression — have been developed almost exclusively on data-center or consumer-**GPU** hardware with 7B–70B+ models. A systematic review of 22 papers in these areas (companion document `research_gap.md`) finds **no study whose primary evaluation regime is CPU-only inference of small, quantized open models (3B–7B)**, and **no compression paper that isolates how much of its speedup comes from decoding (per-token latency) rather than prefill (time-to-first-token).**

This distinction matters because the CPU cost profile is inverted relative to the GPU one. On GPU, prefill dominates (one paper reports prefill at **95.53%** of inference time). On CPU, where weights and the KV cache are streamed from DRAM at limited bandwidth and generation is inherently sequential, **per-token decode dominates end-to-end latency**, and it scales with context length because the KV cache does. Consequently, the KV-reuse and prefetching techniques built for prefill-dominated GPU serving transfer poorly: prefetching can hide at most a few milliseconds of local retrieval behind seconds of decoding, and KV-reuse improves TTFT while leaving decode untouched.

Context compression is the one intervention that attacks **both** phases — it shrinks the KV cache and therefore reduces the memory traffic that dominates CPU decoding. Yet the literature leaves exactly this untested: ACC-RAG reports first-token time only, CORE-RAG reports token count only, ECoRAG reports total latency, and REFRAG — the only paper to isolate decode — requires **64×H100 continual pretraining** and is not reproducible on a CPU workstation.

This proposal introduces **DECAF**, a fully **training-free** adaptive extractive compressor, designed and evaluated entirely on a CPU-only, 16 GB / 4–8 core machine with Qwen2.5-3B/7B in GGUF form. DECAF contributes: (i) the **first prefill-vs-decode attribution** of compression's benefit on CPU; (ii) a **training-free adaptive** compression rate that closes the adaptive-rate gap ACC-RAG's own authors flag as their "largest bottleneck"; and (iii) a **memory-traffic (roofline) model** explaining why CPU compression trade-offs differ from published GPU numbers, plus an honest **transfer study** of the KV-reuse and prefetch techniques.

**Primary Research Question:** On a CPU-only, small-model RAG deployment, is the end-to-end benefit of context compression dominated by decode-time (per-token) reduction rather than prefill (time-to-first-token) reduction — and can a fully training-free adaptive compressor achieve this without degrading answer quality?

---

# 2. Problem Statement

Latency is a first-class constraint for local RAG: every unoptimized prefill or decode pass is felt directly by the user, and on CPU it is the difference between an interactive tool and an unusable one. Three citable 2025–2026 survey-style sources converge on the observation that efficiency under resource constraints is unresolved:

- The **Agentic RAG Survey** (§12.4, "Computational Cost, Efficiency, and Sustainability") states: *"Multi-agent collaboration and iterative retrieval increase latency and resource consumption. Future research must explore cost-aware planning, adaptive inference, and lightweight coordination…"* — and its Table 3 characterizes latency only qualitatively (Low/Moderate/High), with no measured numbers for constrained hardware.
- The same survey (§2.4.3) names retrieval/ranking as a primary latency driver, again without measurements at small-model scale.
- The **RAG Survey** (§4.3, §5.4) treats efficiency as a first-class axis (*"ablating … caching mechanisms … increases inference time up to 4×"*), yet every efficiency method it surveys is evaluated on 7B–70B+ models or GPT-3.5/4-class APIs on unstated GPUs.

At the same time, interest in deploying RAG with **small, open-weight, quantized models on CPU-only hardware** is growing rapidly, driven by privacy, cost and on-device constraints. No paper in the surveyed literature treats this as its primary setting. A method is therefore needed that: (1) compresses retrieved evidence to the minimum content required for a correct answer; (2) is **training-free**, so it works with any off-the-shelf small model; (3) **adapts** its compression rate per query without a trained selector; and (4) is designed and benchmarked under the actual CPU hardware constraints a single researcher or small deployment will face, reporting the metrics that matter there (TTFT, **per-token decode time**, end-to-end latency, peak memory).

---

# 3. Literature Positioning and Research Gap

Between 2025 and 2026, four RAG-latency streams advanced largely in isolation: (i) KV-cache reuse (TurboRAG, CacheBlend, RAGCache, SpecCache, CacheTune, FusionRAG, PCR, CoinRAG); (ii) retrieval–generation overlap and prefetching (Predictive Prefetching, TeleRAG, HedraRAG, RAGO, VoiceAgentRAG); (iii) context compression (REFRAG, CORE-RAG, ACC-RAG, ECoRAG); and (iv) visual-document retrieval compression (HPC-ColPali, DocPC, PULSAR), treated here as out of scope.

***Table 1. Positioning of DECAF against the surveyed 2025–2026 RAG-latency literature***

| Research Stream / Paper | Axis | Training-free? | Runs CPU-only? | Measures decode / TPOT? | Model scale | Hardware in paper |
|---|---|---|---|---|---|---|
| CacheBlend (EuroSys 2025) | KV reuse | ✓ | — | — (prefill-only; benefit grows with batch) | 7B–70B | 2×A40 |
| TurboRAG (EMNLP 2025) | KV reuse | — (888 A100-hrs) | — | — ("orthogonal to decode") | 1.5B–72B | 32×A100 / 1×A100 |
| SpecCache (ACL 2026) | KV reuse | ✓ | — | — (TTFT only) | 1B–14B | 2×A100 |
| CacheTune (2026) | KV reuse | ✓ | — (CPU = cache tier only) | — (TTFT) | 7B–32B | 2×A100 / 2×4090 |
| FusionRAG (SIGMOD 2026) | KV reuse | ✓ | — (Triton kernel) | — | 7B–32B | 4×L20 |
| RAGCache (ACM 2025) | KV reuse | ✓ | — | — (TPOT called out as unaddressed) | 7B–70B | A10G / 2×H800 |
| CoinRAG (2026) | KV reuse | — (~140 GPU-hrs) | — | — (TTFT) | 7B | L40S / RTX PRO 6000 |
| PCR (2026) | KV reuse | ✓ | — (vLLM/CUDA) | — | 7B–14B | RTX 4090 + NVMe |
| Predictive Prefetching (ICML 2026) | Prefetch | — (trained predictor) | — | — | 8B–70B | 8×A100 |
| TeleRAG (MLSys 2026) | Prefetch | ✓ | — (SGLang/CUDA) | — | 3B–22B | 4090 / H100 / 8×H200 |
| HedraRAG (SOSP 2025) | Prefetch | ✓ | — (vLLM + H100) | — | 8B–30B | EPYC + H100 |
| RAGO (ISCA 2025) | Scheduling | ✓ | — (simulation only) | — | 1B–405B | 16–128 XPUs (sim) |
| VoiceAgentRAG (2026) | Prefetch/cache | ✓ | partial (API LLM) | — | GPT-4o-mini | API + Qdrant |
| REFRAG (2025) | Compression | — (64×H100 CPT) | — | ✓ (only one; GPU) | 3B–13B | 64×H100 / 1×A100 |
| CORE-RAG (ICML 2026) | Compression | — (8×H20 GRPO) | — | — (tokens only) | 1.5B | 8×H20 |
| ACC-RAG (EMNLP 2025 F) | Compression | — (~71 h GPU) | — | — (FTIT/TTFT only) | 3B–7B | 1×A100 + 1×A6000 |
| ECoRAG (ACL 2025 F) | Compression | ~ (110M+770M) | — | — (total latency) | reader ≥8B | 8×RTX3090 |
| **DECAF — proposed** | **Compression** | **✓** | **✓** | **✓ (central)** | **3B/7B GGUF** | **CPU, 16 GB** |

No existing row satisfies the four properties DECAF targets simultaneously. Two observations follow directly from the table and from the papers' own reported hardware.

## 3.1 Specific, verified research gaps

1. **No compression paper decomposes its speedup into prefill vs decode.** ACC-RAG measures first-token time (FTIT/TTFT) only; CORE-RAG measures token count only; ECoRAG measures total latency; REFRAG isolates decode (TTIT) but demands 64×H100 continual pretraining. On CPU, where decode dominates, this is precisely the measurement that determines whether compression is worthwhile.
2. **No CPU-only, quantized, small-model evaluation exists** across the 22 papers. Every result is A100/H100/RTX4090-scale with unmatched metrics, making cross-paper latency comparison impossible.
3. **Adaptive compression is always trained.** ACC-RAG's authors call their selector "the largest bottleneck in the entire framework"; CORE-RAG uses GRPO with a 14B judge; REFRAG uses an RL expansion policy. None is training-free, and all are out of reach on a CPU-only, no-training budget.
4. **The GPU-cost assumption does not hold on CPU.** Several prefill-oriented papers explicitly state their gains shrink when decode dominates (CacheBlend, TurboRAG, RAGCache, CoinRAG), yet no study measures the decode-only regime.
5. **Prefetch/KV-reuse transfer to CPU is untested.** Amdahl's law predicts that hiding a few-millisecond local retrieval behind seconds of CPU decoding yields ≈0 end-to-end gain, and that KV-reuse moves only TTFT — but nobody has reported this negative result, so it remains an open, citable question.

---

# 4. Theoretical Framework

End-to-end RAG latency decomposes into four additive stages, and the CPU regime re-weights them relative to the GPU regime:

```
T_total = T_retrieve + T_compress + T_prefill(L) + N_out · TPOT(L)

GPU (published):  T_prefill ≫ rest                        (FusionRAG: prefill = 95.53%; CoinRAG: TTFT dominates)
CPU (hypothesis): N_out · TPOT(L) ≫ T_prefill             → decode-dominated end-to-end
TPOT(L) ≈ (W_model + KV(L)) / BW_mem ,   KV(L) ∝ L · d · layers · bytes     [memory-bandwidth-bound]
T_prefill(L) ≈ O(L²) attention compute (fast batched GEMM) + O(L) weight matmuls
```

where `L` is the effective context length, `N_out` the number of generated tokens, `W_model` the model weight bytes and `KV(L)` the KV-cache bytes. The design objective is to minimize `T_total` subject to a quality floor `Q ≥ Q_min` (EM/F1 relative to an uncompressed, fully-recomputed RAG baseline).

The central falsifiable prediction: because CPU decode is memory-bound, shrinking context by a factor `ρ` should reduce TPOT by roughly `(W + ρ·KV)/(W + KV)` and reduce end-to-end latency **more than a prefill-only model would predict**. DECAF tests this prediction directly by logging TTFT and TPOT separately, and by fitting a roofline/memory-traffic model to the measured achieved bandwidth. It also measures `T_compress` explicitly, since a compressor whose own cost exceeds the decode savings is a net loss — a trade-off measured, never assumed.

---

# 5. Research Questions

1. **RQ1:** On CPU, is the end-to-end benefit of context compression dominated by **decode (TPOT)** reduction rather than prefill (TTFT) reduction?
2. **RQ2:** Can a **training-free** adaptive extractive compressor match or beat fixed-ratio compression at equal answer quality while using fewer tokens?
3. **RQ3:** How does quantization level (Q4/Q5/Q8) interact with compression's latency–quality trade-off on CPU?
4. **RQ4:** What is the accuracy–latency Pareto frontier of DECAF versus no-context, full-context, fixed-ratio, and token-level compression baselines, on 3B versus 7B models?
5. **RQ5 (transfer):** Do KV-cache reuse and retrieval prefetching yield measurable end-to-end gains in the CPU decode-dominated regime?

---

# 6. Research Hypotheses

1. **H1:** Compression's end-to-end gain on CPU is predominantly TPOT-driven (decode share > prefill share), reversing the prefill-dominance reported for GPU serving.
2. **H2:** A training-free adaptive gate (off-the-shelf cross-encoder + a training-free sufficiency check) matches or beats fixed-ratio extraction at equal EM/F1 with fewer retained tokens.
3. **H3:** The accuracy–latency Pareto frontier is steeper on CPU than the GPU literature implies, because TPOT is memory-bound and highly sensitive to context length.
4. **H4:** Off-the-shelf reranker scoring plus a training-free gate recovers most of ECoRAG's trained-module benefit at zero training cost.
5. **H5 (transfer/negative):** KV-cache reuse and retrieval prefetching provide ≈0 net end-to-end benefit on CPU — reuse moves only TTFT, and prefetch costs more than it hides.

---

# 7. Aim and Objectives

**Aim:** to design, implement and empirically validate a training-free adaptive extractive compressor for CPU-only small-model RAG, and to characterize its latency/quality trade-offs with explicit prefill-vs-decode attribution.

- **Objective 1 —** Build a CPU inference harness (llama.cpp / GGUF) with per-stage telemetry: T_retrieve, T_compress, TTFT, **TPOT**, end-to-end latency, peak RSS, and achieved memory bandwidth.
- **Objective 2 —** Implement training-free components: sentence segmentation, question-aware cross-encoder scoring, an adaptive sufficiency gate, and relevance reordering.
- **Objective 3 —** Reproduce fair, same-hardware baselines (full-context, fixed top-k, token-level LLMLingua-2, ECoRAG training-free variant).
- **Objective 4 —** Run the S0–S8 ablation plus the 3B-vs-7B and quantization sweeps.
- **Objective 5 —** Run the CPU transfer micro-study (KV reuse + async prefetch) to test H5.
- **Objective 6 —** Fit the memory-traffic/roofline model that explains the CPU-vs-GPU difference.

---

# 8. Scope and Target Setting

**In scope:** single-turn and multi-hop extractive/open-domain text QA RAG (HotpotQA distractor, 2WikiMultihopQA, Natural Questions, TriviaQA), using provided-context settings (to isolate compression) and a small BM25 index for end-to-end runs; small open-source decoder LLMs (Qwen2.5-3B-Instruct primary; Qwen2.5-7B-Instruct on a subset; Qwen3-4B/8B as an alternative family), GGUF Q4_K_M default (Q5_K_M/Q8_0 sweeps), on a CPU-only 16 GB / 4–8 core machine. All methods are **training-free**.

**Out of scope:** GPU/vLLM serving and data-center batching (RAGO, HedraRAG); visual/multimodal retrieval (HPC-ColPali, DocPC, PULSAR); voice/multi-turn settings (VoiceAgentRAG); and any model fine-tuning, RL training, or continual pretraining — explicitly excluded rather than approximated, because the CPU-only, no-training budget cannot support them.

---

# 9. Proposed System Architecture

The pipeline is organized so that segmentation, scoring, gating and reordering can each be benchmarked independently before composition (mirroring the ablation in §14).

```
Query
  │
  ▼
Retriever (provided context / BM25 over a small corpus)
  │
  ▼
Sentence segmenter (spaCy / NLTK)
  │
  ▼
Question-aware scorer — off-the-shelf cross-encoder (bge-reranker-base, ~110M)
  │
  ▼
Adaptive sufficiency gate  (training-free)
  │  add more sentences until "enough evidence"  (hard budget cap)
  ▼
Relevance reordering (mitigate lost-in-the-middle)
  │
  ▼
Assemble compressed prompt
  │
  ▼
Reader: Qwen2.5-3B / 7B  (GGUF via llama.cpp, n_threads = cores)
  │
  ▼
Answer + per-stage telemetry (TTFT, TPOT, E2E, peak RSS, achieved mem-BW)
```

A separate micro-benchmark branch measures llama.cpp prefix/KV cache and an asynchronous retrieval thread, to test H5.

**Safety/validity boundary:** the compressor may only change *which* retrieved sentences reach the decoder and *when* they are fetched; it never alters sentence content beyond selection. Every decision is logged (retained count, compression ratio, per-stage latency) so answer quality can be attributed back to a specific operating point for failure analysis.

***Table 2. Core modules and research value***

| Module | Function | Baseline it extends |
|---|---|---|
| Sentence segmenter + question-aware scorer | Split retrieved docs; rank sentences by relevance to the query | New composition; uses off-the-shelf `bge-reranker` |
| Training-free sufficiency gate | Decide per query when enough evidence has been included (adaptive ratio) | ECoRAG (ACL 2025 F) / ACC-RAG (EMNLP 2025 F) |
| Relevance reordering | Order retained sentences by relevance to reduce lost-in-the-middle | New |
| CPU telemetry + memory profiler | Per-stage timing, TPOT, peak RSS, achieved memory bandwidth | New — required for Objective 1 |
| Transfer micro-benchmark | Measures KV-reuse and async-prefetch behavior on CPU | CacheBlend / RAGCache / TeleRAG (GPU-era) |

---

# 10. Core Technical Modules

**10.1 Scorer.** Sentence-level, question-aware scoring with an off-the-shelf cross-encoder — `BAAI/bge-reranker-base` (~110M) by default, `bge-reranker-v2-m3` (568M) as a quality variant — run on CPU via sentence-transformers or ONNX. This replaces ECoRAG's trained dual-encoder and is the quantified "cost of training-free."

**10.2 Training-free sufficiency gate (the core novel mechanism).** Three training-free gate designs are implemented and compared:
- **(a) NLI-entailment gate:** a small NLI cross-encoder (`cross-encoder/nli-deberta-v3-small`, ~140M) tests whether the currently retained evidence supports a candidate answer; sentences are added until a confidence threshold is met.
- **(b) Reader self-confidence gate:** the reader (Qwen) answers with the current evidence; if answer-token confidence or first-token entropy is poor, more evidence is added. Uses the reader itself, adding zero extra models.
- **(c) Marginal-gain gate:** evidence is added while the generated answer keeps changing.

All gates carry a hard retention/iteration cap to avoid ECoRAG's acknowledged worst-case reflection loop.

**10.3 Relevance reordering.** Retained sentences are ordered by relevance score to reduce lost-in-the-middle behavior on small models.

**10.4 CPU telemetry and memory profiler.** Logs T_retrieve, T_compress, TTFT, TPOT, end-to-end latency, tokens/s, peak RSS (`resource.getrusage` / psutil), and an achieved-memory-bandwidth estimate used to fit the roofline model of §4.

**10.5 Baselines and diagnostics.** Fixed top-k extraction; LLMLingua-2 token-level compression (training-free, XLM-R); the ECoRAG training-free variant; and the llama.cpp prefix/KV-cache + async-prefetch micro-benchmarks.

---

# 11. Baselines

- **No-context** (parametric-only answer from the target model).
- **Full-context RAG** (all retrieved context, no compression).
- **Fixed top-k sentence selection** (k ∈ {1, 2, 4, 8}).
- **LLMLingua-2** token-level compression (rate sweep; training-free).
- **ECoRAG training-free variant** (off-the-shelf scorer + threshold gate).
- **Fewer-retrieved-docs** (retrieval-side compression).
- **DECAF (full)** — §9/§10.

All baselines are re-run on identical CPU hardware, datasets and metrics — directly resolving the cross-paper incomparability noted in §3.1 (gap 2). Methods requiring GPU training (TurboRAG, REFRAG, CORE-RAG, CoinRAG) are cited qualitatively with their compute barrier stated explicitly rather than approximated at reduced fidelity.

---

# 12. Experimental Methodology

1. Fix the CPU hardware (model, cores, RAM, thread count), the model checkpoints (Qwen2.5-3B/7B) and their GGUF quantizations/hashes, prompt templates, retrieval setup and random seeds before development begins.
2. Build the CPU harness and telemetry; validate it against a manual timing baseline.
3. Reproduce the fixed-ratio, token-level and ECoRAG-training-free baselines.
4. Implement the training-free modules and the three gate variants; select a default from the comparison.
5. Run the S0–S8 ablation (§14) and the 3B-vs-7B, quantization and context-length sweeps.
6. Run the CPU transfer micro-study (§15) and fit the roofline model (§4).
7. Finalize statistical reporting (§18) and the failure analysis.

Eval sets are sized to CPU throughput (~200–300 queries per dataset, with 50 calibration queries); each configuration runs a warmup, then ≥3 repeats per query, reporting medians with confidence intervals. Every stage is logged.

---

# 13. Evaluation Metrics

- **Latency:** time-to-first-token (TTFT), **time-per-output-token (TPOT, ms/token)**, end-to-end latency, P50/P95 tail latency, tokens/second.
- **System:** peak RSS, compression ratio (retained / available tokens), compressor overhead as a fraction of end-to-end latency, achieved memory bandwidth.
- **Quality:** Exact Match (EM) and F1 for QA; ROUGE-L where applicable.
- **Composite:** accuracy-per-second efficiency, reported as a **Pareto frontier** rather than a single number.

---

# 14. Component-Necessity (Ablation) Study

***Table 3. Ablation configurations***

| Config | Components included |
|---|---|
| S0 | No context (parametric only) |
| S1 | Full context (no compression) |
| S2 | Fixed top-k sentence selection |
| S3 | ECoRAG training-free variant |
| S4 | S3 + adaptive sufficiency gate |
| S5 | S4 + reranker scoring |
| S6 | S5 + relevance reordering |
| S7 | Token-level compression (LLMLingua-2) |
| S8 (full DECAF) | Scorer + adaptive gate + reordering |

S8 vs S2 isolates H2 (training-free adaptive vs fixed ratio); the context-length sweep with the TTFT/TPOT split isolates H1; the quant and 3B/7B sweeps address RQ3/RQ4; the gate-variant comparison (a/b/c) selects the default and tests H4.

---

# 15. CPU Transfer Micro-Study (H5)

A focused micro-benchmark measures (i) llama.cpp prefix/KV cache across repeated queries and (ii) an asynchronous retrieval prefetch overlapped with decoding. It reports net end-to-end latency and TTFT, testing whether KV reuse only shifts TTFT and whether prefetch cost exceeds the retrieval latency it hides. This is the proposal's deliberate **negative-result** contribution: it directly tests whether the GPU-era KV-reuse and prefetch techniques transfer to a CPU decode-dominated regime, a question no surveyed paper answers.

---

# 16. Risk Register

***Table 4. Risks and mitigations***

| Risk | Mitigation |
|---|---|
| Compressor (reranker) CPU cost could dominate end-to-end latency | Default to the ~110M `bge-reranker-base`; measure overhead as a %, and prefer the reader-confidence gate (b), which adds no model) |
| 7B model too slow on 4–8 cores for full sweeps | 3B is primary across all configs; 7B runs on a subset for scaling only |
| Adaptive gate may loop / add excessive evidence | Hard retention and iteration caps (ECoRAG's flagged worst case) |
| Quality floor unmet at high compression | Report the Pareto frontier, not a single operating point; keep an evidence floor |
| Dataset scale vs CPU throughput | Small eval sets, fixed seeds, parallelize queries across cores |
| Off-the-shelf scorer weaker than ECoRAG's trained scorer | Position explicitly as the *quantified cost of being training-free*; report the gap |
| Latency numbers are hardware-sensitive and may not transfer | Fully specify CPU, cores, RAM, thread count, GGUF quant and runtime version; report medians with CIs |

---

# 17. Success Criteria

1. All baselines run on identical CPU hardware with a consistent metric set, resolving cross-paper incomparability.
2. **H1** is confirmed or refuted with an explicit TTFT-vs-decode share of the compression gain.
3. The training-free adaptive compressor matches or beats fixed-ratio at equal EM/F1 using fewer tokens (H2).
4. A reproducible accuracy–latency Pareto frontier is produced across 3B/7B and quantization levels (H3/RQ3/RQ4).
5. The transfer micro-study answers H5 in either direction, reported honestly.
6. Code, configurations, seeds and telemetry logs are released in a reproducible form.

---

# 18. Statistical Analysis Plan

Latency comparisons across configurations and baselines use paired bootstrap resampling (matched per query), reported with 95% confidence intervals and effect sizes rather than point estimates. Quality metrics (EM/F1) use paired Wilcoxon signed-rank tests. Holm–Bonferroni correction controls family-wise error across the S0–S8 comparisons. Latency–quality trade-offs are reported as Pareto frontiers, and compressor overhead is reported alongside every speedup so that added complexity is justified by measured benefit.

---

# 19. Reproducibility Plan

- Public release of the compressor, gate, scoring and telemetry code, built on open CPU tooling (llama.cpp / llama-cpp-python, sentence-transformers/ONNX, spaCy/NLTK).
- Exact hardware specification: CPU model, physical cores, RAM, thread count; software: OS, llama.cpp build, library versions.
- Model checkpoints named by hash and quantization (GGUF Q4_K_M/Q5_K_M/Q8_0).
- Fixed random seeds and documented prompt templates.
- Released dataset subsets and held-out splits for HotpotQA, 2WikiMultihopQA, NQ and TriviaQA.
- Per-run telemetry logs (TTFT, TPOT, E2E, RSS, compression ratio) so results can be re-derived.

---

# 20. Timeline and Collaboration Roles

***Table 5. Indicative phased timeline***

| Phase | Activities | Indicative duration |
|---|---|---|
| 1. Harness | CPU harness + per-stage telemetry validated | 2–3 weeks |
| 2. Baselines | Fixed-ratio/k sweep, LLMLingua-2, ECoRAG-training-free | 3–4 weeks |
| 3. Modules | Training-free scorer + gate variants (a/b/c) + reordering | 3–4 weeks |
| 4. Ablation | S0–S8 + 3B/7B + quant + context-length sweeps | 4–5 weeks |
| 5. Transfer + model | KV-reuse/prefetch micro-study; roofline fit | 2–3 weeks |
| 6. Writing | Manuscript, failure analysis, reproducibility artifacts | 3–4 weeks |

***Table 6. Collaboration roles***

| Area | Primary responsibility |
|---|---|
| Research direction, hypothesis framing, manuscript review | Supervisor |
| Harness, baselines, module implementation, experiments | Student (Tasnia Haque) |
| Gate-variant design and ablation review | Joint |
| Statistical analysis review | Joint |
| Venue selection and submission strategy | Joint |
| Reproducibility artifact release | Student (Tasnia Haque) |

---

# 21. Publication Strategy and Conclusion

Target venues fit an **efficiency/NLP** contribution rather than a data-center systems one: ACL/EMNLP (efficiency/retrieval tracks, main or Findings), NAACL, and on-device/efficient-NLP workshops (e.g. ENLSP, SustaiNLP) as a lower-risk fallback. This scope matches the venue tier at which directly comparable single-technique work appears (TurboRAG at EMNLP 2025, SpecCache at ACL 2026, ACC-RAG and ECoRAG at EMNLP/ACL 2025 Findings).

DECAF's contribution is not a new primitive but a **disciplined, decode-first study** of context compression under exactly the constraint — CPU-only, 16 GB, 3B/7B quantized, training-free — that all 22 surveyed papers leave untested. Its value is threefold: the first prefill-vs-decode attribution of compression's benefit on CPU; a training-free adaptive compressor that closes the adaptive-rate gap trained methods leave open; and an honest transfer study of the KV-reuse and prefetch techniques, which theory predicts will not help in this regime. Together these should position the work for a competitive efficiency/NLP venue rather than an incremental single-technique paper.

**Immediate next steps:** (1) confirm the exact CPU/cores/RAM and llama.cpp build with the supervisor; (2) fix dataset subsets, held-out splits and model checkpoint hashes; (3) build the harness and reproduce the S0–S2 baselines; (4) implement and select among the training-free gate variants; (5) run the full ablation and the transfer micro-study, then fit the roofline model.
