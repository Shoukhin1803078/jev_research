# plan.md — DECAF: Decode-Centric, Training-Free Adaptive Context Compression for CPU-Only RAG

Master working plan for a research paper on reducing RAG latency. This is the persistent, version-controlled plan; the supervisor-facing narrative lives in `proposals/draft/DECAF_CPU_RAG_Proposal.md`.

---

## 0. Constraints and assumptions (fixed)

| Item | Value |
|---|---|
| Compute for experiments | **CPU only — no GPU** |
| Memory / cores | **16 GB RAM, 4–8 cores** |
| Models | **Qwen2.5-3B-Instruct** (primary) and **Qwen2.5-7B-Instruct** (subset scaling); Qwen3-4B/8B as an alternative family |
| Quantization | **GGUF Q4_K_M** default; Q5_K_M and Q8_0 as sweeps |
| Runtime | **llama.cpp** (llama-cpp-python) on CPU; HuggingFace/Transformers CPU as a cross-check |
| Training | **Fully training-free** — off-the-shelf models only; no fine-tuning, no RL |
| Corpus | 22 downloaded papers in `papers/` (5 folders) + prior analysis in `research_gap.md` and `chat.md` |

## 1. Thesis

> On CPU, **per-token decode dominates end-to-end RAG latency** (decode is memory-bandwidth-bound), and **context compression reduces it** — but every compression paper in the surveyed literature measures prefill (TTFT), or token count, or total latency, and **never isolates the decode-side (TPOT) benefit**.

DECAF delivers the first **decode-first** characterization of RAG context compression on CPU-only, quantized 3B/7B models, plus a **fully training-free adaptive** extractive compressor that closes the adaptive-rate gap that trained methods (ACC-RAG, CORE-RAG, REFRAG) leave open.

## 2. Why this shape (GPU → CPU inversion, per folder)

The existing `proposals/draft/PACER_Project_Proposal_v2.docx` targets a 24 GB RTX 4090 with vLLM/LMCache and a 3-layer predictor+compression+KV-cache fusion. The CPU-only + training-free constraints invalidate most of it:

- **Folder 01 (KV-cache reuse, 8 papers):** all optimize **prefill/TTFT**, and several state the gain shrinks or vanishes when decode dominates — CacheBlend ("prefill overhead dominant … with larger batch sizes"), TurboRAG ("orthogonal to TurboRAG's prefill savings" for decode), RAGCache ("decoding iterations typically take far less time than the prefill iteration"; TPOT unaddressed), FusionRAG (prefill = 95.53% of inference time), CoinRAG (TTFT dominates given short outputs). All require vLLM / Triton / CUDA / FlashAttention. → **Not portable to CPU, and aimed at the wrong phase.**
- **Folder 02 (prefetch/pipelining, 5 papers):** all hide **retrieval** latency behind generation. On CPU, local retrieval is ~0.1–10 ms while decode is seconds → Amdahl's law caps E2E gain at the retrieval time. VoiceAgentRAG itself: generation "dominates total latency (500–8000 ms)" and the savings are "invisible" without *fast* inference; Predictive Prefetching states benefit "scales with the ratio of decoding cost to retrieval latency"; RAGO shows retrieval <1% in long-sequence regimes. TeleRAG/HedraRAG/Predictive Prefetching are GPU-bound; RAGO is simulation-only. → **Near-zero E2E benefit on CPU honestly reported as a negative result.**
- **Folder 03 (context compression, 4 papers):** the **only** axis that helps CPU, because shorter context shrinks the KV cache and cuts both prefill and the memory-bandwidth-bound per-token decode. **But the decode benefit is never isolated:** ACC-RAG measures FTIT/TTFT only; CORE-RAG measures token count only; ECoRAG measures total latency; REFRAG is the only one to isolate decode (TTIT) but needs **64×H100 continual pretraining** — impossible here. → **The gap DECAF fills.**
- **Folder 04 (visual-document retrieval, 3 papers):** not CPU-viable at 3B–4B backbones (per-page VLM forward pass dominates); HPC-ColPali's "CPU-only" claim covers only Hamming search and its numbers are self-admitted *estimates*. → **Out of scope.**
- **Training wall:** CPU-only rules out TurboRAG (SFT, 888 A100-hrs), CoinRAG (~140 GPU-hrs SFT), REFRAG (64×H100 CPT), CORE-RAG (8×H20 GRPO), ACC-RAG (LoRA compressor + RL selector, ~71 GPU-hrs). Only **ECoRAG's extractive, tiny-module skeleton (~110M scorer + 770M evaluator)** is portable to an unmodified stock Qwen, and it has a training-free variant.

## 3. Literature synthesis (CPU lens)

### 3.1 Positioning table

| Paper (venue) | Axis | Training-free? | Runs CPU-only? | Measures decode/TPOT? | Model scale | Hardware in paper |
|---|---|---|---|---|---|---|
| CacheBlend (EuroSys 2025) | KV reuse | ✓ | ✗ | ✗ (explicitly prefill-only) | 7B–70B | 2×A40 |
| TurboRAG (EMNLP 2025) | KV reuse | ✗ (888 A100-hrs) | ✗ | ✗ ("orthogonal to decode") | 1.5B–72B | 32×A100 train / 1×A100 |
| SpecCache (ACL 2026) | KV reuse | ✓ | ✗ | ✗ (TTFT only) | 1B–14B | 2×A100 |
| CacheTune / AdaptiveKVCacheReuse (2026) | KV reuse | ✓ | ✗ (CPU = cache tier, not inference) | ✗ (TTFT) | 7B–32B | 2×A100 / 2×4090 |
| FusionRAG (SIGMOD 2026) | KV reuse | ✓ | ✗ (Triton Q-Sparse-Attn) | ✗ | 7B–32B | 4×L20 |
| RAGCache (ACM 2025 / arXiv 2024) | KV reuse | ✓ | ✗ | ✗ (notes TPOT unaddressed) | 7B–70B | A10G / 2×H800 |
| CoinRAG (2026) | KV reuse | ✗ (~140 GPU-hrs) | ✗ | ✗ (TTFT) | 7B | RTX PRO 6000 / L40S |
| PCR (2026) | KV reuse | ✓ | ✗ (vLLM/CUDA) | ✗ | 7B–14B | RTX 4090 + NVMe |
| Predictive Prefetching (ICML 2026) | Prefetch | ✗ (trained predictor) | ✗ | ✗ | 8B–70B | 8×A100 |
| TeleRAG (MLSys 2026) | Prefetch | ✓ | ✗ (SGLang/CUDA) | ✗ | 3B–22B | RTX 4090 / H100 / 8×H200 |
| HedraRAG (SOSP 2025) | Prefetch | ✓ | ✗ (vLLM + H100) | ✗ | 8B–30B | EPYC + H100 |
| RAGO (ISCA 2025) | Scheduling | ✓ | ✗ (simulation only) | ✗ | 1B–405B | 16–128 XPUs (sim) |
| VoiceAgentRAG (2026) | Prefetch/cache | ✓ | partial (API LLM) | ✗ | GPT-4o-mini | API + Qdrant Cloud |
| REFRAG (2025) | Compression | ✗ (64×H100 CPT) | ✗ | ✓ (only one; GPU) | 3B–13B | 64×H100 train / 1×A100 |
| CORE-RAG (ICML 2026) | Compression | ✗ (8×H20 GRPO) | ✗ | ✗ (tokens only) | 1.5B→14B reader | 8×H20 |
| ACC-RAG (EMNLP 2025 F) | Compression | ✗ (~71 h GPU) | ✗ | ✗ (FTIT/TTFT only) | 3B–7B | 1×A100 + 1×A6000 |
| ECoRAG (ACL 2025 F) | Compression | ~ (110M+770M) | ✗ | ✗ (total latency) | reader ≥8B | 8×RTX3090 |
| **DECAF (proposed)** | **Compression** | **✓** | **✓** | **✓ (central)** | **3B/7B GGUF** | **CPU, 16 GB** |

### 3.2 Specific, verified gaps

1. **No compression paper decomposes its gain into prefill vs decode.** ACC-RAG = TTFT/FTIT only; CORE-RAG = token count only; ECoRAG = total latency only; REFRAG isolates decode (TTIT) but is GPU-trained.
2. **No CPU-only, quantized, small-model evaluation exists** among the 22 papers — every result is A100/H100/RTX4090-scale.
3. **Adaptive compression is always trained.** ACC-RAG's own authors call the selector "the largest bottleneck in the entire framework"; CORE-RAG uses GRPO; REFRAG uses an RL expand-policy. None is training-free.
4. **Latency numbers are incomparable across papers** (one 4090 → 128 XPUs), and **none report CPU TPOT or peak RSS.**
5. **Prefetch/KV-reuse transfer to a CPU decode-dominated regime is untested** and, by Amdahl, expected to fail — an honest negative-result opportunity.

### 3.3 Citable survey statements (motivation)

- Agentic RAG Survey §12.4: *"Multi-agent collaboration and iterative retrieval increase latency and resource consumption. Future research must explore cost-aware planning, adaptive inference, and lightweight coordination…"*
- Agentic RAG Survey §2.4.3: retrieval/ranking named a latency driver; §9.1/Table 3 gives only qualitative Low/Moderate/High latency — no measured numbers.
- RAG Survey §4.3 + §5.4: efficiency is a first-class axis (*"ablating caching … increases inference time up to 4×"*), yet all surveyed efficiency methods are 7B–70B+/GPT-3.5-class on unstated hardware.

## 4. The system: DECAF

**Name:** DECAF — **D**ecode-Centric, training-free **A**daptive compression **F**or CPU-only RAG.

**Pipeline**

```
Query
  │
  ▼
Retriever (provided context, or BM25 over a small corpus)
  │
  ▼
Sentence segmenter (spaCy/NLTK)
  │
  ▼
Question-aware scorer — off-the-shelf cross-encoder (bge-reranker-base ~110M)
  │
  ▼
Adaptive sufficiency gate (training-free; see §4.2)
  │  add sentences until "enough evidence"
  ▼
Relevance reordering (mitigate lost-in-the-middle)
  │
  ▼
Assemble compressed prompt
  │
  ▼
Reader: Qwen2.5-3B/7B GGUF via llama.cpp (n_threads = cores)
  │
  ▼
Answer + telemetry (per-stage latency, TPOT, peak RSS, achieved mem-BW)
```

A separate micro-benchmark branch (llama.cpp prefix/KV cache; async retrieval thread) tests hypothesis H5.

**Modules**

- **4.1 Scorer** — `BAAI/bge-reranker-base` (~110M) or `bge-reranker-v2-m3` (568M), CPU via sentence-transformers / ONNX.
- **4.2 Training-free sufficiency gate** — implement and compare three:
  - (a) **NLI-entailment gate**: `cross-encoder/nli-deberta-v3-small` (~140M) tests whether current evidence supports a candidate answer; add until threshold.
  - (b) **Reader self-confidence gate**: read with Qwen; add evidence if answer-token confidence / first-token entropy is poor. Zero extra model.
  - (c) **Marginal-gain gate**: add evidence while the generated answer keeps changing.
  - Hard cap on budget/iterations to avoid ECoRAG's worst-case loop.
- **4.3 Reordering** — sort retained sentences by relevance.
- **4.4 CPU telemetry + memory profiler** — per-stage timing, tokens/s, peak RSS (`resource`/psutil), achieved memory bandwidth (for the roofline fit).

## 5. Theoretical framework (CPU latency model)

```
T_total = T_retrieve + T_compress + T_prefill(L) + N_out · TPOT(L)

GPU (published): T_prefill ≫ rest            (FusionRAG: prefill = 95.53%; CoinRAG: TTFT dominates)
CPU (hypothesis): N_out · TPOT(L) ≫ T_prefill  → decode-dominated E2E
TPOT(L) ≈ (W_model + KV(L)) / BW_mem ,        KV(L) ∝ L · d · layers · bytes   [memory-bound]
T_prefill(L) ≈ O(L²) attention compute (fast batched GEMM) + O(L) matmuls
```

Falsifiable prediction: shrinking context by factor ρ cuts TPOT by ≈ `(W + ρ·KV)/(W + KV)` and cuts E2E **more** than a prefill-only model predicts. Objective: minimize T_total s.t. quality Q ≥ Q_min; the compressor's own overhead (T_compress) must not exceed the decode savings — measured directly, not assumed.

## 6. Research questions and hypotheses

**RQs**
- RQ1 — On CPU, is compression's E2E benefit dominated by **decode (TPOT)** rather than prefill (TTFT)?
- RQ2 — Can a **training-free** adaptive extractive compressor match/beat fixed-ratio at equal quality?
- RQ3 — How does quantization (Q4/Q5/Q8) interact with the latency–quality trade-off?
- RQ4 — What is the accuracy–latency Pareto frontier of DECAF vs no-context, full-context, fixed-ratio, and token-level baselines, on 3B vs 7B?
- RQ5 (transfer) — Do KV-cache reuse and retrieval prefetching yield measurable E2E gains on CPU?

**Hypotheses**
- H1 — Compression's E2E gain is predominantly TPOT-driven (prefill share < decode share), reversing GPU prefill-dominance.
- H2 — A training-free adaptive gate matches/beats fixed-ratio at equal EM/F1 with fewer tokens.
- H3 — The accuracy–latency Pareto is steeper on CPU than the GPU literature implies (TPOT is memory-bound and L-sensitive).
- H4 — Off-the-shelf reranker + a training-free gate recovers most of ECoRAG's trained-module benefit at zero training.
- H5 (negative) — KV-reuse and prefetch provide ≈0 net E2E benefit on CPU; reuse moves only TTFT, prefetch costs more than it hides.

## 7. Experimental plan

### 7.1 Datasets
Four standard RAG QA benchmarks, chosen to match the exact datasets the compression baselines use (ECoRAG, ACC-RAG, CORE-RAG) so accuracy is directly comparable:

| Dataset | QA type | Context provided? | Baselines reporting it |
|---|---|---|---|
| HotpotQA (distractor) | Multi-hop | Yes (gold paragraphs) | CacheBlend, ECoRAG, ACC-RAG |
| 2WikiMultihopQA | Multi-hop | Yes | CacheBlend, ECoRAG |
| Natural Questions (NQ) | Single-hop | Optional (small BM25 index) | ECoRAG, CORE-RAG, ACC-RAG |
| TriviaQA | Single-hop | Optional (small BM25 index) | ECoRAG, CORE-RAG, ACC-RAG |

- Retrieval modes: (1) **provided-context** (isolates compression; all S0–S8 run here); (2) **small `rank_bm25` index** over the datasets' own passages for end-to-end runs + the fewer-docs baseline.
- Excluded: the full 21M-passage DPR Wikipedia corpus — it would make retrieval, not prefill/decode, the dominant cost and confound H1.
- Scale: ~200–300 eval queries/dataset + 50 calibration queries (threshold selection only; no training anywhere). 3B on all four; 7B on a subset.
- Protocol: dataset-native fixed prompt, standard EM/F1 scoring, identical across configs.

### 7.2 Setup
- Hardware: fixed CPU (record model, cores, RAM, thread count), 16 GB.
- Models: Qwen2.5-3B-Instruct (all runs) + Qwen2.5-7B-Instruct (subset). GGUF Q4_K_M default; Q5_K_M/Q8_0 sweeps.
- Runtime: llama.cpp (llama-cpp-python), `n_threads` = physical cores; Transformers-CPU cross-check.

### 7.3 Baselines
- No-context (parametric only).
- Full-context RAG.
- Fixed top-k sentences (k ∈ {1,2,4,8}).
- LLMLingua-2 token-level compression (rate sweep; training-free, XLM-R).
- ECoRAG training-free variant (off-the-shelf scorer + threshold).
- Fewer-retrieved-docs.
All re-run on identical CPU hardware.

### 7.4 Metrics
- Latency: TTFT, **TPOT (ms/token)**, E2E, P50/P95, tokens/s.
- System: peak RSS, compression ratio, compressor overhead as % of E2E, achieved memory BW.
- Quality: EM, F1 (QA); ROUGE-L where applicable.
- Composite: accuracy-per-second; Pareto frontier.

### 7.5 Ablation matrix (S0–S8)
| Config | Components |
|---|---|
| S0 | No context |
| S1 | Full context |
| S2 | Fixed top-k |
| S3 | ECoRAG training-free |
| S4 | S3 + adaptive gate |
| S5 | S4 + reranker scoring |
| S6 | S5 + reordering |
| S7 | Token-level (LLMLingua-2) |
| S8 | Full DECAF |

Cross-sweeps: 3B vs 7B; quant Q4/Q5/Q8; context length L ∈ {512, 1K, 2K, 4K}; gate variant (a/b/c). S8 vs S2 isolates H2; the L-sweep + TTFT/TPOT split isolates H1.

### 7.6 CPU transfer micro-study (H5)
Measure (i) llama.cpp prefix/KV cache on repeated queries and (ii) async retrieval prefetch overlapped with decode. Report net E2E + TTFT; test whether reuse only moves TTFT and whether prefetch costs more than it hides.

### 7.7 Statistical plan
Paired bootstrap over queries (95% CI + effect sizes) for latency; paired Wilcoxon for EM/F1; Holm–Bonferroni across the S0–S8 family; Pareto frontiers rather than single numbers; report compressor overhead so complexity is justified.

## 8. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Compressor CPU cost dominates | ~110M reranker; measure overhead; prefer the (free) reader-confidence gate |
| 7B too slow on 4–8 cores | 3B primary; 7B on a subset only |
| Adaptive gate loops | Hard budget/iteration cap |
| Quality floor unmet at high compression | Report Pareto, not a single operating point |
| Dataset scale vs CPU throughput | Small eval sets, fixed seeds, parallelize across cores |
| Off-the-shelf scorer ≠ ECoRAG's trained scorer | Position as the *quantified cost of training-free* |

## 9. Timeline (indicative phases)
1. CPU harness + telemetry — (T_retrieve, T_compress, TTFT, TPOT, E2E, RSS, mem-BW).
2. Baselines + fixed-ratio/k sweep.
3. Training-free modules + gate variants.
4. S0–S8 ablation + 3B/7B + quant sweeps.
5. Transfer micro-study + roofline fit.
6. Writing + reproducibility artifacts.

## 10. Publication strategy
Position as an **efficiency/NLP** contribution (not data-center systems): ACL/EMNLP (efficiency/retrieval), Findings, NAACL; on-device/efficient-NLP workshops (ENLSP, SustaiNLP) as fallback. Novelty = decode-first CPU characterization + a training-free adaptive compressor + an honest transfer study — not a new primitive.

## 11. Repo deliverables and layout
| File | Status | Purpose |
|---|---|---|
| `plan.md` | this file | master working plan |
| `proposals/draft/DECAF_CPU_RAG_Proposal.md` | to create | supervisor-facing proposal (20-section spine) |
| `research_gap.md` | existing | per-paper gap analysis (GPU-era assumptions) |
| `papers/` | existing | 22 source PDFs |

## 12. Verification checklist
1. Every factual claim traces to a paper's actual number/quote (FusionRAG 95.53%; ACC-RAG "largest bottleneck" + FTIT only; ECoRAG 110M+770M + total-latency only; CORE-RAG tokens only; REFRAG 64×H100 + TTIT; VoiceAgentRAG "generation dominates"; survey §12.4/§4.3).
2. Central gap claim is internally consistent: ACC-RAG (TTFT), CORE-RAG (tokens), ECoRAG (total) do not isolate decode; only REFRAG does, and it is GPU-trained.
3. Every proposed component is CPU-runnable and training-free (llama.cpp GGUF; ~110–570M reranker; optional ~140M NLI; no first-party training).
4. Hardware assumptions stated and consistent: 16 GB / 4–8 cores, Qwen2.5-3B primary + 7B subset, Q4_K_M default.
5. Proposed system makes falsifiable predictions (H1–H5) and pre-commits to reporting the negative result (H5).
6. No code is written for this task — planning/proposal documents only.

## 13. Immediate next steps
1. Confirm exact CPU model/cores/RAM and llama.cpp build with the supervisor.
2. Fix the dataset subsets + held-out split and the 3B/7B checkpoint hashes.
3. Build the CPU harness + telemetry, then reproduce the S0–S2 baselines.
4. Implement the training-free gate variants and pick the default (a/b/c).
5. Run S0–S8 + 3B/7B + quant sweeps, then the transfer micro-study, then fit the roofline model.
