i want to search some paper related to handle/reducing execution time of rag that means latency . please focus on only 2025-2026 years


I'll search for RAG latency papers from 2025 and 2026.

Here are RAG latency papers from 2025 and 2026, grouped by where they cut time. Profile your own pipeline first, because the papers disagree on what dominates latency. One shows prefill taking 95.53% of inference time. Another shows retrieval at only 0.01–0.4% of end-to-end latency. A third shows generation dominating when the LLM is called through an API.

## 1. Cutting prefill / time-to-first-token (KV-cache reuse)
- [From Prefix Cache to Fusion RAG Cache](https://arxiv.org/pdf/2601.12904) (SIGMOD 2026) reports prefill at 95.53% of inference time. It reuses cached document KV states while recovering the quality that naive reuse loses.
- [SpecCache: Speculative KV Cache Reuse for Efficient RAG Serving](https://aclanthology.org/2026.acl-long.859/) (ACL 2026) cuts time-to-first-token (TTFT) by 2.17–3.95× and raises throughput 2.7–5.2× against full recomputation.
- [TurboRAG](https://aclanthology.org/2025.emnlp-main.334.pdf) (EMNLP 2025) precomputes KV caches per chunk offline. It reports an 8.6× average TTFT speedup, with a peak of 9.4%×.
- [PCR: Prefetch-Enhanced Cache Reuse for Low-Latency RAG Serving](https://arxiv.org/html/2603.23049v1) (2026)
- [Adaptive KV Cache Reuse for Fast Long-Context LLM Serving](https://arxiv.org/html/2605.24022v1) (2026)
- [CoinRAG](https://arxiv.org/pdf/2608.07458) (2026) targets a P99 TTFT under 100 ms by reusing KV caches of pre-extracted information nuggets.
- [CacheBlend](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/09/eurosys25-final999.pdf) (EuroSys 2025) cuts TTFT 2.2–3.3× by selectively recomputing KV for fused chunks.
- [RAGCache](https://dl.acm.org/doi/10.1145/3768628) was published in November 2025 (first on arXiv in 2024). It caches KV for frequently retrieved documents and overlaps retrieval with generation.

## 2. Overlapping, prefetching and pipelining
- [Predictive Prefetching for RAG](https://arxiv.org/pdf/2605.17989) (ICML 2026) predicts when retrieval will be needed and what to fetch. It reports 43.5% lower end-to-end latency.
- [TeleRAG](https://arxiv.org/pdf/2502.20969) (2025) uses lookahead retrieval, prefetching data while the LLM generates.
- [HedraRAG](https://arxiv.org/pdf/2507.09138) (2025) coordinates generation and retrieval with speculative execution on heterogeneous hardware.
- [RAGO](https://people.csail.mit.edu/suvinay/pubs/2025.rago.isca.pdf) (ISCA 2025) is a systematic RAG serving optimizer. It reports a 55% TTFT reduction against LLM-only serving systems.
- [VoiceAgentRAG](https://arxiv.org/html/2603.02206v1) (2026) uses a pre-fetching agent and a semantic cache. It reports a 316× retrieval speedup, but the authors note generation dominates total time.

## 3. Shortening context (compression)
- [REFRAG](https://arxiv.org/pdf/2509.01092) (2025) reports a 30.75× TTFT speedup by replacing most retrieved chunks with precomputed embeddings.
- [Less Is More: Performance-Driven Context Compression (CORE-RAG)](https://arxiv.org/html/2508.19282v4) (ICML 2026) beats a 10-document baseline using about 3% of the tokens.
- [Enhancing RAG Efficiency with Adaptive Context Compression (ACC-RAG)](https://aclanthology.org/2025.findings-emnlp.1307.pdf) (EMNLP 2025 Findings) varies the compression rate per query.
- [ECoRAG](https://arxiv.org/pdf/2506.05167) (ACL 2025 Findings) compresses based on evidence quality and reports lower latency.
- [A2C-RAG](https://dl.acm.org/doi/10.1145/3819836.3819877) (2026) uses a token budget and reports up to 71% fewer prompt tokens on HotpotQA.
- BRIEF-Pro (ACL 2026 Findings), EXIT (ACL 2025 Findings) and AttnComp (EMNLP 2025 Findings) are also in this line, but I didn't pull links.

## 4. Visual-document retrieval speed (relevant to multimodal RAG)
- [Hierarchical Patch Compression for ColPali](https://arxiv.org/pdf/2506.21601) (2025) prunes and quantizes patch vectors. A secondary summary reports a 32× smaller index and a 2–4× latency gain at over 98% nDCG@5 retention. I haven't checked those numbers in the paper.
- [DocPC](https://arxiv.org/pdf/2608.25434) (2026) represents a whole multi-page document with a single composed encoding instead of indexing page by page.
- [PULSAR](https://arxiv.org/pdf/2608.28572) (2026) is a late-interaction retrieval design for enterprise visual-document RAG.

## Caveat for hosted APIs
The KV-cache reuse papers in section 1 need control over your serving stack, such as vLLM or SGLang. If you call a hosted model API, the practical options are context compression (section 3), prefetching and caching (section 2), and fewer or smaller retrieved chunks.

I can narrow this to one area. I can also check whether these papers report latency on the same hardware and metrics, so you can compare them directly.




---


as i will use opensource llm like qwen 8b , llama small model also i can use huggingface free model for doing my reasearch then which paper should i keep for literature overview and doing a new method or fusioning exixting method. suggest some paper


I'll check which of these papers have open code you can run with small open models.

With Qwen 8B or small Llama models, the strongest research base is the KV-cache and context-compression line of work. Those papers have open implementations you can run, and a few of them leave real gaps for a fusion method. The retrieval-side papers (prefetching, lookahead) are mostly systems work and need more engineering.

## Papers to keep for the literature overview
**Core baselines**
- [TurboRAG](https://aclanthology.org/2025.emnlp-main.334/) (EMNLP 2025): offline per-chunk KV caches.
- [CacheBlend](https://arxiv.org/pdf/2405.16444) (EuroSys 2025): selective KV recomputation when chunks are fused.
- [RAGCache](https://dl.acm.org/doi/10.1145/3768628) (published November 2025): a KV cache for hot documents, plus retrieval-generation overlap.
- [REFRAG](https://arxiv.org/pdf/2509.01092) (2025): replaces most retrieved chunks with embeddings, so the decoder sees fewer tokens.

**Newest competitors you should cite**
- [Fusion RAG Cache](https://arxiv.org/pdf/2601.12904) (SIGMOD 2026): cross-chunk context for cached KV.
- [SpecCache](https://aclanthology.org/2026.acl-long.859/) (ACL 2026): a small speculative model picks which tokens to recompute in the large model.
- [PCR](https://arxiv.org/html/2603.23049v1), [Adaptive KV Cache Reuse](https://arxiv.org/html/2605.24022v1) and [CoinRAG](https://arxiv.org/pdf/2608.07458) (all 2026).

**Pipeline and systems view**
- [Predictive Prefetching](https://arxiv.org/pdf/2605.17989) (ICML 2026)
- [TeleRAG](https://arxiv.org/pdf/2502.20969) (2025)
- [RAGO](https://people.csail.mit.edu/suvinay/pubs/2025.rago.isca.pdf) (ISCA 2025): useful for profiling methodology.

**Compression**
- [ACC-RAG](https://aclanthology.org/2025.findings-emnlp.1307.pdf) (EMNLP 2025 Findings)
- [ECoRAG](https://arxiv.org/pdf/2506.05167) (ACL 2025 Findings)
- [CORE-RAG](https://arxiv.org/html/2508.19282v4) (ICML 2026)

**Background**
- The [RAG survey](https://arxiv.org/pdf/2506.00054) and the [agentic RAG survey](https://arxiv.org/html/2501.09136v4) cover the efficiency and latency framing.

I'd skip vendor and Medium guides, and VoiceAgentRAG unless you target voice.

## Best bases for a new method (open code, small models)
| Paper | Open code | Notes |
|---|---|---|
| TurboRAG | [MooreThreads/TurboRAG](https://github.com/MooreThreads/TurboRAG) | Ships a modified Qwen2 implementation. Its README shows average TTFT of 0.65 s with cache against 4.13 s without, on 10 documents. I didn't check whether it needs fine-tuned weights. |
| CacheBlend | [LMCache](https://github.com/LMCache/LMCache) (Apache-2.0, works with vLLM) | Docs show a Llama-3.1-8B and Mistral-7B setup, with a configurable recompute ratio (0.15 in the docs example). The docs mark the in-process blending mode as deprecated, so use the newer mode. |
| REFRAG | Meta's official repo was announced, but I couldn't confirm a release. | Community reimplementations exist, including [one using Llama-3.2-3B](https://github.com/simulanics/REFRAG) and [one for a single GPU with Llama-3.1-8B](https://github.com/n33levo/refrag-lite). Treat them as unofficial. It needs several training stages, so it is the heaviest to reproduce. |
| CORE-RAG | The paper links an ICML 2026 repo (`ziqiangcui/CORE-RAG-ICML26`). | I haven't opened it. |
| SpecCache | I didn't find code. | The idea is still easy to re-implement on top of LMCache. |

TurboRAG's paper lists storage overhead as a limitation. By my arithmetic from its own formula (28 layers, 8 KV heads, 128 dimensions, FP16), one 512-token chunk takes about 56 MiB. A 10,000-chunk corpus would need roughly 0.5 TB, so check your model's config and your disk budget early.

## Fusion ideas worth testing (my suggestions, untested)
1. **Compress, then cache.** Compress each chunk offline (ACC-RAG, ECoRAG or CORE-style), then store the compressed chunk's KV. Compression and KV reuse are usually studied separately, and compression breaks cache keys when it runs online.
2. **Small-model recompute selection.** Use a small same-family model (for example a 0.6B Qwen) to choose which tokens to recompute, following SpecCache, but apply it on top of TurboRAG-style stored chunks instead of CacheBlend's fixed ratio. Verify the models share a tokenizer.
3. **Predictive KV preloading.** Use Predictive Prefetching's retrieval predictor, but prefetch the chunks' KV from CPU or disk into GPU memory. TurboRAG notes that CPU-to-GPU transfer is its overhead, and TeleRAG shows overlapping transfers with generation helps.
4. **Latency-budget routing.** A per-query router chooses between no retrieval, compressed context, and cached full context, targeting a latency budget. The agentic RAG survey flags latency as a gap there.
5. **Document and multimodal angle.** If you continue the financial document work, measure the accuracy-latency tradeoff on FinRAGBench-V or FinMMDocR. I didn't check whether those papers report latency.

## Practical cautions
- Report TTFT, end-to-end latency and throughput alongside accuracy, and state your hardware. The papers disagree on what dominates latency because they use different setups.
- TurboRAG used LongBench multi-document QA and CacheBlend used 2WikiMQA, so those are safe benchmark choices for comparison.
- Free Hugging Face models still need a GPU. A Qwen 8B in bf16 needs about 16 GB for weights alone, before any KV cache.

Tell me which of the four lines you prefer (KV reuse, compression, prefetching or routing), and I can pull the experimental setup, datasets and baselines from three or four of these papers so you can plan the study.