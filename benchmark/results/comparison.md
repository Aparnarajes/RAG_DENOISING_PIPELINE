# Day 4 RAG Benchmark

- Queries: 20 (5 per category)
- Systems: Standard RAG and Denoised RAG, same corpus, query set, and top-k.
- Generation: deterministic local extractive mode; external API keys are disabled.
- Semantic embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- NLI evaluator: `cross-encoder/nli-deberta-v3-small`
- Pipeline/index and model initialization (outside query timers): 7.983 seconds
- NLI first-inference warm-up (outside query timers): 4.763 seconds
- Query latency: externally timed `pipeline.run` wall-clock duration; mean, median, nearest-rank p95, min, and max are reported in seconds.
- Expected-fact coverage is normalized exact-token presence in the answer, not proof of factual correctness. The composite Expected-Fact Coverage / Abstention Accuracy averages answerable-query fact coverage with correct abstention on unsupported queries.
- For conflict cases, expected facts are drawn from the higher-reliability corpus source; detecting/displaying the opposing claim is evaluated separately.
- Retrieval precision/recall uses one shared initial top-k retrieval for each query; denoised feedback-retry outputs remain recorded separately.
- Independent answer claims are compared with ground-truth relevant corpus evidence using the NLI model at confidence >= 0.65; only evidence meeting the existing source-trust threshold determines claim status.
- Claims supported and contradicted by credible evidence are labeled CONFLICTED, not SUPPORTED. Supported, unsupported, contradicted, and conflicted claim rates use all extracted factual claims as their denominator.
- Unsupported-claim rate excludes contradicted and conflicted claims. The separate unsupported-or-contradicted rate combines only unsupported and contradicted claims; conflicted claims remain separately reported.
- The system's pipeline hallucination-detector findings and independent benchmark claim evaluation are separate measurements.
- Zero-denominator policy: aggregate precision and rates are 0; category metrics with no applicable queries are n/a. Recall is excluded for unsupported queries and reported with its query count.

## Aggregate comparison

| Metric | Standard RAG | Denoised RAG |
|---|---:|---:|
| Retrieved-document precision | 0.183 | 0.183 |
| Retrieved-document recall | 1.000 | 1.000 |
| Mean irrelevant retrieved documents | 4.900 | 4.900 |
| Mean irrelevant documents passed to generation | 4.900 | 0.050 |
| Noise reduction (overall) | 0.000% | 98.980% |
| Expected-fact coverage (fact-weighted token presence) | 1.000 | 1.000 |
| Expected-fact coverage / abstention accuracy | 0.750 | 1.000 |
| Correct unsupported-query abstention rate | 0.000 | 1.000 |
| Pipeline detector hallucination rate | 0.340 | 0.221 |
| Unreliable documents passed (mean) | 1.950 | 0.000 |
| Generation context characters (mean) | 1043.300 | 164.450 |
| Supported-query factual claim rate | 0.188 | 0.667 |
| Supported claim rate (all factual claims) | 0.136 | 0.667 |
| Unsupported claim rate (all factual claims) | 0.742 | 0.037 |
| Contradicted claim rate (all factual claims) | 0.121 | 0.296 |
| Conflicted/unresolved claim rate (all factual claims) | 0.000 | 0.000 |
| Unsupported-query abstention rate | 0.000 | 1.000 |
| Unsupported-or-contradicted claim rate (independent evaluator) | 0.864 | 0.333 |
| Evidence confidence score (mean; descriptive) | 0.196 | 0.539 |

| Latency statistic | Standard RAG (seconds) | Denoised RAG (seconds) |
|---|---:|---:|
| Mean | 0.1581 | 0.1289 |
| Median | 0.1730 | 0.0876 |
| P95 | 0.2473 | 0.2962 |
| Min | 0.0619 | 0.0108 |
| Max | 0.2477 | 0.5724 |

Denoised expected-conflict detection recall: 1.000 (5/5).
Denoised contradiction handling rate (correct conflicts / conflicting queries): 1.000 (5/5).

Denoised feedback retries: 18 across 20 queries (maximum configured retries per query: 1).

## Results by category

| Category | System | Expected-fact coverage | Supported | Unsupported | Contradicted | Conflicted | Claims generated | Unsupported abstention rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| normal | standard | 1.000 | 8 | 21 | 4 | 0 | 33 | n/a |
| normal | denoised | 1.000 | 8 | 0 | 2 | 0 | 10 | n/a |
| noisy_irrelevant | standard | 1.000 | 5 | 25 | 3 | 0 | 33 | n/a |
| noisy_irrelevant | denoised | 1.000 | 5 | 1 | 0 | 0 | 6 | n/a |
| conflicting | standard | 1.000 | 5 | 16 | 9 | 0 | 30 | n/a |
| conflicting | denoised | 1.000 | 5 | 0 | 6 | 0 | 11 | n/a |
| unsupported | standard | n/a | 0 | 36 | 0 | 0 | 36 | 0.000 |
| unsupported | denoised | n/a | 0 | 0 | 0 | 0 | 0 | 1.000 |

## Per-query results

| ID | Category | Standard expected-fact coverage / abstention accuracy | Denoised expected-fact coverage / abstention accuracy | Standard unsupported claims | Denoised unsupported claims | Standard latency (s) | Denoised latency (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Q01 | normal | 1.000 | 1.000 | 3 | 0 | 0.2477 | 0.5724 |
| Q02 | normal | 1.000 | 1.000 | 5 | 0 | 0.1849 | 0.0954 |
| Q03 | normal | 1.000 | 1.000 | 5 | 0 | 0.2473 | 0.1698 |
| Q04 | normal | 1.000 | 1.000 | 6 | 0 | 0.2264 | 0.0886 |
| Q05 | normal | 1.000 | 1.000 | 2 | 0 | 0.1764 | 0.2099 |
| Q06 | noisy_irrelevant | 1.000 | 1.000 | 6 | 0 | 0.1815 | 0.0867 |
| Q07 | noisy_irrelevant | 1.000 | 1.000 | 5 | 0 | 0.1697 | 0.0812 |
| Q08 | noisy_irrelevant | 1.000 | 1.000 | 5 | 0 | 0.1643 | 0.0736 |
| Q09 | noisy_irrelevant | 1.000 | 1.000 | 7 | 0 | 0.2043 | 0.0747 |
| Q10 | noisy_irrelevant | 1.000 | 1.000 | 2 | 1 | 0.0966 | 0.0800 |
| Q11 | conflicting | 1.000 | 1.000 | 3 | 0 | 0.0987 | 0.1964 |
| Q12 | conflicting | 1.000 | 1.000 | 3 | 0 | 0.0939 | 0.1670 |
| Q13 | conflicting | 1.000 | 1.000 | 3 | 0 | 0.0619 | 0.2962 |
| Q14 | conflicting | 1.000 | 1.000 | 3 | 0 | 0.0968 | 0.1642 |
| Q15 | conflicting | 1.000 | 1.000 | 4 | 0 | 0.0931 | 0.1650 |
| Q16 | unsupported | 0.000 | 1.000 | 6 | 0 | 0.1182 | 0.0116 |
| Q17 | unsupported | 0.000 | 1.000 | 8 | 0 | 0.1992 | 0.0108 |
| Q18 | unsupported | 0.000 | 1.000 | 7 | 0 | 0.1072 | 0.0121 |
| Q19 | unsupported | 0.000 | 1.000 | 8 | 0 | 0.1930 | 0.0115 |
| Q20 | unsupported | 0.000 | 1.000 | 7 | 0 | 0.2005 | 0.0115 |

## Limitations

- This is a small, corpus-specific evaluation, not a broad generalization test.
- Expected-fact scoring uses exact normalized token coverage against facts curated from the corpus; paraphrases may score lower.
- Independent claim grounding depends on the selected NLI model and its confidence threshold; it is an evaluator, not a guarantee of factual truth.
- Timing is warm execution time after local model initialization and a single NLI warm-up; it is not cold-start latency and is hardware-dependent.
- The query dataset is curated and therefore should not be treated as a statistically representative sample.

Full raw inputs, pipeline outputs, per-query evaluations, and aggregates are available in `per_query_results.json` and `summary.json`.
