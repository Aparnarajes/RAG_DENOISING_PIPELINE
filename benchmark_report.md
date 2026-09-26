# RAG Denoising Multi-Agent Pipeline Benchmark Report

**Queries Evaluated:** 8 | **Overall Rubric Score:** 89.0 / 100

## 1. Assignment Evaluation Rubric Alignment (Table 3)

| Criterion | Weight | Standard RAG | Denoised RAG | Lift / Improvement | Rubric Pts |
|---|---|---|---|---|---|
| **Retrieval Precision** | 20% | 0.229 | **0.844** | +61.5 pp | 16.9 / 20 |
| **Noise Reduction** | 20% | 4.6 noisy docs | **0.4 noisy docs** | -91.9% noise | 18.4 / 20 |
| **Answer Accuracy** | 25% | 0.681 | **0.749** | +6.8 pp | 18.7 / 25 |
| **Hallucination Reduction** | 20% | 0.474 | **0.000** | -47.4 pp | 20.0 / 20 |
| **System Latency** | 15% | 1.0 ms | **4.0 ms** | +3.0 ms (<10s target) | 15.0 / 15 |

## 2. Additional Performance Metrics

| Metric | Standard RAG | Denoised RAG |
|---|---|---|
| Retrieval Recall | 1.0 | 1.0 |
| Unreliable Docs Passed to LLM | 2.875 | **0** |
| Model Confidence Score | 0.086 | **0.766** |
| Evidence Supported Ratio | 0.526 | **1.0** |
| Feedback Loop Trigger Rate | n/a | 0.5 |

## 3. Per-Query Breakdown

| Query | Std Prec | Den Prec | Noise Drop | Std Acc | Den Acc | Std Halluc | Den Halluc | Feedback |
|---|---|---|---|---|---|---|---|---|
| What are the main causes of climate change?... | 0.5 | **0.75** | 3 -> **1** | 0.762 | **0.653** | 0.125 | **0.0** | — |
| How long does it take to fast charge an EV ba... | 0.333 | **1.0** | 4 -> **0** | 0.803 | **0.878** | 0.667 | **0.0** | — |
| Are solid-state batteries used in electric ca... | 0.167 | **1.0** | 5 -> **0** | 0.738 | **0.831** | 0.5 | **0.0** | Triggered |
| How much capacity do EV batteries lose over t... | 0.167 | **1.0** | 5 -> **0** | 0.742 | **0.854** | 0.667 | **0.0** | Triggered |
| Are electric vehicles more likely to catch fi... | 0.167 | **1.0** | 5 -> **0** | 0.778 | **0.916** | 0.5 | **0.0** | Triggered |
| What happens to EV batteries after they are r... | 0.167 | **0.5** | 5 -> **1** | 0.717 | **0.772** | 0.5 | **0.0** | — |
| How does cold winter weather affect electric ... | 0.167 | **0.5** | 5 -> **1** | 0.547 | **0.628** | 0.333 | **0.0** | — |
| What is the average cost of an electric vehic... | 0.167 | **1.0** | 5 -> **0** | 0.36 | **0.459** | 0.5 | **0.0** | Triggered |
