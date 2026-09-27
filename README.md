<div align="center">

# RAG Denoising Pipeline

### Multi-Agent Retrieval-Augmented Generation for Evidence Filtering, Conflict Detection, and Hallucination Reduction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-69%20passed-brightgreen)](#running-tests)
[![Status](https://img.shields.io/badge/status-research%20project-lightgrey)](#)
[![Dashboard](https://img.shields.io/badge/dashboard-live-orange)](#live-dashboard)

Submitted for: **Multi-Agent AI Systems — Project 2: RAG Denoising Pipeline**

</div>

---

## Table of Contents

- [Abstract](#abstract)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Solution Overview](#solution-overview)
- [System Architecture](#system-architecture)
- [Agent Architecture](#agent-architecture)
- [Standard RAG vs Denoised RAG](#standard-rag-vs-denoised-rag)
- [Evidence Gating](#evidence-gating)
- [Hallucination Detection](#hallucination-detection)
- [Expected Output](#expected-output)
- [Bonus Features](#bonus-features)
- [Evaluation Criteria — Assignment Rubric](#evaluation-criteria--assignment-rubric)
- [Benchmark](#benchmark)
- [Benchmark Results](#benchmark-results)
- [Compliance Notes](#compliance-notes)
- [Live Dashboard](#live-dashboard)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Security](#security)
- [Example Scenarios](#example-scenarios)
- [Limitations](#limitations)
- [Key Findings](#key-findings)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Abstract

This project builds a Retrieval-Augmented Generation (RAG) system that improves answer quality by identifying and discarding noisy or irrelevant retrieved documents before generation occurs.

The objective is to demonstrate how intelligent filtering and verification reduces hallucinations and improves the reliability of LLM-generated answers. The system compares a standard RAG pipeline against a denoised multi-agent RAG pipeline on the same benchmark queries.

---

## Problem Statement

Retrieval-Augmented Generation systems retrieve external documents and provide them as context to a language model. However, retrieved documents are not always useful.

A retrieval system may return:

- Relevant documents
- Partially relevant documents
- Irrelevant documents
- Weak or unreliable sources
- Contradictory documents
- Documents that do not actually support the requested claim

If all retrieved documents are passed directly to the generation stage, irrelevant or conflicting information can influence the final answer.

This project addresses this problem by introducing an evidence-denoising layer between retrieval and generation. Instead of trusting every retrieved document, the system evaluates the evidence before allowing it to reach the answer-generation stage.

---

## Objectives

1. Retrieve potentially relevant documents for a query.
2. Score retrieved documents based on semantic relevance.
3. Verify whether retrieved evidence supports relevant claims.
4. Detect contradictory evidence.
5. Remove irrelevant or insufficiently supported documents.
6. Prevent unsupported evidence from reaching answer generation.
7. Generate answers using verified evidence.
8. Detect unsupported or contradicted claims in generated answers.
9. Abstain when sufficient verified evidence is unavailable.
10. Compare Standard RAG against Denoised Multi-Agent RAG using a fixed benchmark.
11. Measure noise reduction, answer evidence coverage, contradiction handling, hallucination-related claims, and latency.

---

## Solution Overview

The project implements a multi-stage evidence-filtering pipeline.

```text
                         User Query
                              |
                              v
                    +-------------------+
                    | Retrieval Agent   |
                    +---------+---------+
                              |
                              v
                 +------------------------+
                 | Relevance Scoring      |
                 | Agent                  |
                 +-----------+------------+
                             |
                             v
                +--------------------------+
                | Evidence Verification    |
                | Agent                    |
                +------------+-------------+
                             |
                             v
              +-----------------------------+
              | Contradiction Detection     |
              | Agent                       |
              +-------------+---------------+
                            |
                            v
                    +---------------+
                    | Evidence Gate |
                    +-------+-------+
                            |
                  +---------+---------+
                  |                   |
                  v                   v
        Sufficient Evidence     Insufficient Evidence
                  |                   |
                  v                   v
        +-------------------+   +-------------+
        | Answer Generation |   |   Abstain   |
        | Agent             |   |             |
        +---------+---------+   +-------------+
                  |
                  v
        +-----------------------+
        | Hallucination         |
        | Detection Agent       |
        +-----------+-----------+
                    |
                    v
              Final Answer
```

---

## System Architecture

Two pipelines operate on the same corpus and benchmark queries, enabling direct comparison.

<table>
<tr>
<th>Standard RAG</th>
<th>Denoised Multi-Agent RAG</th>
</tr>
<tr>
<td valign="top">

```text
Query
  |
  v
Retrieval
  |
  v
Retrieved Documents
  |
  v
Answer Generation
  |
  v
Final Answer
```

</td>
<td valign="top">

```text
Query
  |
  v
Retrieval
  |
  v
Relevance Scoring
  |
  v
Evidence Verification
  |
  v
Contradiction Detection
  |
  v
Evidence Gate
  |
  +----------+----------+
  v                     v
Generate              Abstain
  |
  v
Hallucination Detection
  |
  v
Final Answer
```

</td>
</tr>
</table>

The purpose of the second pipeline is to ensure that generation receives a smaller and better-supported evidence set.

---

## Agent Architecture

The assignment brief specifies five agents. Each entry below states the brief's requirement alongside this implementation's actual behavior.

### Retrieval Agent

**Specified:** Queries one or more vector stores or search APIs to fetch candidate documents for a given user query. Returns a raw document list with metadata.

**Implemented:** Retrieves the most relevant candidate documents using dense semantic embeddings over a local document corpus.

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- Process: embed corpus → embed query → cosine similarity → top-k retrieval

> Note: the implementation retrieves against a single local corpus rather than multiple vector stores or external search APIs. This is a scoped-down version of the specified agent, adequate for a fixed benchmark but not a multi-source retriever.

### Relevance Scoring Agent

**Specified:** Assigns a relevance score (0–1) to each retrieved document using semantic similarity and query-alignment heuristics. Filters out documents below a configurable threshold.

**Implemented:** Matches the specification. Combines semantic similarity and query-term alignment into a score, and filters documents below a configurable relevance threshold before they reach verification or generation.

### Evidence Verification Agent

**Specified:** Cross-checks claimed facts in documents against trusted sources or knowledge bases to flag unreliable evidence.

**Implemented:** Uses Natural Language Inference (NLI) to determine whether retrieved evidence entails, contradicts, or is neutral toward the claims tied to the query, and factors in source reliability.

- **NLI model:** `cross-encoder/nli-deberta-v3-small`
- Relationships: `ENTAILMENT`, `CONTRADICTION`, `NEUTRAL`

> Note: this is an NLI-based verification approach rather than cross-checking against an external trusted-source knowledge base. It answers "does this evidence support this claim?" rather than "does an independent source confirm this fact?" — a related but distinct method of achieving the same goal.

### Contradiction Detection Agent

**Specified:** Detects logical or factual contradictions between retrieved documents. Resolves conflicts by source priority or majority vote.

**Implemented:** Detects contradictions bidirectionally between claims and evidence. When credible sources disagree, the conflict is **preserved and reported** rather than resolved. If evidence is insufficient to settle the disagreement, the conflict remains unresolved.

> Gap against specification: the brief asks the agent to resolve conflicts (by source priority or majority vote); this implementation deliberately does not resolve them. See [Compliance Notes](#compliance-notes).

### Answer Generation Agent

**Specified:** Generates the final answer using only the verified, denoised document set. Attaches citations and a confidence score.

**Implemented:** Generates the final answer using evidence that survives the denoising pipeline (filtered, verified evidence only). Supports a local extractive generation path, with optional use of an external language-model provider. Abstains when evidence is insufficient rather than generating an unsupported answer.

> Gap against specification: per-claim citation generation is not implemented as a first-class output field. An Evidence Confidence Score exists at the pipeline level, but it represents evidence strength, not a per-answer "model-estimated certainty" score as specified. See [Compliance Notes](#compliance-notes).

### Hallucination Detection Agent (beyond the core five)

Evaluates factual claims made by the generated answer against verified evidence. Claims are classified as:

`SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED`

This agent is not one of the five specified in the brief's core architecture but directly implements the "Hallucination detection" bonus feature.

---

## Standard RAG vs Denoised RAG

| | Standard RAG | Denoised Multi-Agent RAG |
|---|---|---|
| Evidence filtering | None — all retrieved documents used | Relevance scoring, verification, and contradiction checks |
| Conflict handling | Not detected | Detected and preserved (not resolved) |
| Unsupported queries | Answered regardless | Abstains |
| Generation input | Raw retrieved documents | Verified, filtered evidence only |
| Transparency | Limited | Claim-level evidence tracing |

Both pipelines use the same benchmark query set and corpus for comparison.

---

## Evidence Gating

The evidence gate determines whether sufficient verified evidence is available for generation.

```text
Sufficient Evidence      -> Answer Generation
Insufficient Evidence    -> ABSTAIN
```

This prevents the system from deliberately generating an answer from evidence that failed the verification process. The denoised pipeline also preserves the original user query during evidence evaluation, so that retrieval expansion does not incorrectly cause irrelevant documents to become generation evidence.

---

## Hallucination Detection

After an answer is generated, factual claims are evaluated against verified evidence.

```text
Generated Claim -> Compare Against Verified Evidence -> SUPPORTED / CONTRADICTED / UNSUPPORTED
```

This provides claim-level visibility into whether generated content is supported by the evidence used by the system. The benchmark evaluator also performs independent claim-level analysis rather than relying only on the pipeline's own hallucination detector.

---

## Expected Output

Per the assignment brief, every query should return five fields. Status against this implementation:

| # | Required field | Status |
|---|---|---|
| 1 | Retrieved Documents — full list from the retrieval step | Met — shown in the dashboard's document inspection panel |
| 2 | Filtered Documents — subset that survived the denoising pipeline | Met — shown as generation evidence |
| 3 | Relevance Scores — per-document numeric scores with reasoning | Partially met — numeric scores are shown; per-document reasoning text is not explicitly surfaced |
| 4 | Final Answer — generated from the clean document set | Met |
| 5 | Confidence Score — model-estimated certainty of the answer | Partially met — an Evidence Confidence Score is shown, but it reflects evidence strength rather than answer-level certainty (see [Limitations](#limitations)) |

---

## Bonus Features

| Feature | Status |
|---|---|
| Citation generation — every factual claim linked to its source document | Not implemented as a first-class output |
| Hallucination detection — post-generation agent flags unsupported statements | Implemented — dedicated agent with claim-level SUPPORTED / CONTRADICTED / UNSUPPORTED classification |
| Agent feedback loops — low-confidence answers trigger a retrieval retry with an expanded query | Not implemented — the pipeline preserves the original query during evaluation, but there is no confidence-triggered retry loop |
| Benchmarking dashboard — side-by-side comparison of standard vs. denoised RAG on precision, recall, and answer quality | Implemented — live dashboard backed by both pipelines, with benchmark results and category-level breakdowns |

---

## Evaluation Criteria — Assignment Rubric

The assignment brief (Table 3) defines the following rubric. The right-hand column states how this project's benchmark addresses each criterion.

| Criterion | Description | Weight | Covered by this project |
|---|---|---:|---|
| Retrieval Precision | Proportion of retrieved docs that are genuinely relevant | 20% | Not reported as a standalone precision metric. The benchmark reports Noise Reduction instead, which is related but measures a different quantity (see [Compliance Notes](#compliance-notes)) |
| Noise Reduction | Measurable drop in irrelevant context passed to the LLM | 20% | Met — 98.98% noise reduction reported |
| Answer Accuracy | Correctness verified against ground-truth answer set | 25% | Approximated via Expected-Fact Coverage (30/30); this measures fact coverage against ground truth, not full answer correctness |
| Hallucination Reduction | Fewer unsupported claims in denoised vs. standard pipeline | 20% | Met — claim-level breakdown of supported / contradicted / unsupported claims for the denoised pipeline |
| System Latency | End-to-end response time remains acceptable (<10s typical) | 15% | Met — mean, median, and P95 warm latency reported, well under the 10s target |

---

## Benchmark

A fixed benchmark of 20 queries is used to compare Standard RAG and Denoised RAG.

| Category | Number of Queries |
|---|---:|
| Normal | 5 |
| Noisy / Irrelevant | 5 |
| Conflicting | 5 |
| Unsupported | 5 |
| **Total** | **20** |

Both pipelines are evaluated using the same benchmark queries and corpus.

### Evaluation Metrics

**1. Noise Reduction** — Measures how effectively the denoising pipeline reduces irrelevant evidence reaching the generation stage.

**2. Expected-Fact Coverage** — Measures whether expected benchmark facts are covered by the generated response, based on benchmark ground truth. This is not the same as universal factual accuracy and should not be interpreted as a guarantee of correctness on unseen questions.

**3. Unsupported-Query Abstention** — Measures whether the system correctly abstains when sufficient verified evidence is unavailable.

**4. Conflict Detection** — Measures whether expected contradictory evidence is detected.

**5. Claim-Level Evaluation** — Generated claims are independently evaluated using evidence relationships: `SUPPORTED`, `UNSUPPORTED`, `CONTRADICTED`, `CONFLICTED`.

**6. Latency** — Measured using mean, median, and P95. Warm query latency is reported separately from model initialization and first-inference warm-up.

---

## Benchmark Results

### Denoised RAG

| Metric | Result |
|---|---:|
| Queries | 20 |
| Noise reduction | 98.98% |
| Expected-fact coverage | 30 / 30 |
| Unsupported-query abstention | 5 / 5 |
| Expected conflicts detected | 5 / 5 |
| Supported claims | 18 / 27 |
| Contradicted claims | 8 / 27 |
| Unsupported claims | 1 / 27 |
| Conflicted claims | 0 / 27 |

### Warm Latency

| Measurement | Standard RAG | Denoised RAG |
|---|---:|---:|
| Mean | 0.1471 s | 0.1128 s |
| Median | 0.1329 s | 0.0660 s |
| P95 | 0.2452 s | 0.2902 s |

Both pipelines are well within the assignment's 10-second latency target.

### Interpretation

The benchmark results are specific to the current corpus, query set, models, and execution environment. The following should not be interpreted as universal guarantees:

- 98.98% noise reduction
- 30/30 expected-fact coverage
- 5/5 unsupported-query abstention
- 5/5 conflict detection

The benchmark is designed to demonstrate the behavior of the denoising pipeline under the selected evaluation scenarios.

---

## Compliance Notes

This section records the known, deliberate deviations from the assignment brief so grading and future work can address them directly.

1. **Contradiction resolution.** The brief specifies that the Contradiction Detection Agent should resolve conflicts by source priority or majority vote. This implementation instead preserves detected conflicts unresolved and reports them to the user. This was a design choice favoring transparency over automatic resolution, but it does not satisfy the letter of the specification.

2. **Citation generation (bonus).** Not implemented. Generated answers are traceable to their evidence set at the pipeline level, but no per-claim citation is attached to the output.

3. **Agent feedback loop (bonus).** Not implemented. The pipeline preserves the original query during evidence evaluation, but there is no mechanism that retries retrieval with an expanded query when answer confidence is low.

4. **Retrieval Precision metric.** The benchmark reports Noise Reduction, which measures the drop in irrelevant context reaching generation. This is related to but distinct from Retrieval Precision, which measures the proportion of retrieved documents that are genuinely relevant at the retrieval step itself. The two are not interchangeable for rubric-scoring purposes.

5. **Confidence Score semantics.** The brief asks for a "model-estimated certainty of the answer." This project's Evidence Confidence Score reflects the strength of the evidence set, not a calibrated probability that the specific generated answer is correct. This distinction is stated explicitly in [Limitations](#limitations).

6. **Evidence Verification method.** The brief describes cross-checking against trusted sources or knowledge bases. This implementation uses NLI-based entailment/contradiction analysis between evidence and claims instead, which verifies internal consistency of the retrieved set rather than external factual grounding.

---

## Live Dashboard

The project includes a live technical dashboard backed by the actual Python RAG pipelines. The dashboard is not a browser-side RAG simulation.

It provides:

- Standard RAG vs Denoised RAG comparison
- Retrieved-document inspection
- Relevance scores
- Evidence verification status
- Generation evidence
- Contradiction information
- Claim-level hallucination analysis
- Abstention status
- Evidence Confidence Score
- Query latency
- Benchmark results, including category-level results

### Dashboard Architecture

```text
Python ThreadingHTTPServer
        |
        +---- Standard RAG Pipeline
        |
        +---- Denoised RAG Pipeline
        |
        +---- Benchmark Results
        |
        +---- HTML/CSS/JavaScript UI
```

### Running the Dashboard

```bash
python dashboard.py
```

Default address:

```text
http://127.0.0.1:8000
```

Open the address in a browser. The dashboard initializes and reuses the RAG pipelines for live query execution.

---

## Technology Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Retrieval | Sentence Transformers, dense embeddings, cosine similarity |
| NLP / NLI | Transformers, Natural Language Inference |
| Models | `sentence-transformers/all-MiniLM-L6-v2`, `cross-encoder/nli-deberta-v3-small` |
| Dashboard | Python `ThreadingHTTPServer`, HTML, CSS, JavaScript |
| Testing | Python `unittest` |
| Configuration | Environment variables, `.env.example` |

---

## Project Structure

```text
RAG_DENOISING_PIPELINE/
│
├── agents/
│   ├── __init__.py
│   ├── retrieval_agent.py
│   ├── relevance_scoring_agent.py
│   ├── evidence_verification_agent.py
│   ├── contradiction_detection_agent.py
│   ├── answer_generation_agent.py
│   └── hallucination_detection_agent.py
│
├── benchmark/
│   └── results/
│       ├── summary.json
│       ├── per_query_results.json
│       └── comparison.md
│
├── data/
│   ├── documents.json
│   └── queries.json
│
├── pipeline/
│   ├── standard_rag.py
│   └── denoised_rag.py
│
├── tests/
│   └── test_*.py
│
├── utils/
│   └── supporting utilities
│
├── benchmark.py
├── dashboard.py
├── dashboard.html
├── orchestrator.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Aparnarajes/RAG_DENOISING_PIPELINE.git
cd RAG_DENOISING_PIPELINE
```

### 2. Create a Virtual Environment

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The first execution may download the required embedding and NLI models.

---

## Usage

### Running the Benchmark

```bash
python benchmark.py
```

Results are stored under `benchmark/results/`:

```text
summary.json
per_query_results.json
comparison.md
```

### Running the Dashboard

```bash
python dashboard.py
```

Open:

```text
http://127.0.0.1:8000
```

### Running Tests

```bash
python -m unittest discover -s tests -v
```

The final validation run completed with:

```text
69 tests passed
```

Test coverage includes: retrieval behavior, relevance scoring, evidence verification, NLI behavior, contradiction detection, evidence gating, abstention, retry behavior, claim-level hallucination detection, benchmark evaluation, dashboard behavior, input validation, error handling, and configuration.

---

## Configuration

The project supports configuration through environment variables. A template is provided in `.env.example`.

```bash
cp .env.example .env
```

Do not commit `.env` or any real API credentials to the repository. Only the example configuration should be included in version control.

---

## Security

The dashboard includes basic defensive controls, including:

- Query validation
- Query length limits
- JSON validation
- Content-type validation
- Request-size limits
- Fixed API routes
- Generic backend error responses

The dashboard runs on loopback by default. It does not provide authentication and should not be exposed directly to an untrusted public network without additional security controls.

---

## Example Scenarios

### Example 1 — Supported Query

```text
Query -> Retrieved Documents -> Relevant Documents -> Verified Evidence
      -> Answer Generation -> Claim Verification -> Final Answer
```

### Example 2 — Unsupported Query

```text
Query -> Retrieved Documents -> Evidence Filtering
      -> Insufficient Verified Evidence -> ABSTAIN
```

The system avoids intentionally passing unsupported evidence to generation.

### Example 3 — Conflicting Query

```text
Query -> Evidence Retrieval -> Claim Comparison
      -> Contradiction Detected -> Conflict Preserved / Reported
```

The system does not silently resolve an unresolved conflict without sufficient supporting evidence.

---

## Limitations

| # | Limitation |
|---|---|
| 1 | **Limited benchmark** — 20 curated queries; useful for demonstrating intended behavior but not representative of every real-world query. |
| 2 | **Limited corpus** — relatively small; may not represent large-scale production retrieval systems. |
| 3 | **NLI limitations** — the NLI model can make incorrect entailment, contradiction, or neutral classifications. |
| 4 | **Claim extraction** — automatic claim extraction is imperfect and may occasionally split or misinterpret generated statements. |
| 5 | **Evidence Confidence Score** — represents evidence strength based on pipeline characteristics; not a probability that the final answer is correct, and not equivalent to the brief's "model-estimated certainty of the answer." |
| 6 | **Hardware dependence** — latency can vary depending on CPU, memory, Python version, model loading, and execution environment. |
| 7 | **Cold start** — model initialization and first-inference warm-up are separate from reported warm query latency. |
| 8 | **Local dashboard** — configured for local execution by default; does not include authentication. |

---

## Reproducibility

The repository includes the components required to reproduce the benchmark:

- Fixed benchmark queries
- Ground-truth information
- Document corpus
- Standard RAG baseline
- Denoised RAG pipeline
- Saved benchmark results
- Model identifiers
- Configuration options
- Test suite
- Installation instructions
- Dashboard

The same query set and corpus are used when comparing the two RAG pipelines.

---

## Key Findings

The current benchmark demonstrates that the denoising pipeline can:

- Remove a large proportion of irrelevant evidence before generation.
- Detect the expected conflicting evidence in the benchmark.
- Abstain on unsupported benchmark queries.
- Provide claim-level evidence analysis.
- Maintain low warm-query latency, well under the assignment's 10-second target.
- Provide a transparent comparison between Standard RAG and Denoised RAG.

These findings are limited to the current benchmark and should not be interpreted as universal performance guarantees. See [Compliance Notes](#compliance-notes) for known gaps against the assignment specification.

---

## Future Improvements

- Implement citation generation so each factual claim links to its source document
- Implement a confidence-triggered agent feedback loop with query expansion
- Add a standalone Retrieval Precision metric distinct from Noise Reduction
- Resolve contradictions by source priority or majority vote, as an option alongside the current preserve-and-report behavior
- Larger and more diverse benchmark datasets
- External knowledge sources for evidence verification
- Hybrid lexical + dense retrieval
- More advanced reranking models
- Better claim extraction
- Larger NLI models
- Human evaluation
- Production-grade authentication and cloud deployment

---

## Conclusion

This project demonstrates a multi-agent approach to improving Retrieval-Augmented Generation by introducing an evidence-denoising stage between retrieval and answer generation. Instead of treating every retrieved document as trustworthy context, the system evaluates evidence through:

```text
Retrieval -> Relevance Scoring -> Evidence Verification -> Contradiction Detection
          -> Evidence Gate -> Generation / Abstention -> Hallucination Detection
```

The implementation satisfies the core intent of the assignment brief and most of its measurable evaluation criteria, with two documented deviations: contradiction conflicts are preserved rather than resolved, and the citation-generation and agent-feedback-loop bonus features are not yet implemented. These are recorded in [Compliance Notes](#compliance-notes) for transparency and future work.

---

## Author

**Aparna Rajesh**
B.Tech, Artificial Intelligence & Machine Learning

GitHub: [github.com/Aparnarajes](https://github.com/Aparnarajes)

Repository: [github.com/Aparnarajes/RAG_DENOISING_PIPELINE](https://github.com/Aparnarajes/RAG_DENOISING_PIPELINE)
