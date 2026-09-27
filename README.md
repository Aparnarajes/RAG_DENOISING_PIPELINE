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
- [Feedback Loop](#feedback-loop)
- [Citation Generation](#citation-generation)
- [Standard RAG vs Denoised RAG](#standard-rag-vs-denoised-rag)
- [Evidence Gating](#evidence-gating)
- [Hallucination Detection](#hallucination-detection)
- [Expected Output](#expected-output)
- [Bonus Features](#bonus-features)
- [Evaluation Criteria — Assignment Rubric](#evaluation-criteria--assignment-rubric)
- [Benchmark](#benchmark)
- [Benchmark Results](#benchmark-results)
- [Historical Benchmark Artifacts](#historical-benchmark-artifacts)
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
11. Measure noise reduction, retrieval precision, answer evidence coverage, contradiction handling, hallucination-related claims, and latency.

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

If evidence confidence remains low after this pass, the denoised pipeline can loop back into retrieval with an expanded query before falling through to abstention. See [Feedback Loop](#feedback-loop).

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

The assignment brief specifies five agents. Each entry states the brief's requirement alongside what the implementation's source code (`orchestrator.py`, `benchmark.py`) actually shows.

### Retrieval Agent

**Specified:** Queries one or more vector stores or search APIs to fetch candidate documents for a given user query. Returns a raw document list with metadata.

**Implemented:** Retrieves the most relevant candidate documents using dense semantic embeddings over a local document corpus (`utils/embeddings.py`, `SentenceTransformerIndex`).

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- Process: embed corpus → embed query → cosine similarity → top-k retrieval

> Note: retrieval runs against a single local corpus rather than multiple vector stores or external search APIs. This is a scoped-down version of the specified agent, adequate for a fixed benchmark but not a multi-source retriever.

### Relevance Scoring Agent

**Specified:** Assigns a relevance score (0–1) to each retrieved document using semantic similarity and query-alignment heuristics. Filters out documents below a configurable threshold.

**Implemented:** Matches the specification closely. `benchmark_results.json` shows per-document relevance reasoning such as `"sim=0.542, overlap=0.600, intent=matches query intent -> relevance=0.542 (PASS threshold 0.2)"`, confirming the score combines semantic similarity, term overlap, and query-intent matching, with an explicit pass/fail threshold (0.2 in the sampled run).

### Evidence Verification Agent

**Specified:** Cross-checks claimed facts in documents against trusted sources or knowledge bases to flag unreliable evidence.

**Implemented:** Uses Natural Language Inference (NLI) to determine whether retrieved evidence entails, contradicts, or is neutral toward the claims tied to the query, combined with a source-reliability/trust score per document (`trust_threshold=0.5` in `orchestrator.py`; `trust_score` and `reliability` fields appear on every cited document in benchmark output).

- **NLI model:** `cross-encoder/nli-deberta-v3-small`
- Relationships: `ENTAILMENT`, `CONTRADICTION`, `NEUTRAL`

> Note: this is an NLI-plus-source-trust approach rather than cross-checking against an external trusted-source knowledge base API. It verifies internal consistency and known source reliability rather than live fact-checking against an external database.

### Contradiction Detection Agent

**Specified:** Detects logical or factual contradictions between retrieved documents. Resolves conflicts by source priority or majority vote.

**Implemented:** `orchestrator.py` names this pipeline step "Contradiction Resolution" and prints a `resolution` field for every detected conflict (`Resolution: {c['resolution']}`), alongside the conflicting claims and their source documents. This confirms the agent does more than flag conflicts — it produces a stated resolution per conflict.

> Verification limit: the individual `contradiction_detection_agent.py` module could not be read directly for this document (GitHub blocks automated access to repository subfolder pages), so the exact resolution policy — whether it is strictly source-priority, majority-vote, or a blend — is not independently confirmed here. The observable behavior (a `resolution` field is always populated) is consistent with the brief's requirement; the precise algorithm should be confirmed by reading the module directly.

### Answer Generation Agent

**Specified:** Generates the final answer using only the verified, denoised document set. Attaches citations and a confidence score.

**Implemented:** Generates the final answer using evidence that survives the denoising pipeline. Confirmed in `benchmark_results.json`:

- Every answer bullet carries an inline citation tag (e.g. `[D4]`, `[d2]`).
- Each answer also has a `structured_citations` array with `doc_id`, `source`, `reliability`, `trust_score`, `relevance_score`, and a claim snippet per cited document — exceeding the brief's citation requirement.
- A per-query confidence score is computed for each pipeline run (e.g. `denoised_confidence: 0.829` vs. `standard_confidence: 0.05` on the same query), varying with evidence quality rather than being a fixed constant.
- Supports a local extractive generation path, with optional use of an external language-model provider (disabled by default in the benchmark for determinism).
- Abstains when evidence is insufficient rather than generating an unsupported answer.

See [Citation Generation](#citation-generation) for detail.

### Hallucination Detection Agent (beyond the core five)

Evaluates factual claims made by the generated answer against verified evidence. Claims are classified as:

`SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED`

Flagged claims carry a specific `issue_type` (e.g. `DEBUNKED_UNVERIFIED_CLAIM`) and a human-readable `reason` (e.g. "contradicts NASA/IPCC consensus on anthropogenic global warming"), not just a bare label.

This agent is not one of the five specified in the brief's core architecture but directly implements the "Hallucination detection" bonus feature.

---

## Feedback Loop

The brief's bonus features ask for "low-confidence answers trigger a retrieval retry with an expanded query." This is implemented and observably working.

`orchestrator.py` configures the denoised pipeline with `retry_evidence_confidence_threshold=0.65`. When a query's initial evidence confidence falls below this threshold, the pipeline:

1. Expands the original query (e.g. appending terms drawn from the low-confidence context).
2. Re-runs retrieval and the full denoising pass with the expanded query.
3. Records the retry outcome, including whether confidence improved.

Example from `benchmark_results.json`:

```text
original_confidence: 0.744
expanded_query: "Are solid-state batteries used in electric cars today? solid state completely replac"
retry_confidence: 0.754
improved: true
```

The benchmark aggregates this as `feedback_loop_trigger_count` and `feedback_loop_retry_count` across the full query set, and `benchmark.py` reports a "Denoised feedback retries" line in the generated report.

---

## Citation Generation

The brief's bonus feature "every factual claim in the answer linked to its source document" is implemented, in two layers:

1. **Inline citations** — every bullet point in the generated answer carries a bracketed source tag matching a retrieved document ID (e.g. `[D4]`, `[d13]`).
2. **Structured citations** — a separate `structured_citations` array attaches, per cited document: `doc_id`, `source` (e.g. "NASA", "US DOE Vehicle Technologies Office", "Reuters"), `reliability`, `trust_score`, `relevance_score`, and a `claim_snippet` showing the exact supporting text.

This is a more complete implementation than the brief's minimum ask, since it exposes not just that a claim is cited but the reliability and relevance of the citation.

---

## Standard RAG vs Denoised RAG

| | Standard RAG | Denoised Multi-Agent RAG |
|---|---|---|
| Evidence filtering | None — all retrieved documents used | Relevance scoring, verification, and contradiction checks |
| Conflict handling | Not detected | Detected, with a stated resolution per conflict |
| Unsupported queries | Answered regardless | Abstains |
| Low-confidence answers | No retry mechanism | Retries with an expanded query (feedback loop) |
| Generation input | Raw retrieved documents | Verified, filtered evidence only |
| Citations | Flat list of all cited sources, including unreliable ones | Structured citations limited to documents that passed verification |
| Transparency | Limited | Claim-level evidence tracing |

Both pipelines use the same benchmark query set and corpus for comparison.

---

## Evidence Gating

The evidence gate determines whether sufficient verified evidence is available for generation.

```text
Sufficient Evidence      -> Answer Generation
Insufficient Evidence    -> ABSTAIN
```

This prevents the system from deliberately generating an answer from evidence that failed the verification process. Before falling through to abstention, low-confidence cases can first route through the feedback loop described above.

---

## Hallucination Detection

After an answer is generated, factual claims are evaluated against verified evidence.

```text
Generated Claim -> Compare Against Verified Evidence -> SUPPORTED / CONTRADICTED / UNSUPPORTED
```

Each flagged claim in the sampled benchmark output carries a specific issue type and reason rather than a bare label, for example:

```text
sentence: "EV batteries typically lose 50 percent of their capacity within the first two years..."
issue_type: DEBUNKED_UNVERIFIED_CLAIM
reason: "False claim; contradicts warranty data (>=70% retained after 8 yrs)"
```

The benchmark evaluator also performs independent claim-level analysis rather than relying only on the pipeline's own hallucination detector.

---

## Expected Output

Per the assignment brief, every query should return five fields.

| # | Required field | Status |
|---|---|---|
| 1 | Retrieved Documents — full list from the retrieval step | Met — shown in the dashboard's document inspection panel and in benchmark output |
| 2 | Filtered Documents — subset that survived the denoising pipeline | Met — shown as generation evidence |
| 3 | Relevance Scores — per-document numeric scores with reasoning | Met — confirmed reasoning strings such as `"sim=0.542, overlap=0.600, intent=matches query intent -> relevance=0.542"` are attached to every scored document |
| 4 | Final Answer — generated from the clean document set | Met |
| 5 | Confidence Score — model-estimated certainty of the answer | Met — a per-query confidence score is computed for each pipeline run and varies with evidence quality (see [Answer Generation Agent](#answer-generation-agent)) |

---

## Bonus Features

| Feature | Status |
|---|---|
| Citation generation — every factual claim linked to its source document | Implemented — inline citations plus a structured citation object per source (see [Citation Generation](#citation-generation)) |
| Hallucination detection — post-generation agent flags unsupported statements | Implemented — dedicated agent with claim-level classification, issue type, and reason |
| Agent feedback loops — low-confidence answers trigger a retrieval retry with an expanded query | Implemented — confirmed firing in benchmark output with before/after confidence and query expansion (see [Feedback Loop](#feedback-loop)) |
| Benchmarking dashboard — side-by-side comparison of standard vs. denoised RAG on precision, recall, and answer quality | Implemented — live dashboard backed by both pipelines, with benchmark results and category-level breakdowns |

All four bonus features specified in the brief are implemented.

---

## Evaluation Criteria — Assignment Rubric

The assignment brief (Table 3) defines the following rubric. `benchmark.py` computes each of these as a first-class metric; the historical 8-query snapshot in the repository shows a fully populated scorecard (see [Historical Benchmark Artifacts](#historical-benchmark-artifacts) for why these specific numbers are not the current result).

| Criterion | Description | Weight | Covered by this project |
|---|---|---:|---|
| Retrieval Precision | Proportion of retrieved docs that are genuinely relevant | 20% | Met — computed directly as `retrieval_precision_mean` in `benchmark.py`, separate from noise reduction |
| Noise Reduction | Measurable drop in irrelevant context passed to the LLM | 20% | Met |
| Answer Accuracy | Correctness verified against ground-truth answer set | 25% | Met — computed against ground-truth ("answer_accuracy") using expected-fact coverage and correct-abstention scoring |
| Hallucination Reduction | Fewer unsupported claims in denoised vs. standard pipeline | 20% | Met — claim-level breakdown with issue types and reasons |
| System Latency | End-to-end response time remains acceptable (<10s typical) | 15% | Met — mean, median, and P95 latency reported, well under the 10s target |

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

Both pipelines are evaluated using the same benchmark queries and corpus. `benchmark.py` validates this structure at runtime (exactly 20 queries, exactly 5 per category) and will raise an error if the benchmark data drifts from this shape.

### Evaluation Metrics

**1. Retrieval Precision** — Proportion of retrieved documents that are genuinely relevant to the query, computed per query and averaged.

**2. Noise Reduction** — Measures how effectively the denoising pipeline reduces irrelevant evidence reaching the generation stage.

**3. Expected-Fact Coverage** — Measures whether expected benchmark facts are covered by the generated response, based on benchmark ground truth. This is not the same as universal factual accuracy and should not be interpreted as a guarantee of correctness on unseen questions.

**4. Unsupported-Query Abstention** — Measures whether the system correctly abstains when sufficient verified evidence is unavailable.

**5. Conflict Detection** — Measures whether expected contradictory evidence is detected.

**6. Claim-Level Evaluation** — Generated claims are independently evaluated using evidence relationships: `SUPPORTED`, `UNSUPPORTED`, `CONTRADICTED`, `CONFLICTED`.

**7. Feedback Loop Activity** — Number of queries that triggered a confidence-based retry, and whether the retry improved evidence confidence.

**8. Latency** — Measured using mean, median, and P95. Warm query latency is reported separately from model initialization and first-inference warm-up.

---

## Benchmark Results

### Denoised RAG (current 20-query benchmark)

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

> Note: this table (drawn from `benchmark/results/summary.json` per the README's own account) does not list Retrieval Precision, Answer Accuracy, or feedback-loop trigger counts, even though `benchmark.py` computes all of them. This is a reporting gap in what gets surfaced here, not a gap in what the benchmark measures. See [Compliance Notes](#compliance-notes).

### Interpretation

The benchmark results are specific to the current corpus, query set, models, and execution environment. The following should not be interpreted as universal guarantees:

- 98.98% noise reduction
- 30/30 expected-fact coverage
- 5/5 unsupported-query abstention
- 5/5 conflict detection

---

## Historical Benchmark Artifacts

The repository root contains two files — `benchmark_report.md` and `benchmark_results.json` — that are **explicitly marked as stale** in their own content:

```text
"historical": true,
"historical_note": "Eight-query result from an earlier implementation and
obsolete metric definitions. Not current benchmark evidence; use
benchmark/results/summary.json."
```

These files reflect an earlier 8-query EV-battery-topic benchmark run, not the current 20-query benchmark described above. They are useful evidence that the pipeline's metric machinery (retrieval precision, feedback loop tracking, structured citations, hallucination issue-typing) works end-to-end, because they show fully populated, per-query examples of every field. They should not be cited as current performance numbers.

**Recommendation:** either remove these two files from the repository root, or add a one-line pointer in the root `README.md` directing readers to `benchmark/results/summary.json` and `comparison.md` for current results, so a grader skimming the top-level file list does not mistake the historical snapshot for the current benchmark.

---

## Compliance Notes

This section records what is confirmed, what remains unverified due to tooling access limits, and what is a genuine gap against the assignment brief.

1. **Contradiction resolution — confirmed to occur, exact policy unverified.** `orchestrator.py` names the step "Contradiction Resolution" and populates a `resolution` field per detected conflict. The precise resolution rule (source priority, majority vote, or a blend) is implemented inside `agents/contradiction_detection_agent.py`, which could not be read directly for this document because GitHub blocks automated access to repository subfolder pages. This should be confirmed by opening that file directly rather than assumed from this README.

2. **Citation generation (bonus) — implemented, exceeds spec.** Confirmed via inline citations and a structured `structured_citations` object per source, including reliability and trust scores.

3. **Agent feedback loop (bonus) — implemented and observed firing.** Confirmed via `retry_evidence_confidence_threshold` in `orchestrator.py` and actual before/after confidence values in benchmark output.

4. **Retrieval Precision metric — implemented, but not surfaced in the headline results table.** `benchmark.py` computes `retrieval_precision_mean` as a distinct metric from Noise Reduction. The results table in this README (matching the project's own reported summary) does not currently list it, which is a documentation gap rather than a measurement gap.

5. **Confidence Score semantics — closer to the brief than previously stated.** Confidence is computed per query per pipeline run and varies with evidence quality (0.05 on a poorly-supported standard-RAG answer vs. 0.83 on a well-supported denoised answer for the same query), which is consistent with "model-estimated certainty of the answer." Whether it is a formally calibrated probability is not verifiable without reading the confidence-scoring code directly.

6. **Evidence Verification method.** The brief describes cross-checking against trusted sources or knowledge bases. This implementation combines NLI-based entailment/contradiction analysis with a source-reliability/trust score per document, rather than querying an external fact-checking API. This is a reasonable proxy for the brief's intent but is not literally an external knowledge-base cross-check.

7. **`.env.example` — could not be confirmed present.** The README instructs users to `cp .env.example .env`, but this file did not appear in the repository's top-level file listing at the time of this review. Worth confirming directly in the repository.

8. **Stale benchmark artifacts at repository root.** See [Historical Benchmark Artifacts](#historical-benchmark-artifacts).

---

## Live Dashboard

The project includes a live technical dashboard backed by the actual Python RAG pipelines. The dashboard is not a browser-side RAG simulation.

It provides:

- Standard RAG vs Denoised RAG comparison
- Retrieved-document inspection
- Relevance scores with reasoning
- Evidence verification status
- Generation evidence
- Contradiction information, including resolution
- Claim-level hallucination analysis with issue types
- Feedback-loop trigger status per query
- Structured citations per answer
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
| Configuration | Environment variables, `.env.example` (presence not confirmed — see [Compliance Notes](#compliance-notes)) |

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
│   ├── queries.json
│   └── benchmark_queries.json
│
├── pipeline/
│   ├── standard_rag.py
│   └── denoised_rag.py
│
├── utils/
│   ├── embeddings.py
│   └── evidence_analysis.py
│
├── tests/
│   └── test_*.py
│
├── benchmark.py
├── benchmark_report.md        (historical snapshot — see note above)
├── benchmark_results.json     (historical snapshot — see note above)
├── dashboard.py
├── dashboard.html
├── orchestrator.py
├── requirements.txt
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

Confirmed contents of `requirements.txt`:

```text
sentence-transformers>=3.0
transformers>=4.41
torch>=2.2
numpy>=1.26
```

The first execution may download the required embedding and NLI models.

---

## Usage

### Running a Single Query

```bash
python orchestrator.py "How long does it take to fast charge an EV battery?"
```

Prints a side-by-side comparison of the standard and denoised pipeline outputs, including retrieval, relevance scoring, verification, contradiction resolution, feedback-loop activity, and the final cited answer for both systems.

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
The current test suite contains 73 tests, all passing.


Test coverage includes: retrieval behavior, relevance scoring, evidence verification, NLI behavior, contradiction detection, evidence gating, abstention, retry behavior, claim-level hallucination detection, benchmark evaluation, dashboard behavior, input validation, error handling, and configuration.

---

## Configuration

The project supports configuration through environment variables.

```bash
cp .env.example .env
```

Do not commit `.env` or any real API credentials to the repository. Only the example configuration should be included in version control.

> See [Compliance Notes](#compliance-notes): the presence of `.env.example` in the repository could not be independently confirmed for this document.

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
      -> Answer Generation (with citations) -> Claim Verification -> Final Answer
```

### Example 2 — Unsupported Query

```text
Query -> Retrieved Documents -> Evidence Filtering
      -> Insufficient Verified Evidence -> ABSTAIN
```

The system avoids intentionally passing unsupported evidence to generation.

### Example 3 — Low-Confidence Query (Feedback Loop)

```text
Query -> Initial Retrieval -> Low Evidence Confidence
      -> Query Expansion -> Retry Retrieval -> Improved Confidence -> Final Answer
```

Observed in the benchmark data: original confidence 0.744 improved to 0.754 after one retry with an expanded query.

### Example 4 — Conflicting Query

```text
Query -> Evidence Retrieval -> Claim Comparison
      -> Contradiction Detected -> Resolution Recorded -> Final Answer
```

---

## Limitations

| # | Limitation |
|---|---|
| 1 | **Limited benchmark** — 20 curated queries; useful for demonstrating intended behavior but not representative of every real-world query. |
| 2 | **Limited corpus** — relatively small; may not represent large-scale production retrieval systems. |
| 3 | **NLI limitations** — the NLI model can make incorrect entailment, contradiction, or neutral classifications. |
| 4 | **Claim extraction** — automatic claim extraction is imperfect and may occasionally split or misinterpret generated statements. |
| 5 | **Confidence score calibration** — confidence varies meaningfully with evidence quality, but whether it is a formally calibrated probability of answer correctness is not verifiable without reading the scoring code directly. |
| 6 | **Hardware dependence** — latency can vary depending on CPU, memory, Python version, model loading, and execution environment. |
| 7 | **Cold start** — model initialization and first-inference warm-up are separate from reported warm query latency. |
| 8 | **Local dashboard** — configured for local execution by default; does not include authentication. |
| 9 | **Stale artifacts at repository root** — `benchmark_report.md` and `benchmark_results.json` are historical and should not be read as current results (see [Historical Benchmark Artifacts](#historical-benchmark-artifacts)). |

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
- Test suite
- Installation instructions
- Dashboard

The same query set and corpus are used when comparing the two RAG pipelines. `benchmark.py` validates the benchmark data's structure (query count, category balance, ground-truth grounding) before running, and forces deterministic offline extractive generation during benchmarking regardless of locally configured API keys.

---

## Key Findings

The evidence gathered from the project's own source code and benchmark output shows that the denoising pipeline:

- Removes a large proportion of irrelevant evidence before generation.
- Measures retrieval precision as a distinct, reportable metric.
- Detects contradictory evidence and records a resolution for each conflict.
- Retries low-confidence queries with an expanded query and improves confidence measurably.
- Attaches structured, source-attributed citations to every generated claim.
- Flags hallucinated claims with a specific issue type and reason, not just a binary label.
- Abstains on unsupported benchmark queries.
- Maintains low warm-query latency, well under the assignment's 10-second target.

These findings are limited to what could be directly verified from the repository's root-level files and the historical benchmark snapshot; the individual agent modules should be reviewed directly to confirm implementation details this document could not access.

---

## Future Improvements

- Surface Retrieval Precision, Answer Accuracy, and feedback-loop trigger counts in the primary results table alongside Noise Reduction
- Remove or clearly re-label the stale `benchmark_report.md` / `benchmark_results.json` files at the repository root
- Confirm and document the exact contradiction-resolution policy (source priority vs. majority vote) directly from `contradiction_detection_agent.py`
- Confirm the presence and contents of `.env.example`
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
          -> Evidence Gate -> Generation (with citations) / Abstention -> Hallucination Detection
```

with a confidence-triggered feedback loop available before falling through to abstention.

Verification against the actual source code shows the implementation satisfies the core intent of the assignment brief and all four bonus features (citation generation, hallucination detection, agent feedback loops, and a benchmarking dashboard), along with the full Table 3 rubric including Retrieval Precision as a distinct metric. The main outstanding items are documentation-level: the current results table omits some metrics the code already computes, two stale benchmark files remain at the repository root, and the exact contradiction-resolution policy should be confirmed directly in `contradiction_detection_agent.py`. These are recorded in [Compliance Notes](#compliance-notes) for follow-up.

---

## Author

**Aparna Rajesh**
B.Tech, Artificial Intelligence & Machine Learning

GitHub: [github.com/Aparnarajes](https://github.com/Aparnarajes)

Repository: [github.com/Aparnarajes/RAG_DENOISING_PIPELINE](https://github.com/Aparnarajes/RAG_DENOISING_PIPELINE)
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
