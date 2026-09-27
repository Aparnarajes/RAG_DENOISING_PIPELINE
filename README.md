# RAG Denoising Pipeline

### Multi-Agent Retrieval-Augmented Generation for Evidence Filtering, Conflict Detection, and Hallucination Reduction

A multi-agent Retrieval-Augmented Generation (RAG) system designed to reduce the impact of irrelevant, unreliable, and conflicting retrieved documents before answer generation.

This project compares two approaches:

- **Standard RAG** — retrieves documents and directly generates an answer.
- **Denoised Multi-Agent RAG** — evaluates retrieved evidence through multiple specialized agents before allowing evidence to reach the generation stage.

The objective is to improve evidence quality, reduce noisy context, detect contradictions, prevent unsupported generation, and provide transparent evaluation.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Solution Overview](#solution-overview)
- [System Architecture](#system-architecture)
- [Multi-Agent Components](#multi-agent-components)
- [Standard RAG vs Denoised RAG](#standard-rag-vs-denoised-rag)
- [Evidence Gating](#evidence-gating)
- [Hallucination Detection](#hallucination-detection)
- [Benchmark](#benchmark)
- [Benchmark Results](#benchmark-results)
- [Live Dashboard](#live-dashboard)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Benchmark](#running-the-benchmark)
- [Running the Dashboard](#running-the-dashboard)
- [Running Tests](#running-tests)
- [Configuration](#configuration)
- [Example Scenarios](#example-scenarios)
- [Limitations](#limitations)
- [Reproducibility](#reproducibility)
- [Demo Video](#demo-video)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

# Problem Statement

Retrieval-Augmented Generation systems retrieve external documents and provide them as context to a language model.

However, retrieved documents are not always useful.

A retrieval system may return:

- Relevant documents
- Partially relevant documents
- Irrelevant documents
- Weak or unreliable sources
- Contradictory documents
- Documents that do not actually support the requested claim

If all retrieved documents are passed directly to the generation stage, irrelevant or conflicting information can influence the final answer.

This project addresses this problem by introducing an **evidence-denoising layer** between retrieval and generation.

Instead of trusting every retrieved document, the system evaluates the evidence before allowing it to reach the answer-generation stage.

---

# Objectives

The main objectives of this project are:

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

# Solution Overview

The project implements a multi-stage evidence filtering pipeline.

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
                | Evidence Verification   |
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

# System Architecture

The system contains two pipelines that operate on the same corpus and benchmark queries.

## Standard RAG

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

## Denoised Multi-Agent RAG

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
  +----------------------+
  |                      |
  v                      v
Generate               Abstain
  |
  v
Hallucination Detection
  |
  v
Final Answer
```

The purpose of the second pipeline is to ensure that generation receives a smaller and better-supported evidence set.

---

# Multi-Agent Components

## 1. Retrieval Agent

The Retrieval Agent retrieves the most relevant candidate documents for the user query.

The project uses dense semantic embeddings instead of keyword-only retrieval.

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

The system:

1. Creates embeddings for the document corpus.
2. Creates an embedding for the user query.
3. Calculates cosine similarity.
4. Retrieves the top-k candidate documents.

---

## 2. Relevance Scoring Agent

The Relevance Scoring Agent evaluates how relevant each retrieved document is to the original query.

The relevance score combines:

- Semantic similarity
- Query-term alignment

Documents below the configured relevance threshold can be filtered before evidence verification and generation.

The relevance threshold is configurable through the project configuration.

---

## 3. Evidence Verification Agent

The Evidence Verification Agent evaluates whether retrieved evidence supports claims related to the query.

The project uses Natural Language Inference (NLI) to compare claims and evidence.

The NLI relationship can be:

```text
ENTAILMENT
CONTRADICTION
NEUTRAL
```

### NLI Model

```text
cross-encoder/nli-deberta-v3-small
```

The NLI model is separate from the embedding model.

The embedding model is used for retrieval and semantic similarity, while the NLI model is used for evidence relationship analysis.

Source reliability is also considered during evidence verification.

---

## 4. Contradiction Detection Agent

The Contradiction Detection Agent compares claims and evidence to identify conflicting information.

It evaluates evidence in both directions and identifies relationships such as:

```text
ENTAILMENT
CONTRADICTION
NEUTRAL
```

When credible sources disagree, the system preserves the conflict rather than silently selecting one unsupported answer.

If the available evidence is insufficient to resolve the disagreement, the conflict can remain unresolved.

---

## 5. Answer Generation Agent

The Answer Generation Agent generates the final answer using evidence that survives the denoising pipeline.

The generation stage receives filtered and verified evidence rather than the complete retrieved document set.

When sufficient evidence is unavailable, the system can abstain instead of intentionally generating an unsupported answer.

The project supports a local extractive generation path and can optionally use an external language-model provider when configured.

---

## 6. Hallucination Detection Agent

The Hallucination Detection Agent evaluates factual claims made by the generated answer.

Claims are compared against verified evidence.

Claims can be classified as:

```text
SUPPORTED
CONTRADICTED
UNSUPPORTED
```

The dashboard displays claim-level results and associated evidence information.

---

# Standard RAG vs Denoised RAG

## Standard RAG

Standard RAG follows the conventional flow:

```text
User Query
    |
    v
Retrieve Documents
    |
    v
Generate Answer
```

Retrieved documents are passed directly to generation.

---

## Denoised Multi-Agent RAG

The denoised pipeline introduces multiple evidence-quality checks:

```text
User Query
    |
    v
Retrieve Documents
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
    +--------------------+
    |                    |
    v                    v
Generate              Abstain
    |
    v
Hallucination Detection
    |
    v
Final Answer
```

Both pipelines use the same benchmark query set and corpus for comparison.

---

# Evidence Gating

The evidence gate is one of the main components of the denoising system.

The gate determines whether sufficient verified evidence is available for generation.

### If sufficient evidence exists

```text
Verified Evidence
       |
       v
Answer Generation
```

### If sufficient evidence does not exist

```text
Insufficient Verified Evidence
       |
       v
ABSTAIN
```

This prevents the system from deliberately generating an answer from evidence that failed the verification process.

The denoised pipeline also preserves the original user query during evidence evaluation so that retrieval expansion does not incorrectly cause irrelevant documents to become generation evidence.

---

# Hallucination Detection

After an answer is generated, factual claims are evaluated against verified evidence.

Example:

```text
Generated Claim
      |
      v
Compare Against Verified Evidence
      |
      +-----------------------------+
      |              |              |
      v              v              v
  SUPPORTED     CONTRADICTED    UNSUPPORTED
```

This provides claim-level visibility into whether generated content is supported by the evidence used by the system.

The benchmark evaluator also performs independent claim-level analysis rather than relying only on the pipeline's own hallucination detector.

---

# Benchmark

A fixed benchmark of **20 queries** is used to compare Standard RAG and Denoised RAG.

The benchmark contains four categories:

| Category | Number of Queries |
|---|---:|
| Normal | 5 |
| Noisy / Irrelevant | 5 |
| Conflicting | 5 |
| Unsupported | 5 |
| **Total** | **20** |

Both pipelines are evaluated using the same benchmark queries and corpus.

---

# Evaluation Metrics

The benchmark evaluates multiple aspects of the system.

## 1. Noise Reduction

Measures how effectively the denoising pipeline reduces irrelevant evidence reaching the generation stage.

---

## 2. Expected-Fact Coverage

Measures whether expected benchmark facts are covered by the generated response.

This metric is based on the benchmark ground truth.

> Expected-fact coverage is not the same as universal factual accuracy and should not be interpreted as a guarantee of correctness on unseen questions.

---

## 3. Unsupported-Query Abstention

Measures whether the system correctly abstains when sufficient verified evidence is unavailable.

---

## 4. Conflict Detection

Measures whether expected contradictory evidence is detected.

---

## 5. Claim-Level Evaluation

Generated claims are independently evaluated using evidence relationships.

Possible categories include:

```text
SUPPORTED
UNSUPPORTED
CONTRADICTED
CONFLICTED
```

---

## 6. Latency

Query latency is measured using:

- Mean
- Median
- P95

Warm query latency is reported separately from model initialization and first-inference warm-up.

---

# Benchmark Results

The current benchmark contains 20 queries across four categories.

## Denoised RAG Results

| Metric | Result |
|---|---:|
| Queries | 20 |
| Noise reduction | **98.98%** |
| Expected-fact coverage | **30/30** |
| Unsupported-query abstention | **5/5** |
| Expected conflicts detected | **5/5** |
| Supported claims | **18/27** |
| Unsupported claims | **1/27** |
| Contradicted claims | **8/27** |
| Conflicted claims | **0/27** |

### Warm Latency

| Measurement | Standard RAG | Denoised RAG |
|---|---:|---:|
| Mean | 0.1471 s | **0.1128 s** |
| Median | 0.1329 s | **0.0660 s** |
| P95 | 0.2452 s | **0.2902 s** |

### Important Interpretation

The benchmark results are specific to the current corpus, query set, models, and execution environment.

The following should **not** be interpreted as universal guarantees:

- 98.98% noise reduction
- 30/30 expected-fact coverage
- 5/5 unsupported-query abstention
- 5/5 conflict detection

The benchmark is designed to demonstrate the behavior of the denoising pipeline under the selected evaluation scenarios.

---

# Live Dashboard

The project includes a live technical dashboard backed by the actual Python RAG pipelines.

The dashboard is **not a browser-side RAG simulation**.

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
- Benchmark results
- Category-level benchmark results

## Dashboard Architecture

The dashboard uses:

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

---

# Running the Dashboard

Start the dashboard:

```bash
python dashboard.py
```

The default address is:

```text
http://127.0.0.1:8000
```

Open the address in a browser.

The dashboard initializes and reuses the RAG pipelines for live query execution.

---

# Example Scenarios

## Example 1 — Supported Query

When sufficient verified evidence is available:

```text
Query
  |
  v
Retrieved Documents
  |
  v
Relevant Documents
  |
  v
Verified Evidence
  |
  v
Answer Generation
  |
  v
Claim Verification
  |
  v
Final Answer
```

---

## Example 2 — Unsupported Query

When sufficient verified evidence is unavailable:

```text
Query
  |
  v
Retrieved Documents
  |
  v
Evidence Filtering
  |
  v
Insufficient Verified Evidence
  |
  v
ABSTAIN
```

The system avoids intentionally passing unsupported evidence to generation.

---

## Example 3 — Conflicting Query

When retrieved sources disagree:

```text
Query
  |
  v
Evidence Retrieval
  |
  v
Claim Comparison
  |
  v
Contradiction Detected
  |
  v
Conflict Preserved / Reported
```

The system does not silently resolve an unresolved conflict without sufficient supporting evidence.

---

# Technology Stack

## Programming Language

- Python

## Retrieval

- Sentence Transformers
- Dense embeddings
- Cosine similarity

## NLP / NLI

- Transformers
- Sentence Transformers
- Natural Language Inference

## Models

```text
sentence-transformers/all-MiniLM-L6-v2
cross-encoder/nli-deberta-v3-small
```

## Dashboard

- Python `ThreadingHTTPServer`
- HTML
- CSS
- JavaScript

## Testing

- Python `unittest`

## Configuration

- Environment variables
- `.env.example`

---

# Project Structure

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

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Aparnarajes/RAG_DENOISING_PIPELINE.git
cd RAG_DENOISING_PIPELINE
```

## 2. Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The first execution may download the required embedding and NLI models.

---

# Running the Benchmark

Run:

```bash
python benchmark.py
```

The benchmark results are stored under:

```text
benchmark/results/
```

The saved benchmark includes:

```text
summary.json
per_query_results.json
comparison.md
```

---

# Running the Dashboard

Start the server:

```bash
python dashboard.py
```

Open:

```text
http://127.0.0.1:8000
```

---

# Running Tests

Run the complete test suite:

```bash
python -m unittest discover -s tests -v
```

The final validation run completed with:

```text
69 tests passed
```

The test suite covers:

- Retrieval behavior
- Relevance scoring
- Evidence verification
- NLI behavior
- Contradiction detection
- Evidence gating
- Abstention
- Retry behavior
- Claim-level hallucination detection
- Benchmark evaluation
- Dashboard behavior
- Input validation
- Error handling
- Configuration

---

# Configuration

The project supports configuration through environment variables.

A template is provided:

```text
.env.example
```

Create a local `.env` file when configuration values are required.

### Important

Do not commit:

```text
.env
```

or any real API credentials to GitHub.

Only the example configuration should be included in the repository.

---

# Security

The dashboard includes basic defensive controls including:

- Query validation
- Query length limits
- JSON validation
- Content-type validation
- Request-size limits
- Fixed API routes
- Generic backend error responses

The dashboard runs on loopback by default.

It does not provide authentication and should not be exposed directly to an untrusted public network without additional security controls.

---

# Limitations

This project has several limitations.

### 1. Limited Benchmark

The benchmark contains 20 curated queries.

It is useful for demonstrating the intended behavior of the pipeline but does not represent every possible real-world query.

### 2. Limited Corpus

The evaluation corpus is relatively small and may not represent large-scale production retrieval systems.

### 3. NLI Limitations

The NLI model can make incorrect entailment, contradiction, or neutral classifications.

### 4. Claim Extraction

Automatic claim extraction is imperfect and may occasionally split or interpret generated statements incorrectly.

### 5. Evidence Confidence

The Evidence Confidence Score represents evidence strength based on the pipeline's evidence characteristics.

It is not a probability that the final answer is correct.

### 6. Hardware Dependence

Latency can vary depending on CPU, memory, Python version, model loading, and execution environment.

### 7. Cold Start

Model initialization and first-inference warm-up are separate from the reported warm query latency.

### 8. Local Dashboard

The dashboard is configured for local execution by default and does not include authentication.

---

# Reproducibility

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

# Research / Evaluation Methodology

The evaluation follows the following process:

```text
                Fixed Query
                    |
                    v
          +-------------------+
          | Standard RAG      |
          +-------------------+
                    |
                    v
             Standard Answer
                    |
                    |
                    |
          +-------------------+
          | Denoised RAG      |
          +-------------------+
                    |
                    v
        Evidence Filtering
                    |
                    v
          Verified Evidence
                    |
                    v
           Denoised Answer
                    |
                    v
             Evaluation
```

Both systems are evaluated against the same benchmark conditions.

The evaluation focuses on whether the denoising layer:

- Removes irrelevant evidence
- Preserves useful evidence
- Detects conflicts
- Abstains when evidence is insufficient
- Reduces unsupported generated claims
- Maintains practical query latency

---

# Key Findings

The current benchmark demonstrates that the denoising pipeline can:

- Remove a large proportion of irrelevant evidence before generation.
- Detect the expected conflicting evidence in the benchmark.
- Abstain on unsupported benchmark queries.
- Provide claim-level evidence analysis.
- Maintain low warm-query latency on the evaluation environment.
- Provide a transparent comparison between Standard RAG and Denoised RAG.

These findings are limited to the current benchmark and should not be interpreted as universal performance guarantees.

---

# Demo Video

The demonstration covers:

1. Project overview
2. Standard RAG architecture
3. Denoised Multi-Agent RAG architecture
4. Retrieval
5. Relevance scoring
6. Evidence verification
7. Contradiction detection
8. Evidence filtering
9. Answer generation
10. Unsupported-query abstention
11. Hallucination detection
12. Benchmark comparison
13. Live dashboard

### Demo Link

**Add your final demo video link here:**

```text
PASTE-YOUR-DEMO-VIDEO-LINK-HERE
```

---

# Future Improvements

Potential future improvements include:

- Larger and more diverse benchmark datasets
- External knowledge sources
- Hybrid lexical + dense retrieval
- More advanced reranking models
- Better claim extraction
- Larger NLI models
- Human evaluation
- More comprehensive hallucination benchmarks
- Distributed inference
- Production-grade authentication
- Cloud deployment
- Continuous evaluation and feedback loops

---

# Conclusion

This project demonstrates a multi-agent approach to improving Retrieval-Augmented Generation by introducing an evidence-denoising stage between retrieval and answer generation.

Instead of treating every retrieved document as trustworthy context, the system evaluates evidence through:

```text
Retrieval
   ↓
Relevance Scoring
   ↓
Evidence Verification
   ↓
Contradiction Detection
   ↓
Evidence Gate
   ↓
Generation / Abstention
   ↓
Hallucination Detection
```

The primary goal is not simply to generate an answer, but to ensure that the answer-generation stage receives evidence that has passed relevance and verification checks.

The project provides both a Standard RAG baseline and a Denoised Multi-Agent RAG pipeline, together with a fixed benchmark, claim-level evaluation, automated tests, and a live technical dashboard.

---

# Author

**Aparna Rajesh**

B.Tech Artificial Intelligence & Machine Learning

GitHub:

https://github.com/Aparnarajes

---

## Project Repository

https://github.com/Aparnarajes/RAG_DENOISING_PIPELINE
