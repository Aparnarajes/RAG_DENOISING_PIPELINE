# RAG Denoising Pipeline

## 1. Problem statement

Retrieval-Augmented Generation can pass irrelevant, unreliable, or mutually
conflicting documents into answer generation. This project evaluates a
multi-agent pipeline that filters and verifies candidate evidence, reports
conflicts, and abstains when relevant verified evidence does not survive.

## 2. Solution overview

The project runs two pipelines against the same local document corpus:

- **Standard RAG** retrieves candidates and passes them directly to answer
  generation.
- **Denoised Multi-Agent RAG** adds relevance scoring, source/evidence
  verification, contradiction analysis, and an original-query evidence gate
  before generation.

Both pipelines run the claim-level hallucination detector after generation.
The benchmark's independent evaluator is separate from that pipeline detector.

## 3. Architecture

```text
User query
   |
   +--> Standard RAG: Retrieval --> Answer Generation --> Pipeline Claim Check
   |
   +--> Denoised RAG:
        Retrieval
          -> Relevance Scoring
          -> Evidence Verification
          -> Contradiction Detection
          -> Original-Query Evidence Gate
          -> Generate OR Abstain
          -> Pipeline Claim Check

Independent benchmark evaluator: generated claims vs benchmark ground-truth evidence
Dashboard: HTTP UI -> actual Python pipelines; benchmark UI -> saved summary JSON
```

The live dashboard uses Python's standard-library HTTP server. It is not a
browser-side RAG implementation.

## 4. Agent responsibilities

| Component | Responsibility |
|---|---|
| Retrieval Agent | Returns top-k corpus documents using cached dense embeddings. |
| Relevance Scoring Agent | Combines semantic similarity and generic query-term alignment; applies the configurable threshold. |
| Evidence Verification Agent | Evaluates extracted claims against trusted corpus evidence with NLI and source reliability. |
| Contradiction Detection Agent | Detects contradictory claims and reports unresolved conflicts or corroboration-based source preference. |
| Answer Generation Agent | Generates from the supplied documents; uses deterministic extractive output when no optional provider key is configured. |
| Hallucination Detection Agent | Compares answer claims with evidence and reports statuses, NLI confidence, and evidence document IDs. |
| Denoised RAG pipeline | Applies the original-query gate and bounded retry policy; abstains if verified evidence is insufficient. |

## 5. Standard RAG vs. Denoised RAG

Standard RAG is the baseline: all retrieved candidates are generation context.
Denoised RAG scores and verifies documents, preserves conflict information,
and only generates from evidence that passes its original-query relevance,
trust, and verification requirements. The benchmark uses the same corpus,
query text, and initial top-k retrieval for both systems.

## 6. Technology stack

- Python (recommended: CPython 3.12)
- Sentence Transformers, Transformers, PyTorch, and NumPy
- Local JSON corpus and benchmark definitions
- Python standard-library HTTP server and browser HTML/CSS/JavaScript dashboard
- `unittest` test suite

The current workspace validation used Python 3.14.7 with
sentence-transformers 6.1.0, transformers 5.17.0, torch 2.14.0, and NumPy
2.5.3. The environment emitted a PyTorch JIT compatibility warning on Python
3.14; Python 3.12 is recommended for a new installation.

## 7. Models

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`

Used for document/query embeddings, dense retrieval, semantic similarity, and
the semantic guard used by NLI pair comparison. It is not an NLI classifier.

**NLI model:** `cross-encoder/nli-deberta-v3-small`

Used to classify premise/hypothesis pairs as **ENTAILMENT**,
**CONTRADICTION**, or **NEUTRAL**, and to provide class probabilities. A low
semantic-similarity guard maps unrelated non-neutral NLI results to neutral.

Both model identifiers can be overridden with environment variables. Models
are loaded locally and reused by the dashboard process. The first run requires
internet access to download model files from Hugging Face unless they are
already cached.

## 8. Pipeline flow and safety behavior

The Denoised RAG evidence gate re-scores evidence against the original user
query, including evidence discovered during a retry. A retry cannot bypass
the original-query relevance or verification requirements. Retry count is
bounded and configurable.

If no sufficiently relevant, trusted, verified evidence survives, Denoised
RAG returns a structured abstention with no generation evidence. The
`evidence_confidence_score` describes available evidence strength; it is not
answer correctness or model certainty. NLI claim labels and scores are
model-dependent and are not guarantees.

## 9. Benchmark methodology

The benchmark has **20 curated queries**, five per category: normal,
noisy/irrelevant, conflicting, and unsupported. Its corpus and query set are
small and domain/corpus-dependent; results are not statistically definitive.
Queries and expected facts are defined in `data/benchmark_queries.json` and
must not be changed merely to improve a score.

- **Expected-fact coverage** is exact normalized-token presence against
  curated expected facts. It does not prove factual correctness.
- **Expected-Fact Coverage / Abstention Accuracy** averages answerable-query
  expected-fact coverage and correct abstention on unsupported queries.
- **Supported, Unsupported, Contradicted, and Conflicted Claim Rates** are
  independent NLI evaluation metrics using extracted factual claims as their
  denominator. A claim with credible support and contradiction is classified
  as conflicted/unresolved.
- **Unsupported-query abstention rate** is correct abstentions divided by
  unsupported benchmark queries.
- **Noise reduction** compares irrelevant documents before filtering with
  irrelevant documents passed to generation.
- **Contradiction handling** and expected conflict detection are reported
  separately from claim classification.
- Benchmark evaluation compares generated claims with benchmark ground-truth
  evidence; it does not call the pipeline's own hallucination detector.
- Model initialization and NLI warm-up are measured separately from warm
  per-query pipeline latency.

## 10. Current benchmark results

These are the values in `benchmark/results/summary.json` (saved run generated
2026-09-26 23:12 UTC). They are a snapshot, not a promise of identical results
on other hardware, model revisions, or environments.

| Denoised RAG metric | Saved result |
|---|---:|
| Queries | 20 |
| Query categories | 5 normal, 5 noisy/irrelevant, 5 conflicting, 5 unsupported |
| Noise reduction | 98.98% |
| Expected-fact coverage | 30/30 (100%) |
| Expected-Fact Coverage / Abstention Accuracy | 1.000 |
| Unsupported-query abstention | 5/5 (100%) |
| Expected conflict document pairs detected | 5/5 |
| Supported claim rate | 18/27 (66.7%) |
| Unsupported claim rate | 1/27 (3.7%) |
| Contradicted claim rate | 8/27 (29.6%) |
| Conflicted claim rate | 0/27 (0%; no conflicted claim in this run) |

The composite value of 1.000 is not a claim of 100% factual accuracy. It
combines token coverage and correct unsupported-query abstention as defined
above.

### Saved warm latency and initialization

The latest saved run measured pipeline/index initialization at **7.983 s**
and NLI first-inference warm-up at **4.763 s** (both outside query timing).
Startup timing varies with cache and machine load.

| Warm pipeline latency | Standard RAG | Denoised RAG |
|---|---:|---:|
| Mean | 0.1581 s | 0.1289 s |
| Median | 0.1730 s | 0.0876 s |
| p95 (nearest rank) | 0.2473 s | 0.2962 s |

These timing values are hardware- and run-dependent. Model cold-start and
download time are not included in per-query latency.

## 11. Dashboard

`dashboard.py` serves `dashboard.html` and exposes only fixed local endpoints:
live comparison, saved benchmark summary, and demo examples. It initializes
both pipelines once per process and serializes concurrent use of their shared
model instances. The UI displays retrieved documents, evidence stages,
conflicts, claim findings, abstention, evidence confidence, and latency. In
the live claim panel, answer claims are filtered with the benchmark's generic
answer-claim extraction logic so headings, preambles, resolution/confidence
metadata, and duplicate conflict-format fragments are not presented as
factual claims. This is a display-only boundary: the pipeline's raw
hallucination-check output and benchmark evaluation are unchanged. Displayed
factual claims retain their status, evidence document IDs, and available NLI
evidence details.

The default server binds to `127.0.0.1`. It has no authentication and is for
local development; do not expose it to an untrusted network. Static hosting or
opening `dashboard.html` directly cannot run the Python RAG backend.

The benchmark section reads `benchmark/results/summary.json`. Run the
benchmark first if the saved file is unavailable.

## 12. Installation

Install CPython 3.12 and confirm `python --version` reports 3.12. From the
project directory, create and activate a virtual environment, then install
the listed dependencies:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No hidden setup steps are required. The first model-backed run needs internet
access to download uncached embedding and NLI models.

## 13. Configuration

Defaults are applied by the code; `.env.example` is a safe template, not
automatically loaded. Copy it to `.env`, fill only the variables needed, then
export them before starting Python. For Bash/Zsh:

```bash
cp .env.example .env
# Edit .env locally, then:
set -a
source .env
set +a
```

Never commit `.env` or real credentials. `.gitignore` excludes `.env` while
allowing `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `RAG_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Semantic embeddings/retrieval model. |
| `RAG_NLI_MODEL` | `cross-encoder/nli-deberta-v3-small` | NLI classification model. |
| `RAG_RELEVANCE_THRESHOLD` | `0.45` | Relevance filter threshold (0–1); configurable, not claimed to be scientifically optimal. |
| `RAG_MAX_RETRIES` | `1` | Maximum Denoised RAG retries; non-negative integer. |
| `RAG_SOURCE_PRIORITIES` | Built-in source policy | Optional JSON object of source category to score (0–1). |
| `ANTHROPIC_API_KEY` | Unset | Optional Anthropic answer-generation provider. |
| `GEMINI_API_KEY` | Unset | Optional Gemini answer-generation provider. |
| `GOOGLE_API_KEY` | Unset | Alternate Gemini key name. |

With no optional generation key configured, answers use the local deterministic
extractive generator. The benchmark explicitly disables provider keys to keep
its generation path local. Runtime dependencies are explicitly listed in
`requirements.txt`: `sentence-transformers`, `transformers`, `torch`, and
`numpy`; other project components use the Python standard library.

## 14. Running the benchmark

After installation and model download:

```bash
python benchmark.py
```

Current run artifacts are written to:

- `benchmark/results/summary.json`
- `benchmark/results/per_query_results.json`
- `benchmark/results/comparison.md`

Do not confuse these with the root-level historical snapshots described under
Limitations.

## 15. Running the dashboard

In a separate terminal, with the environment activated:

```bash
python dashboard.py
```

Then open <http://127.0.0.1:8000>. To use a different local port:

```bash
python dashboard.py --port 8010
```

The query endpoint accepts JSON, rejects empty questions, limits query text to
2,000 characters, bounds request bodies to 17,024 bytes, and returns generic backend errors
without stack traces or environment values. Optional API keys are used only
for generation and are never returned by the dashboard endpoints.

## 16. Testing

Run the complete unit suite:

```bash
python -m unittest discover -s tests -v
```

The final dashboard claim-display update was validated with **70 passing
tests**. Live dashboard smoke checks also covered a supported query, an
unsupported query that abstains with zero generation evidence, and a
conflicting query whose factual claims retain their statuses and evidence
without showing conflict metadata as claims.

## 17. Limitations and historical artifacts

- The benchmark is small, curated, and corpus-dependent; its scores are not
  statistically definitive and expected-fact token matching may miss
  paraphrases.
- NLI outputs depend on the selected model and thresholds; they can be wrong.
- Latencies depend on hardware and exclude cold model initialization.
- The local dashboard has no authentication and must not be exposed to an
  untrusted network.
- `benchmark_report.md` and `benchmark_results.json` at the repository root
  are **historical** eight-query evaluation artifacts from an earlier
  implementation. Their rubric score, zero hallucination value, accuracy,
  confidence, and latency are obsolete for current system claims. The current
  dashboard does not load them; use `benchmark/results/` instead. They are
  retained as history, not current evidence.
- The former Netlify configuration was removed because a static host cannot
  run the Python backend. No public deployment or authentication setup is
  included.
- Local scratch/debug scripts may be present in a working copy; `.gitignore`
  excludes their known patterns. They are not runtime dependencies.

No claim of perfect accuracy or hallucination-free output is made.

## 18. Project structure

```text
agents/                         Retrieval, relevance, evidence, conflict,
                                generation, and claim-detection components
pipeline/                       Standard and Denoised RAG orchestration
utils/                          Embeddings and shared NLI/source policy
data/documents.json             Local corpus
data/benchmark_queries.json     Fixed 20-query benchmark definition
benchmark.py                    Independent benchmark runner/evaluator
benchmark/results/              Current aggregate, raw, and report outputs
dashboard.py                    Local live-pipeline HTTP server
dashboard.html                  Dashboard UI
tests/                           Unit and dashboard endpoint tests
requirements.txt                Explicit runtime model dependencies
.env.example                    Safe configuration template
```
