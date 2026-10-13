# Intelligent Document Processing Agent

> LangGraph-powered pipeline that classifies, extracts, validates, and routes documents with human-in-the-loop review. Paused reviews persist in Postgres, and a deploy script targets Azure Container Apps.



## What This Does

Feed it any document (invoice, resume, or contract), as text, a PDF, or a photo, and the agent:

1. **Reads** — a PDF's own text layer when it has one; scanned pages and PNG/JPEG photos go to the vision model, and each page records which source produced its text
2. **Ingests** — validates the document is processable
3. **Classifies** — LLM determines document type + confidence score
4. **Extracts** — specialist node fills a per-type schema with structured output; fields the document lacks come back null
5. **Validates** — confidence gate checks extraction quality
6. **Routes** — high confidence → auto-store, low confidence → human review
7. **Stores** — packages clean JSON output ready for any database

The human-in-the-loop pattern uses LangGraph's `interrupt()` — the pipeline literally pauses, saves all state, and waits for a human to approve or correct the extraction. Could be 5 seconds or 5 days.

## Architecture

```
PDF / image → Read (text layer, or vision model for scans)
                    │
Text ───────────────┴→ Ingest → Classify ─┬─→ Extract Invoice  ─┐
                                           ├─→ Extract Resume   ─┼─→ Validate ─┬─→ Store Results
                                           └─→ Extract Contract ─┘             │
                                                                                └─→ Human Review → Store Results
```

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key

### Local Development

```bash
# Clone and install
git clone https://github.com/TerrellBlash/doc-processing-agent.git
cd doc-processing-agent
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY="sk-..."

# Run the server
uvicorn app.main:app --reload --port 8000

# Health check
curl http://localhost:8000/health
```

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | none | Required. |
| `DOC_AGENT_MODEL` | `gpt-6-luna` | OpenAI model for every step. A run can override it with `{"configurable": {"model": ...}}`. |
| `DOC_AGENT_REASONING_EFFORT` | `none` | Reasoning effort for reasoning models (GPT-5 and later). Older models run at temperature 0 instead. Some models don't accept `none` (gpt-6.1-sol needs `low` or higher) and fail every call with a 400 until this is set. |
| `DATABASE_URL` | unset | Postgres for paused reviews. Unset falls back to in-memory checkpoints. |

### Process a Document

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "document_text": "INVOICE #INV-2024-0847\nFrom: CloudTech Solutions\nTotal: $19,593.25",
    "document_filename": "invoice_march.pdf",
    "document_source": "email_attachment"
  }'
```

### Process a PDF or Image

```bash
curl -X POST http://localhost:8000/process/file \
  -F "file=@invoice.pdf" \
  -F "document_source=email_attachment"
```

PDF, PNG and JPEG. The type is detected from the file's bytes, not its name.
`final_output.document_pages` shows each page's source (`text_layer` or
`vision`), so a reviewer knows which pages a model transcribed.

Uploads are untrusted, so each size is checked before the work it would cost:

| Limit | Value | Checked |
|---|---|---|
| Upload size | 20 MB | From `Content-Length`, before the body is read (411 without one) |
| Pages | 20 | Before any page is read or sent to the vision model |
| Rendered page | 2048 px on the long side | Before a scanned page is drawn, whatever size the PDF declares |
| Image | 50 megapixels | Before the image is decoded |
| Text | 120,000 characters | Before the graph runs, since every extraction prompt carries the full text |

Errors return a short message. Details go to the server log, not the client.

### Submit Human Review

```bash
curl -X POST http://localhost:8000/review \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "YOUR_THREAD_ID_HERE",
    "approved": true,
    "corrections": {"total_amount": 19593.25}
  }'
```

### Run Tests

```bash
pytest tests/test_pipeline.py -v
```

#### Local Postgres for testing

The checkpointer test that proves paused reviews survive a restart
(`tests/test_checkpointer_restart.py`) needs a real Postgres instance —
it's skipped automatically if `DATABASE_URL` isn't set. To run it:

```bash
docker compose up -d postgres
export DATABASE_URL=postgresql://docagent:docagent@localhost:5433/docagent
pytest tests/test_checkpointer_restart.py -v
```

## Deploy to Azure

```bash
# Login to Azure
az login

# Set your API key
export OPENAI_API_KEY="sk-..."

# Point at a provisioned Postgres instance (e.g. Azure Database for
# PostgreSQL Flexible Server) so checkpoints survive restarts — see
# "Checkpointing & Persistence" below. Omitting this falls back to
# in-memory checkpointing, which defeats --workers 4 in the Dockerfile.
export DATABASE_URL="postgresql://user:password@host:5432/dbname"

# Deploy (creates all Azure resources)
./deploy.sh
```

See `deploy.sh` for the full Azure Container Apps setup.

## Project Structure

```
doc-processing-agent/
├── app/
│   ├── __init__.py
│   ├── state.py          # State definitions
│   ├── nodes.py          # All node functions (each does one job)
│   ├── schemas.py        # Extraction schemas per document type
│   ├── llm.py            # Model selection (DOC_AGENT_MODEL, per-run override)
│   ├── documents.py      # PDF/image to text: text layer first, vision model for scans
│   ├── graph.py          # Graph wiring (connects nodes with edges), build_checkpointer()
│   └── main.py           # FastAPI server (API endpoints)
├── tests/
│   ├── test_pipeline.py              # Full pipeline tests with sample docs
│   ├── test_confidence.py            # Unit tests for extraction confidence scoring
│   ├── test_extraction.py            # Schema-enforced extraction against a stub model
│   ├── test_llm.py                   # Model selection and per-run override
│   ├── test_documents.py             # PDF/image reading, page sources, limits
│   ├── test_api_file.py              # Upload endpoint: errors and page provenance
│   ├── test_scoring.py               # Eval field-comparison rules
│   ├── test_eval_summary.py          # Eval routing and accuracy metrics
│   ├── test_checkpointer_restart.py  # Proves paused reviews survive a worker restart (needs Postgres)
│   └── test_checkpointer_infra.py    # LLM-free Postgres persistence and setup tests (needs Postgres)
├── evals/
│   ├── README.md         # Dataset, scoring rules, metric definitions, limits
│   ├── build_dataset.py  # Seeded synthetic dataset: 44 docs across text, PDFs, scans
│   ├── run_eval.py       # Runs the pipeline per model; accuracy, routing, latency, cost
│   ├── scoring.py        # Field comparison rules
│   ├── pricing.py        # Token prices for cost estimates
│   ├── pdfgen.py         # Builds text-layer and scanned PDFs for tests and the eval set
│   ├── data/             # The dataset and labels.json
│   └── results/          # One JSON per run, plus a generated summary
├── Dockerfile            # Container config for Azure
├── docker-compose.yml    # Local Postgres for development/testing only
├── deploy.sh             # One-command Azure deployment
├── requirements.txt
└── README.md
```

## Key LangGraph Concepts Demonstrated

| Concept | Where It's Used |
|---------|----------------|
| **State design** | `state.py` — raw data, never formatted prompts |
| **Node functions** | `nodes.py` — each function does one thing |
| **Command routing** | `classify_document()` — LLM decides next node |
| **Structured output** | `schemas.py` — every field required and nullable, so the model must state each one |
| **Confidence gating** | `validate_extraction()` — auto-store vs human review |
| **interrupt()** | `human_review()` — pause/resume across sessions |
| **Retry policies** | `graph.py` — extraction nodes retry on failure |
| **Checkpointing** | `graph.py` — `build_checkpointer()` picks Postgres or MemorySaver, enabling durable execution |

## Checkpointing & Persistence

The pipeline's `human_review` step pauses the graph with LangGraph's
`interrupt()` — state has to live somewhere for that pause to survive
until a human responds. Where it lives depends on `DATABASE_URL`:

- **`DATABASE_URL` set** — `build_checkpointer()` (in `app/graph.py`) builds
  a `psycopg_pool.ConnectionPool` and a `PostgresSaver` on top of it. Every
  worker process's `.setup()` call is idempotent, so all 4 uvicorn workers
  can run it concurrently on startup. A review submitted to a different
  worker than the one that paused it still resolves, and a full container
  restart doesn't lose pending reviews.
- **`DATABASE_URL` not set** — falls back to an in-memory `MemorySaver`,
  logs a `WARNING`, and keeps running. This is fine for local development
  and tests, but **not restart-safe**: a container restart or a request
  landing on a different worker silently loses any paused review. Never
  run it this way in a multi-worker deployment.

For local Postgres setup, see "Local Postgres for testing" above.

## Evaluation

The default model was chosen by running the pipeline over a labeled set of
44 synthetic documents: invoices, resumes and contracts as plain text,
text-layer PDFs and scanned PDFs, plus a poor-scan copy (80 DPI, heavy blur)
of every scanned one. [evals/README.md](evals/README.md) covers the dataset
and scoring. Every run's per-document output is in
[evals/results/](evals/results/README.md).

| Model | Fields correct, poor scans | Fields correct, everything else | Silent errors | Cost per 1,000 docs | p50 seconds |
|---|---|---|---|---|---|
| gpt-4o-mini | 62% | 99% | 2 | $2.30 | 3.9 |
| **gpt-6-luna**, effort none | 88% | 100% | 1 | $0.48 | 5.7 |
| gpt-6-luna, effort low | 93% | 100% | 2 | $0.54 | 7.1 |
| gpt-6.1-sol, effort low | 99% | 100% | 1 | $9.48 | 11.7 |

A silent error is a document stored without review that had something
wrong in it. Those are the expensive ones, because nobody looks at them.

What the numbers say:

- **Clean input is solved.** On plain text, text PDFs and ordinary scans,
  every gpt-6 run extracted every field correctly. gpt-4o-mini missed the
  contract type on two contracts, but the confidence gate sent both to review.
- **Every silent error came from a poor scan.** The model misread a digit or
  a word, and the result looked like a valid value: 2026 read as 2025,
  invoice 91630 as 91830, a (312) area code as (912), "Cardinal Insurance"
  as "Cleartrail Insurance". The confidence gate checks that fields are
  present, not that they're right, so these were stored.
- **gpt-6-luna costs a fifth of gpt-4o-mini.** Most of the gap is image
  tokens: a scanned page costs about 27,000 input tokens on gpt-4o-mini and
  about 3,900 on gpt-6-luna. gpt-6-luna is about 2 seconds slower per
  document.
- **gpt-6.1-sol reads poor scans almost perfectly, at 20 times the cost.**
  Worth it only if most of the input is bad scans and review is expensive.
- **Reasoning effort low didn't clearly help.** Its gain on poor scans is
  within run-to-run noise on 11 documents.

The default is gpt-6-luna at effort none. These are one run each on a small
synthetic set, good for comparing models, not a production accuracy figure.

## Known Limitations

- **The confidence gate can't see misreads.** It counts missing fields, so a
  wrong value read from a bad scan passes as confidently as a right one (see
  Evaluation). Reading low-resolution pages twice and sending disagreements
  to review, or sending every low-resolution page to review, would close
  most of this.

- **A vision refusal becomes the document's text.** gpt-4o-mini refused to
  transcribe two of the poor scans ("I'm sorry, but I can't assist with
  that."). The pipeline classified that text as an unknown document and sent
  it to review, which is safe, but the refusal should be detected and
  reported as unreadable.

- **PDF reading is serialized per process.** PDFium isn't thread-safe, so one
  lock covers every PDFium call. Vision calls, the slow part, still run in
  parallel.

- **Every schema field counts toward confidence.** A document type's fields
  are all treated as required, so an NDA with no payment clause has a null
  `payment_terms` and goes to human review. Per-type optional fields would
  fix this.

## Production Upgrades

For a real production deployment, consider:

- **Authentication**: Add API key auth or OAuth to the FastAPI endpoints
- **Observability**: Connect LangSmith for tracing and debugging
- **Streaming**: Add SSE endpoints to stream progress updates to a frontend

## Tech Stack

- **LangGraph** — agent orchestration with state machines
- **LangChain** — LLM integration (OpenAI, swappable)
- **FastAPI** — async API server
- **Docker** — containerization
- **Azure Container Apps** — cloud deployment

## License

MIT
