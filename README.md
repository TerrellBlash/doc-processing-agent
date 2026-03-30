# Intelligent Document Processing Agent

> LangGraph-powered pipeline that classifies, extracts, validates, and routes documents with human-in-the-loop review. Deployed to Azure Container Apps.

Built as a portfolio project for the IBM AI Engineering Professional Certificate.

## What This Does

Feed it any document (invoice, resume, or contract) and the agent:

1. **Ingests** — validates the document is processable
2. **Classifies** — LLM determines document type + confidence score
3. **Extracts** — specialist node pulls structured data based on doc type
4. **Validates** — confidence gate checks extraction quality
5. **Routes** — high confidence → auto-store, low confidence → human review
6. **Stores** — packages clean JSON output ready for any database

The human-in-the-loop pattern uses LangGraph's `interrupt()` — the pipeline literally pauses, saves all state, and waits for a human to approve or correct the extraction. Could be 5 seconds or 5 days.

## Architecture

```
Document → Ingest → Classify ─┬─→ Extract Invoice  ─┐
                               ├─→ Extract Resume   ─┼─→ Validate ─┬─→ Store Results
                               └─→ Extract Contract ─┘             │
                                                                    └─→ Human Review → Store Results
```

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key (or swap for Gemini/Claude in `app/nodes.py`)

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

## Deploy to Azure

```bash
# Login to Azure
az login

# Set your API key
export OPENAI_API_KEY="sk-..."

# Deploy (creates all Azure resources)
./deploy.sh
```

See `deploy.sh` for the full Azure Container Apps setup.

## Project Structure

```
doc-processing-agent/
├── app/
│   ├── __init__.py
│   ├── state.py          # State definitions (the shared notebook)
│   ├── nodes.py          # All node functions (each does one job)
│   ├── graph.py          # Graph wiring (connects nodes with edges)
│   └── main.py           # FastAPI server (API endpoints)
├── tests/
│   └── test_pipeline.py  # Full pipeline tests with sample docs
├── Dockerfile            # Container config for Azure
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
| **Confidence gating** | `validate_extraction()` — auto-store vs human review |
| **interrupt()** | `human_review()` — pause/resume across sessions |
| **Retry policies** | `graph.py` — extraction nodes retry on failure |
| **Checkpointing** | `graph.py` — MemorySaver enables durable execution |

## Production Upgrades

For a real production deployment, consider:

- **Persistent checkpointer**: Swap `MemorySaver` for `PostgresCheckpointer` so state survives container restarts
- **PDF parsing**: Add PyMuPDF or pdf-parse to handle actual PDF files, not just text
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
