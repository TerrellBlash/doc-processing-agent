# LEARNING.md

This repo doubles as the hands-on spine for IBM "Fundamentals of AI Agents Using
RAG and LangChain" (Course 12 of 13, Generative AI Engineering with LLMs). Course
work lands here as real, disciplined commits rather than throwaway exercises. This
file tracks what each unit added, where the RAG subsystem lives, and what is
deliberately not RAG.

## Where the RAG subsystem lives

Planned, not yet built. RAG will enter at one architectural seam: a new
`retrieve_context` node between `classify` and the extract step. The retrieval
primitives (FAISS index, encoder) will live in `app/retrieval.py`. This section
will be updated with concrete file and node references as those land.

## What is deliberately not RAG

The existing extraction pipeline pulls structured data *from* a single document.
That is extraction, not retrieval. RAG retrieves context *from a knowledge base*
of many documents to augment generation. These are different operations. The
existing nodes (ingest, classify, extract, validate, store) are not RAG and will
not be relabeled as such. New RAG work is additive: new nodes, new state fields,
existing logic unchanged.

## Branch convention

- Each addition lands on a `feature/rag-*` branch.
- One well-formed commit per logical addition. No `git add -A`; stage only the
  files that belong to the change.
- Merge to main when the unit's work is complete and reviewed.

## Roadmap (planned, not done)

These units are planned. Each gets a detailed plan when approved, before any code.
Entries move into the Unit log below only after the work actually lands.

- **Unit 0** - This file and the branch convention. Markdown only.
- **Unit 1** - Agent foundations (theory). A section here on why this repo is a
  workflow, not a ReAct-style agent.
- **Unit 2** - Tool calling. One tool (vendor lookup), new `enrich_extraction`
  node, LLM tool-binding decides when to call.
- **Unit 3** - RAG from primitives. `app/retrieval.py`: FAISS + sentence-
  transformers, `KnowledgeBase` with `add_documents` and `search`. Standalone,
  not yet wired to the graph.
- **Unit 4** - RAG integrated. `retrieve_context` node between classify and
  extract, new `retrieved_context` state field, extract prompts read it.
- **Unit 5** - Prompt engineering. Few-shot examples and chain-of-thought in the
  extract node prompts.
- **Unit 6** - LangChain refactor. `KnowledgeBase` internals rewrapped on
  LangChain's FAISS vectorstore and HuggingFaceEmbeddings, public interface
  unchanged. Framework-familiarity unit, not a repo-improvement unit.
- **Unit 7** - Evaluation harness. `tests/test_rag_quality.py` measuring
  extraction accuracy across RAG / few-shot / chain-of-thought conditions, with
  an `EVALUATION.md` write-up. Not from the course; deliberate addition.
- **Unit 8** - Capstone alignment. `POST /qa` endpoint answering questions over a
  previously processed document via `KnowledgeBase`. Head start on Course 13.

## Unit log (landed work)

### Unit 0 - Scaffold

Added this file and established the `feature/rag-*` branch convention. No code, no
graph changes. Sets up the structure that later units append to.
