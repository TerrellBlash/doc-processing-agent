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

### Unit 1 - Workflow vs agent (theory)

This section reconciles the repo against IBM Course 12 Module 1, which introduces
AI agents and tool calling. No code or graph changes; documentation only.

**The term "agent" is contested.** There is no single accepted definition, so
locating this repo on the workflow-agent line requires picking a definition first.
Two common ones, both relevant to Module 1:

- **Tool-use definition.** An agent is an LLM that can call tools to act on
  external systems, rather than only producing text. Module 1's framing leans this
  way: tool calling turns an LLM from a passive responder into an active problem
  solver. (inference: this is the emphasis of the module's description; the
  lectures may state a more precise definition. Correct here if so.)
- **LLM-as-controller definition.** An agent is a system where the LLM decides
  control flow, which steps run and in what order, instead of a developer fixing
  the flow in code.

**Where this repo sits, under each:**

- *Under the tool-use definition:* the repo currently has zero agentic nodes. No
  node calls a tool. Unit 2 will add the first: `enrich_extraction`, a tool-bound
  LLM that decides whether to look up a vendor. So under this definition the repo
  gains one agentic node at Unit 2, and remains a workflow everywhere else.
- *Under the LLM-as-controller definition:* the repo is a workflow throughout,
  before and after Unit 2. Control flow is fixed in `app/graph.py`. The `classify`
  node selects from a closed, developer-defined set (`extract_invoice`,
  `extract_resume`, `extract_contract`, `human_review`) via a fixed
  `Command[Literal[...]]` return (`app/nodes.py:64`). That is routing among known
  options, not the LLM choosing the program. Unit 2's tool node decides one local
  thing (call the tool or not); it never decides which nodes run.

**Why the distinction matters here.** Under either definition, deterministic
routing is the correct design for this problem. The document types are known in
advance. An LLM improvising control flow over a known structure would add
nondeterminism without adding capability. This repo is a workflow by choice, not
by limitation. The one place real LLM discretion is warranted is a bounded
enrichment decision (Unit 2), and that is deliberately the only place it enters.
