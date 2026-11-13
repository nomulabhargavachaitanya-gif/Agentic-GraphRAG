# GraphRAG for Competitive Programming

> Agentic self-correcting retrieval over algorithmic knowledge — graph traversal + vector search with grading, fallback, and contradiction detection.

## Status: All 8 Phases ✅

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Schema design + Docker infra + manual graph seed + Cypher queries | ✅ Done |
| 2 | Ingestion pipeline (20 editorials → entities → Neo4j) | ✅ Done |
| 3 | Vector store + chunk embeddings linked to graph | ✅ Done |
| 4 | Query router + hybrid retrieval | ✅ Done |
| 5 | **Self-grading loop** + fallback/retry (the agentic core) | ✅ Done |
| 6 | Contradiction detection path | ✅ Done |
| 7 | Eval harness + metrics | ✅ Done |
| 8 | Streamlit demo UI | ✅ Done |

## Quick Start

### 0. Configure Gemini (optional)

Copy `.env.example` to `.env` and set `GEMINI_API_KEY` (free key from Google AI Studio). Without a key the system falls back to rules-based extraction and structured answers. Default model: `gemini-3.5-flash-lite` (override with `GEMINI_MODEL`).

### 1. Start infrastructure

```bash
docker compose up -d
```

This brings up **Neo4j** on `localhost:7474` (browser) / `localhost:7687` (Bolt).

Neo4j credentials: `neo4j` / `graphrag_password`

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Load the knowledge graph

```bash
python -m ingestion.load_graph
```

This inserts 8 algorithmic patterns (Two Pointers, Sliding Window, Binary Search, Prefix Sum, Fenwick Tree, Segment Tree, Sparse Table, Suffix Array) with:
- `PREREQUISITE_OF` edges (e.g. Prefix Sum → Fenwick Tree)
- `ALTERNATIVE_TO` edges with tradeoff descriptions
- `SAME_COMPLEXITY_CLASS` edges
- 5 Problem nodes linked via `SOLVED_WITH`
- 8 C++ CodeTemplate nodes linked via `HAS_TEMPLATE`
- 4 Source nodes linked via `EXTRACTED_FROM`

### 4. Run ingestion pipeline (Phase 2)

```bash
# Offline mode (no API key needed):
python -m ingestion.pipeline --mode rules --dry-run

# With live Neo4j:
python -m ingestion.pipeline --mode rules

# LLM extraction with Gemini (free tier; needs GEMINI_API_KEY in .env):
python -m ingestion.pipeline --mode llm
```

### 5. Build vector index (Phase 3)

```bash
python -m retrieval.vector_store
```

### 6. Test the query router (Phase 4)

```bash
python -m retrieval.query_router
```

Classifies queries into: `graph_hop`, `code_template`, `semantic`, `hybrid`, or `contradiction`.

### 7. Run the agentic RAG system & contradiction detector (Phases 5 & 6)

```bash
python -m agent.rag_agent
python -m agent.contradiction
```

Runs the full retrieve → grade → self-correct → synthesize loop on benchmark queries, plus corpus-wide complexity contradiction scanning. The agent:
- Routes each query to the optimal retrieval strategy
- Grades retrieved context for relevance & sufficiency
- Retries with wider search on low scores
- Detects conflicting Big-O complexity claims across sources
- Honestly abstains when a topic is not in the corpus

### 8. Run evaluation harness (Phase 7)

```bash
python -m eval.eval_harness
```

Runs 30 benchmark queries (graph-hop, code, hybrid, semantic, contradiction, out-of-corpus). Latest: 30/30 pass, 100% routing/recall/code-hit/abstention, 72% avg retrieval score. Evaluates routing accuracy, pattern recall, code hit rate, abstention precision, and generates a structured report saved to `eval/eval_report.json`.

### 9. Launch the demo UI (Phase 8)

```bash
streamlit run app.py   # opens http://localhost:8501
```

## Graph Schema

```
(Pattern)──PREREQUISITE_OF──►(Pattern)
    │                             │
    ├──ALTERNATIVE_TO────────────►│
    │                             │
    ├──SAME_COMPLEXITY_CLASS─────►│
    │
    ├──HAS_TEMPLATE──►(CodeTemplate)──EXTRACTED_FROM──►(Source)
    │
(Problem)──SOLVED_WITH──►(Pattern)

(Source)──CONTRADICTS──►(Source)   ← populated by Phase 6
```

## Architecture

```
          User Query
              │
              ▼
       ┌─────────────┐
       │ Query Router │  ← Phase 4: keyword + pattern classification
       └──────┬──────┘
              │
    ┌─────────┼─────────┐
    ▼                    ▼
┌────────┐        ┌──────────┐
│ Graph  │        │  Vector  │
│Retriever│        │Retriever │
│(Neo4j) │        │ (FAISS)  │
└────┬───┘        └────┬─────┘
     │                  │
     └────────┬─────────┘
              ▼
       ┌─────────────┐
       │   Hybrid     │  ← Phase 4: context fusion
       │   Fusion     │
       └──────┬──────┘
              ▼
       ┌─────────────┐
       │ Self-Grading │  ← Phase 5: relevance + sufficiency scoring
       │    Agent     │◄──── retry loop (widen k, expand hops)
       └──────┬──────┘
              │
    ┌─────────┼──────────┐
    ▼         ▼          ▼
 Answer   Contradiction  Abstain
 + Code   + Attribution  ("Not in corpus")
```

## Project Structure

```
rag_project/
├── ingestion/
│   ├── load_graph.py          ← Phase 1: graph schema + seed data
│   ├── scrape.py              ← Phase 2: editorial scraper
│   ├── parse_cpp.py           ← Phase 2: C++ code parser
│   ├── extract_entities.py    ← Phase 2: entity extraction
│   └── pipeline.py            ← Phase 2: end-to-end pipeline
├── retrieval/
│   ├── graph_retriever.py     ← Phase 1: Cypher traversal engine
│   ├── vector_store.py        ← Phase 3: FAISS index + embeddings
│   ├── vector_retriever.py    ← Phase 3: vector search interface
│   ├── query_router.py        ← Phase 4: query classification
│   └── hybrid_retriever.py    ← Phase 4: joint graph+vector retrieval
├── llm/
│   └── gemini.py              ← Gemini client with retry/backoff
├── agent/
│   ├── rag_agent.py           ← Phase 5: self-correcting agentic loop
│   └── contradiction.py       ← Phase 6: Big-O contradiction detector
├── app.py                     ← Phase 8: Streamlit demo UI
├── eval/
│   ├── test_queries.json      ← 30 benchmark test cases
│   └── eval_harness.py        ← Phase 7: evaluation + metrics
├── data/
│   ├── editorials.json        ← 20 curated CP editorials
│   └── vector_index/          ← persisted FAISS index
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Author

**N Bhargava** — B.Tech DSAI, IIIT Bangalore
