# GraphRAG for Competitive Programming

> Agentic self-correcting retrieval over algorithmic knowledge — graph traversal + vector search with grading, fallback, and contradiction detection.

## Status: Phase 1 ✅

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Schema design + Docker infra + manual graph seed + Cypher queries | ✅ Done |
| 2 | Ingestion pipeline (20 editorials → entities → Neo4j) | ⬜ |
| 3 | Vector store + chunk embeddings linked to graph | ⬜ |
| 4 | Query router + hybrid retrieval | ⬜ |
| 5 | **Self-grading loop** + fallback/retry (the agentic core) | ⬜ |
| 6 | Contradiction detection path | ⬜ |
| 7 | Eval harness + metrics | ⬜ |
| 8 | Streamlit demo UI (optional) | ⬜ |

## Quick Start

### 1. Start infrastructure

```bash
docker compose up -d
```

This brings up:
- **Neo4j** on `localhost:7474` (browser) / `localhost:7687` (Bolt)
- **Qdrant** on `localhost:6333` (REST) / `localhost:6334` (gRPC)

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

### 4. Run test queries

```bash
python -m retrieval.graph_retriever
```

Runs three queries from the test suite:

1. **Pattern comparison**: "How does a segment tree differ from a Fenwick tree?" → ALTERNATIVE_TO traversal + both templates
2. **Prerequisite chain**: "What do I need to know before suffix arrays?" → recursive PREREQUISITE_OF chain
3. **Complexity lookup**: "Time complexity of sparse table?" → direct property lookup with full-text fallback

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

See the full spec for the complete architecture diagram. The key insight: after retrieval, a **self-grading module** evaluates whether results are relevant, sufficient, and non-contradictory — then either synthesizes an answer, retries with wider scope, or honestly abstains.

## Project Structure

```
graphrag-cp/
├── ingestion/
│   └── load_graph.py          ← Phase 1: manual graph seed
├── retrieval/
│   └── graph_retriever.py     ← Phase 1: Cypher queries
├── agent/                     ← Phase 5: grading + synthesis
├── api/                       ← Phase 4: FastAPI
├── eval/
│   └── test_queries.json      ← all 6 test cases
├── docker-compose.yml
├── requirements.txt
└── README.md
```
