# Agentic GraphRAG for Competitive Programming

> **Self-correcting hybrid Graph + Vector Retrieval-Augmented Generation (RAG) engine over algorithmic editorials and C++ implementations — featuring deterministic query routing, multi-hop Neo4j Cypher traversal, FAISS dense retrieval, automated context grading with iterative search widening, Big-O contradiction detection, and calibrated abstention.**

[![Live Demo on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agentic-graphrag.streamlit.app/) — **[https://agentic-graphrag.streamlit.app/](https://agentic-graphrag.streamlit.app/)**

---

## Overview & Motivation

Standard vector-only RAG systems struggle with competitive programming and algorithmic knowledge for three structural reasons:

1. **Relational & Multi-Hop Reasoning**: Answering *"What do I need to learn before Suffix Arrays?"* or *"When should I prefer a Fenwick Tree over a Segment Tree?"* requires traversing explicit prerequisite chains (`PREREQUISITE_OF`) and trade-off edges (`ALTERNATIVE_TO`) rather than matching surface-level text similarity.
2. **Code vs. Prose Modality Gap**: Queries asking for compilable C++ boilerplate require exact data structure templates linked to their canonical pattern, whereas conceptual or complexity queries require prose explanations and mathematical bounds.
3. **Conflicting Claims & Out-of-Corpus Hallucinations**: Community editorials frequently state problem-specific or inaccurate asymptotic bounds (e.g., claiming $O(n \log n)$ per Segment Tree query instead of $O(\log n)$), and standard LLMs readily hallucinate explanations for unindexed algorithms instead of abstaining.

**Agentic GraphRAG** solves these challenges by combining a **Neo4j Knowledge Graph** with a **FAISS Vector Store** inside a **self-grading agentic loop** powered by **Google Gemini**. Every query is classified into one of five retrieval strategies, graded across three quantitative dimensions (*Relevance*, *Sufficiency*, *Coverage*), automatically retried with a wider retrieval window (`k` doubling) when evidence is weak, checked for conflicting Big-O claims across sources, and forced to abstain when the topic lies outside the corpus.

---

## Key Capabilities

- **5-Way Deterministic Query Router (`retrieval/query_router.py`)**
  Classifies incoming questions using word-boundary regex alias resolution (19 algorithmic pattern families) and intent signatures into:
  - `graph_hop` — Multi-hop prerequisite chains (`PREREQUISITE_OF*1..10`) and pairwise trade-off comparisons (`ALTERNATIVE_TO`).
  - `code_template` — Verified C++ template lookup from Neo4j (`HAS_TEMPLATE`) fused with filtered FAISS + BM25 code-chunk search.
  - `hybrid` — Joint dense FAISS + lexical Okapi BM25 Reciprocal Rank Fusion (RRF) over editorial chunks followed by automatic entity extraction and 1-hop graph property expansion.
  - `semantic` — Dense `all-MiniLM-L6-v2` (384-d L2-normalized) + Okapi BM25 RRF similarity search.
  - `contradiction` — Multi-source retrieval + canonical graph complexity lookup for conflict verification.

- **Dense FAISS + Okapi BM25 Reciprocal Rank Fusion (`retrieval/vector_store.py`)**
  Combines 384-d unit-normalized dense embeddings with an in-memory Okapi BM25 ($k_1=1.5, b=0.75$) inverted index via Reciprocal Rank Fusion ($\text{RRF}(d) = \frac{1}{k + r_{\text{dense}}(d)} + \frac{w}{k + r_{\text{bm25}}(d)}$), boosting exact algorithmic identifier matches without inflating out-of-corpus noise.

- **Self-Grading & Iterative Search Widening (`agent/rag_agent.py`)**
  Evaluates retrieved context without extra LLM latency using token/content overlap, cosine similarity calibration ($\cos\theta = 1 - \|\mathbf{u}-\mathbf{v}\|_2^2 / 2$), and pattern coverage:
  - **Sufficient ($\ge 0.60$)**: Proceeds directly to grounded Gemini synthesis.
  - **Partial ($0.25 \le \text{score} < 0.60$)**: Triggers self-correction by doubling the retrieval window ($k \to 2k$, up to $k=20$) and re-grading (up to 2 retries).
  - **Abstain ($< 0.25$)**: Halts synthesis after retry and returns an explicit, hallucination-free abstention notice.

- **Two-Stage Big-O Contradiction Detector (`agent/contradiction.py`)**
  Extracts sentence-level asymptotic complexity claims (`O(...)`) per operation (`build`, `query`, `update`), compares claims across editorial sources and the canonical graph node, and uses Gemini in JSON mode to filter out false positives (such as total runtime vs. per-query complexity) before surfacing genuine conflicts with full source attribution.

- **Dual-Mode Ingestion Pipeline (`ingestion/`)**
  Parses raw Codeforces and LeetCode editorials (`data/editorials.json`), separates prose from C++ code blocks (`parse_cpp.py`), extracts patterns, relationships, and annotated templates via either **Gemini structured JSON extraction** (`--mode llm`) or **deterministic rule-based extraction** (`--mode rules`), normalizes canonical names, and merges entities idempotently into Neo4j.

- **Graceful Offline / Cloud Fallback (`retrieval/graph_retriever.py`)**
  Runs against live Neo4j when available and automatically falls back to `InMemoryGraphRetriever` (seeded with both canonical graph edges and `data/extractions.json`) when deployed to serverless environments without a database container.

---

## System Architecture

```
                         User Question
                               │
                               ▼
                  ┌─────────────────────────┐
                  │      Query Router       │
                  │ (Alias + Intent Engine) │
                  └────────────┬────────────┘
                               │
         ┌──────────────┬──────┴──────┬──────────────┬──────────────┐
         ▼              ▼             ▼              ▼              ▼
   [graph_hop]   [code_template]  [hybrid]      [semantic]   [contradiction]
         │              │             │              │              │
         ▼              ▼             ▼              ▼              ▼
  ┌──────────────────────────┐                 ┌──────────────────────────┐
  │  Neo4j Knowledge Graph   │◄───pattern_ids──┤  FAISS + BM25 RRF Store  │
  │  • 15 Pattern nodes      │                 │  • 68 embedded chunks    │
  │  • PREREQUISITE_OF       │                 │    (48 prose, 20 C++)    │
  │  • ALTERNATIVE_TO        │                 │  • all-MiniLM-L6-v2      │
  │  • 37 CodeTemplate nodes │                 │  • Okapi BM25 + RRF      │
  └─────────────┬────────────┘                 └─────────────┬────────────┘
                └─────────────────────┬──────────────────────┘
                                      ▼
                       ┌─────────────────────────────┐
                       │   Hybrid Context Fusion     │
                       └──────────────┬──────────────┘
                                      ▼
                       ┌─────────────────────────────┐
                       │     Context Grader          │◄──── Self-Correction Loop
                       │  (Relevance / Sufficiency / │      (widen k → 2k, max 20)
                       │   Coverage Scoring)         │
                       └──────────────┬──────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
     [verdict: sufficient]   [route: contradiction]   [verdict: abstain]
              │                       │                       │
              ▼                       ▼                       ▼
     Grounded Answer +       Two-Stage Big-O         Honest Abstention
     C++ Code Synthesis      Conflict Resolution     ("Not in corpus")
```

---

## Knowledge Graph Schema

### Node Labels
| Label | Key Properties | Description |
| :--- | :--- | :--- |
| `Pattern` | `name`, `category`, `time_complexity`, `space_complexity`, `description` | Canonical algorithmic paradigm or data structure (e.g., *Segment Tree*, *Fenwick Tree*, *Suffix Array*, *Sliding Window*). |
| `CodeTemplate` | `id`, `code`, `language`, `annotated_explanation` | Compilable C++ implementation linked to its parent pattern and editorial source. |
| `Problem` | `id`, `title`, `platform`, `difficulty`, `url` | Competitive programming problem from Codeforces or LeetCode (e.g., `CF-1354D`, `LC-307`). |
| `Source` | `url`, `platform`, `author`, `date_published` | Provenance record for every ingested editorial or reference article. |

### Relationships
```
(Pattern)──[:PREREQUISITE_OF]────────►(Pattern)
(Pattern)──[:ALTERNATIVE_TO {tradeoff}]──►(Pattern)
(Pattern)──[:SAME_COMPLEXITY_CLASS]──►(Pattern)
(Pattern)──[:HAS_TEMPLATE]───────────►(CodeTemplate)──[:EXTRACTED_FROM]──►(Source)
(Problem)──[:SOLVED_WITH]────────────►(Pattern)
(Source)───[:CONTRADICTS {pattern, operation, claim_a, claim_b}]──►(Source)
```

---

## Evaluation Benchmark (30 Queries)

The evaluation suite (`eval/eval_harness.py` + `eval/test_queries.json`) tests the full pipeline across **30 benchmark queries** spanning all 5 routing categories:
- **7 Graph-Hop queries** (transitive prerequisite chains & pairwise data structure trade-offs)
- **7 Code Template queries** (C++ implementations for Two Pointers, Binary Search, Segment Tree, Sliding Window, Prefix Sum, Fenwick Tree, BFS)
- **8 Hybrid complexity & conceptual queries** (time/space complexity + editorial usage)
- **3 Contradiction queries** (conflicting vs. consistent Big-O claims across sources)
- **5 Out-of-Corpus queries** (*Aho-Corasick*, *Dinic's Max Flow*, *Fast Fourier Transform*, *Mo's Algorithm*, *Quantum Annealing* — verifying zero hallucination)

### Latest Benchmark & 3-Way Ablation Results (`eval/eval_report.json`)

To quantify the contribution of the knowledge graph, Reciprocal Rank Fusion, and the self-grading agentic loop, `eval/eval_harness.py` evaluates all 30 queries across three system configurations:

1. **Vanilla Vector-Only RAG (`vector_only`)**: Pure dense FAISS top-$k$ retrieval ($k=5$), no query routing, no graph traversal, no self-grading, and no abstention gate.
2. **Static Hybrid Graph + RRF Vector (`hybrid_no_agent`)**: 5-way Router + Neo4j/In-Memory Graph + FAISS/BM25 RRF, without iterative search widening, contradiction verification, or calibrated abstention.
3. **Full Agentic GraphRAG (`full_agentic`)**: Complete pipeline with 5-way Router, Graph + FAISS/BM25 RRF, Context Grader + Search Widening ($k \to 2k$), Two-Stage Big-O Contradiction Detector, and Calibrated Abstention.

| Metric | 1. Vanilla Vector-Only | 2. Static Hybrid (Graph + RRF) | 3. Full Agentic GraphRAG |
| :--- | :---: | :---: | :---: |
| **Overall Pass Rate** | 53.3% (16 / 30) | 73.3% (22 / 30) | **100.0% (30 / 30)** |
| **Routing Accuracy** | 16.7% | 100.0% | **100.0% (30 / 30)** |
| **Graph-Hop Prerequisite / Trade-off Recall** | 78.6% | 100.0% | **100.0%** |
| **C++ Code Template Hit Rate** | 75.0% | 100.0% | **100.0%** |
| **Big-O Contradiction Verification** | 0.0% | 0.0% | **100.0% (3 / 3)** |
| **Out-of-Corpus Abstention (Zero Hallucination)** | 0.0% (0 / 5) | 0.0% (0 / 5) | **100.0% (5 / 5)** |

> **Key Takeaway:** Adding the 5-way router, knowledge graph traversal, and BM25/Dense RRF lifts pass rate from **53.3% $\rightarrow$ 73.3% (+20.0 pp)** by resolving transitive prerequisite chains and C++ template retrieval. Adding the self-grading agentic loop, Big-O conflict verifier, and calibrated abstention gate lifts pass rate from **73.3% $\rightarrow$ 100.0% (+26.7 pp, +46.7 pp overall over Vector-Only)** while eliminating 100% of out-of-corpus hallucinations.

---

## Project Structure

```
Agentic-GraphRAG/
├── agent/
│   ├── rag_agent.py           # Self-correcting retrieve → grade → retry → synthesize loop
│   └── contradiction.py       # Sentence-level Big-O claim extractor + LLM conflict verifier
├── retrieval/
│   ├── query_router.py        # 5-way deterministic query classifier + pattern alias engine
│   ├── hybrid_retriever.py    # Joint Neo4j + FAISS orchestrator & context fusion
│   ├── graph_retriever.py     # Cypher traversal engine + InMemoryGraphRetriever fallback
│   ├── vector_store.py        # FAISS IndexFlatL2 builder, chunker, and metadata persistence
│   └── vector_retriever.py    # Filtered vector search interface (prose vs. code, pattern filters)
├── ingestion/
│   ├── load_graph.py          # Neo4j schema constraints, indexes, and canonical seed graph
│   ├── scrape.py              # Editorial loader and normalizer
│   ├── parse_cpp.py           # C++ code block extractor and prose splitter
│   ├── extract_entities.py    # Gemini JSON entity extractor + deterministic rules fallback
│   └── pipeline.py            # End-to-end ingestion & deduplication pipeline
├── llm/
│   └── gemini.py              # Google GenAI client with exponential backoff & Streamlit secrets support
├── eval/
│   ├── test_queries.json      # 30 labeled benchmark queries across all 5 query types
│   ├── eval_harness.py        # Automated evaluation runner & 3-way ablation benchmark
│   └── eval_report.json       # Latest serialized evaluation & ablation results
├── tests/
│   └── test_graphrag.py       # Automated unit & integration tests (router, RRF, graph, grader)
├── data/
│   ├── editorials.json        # 20 curated Codeforces & LeetCode editorials with C++ solutions
│   ├── extractions.json       # Extracted patterns, relationships, and code templates
│   └── vector_index/          # Persisted 68-chunk FAISS + BM25 index & metadata
├── app.py                     # Interactive Streamlit UI with subgraph visualization & ablation mode switcher
├── docker-compose.yml         # Local Neo4j 5.23 + APOC container configuration
└── requirements.txt           # Python dependencies
```

---

## Quick Start

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/nomulabhargavachaitanya-gif/Agentic-GraphRAG.git
cd Agentic-GraphRAG
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and add your free Google AI Studio key:

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=your_gemini_api_key_here
# Optional override (defaults to gemini-3.5-flash-lite):
# GEMINI_MODEL=gemini-3.5-flash-lite
```

*(If `GEMINI_API_KEY` is omitted, the entire pipeline still runs offline using rule-based entity extraction and structured context responses.)*

### 3. Start Neo4j & Load the Knowledge Graph (Optional for Local Graph Mode)

If Docker is installed, start the Neo4j container and populate the graph:

```bash
docker compose up -d
python -m ingestion.load_graph
python -m ingestion.pipeline --mode llm     # or --mode rules for offline extraction
python -m retrieval.vector_store            # rebuilds data/vector_index/
```

- **Neo4j Browser**: `http://localhost:7474` (credentials: `neo4j` / `graphrag_password`)
- **Bolt URI**: `bolt://localhost:7687`
- *Note*: If Docker is not running, the system automatically falls back to `InMemoryGraphRetriever` using the pre-built `data/extractions.json` and `data/vector_index/`.

### 4. Launch the Web UI

```bash
streamlit run app.py
```

Open **`http://localhost:8501`** in your browser to test queries, inspect relevance/sufficiency/coverage scores, view the active retrieval path, and expand source attributions or contradiction analyses.

### 5. Run CLI Demos & Evaluation Suite

```bash
# Test query classification across sample questions
python -m retrieval.query_router

# Run the self-correcting RAG agent on core benchmark queries
python -m agent.rag_agent

# Scan the corpus for conflicting Big-O complexity claims
python -m agent.contradiction

# Run the automated unit & integration test suite
python -m unittest discover -s tests -v

# Run the full 30-query benchmark + 3-way ablation evaluation harness
python -m eval.eval_harness
```

---

## Cloud Deployment (Streamlit Community Cloud / Hugging Face Spaces)

**Live App**: **[https://agentic-graphrag.streamlit.app/](https://agentic-graphrag.streamlit.app/)**

The repository is pre-configured for zero-setup cloud deployment:
- `data/vector_index/` and `data/extractions.json` are committed to the repository (and `app.py` automatically rebuilds the FAISS index on boot if ever absent).
- When no external Neo4j server is configured, `HybridRetriever` seamlessly falls back to `InMemoryGraphRetriever`.
- **To deploy on [Streamlit Community Cloud](https://share.streamlit.io)**:
  1. Create a new app pointing to `nomulabhargavachaitanya-gif/Agentic-GraphRAG` (`main` branch, `app.py`).
  2. Under **Advanced settings → Secrets**, add:
     ```toml
     GEMINI_API_KEY = "your_gemini_api_key"
     ```
     *(Optional: add `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` to connect a free Neo4j AuraDB cloud instance.)*

---

## Author

**N Bhargava** — B.Tech in Data Science & Artificial Intelligence (DSAI), IIIT Bangalore
