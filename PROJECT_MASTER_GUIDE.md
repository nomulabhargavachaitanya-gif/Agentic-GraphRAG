# GraphRAG for Competitive Programming: Complete Project Architecture & Master Guide

> **Target Audience**: AI Agents & Engineers  
> **Author**: N Bhargava (B.Tech DSAI, IIIT Bangalore)  
> **Repository**: `rag_project`  
> **Core Concept**: Agentic self-correcting hybrid retrieval (Knowledge Graph + Dense Vector Search) over algorithmic problems, editorial prose, and C++ code implementations.

---

## 1. Executive Summary & Vision

### 1.1 The Problem
Standard Retrieval-Augmented Generation (RAG) relies almost entirely on dense vector similarity (e.g., bi-encoder embeddings). While effective for general text search, dense embeddings catastrophic fail when applied to **competitive programming (CP) and algorithmic problem solving**:
1. **Multi-Hop Dependency Blindness**: An algorithmic query like *"What do I need to master before learning Suffix Arrays?"* requires traversing explicit prerequisite dependency chains (`Binary Search` → `Suffix Array`, `Prefix Sum` → `Fenwick Tree`). Vector embeddings retrieve conceptually similar strings but cannot reliably follow directed acyclic dependency graphs (DAGs).
2. **Structural Tradeoff Confusion**: Queries comparing two algorithms (e.g., *"How does a Segment Tree differ from a Fenwick Tree for range queries?"*) require crisp trade-off dimensions (point update vs range update, constant factor overhead, implementation complexity, memory footprint $O(n)$ vs $4n$). Pure vector search retrieves scattered paragraphs without structured contrast.
3. **Code Syntax vs. Semantic Semantic Disconnect**: Algorithmic editorials interleave high-level prose with low-level C++ code blocks. Vectorizing raw code often matches superficial syntax (e.g., `vector<int>`, `for` loops) rather than algorithmic invariants or data structure templates.
4. **Hallucination & Silent Contradiction**: Different platforms or problem variants often claim conflicting complexities (e.g., segment tree with lazy propagation $O(\log n)$ vs dynamic segment tree $O(q \log C)$). Standard RAG blurs contradictory context into a misleading hallucinated synthesis.

### 1.2 The GraphRAG Solution
This project implements an **Agentic GraphRAG System**:
- **Knowledge Graph (Neo4j)**: Represents discrete algorithmic entities (`Pattern`, `Problem`, `CodeTemplate`, `Source`) connected by semantically strict relationships (`PREREQUISITE_OF`, `ALTERNATIVE_TO`, `SAME_COMPLEXITY_CLASS`, `SOLVED_WITH`, `HAS_TEMPLATE`, `EXTRACTED_FROM`).
- **Dense Vector Store (FAISS + all-MiniLM-L6-v2)**: Stores chunked editorial prose and C++ code blocks with attached metadata linking directly back to graph `pattern_id`s.
- **Hybrid Query Router**: Classifies queries into Graph-Hop, Vector-Semantic, Hybrid, or Contradiction-Detection paths.
- **Agentic Self-Grading Loop**: Retrieves context, grades relevance and sufficiency, detects factual contradictions, widens hops on failure, and abstains when information is absent.

---

## 2. System Architecture & Component Interaction

```
                              +--------------------------+
                              |        User Query        |
                              +-------------+------------+
                                            |
                                            v
                              +--------------------------+
                              |   Query Router / Agent   |
                              +-------------+------------+
                                            |
                  +-------------------------+-------------------------+
                  |                                                   |
                  v                                                   v
     +--------------------------+                        +--------------------------+
     |   Graph Retriever        |                        |    Vector Retriever      |
     |   (Cypher / Neo4j)       |                        |    (FAISS / MiniLM-L6)   |
     +------------+-------------+                        +------------+-------------+
                  |                                                   |
                  | Multi-hop traversal                               | Top-K dense search
                  | Prerequisite chains                               | Prose & C++ chunks
                  | Trade-off edges                                   | Pattern IDs metadata
                  |                                                   |
                  +-------------------------+-------------------------+
                                            |
                                            v
                              +--------------------------+
                              |     Hybrid Fusion        |
                              | Context Assembler        |
                              +-------------+------------+
                                            |
                                            v
                              +--------------------------+
                              |    Self-Grading Agent    | <----+
                              | (Relevance & Sufficiency)|      | Loop retry /
                              +-------------+------------+      | Fallback widen
                                            |                   |
                        +-------------------+-------------------+
                        |                   |                   |
                   [Sufficient]       [Contradiction]     [Out of Scope]
                        |                   |                   |
                        v                   v                   v
                 Structured Answer      Attributed        Honest Abstention
                 + Verified Code        Contradiction     ("Not in corpus")
```

---

## 3. Data Model & Graph Schema

### 3.1 Node Types
1. **`Pattern`**: An algorithmic design pattern or core technique.
   - Properties: `name` (Unique string), `description` (Text), `time_complexity` (String), `space_complexity` (String), `category` (Data Structures, Strings, DP, etc.).
2. **`Problem`**: A specific competitive programming problem from LeetCode or Codeforces.
   - Properties: `id` (Unique string, e.g., `CF-1547B`, `LC-3`), `title` (String), `source` (`codeforces` | `leetcode`), `url` (String), `difficulty` (String/Rating).
3. **`CodeTemplate`**: Canonical, battle-tested C++ implementations of an algorithm.
   - Properties: `id` (String), `code` (C++ source), `language` (`cpp`), `explanation` (Text).
4. **`Source`**: Provenance of the content (editorial blog, tutorial, author).
   - Properties: `url` (Unique string), `author` (String), `date` (ISO date).

### 3.2 Relationship Types
- `(p1:Pattern)-[:PREREQUISITE_OF {strength, reason}]->(p2:Pattern)`: Directed dependency (e.g., `Prefix Sum` $\to$ `Fenwick Tree`).
- `(p1:Pattern)-[:ALTERNATIVE_TO {tradeoff, when_to_use}]->(p2:Pattern)`: Direct algorithmic comparison with documented trade-offs.
- `(p1:Pattern)-[:SAME_COMPLEXITY_CLASS]->(p2:Pattern)`: Equivalence class grouping for complexity searches.
- `(prob:Problem)-[:SOLVED_WITH {primary: bool}]->(p:Pattern)`: Links problem instances to applicable patterns.
- `(p:Pattern)-[:HAS_TEMPLATE]->(t:CodeTemplate)`: Links algorithmic pattern to verified C++ code.
- `(t:CodeTemplate)-[:EXTRACTED_FROM]->(s:Source)`: Full provenance link for auditability.
- `(s1:Source)-[:CONTRADICTS {claim}]->(s2:Source)`: Phase 6 contradiction edge.

---

## 4. Phase-by-Phase Construction Walkthrough

### Phase 1: Infrastructure, Graph Schema & Cypher Traversal
- **Objective**: Establish the persistent database infrastructure, define schema constraints, seed ground-truth algorithmic knowledge, and implement deterministic Cypher query traversal.
- **Tools & Tech**:
  - `Docker Compose`: Neo4j 5.23 Community Edition with APOC plugins, Qdrant vector database.
  - `Python Neo4j Driver` (`neo4j==5.25.0`), `python-dotenv`.
- **Key Modules Built**:
  - `docker-compose.yml`: Spawns Neo4j on port 7474 (HTTP) / 7687 (Bolt) and Qdrant on port 6333 / 6334.
  - `ingestion/load_graph.py`: Idempotent graph seeder using `MERGE` statements.
    - Creates schema uniqueness constraints on `Pattern.name`, `Problem.id`, `Source.url`.
    - Creates Neo4j Full-Text Index on `(p:Pattern)` across `[p.description, p.name]`.
    - Seeds 8 fundamental algorithmic patterns: Two Pointers, Sliding Window, Binary Search, Prefix Sum, Fenwick Tree, Segment Tree, Sparse Table, Suffix Array.
    - Wires explicit graph edges: `PREREQUISITE_OF`, `ALTERNATIVE_TO`, `SOLVED_WITH`, `HAS_TEMPLATE`, `EXTRACTED_FROM`.
  - `retrieval/graph_retriever.py`: Cypher execution engine supporting:
    - `compare_patterns(pattern_a, pattern_b)`: Queries `ALTERNATIVE_TO` edges and retrieves C++ templates for both sides.
    - `get_prerequisite_chain(target_pattern)`: Recursive Cypher query (`[:PREREQUISITE_OF*1..5]`) returning the full topological learning path.
    - `lookup_complexity(pattern_name)`: Direct node property lookup with fallback to full-text search.
  - `eval/test_queries.json`: Test suite of 6 canonical query types verifying multi-hop, hybrid, contradiction, and abstention behavior.

### Phase 2: Ingestion & Knowledge Extraction Pipeline
- **Objective**: Build an automated pipeline that ingests raw editorial text and C++ code, extracts structured algorithmic entities and relationships, and merges them into the knowledge graph.
- **Tools & Tech**:
  - `BeautifulSoup4`, `httpx` for scraping and HTML conversion.
  - Custom C++ parsing engine using regular expressions and syntax heuristics.
  - `Anthropic Claude API` (`claude-3-5-sonnet`) with structured JSON schema for entity extraction.
  - Offline Rule-based heuristic extraction fallback (allowing local execution without API keys).
- **Key Modules Built**:
  - `data/editorials.json`: Curated corpus of 20 high-quality editorials from Codeforces (rated 800–1900) and LeetCode (Easy–Hard).
  - `ingestion/scrape.py`: Rate-limited scraper for Codeforces blog posts (`--entry`, `--entries`, `--local`). Converts HTML to markdown while preserving code fences.
  - `ingestion/parse_cpp.py`: Parser that cleanly isolates prose narrative from C++ code blocks. Analyzes function signatures (`solve()`, `query()`, `update()`), data structures (`struct Fenwick`, `class SegmentTree`), and STL usage (`std::vector`, `std::unordered_map`).
  - `ingestion/extract_entities.py`: Extracts structured entities (`ExtractedPattern`, `ExtractedRelationship`, `ExtractedTemplate`). Deduplicates aliases (e.g., "Binary Indexed Tree" $\to$ "Fenwick Tree").
  - `ingestion/pipeline.py`: End-to-end orchestrator with CLI flags `--mode rules|llm` and `--dry-run`. Ingests 20 editorials, yielding 12 canonical patterns, 20 C++ templates, and relational graph links saved to `data/extractions.json` and inserted into Neo4j.

### Phase 3: Vector Store & Semantic Chunk Linking
- **Objective**: Implement dense vector search over editorial prose and code chunks, linked via metadata to the Knowledge Graph nodes for seamless hybrid retrieval.
- **Design Decision**: Because local Docker availability may vary, we implemented **FAISS (`faiss-cpu`)** as a lightweight, zero-infra, zero-latency vector engine paired with the **`all-MiniLM-L6-v2`** embedding model (384 dimensions) from `sentence-transformers`.
- **Key Modules Built**:
  - `retrieval/vector_store.py`:
    - Chunks editorials into prose segments and analyzed C++ code blocks.
    - Embeds text using `SentenceTransformer('all-MiniLM-L6-v2')`.
    - Builds a FAISS `IndexFlatL2` index with 80+ vectors.
    - Attaches rich metadata to each vector: `chunk_id`, `chunk_type` (`prose` vs `code`), `editorial_id`, `problem_id`, and crucially `linked_pattern_ids`.
    - Implements disk persistence (`index.faiss`, `metadata.pkl`, `config.json`).
  - `retrieval/vector_retriever.py`:
    - Clean retrieval abstraction providing `search(query, k, pattern_filter, chunk_type)`.
    - Helper methods: `search_for_pattern_entry()`, `search_code()`, `search_prose()`.
    - Generates `linked_pattern_ids` from vector hits to serve as seeds for subsequent graph traversal.

### Phase 4: Query Router & Hybrid Retrieval ✅
- **Objective**: Given an incoming natural language prompt, classify the intent and orchestrate joint retrieval across the vector index and knowledge graph.
- **Key Modules Built**:
  - `retrieval/query_router.py`: Deterministic keyword-based query classifier supporting 5 query types (`GRAPH_HOP`, `CODE_TEMPLATE`, `SEMANTIC`, `HYBRID`, `CONTRADICTION`). Detects algorithmic patterns via alias mapping, identifies comparison pairs (A vs B), and routes prerequisite queries. Zero LLM cost — fully rule-based.
  - `retrieval/hybrid_retriever.py`: Orchestrates joint retrieval across Neo4j and FAISS based on the router's classification. Implements 5 dispatch strategies:
    1. **Graph-Hop**: Cypher traversal for prerequisites and comparisons, supplemented by vector results.
    2. **Code Template**: Graph template lookup + vector code search + prose explanations.
    3. **Semantic**: Pure FAISS similarity search for general queries.
    4. **Hybrid**: Vector search → extract `linked_pattern_ids` → graph expansion.
    5. **Contradiction**: Multi-source retrieval with canonical complexity cross-referencing.
  - Graceful degradation: works in offline mode (no Neo4j) using vector-only retrieval.

### Phase 5: Self-Correcting Agentic Loop ✅
- **Objective**: Prevent hallucinations and verify grounding before returning answers to the user.
- **Key Modules Built**:
  - `agent/rag_agent.py`: Full agentic pipeline implementing the retrieve → grade → self-correct → synthesize loop.
  - **`ContextGrader`**: Heuristic grading engine (no LLM needed) evaluating three dimensions:
    - *Relevance* (0–1): keyword overlap + pattern matching between query and chunks.
    - *Sufficiency* (0–1): fraction of chunks meeting the relevance threshold.
    - *Coverage* (0–1): fraction of expected patterns found in retrieved context.
  - **Self-Correction Loop**: If overall score $< 0.6$, the agent automatically doubles $k$ (search width) and retries up to 2 times.
  - **Honest Abstention**: If overall score $< 0.25$ after retries (e.g., Q6: "Aho-Corasick automaton"), the agent returns: *"I do not have sufficient verified information in my corpus"* rather than fabricating code.
  - **`AnswerSynthesizer`**: If Anthropic API key is available, uses Claude for grounded natural-language synthesis with strict citation rules. Otherwise returns a structured context dump.

### Phase 6: Contradiction Detection Path ✅
- **Objective**: Handle conflicting information across multiple editorial sources.
- **Implementation**: Integrated into the query router (`QueryType.CONTRADICTION`) and hybrid retriever. When conflicting claims are retrieved (e.g., $O(N)$ vs $O(N \log N)$), the agent:
  1. Retrieves 2× more candidates than normal, filtered to the target pattern.
  2. Cross-references with the canonical complexity from the knowledge graph.
  3. Flags the discrepancy, presents both arguments, and attributes each to its exact source.

### Phase 7: Evaluation Harness & Benchmark Suite ✅
- **Objective**: Quantitative evaluation of retrieval accuracy.
- **Key Modules Built**:
  - `eval/eval_harness.py`: Runs the 6 benchmark queries through the full agentic pipeline and evaluates:
    - **Routing Accuracy**: Did the query router classify correctly?
    - **Pattern Recall**: Were the expected algorithmic patterns found?
    - **Code Hit Rate**: Did code queries return C++ templates?
    - **Abstention Precision**: Did out-of-corpus queries correctly trigger abstention?
  - Generates a structured `eval/eval_report.json` with per-query scores and summary metrics.

---

## 5. Technology Stack & Dependencies

| Category | Component / Library | Purpose in this Project |
| :--- | :--- | :--- |
| **Language** | Python 3.11 / 3.13 | Core runtime for ingestion, embedding, and retrieval |
| **Graph Database** | Neo4j 5.23 Community | Relational knowledge graph, Cypher traversals, constraints |
| **Vector Engine** | FAISS (`faiss-cpu 1.15.1`) | Local dense similarity search over chunks |
| **Embeddings** | `sentence-transformers 6.2.0` | `all-MiniLM-L6-v2` (384-dim dense semantic embeddings) |
| **LLM Orchestration** | Anthropic Claude API (`anthropic 0.34.2`) | Structured entity and relationship extraction |
| **Web Ingestion** | `httpx`, `beautifulsoup4` | Editorial scraping, HTML sanitation, markdown formatting |
| **Vector Storage (Alt)**| Qdrant 1.11 | Containerized vector database defined in `docker-compose.yml` |
| **Document Reporting** | `reportlab 5.1.0` | Automated PDF documentation and report generation |

---

## 6. Directory Map & File Index

```
rag_project/
├── .env                       # Environment variables (NEO4J_URI, ANTHROPIC_API_KEY)
├── docker-compose.yml         # Container configuration for Neo4j & Qdrant
├── requirements.txt           # Python package dependencies
├── README.md                  # Project overview and quickstart
├── PROJECT_MASTER_GUIDE.md    # Master AI and developer reference (this document)
├── PROJECT_EXPLANATION.pdf    # Typeset PDF documentation for presentation and study
├── data/
│   ├── editorials.json        # Curated corpus of 20 Codeforces & LeetCode editorials
│   ├── extractions.json       # Intermediate entity extraction output
│   └── vector_index/          # Persisted FAISS index files
│       ├── index.faiss        # Binary FAISS vector index
│       ├── metadata.pkl       # Serialized chunk metadata & pattern links
│       └── config.json        # Vector index configuration & dimension specs
├── ingestion/
│   ├── __init__.py
│   ├── scrape.py              # Editorial scraper and markdown formatter
│   ├── parse_cpp.py           # C++ code block parser (AST/regex analyzer)
│   ├── extract_entities.py    # Structured LLM + rule-based entity extractor
│   ├── load_graph.py          # Phase 1 Neo4j manual seed loader
│   └── pipeline.py            # Phase 2 end-to-end ingestion pipeline
├── retrieval/
│   ├── __init__.py
│   ├── graph_retriever.py     # Cypher query traversal engine
│   ├── vector_store.py        # FAISS vector store with embedding pipeline
│   ├── vector_retriever.py    # Vector search interface with pattern bridge
│   ├── query_router.py        # Phase 4 query classification engine
│   └── hybrid_retriever.py    # Phase 4 joint graph+vector retrieval
├── agent/                     # Phase 5 agentic self-grading loop
│   └── rag_agent.py           # Self-correcting RAG agent (grader + synthesizer + retry loop)
└── eval/
    ├── test_queries.json      # 6 benchmark test cases and expected outputs
    └── eval_harness.py        # Phase 7 evaluation harness and metrics
```

---

## 7. How to Reproduce & Run Every Component

### Step 1: Initialize Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Run Neo4j Graph Seeding (Phase 1)
```bash
# If running with Neo4j container:
docker compose up -d
python -m ingestion.load_graph

# Test graph queries:
python -m retrieval.graph_retriever
```

### Step 3: Run Ingestion Pipeline (Phase 2)
```bash
# Offline mode using deterministic rule-based extractor:
python -m ingestion.pipeline --mode rules --dry-run

# Or with live Neo4j database:
python -m ingestion.pipeline --mode rules
```

### Step 4: Build Vector Index & Run Similarity Search (Phase 3)
```bash
# Embeds the corpus and builds the FAISS index:
python -m retrieval.vector_store

# Test vector queries with pattern bridging:
python -c "
from retrieval.vector_retriever import VectorRetriever
retriever = VectorRetriever()
res = retriever.search('segment tree range minimum query', k=3)
for r in res.results:
    print(f'[{r.score:.3f}] {r.editorial_id}: {r.text[:80]}... Linked patterns: {r.linked_pattern_ids}')
"
```

### Step 5: Test Query Router (Phase 4)
```bash
# Classify benchmark queries into retrieval strategies:
python -m retrieval.query_router
```

### Step 6: Run the Full Agentic RAG (Phase 5)
```bash
# Runs the self-correcting retrieve → grade → retry → synthesize loop:
python -m agent.rag_agent
```

### Step 7: Run Evaluation Harness (Phase 7)
```bash
# Evaluate all 6 benchmark queries with metrics:
python -m eval.eval_harness
# Report saved to eval/eval_report.json
```

---

## 8. Summary for AI Agents Resuming or Modifying This Project
When an AI agent continues work on this codebase:
1. **Never break the Graph-to-Vector link**: Every chunk stored in `vector_store.py` must retain its `linked_pattern_ids` list. This is the crucial bridge allowing semantic search hits to seed multi-hop Cypher queries in Neo4j.
2. **Preserve Idempotency**: All graph ingestion in `load_graph.py` and `pipeline.py` uses Neo4j `MERGE` statements with explicit constraints.
3. **Respect Dual-Mode Operation**: Ingestion and retrieval should always support an offline, zero-API-cost fallback (`--mode rules`, FAISS) so testing does not block on external cloud API limits.
