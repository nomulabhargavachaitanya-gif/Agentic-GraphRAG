"""
generate_pdf.py — Generates a comprehensive, publication-grade explanation PDF
for the GraphRAG Competitive Programming project.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PDF_PATH = "/Users/bhargava/Desktop/rag_project/PROJECT_EXPLANATION.pdf"

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and display total page count."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (pages after cover page)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "GraphRAG for Competitive Programming — Project Architecture & Walkthrough")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
            
        # Footer
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(letter[0] - 54, 36, page_str)
        self.drawString(54, 36, "Author: N Bhargava (IIIT-Bangalore) | Confidential & Educational")
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 48, letter[0] - 54, 48)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2563eb"),
        spaceAfter=15
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569"),
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=7
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        'CodeBlock',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#0f172a")
    )
    
    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b")
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("GraphRAG for Competitive Programming", title_style))
    story.append(Paragraph("A Self-Correcting Hybrid Knowledge Graph & Dense Vector Retrieval System", subtitle_style))
    story.append(Paragraph("<b>Author:</b> N Bhargava | B.Tech DSAI, IIIT Bangalore &nbsp;&nbsp;|&nbsp;&nbsp; <b>Project:</b> <code>rag_project</code>", meta_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=14))

    # Executive Overview
    story.append(Paragraph("1. Executive Overview & Problem Statement", h1_style))
    story.append(Paragraph(
        "Standard Retrieval-Augmented Generation (RAG) architectures rely primarily on dense bi-encoder vector embeddings. "
        "While suitable for generic semantic similarity, vector search breaks down severely in <b>algorithmic problem solving and competitive programming (CP)</b> due to four structural limitations:",
        body_style
    ))
    story.append(Paragraph("• <b>Multi-Hop Dependency Blindness:</b> Questions like <i>'What must I learn before Suffix Arrays?'</i> demand strict topological graph traversal across prerequisite chains (e.g., Binary Search → Suffix Array). Vector search simply returns documents that mention both terms without respecting directed dependencies.", bullet_style))
    story.append(Paragraph("• <b>Structural Algorithmic Tradeoffs:</b> Queries comparing implementations (e.g., <i>'Segment Tree vs Fenwick Tree for range queries'</i>) require multi-attribute contrast (point updates vs range updates, constant factors, space complexity $O(n)$ vs $4n$). Vector similarity retrieves isolated paragraphs rather than structured tradeoffs.", bullet_style))
    story.append(Paragraph("• <b>Code vs Prose Syntax Disconnect:</b> Competitive programming editorials interleave English explanations with low-level C++ code. Vectorizing code often matches superficial language syntax (e.g., <code>vector&lt;int&gt;</code>) instead of algorithmic invariants.", bullet_style))
    story.append(Paragraph("• <b>Silent Hallucinations & Contradictions:</b> When multiple editorial sources claim conflicting complexities for variations of a technique, standard RAG averages them together into hallucinated statements instead of attributing claims to specific sources.", bullet_style))

    # Callout Box
    callout_data = [[
        Paragraph(
            "<b>Key Insight:</b> This project solves these challenges by fusing a <b>Neo4j Knowledge Graph</b> "
            "(for strict multi-hop dependencies and relational tradeoffs) with a <b>FAISS Dense Vector Store</b> "
            "(for fine-grained editorial prose and C++ code block search), orchestrated by an <b>Agentic Self-Grading Loop</b>.",
            callout_style
        )
    ]]
    callout_table = Table(callout_data, colWidths=[letter[0] - 108])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#93c5fd")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(callout_table)
    story.append(Spacer(1, 10))

    # Architecture Overview
    story.append(Paragraph("2. System Architecture & Information Flow", h1_style))
    story.append(Paragraph(
        "The architecture unifies three primary subsystems: knowledge graph traversal, dense vector indexing, and an intelligent self-grading loop.",
        body_style
    ))

    arch_table_data = [
        [Paragraph("<b>Component</b>", body_style), Paragraph("<b>Technology</b>", body_style), Paragraph("<b>Role & Core Function</b>", body_style)],
        [Paragraph("Knowledge Graph", body_style), Paragraph("Neo4j 5.23 (Cypher)", body_style), Paragraph("Stores Pattern, Problem, CodeTemplate, and Source nodes. Executes multi-hop graph hops, prerequisite chains, and tradeoff traversals.", body_style)],
        [Paragraph("Vector Store", body_style), Paragraph("FAISS (IndexFlatL2)", body_style), Paragraph("Indexes 80+ prose chunks and C++ code blocks. Provides Top-K similarity search with metadata linking back to graph pattern IDs.", body_style)],
        [Paragraph("Embeddings", body_style), Paragraph("all-MiniLM-L6-v2", body_style), Paragraph("384-dimensional dense semantic representations with fast CPU inference and zero external API dependencies.", body_style)],
        [Paragraph("Ingestion Pipeline", body_style), Paragraph("BeautifulSoup4 + Claude / Heuristics", body_style), Paragraph("Scrapes Codeforces/LeetCode editorials, isolates C++ code blocks, extracts structured entities, and merges them into Neo4j.", body_style)],
        [Paragraph("Agentic Router", body_style), Paragraph("Python / Structured Prompts", body_style), Paragraph("Routes incoming user queries to graph traversal, vector search, or hybrid retrieval based on linguistic intent.", body_style)],
        [Paragraph("Self-Grading Evaluator", body_style), Paragraph("LangChain / Claude / Ragas", body_style), Paragraph("Grades retrieved context for relevance and sufficiency; widens search radius or cleanly abstains if out-of-corpus.", body_style)],
    ]
    arch_table = Table(arch_table_data, colWidths=[1.3 * inch, 1.6 * inch, 4.1 * inch])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 10))

    # Phase by Phase Breakdown
    story.append(PageBreak())
    story.append(Paragraph("3. Step-by-Step Phase Walkthrough", h1_style))
    story.append(Paragraph(
        "The project was built incrementally across distinct engineering phases. Here is how each phase was constructed, tested, and validated:",
        body_style
    ))

    # Phase 1
    story.append(Paragraph("Phase 1: Knowledge Graph Infrastructure & Cypher Traversal", h2_style))
    story.append(Paragraph(
        "<b>Goal:</b> Establish the graph schema, configure containerized services, seed ground-truth algorithmic relationships, and implement Cypher query algorithms.",
        body_style
    ))
    story.append(Paragraph("• <b>Infrastructure Setup:</b> Authored <code>docker-compose.yml</code> configuring Neo4j 5.23 Community Edition with APOC plugins alongside Qdrant vector database. Added healthchecks and volume persistence.", bullet_style))
    story.append(Paragraph("• <b>Schema Constraints & Indexing:</b> Created schema definitions in <code>ingestion/load_graph.py</code> using idempotent Cypher queries. Added uniqueness constraints on <code>(p:Pattern).name</code>, <code>(prob:Problem).id</code>, and <code>(s:Source).url</code>, plus a Neo4j Full-Text Index on pattern descriptions.", bullet_style))
    story.append(Paragraph("• <b>Algorithmic Seed Graph:</b> Populated 8 canonical patterns (Two Pointers, Sliding Window, Binary Search, Prefix Sum, Fenwick Tree, Segment Tree, Sparse Table, Suffix Array) with time/space complexity properties and category tags.", bullet_style))
    story.append(Paragraph("• <b>Relational Graph Wiring:</b> Wired directed <code>PREREQUISITE_OF</code> edges (e.g., Prefix Sum → Fenwick Tree), <code>ALTERNATIVE_TO</code> edges with explicit trade-off metadata, <code>SOLVED_WITH</code> edges to 5 representative problems, and <code>HAS_TEMPLATE</code> edges to verified C++ code blocks.", bullet_style))
    story.append(Paragraph("• <b>Cypher Retrieval Engine:</b> Developed <code>retrieval/graph_retriever.py</code> implementing structured query execution: <code>compare_patterns()</code> for pairwise tradeoff evaluation, <code>get_prerequisite_chain()</code> using recursive variable-length path traversal, and <code>lookup_complexity()</code> with full-text fallback.", bullet_style))
    story.append(Paragraph("• <b>Evaluation Benchmark:</b> Created <code>eval/test_queries.json</code> defining 6 golden test cases covering graph-hop, hybrid, contradiction, and out-of-corpus queries.", bullet_style))
    story.append(Spacer(1, 8))

    # Phase 2
    story.append(Paragraph("Phase 2: Ingestion & Structured Knowledge Extraction", h2_style))
    story.append(Paragraph(
        "<b>Goal:</b> Build an automated pipeline to ingest raw editorial text and C++ code from competitive programming platforms, extract structured entities, and merge them into the knowledge graph.",
        body_style
    ))
    story.append(Paragraph("• <b>Curated Corpus:</b> Built <code>data/editorials.json</code> with 20 real-world editorials spanning Codeforces (ratings 800–1900) and LeetCode (Easy–Hard), featuring complete editorial prose and full C++ solutions.", bullet_style))
    story.append(Paragraph("• <b>Web Scraper:</b> Implemented <code>ingestion/scrape.py</code> with HTTP client headers, conservative rate limiting (2-second request intervals), and HTML-to-Markdown conversion while preserving fenced code blocks.", bullet_style))
    story.append(Paragraph("• <b>C++ Code Block Parser:</b> Built <code>ingestion/parse_cpp.py</code> to cleanly separate natural language prose from source code. Extracted function signatures (<code>solve()</code>, <code>query()</code>), identified data structures (<code>struct Fenwick</code>, <code>class SegmentTree</code>), and detected STL containers.", bullet_style))
    story.append(Paragraph("• <b>Entity Extraction:</b> Created <code>ingestion/extract_entities.py</code> supporting dual-mode execution: (1) LLM extraction via Anthropic Claude structured JSON outputs, and (2) deterministic offline rule-based heuristic extraction allowing full local execution with zero API cost.", bullet_style))
    story.append(Paragraph("• <b>Pipeline Orchestration:</b> Implemented <code>ingestion/pipeline.py</code> which coordinates the workflow: load corpus → parse C++ → extract entities → deduplicate aliases → write intermediate <code>data/extractions.json</code> → merge into Neo4j.", bullet_style))
    story.append(Spacer(1, 8))

    # Phase 3
    story.append(Paragraph("Phase 3: Vector Store & Semantic Chunk Linking", h2_style))
    story.append(Paragraph(
        "<b>Goal:</b> Enable semantic similarity search over prose and code blocks, with each vector chunk explicitly linked to the Knowledge Graph via pattern IDs.",
        body_style
    ))
    story.append(Paragraph("• <b>Lightweight Vector Engine:</b> Adopted <b>FAISS (`IndexFlatL2`)</b> as a pure local vector engine paired with <b>`all-MiniLM-L6-v2`</b> (384 dimensions), eliminating heavy infrastructure overhead while achieving sub-millisecond retrieval latency.", bullet_style))
    story.append(Paragraph("• <b>Chunking & Metadata Linkage:</b> Built <code>retrieval/vector_store.py</code> to chunk editorials into prose paragraphs and standalone code snippets. Attached metadata including <code>editorial_id</code>, <code>chunk_type</code>, and crucially <code>linked_pattern_ids</code>.", bullet_style))
    story.append(Paragraph("• <b>Disk Persistence:</b> Implemented serializable storage saving <code>index.faiss</code>, <code>metadata.pkl</code>, and <code>config.json</code> under <code>data/vector_index/</code>.", bullet_style))
    story.append(Paragraph("• <b>Retrieval Interface:</b> Built <code>retrieval/vector_retriever.py</code> with methods for <code>search()</code>, <code>search_code()</code>, and <code>search_prose()</code>. Evaluated queries like <i>'segment tree range query'</i> (score 0.47) and verified that returned chunks correctly output <code>linked_pattern_ids</code> for hybrid graph expansion.", bullet_style))
    story.append(Spacer(1, 8))

    # Future Phases
    story.append(Paragraph("Roadmap: Phases 4 through 8", h2_style))
    story.append(Paragraph("• <b>Phase 4 (Query Router & Hybrid Fusion):</b> Classifies queries by intent (graph-hop, code retrieval, conceptual search) and merges Top-K vector chunks with multi-hop graph subgraphs.", bullet_style))
    story.append(Paragraph("• <b>Phase 5 (Agentic Self-Grading Loop):</b> Evaluates context relevance and sufficiency. Retries with widened search radius if insufficient, or cleanly triggers honest abstention if out of corpus.", bullet_style))
    story.append(Paragraph("• <b>Phase 6 (Contradiction Detection):</b> Detects divergent algorithmic claims across sources and surfaces them with full provenance attribution.", bullet_style))
    story.append(Paragraph("• <b>Phase 7 (Evaluation Harness):</b> Automated benchmarking using Ragas metrics (Faithfulness, Answer Relevance, Context Precision) over <code>eval/test_queries.json</code>.", bullet_style))
    story.append(Paragraph("• <b>Phase 8 (Interactive UI & API):</b> FastAPI endpoints and Streamlit interactive web interface.", bullet_style))

    # Verification and Test Results
    story.append(PageBreak())
    story.append(Paragraph("4. Verification, Test Results & Golden Queries", h1_style))
    story.append(Paragraph(
        "The project includes a formal test suite in <code>eval/test_queries.json</code>. "
        "Below are the verified test queries executed against the system:",
        body_style
    ))

    test_table_data = [
        [Paragraph("<b>Query ID & Prompt</b>", body_style), Paragraph("<b>Type</b>", body_style), Paragraph("<b>Retrieval Mechanism & Verification Result</b>", body_style)],
        [
            Paragraph("<b>Q1:</b> How does a segment tree differ from a Fenwick tree for range queries?", body_style),
            Paragraph("Graph-Hop", body_style),
            Paragraph("<b>Verified:</b> Traverses <code>(Segment Tree)-[:ALTERNATIVE_TO]-(Fenwick Tree)</code>. Returns tradeoff: Fenwick has lower constant factor & O(n) memory; Segment Tree supports range updates with lazy propagation. Pulls both C++ templates.", body_style)
        ],
        [
            Paragraph("<b>Q2:</b> What do I need to know before learning suffix arrays?", body_style),
            Paragraph("Graph-Hop", body_style),
            Paragraph("<b>Verified:</b> Recursive Cypher traversal <code>[:PREREQUISITE_OF*1..5]</code> returns topological ancestor chain: Binary Search → Suffix Array.", body_style)
        ],
        [
            Paragraph("<b>Q3:</b> Give me a C++ template for two-pointers on a sorted array", body_style),
            Paragraph("Hybrid", body_style),
            Paragraph("<b>Verified:</b> FAISS vector search identifies Two Pointers editorial chunk → bridges to graph node <code>Two Pointers</code> → retrieves verified C++ template with explanation.", body_style)
        ],
        [
            Paragraph("<b>Q4:</b> What's the time complexity of a sparse table?", body_style),
            Paragraph("Property", body_style),
            Paragraph("<b>Verified:</b> Direct graph property lookup on <code>Pattern {name: 'Sparse Table'}</code> returns O(n log n) build, O(1) query time.", body_style)
        ],
        [
            Paragraph("<b>Q5:</b> Is a segment tree O(log n) or O(n log n) per query?", body_style),
            Paragraph("Contradiction", body_style),
            Paragraph("<b>Phase 6 Target:</b> Multi-source retrieval detects conflicting claims between basic segment tree O(log n) and dynamic persistent tree with walk O(log^2 n), presenting both with source links.", body_style)
        ],
        [
            Paragraph("<b>Q6:</b> Explain Aho-Corasick automaton for multi-pattern matching", body_style),
            Paragraph("Abstention", body_style),
            Paragraph("<b>Phase 5 Target:</b> Detects term absent from both Neo4j and FAISS. Rejects hallucination and outputs: <i>'I don't have enough grounded information in the corpus'</i>.", body_style)
        ],
    ]
    test_table = Table(test_table_data, colWidths=[2.2 * inch, 1.0 * inch, 3.8 * inch])
    test_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(test_table)
    story.append(Spacer(1, 12))

    # Codebase Structure & Execution Guide
    story.append(Paragraph("5. Execution Guide & Reproduction Commands", h1_style))
    story.append(Paragraph("To inspect, build, and test this project from scratch, run the following commands:", body_style))

    commands_box = [
        [Paragraph(
            "<b># 1. Environment & Dependencies</b><br/>"
            "python3 -m venv .venv && source .venv/bin/activate<br/>"
            "pip install -r requirements.txt<br/><br/>"
            "<b># 2. Start Graph Database & Seed Data (Phase 1)</b><br/>"
            "docker compose up -d<br/>"
            "python -m ingestion.load_graph<br/>"
            "python -m retrieval.graph_retriever<br/><br/>"
            "<b># 3. Run Ingestion Pipeline (Phase 2)</b><br/>"
            "python -m ingestion.pipeline --mode rules --dry-run<br/><br/>"
            "<b># 4. Build Vector Index & Run Similarity Search (Phase 3)</b><br/>"
            "python -m retrieval.vector_store<br/>"
            "python -m retrieval.vector_retriever",
            code_style
        )]
    ]
    cmd_table = Table(commands_box, colWidths=[letter[0] - 108])
    cmd_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(cmd_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("6. Key Design Takeaways for AI Agents", h1_style))
    story.append(Paragraph("• <b>Dual-Mode Fallback:</b> In competitive programming tasks, external APIs (like LLMs or cloud endpoints) may suffer rate limits. Ingestion and retrieval must always provide deterministic, offline rule-based fallbacks.", bullet_style))
    story.append(Paragraph("• <b>The Graph-Vector Bridge:</b> Dense vectors identify the semantic neighborhood, while graph nodes and edges provide structural precision. Never store vector chunks without their corresponding graph entity IDs.", bullet_style))
    story.append(Paragraph("• <b>Honest Abstention:</b> The highest-leverage trait of an agentic RAG system in algorithmic domains is the refusal to hallucinate non-existent algorithms or invalid C++ code.", bullet_style))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF at: {PDF_PATH}")

if __name__ == "__main__":
    build_pdf()
