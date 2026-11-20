"""Interactive Streamlit UI for the Agentic GraphRAG system.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Agentic GraphRAG", page_icon="🕸️", layout="wide")

# Bridge Streamlit Cloud secrets into os.environ before loading modules
try:
    for _k in ("GEMINI_API_KEY", "GEMINI_MODEL", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass

from agent.rag_agent import RAGAgent
from ingestion.load_graph import ALTERNATIVE_EDGES, PATTERNS, PREREQUISITE_EDGES
from retrieval.vector_store import INDEX_DIR, VectorStore

EXAMPLES = [
    "What are the prerequisites for a Fenwick Tree?",
    "What do I need to know before learning suffix arrays?",
    "Compare Segment Tree and Fenwick Tree",
    "Give me a C++ template for Segment Tree",
    "Is a segment tree O(log n) or O(n log n) per query?",
    "How does quantum annealing solve knapsack?",
]

MODE_LABELS = {
    "Full Agentic GraphRAG (Router + Graph + RRF + Grader)": "full_agentic",
    "Static Hybrid (Graph + RRF Vector, No Agentic Gate)": "hybrid_no_agent",
    "Vanilla Vector-Only Baseline (FAISS Top-k Only)": "vector_only",
}


@st.cache_resource(show_spinner="Initialising agent (knowledge graph + FAISS/BM25 RRF index)...")
def get_agent() -> RAGAgent:
    if not (INDEX_DIR / "index.faiss").exists():
        store = VectorStore()
        store.build_from_editorials()
        store.save(INDEX_DIR)
    return RAGAgent()


def build_subgraph_dot(active_patterns: list[str], full_graph: bool = False) -> str:
    """Build a Graphviz DOT specification for the active subgraph or full knowledge graph."""
    active_set = set(active_patterns)

    # Collect 1-hop and ancestor/descendant edges around active patterns
    edges_prereq: list[tuple[str, str]] = []
    edges_alt: list[tuple[str, str, str]] = []
    nodes: set[str] = set(active_set)

    if full_graph or not active_set:
        for p in PATTERNS:
            nodes.add(p["name"])
        edges_prereq = list(PREREQUISITE_EDGES)
        edges_alt = list(ALTERNATIVE_EDGES)
    else:
        # Include transitive prerequisites of active patterns
        frontier = set(active_set)
        for _ in range(4):
            next_frontier = set()
            for src, dst in PREREQUISITE_EDGES:
                if dst in frontier or src in frontier:
                    if (src, dst) not in edges_prereq:
                        edges_prereq.append((src, dst))
                    nodes.add(src)
                    nodes.add(dst)
                    if dst in frontier:
                        next_frontier.add(src)
            frontier = next_frontier
        for a, b, desc in ALTERNATIVE_EDGES:
            if a in active_set or b in active_set:
                edges_alt.append((a, b, desc))
                nodes.add(a)
                nodes.add(b)

    complexity_map = {p["name"]: p["time_complexity"] for p in PATTERNS}
    lines = [
        "digraph KnowledgeGraph {",
        '  rankdir=LR;',
        '  bgcolor="transparent";',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.18,0.10"];',
        '  edge [fontname="Helvetica", fontsize=9];',
    ]

    for n in sorted(nodes):
        cx = complexity_map.get(n, "")
        label = f"{n}\\n[{cx}]" if cx else n
        if n in active_set:
            lines.append(f'  "{n}" [label="{label}", fillcolor="#1e40af", fontcolor="white", color="#1e3a8a", penwidth=2.0];')
        else:
            lines.append(f'  "{n}" [label="{label}", fillcolor="#e0f2fe", fontcolor="#0f172a", color="#0284c7"];')

    for src, dst in edges_prereq:
        highlight = src in active_set and dst in active_set
        color = "#2563eb" if highlight else "#64748b"
        pw = "2.2" if highlight else "1.3"
        lines.append(f'  "{src}" -> "{dst}" [label="PREREQUISITE_OF", color="{color}", fontcolor="{color}", penwidth={pw}];')

    for a, b, _desc in edges_alt:
        highlight = a in active_set and b in active_set
        color = "#d97706" if highlight else "#94a3b8"
        pw = "2.2" if highlight else "1.3"
        lines.append(
            f'  "{a}" -> "{b}" [dir=both, style=dashed, label="ALTERNATIVE_TO", color="{color}", fontcolor="{color}", penwidth={pw}];'
        )

    lines.append("}")
    return "\n".join(lines)


def run_query(question_text: str, top_k: int, mode_key: str) -> None:
    if not question_text.strip():
        return
    try:
        agent = get_agent()
        with st.spinner("Routing, retrieving (Graph + FAISS/BM25 RRF), grading, synthesizing..."):
            st.session_state["last_response"] = agent.query(
                question_text.strip(), k=top_k, mode=mode_key
            )
            st.session_state["last_question"] = question_text.strip()
    except Exception as e:
        st.session_state["last_error"] = str(e)
        st.session_state.pop("last_response", None)


with st.sidebar:
    st.title("🕸️ Agentic GraphRAG")
    st.caption(
        "5-way deterministic query router + Neo4j knowledge graph + "
        "FAISS/BM25 Reciprocal Rank Fusion + self-grading loop, "
        "Big-O contradiction detection, and calibrated abstention."
    )
    mode_label = st.selectbox("Pipeline Mode (Ablation)", list(MODE_LABELS.keys()), index=0)
    selected_mode = MODE_LABELS[mode_label]
    k = st.slider("Initial Top-k chunks", 1, 15, 5)
    st.subheader("Sample Queries")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["question_input"] = ex
            st.session_state.pop("last_error", None)
            run_query(ex, k, selected_mode)

st.header("Competitive Programming Knowledge Assistant")

with st.form("query_form", clear_on_submit=False):
    col_in, col_btn = st.columns([5, 1], vertical_alignment="bottom")
    question = col_in.text_input(
        "Ask an algorithmic or C++ implementation question",
        key="question_input",
        placeholder="e.g. How does a segment tree differ from a Fenwick tree?",
    )
    submitted = col_btn.form_submit_button("Ask", type="primary", use_container_width=True)

if submitted and question.strip():
    st.session_state.pop("last_error", None)
    run_query(question, k, selected_mode)

if "last_error" in st.session_state:
    st.error(f"Agent failed: {st.session_state['last_error']}")
    st.info(
        "Make sure the vector index is built (`python -m retrieval.vector_store`)."
    )

resp = st.session_state.get("last_response")
if resp is not None:
    if resp.abstained:
        st.warning("Calibrated Abstention: topic not found in the verified corpus (zero hallucination gate).")
    if resp.contradiction_detected:
        st.error("Big-O Contradiction Detected: sources make conflicting complexity claims.")

    st.markdown(resp.answer)

    st.divider()
    g = resp.grading
    routing = resp.metadata.get("routing", {})
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Route", routing.get("type", "—"))
    c2.metric("Relevance", f"{g.relevance_score:.2f}")
    c3.metric("Sufficiency", f"{g.sufficiency_score:.2f}")
    c4.metric("Coverage", f"{g.coverage_score:.2f}")
    c5.metric("Retries", resp.retries_used)
    c6.metric("Mode", resp.metadata.get("mode", "full_agentic"))
    st.caption(f"Verdict: **{g.verdict}** — {g.reason} | Path: `{resp.retrieval_path}`")

    # Determine active graph patterns for visualization
    active_pats = list(routing.get("detected_patterns", []))
    for src in resp.sources:
        for p in src.get("patterns", []):
            if p not in active_pats:
                active_pats.append(p)

    if active_pats and not resp.abstained:
        with st.expander("🕸️ Knowledge Graph Subgraph Traversal", expanded=True):
            st.caption("Dark blue nodes indicate patterns matched/traversed for this query; solid arrows show `PREREQUISITE_OF` chains and dashed arrows show `ALTERNATIVE_TO` trade-offs.")
            st.graphviz_chart(build_subgraph_dot(active_pats[:4], full_graph=False), use_container_width=True)

    cx_rep = resp.metadata.get("contradiction")
    if cx_rep and cx_rep.sources:
        with st.expander("Contradiction Analysis", expanded=resp.contradiction_detected):
            st.markdown(cx_rep.to_markdown())

    with st.expander(f"Attributed Sources ({len(resp.sources)})"):
        st.json(resp.sources)

with st.expander("📊 3-Way Ablation Benchmark (30 Queries) & Full Knowledge Graph"):
    eval_path = Path(__file__).resolve().parent / "eval" / "eval_report.json"
    if eval_path.exists():
        with open(eval_path) as f:
            rep_data = json.load(f)
        ab = rep_data.get("ablation")
        if ab:
            st.markdown("#### Ablation Study: Vanilla Vector-Only vs. Static Hybrid vs. Full Agentic GraphRAG")
            rows = []
            label_map = {
                "vector_only": "1. Vanilla Vector-Only (FAISS Top-k)",
                "hybrid_no_agent": "2. Static Hybrid (Router + Graph + RRF)",
                "full_agentic": "3. Full Agentic GraphRAG (Ours)",
            }
            for k_mode, title in label_map.items():
                if k_mode in ab:
                    m = ab[k_mode]
                    rows.append({
                        "Configuration": title,
                        "Pass Rate": f"{m['passed']}/{m['total']} ({m['pass_rate']:.1%})",
                        "Routing Acc": f"{m['routing_accuracy']:.1%}",
                        "Graph-Hop Recall": f"{m['graph_hop_recall']:.1%}",
                        "Code Hit Rate": f"{m['code_hit_rate']:.1%}",
                        "Contradiction Verif.": f"{m['contradiction_accuracy']:.1%}",
                        "Out-of-Corpus Abstain": f"{m['abstention_accuracy']:.1%}",
                    })
            st.table(rows)
    st.markdown("#### Complete Canonical Knowledge Graph")
    st.graphviz_chart(build_subgraph_dot([], full_graph=True), use_container_width=True)
