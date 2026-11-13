"""Phase 8: Streamlit demo UI for the Agentic GraphRAG system.

Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from agent.rag_agent import RAGAgent

st.set_page_config(page_title="Agentic GraphRAG", page_icon="🕸️", layout="wide")

EXAMPLES = [
    "What are the prerequisites for a Fenwick Tree?",
    "Give me a C++ template for Segment Tree",
    "Compare Segment Tree and Fenwick Tree",
    "Explain sliding window technique",
    "Which is correct: sparse table complexity claims?",
    "How does quantum annealing solve knapsack?",
]


@st.cache_resource(show_spinner="Initialising agent (graph + vector index)...")
def get_agent() -> RAGAgent:
    return RAGAgent()


with st.sidebar:
    st.title("🕸️ Agentic GraphRAG")
    st.caption("Graph traversal + vector search with self-grading, retry, "
               "contradiction detection and abstention.")
    k = st.slider("Top-k chunks", 1, 15, 5)
    st.subheader("Examples")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["question"] = ex

st.header("Competitive Programming Knowledge Assistant")
question = st.text_input("Ask a question", key="question")

if st.button("Ask", type="primary") and question.strip():
    try:
        agent = get_agent()
        with st.spinner("Retrieving, grading, synthesizing..."):
            resp = agent.query(question, k=k)
    except Exception as e:  # e.g. Neo4j not running
        st.error(f"Agent failed: {e}")
        st.info("Make sure `docker compose up -d` is running and the vector "
                "index is built (`python -m retrieval.vector_store`).")
        st.stop()

    if resp.abstained:
        st.warning("Agent abstained: topic not found in corpus.")
    if resp.contradiction_detected:
        st.error("Contradiction detected between sources.")

    st.markdown(resp.answer)

    g = resp.grading
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Relevance", f"{g.relevance_score:.2f}")
    c2.metric("Sufficiency", f"{g.sufficiency_score:.2f}")
    c3.metric("Coverage", f"{g.coverage_score:.2f}")
    c4.metric("Retries", resp.retries_used)
    st.caption(f"Verdict: **{g.verdict}** — {g.reason}")

    with st.expander("Retrieval path"):
        st.code(resp.retrieval_path)
    with st.expander(f"Sources ({len(resp.sources)})"):
        st.json(resp.sources)
