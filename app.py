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
    "Is a segment tree O(log n) or O(n log n) per query?",
    "How does quantum annealing solve knapsack?",
]


@st.cache_resource(show_spinner="Initialising agent (graph + vector index)...")
def get_agent() -> RAGAgent:
    return RAGAgent()


def run_query(question_text: str, top_k: int) -> None:
    if not question_text.strip():
        return
    try:
        agent = get_agent()
        with st.spinner("Retrieving, grading, synthesizing..."):
            st.session_state["last_response"] = agent.query(question_text.strip(), k=top_k)
            st.session_state["last_question"] = question_text.strip()
    except Exception as e:
        st.session_state["last_error"] = str(e)
        st.session_state.pop("last_response", None)


with st.sidebar:
    st.title("🕸️ Agentic GraphRAG")
    st.caption(
        "Graph traversal + vector search with self-grading, retry, "
        "contradiction detection and abstention."
    )
    k = st.slider("Top-k chunks", 1, 15, 5)
    st.subheader("Examples")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["question_input"] = ex
            st.session_state.pop("last_error", None)
            run_query(ex, k)

st.header("Competitive Programming Knowledge Assistant")

with st.form("query_form", clear_on_submit=False):
    col_in, col_btn = st.columns([5, 1], vertical_alignment="bottom")
    question = col_in.text_input(
        "Ask a question",
        key="question_input",
        placeholder="e.g. How does a segment tree differ from a Fenwick tree?",
    )
    submitted = col_btn.form_submit_button("Ask", type="primary", use_container_width=True)

if submitted and question.strip():
    st.session_state.pop("last_error", None)
    run_query(question, k)

if "last_error" in st.session_state:
    st.error(f"Agent failed: {st.session_state['last_error']}")
    st.info(
        "Make sure `docker compose up -d` is running and the vector "
        "index is built (`python -m retrieval.vector_store`)."
    )

resp = st.session_state.get("last_response")
if resp is not None:
    if resp.abstained:
        st.warning("Agent abstained: topic not found in corpus.")
    if resp.contradiction_detected:
        st.error("Contradiction detected between sources.")

    st.markdown(resp.answer)

    st.divider()
    g = resp.grading
    routing = resp.metadata.get("routing", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Route", routing.get("type", "—"))
    c2.metric("Relevance", f"{g.relevance_score:.2f}")
    c3.metric("Sufficiency", f"{g.sufficiency_score:.2f}")
    c4.metric("Coverage", f"{g.coverage_score:.2f}")
    c5.metric("Retries", resp.retries_used)
    st.caption(f"Verdict: **{g.verdict}** — {g.reason}")

    cx_rep = resp.metadata.get("contradiction")
    if cx_rep and cx_rep.sources:
        with st.expander("Contradiction analysis", expanded=resp.contradiction_detected):
            st.markdown(cx_rep.to_markdown())
    with st.expander("Retrieval path"):
        st.code(resp.retrieval_path)
    with st.expander(f"Sources ({len(resp.sources)})"):
        st.json(resp.sources)
