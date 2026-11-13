"""
rag_agent.py — Phase 5: Self-Correcting Agentic RAG Loop.

Implements the core agentic loop that:
  1. Receives a user query
  2. Dispatches to the hybrid retriever (Phase 4)
  3. Grades retrieved context for relevance & sufficiency
  4. Self-corrects: widens search, retries, or abstains honestly
  5. Synthesizes a grounded, cited answer

The grading + retry loop is fully deterministic (no LLM needed for grading),
using heuristic scoring on retrieved contexts. An optional LLM synthesis
step uses the Anthropic API if available, otherwise returns structured
context directly.

Design goals:
  - Zero hallucination: abstains when corpus coverage is insufficient
  - Full attribution: every claim linked to source chunks
  - Self-correcting: automatically widens retrieval on low scores
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from retrieval.query_router import QueryRouter, QueryType
from retrieval.hybrid_retriever import HybridRetriever, HybridRetrievalResult, RetrievedContext

from llm import gemini
from agent.contradiction import ContradictionDetector


# ── Configuration ────────────────────────────────────────────────────────

RELEVANCE_THRESHOLD = 0.4       # min relevance score per context chunk
SUFFICIENCY_THRESHOLD = 0.6     # min overall grading score to accept
MAX_RETRIES = 2                 # max self-correction retries
ABSTENTION_THRESHOLD = 0.25     # below this → honest abstention

# LLM synthesis model


# ── Result Types ─────────────────────────────────────────────────────────

@dataclass
class GradingResult:
    """Result of grading retrieved context."""
    relevance_score: float        # 0.0–1.0: are chunks relevant to the query?
    sufficiency_score: float      # 0.0–1.0: is there enough info to answer?
    coverage_score: float         # 0.0–1.0: do chunks cover the query's scope?
    has_code: bool                # did we retrieve compilable code?
    has_graph_data: bool          # did graph traversal contribute?
    num_relevant_chunks: int
    total_chunks: int
    verdict: str                  # "sufficient", "partial", "insufficient", "abstain"
    reason: str

    @property
    def overall_score(self) -> float:
        return (self.relevance_score + self.sufficiency_score + self.coverage_score) / 3.0


@dataclass
class AgentResponse:
    """Final response from the agentic RAG system."""
    query: str
    answer: str
    sources: list[dict]             # attribution list
    grading: GradingResult
    retrieval_path: str             # routing + retrieval strategy used
    retries_used: int
    abstained: bool = False
    contradiction_detected: bool = False
    metadata: dict = field(default_factory=dict)


# ── Grader ───────────────────────────────────────────────────────────────

class ContextGrader:
    """
    Heuristic grader for retrieved context quality.

    Evaluates relevance, sufficiency, and coverage without an LLM call,
    using keyword overlap, pattern matching, and structural checks.
    """

    def grade(
        self,
        query: str,
        contexts: list[RetrievedContext],
        routing_decision=None,
    ) -> GradingResult:
        """Grade the retrieved contexts against the query."""
        if not contexts:
            return GradingResult(
                relevance_score=0.0,
                sufficiency_score=0.0,
                coverage_score=0.0,
                has_code=False,
                has_graph_data=False,
                num_relevant_chunks=0,
                total_chunks=0,
                verdict="abstain",
                reason="No contexts retrieved.",
            )

        q_lower = query.lower()
        import re as _re
        _stop = {
            "what", "is", "are", "the", "a", "an", "for", "of", "in", "on", "to",
            "how", "does", "do", "i", "need", "should", "learn", "before", "know",
            "explain", "give", "me", "show", "compare", "between", "from", "with",
            "and", "or", "by", "this", "these", "used", "work", "can", "which",
        }
        q_tokens = [w for w in _re.findall(r"[a-z0-9\+]+", q_lower) if len(w) > 1]
        q_words = set(q_tokens)
        q_content_words = {w for w in q_words if w not in _stop} or q_words

        expected_patterns = set()
        if routing_decision and routing_decision.detected_patterns:
            expected_patterns = set(routing_decision.detected_patterns)

        relevant_chunks = 0
        has_code = False
        has_graph = False
        relevance_scores = []

        for ctx in contexts:
            # Check if context text overlaps with query keywords
            ctx_lower = ctx.text.lower()
            ctx_words = set(_re.findall(r"[a-z0-9\+]+", ctx_lower))
            word_overlap = len(q_words & ctx_words) / max(len(q_words), 1)
            content_overlap = len(q_content_words & ctx_words) / max(len(q_content_words), 1)

            # Check if any detected patterns appear in the context
            pattern_match = (
                bool(set(ctx.pattern_ids) & expected_patterns)
                if expected_patterns
                else any(pid.lower() in ctx_lower for pid in ctx.pattern_ids)
            )

            # For pure semantic queries (no known patterns detected), require
            # either strong vector similarity (>=0.55) or meaningful content-word
            # overlap so out-of-corpus topics don't pass on background similarity.
            if not expected_patterns and ctx.source_type != "graph":
                if ctx.relevance_score < 0.55 and content_overlap == 0.0:
                    chunk_relevance = ctx.relevance_score * 0.4
                else:
                    chunk_relevance = min(1.0, ctx.relevance_score * 0.5 + content_overlap * 0.4)
            else:
                chunk_relevance = min(
                    1.0,
                    ctx.relevance_score * 0.5
                    + word_overlap * 0.3
                    + (0.2 if pattern_match else 0.0),
                )
            relevance_scores.append(chunk_relevance)

            if chunk_relevance >= RELEVANCE_THRESHOLD:
                relevant_chunks += 1

            if ctx.source_type in ("vector_code", "graph_template"):
                has_code = True
            if ctx.source_type == "graph":
                has_graph = True

        # Aggregate scores
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        sufficiency = min(1.0, relevant_chunks / max(1, min(3, len(contexts))))

        # Coverage: how many of the query's detected patterns are represented?
        if expected_patterns:
            found_patterns = set()
            for ctx in contexts:
                found_patterns.update(ctx.pattern_ids)
            coverage = len(found_patterns & expected_patterns) / len(expected_patterns)
        else:
            # No specific patterns expected — coverage based on chunk count
            coverage = min(1.0, relevant_chunks / 2.0)

        overall = (avg_relevance + sufficiency + coverage) / 3.0

        # Determine verdict
        if overall >= SUFFICIENCY_THRESHOLD:
            verdict = "sufficient"
            reason = f"Retrieved {relevant_chunks} relevant chunks with {overall:.0%} overall score."
        elif overall >= ABSTENTION_THRESHOLD:
            verdict = "partial"
            reason = f"Partial coverage ({overall:.0%}). May need wider search."
        else:
            verdict = "abstain"
            reason = f"Insufficient coverage ({overall:.0%}). Topic may not be in corpus."

        return GradingResult(
            relevance_score=avg_relevance,
            sufficiency_score=sufficiency,
            coverage_score=coverage,
            has_code=has_code,
            has_graph_data=has_graph,
            num_relevant_chunks=relevant_chunks,
            total_chunks=len(contexts),
            verdict=verdict,
            reason=reason,
        )


# ── Answer Synthesizer ───────────────────────────────────────────────────

class AnswerSynthesizer:
    """
    Synthesizes a final answer from graded contexts.

    If a Gemini API key is available, uses Gemini for natural language
    synthesis. Otherwise, returns a structured context dump.
    """

    def __init__(self):
        pass

    @property
    def has_llm(self) -> bool:
        return gemini.available()

    def synthesize(
        self,
        query: str,
        contexts: list[RetrievedContext],
        grading: GradingResult,
        routing_decision=None,
        contradiction_report=None,
    ) -> str:
        """Generate a final answer from the retrieved and graded contexts."""
        # Abstention
        if grading.verdict == "abstain":
            return self._abstention_response(query)

        # Build context block
        context_block = self._format_contexts(contexts, grading)

        # LLM synthesis
        cx_md = contradiction_report.to_markdown() if contradiction_report else ""
        if cx_md:
            context_block += "\n\nContradiction analysis (automated):\n" + cx_md

        if self.has_llm:
            return self._llm_synthesize(query, context_block, routing_decision)

        # Structured fallback
        out = self._structured_response(query, contexts, grading, routing_decision)
        if cx_md:
            out = "### Contradiction Analysis\n\n" + cx_md + "\n\n" + out
        return out

    def _abstention_response(self, query: str) -> str:
        return (
            f"I do not have sufficient verified information in my corpus to answer: "
            f"\"{query}\"\n\n"
            f"This topic may not be covered by the ingested editorials and "
            f"algorithmic patterns. I prefer to abstain rather than risk "
            f"providing ungrounded or hallucinated information."
        )

    def _format_contexts(
        self, contexts: list[RetrievedContext], grading: GradingResult
    ) -> str:
        """Format relevant contexts into a prompt-ready block."""
        parts = []
        for i, ctx in enumerate(contexts, 1):
            if ctx.relevance_score < RELEVANCE_THRESHOLD * 0.5:
                continue  # skip very low quality chunks
            source_tag = f"[{ctx.source_type}]"
            patterns = ", ".join(ctx.pattern_ids) if ctx.pattern_ids else "none"
            parts.append(
                f"--- Context {i} {source_tag} (patterns: {patterns}) ---\n"
                f"{ctx.text}"
            )
        return "\n\n".join(parts)

    def _llm_synthesize(
        self, query: str, context_block: str, routing_decision=None
    ) -> str:
        """Use Gemini to synthesize a grounded answer."""
        query_type = routing_decision.query_type.value if routing_decision else "unknown"

        system_prompt = (
            "You are a competitive programming expert answering questions based "
            "STRICTLY on the provided context. Rules:\n"
            "1. Only use information from the provided context.\n"
            "2. If the context doesn't fully answer the question, say what you "
            "can and note what's missing.\n"
            "3. Cite patterns and sources explicitly.\n"
            "4. For code templates, preserve them exactly as retrieved.\n"
            "5. If complexities are mentioned, state them precisely.\n"
            "6. Never fabricate algorithms, complexities, or code not in context.\n"
            "7. If a 'Contradiction analysis' section is present, present each "
            "conflicting claim with its source, then say which claim the context "
            "supports (prefer the canonical graph value) and why."
        )

        user_prompt = (
            f"Query type: {query_type}\n"
            f"Question: {query}\n\n"
            f"Retrieved Context:\n{context_block}\n\n"
            f"Provide a clear, well-structured answer grounded in the context above."
        )

        try:
            return gemini.generate(user_prompt, system=system_prompt, max_tokens=2048)
        except Exception as e:
            return f"[LLM synthesis failed: {e}]\n\n" + self._structured_response(
                query, [], None, routing_decision  # type: ignore
            )

    def _structured_response(
        self,
        query: str,
        contexts: list[RetrievedContext],
        grading: Optional[GradingResult],
        routing_decision=None,
    ) -> str:
        """Build a structured answer without LLM."""
        parts = [f"## Answer: {query}\n"]

        if routing_decision:
            parts.append(
                f"**Query Type**: {routing_decision.query_type.value} "
                f"(confidence: {routing_decision.confidence:.0%})\n"
            )
            if routing_decision.detected_patterns:
                parts.append(
                    f"**Detected Patterns**: {', '.join(routing_decision.detected_patterns)}\n"
                )

        if grading:
            parts.append(
                f"**Retrieval Quality**: {grading.verdict} "
                f"({grading.overall_score:.0%} overall, "
                f"{grading.num_relevant_chunks}/{grading.total_chunks} relevant chunks)\n"
            )

        parts.append("\n### Retrieved Context\n")

        for i, ctx in enumerate(contexts, 1):
            patterns = ", ".join(ctx.pattern_ids) if ctx.pattern_ids else "—"
            parts.append(f"**[{i}] {ctx.source_type}** (patterns: {patterns})")
            # Truncate very long code blocks
            text = ctx.text
            if len(text) > 500 and ctx.source_type in ("vector_code", "graph_template"):
                text = text[:500] + "\n... [truncated]"
            parts.append(f"```\n{text}\n```\n")

        return "\n".join(parts)


# ── Agentic RAG System ───────────────────────────────────────────────────

class RAGAgent:
    """
    Self-correcting agentic RAG system.

    Implements the retrieve → grade → retry/widen → synthesize loop
    described in Phase 5 of the project architecture.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever | None = None,
        grader: ContextGrader | None = None,
        synthesizer: AnswerSynthesizer | None = None,
    ):
        self.retriever = hybrid_retriever or HybridRetriever()
        self.grader = grader or ContextGrader()
        self.synthesizer = synthesizer or AnswerSynthesizer()
        self.contradiction_detector = ContradictionDetector()

    def query(self, question: str, k: int = 5, verbose: bool = False) -> AgentResponse:
        """
        Process a user query through the full agentic RAG pipeline.

        1. Route & retrieve (hybrid)
        2. Grade context quality
        3. Self-correct if needed (widen search, retry)
        4. Synthesize grounded answer or abstain
        """
        retries = 0
        current_k = k
        best_result: Optional[HybridRetrievalResult] = None
        best_grading: Optional[GradingResult] = None

        while retries <= MAX_RETRIES:
            if verbose:
                print(f"  [Agent] Attempt {retries + 1}/{MAX_RETRIES + 1} (k={current_k})")

            # Step 1: Retrieve
            result = self.retriever.retrieve(question, k=current_k)

            if verbose:
                print(f"  [Agent] Route: {result.routing.query_type.value}, "
                      f"Contexts: {len(result.contexts)}")

            # Step 2: Grade
            grading = self.grader.grade(
                question, result.contexts, result.routing
            )

            if verbose:
                print(f"  [Agent] Grade: {grading.verdict} "
                      f"(overall={grading.overall_score:.2f}, "
                      f"relevant={grading.num_relevant_chunks}/{grading.total_chunks})")

            # Track best attempt
            if best_grading is None or grading.overall_score > best_grading.overall_score:
                best_result = result
                best_grading = grading

            # Step 3: Check if sufficient
            if grading.verdict == "sufficient":
                break

            if grading.verdict == "abstain" and retries >= 1:
                # Already retried once with wider search — give up
                break

            if retries >= MAX_RETRIES:
                break

            # Step 4: Self-correct — widen the search
            retries += 1
            current_k = min(current_k * 2, 20)  # double k, cap at 20

            if verbose:
                print(f"  [Agent] Retrying with k={current_k}...")

        # Use best result
        assert best_result is not None
        assert best_grading is not None

        # Step 5: Synthesize answer
        is_abstention = best_grading.verdict == "abstain"
        contradiction_report = None
        if best_result.routing.query_type == QueryType.CONTRADICTION and not is_abstention:
            target = best_result.routing.target_pattern or (
                best_result.routing.detected_patterns[0]
                if best_result.routing.detected_patterns else None)
            contradiction_report = self.contradiction_detector.detect(target, best_result.contexts)
        is_contradiction = bool(contradiction_report and contradiction_report.detected)

        answer = self.synthesizer.synthesize(
            question,
            best_result.contexts,
            best_grading,
            best_result.routing,
            contradiction_report=contradiction_report,
        )

        # Build source attribution
        sources = []
        for ctx in best_result.contexts:
            src = {
                "type": ctx.source_type,
                "patterns": ctx.pattern_ids,
                "relevance": round(ctx.relevance_score, 3),
            }
            if ctx.source_url:
                src["url"] = ctx.source_url
            if "editorial_id" in ctx.metadata:
                src["editorial_id"] = ctx.metadata["editorial_id"]
            sources.append(src)

        return AgentResponse(
            query=question,
            answer=answer,
            sources=sources,
            grading=best_grading,
            retrieval_path=best_result.retrieval_path,
            retries_used=retries,
            abstained=is_abstention,
            contradiction_detected=is_contradiction,
            metadata={
                "routing": {
                    "type": best_result.routing.query_type.value,
                    "confidence": best_result.routing.confidence,
                    "detected_patterns": best_result.routing.detected_patterns,
                },
                "contradiction": contradiction_report,
            },
        )

    def batch_query(
        self, queries: list[str], k: int = 5, verbose: bool = False
    ) -> list[AgentResponse]:
        """Process multiple queries sequentially."""
        return [self.query(q, k=k, verbose=verbose) for q in queries]


# ── CLI demo ─────────────────────────────────────────────────────────────

def demo():
    """Run the agentic RAG on all benchmark queries."""
    print("=" * 72)
    print("Phase 5 — Self-Correcting Agentic RAG Demo")
    print("=" * 72)

    agent = RAGAgent()

    if not agent.retriever.has_vector:
        print("\n⚠ Vector index not found. Run: python -m retrieval.vector_store")
        return

    test_queries = [
        # Q1: Graph-hop comparison
        "How does a segment tree differ from a Fenwick tree for range queries?",
        # Q2: Prerequisite chain
        "What do I need to know before learning suffix arrays?",
        # Q3: Code template (hybrid)
        "Give me a C++ template for two-pointers on a sorted array",
        # Q4: Complexity lookup
        "What's the time complexity of a sparse table?",
        # Q5: Contradiction
        "Is a segment tree O(log n) or O(n log n) per query?",
        # Q6: Out-of-corpus → abstention
        "Explain the Aho-Corasick automaton for multi-pattern string matching",
    ]

    print(f"\nRunning {len(test_queries)} benchmark queries...\n")

    for i, q in enumerate(test_queries, 1):
        print(f"{'─' * 72}")
        print(f"Q{i}: \"{q}\"")
        print(f"{'─' * 72}")

        response = agent.query(q, k=5, verbose=True)

        print(f"\n  Route:     {response.metadata['routing']['type']} "
              f"({response.metadata['routing']['confidence']:.0%})")
        print(f"  Path:      {response.retrieval_path}")
        print(f"  Verdict:   {response.grading.verdict} "
              f"({response.grading.overall_score:.0%})")
        print(f"  Retries:   {response.retries_used}")
        print(f"  Abstained: {response.abstained}")
        print(f"  Sources:   {len(response.sources)}")

        # Show first 200 chars of answer
        answer_preview = response.answer[:200].replace("\n", " ")
        print(f"\n  Answer: {answer_preview}...")
        print()

    print("=" * 72)
    print("✓ Agentic RAG benchmark complete.")
    print("=" * 72)


if __name__ == "__main__":
    demo()

