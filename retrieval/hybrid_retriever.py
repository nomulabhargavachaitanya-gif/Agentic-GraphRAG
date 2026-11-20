"""
hybrid_retriever.py — Hybrid Graph + Dense/BM25 RRF Retrieval Engine.

Orchestrates joint retrieval across the Knowledge Graph (Neo4j or In-Memory
Graph Fallback) and Vector Store (FAISS + Okapi BM25 Reciprocal Rank Fusion)
based on the QueryRouter's classification:

  GRAPH_HOP     → Cypher multi-hop traversal (prerequisites, comparisons) + RRF vector context
  CODE_TEMPLATE → Verified graph C++ template lookup + filtered vector code search
  SEMANTIC      → Dense + BM25 RRF vector search
  HYBRID        → RRF vector search → pattern entity extraction → 1-hop graph expansion
  CONTRADICTION → Canonical graph complexity lookup + multi-source vector retrieval
"""

from __future__ import annotations

from dataclasses import dataclass, field

from retrieval.query_router import QueryRouter, QueryType, RoutingDecision
from retrieval.vector_retriever import VectorRetriever


def _l2_to_similarity(dist: float) -> float:
    """Convert squared L2 distance on unit-normalised vectors to [0, 1] similarity.

    For unit vectors ||u - v||^2 = 2(1 - cos(u, v)), so cos = 1 - dist / 2.
    """
    return max(0.0, min(1.0, 1.0 - dist / 2.0))


# ── Result Types ─────────────────────────────────────────────────────────

@dataclass
class RetrievedContext:
    """A single piece of retrieved context with provenance."""
    text: str
    source_type: str           # "graph", "vector_prose", "vector_code", "graph_template"
    relevance_score: float     # 0.0–1.0 (normalized)
    pattern_ids: list[str] = field(default_factory=list)
    source_url: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class HybridRetrievalResult:
    """Complete result from the hybrid retrieval pipeline."""
    query: str
    routing: RoutingDecision
    contexts: list[RetrievedContext]
    graph_data: dict = field(default_factory=dict)
    total_contexts: int = 0
    retrieval_path: str = ""

    def to_prompt_context(self) -> str:
        """Format all retrieved contexts into a single prompt-ready string."""
        parts = []
        for i, ctx in enumerate(self.contexts, 1):
            header = f"[Source {i}: {ctx.source_type}]"
            if ctx.pattern_ids:
                header += f" (patterns: {', '.join(ctx.pattern_ids)})"
            if ctx.source_url:
                header += f" (url: {ctx.source_url})"
            parts.append(f"{header}\n{ctx.text}")
        return "\n\n---\n\n".join(parts)


# ── Hybrid Retriever ─────────────────────────────────────────────────────

class HybridRetriever:
    """
    Orchestrates retrieval across graph and vector stores with RRF fusion
    and chunk deduplication.
    """

    def __init__(
        self,
        graph_retriever=None,
        vector_retriever: VectorRetriever | None = None,
        router: QueryRouter | None = None,
    ):
        if graph_retriever is not None:
            self.graph = graph_retriever
        else:
            try:
                from retrieval.graph_retriever import GraphRetriever
                self.graph = GraphRetriever()
            except Exception:
                try:
                    from retrieval.graph_retriever import InMemoryGraphRetriever
                    self.graph = InMemoryGraphRetriever()
                except Exception:
                    self.graph = None
        self.vector = vector_retriever or VectorRetriever()
        self.router = router or QueryRouter()

    @property
    def has_graph(self) -> bool:
        return self.graph is not None

    @property
    def has_vector(self) -> bool:
        return self.vector.is_ready

    @staticmethod
    def _deduplicate(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        seen: set[tuple[str, str]] = set()
        unique: list[RetrievedContext] = []
        for ctx in contexts:
            sig = (ctx.source_type, ctx.text[:160].strip())
            if sig not in seen:
                seen.add(sig)
                unique.append(ctx)
        return unique

    def retrieve(self, query: str, k: int = 5) -> HybridRetrievalResult:
        """Classify the query and dispatch to the selected retrieval strategy."""
        decision = self.router.classify(query)

        dispatch = {
            QueryType.GRAPH_HOP: self._retrieve_graph_hop,
            QueryType.CODE_TEMPLATE: self._retrieve_code_template,
            QueryType.SEMANTIC: self._retrieve_semantic,
            QueryType.HYBRID: self._retrieve_hybrid,
            QueryType.CONTRADICTION: self._retrieve_contradiction,
        }

        handler = dispatch.get(decision.query_type, self._retrieve_semantic)
        result = handler(decision, k=k)
        result.contexts = self._deduplicate(result.contexts)
        result.total_contexts = len(result.contexts)
        return result

    def retrieve_vector_only(self, query: str, k: int = 5) -> HybridRetrievalResult:
        """Pure dense vector baseline without routing or graph traversal (for ablation)."""
        decision = RoutingDecision(
            query=query,
            query_type=QueryType.SEMANTIC,
            confidence=1.0,
            detected_patterns=[],
            reason="Vanilla vector-only baseline (ablation).",
        )
        contexts: list[RetrievedContext] = []
        if self.has_vector:
            results = self.vector.store.search(query, k=k, use_rrf=False)
            for r in results:
                contexts.append(RetrievedContext(
                    text=r.text,
                    source_type=f"vector_{r.metadata.chunk_type}",
                    relevance_score=_l2_to_similarity(r.score),
                    pattern_ids=r.metadata.pattern_ids,
                    source_url=r.metadata.source_url,
                    metadata={"editorial_id": r.metadata.editorial_id},
                ))
        return HybridRetrievalResult(
            query=query,
            routing=decision,
            contexts=contexts,
            total_contexts=len(contexts),
            retrieval_path=f"vector_only(k={k})",
        )

    # ── Strategy 1: Graph-Hop ────────────────────────────────────────

    def _retrieve_graph_hop(
        self, decision: RoutingDecision, k: int = 5
    ) -> HybridRetrievalResult:
        """Handle prerequisite chains and pattern comparisons."""
        contexts: list[RetrievedContext] = []
        graph_data: dict = {}

        if decision.pattern_a and decision.pattern_b:
            if self.has_graph:
                try:
                    result = self.graph.compare_patterns(
                        decision.pattern_a, decision.pattern_b
                    )
                    if result:
                        graph_data["comparison"] = _safe_asdict(result)
                        contexts.append(RetrievedContext(
                            text=(
                                f"Comparison: {result.pattern_a.name} vs {result.pattern_b.name}\n"
                                f"Relationship: {result.relationship_type}\n"
                                f"Tradeoff: {result.tradeoff}\n\n"
                                f"{result.pattern_a.name}:\n"
                                f"  Time: {result.pattern_a.time_complexity}\n"
                                f"  Space: {result.pattern_a.space_complexity}\n"
                                f"  Description: {result.pattern_a.description}\n\n"
                                f"{result.pattern_b.name}:\n"
                                f"  Time: {result.pattern_b.time_complexity}\n"
                                f"  Space: {result.pattern_b.space_complexity}\n"
                                f"  Description: {result.pattern_b.description}"
                            ),
                            source_type="graph",
                            relevance_score=1.0,
                            pattern_ids=[decision.pattern_a, decision.pattern_b],
                        ))

                        for tmpl in getattr(result, "templates", []):
                            contexts.append(RetrievedContext(
                                text=f"// {tmpl.pattern} — C++ Template\n{tmpl.code}",
                                source_type="graph_template",
                                relevance_score=0.95,
                                pattern_ids=[tmpl.pattern],
                                source_url=tmpl.source_url or "",
                                metadata={"explanation": tmpl.explanation},
                            ))
                except Exception:
                    pass

            if self.has_vector:
                self._add_vector_contexts(
                    contexts, decision.query, k=k,
                    pattern_filter=None,
                )

            return HybridRetrievalResult(
                query=decision.query,
                routing=decision,
                contexts=contexts,
                graph_data=graph_data,
                retrieval_path=f"graph_comparison({decision.pattern_a}, {decision.pattern_b}) + rrf_vector",
            )

        if decision.target_pattern:
            if self.has_graph:
                try:
                    result = self.graph.prerequisite_chain(decision.target_pattern)
                    if result:
                        graph_data["prerequisite_chain"] = _safe_asdict(result)
                        chain_str = " → ".join(p.name for p in result.chain)
                        contexts.append(RetrievedContext(
                            text=(
                                f"Prerequisite chain to learn '{result.target}' "
                                f"(depth={result.depth}):\n\n{chain_str}\n\n"
                                + "\n".join(
                                    f"• {p.name}: {p.description} "
                                    f"[Time: {p.time_complexity}, Space: {p.space_complexity}]"
                                    for p in result.chain
                                )
                            ),
                            source_type="graph",
                            relevance_score=1.0,
                            pattern_ids=[p.name for p in result.chain],
                        ))
                except Exception:
                    pass

            if self.has_vector:
                self._add_vector_contexts(
                    contexts, decision.query, k=k,
                    pattern_filter=decision.target_pattern,
                )

            return HybridRetrievalResult(
                query=decision.query,
                routing=decision,
                contexts=contexts,
                graph_data=graph_data,
                retrieval_path=f"graph_prerequisites({decision.target_pattern}) + rrf_vector",
            )

        if decision.detected_patterns and self.has_graph:
            pattern = decision.detected_patterns[0]
            try:
                neighbors = self.graph.get_neighbors(pattern)
                if neighbors:
                    graph_data["neighbors"] = neighbors
                    neighbor_text = "\n".join(
                        f"• {n['name']} ({n['relationship']})"
                        for n in neighbors
                    )
                    contexts.append(RetrievedContext(
                        text=f"Graph neighbors of '{pattern}':\n{neighbor_text}",
                        source_type="graph",
                        relevance_score=0.85,
                        pattern_ids=[pattern] + [n["name"] for n in neighbors],
                    ))
            except Exception:
                pass

        if self.has_vector:
            self._add_vector_contexts(contexts, decision.query, k=k)

        return HybridRetrievalResult(
            query=decision.query,
            routing=decision,
            contexts=contexts,
            graph_data=graph_data,
            retrieval_path="graph_neighbors + rrf_vector",
        )

    # ── Strategy 2: Code Template ────────────────────────────────────

    def _retrieve_code_template(
        self, decision: RoutingDecision, k: int = 5
    ) -> HybridRetrievalResult:
        """Retrieve C++ code templates from both graph and vector."""
        contexts: list[RetrievedContext] = []
        graph_data: dict = {}

        if self.has_graph and decision.target_pattern:
            try:
                tmpl = self.graph.get_template(decision.target_pattern)
                if tmpl:
                    graph_data["template"] = _safe_asdict(tmpl)
                    contexts.append(RetrievedContext(
                        text=f"// {tmpl.pattern} — Verified C++ Template\n{tmpl.code}",
                        source_type="graph_template",
                        relevance_score=1.0,
                        pattern_ids=[tmpl.pattern],
                        source_url=tmpl.source_url or "",
                        metadata={"explanation": tmpl.explanation},
                    ))
            except Exception:
                pass

        if self.has_vector:
            code_results = self.vector.search_code(
                decision.query, k=k,
                pattern=decision.target_pattern,
            )
            for r in code_results:
                contexts.append(RetrievedContext(
                    text=r.text,
                    source_type="vector_code",
                    relevance_score=_l2_to_similarity(r.score),
                    pattern_ids=r.metadata.pattern_ids,
                    source_url=r.metadata.source_url,
                    metadata={
                        "editorial_id": r.metadata.editorial_id,
                        "bm25_score": r.bm25_score,
                        "rrf_score": r.rrf_score,
                    },
                ))

            prose_results = self.vector.search_prose(
                decision.query, k=min(k, 2),
                pattern=decision.target_pattern,
            )
            for r in prose_results:
                contexts.append(RetrievedContext(
                    text=r.text,
                    source_type="vector_prose",
                    relevance_score=_l2_to_similarity(r.score),
                    pattern_ids=r.metadata.pattern_ids,
                    source_url=r.metadata.source_url,
                    metadata={
                        "editorial_id": r.metadata.editorial_id,
                        "bm25_score": r.bm25_score,
                        "rrf_score": r.rrf_score,
                    },
                ))

        return HybridRetrievalResult(
            query=decision.query,
            routing=decision,
            contexts=contexts,
            graph_data=graph_data,
            retrieval_path=f"graph_template({decision.target_pattern}) + rrf_vector_code",
        )

    # ── Strategy 3: Pure Semantic ────────────────────────────────────

    def _retrieve_semantic(
        self, decision: RoutingDecision, k: int = 5
    ) -> HybridRetrievalResult:
        """Vector + BM25 RRF search without graph traversal."""
        contexts: list[RetrievedContext] = []

        if self.has_vector:
            self._add_vector_contexts(contexts, decision.query, k=k)

        return HybridRetrievalResult(
            query=decision.query,
            routing=decision,
            contexts=contexts,
            retrieval_path="vector_bm25_rrf",
        )

    # ── Strategy 4: Hybrid ───────────────────────────────────────────

    def _retrieve_hybrid(
        self, decision: RoutingDecision, k: int = 5
    ) -> HybridRetrievalResult:
        """RRF vector search → extract pattern IDs → 1-hop graph expansion."""
        contexts: list[RetrievedContext] = []
        graph_data: dict = {}

        linked_patterns: list[str] = []
        if self.has_vector:
            v_result = self.vector.search(decision.query, k=k)
            linked_patterns = v_result.linked_pattern_ids

            for r in v_result.results:
                contexts.append(RetrievedContext(
                    text=r.text,
                    source_type=f"vector_{r.metadata.chunk_type}",
                    relevance_score=_l2_to_similarity(r.score),
                    pattern_ids=r.metadata.pattern_ids,
                    source_url=r.metadata.source_url,
                    metadata={
                        "editorial_id": r.metadata.editorial_id,
                        "bm25_score": r.bm25_score,
                        "rrf_score": r.rrf_score,
                    },
                ))

        all_patterns = list(
            dict.fromkeys(decision.detected_patterns + linked_patterns)
        )

        if self.has_graph and all_patterns:
            for pattern in all_patterns[:3]:
                try:
                    cx = self.graph.pattern_complexity(pattern)
                    if cx:
                        graph_data[f"complexity_{pattern}"] = _safe_asdict(cx)
                        contexts.append(RetrievedContext(
                            text=(
                                f"{cx.pattern.name}:\n"
                                f"  Time complexity: {cx.time_complexity}\n"
                                f"  Space complexity: {cx.space_complexity}\n"
                                f"  Description: {cx.pattern.description}"
                            ),
                            source_type="graph",
                            relevance_score=0.92,
                            pattern_ids=[pattern],
                        ))
                except Exception:
                    pass

                try:
                    neighbors = self.graph.get_neighbors(pattern)
                    if neighbors:
                        graph_data[f"neighbors_{pattern}"] = neighbors
                except Exception:
                    pass

        return HybridRetrievalResult(
            query=decision.query,
            routing=decision,
            contexts=contexts,
            graph_data=graph_data,
            retrieval_path=f"rrf_vector(k={k}) → graph_expand({all_patterns[:3]})",
        )

    # ── Strategy 5: Contradiction Detection ──────────────────────────

    def _retrieve_contradiction(
        self, decision: RoutingDecision, k: int = 5
    ) -> HybridRetrievalResult:
        """Retrieve multiple sources + canonical graph node for conflict verification."""
        contexts: list[RetrievedContext] = []
        graph_data: dict = {}

        target = decision.target_pattern or (
            decision.detected_patterns[0] if decision.detected_patterns else None
        )

        if self.has_vector and target:
            v_result = self.vector.search(
                decision.query, k=k * 2, pattern_filter=target
            )
            for r in v_result.results:
                contexts.append(RetrievedContext(
                    text=r.text,
                    source_type=f"vector_{r.metadata.chunk_type}",
                    relevance_score=_l2_to_similarity(r.score),
                    pattern_ids=r.metadata.pattern_ids,
                    source_url=r.metadata.source_url,
                    metadata={
                        "editorial_id": r.metadata.editorial_id,
                        "requires_contradiction_check": True,
                        "bm25_score": r.bm25_score,
                        "rrf_score": r.rrf_score,
                    },
                ))

        if self.has_graph and target:
            try:
                cx = self.graph.pattern_complexity(target)
                if cx:
                    graph_data["canonical_complexity"] = _safe_asdict(cx)
                    contexts.insert(0, RetrievedContext(
                        text=(
                            f"Canonical complexity for {cx.pattern.name}:\n"
                            f"  Time: {cx.time_complexity}\n"
                            f"  Space: {cx.space_complexity}"
                        ),
                        source_type="graph",
                        relevance_score=1.0,
                        pattern_ids=[target],
                        metadata={"is_canonical": True},
                    ))
            except Exception:
                pass

        return HybridRetrievalResult(
            query=decision.query,
            routing=decision,
            contexts=contexts,
            graph_data=graph_data,
            retrieval_path=f"contradiction_search({target})",
        )

    # ── Shared Helpers ───────────────────────────────────────────────

    def _add_vector_contexts(
        self,
        contexts: list[RetrievedContext],
        query: str,
        k: int = 5,
        pattern_filter: str | None = None,
    ):
        v_result = self.vector.search(
            query, k=k, pattern_filter=pattern_filter
        )
        for r in v_result.results:
            contexts.append(RetrievedContext(
                text=r.text,
                source_type=f"vector_{r.metadata.chunk_type}",
                relevance_score=_l2_to_similarity(r.score),
                pattern_ids=r.metadata.pattern_ids,
                source_url=r.metadata.source_url,
                metadata={
                    "editorial_id": r.metadata.editorial_id,
                    "bm25_score": r.bm25_score,
                    "rrf_score": r.rrf_score,
                },
            ))


def _safe_asdict(obj) -> dict:
    try:
        from dataclasses import asdict
        return asdict(obj)
    except Exception:
        return {"repr": str(obj)}


# ── CLI demo ─────────────────────────────────────────────────────────────

def demo():
    """Run the hybrid retriever on benchmark queries."""
    print("=" * 72)
    print("Hybrid Retrieval Demo (Graph + FAISS/BM25 RRF)")
    print("=" * 72)

    retriever = HybridRetriever()

    if not retriever.has_vector:
        print("\n⚠ Vector index not found. Run: python -m retrieval.vector_store")
        return

    test_queries = [
        "How does a segment tree differ from a Fenwick tree for range queries?",
        "What do I need to know before learning suffix arrays?",
        "Give me a C++ template for two-pointers on a sorted array",
        "What's the time complexity of a sparse table?",
        "How does sliding window work for substring problems?",
        "Explain the Aho-Corasick automaton for multi-pattern string matching",
    ]

    for q in test_queries:
        result = retriever.retrieve(q, k=3)
        print(f"\nQ: \"{q}\"")
        print(f"  Route:    {result.routing.query_type.value} "
              f"({result.routing.confidence:.0%})")
        print(f"  Path:     {result.retrieval_path}")
        print(f"  Contexts: {result.total_contexts}")
        for ctx in result.contexts[:2]:
            preview = ctx.text[:80].replace("\n", " ")
            print(f"    [{ctx.source_type}] {ctx.relevance_score:.2f}: \"{preview}...\"")

    print("\n✓ Hybrid retrieval demo complete.")


if __name__ == "__main__":
    demo()
