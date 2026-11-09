"""
query_router.py — Phase 4: Query Classification & Hybrid Retrieval Router.

Classifies incoming natural-language queries into one of four retrieval
strategies and dispatches to the appropriate retriever pipeline:

  1. GRAPH_HOP     — prerequisite chains, comparisons, neighbor lookups
  2. CODE_TEMPLATE — C++ implementation requests, boilerplate, templates
  3. SEMANTIC      — general algorithmic concept / editorial prose queries
  4. HYBRID        — needs both vector similarity + graph expansion
  5. CONTRADICTION — conflicting complexity / correctness claims

The router uses lightweight keyword/pattern heuristics (no LLM call),
making it deterministic, fast, and free to run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QueryType(str, Enum):
    GRAPH_HOP = "graph_hop"
    CODE_TEMPLATE = "code_template"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    CONTRADICTION = "contradiction"


@dataclass
class RoutingDecision:
    """Result of classifying a query."""
    query: str
    query_type: QueryType
    confidence: float                  # 0.0–1.0
    detected_patterns: list[str]       # algorithm names mentioned in the query
    reason: str                        # human-readable explanation

    # Extracted intent signals
    pattern_a: Optional[str] = None    # for GRAPH_HOP comparisons
    pattern_b: Optional[str] = None
    target_pattern: Optional[str] = None  # for prerequisite lookups


# ── Known pattern aliases ────────────────────────────────────────────────

PATTERN_ALIASES: dict[str, str] = {
    # Canonical name → aliases
    "two pointers": "Two Pointers",
    "two pointer": "Two Pointers",
    "two-pointers": "Two Pointers",
    "two-pointer": "Two Pointers",
    "sliding window": "Sliding Window",
    "sliding-window": "Sliding Window",
    "binary search": "Binary Search",
    "prefix sum": "Prefix Sum",
    "prefix sums": "Prefix Sum",
    "fenwick tree": "Fenwick Tree",
    "binary indexed tree": "Fenwick Tree",
    "bit": "Fenwick Tree",
    "segment tree": "Segment Tree",
    "segtree": "Segment Tree",
    "sparse table": "Sparse Table",
    "suffix array": "Suffix Array",
    "suffix arrays": "Suffix Array",
    "dp": "Dynamic Programming",
    "dynamic programming": "Dynamic Programming",
    "bfs": "BFS",
    "dfs": "DFS",
    "dijkstra": "Dijkstra",
    "union find": "Union Find",
    "disjoint set": "Union Find",
    "dsu": "Union Find",
    "trie": "Trie",
    "topological sort": "Topological Sort",
    "monotonic stack": "Monotonic Stack",
    "monotonic queue": "Monotonic Queue",
}


# ── Keyword sets for classification ─────────────────────────────────────

GRAPH_HOP_KEYWORDS = {
    "prerequisite", "prerequisites", "before learning", "need to know",
    "depend", "dependency", "dependencies", "learning path", "learning order",
    "difference between", "differ", "compare", "comparison", "versus", "vs",
    "tradeoff", "trade-off", "trade off", "alternative",
    "related to", "neighbor", "neighbours", "neighbors",
}

CODE_TEMPLATE_KEYWORDS = {
    "template", "implementation", "implement", "boilerplate",
    "c++ code", "cpp code", "code for", "give me code",
    "code template", "snippet", "how to code", "write code",
    "source code", "sample code", "example code",
}

CONTRADICTION_KEYWORDS = {
    "contradiction", "conflicting", "which is correct",
    "inconsistent", "disagree", "wrong complexity",
    "o(n) or o(n log n)", "o(log n) or o(n)",
    "is it o(", "really o(",
}

COMPLEXITY_KEYWORDS = {
    "complexity", "time complexity", "space complexity",
    "big o", "big-o", "asymptotic", "runtime",
}


class QueryRouter:
    """
    Deterministic query classifier for the GraphRAG system.

    Routes queries to the optimal retrieval strategy based on keyword
    analysis and pattern detection.
    """

    def __init__(self, known_patterns: list[str] | None = None):
        """
        Args:
            known_patterns: Optional list of pattern names in the graph.
                If not provided, uses the built-in PATTERN_ALIASES.
        """
        self._aliases = dict(PATTERN_ALIASES)
        if known_patterns:
            for p in known_patterns:
                self._aliases[p.lower()] = p

    def classify(self, query: str) -> RoutingDecision:
        """
        Classify a query into a retrieval strategy.

        Returns a RoutingDecision with the query type, detected patterns,
        and optional pattern_a / pattern_b for comparison queries.
        """
        q_lower = query.lower().strip()
        detected = self._detect_patterns(q_lower)

        # ── Stage 1: Contradiction detection ─────────────────────────
        if self._has_keywords(q_lower, CONTRADICTION_KEYWORDS) or re.search(r"o\(.+?\)\s+or\s+o\(.+?\)", q_lower):
            return RoutingDecision(
                query=query,
                query_type=QueryType.CONTRADICTION,
                confidence=0.85,
                detected_patterns=detected,
                reason="Query contains contradiction/conflict keywords or alternative complexity claims.",
                target_pattern=detected[0] if detected else None,
            )

        # ── Stage 2: Graph-hop (comparisons & prerequisites) ────────
        if self._has_keywords(q_lower, GRAPH_HOP_KEYWORDS):
            # Check for A vs B comparison
            a, b = self._detect_comparison_pair(q_lower, detected)
            if a and b:
                return RoutingDecision(
                    query=query,
                    query_type=QueryType.GRAPH_HOP,
                    confidence=0.95,
                    detected_patterns=detected,
                    reason=f"Comparison query between '{a}' and '{b}'.",
                    pattern_a=a,
                    pattern_b=b,
                )

            # Prerequisite chain query
            if self._has_keywords(q_lower, {"prerequisite", "before learning",
                                            "need to know", "learning path",
                                            "learning order", "depend"}):
                target = detected[-1] if detected else None
                return RoutingDecision(
                    query=query,
                    query_type=QueryType.GRAPH_HOP,
                    confidence=0.90,
                    detected_patterns=detected,
                    reason=f"Prerequisite/dependency query for '{target}'.",
                    target_pattern=target,
                )

            # Generic graph-hop
            return RoutingDecision(
                query=query,
                query_type=QueryType.GRAPH_HOP,
                confidence=0.80,
                detected_patterns=detected,
                reason="Query contains graph traversal keywords.",
                target_pattern=detected[0] if detected else None,
            )

        # ── Stage 3: Code template requests ──────────────────────────
        if self._has_keywords(q_lower, CODE_TEMPLATE_KEYWORDS):
            return RoutingDecision(
                query=query,
                query_type=QueryType.CODE_TEMPLATE,
                confidence=0.90,
                detected_patterns=detected,
                reason="Query requests code template/implementation.",
                target_pattern=detected[0] if detected else None,
            )

        # ── Stage 4: Complexity lookup → hybrid (graph + vector) ─────
        if self._has_keywords(q_lower, COMPLEXITY_KEYWORDS) and detected:
            return RoutingDecision(
                query=query,
                query_type=QueryType.HYBRID,
                confidence=0.85,
                detected_patterns=detected,
                reason="Complexity query — graph property lookup + vector context.",
                target_pattern=detected[0],
            )

        # ── Stage 5: Hybrid if patterns detected ────────────────────
        if len(detected) >= 1:
            return RoutingDecision(
                query=query,
                query_type=QueryType.HYBRID,
                confidence=0.70,
                detected_patterns=detected,
                reason="Pattern(s) detected; using hybrid graph+vector retrieval.",
                target_pattern=detected[0],
            )

        # ── Stage 6: Pure semantic fallback ──────────────────────────
        return RoutingDecision(
            query=query,
            query_type=QueryType.SEMANTIC,
            confidence=0.60,
            detected_patterns=detected,
            reason="No specific patterns or graph keywords detected; using vector search.",
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _detect_patterns(self, q_lower: str) -> list[str]:
        """Find all known algorithmic patterns mentioned in the query."""
        found: dict[str, int] = {}  # canonical_name → first occurrence position
        for alias, canonical in sorted(
            self._aliases.items(), key=lambda x: -len(x[0])
        ):
            pos = q_lower.find(alias)
            if pos >= 0 and canonical not in found:
                found[canonical] = pos
        # Return in order of appearance
        return [name for name, _ in sorted(found.items(), key=lambda x: x[1])]

    def _has_keywords(self, q_lower: str, keywords: set[str]) -> bool:
        """Check if any keyword appears in the query."""
        return any(kw in q_lower for kw in keywords)

    def _detect_comparison_pair(
        self, q_lower: str, detected: list[str]
    ) -> tuple[Optional[str], Optional[str]]:
        """Detect if the query is comparing two patterns."""
        if len(detected) >= 2:
            return detected[0], detected[1]

        # Try "X vs Y" / "X versus Y" / "X or Y" patterns
        vs_match = re.search(
            r"(\w[\w\s]+?)\s+(?:vs\.?|versus|or|compared to|differ\w* from)\s+(\w[\w\s]+)",
            q_lower,
        )
        if vs_match:
            left = self._resolve_pattern(vs_match.group(1).strip())
            right = self._resolve_pattern(vs_match.group(2).strip())
            if left and right:
                return left, right
        return None, None

    def _resolve_pattern(self, text: str) -> Optional[str]:
        """Try to resolve a text fragment to a known pattern name."""
        text_lower = text.lower().strip()
        # Direct match
        if text_lower in self._aliases:
            return self._aliases[text_lower]
        # Substring match (best effort)
        for alias, canonical in sorted(
            self._aliases.items(), key=lambda x: -len(x[0])
        ):
            if alias in text_lower:
                return canonical
        return None


# ── CLI demo ─────────────────────────────────────────────────────────────

def demo():
    """Classify the benchmark queries from test_queries.json."""
    router = QueryRouter()

    test_queries = [
        "How does a segment tree differ from a Fenwick tree for range queries?",
        "What do I need to know before learning suffix arrays?",
        "Give me a C++ template for two-pointers on a sorted array",
        "What's the time complexity of a sparse table?",
        "Is a segment tree O(log n) or O(n log n) per query?",
        "Explain the Aho-Corasick automaton for multi-pattern string matching",
        "How does sliding window work for substring problems?",
        "What problems can I solve with binary search on the answer?",
    ]

    print("=" * 72)
    print("Phase 4 — Query Router Classification Demo")
    print("=" * 72)

    for q in test_queries:
        decision = router.classify(q)
        print(f"\n  Q: \"{q}\"")
        print(f"  → Type:       {decision.query_type.value}")
        print(f"  → Confidence: {decision.confidence:.0%}")
        print(f"  → Patterns:   {decision.detected_patterns}")
        if decision.pattern_a:
            print(f"  → Compare:    {decision.pattern_a} ↔ {decision.pattern_b}")
        if decision.target_pattern:
            print(f"  → Target:     {decision.target_pattern}")
        print(f"  → Reason:     {decision.reason}")

    print(f"\n✓ Classified {len(test_queries)} queries.")


if __name__ == "__main__":
    demo()

