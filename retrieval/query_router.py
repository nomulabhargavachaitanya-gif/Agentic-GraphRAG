"""
query_router.py — Deterministic 5-Way Query Classifier & Alias Engine.

Classifies incoming natural-language queries into one of five retrieval
strategies and dispatches to the appropriate retriever pipeline:

  1. GRAPH_HOP     — prerequisite chains, pairwise comparisons, neighbor lookups
  2. CODE_TEMPLATE — C++ implementation requests, boilerplate, templates
  3. SEMANTIC      — general algorithmic concept / editorial prose queries
  4. HYBRID        — joint vector similarity + knowledge-graph expansion
  5. CONTRADICTION — conflicting complexity / correctness claims across sources

Uses word-boundary regex alias resolution and intent signatures (no LLM call),
making routing deterministic and sub-millisecond.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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
    detected_patterns: list[str]       # canonical algorithm names in order of appearance
    reason: str                        # human-readable explanation

    # Extracted intent signals
    pattern_a: Optional[str] = None    # for GRAPH_HOP comparisons
    pattern_b: Optional[str] = None
    target_pattern: Optional[str] = None  # for prerequisite or single-pattern lookups


# ── Known pattern aliases ────────────────────────────────────────────────

PATTERN_ALIASES: dict[str, str] = {
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
    "fenwick": "Fenwick Tree",
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
    "breadth-first search": "BFS",
    "breadth first search": "BFS",
    "dfs": "DFS",
    "depth-first search": "DFS",
    "depth first search": "DFS",
    "dijkstra": "Dijkstra",
    "union find": "Union Find",
    "disjoint set": "Union Find",
    "dsu": "Union Find",
    "trie": "Trie",
    "topological sort": "Topological Sort",
    "monotonic stack": "Monotonic Stack",
    "monotonic queue": "Monotonic Queue",
    "greedy": "Greedy",
    "priority queue": "Priority Queue",
    "sweep line": "Sweep Line",
    "grid simulation": "Grid Simulation",
}


# ── Keyword sets for classification ─────────────────────────────────────

GRAPH_HOP_KEYWORDS = {
    "prerequisite", "prerequisites", "before learning", "need to know",
    "depend", "dependency", "dependencies", "learning path", "learning order",
    "difference between", "differ", "compare", "comparison", "versus", "vs",
    "tradeoff", "trade-off", "trade off", "alternative", "alternatives",
    "learn before", "should learn", "need to learn", "required before",
    "learn first", "related to", "neighbor", "neighbours", "neighbors",
}

PREREQ_INTENT_KEYWORDS = {
    "prerequisite", "prerequisites", "before learning", "need to know",
    "learning path", "learning order", "depend", "dependency", "dependencies",
    "learn before", "should learn", "need to learn", "required before",
    "learn first",
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


def _compile_alias_regex(alias: str) -> re.Pattern[str]:
    """Compile a word-boundary regex for an alias (allowing optional plural 's')."""
    # Avoid matching "-bit" as in "8-bit" or "64-bit"
    return re.compile(r"(?<![a-z0-9\-])\b" + re.escape(alias) + r"s?\b(?![\-a-z0-9])", re.I)


class QueryRouter:
    """
    Deterministic query classifier for the GraphRAG system.

    Routes queries to the optimal retrieval strategy based on intent signatures
    and word-boundary pattern alias resolution.
    """

    def __init__(self, known_patterns: list[str] | None = None):
        self._aliases = dict(PATTERN_ALIASES)
        if known_patterns:
            for p in known_patterns:
                self._aliases[p.lower()] = p
        # Sort longest alias first so "binary indexed tree" matches before "bit"
        self._compiled_aliases: list[tuple[str, re.Pattern[str], str]] = [
            (alias, _compile_alias_regex(alias), canonical)
            for alias, canonical in sorted(self._aliases.items(), key=lambda x: -len(x[0]))
        ]

    def classify(self, query: str) -> RoutingDecision:
        """
        Classify a query into a retrieval strategy.

        Returns a RoutingDecision with the query type, detected patterns,
        and optional pattern_a / pattern_b for comparison queries.
        """
        q_lower = query.lower().strip()
        detected = self._detect_patterns(q_lower)

        # ── Stage 1: Contradiction detection ─────────────────────────
        if self._has_keywords(q_lower, CONTRADICTION_KEYWORDS) or re.search(
            r"o\(.+?\)\s+or\s+o\(.+?\)", q_lower
        ):
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
            # Pairwise relationship query ("Is X a prerequisite for Y?")
            # vs comparison query ("Compare X and Y")
            a, b = self._detect_comparison_pair(q_lower, detected)
            if a and b and not self._has_keywords(
                q_lower, {"before learning", "need to know", "learn before", "should i learn before"}
            ):
                return RoutingDecision(
                    query=query,
                    query_type=QueryType.GRAPH_HOP,
                    confidence=0.95,
                    detected_patterns=detected,
                    reason=f"Comparison query between '{a}' and '{b}'.",
                    pattern_a=a,
                    pattern_b=b,
                )

            if self._has_keywords(q_lower, PREREQ_INTENT_KEYWORDS):
                target = detected[-1] if detected else None
                return RoutingDecision(
                    query=query,
                    query_type=QueryType.GRAPH_HOP,
                    confidence=0.90,
                    detected_patterns=detected,
                    reason=f"Prerequisite/dependency query for '{target}'.",
                    target_pattern=target,
                )

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
                confidence=0.75,
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
        """Find all known algorithmic patterns mentioned in the query using word boundaries."""
        found: dict[str, int] = {}
        occupied: list[tuple[int, int]] = []
        for _alias, pattern_re, canonical in self._compiled_aliases:
            m = pattern_re.search(q_lower)
            if not m:
                continue
            span = m.span()
            # Skip if this match lies inside an already-matched longer alias span
            if any(s <= span[0] and span[1] <= e for s, e in occupied):
                continue
            occupied.append(span)
            if canonical not in found or span[0] < found[canonical]:
                found[canonical] = span[0]
        return [name for name, _ in sorted(found.items(), key=lambda x: x[1])]

    def _has_keywords(self, q_lower: str, keywords: set[str]) -> bool:
        """Check if any keyword phrase appears on word boundaries in the query."""
        for kw in keywords:
            if " " in kw or "(" in kw or "-" in kw:
                if kw in q_lower:
                    return True
            elif re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
                return True
        return False

    def _detect_comparison_pair(
        self, q_lower: str, detected: list[str]
    ) -> tuple[Optional[str], Optional[str]]:
        """Detect if the query is comparing two patterns."""
        if len(detected) >= 2:
            return detected[0], detected[1]

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
        matches = self._detect_patterns(text.lower().strip())
        return matches[0] if matches else None


# ── CLI demo ─────────────────────────────────────────────────────────────

def demo():
    """Classify sample benchmark queries."""
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
    print("Query Router Classification Demo")
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
