"""Automated unit and integration test suite for Agentic-GraphRAG."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from agent.contradiction import ContradictionDetector, extract_claims, normalize_big_o
from agent.rag_agent import ContextGrader, RAGAgent
from ingestion.parse_cpp import analyze_code_block, parse_editorial
from retrieval.graph_retriever import InMemoryGraphRetriever
from retrieval.hybrid_retriever import HybridRetriever, RetrievedContext
from retrieval.query_router import QueryRouter, QueryType


class TestQueryRouter(unittest.TestCase):
    def setUp(self) -> None:
        self.router = QueryRouter()

    def test_word_boundary_alias_no_false_positives(self) -> None:
        """Words containing 'bit', 'dp', 'bfs', or 'dfs' as substrings must not false-match."""
        noisy_queries = [
            "Explain how quantum annealing explores an arbitrary 64-bit search space",
            "Is it a good habit to use recursion in orbital mechanics?",
            "Explain the Aho-Corasick automaton for multi-pattern string matching",
        ]
        for q in noisy_queries:
            decision = self.router.classify(q)
            self.assertEqual(
                decision.detected_patterns,
                [],
                f"Expected no detected patterns for '{q}', got {decision.detected_patterns}",
            )
            self.assertEqual(decision.query_type, QueryType.SEMANTIC)

    def test_short_acronyms_match_on_word_boundaries(self) -> None:
        self.assertIn("Fenwick Tree", self.router.classify("How does a BIT support point updates?").detected_patterns)
        self.assertIn("Dynamic Programming", self.router.classify("Explain DP state transitions").detected_patterns)
        self.assertIn("BFS", self.router.classify("Give me a C++ template for BFS").detected_patterns)

    def test_five_way_classification(self) -> None:
        d1 = self.router.classify("How does a segment tree differ from a Fenwick tree?")
        self.assertEqual(d1.query_type, QueryType.GRAPH_HOP)
        self.assertEqual((d1.pattern_a, d1.pattern_b), ("Segment Tree", "Fenwick Tree"))

        d2 = self.router.classify("What do I need to know before learning suffix arrays?")
        self.assertEqual(d2.query_type, QueryType.GRAPH_HOP)
        self.assertEqual(d2.target_pattern, "Suffix Array")

        d3 = self.router.classify("Give me a C++ template for two-pointers on a sorted array")
        self.assertEqual(d3.query_type, QueryType.CODE_TEMPLATE)
        self.assertEqual(d3.target_pattern, "Two Pointers")

        d4 = self.router.classify("What's the time complexity of a sparse table?")
        self.assertEqual(d4.query_type, QueryType.HYBRID)
        self.assertEqual(d4.target_pattern, "Sparse Table")

        d5 = self.router.classify("Is a segment tree O(log n) or O(n log n) per query?")
        self.assertEqual(d5.query_type, QueryType.CONTRADICTION)
        self.assertEqual(d5.target_pattern, "Segment Tree")


class TestGraphAndContradiction(unittest.TestCase):
    def test_in_memory_graph_prerequisites_and_comparisons(self) -> None:
        g = InMemoryGraphRetriever()
        chain = g.prerequisite_chain("Suffix Array")
        self.assertIsNotNone(chain)
        assert chain is not None
        names = [p.name for p in chain.chain]
        self.assertIn("Binary Search", names)
        self.assertEqual(names[-1], "Suffix Array")

        cmp_res = g.compare_patterns("Segment Tree", "Fenwick Tree")
        self.assertIsNotNone(cmp_res)
        assert cmp_res is not None
        self.assertEqual(cmp_res.relationship_type, "ALTERNATIVE_TO")
        self.assertGreaterEqual(len(cmp_res.templates), 2)

    def test_big_o_claim_extraction_and_conflict_detection(self) -> None:
        self.assertEqual(normalize_big_o("O(n * log(n))"), "o(nlogn)")
        c1 = extract_claims(
            "Some sources claim Segment Tree queries are O(n log n) but they are O(log n) per query.",
            pattern="Segment Tree",
        )
        self.assertIn(("query", "o(nlogn)"), c1)
        self.assertIn(("query", "o(logn)"), c1)

        ctxs = [
            RetrievedContext(
                text="Canonical complexity for Segment Tree:\n  Time: O(log n) per query/update",
                source_type="graph",
                relevance_score=1.0,
                pattern_ids=["Segment Tree"],
                metadata={"is_canonical": True},
            ),
            RetrievedContext(
                text="In this blog post, Segment Tree range queries take O(n log n) per query.",
                source_type="vector_prose",
                relevance_score=0.85,
                pattern_ids=["Segment Tree"],
                metadata={"editorial_id": "CF-TEST"},
            ),
        ]
        rep = ContradictionDetector(use_llm=False).detect("Segment Tree", ctxs)
        self.assertTrue(rep.detected)
        self.assertGreaterEqual(len(rep.conflicts), 1)
        self.assertEqual(rep.conflicts[0].operation, "query")

    def test_cpp_parser(self) -> None:
        sample = "Editorial text.\n\n```cpp\nstruct SegTree {\n    vector<int> t;\n    int query(int l, int r) { return 0; }\n};\n```\n"
        parsed = parse_editorial("ED-1", sample)
        self.assertEqual(len(parsed.prose_chunks), 1)
        self.assertEqual(len(parsed.code_blocks), 1)
        cb = analyze_code_block(parsed.code_blocks[0].code)
        self.assertIn("SegTree", cb.data_structures)
        self.assertIn("vector", cb.stl_containers)


class TestAgentAndAblationModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent = RAGAgent(
            hybrid_retriever=HybridRetriever(graph_retriever=InMemoryGraphRetriever())
        )

    def test_rrf_scores_populated(self) -> None:
        res = self.agent.retriever.vector.search("Fenwick tree point update", k=3)
        self.assertGreater(len(res.results), 0)
        self.assertGreater(res.results[0].rrf_score, 0.0)

    def test_out_of_corpus_abstains_with_empty_sources_in_agentic_mode(self) -> None:
        q = "Explain the Aho-Corasick automaton for multi-pattern string matching"
        resp_agentic = self.agent.query(q, k=5, mode="full_agentic")
        self.assertTrue(resp_agentic.abstained)
        self.assertEqual(resp_agentic.sources, [])
        self.assertGreaterEqual(resp_agentic.retries_used, 1)

        # In vector_only baseline mode, abstention is disabled
        resp_vec = self.agent.query(q, k=5, mode="vector_only")
        self.assertFalse(resp_vec.abstained)

    def test_retry_widening_on_partial_context(self) -> None:
        grader = ContextGrader()
        partial_ctxs = [
            RetrievedContext(
                text="Binary search divides the sorted range in half.",
                source_type="vector_prose",
                relevance_score=0.45,
                pattern_ids=["Binary Search"],
            ),
            RetrievedContext(
                text="Unrelated background note on grid simulation.",
                source_type="vector_prose",
                relevance_score=0.20,
                pattern_ids=["Grid Simulation"],
            ),
            RetrievedContext(
                text="Another low-relevance snippet.",
                source_type="vector_prose",
                relevance_score=0.15,
                pattern_ids=[],
            ),
        ]
        decision = self.agent.retriever.router.classify(
            "Compare Binary Search and Suffix Array complexity"
        )
        g = grader.grade(decision.query, partial_ctxs, decision)
        self.assertEqual(g.verdict, "partial")


if __name__ == "__main__":
    unittest.main()
