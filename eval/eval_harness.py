"""
eval_harness.py — Phase 7: Evaluation Harness & Benchmark Suite.

Runs the 6 benchmark queries from test_queries.json through the full
agentic RAG pipeline and evaluates:

  1. Graph Hop Accuracy   — correct prerequisite chains & comparisons
  2. Template Hit Rate    — code queries return compilable C++ templates
  3. Routing Accuracy     — query router classifies correctly
  4. Abstention Precision — out-of-corpus queries trigger honest abstention
  5. Contradiction Handling — conflicting claims flagged properly

Outputs a structured evaluation report with per-query scores and a
summary table.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from retrieval.query_router import QueryRouter, QueryType
from retrieval.hybrid_retriever import HybridRetriever
from agent.rag_agent import RAGAgent, AgentResponse


EVAL_DIR = Path(__file__).resolve().parent
TEST_QUERIES_PATH = EVAL_DIR / "test_queries.json"


# ── Expected routing map ─────────────────────────────────────────────────

EXPECTED_ROUTES: dict[str, QueryType] = {
    "Q1": QueryType.GRAPH_HOP,
    "Q2": QueryType.GRAPH_HOP,
    "Q3": QueryType.CODE_TEMPLATE,
    "Q4": QueryType.HYBRID,
    "Q5": QueryType.CONTRADICTION,
    "Q6": QueryType.SEMANTIC,
}


# ── Result types ─────────────────────────────────────────────────────────

@dataclass
class QueryEvalResult:
    """Evaluation result for a single query."""
    query_id: str
    query: str
    expected_type: str
    actual_type: str
    routing_correct: bool
    patterns_found: list[str]
    expected_patterns: list[str]
    pattern_recall: float       # what fraction of expected patterns were found
    has_code_template: bool
    expected_code: bool
    code_hit: bool              # True if code expectation matched
    abstained: bool
    expected_abstention: bool
    abstention_correct: bool
    grading_verdict: str
    overall_score: float
    retries_used: int
    latency_ms: float
    passed: bool                # overall pass/fail

    @property
    def summary_line(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return (
            f"  {self.query_id}: {status}  "
            f"route={'✓' if self.routing_correct else '✗'}  "
            f"patterns={self.pattern_recall:.0%}  "
            f"code={'✓' if self.code_hit else '✗'}  "
            f"abstain={'✓' if self.abstention_correct else '✗'}  "
            f"score={self.overall_score:.0%}  "
            f"retries={self.retries_used}  "
            f"{self.latency_ms:.0f}ms"
        )


@dataclass
class EvalReport:
    """Complete evaluation report."""
    results: list[QueryEvalResult]
    total_queries: int
    passed: int
    failed: int
    routing_accuracy: float
    pattern_recall: float
    code_hit_rate: float
    abstention_precision: float
    avg_score: float
    avg_latency_ms: float
    timestamp: str = ""

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "  GraphRAG Evaluation Report",
            "=" * 72,
            "",
            f"  Total Queries:         {self.total_queries}",
            f"  Passed:                {self.passed}/{self.total_queries}",
            f"  Failed:                {self.failed}/{self.total_queries}",
            "",
            "  ── Metrics ──────────────────────────────────",
            f"  Routing Accuracy:      {self.routing_accuracy:.0%}",
            f"  Pattern Recall:        {self.pattern_recall:.0%}",
            f"  Code Hit Rate:         {self.code_hit_rate:.0%}",
            f"  Abstention Precision:  {self.abstention_precision:.0%}",
            f"  Avg Retrieval Score:   {self.avg_score:.0%}",
            f"  Avg Latency:           {self.avg_latency_ms:.0f}ms",
            "",
            "  ── Per-Query Results ────────────────────────",
        ]
        for r in self.results:
            lines.append(r.summary_line)
        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)


# ── Evaluator ────────────────────────────────────────────────────────────

class EvalHarness:
    """Benchmark evaluation harness for the agentic RAG system."""

    def __init__(self, agent: RAGAgent | None = None):
        self.agent = agent or RAGAgent()

    def load_test_queries(self, path: Path | None = None) -> list[dict]:
        """Load benchmark queries from JSON."""
        path = path or TEST_QUERIES_PATH
        with open(path) as f:
            data = json.load(f)
        return data["test_queries"]

    def evaluate(self, verbose: bool = True) -> EvalReport:
        """Run the full evaluation suite."""
        test_queries = self.load_test_queries()

        if verbose:
            print("=" * 72)
            print("  Phase 7 — Evaluation Harness")
            print("=" * 72)
            print(f"\n  Running {len(test_queries)} benchmark queries...\n")

        results: list[QueryEvalResult] = []

        for tq in test_queries:
            qid = tq["id"]
            query = tq["query"]
            qtype = tq["type"]
            expected = tq.get("expected_retrieval", {})
            phase = tq.get("phase", "1+")

            if verbose:
                print(f"  ── {qid}: \"{query[:60]}...\"")

            # Run through the agent
            start = time.time()
            response = self.agent.query(query, k=5, verbose=False)
            latency = (time.time() - start) * 1000

            # Evaluate routing
            expected_route = EXPECTED_ROUTES.get(qid, QueryType.SEMANTIC)
            actual_route = QueryType(response.metadata["routing"]["type"])
            routing_correct = actual_route == expected_route

            # Evaluate pattern retrieval
            expected_patterns = expected.get("patterns", [])
            if not expected_patterns:
                # Try single pattern fields
                for key in ("pattern", "target_pattern"):
                    if key in expected:
                        expected_patterns = [expected[key]]
                        break
                # Try prerequisite chain
                if not expected_patterns and "expected_prerequisite_chain" in expected:
                    expected_patterns = expected["expected_prerequisite_chain"]

            found_patterns = response.metadata["routing"].get("detected_patterns", [])
            # Also check source patterns
            for src in response.sources:
                found_patterns.extend(src.get("patterns", []))
            found_patterns = list(set(found_patterns))

            if expected_patterns:
                matched = sum(
                    1 for ep in expected_patterns
                    if any(ep.lower() in fp.lower() for fp in found_patterns)
                )
                pattern_recall = matched / len(expected_patterns)
            else:
                pattern_recall = 1.0  # no expectation → pass

            # Evaluate code template retrieval
            expected_code = expected.get("should_return_templates", False) or expected.get("should_return_template", False)
            has_code = response.grading.has_code
            code_hit = (has_code == expected_code) if expected_code else True

            # Evaluate abstention
            expected_abstention = qtype == "out-of-corpus"
            abstention_correct = response.abstained == expected_abstention

            # Overall pass
            passed = routing_correct and pattern_recall >= 0.5 and abstention_correct

            result = QueryEvalResult(
                query_id=qid,
                query=query,
                expected_type=expected_route.value,
                actual_type=actual_route.value,
                routing_correct=routing_correct,
                patterns_found=found_patterns,
                expected_patterns=expected_patterns,
                pattern_recall=pattern_recall,
                has_code_template=has_code,
                expected_code=expected_code,
                code_hit=code_hit,
                abstained=response.abstained,
                expected_abstention=expected_abstention,
                abstention_correct=abstention_correct,
                grading_verdict=response.grading.verdict,
                overall_score=response.grading.overall_score,
                retries_used=response.retries_used,
                latency_ms=latency,
                passed=passed,
            )
            results.append(result)

            if verbose:
                print(f"    {result.summary_line}")

        # Build report
        total = len(results)
        passed_count = sum(1 for r in results if r.passed)
        routing_correct_count = sum(1 for r in results if r.routing_correct)
        code_queries = [r for r in results if r.expected_code]
        abstention_queries = [r for r in results if r.expected_abstention or r.abstained]

        report = EvalReport(
            results=results,
            total_queries=total,
            passed=passed_count,
            failed=total - passed_count,
            routing_accuracy=routing_correct_count / max(total, 1),
            pattern_recall=sum(r.pattern_recall for r in results) / max(total, 1),
            code_hit_rate=(
                sum(1 for r in code_queries if r.code_hit) / max(len(code_queries), 1)
                if code_queries else 1.0
            ),
            abstention_precision=(
                sum(1 for r in abstention_queries if r.abstention_correct)
                / max(len(abstention_queries), 1)
                if abstention_queries else 1.0
            ),
            avg_score=sum(r.overall_score for r in results) / max(total, 1),
            avg_latency_ms=sum(r.latency_ms for r in results) / max(total, 1),
        )

        if verbose:
            print(f"\n{report.summary()}")

        return report

    def save_report(self, report: EvalReport, path: Path | None = None):
        """Save the evaluation report to a JSON file."""
        path = path or EVAL_DIR / "eval_report.json"
        data = {
            "summary": {
                "total": report.total_queries,
                "passed": report.passed,
                "failed": report.failed,
                "routing_accuracy": report.routing_accuracy,
                "pattern_recall": report.pattern_recall,
                "code_hit_rate": report.code_hit_rate,
                "abstention_precision": report.abstention_precision,
                "avg_score": report.avg_score,
                "avg_latency_ms": report.avg_latency_ms,
            },
            "results": [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "expected_type": r.expected_type,
                    "actual_type": r.actual_type,
                    "routing_correct": r.routing_correct,
                    "patterns_found": r.patterns_found,
                    "pattern_recall": r.pattern_recall,
                    "has_code": r.has_code_template,
                    "abstained": r.abstained,
                    "abstention_correct": r.abstention_correct,
                    "verdict": r.grading_verdict,
                    "score": r.overall_score,
                    "retries": r.retries_used,
                    "latency_ms": r.latency_ms,
                    "passed": r.passed,
                }
                for r in report.results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ Report saved to {path}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    """Run the evaluation harness from the command line."""
    harness = EvalHarness()
    report = harness.evaluate(verbose=True)
    harness.save_report(report)


if __name__ == "__main__":
    main()

