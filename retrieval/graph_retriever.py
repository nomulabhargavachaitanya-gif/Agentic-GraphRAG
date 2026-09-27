"""
graph_retriever.py — Cypher-based retrieval for GraphRAG Phase 1.

Implements three queries matching test cases from the spec (§7):

  Query 1: "How does a segment tree differ from a Fenwick tree for range queries?"
           → graph-hop via ALTERNATIVE_TO edge, returns both templates.

  Query 2: "What do I need to know before learning suffix arrays?"
           → graph-hop via recursive PREREQUISITE_OF chain.

  Query 4: "What's the time complexity of a sparse table?"
           → property lookup on a Pattern node (+ full-text search fallback).

Each function returns structured dicts ready for downstream synthesis.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag_password")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PatternInfo:
    name: str
    description: str
    time_complexity: str
    space_complexity: str
    category: str


@dataclass
class TemplateInfo:
    pattern: str
    code: str
    explanation: str
    source_url: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result for Query 1: comparing two patterns."""
    pattern_a: PatternInfo
    pattern_b: PatternInfo
    relationship_type: str
    tradeoff: str
    templates: list[TemplateInfo] = field(default_factory=list)


@dataclass
class PrerequisiteChain:
    """Result for Query 2: prerequisite chain to reach a target pattern."""
    target: str
    chain: list[PatternInfo]  # ordered from root prerequisite → target
    depth: int


@dataclass
class ComplexityResult:
    """Result for Query 4: complexity lookup."""
    pattern: PatternInfo
    time_complexity: str
    space_complexity: str


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class GraphRetriever:
    """Cypher-based retriever over the algorithmic knowledge graph."""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    # --- Query 1 ---------------------------------------------------------
    # "How does a segment tree differ from a Fenwick tree for range queries?"

    def compare_patterns(self, pattern_a: str, pattern_b: str) -> Optional[ComparisonResult]:
        """
        Find the ALTERNATIVE_TO (or any direct) relationship between two
        patterns and return both patterns' info, the tradeoff description,
        and their code templates.
        """
        with self.driver.session() as session:
            # Find the relationship (check both directions for ALTERNATIVE_TO)
            result = session.run(
                """
                MATCH (a:Pattern {name: $a}), (b:Pattern {name: $b})
                OPTIONAL MATCH (a)-[r:ALTERNATIVE_TO]-(b)
                RETURN a {.*} AS pa,
                       b {.*} AS pb,
                       type(r) AS rel_type,
                       r.tradeoff AS tradeoff
                """,
                a=pattern_a,
                b=pattern_b,
            )
            record = result.single()
            if record is None:
                return None

            pa = PatternInfo(**{k: record["pa"][k] for k in PatternInfo.__dataclass_fields__})
            pb = PatternInfo(**{k: record["pb"][k] for k in PatternInfo.__dataclass_fields__})

            # Fetch templates for both patterns
            templates = []
            tmpl_result = session.run(
                """
                MATCH (p:Pattern)-[:HAS_TEMPLATE]->(ct:CodeTemplate)
                WHERE p.name IN [$a, $b]
                OPTIONAL MATCH (ct)-[:EXTRACTED_FROM]->(s:Source)
                RETURN p.name AS pattern,
                       ct.code AS code,
                       ct.annotated_explanation AS explanation,
                       s.url AS source_url
                """,
                a=pattern_a,
                b=pattern_b,
            )
            for tmpl_rec in tmpl_result:
                templates.append(
                    TemplateInfo(
                        pattern=tmpl_rec["pattern"],
                        code=tmpl_rec["code"],
                        explanation=tmpl_rec["explanation"],
                        source_url=tmpl_rec["source_url"],
                    )
                )

            return ComparisonResult(
                pattern_a=pa,
                pattern_b=pb,
                relationship_type=record["rel_type"] or "NONE",
                tradeoff=record["tradeoff"] or "No direct relationship found.",
                templates=templates,
            )

    # --- Query 2 ---------------------------------------------------------
    # "What do I need to know before learning suffix arrays?"

    def prerequisite_chain(self, target_pattern: str, max_depth: int = 10) -> Optional[PrerequisiteChain]:
        """
        Traverse PREREQUISITE_OF edges backwards from `target_pattern` to
        find the full prerequisite chain (transitive closure).

        Uses a variable-length path to find ALL ancestors, then returns
        them topologically ordered (roots first).
        """
        with self.driver.session() as session:
            # Variable-length path: (ancestor)-[:PREREQUISITE_OF*1..N]->(target)
            result = session.run(
                """
                MATCH path = (root:Pattern)-[:PREREQUISITE_OF*1..10]->(target:Pattern {name: $target})
                WHERE NOT EXISTS {
                    MATCH (prereq:Pattern)-[:PREREQUISITE_OF]->(root)
                    WHERE prereq <> target  // avoid cycles
                }
                WITH path, length(path) AS depth
                ORDER BY depth DESC
                LIMIT 1
                UNWIND nodes(path) AS node
                RETURN node {.*} AS pattern_info
                """,
                target=target_pattern,
            )
            records = list(result)
            if not records:
                # Check if the pattern itself exists
                exists = session.run(
                    "MATCH (p:Pattern {name: $name}) RETURN p",
                    name=target_pattern,
                ).single()
                if exists:
                    # Pattern exists but has no prerequisites
                    return PrerequisiteChain(target=target_pattern, chain=[], depth=0)
                return None

            chain = []
            for rec in records:
                info = rec["pattern_info"]
                chain.append(
                    PatternInfo(**{k: info[k] for k in PatternInfo.__dataclass_fields__})
                )

            return PrerequisiteChain(
                target=target_pattern,
                chain=chain,
                depth=len(chain) - 1,  # edges, not nodes
            )

    # --- Query 4 ---------------------------------------------------------
    # "What's the time complexity of a sparse table?"

    def pattern_complexity(self, pattern_name: str) -> Optional[ComplexityResult]:
        """
        Look up time and space complexity for a named pattern.
        Falls back to full-text search if exact name doesn't match.
        """
        with self.driver.session() as session:
            # Try exact match first
            result = session.run(
                """
                MATCH (p:Pattern {name: $name})
                RETURN p {.*} AS info
                """,
                name=pattern_name,
            )
            record = result.single()

            # Fallback to full-text search
            if record is None:
                result = session.run(
                    """
                    CALL db.index.fulltext.queryNodes('pattern_description', $query)
                    YIELD node, score
                    WHERE score > 0.5
                    RETURN node {.*} AS info
                    ORDER BY score DESC
                    LIMIT 1
                    """,
                    query=pattern_name,
                )
                record = result.single()

            if record is None:
                return None

            info = record["info"]
            pi = PatternInfo(**{k: info[k] for k in PatternInfo.__dataclass_fields__})

            return ComplexityResult(
                pattern=pi,
                time_complexity=pi.time_complexity,
                space_complexity=pi.space_complexity,
            )

    # --- General utilities ------------------------------------------------

    def get_pattern(self, name: str) -> Optional[PatternInfo]:
        """Fetch a single pattern by exact name."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (p:Pattern {name: $name}) RETURN p {.*} AS info",
                name=name,
            )
            record = result.single()
            if record is None:
                return None
            info = record["info"]
            return PatternInfo(**{k: info[k] for k in PatternInfo.__dataclass_fields__})

    def get_neighbors(self, pattern_name: str, rel_type: str | None = None) -> list[dict]:
        """Get all neighbors of a pattern, optionally filtered by relationship type."""
        rel_filter = f":{rel_type}" if rel_type else ""
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH (p:Pattern {{name: $name}})-[r{rel_filter}]-(neighbor:Pattern)
                RETURN neighbor.name AS name,
                       type(r) AS relationship,
                       properties(r) AS rel_props
                """,
                name=pattern_name,
            )
            return [dict(record) for record in result]

    def get_template(self, pattern_name: str) -> Optional[TemplateInfo]:
        """Get the C++ template for a pattern."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Pattern {name: $name})-[:HAS_TEMPLATE]->(ct:CodeTemplate)
                OPTIONAL MATCH (ct)-[:EXTRACTED_FROM]->(s:Source)
                RETURN ct.code AS code,
                       ct.annotated_explanation AS explanation,
                       s.url AS source_url
                """,
                name=pattern_name,
            )
            record = result.single()
            if record is None:
                return None
            return TemplateInfo(
                pattern=pattern_name,
                code=record["code"],
                explanation=record["explanation"],
                source_url=record["source_url"],
            )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _pretty(obj) -> str:
    """Pretty-print a dataclass or dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return json.dumps(asdict(obj), indent=2, default=str)
    return json.dumps(obj, indent=2, default=str)


def demo():
    """Run the three spec test queries and print results."""
    retriever = GraphRetriever()

    # ── Query 1 ──────────────────────────────────────────────────────────
    print("=" * 72)
    print("QUERY 1: How does a segment tree differ from a Fenwick tree?")
    print("=" * 72)
    result = retriever.compare_patterns("Segment Tree", "Fenwick Tree")
    if result:
        print(f"\nRelationship: {result.relationship_type}")
        print(f"Tradeoff: {result.tradeoff}")
        print(f"\n--- {result.pattern_a.name} ---")
        print(f"  Time:  {result.pattern_a.time_complexity}")
        print(f"  Space: {result.pattern_a.space_complexity}")
        print(f"  Desc:  {result.pattern_a.description}")
        print(f"\n--- {result.pattern_b.name} ---")
        print(f"  Time:  {result.pattern_b.time_complexity}")
        print(f"  Space: {result.pattern_b.space_complexity}")
        print(f"  Desc:  {result.pattern_b.description}")
        print(f"\n{len(result.templates)} code template(s) returned.")
        for t in result.templates:
            print(f"\n  [{t.pattern}] (source: {t.source_url})")
            print(f"  {t.explanation[:100]}...")
    else:
        print("  No relationship found.")

    # ── Query 2 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("QUERY 2: What do I need to know before learning suffix arrays?")
    print("=" * 72)
    result = retriever.prerequisite_chain("Suffix Array")
    if result:
        print(f"\nPrerequisite chain to '{result.target}' (depth={result.depth}):\n")
        for i, p in enumerate(result.chain):
            prefix = "  └─► " if i == len(result.chain) - 1 else "  ├── "
            suffix = "  ★ (target)" if p.name == result.target else ""
            print(f"{prefix}{p.name}  [{p.time_complexity}]{suffix}")
    else:
        print("  Pattern not found or no prerequisites.")

    # ── Query 4 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("QUERY 4: What's the time complexity of a sparse table?")
    print("=" * 72)
    result = retriever.pattern_complexity("Sparse Table")
    if result:
        print(f"\n  Pattern: {result.pattern.name}")
        print(f"  Time complexity:  {result.time_complexity}")
        print(f"  Space complexity: {result.space_complexity}")
        print(f"  Description: {result.pattern.description}")
    else:
        print("  Pattern not found.")

    retriever.close()
    print("\n✓ All queries complete.")


if __name__ == "__main__":
    demo()
