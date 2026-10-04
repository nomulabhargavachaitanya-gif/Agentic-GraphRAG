"""
pipeline.py — End-to-end ingestion pipeline.

Orchestrates: load corpus → parse → extract entities → deduplicate → load into Neo4j.

Usage:
    # Full pipeline with rule-based extraction (no API key needed)
    python -m ingestion.pipeline --mode rules

    # Full pipeline with LLM extraction (requires ANTHROPIC_API_KEY)
    python -m ingestion.pipeline --mode llm

    # Dry run: extract but don't write to Neo4j
    python -m ingestion.pipeline --mode rules --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ingestion.scrape import load_local_corpus, Editorial
from ingestion.parse_cpp import parse_editorial, ParsedEditorial
from ingestion.extract_entities import (
    ExtractionResult,
    extract_with_llm,
    extract_with_rules,
    deduplicate_patterns,
)

# Optional Neo4j import (might not be running)
try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag_password")


def run_pipeline(
    mode: str = "rules",
    dry_run: bool = False,
    save_extractions: bool = True,
) -> list[ExtractionResult]:
    """
    Run the full ingestion pipeline.

    Args:
        mode: "rules" for rule-based extraction, "llm" for Claude API.
        dry_run: If True, skip Neo4j loading.
        save_extractions: If True, save extraction results to JSON.

    Returns:
        List of ExtractionResult objects.
    """
    start_time = time.time()
    output_dir = Path(__file__).resolve().parent.parent / "data"

    # ── Step 1: Load corpus ──────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading editorial corpus")
    print("=" * 60)
    editorials = load_local_corpus()
    print(f"  Loaded {len(editorials)} editorials\n")

    # ── Step 2: Parse editorials ─────────────────────────────────────────
    print("=" * 60)
    print("STEP 2: Parsing editorials (prose + code extraction)")
    print("=" * 60)
    parsed: list[ParsedEditorial] = []
    for ed in editorials:
        p = parse_editorial(ed.id, ed.editorial)
        parsed.append(p)
        print(
            f"  [{ed.id}] {len(p.prose_chunks)} prose chunks, "
            f"{len(p.code_blocks)} code blocks"
        )
    print()

    # ── Step 3: Extract entities ─────────────────────────────────────────
    print("=" * 60)
    print(f"STEP 3: Extracting entities (mode={mode})")
    print("=" * 60)
    all_results: list[ExtractionResult] = []
    extract_fn = extract_with_llm if mode == "llm" else extract_with_rules

    for ed in editorials:
        try:
            result = extract_fn(
                editorial_id=ed.id,
                editorial_text=ed.editorial,
                source_url=ed.url,
            )
            all_results.append(result)
            patterns = ", ".join(p.name for p in result.patterns) or "none"
            print(f"  [{ed.id}] → {patterns}")
        except Exception as e:
            print(f"  [{ed.id}] ✗ Extraction failed: {e}")
    print()

    # ── Step 4: Deduplicate ──────────────────────────────────────────────
    print("=" * 60)
    print("STEP 4: Deduplicating patterns")
    print("=" * 60)
    unique_patterns = deduplicate_patterns(all_results)
    all_relationships = []
    seen_rels = set()
    for result in all_results:
        for rel in result.relationships:
            key = (rel.from_pattern, rel.to_pattern, rel.rel_type)
            if key not in seen_rels:
                seen_rels.add(key)
                all_relationships.append(rel)

    all_templates = [t for r in all_results for t in r.code_templates]

    print(f"  Unique patterns:   {len(unique_patterns)}")
    print(f"  Unique relations:  {len(all_relationships)}")
    print(f"  Code templates:    {len(all_templates)}")
    print(f"  Patterns: {', '.join(sorted(unique_patterns.keys()))}")
    print()

    # ── Step 5: Save extractions ─────────────────────────────────────────
    if save_extractions:
        extraction_path = output_dir / "extractions.json"
        with open(extraction_path, "w") as f:
            json.dump(
                {
                    "patterns": {k: asdict(v) for k, v in unique_patterns.items()},
                    "relationships": [asdict(r) for r in all_relationships],
                    "code_templates": [asdict(t) for t in all_templates],
                    "source_editorials": len(editorials),
                    "extraction_mode": mode,
                },
                f,
                indent=2,
            )
        print(f"✓ Saved extractions to {extraction_path}\n")

    # ── Step 6: Load into Neo4j ──────────────────────────────────────────
    if not dry_run:
        print("=" * 60)
        print("STEP 6: Loading into Neo4j")
        print("=" * 60)
        if not HAS_NEO4J:
            print("  ✗ neo4j Python driver not installed. Skipping graph load.")
        else:
            try:
                _load_extractions_to_neo4j(
                    unique_patterns, all_relationships, all_templates, all_results
                )
            except Exception as e:
                print(f"  ✗ Neo4j connection failed: {e}")
                print("  (Is Docker running? Try: docker compose up -d)")
        print()
    else:
        print("(Dry run — skipping Neo4j load)\n")

    elapsed = time.time() - start_time
    print(f"✓ Pipeline complete in {elapsed:.1f}s")
    return all_results


def _load_extractions_to_neo4j(
    patterns: dict,
    relationships: list,
    templates: list,
    results: list[ExtractionResult],
):
    """Write extracted entities to Neo4j."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print(f"  ✓ Connected to Neo4j at {NEO4J_URI}")

    with driver.session() as session:
        # Merge patterns
        for name, pat in patterns.items():
            session.run(
                """
                MERGE (p:Pattern {name: $name})
                SET p.description      = $desc,
                    p.time_complexity  = $time,
                    p.space_complexity = $space,
                    p.category         = $cat
                """,
                name=name,
                desc=pat.description,
                time=pat.time_complexity,
                space=pat.space_complexity,
                cat=pat.category,
            )
        print(f"  ✓ Merged {len(patterns)} patterns")

        # Merge relationships
        rel_count = 0
        for rel in relationships:
            session.run(
                f"""
                MATCH (a:Pattern {{name: $from_p}})
                MATCH (b:Pattern {{name: $to_p}})
                MERGE (a)-[:{rel.rel_type} {{detail: $detail}}]->(b)
                """,
                from_p=rel.from_pattern,
                to_p=rel.to_pattern,
                detail=rel.detail,
            )
            rel_count += 1
        print(f"  ✓ Merged {rel_count} relationships")

        # Merge source nodes + code templates
        tmpl_count = 0
        for result in results:
            # Source node
            if result.source_url:
                session.run(
                    """
                    MERGE (s:Source {url: $url})
                    SET s.platform = 'ingestion_pipeline',
                        s.author   = 'auto-extracted'
                    """,
                    url=result.source_url,
                )

            # Problem node (from editorial ID)
            editorial_id = result.editorial_id
            session.run(
                """
                MERGE (prob:Problem {id: $id})
                SET prob.source = $source
                """,
                id=editorial_id,
                source="codeforces" if "CF" in editorial_id else "leetcode",
            )

            # Link problem to detected patterns
            for pat in result.patterns:
                session.run(
                    """
                    MATCH (prob:Problem {id: $prob_id})
                    MATCH (pat:Pattern {name: $pat_name})
                    MERGE (prob)-[:SOLVED_WITH]->(pat)
                    """,
                    prob_id=editorial_id,
                    pat_name=pat.name,
                )

            # Code templates
            for tmpl in result.code_templates:
                session.run(
                    """
                    MERGE (ct:CodeTemplate {pattern_name: $pattern, editorial_id: $ed_id})
                    SET ct.code = $code,
                        ct.language = 'cpp',
                        ct.annotated_explanation = $expl
                    """,
                    pattern=tmpl.pattern,
                    ed_id=editorial_id,
                    code=tmpl.code,
                    expl=tmpl.explanation,
                )
                # Link template to pattern
                session.run(
                    """
                    MATCH (pat:Pattern {name: $pattern})
                    MATCH (ct:CodeTemplate {pattern_name: $pattern, editorial_id: $ed_id})
                    MERGE (pat)-[:HAS_TEMPLATE]->(ct)
                    """,
                    pattern=tmpl.pattern,
                    ed_id=editorial_id,
                )
                # Link template to source
                if result.source_url:
                    session.run(
                        """
                        MATCH (ct:CodeTemplate {pattern_name: $pattern, editorial_id: $ed_id})
                        MATCH (s:Source {url: $url})
                        MERGE (ct)-[:EXTRACTED_FROM]->(s)
                        """,
                        pattern=tmpl.pattern,
                        ed_id=editorial_id,
                        url=result.source_url,
                    )
                tmpl_count += 1
        print(f"  ✓ Merged {tmpl_count} code templates with source links")

    driver.close()
    print("  ✓ Neo4j load complete")


def main():
    parser = argparse.ArgumentParser(description="Run the ingestion pipeline")
    parser.add_argument(
        "--mode",
        choices=["rules", "llm"],
        default="rules",
        help="Extraction mode: 'rules' (no API key) or 'llm' (Claude)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract entities but don't load into Neo4j",
    )
    args = parser.parse_args()
    run_pipeline(mode=args.mode, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
