"""
extract_entities.py — LLM-based entity and relationship extraction.

Takes parsed editorials (prose + code blocks) and uses Claude to extract
structured entities (Patterns, Relationships, CodeTemplates) matching
the graph schema defined in the spec.

Supports:
  - Full LLM extraction via Anthropic Claude API
  - Rule-based fallback extraction (no API key needed) for testing/demo

Output schema matches section 8 of the spec:
{
    "patterns": [{"name", "description", "time_complexity", "space_complexity", "category"}],
    "relationships": [{"from", "to", "type", "detail"}],
    "code_templates": [{"pattern", "code", "explanation"}]
}
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# ── Output types ─────────────────────────────────────────────────────────

@dataclass
class ExtractedPattern:
    name: str
    description: str
    time_complexity: str
    space_complexity: str
    category: str


@dataclass
class ExtractedRelationship:
    from_pattern: str
    to_pattern: str
    rel_type: str  # PREREQUISITE_OF | ALTERNATIVE_TO | SAME_COMPLEXITY_CLASS
    detail: str


@dataclass
class ExtractedTemplate:
    pattern: str
    code: str
    explanation: str


@dataclass
class ExtractionResult:
    """Full extraction result for one editorial."""
    editorial_id: str
    patterns: list[ExtractedPattern] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    code_templates: list[ExtractedTemplate] = field(default_factory=list)
    source_url: str = ""
    extraction_method: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


# ── LLM extraction prompt ───────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are an expert competitive programming analyst. Given an editorial (explanation + code), extract structured entities for a knowledge graph.

You MUST return valid JSON matching this exact schema:
{
    "patterns": [
        {
            "name": "Pattern Name (e.g. Two Pointers, Segment Tree)",
            "description": "1-2 sentence description of the technique",
            "time_complexity": "e.g. O(n log n)",
            "space_complexity": "e.g. O(n)",
            "category": "one of: array, tree, string, graph, search, dp, greedy, math, other"
        }
    ],
    "relationships": [
        {
            "from": "Pattern A",
            "to": "Pattern B",
            "type": "PREREQUISITE_OF or ALTERNATIVE_TO or SAME_COMPLEXITY_CLASS",
            "detail": "Brief explanation of the relationship"
        }
    ],
    "code_templates": [
        {
            "pattern": "Pattern Name this implements",
            "code": "The C++ code (copy from editorial)",
            "explanation": "What this code does and how"
        }
    ]
}

Rules:
- Use CANONICAL pattern names: "Two Pointers", "Sliding Window", "Binary Search", "Prefix Sum", "Fenwick Tree", "Segment Tree", "Sparse Table", "Suffix Array", "DFS", "BFS", "Dynamic Programming", "Greedy", etc.
- Only extract patterns that are actually used in the solution, not just mentioned.
- If the editorial mentions alternatives, create ALTERNATIVE_TO relationships.
- If pattern A is needed to understand pattern B, create PREREQUISITE_OF.
- Extract the main code block as a template. Skip trivial I/O code.
- If no standard pattern applies, use "other" category and a descriptive name.
"""


def extract_with_llm(editorial_id: str, editorial_text: str, source_url: str = "") -> ExtractionResult:
    """
    Extract entities using Claude API.

    Requires ANTHROPIC_API_KEY in environment.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package required for LLM extraction. "
            "Install: pip install anthropic"
        )

    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Add it to .env or environment. "
            "Alternatively, use extract_with_rules() for rule-based extraction."
        )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_prompt = f"""Analyze this competitive programming editorial and extract entities.

Editorial ID: {editorial_id}

{editorial_text}

Return ONLY the JSON object, no markdown fences, no extra text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=EXTRACTION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Parse the response
    raw_text = response.content[0].text.strip()

    # Handle potential markdown fencing
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*\n?", "", raw_text)
        raw_text = re.sub(r"\n?```\s*$", "", raw_text)

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"  ✗ Failed to parse LLM response for {editorial_id}: {e}")
        print(f"  Raw response: {raw_text[:200]}...")
        return ExtractionResult(editorial_id=editorial_id, extraction_method="llm_failed")

    return _parse_extraction_json(editorial_id, data, source_url, method="llm")


# ── Rule-based fallback extraction ───────────────────────────────────────

# Known pattern keywords → canonical names
_PATTERN_KEYWORDS: dict[str, str] = {
    "two pointer": "Two Pointers",
    "two-pointer": "Two Pointers",
    "sliding window": "Sliding Window",
    "binary search": "Binary Search",
    "lower_bound": "Binary Search",
    "upper_bound": "Binary Search",
    "prefix sum": "Prefix Sum",
    "prefix[": "Prefix Sum",
    "fenwick": "Fenwick Tree",
    "binary indexed tree": "Fenwick Tree",
    "bit.": "Fenwick Tree",
    "segment tree": "Segment Tree",
    "segtree": "Segment Tree",
    "sparse table": "Sparse Table",
    "suffix array": "Suffix Array",
    "suffix_array": "Suffix Array",
    "dfs": "DFS",
    "bfs": "BFS",
    "dynamic programming": "Dynamic Programming",
    "greedy": "Greedy",
    "priority_queue": "Priority Queue",
    "sweep line": "Sweep Line",
}

_CATEGORY_MAP: dict[str, str] = {
    "Two Pointers": "array",
    "Sliding Window": "array",
    "Binary Search": "search",
    "Prefix Sum": "array",
    "Fenwick Tree": "tree",
    "Segment Tree": "tree",
    "Sparse Table": "tree",
    "Suffix Array": "string",
    "DFS": "graph",
    "BFS": "graph",
    "Dynamic Programming": "dp",
    "Greedy": "greedy",
    "Priority Queue": "tree",
    "Sweep Line": "array",
}

# Complexity extraction from editorial text
_COMPLEXITY_RE = re.compile(r"[Tt]ime[:\s]*(?:complexity[:\s]*)?(O\([^)]+\))")
_SPACE_RE = re.compile(r"[Ss]pace[:\s]*(?:complexity[:\s]*)?(O\([^)]+\))")


def extract_with_rules(editorial_id: str, editorial_text: str, source_url: str = "") -> ExtractionResult:
    """
    Rule-based entity extraction. No LLM needed.

    Uses keyword matching to identify patterns and regex to extract
    complexities. Less accurate than LLM but works offline.
    """
    text_lower = editorial_text.lower()

    # Identify patterns mentioned
    detected_patterns: dict[str, str] = {}  # canonical_name → first match context
    for keyword, canonical in _PATTERN_KEYWORDS.items():
        if keyword in text_lower:
            if canonical not in detected_patterns:
                # Grab surrounding context (±100 chars)
                idx = text_lower.index(keyword)
                context = editorial_text[max(0, idx - 100): idx + 100]
                detected_patterns[canonical] = context

    # Extract complexities from the full text
    time_match = _COMPLEXITY_RE.search(editorial_text)
    space_match = _SPACE_RE.search(editorial_text)
    time_complexity = time_match.group(1) if time_match else "unknown"
    space_complexity = space_match.group(1) if space_match else "unknown"

    # Build pattern objects
    patterns = []
    for name in detected_patterns:
        patterns.append(ExtractedPattern(
            name=name,
            description=f"Detected via keyword matching in editorial {editorial_id}",
            time_complexity=time_complexity,
            space_complexity=space_complexity,
            category=_CATEGORY_MAP.get(name, "other"),
        ))

    # Extract code blocks as templates
    code_blocks = re.findall(r"```(?:cpp|c\+\+)?\s*\n(.*?)```", editorial_text, re.DOTALL)
    templates = []
    primary_pattern = patterns[0].name if patterns else "Unknown"
    for i, code in enumerate(code_blocks):
        templates.append(ExtractedTemplate(
            pattern=primary_pattern,
            code=code.strip(),
            explanation=f"Code block {i + 1} from editorial {editorial_id}",
        ))

    # Infer relationships between detected patterns
    relationships = _infer_relationships(list(detected_patterns.keys()))

    return ExtractionResult(
        editorial_id=editorial_id,
        patterns=patterns,
        relationships=relationships,
        code_templates=templates,
        source_url=source_url,
        extraction_method="rule_based",
    )


def _infer_relationships(pattern_names: list[str]) -> list[ExtractedRelationship]:
    """
    Infer relationships between co-occurring patterns using domain knowledge.
    """
    rels: list[ExtractedRelationship] = []

    # Known prerequisite pairs
    known_prereqs = {
        ("Prefix Sum", "Fenwick Tree"),
        ("Prefix Sum", "Sparse Table"),
        ("Binary Search", "Suffix Array"),
        ("Binary Search", "Segment Tree"),
        ("Two Pointers", "Sliding Window"),
    }

    # Known alternatives
    known_alts = {
        ("Segment Tree", "Fenwick Tree"),
        ("Segment Tree", "Sparse Table"),
        ("Two Pointers", "Sliding Window"),
    }

    name_set = set(pattern_names)
    for a, b in known_prereqs:
        if a in name_set and b in name_set:
            rels.append(ExtractedRelationship(
                from_pattern=a,
                to_pattern=b,
                rel_type="PREREQUISITE_OF",
                detail=f"{a} is a prerequisite for understanding {b}",
            ))

    for a, b in known_alts:
        if a in name_set and b in name_set:
            rels.append(ExtractedRelationship(
                from_pattern=a,
                to_pattern=b,
                rel_type="ALTERNATIVE_TO",
                detail=f"{a} and {b} can solve overlapping problem classes",
            ))

    return rels


# ── Shared JSON parser ──────────────────────────────────────────────────

def _parse_extraction_json(
    editorial_id: str,
    data: dict,
    source_url: str,
    method: str,
) -> ExtractionResult:
    """Parse the extraction JSON into typed dataclasses."""
    patterns = [
        ExtractedPattern(**p)
        for p in data.get("patterns", [])
    ]
    relationships = [
        ExtractedRelationship(
            from_pattern=r["from"],
            to_pattern=r["to"],
            rel_type=r["type"],
            detail=r.get("detail", ""),
        )
        for r in data.get("relationships", [])
    ]
    templates = [
        ExtractedTemplate(**t)
        for t in data.get("code_templates", [])
    ]

    return ExtractionResult(
        editorial_id=editorial_id,
        patterns=patterns,
        relationships=relationships,
        code_templates=templates,
        source_url=source_url,
        extraction_method=method,
    )


# ── Deduplication ────────────────────────────────────────────────────────

def deduplicate_patterns(results: list[ExtractionResult]) -> dict[str, ExtractedPattern]:
    """
    Deduplicate patterns across multiple extraction results.

    Uses canonical name matching. In Phase 3+ this will use embedding
    similarity to catch near-duplicates like "Two Pointer" vs "Two Pointers".
    """
    seen: dict[str, ExtractedPattern] = {}
    # Normalize: strip whitespace, title-case
    for result in results:
        for pattern in result.patterns:
            key = pattern.name.strip().title()
            # Keep the pattern with the most informative description
            if key not in seen or len(pattern.description) > len(seen[key].description):
                pattern.name = key
                seen[key] = pattern
    return seen


# ── CLI demo ─────────────────────────────────────────────────────────────

def demo():
    """Run rule-based extraction on the local corpus."""
    from pathlib import Path

    corpus_path = Path(__file__).resolve().parent.parent / "data" / "editorials.json"
    with open(corpus_path) as f:
        editorials = json.load(f)

    print(f"Extracting entities from {len(editorials)} editorials (rule-based)...\n")

    all_results: list[ExtractionResult] = []
    for entry in editorials:
        result = extract_with_rules(
            editorial_id=entry["id"],
            editorial_text=entry["editorial"],
            source_url=entry.get("url", ""),
        )
        all_results.append(result)
        patterns_str = ", ".join(p.name for p in result.patterns) or "none detected"
        print(
            f"  [{entry['id']}] patterns: {patterns_str} | "
            f"{len(result.code_templates)} templates, "
            f"{len(result.relationships)} relationships"
        )

    # Deduplicate
    unique_patterns = deduplicate_patterns(all_results)
    all_rels = [r for res in all_results for r in res.relationships]
    all_templates = [t for res in all_results for t in res.code_templates]

    print(f"\n--- Extraction Summary ---")
    print(f"  Unique patterns:  {len(unique_patterns)}")
    print(f"  Relationships:    {len(all_rels)}")
    print(f"  Code templates:   {len(all_templates)}")
    print(f"  Patterns found:   {', '.join(sorted(unique_patterns.keys()))}")


if __name__ == "__main__":
    demo()
