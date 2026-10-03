"""
parse_cpp.py — C++ code block extraction and analysis.

Extracts fenced code blocks from editorial markdown, identifies function
signatures, data structures, and key algorithmic constructs.

Uses regex-based extraction as the primary method, with optional tree-sitter
integration for deeper AST analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class CodeBlock:
    """A single C++ code block extracted from an editorial."""
    code: str
    function_signatures: list[str] = field(default_factory=list)
    data_structures: list[str] = field(default_factory=list)
    stl_containers: list[str] = field(default_factory=list)
    complexity_hints: list[str] = field(default_factory=list)


@dataclass
class ParsedEditorial:
    """An editorial split into prose and code components."""
    editorial_id: str
    prose_chunks: list[str]
    code_blocks: list[CodeBlock]
    raw_text: str


# ── Regex patterns for C++ analysis ──────────────────────────────────────

# Function signatures: return_type function_name(params)
_FUNC_SIG_RE = re.compile(
    r"(?:^|\n)\s*(?:(?:inline|static|virtual|constexpr|auto|void|int|long\s+long|"
    r"bool|string|double|float|vector<[^>]+>|pair<[^>]+>)\s+)"
    r"(\w+)\s*\([^)]*\)",
    re.MULTILINE,
)

# Struct/class definitions
_STRUCT_RE = re.compile(
    r"\b(?:struct|class)\s+(\w+)\s*\{",
    re.MULTILINE,
)

# STL container usage
_STL_RE = re.compile(
    r"\b(vector|map|unordered_map|set|unordered_set|multiset|"
    r"priority_queue|queue|stack|deque|array|bitset)\s*<",
)

# Complexity hints in comments
_COMPLEXITY_RE = re.compile(
    r"O\([^)]+\)",
    re.IGNORECASE,
)

# Fenced code block extraction (```cpp ... ``` or ``` ... ```)
_CODE_FENCE_RE = re.compile(
    r"```(?:cpp|c\+\+)?\s*\n(.*?)```",
    re.DOTALL,
)


def extract_code_blocks(markdown_text: str) -> list[str]:
    """Extract all fenced code blocks from markdown text."""
    return _CODE_FENCE_RE.findall(markdown_text)


def analyze_code_block(code: str) -> CodeBlock:
    """
    Analyze a C++ code block to extract structural information.

    Extracts:
      - Function signatures
      - Struct/class definitions
      - STL container usage
      - Complexity annotations from comments
    """
    # Function signatures
    func_sigs = []
    for match in _FUNC_SIG_RE.finditer(code):
        # Get the full match, clean whitespace
        full_sig = match.group(0).strip()
        # Truncate at the opening brace
        if "{" in full_sig:
            full_sig = full_sig[: full_sig.index("{")].strip()
        func_sigs.append(full_sig)

    # Data structures
    structs = [m.group(1) for m in _STRUCT_RE.finditer(code)]

    # STL containers
    stl = list(set(m.group(1) for m in _STL_RE.finditer(code)))

    # Complexity hints
    complexities = list(set(_COMPLEXITY_RE.findall(code)))

    return CodeBlock(
        code=code.strip(),
        function_signatures=func_sigs,
        data_structures=structs,
        stl_containers=sorted(stl),
        complexity_hints=complexities,
    )


def parse_editorial(editorial_id: str, text: str) -> ParsedEditorial:
    """
    Parse an editorial text into prose chunks and analyzed code blocks.

    Splits on code fences, extracting prose segments and code blocks
    alternately.
    """
    raw_codes = extract_code_blocks(text)
    analyzed_blocks = [analyze_code_block(code) for code in raw_codes]

    # Split prose: everything between code blocks
    prose_parts = _CODE_FENCE_RE.split(text)
    prose_chunks = []
    for part in prose_parts:
        cleaned = part.strip()
        if cleaned and cleaned not in raw_codes:
            # Further split long prose into paragraph-level chunks
            paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
            prose_chunks.extend(paragraphs)

    return ParsedEditorial(
        editorial_id=editorial_id,
        prose_chunks=prose_chunks,
        code_blocks=analyzed_blocks,
        raw_text=text,
    )


def demo():
    """Demo: parse the local editorial corpus."""
    import json
    from pathlib import Path

    corpus_path = Path(__file__).resolve().parent.parent / "data" / "editorials.json"
    with open(corpus_path) as f:
        editorials = json.load(f)

    print(f"Parsing {len(editorials)} editorials...\n")

    total_code_blocks = 0
    total_prose_chunks = 0
    total_structs = 0

    for entry in editorials:
        parsed = parse_editorial(entry["id"], entry["editorial"])
        total_code_blocks += len(parsed.code_blocks)
        total_prose_chunks += len(parsed.prose_chunks)

        structs = []
        funcs = []
        for cb in parsed.code_blocks:
            structs.extend(cb.data_structures)
            funcs.extend(cb.function_signatures)
            total_structs += len(cb.data_structures)

        struct_str = f" structs={structs}" if structs else ""
        func_count = len(funcs)
        print(
            f"  [{entry['id']}] {len(parsed.prose_chunks)} prose chunks, "
            f"{len(parsed.code_blocks)} code blocks, "
            f"{func_count} functions{struct_str}"
        )

    print(f"\n✓ Totals: {total_prose_chunks} prose chunks, "
          f"{total_code_blocks} code blocks, {total_structs} structs detected")


if __name__ == "__main__":
    demo()
