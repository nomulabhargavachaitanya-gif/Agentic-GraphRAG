"""
Phase 6: Contradiction detection.

Compares Big-O complexity claims made by different sources about the same
pattern operation (build / query / update / ...). A conflict is two sources
asserting *different* complexities for the *same* operation.

Pipeline:
  1. Rule-based claim extraction: every O(...) expression is tagged with the
     nearest operation keyword ("query", "update", "build", ...).
  2. Candidate conflicts = same (operation) with different normalised Big-O
     across sources (including the canonical graph property).
  3. If a Gemini key is available, the LLM verifies candidates to filter out
     false positives (e.g. per-query vs. total complexity).

CLI:
  python -m agent.contradiction            # scan corpus, print conflicts
  python -m agent.contradiction --write    # also persist CONTRADICTS edges
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

BIG_O = re.compile(r"O\((?:[^()]|\([^()]*\))*\)")

# canonical operation name -> keywords that signal it
OPERATIONS: dict[str, tuple[str, ...]] = {
    "query": ("query", "queries", "lookup", "lookups"),
    "update": ("update", "updates"),
    "build": ("build", "construction", "preprocessing", "precomputation"),
}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_SUMMARY_PREFIX = re.compile(r"(time|space)\s*(complexity)?\s*:\s*$", re.I)


def normalize_big_o(expr: str) -> str:
    s = expr.lower().replace(" ", "").replace("*", "")
    return s.replace("log(n)", "logn")


def _ops_in(segment: str) -> list[str]:
    seg = segment.lower()
    return [op for op, kws in OPERATIONS.items()
            if any(re.search(r"\b" + re.escape(k) + r"\b", seg) for k in kws)]


def _mentions(sentence: str, pattern: str | None) -> bool:
    if not pattern:
        return True
    from retrieval.query_router import PATTERN_ALIASES
    low = sentence.lower()
    aliases = {pattern.lower()} | {a for a, c in PATTERN_ALIASES.items() if c == pattern}
    return any(re.search(r"\b" + re.escape(a) + r"s?\b", low) for a in aliases)


def extract_claims(text: str, pattern: str | None = None,
                   skip_summaries: bool = True) -> list[tuple[str, str]]:
    """
    Return [(operation, normalised_big_o)] claims about `pattern`.

    Works sentence by sentence. An O(...) is attributed to the operation
    keywords that follow it (up to the next O(...)), else those preceding it.
    "Time: O(..)" / "Space: O(..)" summaries describe a whole problem's
    solution rather than a pattern operation, so they are skipped.
    """
    claims: list[tuple[str, str]] = []
    for sent in _SENT_SPLIT.split(text):
        matches = list(BIG_O.finditer(sent))
        if not matches or not _mentions(sent, pattern):
            continue
        for i, m in enumerate(matches):
            before = sent[matches[i - 1].end() if i else 0: m.start()]
            if skip_summaries and _SUMMARY_PREFIX.search(before):
                continue
            after = sent[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(sent)]
            ops = _ops_in(after) or _ops_in(before)
            for op in ops:
                c = (op, normalize_big_o(m.group(0)))
                if c not in claims:
                    claims.append(c)
    return claims


@dataclass
class SourceClaims:
    source: str               # editorial id or "canonical (graph)"
    url: str
    claims: list[tuple[str, str]]


@dataclass
class Conflict:
    operation: str
    claim_a: str
    source_a: str
    claim_b: str
    source_b: str
    url_a: str = ""
    url_b: str = ""


@dataclass
class ContradictionReport:
    pattern: str | None
    detected: bool
    conflicts: list[Conflict] = field(default_factory=list)
    sources: list[SourceClaims] = field(default_factory=list)
    explanation: str = ""
    method: str = "rules"

    def to_markdown(self) -> str:
        if not self.sources:
            return "_No complexity claims found to compare._"
        lines = []
        if self.detected:
            lines.append(f"**Conflicting complexity claims for {self.pattern}:**")
            for c in self.conflicts:
                lines.append(f"- *{c.operation}*: `{c.claim_a}` ({c.source_a}) vs "
                             f"`{c.claim_b}` ({c.source_b})")
        else:
            lines.append(f"**No conflicting claims found for {self.pattern}** — "
                         "all sources agree on each operation.")
        if self.explanation:
            lines.append(f"\n{self.explanation}")
        return "\n".join(lines)


class ContradictionDetector:
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

    def collect(self, contexts, pattern: str | None = None) -> list[SourceClaims]:
        """Group claims by source from RetrievedContext objects."""
        by_source: dict[str, SourceClaims] = {}
        for ctx in contexts:
            if ctx.metadata.get("is_canonical"):
                name = "canonical (graph)"
            else:
                name = ctx.metadata.get("editorial_id") or ctx.source_url or ctx.source_type
            canonical = bool(ctx.metadata.get("is_canonical"))
            claims = extract_claims(ctx.text, None if canonical else pattern,
                                    skip_summaries=not canonical)
            if not claims:
                continue
            sc = by_source.setdefault(name, SourceClaims(name, ctx.source_url or "", []))
            for c in claims:
                if c not in sc.claims:
                    sc.claims.append(c)
        return list(by_source.values())

    def _rule_conflicts(self, sources: list[SourceClaims]) -> list[Conflict]:
        """Same operation, different Big-O — across sources or within one source."""
        by_op: dict[str, list[tuple[str, str, str]]] = {}
        for sc in sources:
            for op, o in sc.claims:
                by_op.setdefault(op, []).append((o, sc.source, sc.url))
        conflicts: list[Conflict] = []
        for op, items in by_op.items():
            values = {}
            for o, src, url in items:
                values.setdefault(o, (src, url))
            vals = list(values.items())
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    (oa, (sa, ua)), (ob, (sb, ub)) = vals[i], vals[j]
                    conflicts.append(Conflict(op, oa, sa, ob, sb, ua, ub))
        return conflicts

    def _llm_verify(self, pattern: str, conflicts: list[Conflict], contexts) -> tuple[list[Conflict], str] | None:
        from llm import gemini
        if not gemini.available():
            return None
        excerpts = "\n\n".join(
            f"[{ctx.metadata.get('editorial_id') or ('canonical' if ctx.metadata.get('is_canonical') else ctx.source_type)}] {ctx.text[:600]}"
            for ctx in contexts
        )
        cands = "\n".join(f"{i}. {c.operation}: {c.claim_a} ({c.source_a}) vs {c.claim_b} ({c.source_b})"
                          for i, c in enumerate(conflicts))
        prompt = (
            f"Pattern: {pattern}\n\nSource excerpts:\n{excerpts}\n\n"
            f"Candidate complexity conflicts:\n{cands}\n\n"
            "For each candidate decide if it represents a complexity conflict or "
            "disputed claim to surface to the user. Keep a candidate in 'genuine' if:\n"
            "  (a) two sources claim different complexities for the SAME operation "
            "under the same conditions, OR\n"
            "  (b) a source explicitly mentions and refutes a conflicting complexity "
            "claim made by other sources (e.g. 'Some sources claim X is O(n log n) "
            "but it is O(log n)').\n"
            "Exclude a candidate ONLY if the two Big-O expressions actually refer to "
            "different things (e.g. total time vs per-query time, or different problem "
            "variants).\n"
            'Return JSON: {"genuine": [indices], "explanation": "one or two sentences"}'
        )
        try:
            data = json.loads(gemini.generate(prompt, json_mode=True, max_tokens=512))
            keep = {int(i) for i in data.get("genuine", [])}
            return [c for i, c in enumerate(conflicts) if i in keep], data.get("explanation", "")
        except Exception:
            return None

    def detect(self, pattern: str | None, contexts) -> ContradictionReport:
        sources = self.collect(contexts, pattern)
        conflicts = self._rule_conflicts(sources)
        report = ContradictionReport(pattern=pattern, detected=bool(conflicts),
                                     conflicts=conflicts, sources=sources)
        if conflicts and self.use_llm:
            verified = self._llm_verify(pattern or "", conflicts, contexts)
            if verified is not None:
                report.conflicts, report.explanation = verified
                report.detected = bool(report.conflicts)
                report.method = "rules+llm"
        return report


# ── Corpus scan → CONTRADICTS edges ──────────────────────────────────────

def scan(write: bool = False, use_llm: bool = True) -> list[ContradictionReport]:
    from retrieval.hybrid_retriever import HybridRetriever, RetrievedContext

    hr = HybridRetriever()
    detector = ContradictionDetector(use_llm=use_llm)
    patterns = sorted({p for m in hr.vector.store.metadata for p in m.pattern_ids})
    reports = []
    for pat in patterns:
        ctxs = [
            RetrievedContext(text=m.text, source_type=f"vector_{m.chunk_type}", relevance_score=1.0,
                             pattern_ids=m.pattern_ids, source_url=m.source_url,
                             metadata={"editorial_id": m.editorial_id})
            for m in hr.vector.store.metadata if pat in m.pattern_ids
        ]
        rep = detector.detect(pat, ctxs)
        reports.append(rep)
        status = f"{len(rep.conflicts)} conflict(s)" if rep.detected else "consistent"
        print(f"  {pat:22s} {len(rep.sources):2d} sources with claims → {status}")
        for c in rep.conflicts:
            print(f"      {c.operation}: {c.claim_a} ({c.source_a}) vs {c.claim_b} ({c.source_b})")

    if write and hr.has_graph:
        n = 0
        with hr.graph.driver.session() as s:
            s.run("MATCH ()-[r:CONTRADICTS]->() DELETE r")
            for rep in reports:
                for c in rep.conflicts:
                    if not (c.url_a and c.url_b) or c.url_a == c.url_b:
                        continue
                    s.run(
                        "MATCH (a:Source {url:$ua}), (b:Source {url:$ub}) "
                        "MERGE (a)-[r:CONTRADICTS {pattern:$p, operation:$op}]->(b) "
                        "SET r.claim_a=$ca, r.claim_b=$cb",
                        ua=c.url_a, ub=c.url_b, p=rep.pattern, op=c.operation,
                        ca=c.claim_a, cb=c.claim_b,
                    )
                    n += 1
        print(f"\n  ✓ Wrote {n} CONTRADICTS edge(s) to Neo4j")
    return reports


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="persist CONTRADICTS edges")
    ap.add_argument("--no-llm", action="store_true", help="rules only")
    a = ap.parse_args()
    scan(write=a.write, use_llm=not a.no_llm)
