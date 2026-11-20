"""
vector_store.py — Hybrid FAISS Dense + Okapi BM25 Vector Store with RRF.

Embeds editorial prose chunks and C++ code blocks into a FAISS IndexFlatL2
index (unit-normalized 384-d embeddings) paired with an in-memory Okapi BM25
inverted index, fusing dense semantic rank and lexical BM25 rank via
Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import json
import math
import pickle
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment, misc]


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_DIR = DATA_DIR / "vector_index"

DEFAULT_MODEL = "all-MiniLM-L6-v2"

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and",
    "in", "that", "have", "for", "on", "with", "as", "at", "by", "from",
    "or", "this", "it", "can", "we", "what", "how", "does", "do", "give",
    "me", "show", "explain", "which", "when", "should", "need", "before",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in _STOPWORDS]


@dataclass
class ChunkMetadata:
    """Metadata for a single embedded chunk."""
    editorial_id: str
    chunk_index: int
    chunk_type: str           # "prose" or "code"
    pattern_ids: list[str]    # linked canonical pattern names
    source_url: str = ""
    text: str = ""


ChunkMetadata.__module__ = "retrieval.vector_store"


class _MetadataUnpickler(pickle.Unpickler):
    """Custom unpickler to robustly load ChunkMetadata regardless of __main__ context."""
    def find_class(self, module, name):
        if name == "ChunkMetadata":
            return ChunkMetadata
        return super().find_class(module, name)


@dataclass
class SearchResult:
    """A single search result from the vector store."""
    text: str
    score: float              # Calibrated squared-L2 distance (lower = more similar)
    metadata: ChunkMetadata
    rank: int = 0
    bm25_score: float = 0.0
    rrf_score: float = 0.0


class VectorStore:
    """FAISS dense vector store + Okapi BM25 lexical scorer with Reciprocal Rank Fusion."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        if faiss is None:
            raise ImportError("faiss-cpu required. Install: pip install faiss-cpu")
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers required. Install: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension: int = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()
        self.index: faiss.IndexFlatL2 = faiss.IndexFlatL2(self.dimension)
        self.metadata: list[ChunkMetadata] = []

        # Okapi BM25 statistics
        self._doc_freqs: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._avgdl: float = 1.0
        self._idf: dict[str, float] = {}

    @property
    def size(self) -> int:
        return self.index.ntotal

    # ── Okapi BM25 ───────────────────────────────────────────────────────

    def _rebuild_bm25(self) -> None:
        self._doc_freqs = []
        self._doc_lens = []
        df: Counter[str] = Counter()
        for meta in self.metadata:
            header = " ".join(meta.pattern_ids)
            tokens = _tokenize(f"{header} {meta.text}")
            counts = Counter(tokens)
            self._doc_freqs.append(counts)
            self._doc_lens.append(len(tokens))
            for term in counts:
                df[term] += 1
        n_docs = len(self.metadata)
        self._avgdl = (sum(self._doc_lens) / n_docs) if n_docs else 1.0
        self._idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def bm25_score_all(self, query: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
        """Compute Okapi BM25 score for every chunk against `query`."""
        if not self._doc_freqs and self.metadata:
            self._rebuild_bm25()
        q_terms = _tokenize(query)
        if not q_terms or not self._doc_freqs:
            return [0.0] * len(self.metadata)
        scores: list[float] = []
        avgdl = max(self._avgdl, 1.0)
        for counts, dl in zip(self._doc_freqs, self._doc_lens):
            s = 0.0
            norm = k1 * (1.0 - b + b * (dl / avgdl))
            for term in q_terms:
                tf = counts.get(term, 0)
                if tf > 0:
                    s += self._idf.get(term, 0.0) * (tf * (k1 + 1.0)) / (tf + norm)
            scores.append(s)
        return scores

    # ── Embedding ────────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into L2-normalized float32 vectors."""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.embed([text])[0]

    # ── Index building ───────────────────────────────────────────────────

    def add_chunks(
        self,
        texts: list[str],
        metadatas: list[ChunkMetadata],
    ):
        """Add chunks with metadata to the FAISS and BM25 indexes."""
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have same length")

        embeddings = self.embed(texts)
        self.index.add(embeddings)
        self.metadata.extend(metadatas)
        self._rebuild_bm25()

    @staticmethod
    def _detect_chunk_patterns(text: str, fallback: list[str]) -> list[str]:
        """Detect canonical pattern names mentioned in a single chunk."""
        from ingestion.extract_entities import _PATTERN_KEYWORDS
        text_lower = text.lower()
        found: list[str] = []
        for keyword, canonical in _PATTERN_KEYWORDS.items():
            if keyword in text_lower and canonical not in found:
                found.append(canonical)
        return found if found else list(fallback[:2])

    def build_from_editorials(self, editorials_path: Path | None = None):
        """
        Build the vector index from the editorial corpus + extraction results.
        """
        editorials_path = editorials_path or DATA_DIR / "editorials.json"
        extractions_path = DATA_DIR / "extractions.json"

        with open(editorials_path) as f:
            editorials = json.load(f)

        pattern_map: dict[str, list[str]] = {}
        if extractions_path.exists():
            with open(extractions_path) as f:
                extractions = json.load(f)
            for tmpl in extractions.get("code_templates", []):
                pattern = tmpl.get("pattern", "")
                if pattern:
                    for ed in editorials:
                        ed_text = ed.get("editorial", "")
                        if tmpl.get("code", "")[:50] in ed_text:
                            pattern_map.setdefault(ed["id"], [])
                            if pattern not in pattern_map[ed["id"]]:
                                pattern_map[ed["id"]].append(pattern)

        from ingestion.extract_entities import _PATTERN_KEYWORDS
        for ed in editorials:
            if ed["id"] not in pattern_map:
                pattern_map[ed["id"]] = []
            text_lower = ed.get("editorial", "").lower()
            for keyword, canonical in _PATTERN_KEYWORDS.items():
                if keyword in text_lower and canonical not in pattern_map[ed["id"]]:
                    pattern_map[ed["id"]].append(canonical)

        all_texts: list[str] = []
        all_metas: list[ChunkMetadata] = []
        code_fence_re = re.compile(r"```(?:cpp|c\+\+)?\s*\n(.*?)```", re.DOTALL)

        for ed in editorials:
            ed_id = ed["id"]
            ed_text = ed.get("editorial", "")
            source_url = ed.get("url", "")
            ed_patterns = pattern_map.get(ed_id, [])

            parts = code_fence_re.split(ed_text)
            code_blocks = [c.strip() for c in code_fence_re.findall(ed_text) if c.strip()]

            chunk_idx = 0
            for part in parts:
                cleaned = part.strip()
                if not cleaned or cleaned in code_blocks:
                    continue
                paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
                for para in paragraphs:
                    if len(para) < 20:
                        continue
                    chunk_pats = self._detect_chunk_patterns(para, ed_patterns)
                    all_texts.append(para)
                    all_metas.append(ChunkMetadata(
                        editorial_id=ed_id,
                        chunk_index=chunk_idx,
                        chunk_type="prose",
                        pattern_ids=chunk_pats,
                        source_url=source_url,
                        text=para,
                    ))
                    chunk_idx += 1

            for code in code_blocks:
                code = code.strip()
                if len(code) < 20:
                    continue
                chunk_pats = self._detect_chunk_patterns(code, ed_patterns)
                all_texts.append(code)
                all_metas.append(ChunkMetadata(
                    editorial_id=ed_id,
                    chunk_index=chunk_idx,
                    chunk_type="code",
                    pattern_ids=chunk_pats,
                    source_url=source_url,
                    text=code,
                ))
                chunk_idx += 1

        print(f"  Embedding {len(all_texts)} chunks ({self.model_name})...")
        self.add_chunks(all_texts, all_metas)
        print(f"  ✓ Index built: {self.size} vectors, dimension={self.dimension}")

    # ── Search (Dense FAISS + Okapi BM25 Reciprocal Rank Fusion) ─────────

    def search(
        self,
        query: str,
        k: int = 5,
        pattern_filter: Optional[str] = None,
        chunk_type: Optional[str] = None,
        use_rrf: bool = True,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """
        Search for chunks using FAISS dense similarity fused with Okapi BM25
        via Reciprocal Rank Fusion (RRF).
        """
        if self.size == 0:
            return []

        query_vec = self.embed_single(query).reshape(1, -1)
        distances, indices = self.index.search(query_vec, self.size)
        bm25_all = self.bm25_score_all(query) if use_rrf else [0.0] * len(self.metadata)

        # Filter candidates first
        candidates: list[tuple[int, float, float]] = []  # (idx, l2_dist, bm25)
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            if pattern_filter and pattern_filter not in meta.pattern_ids:
                continue
            if chunk_type and meta.chunk_type != chunk_type:
                continue
            candidates.append((int(idx), float(dist), bm25_all[int(idx)]))

        if not candidates:
            return []

        if not use_rrf:
            chosen = candidates[:k]
            return [
                SearchResult(
                    text=self.metadata[idx].text,
                    score=dist,
                    metadata=self.metadata[idx],
                    rank=i + 1,
                )
                for i, (idx, dist, _b) in enumerate(chosen)
            ]

        # Dense rank (1-indexed, already sorted by L2 distance ascending)
        dense_rank = {idx: r + 1 for r, (idx, _d, _b) in enumerate(candidates)}
        # BM25 rank (1-indexed, sorted by BM25 descending)
        bm25_sorted = sorted(candidates, key=lambda x: -x[2])
        bm25_rank = {idx: r + 1 for r, (idx, _d, _b) in enumerate(bm25_sorted)}

        scored: list[tuple[float, int, float, float]] = []
        for idx, dist, b_score in candidates:
            rrf = 1.0 / (rrf_k + dense_rank[idx])
            if b_score > 0.0:
                rrf += 0.65 / (rrf_k + bm25_rank[idx])
            # Gentle lexical calibration when strong BM25 match exists,
            # without inflating out-of-corpus queries (where b_score is near 0).
            adj_dist = dist
            if b_score >= 3.0 and dist < 1.15:
                adj_dist = max(0.0, dist - 0.10 * min(1.0, b_score / 10.0))
            scored.append((rrf, idx, adj_dist, b_score))

        scored.sort(key=lambda x: (-x[0], x[2]))

        results: list[SearchResult] = []
        for i, (rrf, idx, adj_dist, b_score) in enumerate(scored[:k]):
            meta = self.metadata[idx]
            results.append(SearchResult(
                text=meta.text,
                score=adj_dist,
                metadata=meta,
                rank=i + 1,
                bm25_score=round(b_score, 3),
                rrf_score=round(rrf, 5),
            ))
        return results

    def get_pattern_ids_from_results(self, results: list[SearchResult]) -> list[str]:
        """Extract unique pattern IDs from search results (for graph follow-up)."""
        seen: set[str] = set()
        pattern_ids: list[str] = []
        for r in results:
            for pid in r.metadata.pattern_ids:
                if pid not in seen:
                    seen.add(pid)
                    pattern_ids.append(pid)
        return pattern_ids

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: Path | None = None):
        """Save the FAISS index and metadata to disk."""
        path = path or INDEX_DIR
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump([asdict(m) for m in self.metadata], f)
        with open(path / "config.json", "w") as f:
            json.dump({"model_name": self.model_name, "dimension": self.dimension}, f)

        print(f"  ✓ Saved vector store to {path} ({self.size} vectors)")

    def load(self, path: Path | None = None):
        """Load a saved FAISS index and metadata, then build the BM25 index."""
        path = path or INDEX_DIR

        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "metadata.pkl", "rb") as f:
            raw_list = _MetadataUnpickler(f).load()
            self.metadata = [
                m if isinstance(m, ChunkMetadata) else ChunkMetadata(**m)
                for m in raw_list
            ]
        self._rebuild_bm25()
        print(f"  ✓ Loaded vector store from {path} ({self.size} vectors)")


# ── CLI demo ─────────────────────────────────────────────────────────────

def demo():
    """Build the index and run sample queries."""
    print("=" * 60)
    print("Building vector store from editorial corpus")
    print("=" * 60)

    store = VectorStore()
    store.build_from_editorials()
    store.save()

    test_queries = [
        ("two pointer technique for sorted arrays", None, None),
        ("segment tree range query implementation", None, None),
        ("sliding window for substring problems", None, "prose"),
        ("fenwick tree binary indexed tree", None, "code"),
        ("prefix sum subarray", "Prefix Sum", None),
    ]

    print(f"\n{'=' * 60}")
    print("Running sample queries (Dense FAISS + BM25 RRF)")
    print("=" * 60)

    for query, pattern_filter, chunk_type in test_queries:
        filter_str = ""
        if pattern_filter:
            filter_str += f" [pattern={pattern_filter}]"
        if chunk_type:
            filter_str += f" [type={chunk_type}]"
        print(f"\n  Q: \"{query}\"{filter_str}")

        results = store.search(query, k=3, pattern_filter=pattern_filter, chunk_type=chunk_type)
        if not results:
            print("    No results found.")
            continue

        for r in results:
            text_preview = r.text[:80].replace("\n", " ")
            patterns = ", ".join(r.metadata.pattern_ids[:3]) or "none"
            print(
                f"    [{r.rank}] dist={r.score:.4f} bm25={r.bm25_score:.2f} "
                f"rrf={r.rrf_score:.4f} type={r.metadata.chunk_type} patterns=[{patterns}]"
            )
            print(f"        \"{text_preview}...\"")

        linked_patterns = store.get_pattern_ids_from_results(results)
        if linked_patterns:
            print(f"    → Graph follow-up patterns: {linked_patterns}")

    print(f"\n✓ Demo complete. Index has {store.size} vectors.")


if __name__ == "__main__":
    demo()
