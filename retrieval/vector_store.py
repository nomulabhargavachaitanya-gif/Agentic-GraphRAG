"""
vector_store.py — FAISS-based vector store with sentence-transformer embeddings.

Embeds editorial prose chunks and code blocks, stores them in a FAISS index
with metadata linking each chunk back to the knowledge graph via pattern_id.

Supports:
  - Building the index from parsed editorials
  - Similarity search with optional pattern_id filtering
  - Persistence (save/load from disk)
  - Hybrid retrieval support (returns pattern_ids for graph follow-up)
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
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

# Default embedding model — fast, good quality, 384 dims
DEFAULT_MODEL = "all-MiniLM-L6-v2"


@dataclass
class ChunkMetadata:
    """Metadata for a single embedded chunk."""
    editorial_id: str
    chunk_index: int
    chunk_type: str           # "prose" or "code"
    pattern_ids: list[str]    # linked pattern names from extraction
    source_url: str = ""
    text: str = ""            # the original text (for retrieval display)


@dataclass
class SearchResult:
    """A single search result from the vector store."""
    text: str
    score: float              # L2 distance (lower = more similar)
    metadata: ChunkMetadata
    rank: int = 0


class VectorStore:
    """FAISS-backed vector store with metadata and pattern linking."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        if faiss is None:
            raise ImportError("faiss-cpu required. Install: pip install faiss-cpu")
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers required. Install: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.dimension: int = self.model.get_sentence_embedding_dimension()
        self.index: faiss.IndexFlatL2 = faiss.IndexFlatL2(self.dimension)
        self.metadata: list[ChunkMetadata] = []

    @property
    def size(self) -> int:
        return self.index.ntotal

    # ── Embedding ────────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into vectors."""
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
        """Add chunks with metadata to the index."""
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must have same length")

        embeddings = self.embed(texts)
        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def build_from_editorials(self, editorials_path: Path | None = None):
        """
        Build the vector index from the editorial corpus + extraction results.

        Reads editorials.json for text content and extractions.json for
        pattern linkage (the pattern_id metadata field).
        """
        editorials_path = editorials_path or DATA_DIR / "editorials.json"
        extractions_path = DATA_DIR / "extractions.json"

        with open(editorials_path) as f:
            editorials = json.load(f)

        # Load extraction results for pattern linkage
        pattern_map: dict[str, list[str]] = {}  # editorial_id → [pattern_names]
        if extractions_path.exists():
            with open(extractions_path) as f:
                extractions = json.load(f)
            # Build map from code_templates (which have editorial context)
            for tmpl in extractions.get("code_templates", []):
                pattern = tmpl.get("pattern", "")
                # Templates are numbered per editorial, so we track patterns globally
                if pattern:
                    for ed in editorials:
                        ed_text = ed.get("editorial", "")
                        if tmpl.get("code", "")[:50] in ed_text:
                            pattern_map.setdefault(ed["id"], [])
                            if pattern not in pattern_map[ed["id"]]:
                                pattern_map[ed["id"]].append(pattern)

        # Also do keyword-based pattern detection as fallback
        from ingestion.extract_entities import _PATTERN_KEYWORDS
        for ed in editorials:
            if ed["id"] not in pattern_map:
                pattern_map[ed["id"]] = []
            text_lower = ed.get("editorial", "").lower()
            for keyword, canonical in _PATTERN_KEYWORDS.items():
                if keyword in text_lower and canonical not in pattern_map[ed["id"]]:
                    pattern_map[ed["id"]].append(canonical)

        # Process each editorial
        import re
        all_texts: list[str] = []
        all_metas: list[ChunkMetadata] = []
        code_fence_re = re.compile(r"```(?:cpp|c\+\+)?\s*\n(.*?)```", re.DOTALL)

        for ed in editorials:
            ed_id = ed["id"]
            ed_text = ed.get("editorial", "")
            source_url = ed.get("url", "")
            patterns = pattern_map.get(ed_id, [])

            # Extract prose chunks (split on code fences, then paragraphs)
            parts = code_fence_re.split(ed_text)
            code_blocks = code_fence_re.findall(ed_text)

            chunk_idx = 0
            for i, part in enumerate(parts):
                cleaned = part.strip()
                if not cleaned or cleaned in code_blocks:
                    continue
                # Split long prose into paragraphs
                paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
                for para in paragraphs:
                    if len(para) < 20:  # skip tiny fragments
                        continue
                    all_texts.append(para)
                    all_metas.append(ChunkMetadata(
                        editorial_id=ed_id,
                        chunk_index=chunk_idx,
                        chunk_type="prose",
                        pattern_ids=patterns,
                        source_url=source_url,
                        text=para,
                    ))
                    chunk_idx += 1

            # Add code blocks as separate chunks
            for code in code_blocks:
                code = code.strip()
                if len(code) < 20:
                    continue
                all_texts.append(code)
                all_metas.append(ChunkMetadata(
                    editorial_id=ed_id,
                    chunk_index=chunk_idx,
                    chunk_type="code",
                    pattern_ids=patterns,
                    source_url=source_url,
                    text=code,
                ))
                chunk_idx += 1

        print(f"  Embedding {len(all_texts)} chunks ({self.model_name})...")
        self.add_chunks(all_texts, all_metas)
        print(f"  ✓ Index built: {self.size} vectors, dimension={self.dimension}")

    # ── Search ───────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = 5,
        pattern_filter: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Search for similar chunks.

        Args:
            query: Natural language query.
            k: Number of results to return.
            pattern_filter: If set, only return chunks linked to this pattern.
            chunk_type: If set, filter by "prose" or "code".

        Returns:
            List of SearchResult, sorted by similarity (best first).
        """
        if self.size == 0:
            return []

        query_vec = self.embed_single(query).reshape(1, -1)

        # If filtering, search more candidates then filter
        search_k = k * 5 if (pattern_filter or chunk_type) else k
        search_k = min(search_k, self.size)

        distances, indices = self.index.search(query_vec, search_k)

        results: list[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]

            # Apply filters
            if pattern_filter and pattern_filter not in meta.pattern_ids:
                continue
            if chunk_type and meta.chunk_type != chunk_type:
                continue

            results.append(SearchResult(
                text=meta.text,
                score=float(dist),
                metadata=meta,
                rank=len(results) + 1,
            ))

            if len(results) >= k:
                break

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
            pickle.dump(self.metadata, f)
        with open(path / "config.json", "w") as f:
            json.dump({"model_name": self.model_name, "dimension": self.dimension}, f)

        print(f"  ✓ Saved vector store to {path} ({self.size} vectors)")

    def load(self, path: Path | None = None):
        """Load a saved FAISS index and metadata."""
        path = path or INDEX_DIR

        self.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
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

    # Sample queries
    test_queries = [
        ("two pointer technique for sorted arrays", None, None),
        ("segment tree range query implementation", None, None),
        ("sliding window for substring problems", None, "prose"),
        ("fenwick tree binary indexed tree", None, "code"),
        ("prefix sum subarray", "Prefix Sum", None),
    ]

    print(f"\n{'=' * 60}")
    print("Running sample queries")
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
                f"    [{r.rank}] score={r.score:.4f} type={r.metadata.chunk_type} "
                f"patterns=[{patterns}]"
            )
            print(f"        \"{text_preview}...\"")

        # Show pattern IDs for graph follow-up
        linked_patterns = store.get_pattern_ids_from_results(results)
        if linked_patterns:
            print(f"    → Graph follow-up patterns: {linked_patterns}")

    print(f"\n✓ Demo complete. Index has {store.size} vectors.")


if __name__ == "__main__":
    demo()
