"""
vector_retriever.py — Vector similarity retrieval over editorial chunks.

Wraps VectorStore to provide a retrieval interface matching the graph_retriever
pattern. Returns structured results with pattern linkage for hybrid retrieval.

Usage:
    retriever = VectorRetriever()        # loads saved index
    results = retriever.search("two pointer technique", k=5)
    patterns = retriever.get_linked_patterns(results)  # for graph follow-up
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from retrieval.vector_store import VectorStore, SearchResult, INDEX_DIR


@dataclass
class VectorRetrievalResult:
    """Structured result from vector retrieval."""
    query: str
    results: list[SearchResult]
    linked_pattern_ids: list[str]
    total_results: int
    retrieval_type: str = "vector"


class VectorRetriever:
    """
    Vector similarity retriever over embedded editorial chunks.

    Loads a pre-built FAISS index and provides search with optional
    pattern and chunk-type filtering.
    """

    def __init__(self, index_path: Path | None = None, model_name: str = "all-MiniLM-L6-v2"):
        self.store = VectorStore(model_name=model_name)
        load_path = index_path or INDEX_DIR
        if (load_path / "index.faiss").exists():
            self.store.load(load_path)
        else:
            print(
                f"  ⚠ No saved index at {load_path}. "
                "Run: python -m retrieval.vector_store"
            )

    @property
    def is_ready(self) -> bool:
        return self.store.size > 0

    def search(
        self,
        query: str,
        k: int = 5,
        pattern_filter: Optional[str] = None,
        chunk_type: Optional[str] = None,
    ) -> VectorRetrievalResult:
        """
        Search for chunks similar to the query.

        Args:
            query: Natural language query string.
            k: Maximum results to return.
            pattern_filter: Only return chunks linked to this pattern.
            chunk_type: Filter by "prose" or "code".

        Returns:
            VectorRetrievalResult with results and linked pattern IDs.
        """
        results = self.store.search(
            query=query,
            k=k,
            pattern_filter=pattern_filter,
            chunk_type=chunk_type,
        )

        linked_patterns = self.store.get_pattern_ids_from_results(results)

        return VectorRetrievalResult(
            query=query,
            results=results,
            linked_pattern_ids=linked_patterns,
            total_results=len(results),
        )

    def search_for_pattern_entry(self, query: str, k: int = 3) -> list[str]:
        """
        Search to find the most relevant pattern names for a query.

        Used by the hybrid retriever to find graph entry points:
        vector search → pattern IDs → graph traversal from those nodes.
        """
        result = self.search(query, k=k)
        return result.linked_pattern_ids

    def search_code(self, query: str, k: int = 5, pattern: Optional[str] = None) -> list[SearchResult]:
        """Search specifically for code blocks."""
        result = self.search(query, k=k, pattern_filter=pattern, chunk_type="code")
        return result.results

    def search_prose(self, query: str, k: int = 5, pattern: Optional[str] = None) -> list[SearchResult]:
        """Search specifically for prose explanations."""
        result = self.search(query, k=k, pattern_filter=pattern, chunk_type="prose")
        return result.results


def demo():
    """Test the vector retriever."""
    retriever = VectorRetriever()
    if not retriever.is_ready:
        print("Index not built. Run: python -m retrieval.vector_store")
        return

    queries = [
        "Give me a C++ template for two-pointers on a sorted array",
        "How does sliding window work for substring problems?",
        "What's the time complexity of a sparse table?",
        "Fenwick tree point update range query",
    ]

    for q in queries:
        print(f"\nQ: \"{q}\"")
        result = retriever.search(q, k=3)
        print(f"  Found {result.total_results} results, patterns: {result.linked_pattern_ids}")
        for r in result.results:
            preview = r.text[:70].replace("\n", " ")
            print(f"    [{r.rank}] {r.score:.4f} ({r.metadata.chunk_type}) \"{preview}...\"")

        # Show code-only results
        code_results = retriever.search_code(q, k=1)
        if code_results:
            print(f"  Best code match: [{code_results[0].metadata.editorial_id}]")

    print("\n✓ Vector retriever demo complete.")


if __name__ == "__main__":
    demo()
