"""
load_graph.py — Phase 1 manual seed of the GraphRAG knowledge graph.

Creates the schema (constraints + indexes), inserts ~8 algorithmic pattern nodes
with realistic properties, plus representative Problem, CodeTemplate, and Source
nodes, then wires them together with PREREQUISITE_OF, ALTERNATIVE_TO,
SAME_COMPLEXITY_CLASS, SOLVED_WITH, HAS_TEMPLATE, and EXTRACTED_FROM edges.

Idempotent: uses MERGE so re-running is safe.
"""

from __future__ import annotations

import os
import textwrap

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphrag_password")

# ---------------------------------------------------------------------------
# Schema: constraints & indexes
# ---------------------------------------------------------------------------

SCHEMA_STATEMENTS = [
    # Uniqueness constraints (also create indexes implicitly)
    "CREATE CONSTRAINT pattern_name IF NOT EXISTS FOR (p:Pattern) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT problem_id IF NOT EXISTS FOR (p:Problem) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT source_url IF NOT EXISTS FOR (s:Source) REQUIRE s.url IS UNIQUE",
    # Full-text index for semantic-ish search over pattern descriptions (Phase 4 will use vector search)
    """
    CREATE FULLTEXT INDEX pattern_description IF NOT EXISTS
    FOR (p:Pattern)
    ON EACH [p.description, p.name]
    """,
]

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

PATTERNS = [
    {
        "name": "Two Pointers",
        "description": (
            "Technique that uses two indices scanning a sorted (or linear) "
            "structure from both ends or in the same direction to find pairs, "
            "sub-arrays, or partitions satisfying a condition in O(n) time."
        ),
        "time_complexity": "O(n)",
        "space_complexity": "O(1)",
        "category": "array",
    },
    {
        "name": "Sliding Window",
        "description": (
            "Maintains a variable- or fixed-size window over a sequence, "
            "expanding/contracting one end at a time to compute aggregates "
            "(max, sum, count of distinct) in amortized O(n)."
        ),
        "time_complexity": "O(n)",
        "space_complexity": "O(k)  # k = window size or alphabet",
        "category": "array",
    },
    {
        "name": "Binary Search",
        "description": (
            "Divide-and-conquer search on a monotonic predicate or sorted "
            "sequence. Reduces the search space by half each iteration. "
            "Also used for 'binary search on the answer' (parametric search)."
        ),
        "time_complexity": "O(log n)",
        "space_complexity": "O(1)",
        "category": "search",
    },
    {
        "name": "Prefix Sum",
        "description": (
            "Pre-computes cumulative sums so that any sub-array sum can be "
            "answered in O(1) after O(n) preprocessing. Generalises to 2-D "
            "prefix sums and prefix XOR."
        ),
        "time_complexity": "O(n) build, O(1) query",
        "space_complexity": "O(n)",
        "category": "array",
    },
    {
        "name": "Fenwick Tree",
        "description": (
            "Binary Indexed Tree (BIT). Supports point updates and prefix "
            "queries (sum, XOR, min with caveats) in O(log n) each, using "
            "O(n) space. Simpler constant factor than a segment tree for "
            "commutative, invertible operations."
        ),
        "time_complexity": "O(log n) per query/update",
        "space_complexity": "O(n)",
        "category": "tree",
    },
    {
        "name": "Segment Tree",
        "description": (
            "Balanced binary tree over an array supporting arbitrary range "
            "queries (min, max, sum, GCD) and point or range updates in "
            "O(log n). Supports lazy propagation for range updates."
        ),
        "time_complexity": "O(log n) per query/update",
        "space_complexity": "O(n)",
        "category": "tree",
    },
    {
        "name": "Sparse Table",
        "description": (
            "Static data structure for idempotent range queries (min, max, "
            "GCD) in O(1) per query after O(n log n) preprocessing. Does "
            "NOT support updates."
        ),
        "time_complexity": "O(n log n) build, O(1) query",
        "space_complexity": "O(n log n)",
        "category": "tree",
    },
    {
        "name": "Suffix Array",
        "description": (
            "Sorted array of all suffixes of a string, enabling O(m log n) "
            "substring search. Often paired with an LCP array for O(1) LCP "
            "queries. Can replace suffix trees with lower memory."
        ),
        "time_complexity": "O(n log n) or O(n) build",
        "space_complexity": "O(n)",
        "category": "string",
    },
]

# Domain-accurate prerequisite edges: (from, to) means `from` is a prerequisite of `to`.
PREREQUISITE_EDGES = [
    ("Prefix Sum", "Fenwick Tree"),
    ("Prefix Sum", "Sparse Table"),
    ("Binary Search", "Suffix Array"),
    ("Binary Search", "Segment Tree"),
    ("Two Pointers", "Sliding Window"),
    ("Fenwick Tree", "Segment Tree"),  # conceptually, BIT is simpler; understanding it first helps
]

# Alternative-to edges (symmetric): both solve overlapping problem classes with different trade-offs.
ALTERNATIVE_EDGES = [
    (
        "Segment Tree",
        "Fenwick Tree",
        "Fenwick Tree has a smaller constant and simpler code but only supports "
        "invertible operations; Segment Tree is more general (lazy propagation, "
        "arbitrary merge functions).",
    ),
    (
        "Segment Tree",
        "Sparse Table",
        "Sparse Table gives O(1) queries for idempotent operations on static data; "
        "Segment Tree is O(log n) but supports updates.",
    ),
    (
        "Two Pointers",
        "Sliding Window",
        "Two Pointers typically scans from both ends inward; Sliding Window "
        "maintains a contiguous sub-array/window moving in one direction. "
        "Overlapping technique set, different framing.",
    ),
]

# Same-complexity-class edges (both O(log n) per op)
SAME_COMPLEXITY_EDGES = [
    ("Fenwick Tree", "Segment Tree"),
]

# ---------------------------------------------------------------------------
# Problems (a handful to make the graph non-trivial)
# ---------------------------------------------------------------------------

PROBLEMS = [
    {
        "id": "CF-1920C",
        "title": "Partitioning into Two Subsequences",
        "source": "codeforces",
        "url": "https://codeforces.com/problemset/problem/1920/C",
        "difficulty": "1600",
        "patterns": ["Two Pointers", "Prefix Sum"],
    },
    {
        "id": "LC-239",
        "title": "Sliding Window Maximum",
        "source": "leetcode",
        "url": "https://leetcode.com/problems/sliding-window-maximum/",
        "difficulty": "Hard",
        "patterns": ["Sliding Window", "Segment Tree"],
    },
    {
        "id": "CF-1354D",
        "title": "Multiset",
        "source": "codeforces",
        "url": "https://codeforces.com/problemset/problem/1354/D",
        "difficulty": "1900",
        "patterns": ["Binary Search", "Fenwick Tree"],
    },
    {
        "id": "LC-307",
        "title": "Range Sum Query - Mutable",
        "source": "leetcode",
        "url": "https://leetcode.com/problems/range-sum-query-mutable/",
        "difficulty": "Medium",
        "patterns": ["Segment Tree", "Fenwick Tree"],
    },
    {
        "id": "CF-1923E",
        "title": "Count Paths",
        "source": "codeforces",
        "url": "https://codeforces.com/problemset/problem/1923/E",
        "difficulty": "2100",
        "patterns": ["Suffix Array"],
    },
]

# ---------------------------------------------------------------------------
# Code templates (realistic C++ snippets)
# ---------------------------------------------------------------------------

CODE_TEMPLATES = [
    {
        "pattern": "Two Pointers",
        "code": textwrap.dedent("""\
            // Two-pointer on sorted array: find pair summing to target
            int lo = 0, hi = (int)a.size() - 1;
            while (lo < hi) {
                long long s = a[lo] + a[hi];
                if (s == target) { /* found */ ++lo; --hi; }
                else if (s < target) ++lo;
                else --hi;
            }
        """),
        "explanation": (
            "Classic two-pointer on a sorted array. `lo` starts at the "
            "beginning, `hi` at the end. If the current sum is too small we "
            "advance `lo`; if too large we retreat `hi`. O(n) time, O(1) space."
        ),
    },
    {
        "pattern": "Segment Tree",
        "code": textwrap.dedent("""\
            struct SegTree {
                int n;
                vector<long long> t;
                SegTree(int n) : n(n), t(4 * n, 0) {}

                void build(vector<int>& a, int v, int tl, int tr) {
                    if (tl == tr) { t[v] = a[tl]; return; }
                    int tm = (tl + tr) / 2;
                    build(a, 2*v, tl, tm);
                    build(a, 2*v+1, tm+1, tr);
                    t[v] = t[2*v] + t[2*v+1];
                }

                void update(int v, int tl, int tr, int pos, int val) {
                    if (tl == tr) { t[v] = val; return; }
                    int tm = (tl + tr) / 2;
                    if (pos <= tm) update(2*v, tl, tm, pos, val);
                    else update(2*v+1, tm+1, tr, pos, val);
                    t[v] = t[2*v] + t[2*v+1];
                }

                long long query(int v, int tl, int tr, int l, int r) {
                    if (l > r) return 0;
                    if (l == tl && r == tr) return t[v];
                    int tm = (tl + tr) / 2;
                    return query(2*v, tl, tm, l, min(r, tm))
                         + query(2*v+1, tm+1, tr, max(l, tm+1), r);
                }
            };
        """),
        "explanation": (
            "Recursive segment tree supporting point update and range sum "
            "query, both in O(log n). Array is 1-indexed internally. "
            "Build is O(n). Extend with lazy propagation for range updates."
        ),
    },
    {
        "pattern": "Fenwick Tree",
        "code": textwrap.dedent("""\
            struct BIT {
                int n;
                vector<long long> tree;
                BIT(int n) : n(n), tree(n + 1, 0) {}

                void update(int i, long long delta) {
                    for (; i <= n; i += i & (-i))
                        tree[i] += delta;
                }

                long long query(int i) {
                    long long s = 0;
                    for (; i > 0; i -= i & (-i))
                        s += tree[i];
                    return s;
                }

                long long query(int l, int r) {
                    return query(r) - query(l - 1);
                }
            };
        """),
        "explanation": (
            "Binary Indexed Tree (Fenwick Tree) for prefix sums. "
            "1-indexed. Point update and prefix query in O(log n). "
            "Range query via prefix difference. Simpler and faster "
            "constant than segment tree for sum queries."
        ),
    },
    {
        "pattern": "Binary Search",
        "code": textwrap.dedent("""\
            // Binary search on a monotonic predicate f(x)
            // Finds the smallest x in [lo, hi] such that f(x) is true
            auto lo = 0LL, hi = 1e18;
            while (hi - lo > 1) {
                auto mid = lo + (hi - lo) / 2;
                if (predicate(mid)) hi = mid;
                else lo = mid;
            }
            // answer is hi (first true)
        """),
        "explanation": (
            "Parametric / binary-search-on-answer template. Works for "
            "any monotonic predicate. The loop invariant is: f(lo)=false, "
            "f(hi)=true. After the loop, hi is the smallest value where "
            "the predicate holds."
        ),
    },
    {
        "pattern": "Sparse Table",
        "code": textwrap.dedent("""\
            struct SparseTable {
                vector<vector<int>> table;
                vector<int> lg;
                int n;

                void build(vector<int>& a) {
                    n = a.size();
                    int K = __lg(n) + 1;
                    table.assign(K, vector<int>(n));
                    lg.assign(n + 1, 0);
                    for (int i = 2; i <= n; i++) lg[i] = lg[i/2] + 1;
                    table[0] = a;
                    for (int k = 1; k < K; k++)
                        for (int i = 0; i + (1 << k) <= n; i++)
                            table[k][i] = min(table[k-1][i],
                                              table[k-1][i + (1 << (k-1))]);
                }

                int query(int l, int r) {  // [l, r] inclusive, 0-indexed
                    int k = lg[r - l + 1];
                    return min(table[k][l], table[k][r - (1 << k) + 1]);
                }
            };
        """),
        "explanation": (
            "Sparse table for static range minimum queries (RMQ). "
            "Build in O(n log n), query in O(1) using overlapping blocks. "
            "Works for any idempotent function (min, max, GCD). "
            "Does NOT support updates."
        ),
    },
    {
        "pattern": "Suffix Array",
        "code": textwrap.dedent("""\
            // O(n log^2 n) suffix array construction
            vector<int> suffix_array(const string& s) {
                int n = s.size();
                vector<int> sa(n), rank_(n), tmp(n);
                iota(sa.begin(), sa.end(), 0);
                for (int i = 0; i < n; i++) rank_[i] = s[i];

                for (int gap = 1; gap < n; gap <<= 1) {
                    auto cmp = [&](int a, int b) {
                        if (rank_[a] != rank_[b]) return rank_[a] < rank_[b];
                        int ra = a + gap < n ? rank_[a + gap] : -1;
                        int rb = b + gap < n ? rank_[b + gap] : -1;
                        return ra < rb;
                    };
                    sort(sa.begin(), sa.end(), cmp);
                    tmp[sa[0]] = 0;
                    for (int i = 1; i < n; i++)
                        tmp[sa[i]] = tmp[sa[i-1]] + cmp(sa[i-1], sa[i]);
                    rank_ = tmp;
                }
                return sa;
            }
        """),
        "explanation": (
            "Suffix array built via successive doubling of gap width. "
            "O(n log^2 n) with std::sort; can be optimised to O(n log n) "
            "with radix sort. Pair with LCP array (Kasai's algorithm) for "
            "efficient substring queries."
        ),
    },
    {
        "pattern": "Prefix Sum",
        "code": textwrap.dedent("""\
            // 1-D prefix sum
            vector<long long> prefix(n + 1, 0);
            for (int i = 0; i < n; i++)
                prefix[i + 1] = prefix[i] + a[i];
            // sum of a[l..r] (0-indexed, inclusive) = prefix[r+1] - prefix[l]
        """),
        "explanation": (
            "Standard prefix sum. O(n) build, O(1) range sum query. "
            "Generalises to 2-D (row prefix then column prefix) and to "
            "prefix XOR for XOR range queries."
        ),
    },
    {
        "pattern": "Sliding Window",
        "code": textwrap.dedent("""\
            // Maximum sum sub-array of size k
            long long window_sum = 0, best = LLONG_MIN;
            for (int i = 0; i < n; i++) {
                window_sum += a[i];
                if (i >= k) window_sum -= a[i - k];
                if (i >= k - 1) best = max(best, window_sum);
            }
        """),
        "explanation": (
            "Fixed-size sliding window computing the maximum sub-array sum "
            "of length k. Extend to variable-size by tracking a condition "
            "(e.g. at most K distinct elements) and shrinking the left "
            "pointer when the condition is violated."
        ),
    },
]

# ---------------------------------------------------------------------------
# Sources (editorial references)
# ---------------------------------------------------------------------------

SOURCES = [
    {
        "url": "https://codeforces.com/blog/entry/44351",
        "author": "PrinceOfPersia",
        "date_published": "2016-04-17",
        "platform": "codeforces",
    },
    {
        "url": "https://cp-algorithms.com/data_structures/segment_tree.html",
        "author": "cp-algorithms contributors",
        "date_published": "2023-08-15",
        "platform": "cp-algorithms",
    },
    {
        "url": "https://codeforces.com/blog/entry/18051",
        "author": "Al.Cash",
        "date_published": "2015-02-10",
        "platform": "codeforces",
    },
    {
        "url": "https://cp-algorithms.com/data_structures/sparse-table.html",
        "author": "cp-algorithms contributors",
        "date_published": "2023-06-20",
        "platform": "cp-algorithms",
    },
]

# Map templates to sources for EXTRACTED_FROM edges
TEMPLATE_SOURCE_MAP = {
    "Two Pointers": "https://codeforces.com/blog/entry/44351",
    "Binary Search": "https://codeforces.com/blog/entry/44351",
    "Sliding Window": "https://codeforces.com/blog/entry/44351",
    "Prefix Sum": "https://codeforces.com/blog/entry/44351",
    "Segment Tree": "https://cp-algorithms.com/data_structures/segment_tree.html",
    "Fenwick Tree": "https://codeforces.com/blog/entry/18051",
    "Sparse Table": "https://cp-algorithms.com/data_structures/sparse-table.html",
    "Suffix Array": "https://cp-algorithms.com/data_structures/sparse-table.html",  # placeholder
}


# ---------------------------------------------------------------------------
# Neo4j loader
# ---------------------------------------------------------------------------


def _run_schema(tx):
    """Create constraints and indexes (idempotent)."""
    for stmt in SCHEMA_STATEMENTS:
        tx.run(stmt.strip())


def _load_patterns(tx):
    """MERGE all Pattern nodes."""
    for p in PATTERNS:
        tx.run(
            """
            MERGE (pat:Pattern {name: $name})
            SET pat.description      = $description,
                pat.time_complexity  = $time_complexity,
                pat.space_complexity = $space_complexity,
                pat.category         = $category
            """,
            **p,
        )


def _load_prerequisite_edges(tx):
    for src, dst in PREREQUISITE_EDGES:
        tx.run(
            """
            MATCH (a:Pattern {name: $src}), (b:Pattern {name: $dst})
            MERGE (a)-[:PREREQUISITE_OF]->(b)
            """,
            src=src,
            dst=dst,
        )


def _load_alternative_edges(tx):
    for src, dst, tradeoff in ALTERNATIVE_EDGES:
        tx.run(
            """
            MATCH (a:Pattern {name: $src}), (b:Pattern {name: $dst})
            MERGE (a)-[:ALTERNATIVE_TO {tradeoff: $tradeoff}]->(b)
            """,
            src=src,
            dst=dst,
            tradeoff=tradeoff,
        )


def _load_same_complexity_edges(tx):
    for src, dst in SAME_COMPLEXITY_EDGES:
        tx.run(
            """
            MATCH (a:Pattern {name: $src}), (b:Pattern {name: $dst})
            MERGE (a)-[:SAME_COMPLEXITY_CLASS]->(b)
            """,
            src=src,
            dst=dst,
        )


def _load_problems(tx):
    for prob in PROBLEMS:
        tx.run(
            """
            MERGE (p:Problem {id: $id})
            SET p.title      = $title,
                p.source     = $source,
                p.url        = $url,
                p.difficulty = $difficulty
            """,
            id=prob["id"],
            title=prob["title"],
            source=prob["source"],
            url=prob["url"],
            difficulty=prob["difficulty"],
        )
        for pat_name in prob["patterns"]:
            tx.run(
                """
                MATCH (prob:Problem {id: $prob_id}), (pat:Pattern {name: $pat_name})
                MERGE (prob)-[:SOLVED_WITH]->(pat)
                """,
                prob_id=prob["id"],
                pat_name=pat_name,
            )


def _load_sources(tx):
    for src in SOURCES:
        tx.run(
            """
            MERGE (s:Source {url: $url})
            SET s.author         = $author,
                s.date_published = $date_published,
                s.platform       = $platform
            """,
            **src,
        )


def _load_code_templates(tx):
    for tmpl in CODE_TEMPLATES:
        # Use a composite key of pattern name (since we have one template per pattern in Phase 1)
        tx.run(
            """
            MERGE (ct:CodeTemplate {pattern_name: $pattern})
            SET ct.code        = $code,
                ct.language    = 'cpp',
                ct.annotated_explanation = $explanation
            """,
            pattern=tmpl["pattern"],
            code=tmpl["code"],
            explanation=tmpl["explanation"],
        )
        # HAS_TEMPLATE edge
        tx.run(
            """
            MATCH (pat:Pattern {name: $pattern}), (ct:CodeTemplate {pattern_name: $pattern})
            MERGE (pat)-[:HAS_TEMPLATE]->(ct)
            """,
            pattern=tmpl["pattern"],
        )
        # EXTRACTED_FROM edge
        source_url = TEMPLATE_SOURCE_MAP.get(tmpl["pattern"])
        if source_url:
            tx.run(
                """
                MATCH (ct:CodeTemplate {pattern_name: $pattern}), (s:Source {url: $url})
                MERGE (ct)-[:EXTRACTED_FROM]->(s)
                """,
                pattern=tmpl["pattern"],
                url=source_url,
            )


def load_all(uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
    """Connect to Neo4j and load the full Phase 1 graph."""
    driver = GraphDatabase.driver(uri, auth=(user, password))

    # Verify connectivity
    driver.verify_connectivity()
    print(f"✓ Connected to Neo4j at {uri}")

    with driver.session() as session:
        # Schema
        session.execute_write(_run_schema)
        print("✓ Schema constraints and indexes created")

        # Nodes
        session.execute_write(_load_patterns)
        print(f"✓ Loaded {len(PATTERNS)} Pattern nodes")

        session.execute_write(_load_problems)
        print(f"✓ Loaded {len(PROBLEMS)} Problem nodes")

        session.execute_write(_load_sources)
        print(f"✓ Loaded {len(SOURCES)} Source nodes")

        session.execute_write(_load_code_templates)
        print(f"✓ Loaded {len(CODE_TEMPLATES)} CodeTemplate nodes")

        # Edges
        session.execute_write(_load_prerequisite_edges)
        print(f"✓ Created {len(PREREQUISITE_EDGES)} PREREQUISITE_OF edges")

        session.execute_write(_load_alternative_edges)
        print(f"✓ Created {len(ALTERNATIVE_EDGES)} ALTERNATIVE_TO edges")

        session.execute_write(_load_same_complexity_edges)
        print(f"✓ Created {len(SAME_COMPLEXITY_EDGES)} SAME_COMPLEXITY_CLASS edges")

        # Summary
        result = session.run(
            """
            MATCH (n)
            RETURN labels(n)[0] AS label, count(n) AS cnt
            ORDER BY cnt DESC
            """
        )
        print("\n--- Graph summary ---")
        for record in result:
            print(f"  {record['label']}: {record['cnt']} nodes")

        result = session.run(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS cnt
            ORDER BY cnt DESC
            """
        )
        for record in result:
            print(f"  {record['rel_type']}: {record['cnt']} edges")

    driver.close()
    print("\n✓ Done. Graph loaded successfully.")


if __name__ == "__main__":
    load_all()
