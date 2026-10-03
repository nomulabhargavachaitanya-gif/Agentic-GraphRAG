"""
scrape.py — Codeforces editorial scraper.

Scrapes editorial blog posts from Codeforces and produces structured JSON
matching the format in data/editorials.json. Respects rate limits and ToS
by using conservative delays between requests.

Usage:
    # Scrape a single editorial by blog entry ID
    python -m ingestion.scrape --entry 92064

    # Scrape a list of entries
    python -m ingestion.scrape --entries 92064 85547 113245

    # Load from the pre-curated local corpus (default, no network)
    python -m ingestion.scrape --local
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment, misc]


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EDITORIALS_PATH = DATA_DIR / "editorials.json"

# Codeforces rate limiting: 1 request per 2 seconds minimum
CF_BASE_URL = "https://codeforces.com"
CF_DELAY_SECONDS = 2.0


@dataclass
class Editorial:
    """A single parsed editorial entry."""
    id: str
    title: str
    source: str
    url: str
    difficulty: str
    author: str
    date_published: str
    editorial: str


def load_local_corpus() -> list[Editorial]:
    """Load the pre-curated editorial corpus from data/editorials.json."""
    if not EDITORIALS_PATH.exists():
        raise FileNotFoundError(
            f"Local corpus not found at {EDITORIALS_PATH}. "
            "Run scraping first or ensure data/editorials.json exists."
        )
    with open(EDITORIALS_PATH) as f:
        raw = json.load(f)
    return [Editorial(**entry) for entry in raw]


def scrape_cf_blog_entry(entry_id: int) -> Optional[Editorial]:
    """
    Scrape a single Codeforces blog entry and extract editorial content.

    Returns an Editorial if successful, None if the page couldn't be parsed.
    """
    if httpx is None:
        raise ImportError("httpx is required for scraping. Install: pip install httpx")
    if BeautifulSoup is None:
        raise ImportError("beautifulsoup4 is required. Install: pip install beautifulsoup4")

    url = f"{CF_BASE_URL}/blog/entry/{entry_id}"
    print(f"  Fetching {url}...")

    try:
        response = httpx.get(url, follow_redirects=True, timeout=15.0)
        response.raise_for_status()
    except httpx.HTTPError as e:
        print(f"  ✗ HTTP error for entry {entry_id}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title
    title_el = soup.select_one(".title.aCodeforcesContentRecovery, .blogTitle")
    title = title_el.get_text(strip=True) if title_el else f"Blog Entry {entry_id}"

    # Extract author
    author_el = soup.select_one(".info .rated-user, .info a")
    author = author_el.get_text(strip=True) if author_el else "unknown"

    # Extract the blog content div
    content_el = soup.select_one(".ttypography")
    if content_el is None:
        print(f"  ✗ Could not find content for entry {entry_id}")
        return None

    # Convert HTML to cleaned text with code blocks preserved
    editorial_text = _html_to_markdown(content_el)

    return Editorial(
        id=f"CF-{entry_id}",
        title=title,
        source="codeforces",
        url=url,
        difficulty="unknown",
        author=author,
        date_published="unknown",
        editorial=editorial_text,
    )


def _html_to_markdown(element) -> str:
    """
    Convert a BeautifulSoup element to a rough markdown representation,
    preserving code blocks as fenced ```cpp blocks.
    """
    parts: list[str] = []

    for child in element.children:
        if hasattr(child, "name") and child.name is not None:
            if child.name == "pre":
                code = child.get_text()
                parts.append(f"\n```cpp\n{code.strip()}\n```\n")
            elif child.name in ("h1", "h2", "h3"):
                text = child.get_text(strip=True)
                level = int(child.name[1])
                parts.append(f"\n{'#' * level} {text}\n")
            elif child.name == "ul":
                for li in child.find_all("li", recursive=False):
                    parts.append(f"- {li.get_text(strip=True)}")
            elif child.name == "ol":
                for i, li in enumerate(child.find_all("li", recursive=False), 1):
                    parts.append(f"{i}. {li.get_text(strip=True)}")
            else:
                text = child.get_text(separator=" ", strip=True)
                if text:
                    parts.append(text)
        elif hasattr(child, "strip"):
            text = child.strip()
            if text:
                parts.append(text)

    return "\n\n".join(parts)


def scrape_multiple(entry_ids: list[int]) -> list[Editorial]:
    """Scrape multiple Codeforces blog entries with rate limiting."""
    results: list[Editorial] = []
    for i, entry_id in enumerate(entry_ids):
        if i > 0:
            print(f"  (waiting {CF_DELAY_SECONDS}s for rate limit...)")
            time.sleep(CF_DELAY_SECONDS)
        editorial = scrape_cf_blog_entry(entry_id)
        if editorial:
            results.append(editorial)
    return results


def save_editorials(editorials: list[Editorial], path: Path | None = None):
    """Save editorials to JSON."""
    path = path or EDITORIALS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(e) for e in editorials], f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(editorials)} editorials to {path}")


def main():
    parser = argparse.ArgumentParser(description="Scrape Codeforces editorials")
    parser.add_argument("--local", action="store_true", help="Load from local corpus (no network)")
    parser.add_argument("--entry", type=int, help="Single blog entry ID to scrape")
    parser.add_argument("--entries", type=int, nargs="+", help="Multiple blog entry IDs")
    parser.add_argument("--output", type=str, help="Output JSON path")
    args = parser.parse_args()

    if args.local:
        editorials = load_local_corpus()
        print(f"✓ Loaded {len(editorials)} editorials from local corpus")
        for e in editorials:
            code_blocks = len(re.findall(r"```cpp", e.editorial))
            print(f"  [{e.id}] {e.title} ({e.source}, {code_blocks} code blocks)")
        return

    ids_to_scrape: list[int] = []
    if args.entry:
        ids_to_scrape = [args.entry]
    elif args.entries:
        ids_to_scrape = args.entries
    else:
        parser.print_help()
        sys.exit(1)

    editorials = scrape_multiple(ids_to_scrape)
    if editorials:
        out_path = Path(args.output) if args.output else EDITORIALS_PATH
        save_editorials(editorials, out_path)
    else:
        print("✗ No editorials scraped.")


if __name__ == "__main__":
    main()
