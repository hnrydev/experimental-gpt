"""
Download public-domain texts from Project Gutenberg and build data/input.txt.

Default list mixes **novels** (long narrative) with **essays, drama, and science**
so the char-level LM sees more registers than one kind of 19c prose — better
statistical variety, not a substitute for a chat-tuned or instruction-tuned model.

Only use texts you are allowed to use. Politeness: delay between requests;
set a descriptive User-Agent. See https://www.gutenberg.org/policy/robot_access.html

Usage:
  python scripts/fetch_corpus.py
  python scripts/fetch_corpus.py --max-chars 2000000
  python scripts/fetch_corpus.py --books novels   # first batch only
  python scripts/fetch_corpus.py --books diverse  # second batch (after novels)
  python scripts/fetch_corpus.py --books all      # default: novels then diverse
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

# (url, short label for logs) — `novels` = long English fiction; `diverse` = history,
# science, philosophy, essays, drama, vernacular (still English text from Gutenberg).

BOOKS_NOVELS: list[tuple[str, str]] = [
    ("https://www.gutenberg.org/files/1342/1342-0.txt", "Pride and Prejudice (Austen)"),
    ("https://www.gutenberg.org/files/11/11-0.txt", "Alice in Wonderland (Carroll)"),
    ("https://www.gutenberg.org/files/84/84-0.txt", "Frankenstein (Shelley)"),
    ("https://www.gutenberg.org/files/161/161-0.txt", "Sense and Sensibility (Austen)"),
    ("https://www.gutenberg.org/files/2701/2701-0.txt", "Moby-Dick (Melville)"),
    ("https://www.gutenberg.org/files/345/345-0.txt", "Dracula (Stoker)"),
    ("https://www.gutenberg.org/files/98/98-0.txt", "A Tale of Two Cities (Dickens)"),
    ("https://www.gutenberg.org/files/76/76-0.txt", "The Adventures of Tom Sawyer (Twain)"),
]

# Non-fiction, drama, mixed registers — more "topics" for the n-gram/char prior.
BOOKS_DIVERSE: list[tuple[str, str]] = [
    ("https://www.gutenberg.org/files/1228/1228-0.txt", "On the Origin of Species (Darwin)"),
    ("https://www.gutenberg.org/files/2680/2680-0.txt", "Meditations (Aurelius, tr. Long)"),
    ("https://www.gutenberg.org/files/205/205-0.txt", "Walden (Thoreau)"),
    (
        "https://www.gutenberg.org/files/16643/16643-0.txt",
        "Essays, First Series (Emerson)",
    ),
    (
        "https://www.gutenberg.org/files/18/18-0.txt",
        "The Federalist Papers (Hamilton, Madison, Jay)",
    ),
    ("https://www.gutenberg.org/files/1524/1524-0.txt", "Hamlet (Shakespeare)"),
    (
        "https://www.gutenberg.org/files/132/132-0.txt",
        "The Art of War (Sun Tzu, Giles tr.)",
    ),
    (
        "https://www.gutenberg.org/files/730/730-0.txt",
        "The Autocrat of the Breakfast-Table (O.W. Holmes)",
    ),
    (
        "https://www.gutenberg.org/files/1952/1952-0.txt",
        "The Yellow Wallpaper (Gilman)",
    ),
]


def _book_list(which: str) -> list[tuple[str, str]]:
    w = which.strip().lower()
    if w in ("all", "mixed"):
        return list(BOOKS_NOVELS) + list(BOOKS_DIVERSE)
    if w in ("novels", "fiction"):
        return list(BOOKS_NOVELS)
    if w in ("diverse", "nonfiction", "essays", "non-fiction"):
        return list(BOOKS_DIVERSE)
    raise SystemExit(
        f"Unknown --books {which!r}; use all, mixed, novels, or diverse (synonyms: fiction, nonfiction, essays)"
    )

USER_AGENT = "baby-gpt/1.0 (local educational use; contact: not-a-bot@localhost)"
REQUEST_DELAY_S = 2.0

def extract_gutenberg_core(raw: str) -> str:
    """Keep only the book body between Gutenberg START/END marker lines."""
    lines = raw.splitlines(keepends=True)
    start_i = end_i = None
    for i, line in enumerate(lines):
        u = line.upper()
        if "***" in line and "START OF" in u and "PROJECT GUTENBERG" in u:
            start_i = i + 1
            break
    for i, line in enumerate(lines):
        if start_i is None:
            break
        if i < start_i:
            continue
        u = line.upper()
        if "***" in line and "END OF" in u and "PROJECT GUTENBERG" in u:
            end_i = i
            break
    if start_i is not None and end_i is not None and end_i > start_i:
        return "".join(lines[start_i:end_i])
    if start_i is not None:
        return "".join(lines[start_i:])
    return raw


def fetch_url(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    return data.decode("utf-8", errors="replace")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/input.txt"), help="Output file (UTF-8)")
    p.add_argument(
        "--max-chars",
        type=int,
        default=3_500_000,
        help="Stop after this many total characters (keeps runs smaller / faster to train on CPU).",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write input.txt.bak if input.txt already exists",
    )
    p.add_argument(
        "--books",
        type=str,
        default="all",
        help="Which URL list: all (novels then diverse, default), novels, or diverse. "
        "Synonyms: mixed=all, fiction=novels, nonfiction|essays=diverse",
    )
    args = p.parse_args()
    books = _book_list(args.books)

    root = Path(__file__).resolve().parents[1]
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.is_file() and not args.no_backup:
        bak = out.with_suffix(out.suffix + ".bak")
        bak.write_bytes(out.read_bytes())
        print(f"Backed up previous corpus to {bak}")

    parts: list[str] = []
    total = 0
    for url, label in books:
        if total >= args.max_chars:
            break
        print(f"Fetching {label} ...")
        try:
            raw = fetch_url(url)
        except Exception as e:
            print(f"  skip ({e})")
            time.sleep(REQUEST_DELAY_S)
            continue
        body = extract_gutenberg_core(raw)
        header = f"\n\n===== {label} =====\n\n"
        chunk = header + body
        if total + len(chunk) > args.max_chars:
            chunk = chunk[: max(0, args.max_chars - total)]
        parts.append(chunk)
        total += len(chunk)
        print(f"  +{len(chunk):,} chars (total {total:,})")
        time.sleep(REQUEST_DELAY_S)

    text = "".join(parts)
    if not text.strip():
        raise SystemExit("No text downloaded. Check your network or Gutenberg availability.")
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {len(text):,} characters to {out}")


if __name__ == "__main__":
    main()
