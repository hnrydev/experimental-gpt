"""
Download clear, *modern* encyclopedic text from Wikipedia for LM training.

Default: **Simple English Wikipedia** (shorter sentences, easier grammar — often a
better match for a small model than long literary prose). Optional: full English
Wikipedia for longer, denser articles (science, history, etc.).

Licensing: text is **CC BY-SA 4.0**; if you redistribute the training file or a
model substantially based on it, review attribution and share-alike:
https://creativecommons.org/licenses/by-sa/4.0/

Respect the Wikimedia **User-Agent policy**: identify your script; do not hammer
the API. One request per title, with a short delay between.

Usage:
  python scripts/fetch_wikipedia_corpus.py
  python scripts/fetch_wikipedia_corpus.py --mode en --max-chars 2000000
  python scripts/fetch_wikipedia_corpus.py --out data/wikipedia_corpus.txt
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# One title per request: on many wikis, TextExtracts + multi-title returns a full
# `extract` only for the first page; batching silently drops the rest. Sequential
# requests are slower but reliable. See https://www.mediawiki.org/wiki/Extension:TextExtracts
REQUEST_DELAY_S = 0.4
USER_AGENT = "baby-gpt/1.0 (local LM training; https://github.com/; contact N/A)"
API_SIMPLE = "https://simple.wikipedia.org/w/api.php"
API_EN = "https://en.wikipedia.org/w/api.php"

# Simple English: core school-level topics, nature, health basics — short clear prose.
# Titles are exact (disambiguation may still hit); missing pages are skipped in logs.
TITLES_SIMPLE: list[str] = [
    "Earth",
    "Water",
    "Sun",
    "Energy",
    "Air",
    "Fire",
    "Life",
    "Cell (biology)",
    "Human body",
    "Heart",
    "Brain",
    "Blood",
    "Child",
    "Family",
    "Food",
    "Health",
    "Disease",
    "Sleep",
    "Exercise",
    "Water cycle",
    "Weather",
    "Ocean",
    "Animal",
    "Mammal",
    "Plant",
    "Fruit",
    "Insect",
    "Bird",
    "Fish",
    "Science",
    "Chemistry",
    "Physics",
    "Mathematics",
    "Time",
    "Space",
    "Light",
    "Sound",
    "Art",
    "Music",
    "Book",
    "History",
    "Geography",
    "Country",
    "City",
    "Language",
    "English language",
    "School",
    "University",
    "Computer",
    "Internet",
    "World War II",
    "Climate",
]

# More Simple English: sports, travel, space, tools, government, time, more places — same register.
# Order merged + deduplicated in main; missing titles are skipped.
TITLES_SIMPLE_EXTRA: list[str] = [
    "Football",
    "Basketball",
    "Tennis",
    "Swimming",
    "Running",
    "Bicycle",
    "Car",
    "Bus",
    "Train",
    "Airplane",
    "Ship",
    "Moon",
    "Star",
    "Planet",
    "Mars",
    "Galaxy",
    "Solar System",
    "Jupiter",
    "Atom",
    "Gravity",
    "Speed",
    "Tool",
    "Machine",
    "Gold",
    "Rock",
    "Tree",
    "Forest",
    "Grass",
    "Desert",
    "Mountain",
    "River",
    "Lake",
    "Beach",
    "Volcano",
    "Earthquake",
    "Island",
    "Winter",
    "Spring",
    "Summer",
    "Autumn",
    "Day",
    "Week",
    "Month",
    "Year",
    "Doctor",
    "Nurse",
    "Teacher",
    "Police",
    "President",
    "Law",
    "Democracy",
    "Government",
    "War",
    "United Nations",
    "Europe",
    "Asia",
    "Africa",
    "North America",
    "South America",
    "United States",
    "United Kingdom",
    "Canada",
    "Australia",
    "China",
    "India",
    "Japan",
    "Money",
    "Religion",
    "Christianity",
    "Hinduism",
    "Islam",
    "Buddhism",
    "God",
    "Church",
    "Temple",
    "Mosque",
    "House",
    "Room",
    "Clothing",
    "Shoe",
    "Hat",
    "Glasses",
    "Bread",
    "Rice",
    "Milk",
    "Sugar",
    "Meat",
    "Salt",
    "Cooking",
    "Emotion",
    "Love",
    "Anger",
    "Fear",
    "Dream",
    "Memory",
    "Television",
    "Newspaper",
    "Telephone",
    "Photograph",
    "Video",
    "Color",
    "Size",
    "Number",
    "Line",
    "Map",
    "Engine",
    "Road",
]

# Even more Simple English: animals, school subjects, everyday words (merged + deduped).
TITLES_SIMPLE_MORE: list[str] = [
    "Elephant",
    "Lion",
    "Dog",
    "Cat",
    "Sheep",
    "Pig",
    "Cow",
    "Horse",
    "Chicken",
    "Snake",
    "Spider",
    "Whale",
    "Shark",
    "Dolphin",
    "Butterfly",
    "Honey bee",
    "Frog",
    "Turtle",
    "Agriculture",
    "Farmer",
    "Crop",
    "Farming",
    "Hunger",
    "Housing",
    "Elderly",
    "Village",
    "Tourist",
    "Hotel",
    "Park",
    "Museum",
    "Library",
    "Poetry",
    "Dance",
    "Singing",
    "Guitar",
    "Paint",
    "Sculpture",
    "Fiction",
    "Story",
    "Letter",
    "Alcohol",
    "Cigarette",
    "Vaccine",
    "Bacteria",
    "Malaria",
    "Flood",
    "Storm",
    "Drought",
    "Ice",
    "Heat",
    "Alps",
    "Himalaya",
    "Nile",
    "Amazon River",
    "Pacific Ocean",
    "Arctic",
    "Antarctica",
    "Greenland",
    "Egypt",
    "Brazil",
    "Mexico",
    "Russia",
    "Spain",
    "Italy",
    "South Korea",
    "Vietnam",
    "Thailand",
    "South Africa",
    "Nigeria",
    "Oxygen",
    "Chemistry",
    "Algebra",
    "Geometry",
    "Grammar",
    "Punctuation",
    "Pronoun",
    "Noun",
    "Verb",
    "Sentence",
    "Parliament",
    "Prison",
    "Marriage",
    "Divorce",
    "Wedding",
    "Birthday",
    "Holiday",
    "Christmas",
    "New Year",
    "Easter",
    "Sport",
    "Yoga",
    "Chess",
    "Olympic Games",
    "FIFA World Cup",
    "Film",
    "Poverty",
    "Tooth",
    "Bone",
    "Lung",
    "Stomach",
    "Skin",
]

# Deduplicate while building final list (any number of title lists).
def _merge_titles(*lists: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for lst in lists:
        for t in lst:
            t = t.strip()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t)
    return out


# English Wikipedia: richer, longer; still non-fiction, modern(ish) standard written English.
TITLES_EN: list[str] = [
    "Photosynthesis",
    "Evolution",
    "DNA",
    "Protein",
    "Climate change",
    "Ecosystem",
    "Photosphere",
    "Nervous system",
    "Cardiovascular system",
    "Vaccine",
    "Bacteria",
    "Virus",
    "Plate tectonics",
    "Atmosphere of Earth",
    "Ocean",
    "Artificial intelligence",
    "Machine learning",
    "Internet",
    "Database",
    "Algorithm",
    "Mathematics",
    "Statistics",
    "Philosophy",
    "Economics",
    "Democracy",
    "Constitution",
    "Human rights",
    "Agriculture",
    "Engineering",
    "Architecture",
    "Geography of Earth",
    "History of science",
]

# More English Wikipedia articles: medicine, technology, social science, more STEM.
TITLES_EN_EXTRA: list[str] = [
    "Neuroscience",
    "Immunology",
    "Ecology",
    "Geology",
    "Meteorology",
    "Astronomy",
    "Cell biology",
    "Organic chemistry",
    "Periodic table",
    "Quantum mechanics",
    "General relativity",
    "Thermodynamics",
    "Electromagnetism",
    "Nuclear fission",
    "Renewable energy",
    "Greenhouse effect",
    "Biodiversity",
    "Ecosystem service",
    "Epidemiology",
    "Public health",
    "Mental health",
    "Pandemic",
    "CRISPR",
    "Antibiotic resistance",
    "Natural language processing",
    "Deep learning",
    "Data structure",
    "Information security",
    "World Wide Web",
    "Transport Layer Security",
    "Cognitive science",
    "Linguistics",
    "Sociology",
    "Macroeconomics",
    "Game theory",
    "International law",
    "Supreme Court of the United States",
    "Parliament of the United Kingdom",
    "European Union",
    "Nuclear power",
    "Electricity grid",
    "Civil engineering",
    "Aerospace engineering",
    "Molecular biology",
    "History of the Internet",
    "Industrial Revolution",
    "Renaissance",
    "Age of Enlightenment",
    "Cold War",
    "Sustainable development",
    "Poverty",
    "Literacy",
    "Infant mortality",
    "Coral reef",
    "Hydroelectricity",
    "Lithium-ion battery",
]


def _api_request(api_base: str, titles: list[str]) -> dict:
    # exchars caps huge articles so one article does not dominate the file.
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "exintro": 0,
        "exlimit": 1,
        "exchars": 16000,
        "redirects": 1,
        "titles": "|".join(titles),
    }
    url = api_base + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _clean_extract(text: str) -> str:
    if not text:
        return ""
    # Occasional templates/refs leak into extract; light cleanup only.
    text = re.sub(r"\{\{[^}]*\}\}", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


def fetch_batches(api_base: str, titles: list[str], max_chars: int) -> list[tuple[str, str]]:
    """List of (normalized_title, extract) within max_chars total. One title per HTTP request."""
    out: list[tuple[str, str]] = []
    total = 0
    for raw_title in titles:
        if total >= max_chars:
            break
        try:
            data = _api_request(api_base, [raw_title])
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            print(f"  API error ({raw_title!r}): {e}")
            time.sleep(REQUEST_DELAY_S * 2)
            continue
        time.sleep(REQUEST_DELAY_S)
        q = data.get("query", {})
        pages = q.get("pages", [])
        for page in pages:
            if page.get("missing") or "invalid" in page:
                t = page.get("title", "?")
                print(f"  skip (missing/invalid): {t}")
                continue
            title = page.get("title", "Untitled")
            ex = _clean_extract(page.get("extract") or "")
            if not ex:
                print(f"  skip (empty extract): {title}")
                continue
            chunk_len = len(ex) + len(title) + 40
            if total + chunk_len > max_chars:
                ex = ex[: max(0, max_chars - total - len(title) - 40)]
            if not ex:
                return out
            out.append((title, ex))
            total += len(ex) + len(title) + 40
            if total >= max_chars:
                return out
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch Wikipedia (Simple or en) text for training.")
    p.add_argument(
        "--mode",
        choices=("simple", "en"),
        default="simple",
        help="simple = Simple English (clear, short sentences); en = English Wikipedia (longer, denser).",
    )
    p.add_argument("--out", type=Path, default=Path("data/wikipedia_corpus.txt"), help="Output path (UTF-8)")
    p.add_argument(
        "--max-chars",
        type=int,
        default=2_500_000,
        help="Stop after approximately this many characters of article text (plus headers).",
    )
    args = p.parse_args()

    api = API_SIMPLE if args.mode == "simple" else API_EN
    if args.mode == "simple":
        titles = _merge_titles(TITLES_SIMPLE, TITLES_SIMPLE_EXTRA, TITLES_SIMPLE_MORE)
    else:
        titles = _merge_titles(TITLES_EN, TITLES_EN_EXTRA)
    label = "Simple English Wikipedia" if args.mode == "simple" else "English Wikipedia"
    print(f"Fetching {label} ({len(titles)} titles, one request each, {REQUEST_DELAY_S}s between) ...")

    root = Path(__file__).resolve().parents[1]
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parts: list[str] = [
        "# Auto-generated. Not hand-edited. Remove this file to exclude from training.\n",
        "#\n",
        "# Text: Wikipedia extract(s) from the MediaWiki API. Licensed under\n",
        "# CC BY-SA 4.0. Wikimedia Foundation, Wikipedia contributors. See\n",
        "# https://creativecommons.org/licenses/by-sa/4.0/ for terms.\n",
        f"# Source: {label}.\n",
        "#\n",
        f"# Command: --mode {args.mode} --max-chars {args.max_chars}\n",
        "\n",
    ]
    for title, body in fetch_batches(api, titles, int(args.max_chars)):
        section = f"\n\n===== {title} ({label}) =====\n\n{body}\n"
        parts.append(section)
        print(f"  + {len(body):,} chars | {title}")

    text = "".join(parts)
    if not text.strip() or len(text) < 200:
        raise SystemExit("No usable text. Check network, titles, or API block.")

    out_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {len(text):,} characters to {out_path}")


if __name__ == "__main__":
    main()
