"""
Local, rate-limit-free post-processing for char-LM output.

A tiny LM can repeat one letter hundreds of times; that is not "grammar" but
looks broken. This module applies simple **deterministic** rules:

- Cap runs of the **same** letter/digit to a few characters (stops "iiii…" spam).
- Tame long runs of `?` `!` `.` / `-`.
- If the **end** of the string is the same multichar span repeated back-to-back
  (a common greedy / undertrained loop), keep one copy — does not fix *wrong* text,
  only tail repetition.

**Not grammar repair:** a real “grammatical rules” layer would need a parse of
valid sentences. Char-salad is not parseable, so rules cannot *fix* meaning.
Optional `light_surface_english` only does **cosmetic** tweaks (spaces, `I`, caps
after `. `) when the string is already mostly text-like — it is not a fix for nonsense.
"""

from __future__ import annotations

import re


def collapse_excessive_repeats(s: str, max_same: int = 3) -> str:
    if not s or max_same < 1:
        return s
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        j = i + 1
        while j < n and s[j] == ch:
            j += 1
        run = j - i
        if ch.isalnum() and run > max_same:
            out.append(ch * max_same)
        else:
            out.append(s[i:j])
        i = j
    return "".join(out)


def collapse_repeating_substring_runs(
    s: str, min_unit: int = 2, max_unit: int = 6, max_passes: int = 32
) -> str:
    """
    Collapse (UNIT)(UNIT)(UNIT)... to a single UNIT for UNIT length in [min_unit, max_unit].
    Catches alternating spam like 'CyCyCy...' and 'from from from' without a single long char run.
    """
    if not s or len(s) < 3 * min_unit:
        return s
    t = s
    for _ in range(max_passes):
        m = re.search(rf"(.{{{min_unit},{max_unit}}})(\1){{2,}}", t)
        if not m:
            break
        u = m.group(1)
        t = t[: m.start()] + u + t[m.end() :]
    return t


def trim_consecutive_duplicate_tail(s: str, min_unit: int = 10) -> str:
    """
    Remove a trailing half when it equals the half before it, repeatedly.

    Example: '...A B C A B C' with len(A B C) >= `min_unit` -> '...A B C'
    (greedy/weak LMs often loop a short phrase; this is a *display* mitigation).
    """
    if not s or len(s) < 2 * min_unit:
        return s
    out = s
    # cap inner scan so pathological 100k strings stay cheap
    for _ in range(256):
        n = len(out)
        if n < 2 * min_unit:
            break
        changed = False
        high = min(n // 2, 800)
        for L in range(high, min_unit - 1, -1):
            if n < 2 * L:
                continue
            a, b = out[n - 2 * L : n - L], out[n - L : n]
            if a == b:
                out = out[: n - L]
                changed = True
                break
        if not changed:
            break
    return out


def light_regex_grammar_fixes(s: str) -> str:
    """
    High-precision string fixes (not a parser). Fixes a few common LM glitches:
    wrong “a/an” before a small set of vowel-initial words, “he go” → “he goes”, “they goes” → “they go”.
    """
    if not s:
        return s
    # “a apple” → “an apple” (small word list; “a university” stays — /juː/ sound)
    _an_words = (
        "apple",
        "egg",
        "orange",
        "umbrella",
        "island",
        "owl",
        "eagle",
        "ice",
        "idea",
        "hour",
        "honest",
        "honor",
        "octopus",
        "elephant",
        "actor",
        "artist",
        "angel",
        "arrow",
        "echo",
        "entry",
        "exit",
        "oven",
    )
    alt = "|".join(re.escape(w) for w in _an_words)

    def _a_to_an(m) -> str:
        art, w = m.group(1), m.group(2)
        return ("An " if art == "A" else "an ") + w

    s = re.sub(rf"\b([Aa]) ({alt})\b", _a_to_an, s, flags=re.IGNORECASE)
    s = re.sub(r"\b(he|she|it) go\b", lambda m: f"{m.group(1)} goes", s, flags=re.IGNORECASE)
    s = re.sub(r"\bthey goes\b", "they go", s, flags=re.IGNORECASE)
    return s


def light_surface_english(s: str) -> str:
    """
    Cheap cosmetic passes that look a bit more like typed English.
    Does not verify grammar; does not fix wrong or non-words.
    """
    if not s:
        return s
    parts = s.splitlines(keepends=True)
    buf: list[str] = []
    for part in parts:
        buf.append(re.sub(r" {2,}", " ", part))
    t = "".join(buf)
    t = re.sub(r"\bi\b", "I", t)
    t = re.sub(
        r"(^|[.!?]\s+)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        t,
        flags=re.MULTILINE,
    )
    return t


def collapse_punctuation_runs(s: str) -> str:
    """No more than two identical punctuation marks in a row (???, !!!, ...)."""
    s = re.sub(r"\?{3,}", "??", s)
    s = re.sub(r"!{3,}", "!!", s)
    s = re.sub(r"\.{4,}", "...", s)
    s = re.sub(r"-{4,}", "---", s)
    return s


def local_sensible_postprocess(
    s: str,
    max_same: int = 3,
    trim_dup_tail: bool = True,
    surface_english: bool = False,
    grammar_tweaks: bool = True,
) -> str:
    """Apply all local heuristics (no network)."""
    s = collapse_excessive_repeats(s, max_same=max_same)
    s = collapse_repeating_substring_runs(s, min_unit=2, max_unit=6)
    if grammar_tweaks:
        s = light_regex_grammar_fixes(s)
    s = collapse_punctuation_runs(s)
    if trim_dup_tail:
        s = trim_consecutive_duplicate_tail(s, min_unit=10)
    if surface_english:
        s = light_surface_english(s)
    return s


def trim_excessive_leading_newlines(s: str, max_leading: int = 2) -> str:
    """Cap a long run of newlines at the start of a string (display helper)."""
    if not s or max_leading < 1:
        return s
    return re.sub(r"^\n{" + str(max_leading + 1) + r",}", "\n" * max_leading, s)
