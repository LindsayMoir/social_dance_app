from __future__ import annotations

import re
from typing import Dict, List, Set


_STYLE_SYNONYMS: Dict[str, Set[str]] = {
    # canonical: synonyms/variants
    "west coast swing": {"west coast swing", "wcs"},
    "east coast swing": {"east coast swing", "ecs"},
    "salsa": {"salsa"},
    "bachata": {"bachata"},
    "kizomba": {"kizomba"},
    "tango": {"tango", "argentine tango"},
    "argentine tango": {"argentine tango", "tango"},
    "lindy hop": {"lindy hop", "lindy"},
    "lindy": {"lindy", "lindy hop"},
    "zouk": {"zouk"},
}


def _is_all_styles_request(text: str) -> bool:
    """Return True when user asks for broad style coverage instead of a specific style filter."""
    t = (text or "").lower()
    broad_patterns = (
        "all dance events",
        "all dances",
        "all styles",
        "any dance style",
        "any styles",
        "not just",
        "not only",
    )
    return any(p in t for p in broad_patterns)


def wants_all_styles(text: str) -> bool:
    """Public helper for broad style requests."""
    return _is_all_styles_request(text)


def _is_negated_style_mention(text: str, start_idx: int) -> bool:
    """Detect obvious negation immediately before a style mention."""
    window = text[max(0, start_idx - 24):start_idx]
    return bool(
        re.search(
            r"(?:\bnot\s+(?:just|only)?\s*|\bno\s+|\bwithout\s+|\bexcluding?\s+|\bexcept\s+)$",
            window,
            flags=re.IGNORECASE,
        )
    )


def _is_excluded_style_mention(text: str, start_idx: int) -> bool:
    """Detect exclusion language near a style token."""
    window = text[max(0, start_idx - 24):start_idx]
    return bool(
        re.search(
            r"(?:\bno\s+|\bwithout\s+|\bexcluding?\s+|\bexcept\s+)$",
            window,
            flags=re.IGNORECASE,
        )
    )


def _styles_from_text(text: str) -> List[str]:
    """Return a list of canonical styles mentioned in the text (case-insensitive)."""
    t = (text or "").lower()
    if _is_all_styles_request(t):
        return []

    found: List[str] = []
    for canonical, syns in _STYLE_SYNONYMS.items():
        for syn in syns:
            pattern = rf"\b{re.escape(syn)}\b"
            matches = list(re.finditer(pattern, t, flags=re.IGNORECASE))
            if not matches:
                continue
            if any(not _is_negated_style_mention(t, m.start()) for m in matches):
                if canonical not in found:
                    found.append(canonical)
                break
    return found


def detect_excluded_styles_in_text(text: str) -> List[str]:
    """Return canonical styles that are explicitly excluded by the user text."""
    t = (text or "").lower()
    found: List[str] = []
    for canonical, syns in _STYLE_SYNONYMS.items():
        for syn in syns:
            pattern = rf"\b{re.escape(syn)}\b"
            matches = list(re.finditer(pattern, t, flags=re.IGNORECASE))
            if not matches:
                continue
            if any(_is_excluded_style_mention(t, m.start()) for m in matches):
                if canonical not in found:
                    found.append(canonical)
                break
    return found


def _or_group_for_styles(styles: List[str]) -> str:
    """Build a single OR group over dance_style ILIKE conditions for the styles (with synonyms)."""
    parts: List[str] = []
    seen: Set[str] = set()
    for canonical in styles:
        for syn in _STYLE_SYNONYMS.get(canonical, {canonical}):
            # normalize spacing around hyphens
            syn_norm = syn.replace("-", " ")
            key = syn_norm.lower()
            if key in seen:
                continue
            seen.add(key)
            parts.append(f"dance_style ILIKE '%{syn_norm}%'")
    if not parts:
        return ""
    return "( " + " OR ".join(parts) + " )"


def enforce_dance_style(sql: str, user_text: str) -> str:
    """
    Idempotently add a dance_style filter when the user_text explicitly mentions one or more styles.

    - If no explicit style is mentioned, return SQL unchanged.
    - If SQL already contains a dance_style condition, return unchanged (do not duplicate).
    - Otherwise, insert a single OR group for the style(s) before ORDER BY/LIMIT.
    """
    if not sql:
        return sql

    # Only consider dance_style present if it appears in a condition, not just in SELECT list
    if re.search(r"(?i)\bdance_style\s*(=|ILIKE|LIKE|IN|IS)\b", sql):
        return sql

    if _is_all_styles_request(user_text):
        return sql

    styles = _styles_from_text(user_text)
    if not styles:
        return sql

    group = _or_group_for_styles(styles)
    if not group:
        return sql

    insert_pt = len(sql)
    m2 = re.search(r"(?i)\bORDER\s+BY\b", sql)
    if m2:
        insert_pt = m2.start()

    prefix = sql[:insert_pt].rstrip()
    suffix = sql[insert_pt:]

    if re.search(r"(?i)\bWHERE\b", prefix):
        prefix += f" AND {group}"
    else:
        prefix += f" WHERE {group}"

    return prefix + " " + suffix.lstrip()



def detect_styles_in_text(text: str) -> list:
    """
    Return a list of canonical dance styles detected in text.
    Uses the same synonym map as enforcement to keep behavior consistent.
    """
    return _styles_from_text(text)
