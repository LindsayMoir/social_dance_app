from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
import re
from typing import Any, Dict, List

from date_calculator import resolve_temporal_from_text
from utils.sql_filters import detect_excluded_styles_in_text, detect_styles_in_text, wants_all_styles


DEFAULT_RESULT_LIMIT = 30


@dataclass
class QueryConstraints:
    temporal_phrase: str = ""
    start_date: str = ""
    end_date: str = ""
    time_filter: str = ""
    end_time_filter: str = ""
    include_styles: List[str] = field(default_factory=list)
    exclude_styles: List[str] = field(default_factory=list)
    all_styles: bool = False
    include_event_types: List[str] = field(default_factory=list)
    exclude_event_types: List[str] = field(default_factory=list)
    location_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "QueryConstraints":
        if not isinstance(data, dict):
            return cls()
        return cls(
            temporal_phrase=str(data.get("temporal_phrase") or ""),
            start_date=str(data.get("start_date") or ""),
            end_date=str(data.get("end_date") or ""),
            time_filter=str(data.get("time_filter") or ""),
            end_time_filter=str(data.get("end_time_filter") or ""),
            include_styles=list(data.get("include_styles") or []),
            exclude_styles=list(data.get("exclude_styles") or []),
            all_styles=bool(data.get("all_styles")),
            include_event_types=list(data.get("include_event_types") or []),
            exclude_event_types=list(data.get("exclude_event_types") or []),
            location_terms=list(data.get("location_terms") or []),
        )


_EVENT_TYPE_PATTERNS = {
    "social dance": [r"\bsocial dance\b", r"\bsocial dances\b", r"\bsocial\b"],
    "class": [r"\bclass\b", r"\bclasses\b", r"\blesson\b", r"\blessons\b"],
    "workshop": [r"\bworkshop\b", r"\bworkshops\b"],
    "live music": [r"\blive music\b", r"\blive band\b", r"\blive bands\b"],
}

_LOCATION_STOPWORDS = {
    "the",
    "a",
    "an",
    "events",
    "event",
    "dance",
    "dances",
    "music",
    "please",
    "tonight",
    "today",
    "tomorrow",
    "week",
    "weekend",
    "month",
}


def _looks_temporal_term(term: str, current_date: str) -> bool:
    """Return True when a candidate location term is actually temporal language."""
    phrase = str(term or "").strip().lower()
    if not phrase:
        return False
    if resolve_temporal_from_text(phrase, current_date):
        return True
    temporal_tokens = (
        "week",
        "weekend",
        "month",
        "today",
        "tomorrow",
        "tonight",
        "yesterday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    )
    return any(tok in phrase for tok in temporal_tokens)


def _detect_event_types_from_text(text: str) -> List[str]:
    found: List[str] = []
    text_l = str(text or "").lower()
    for canonical, patterns in _EVENT_TYPE_PATTERNS.items():
        if any(re.search(p, text_l) for p in patterns):
            found.append(canonical)
    return list(dict.fromkeys(found))


def _detect_excluded_event_types(text: str) -> List[str]:
    text_l = str(text or "").lower()
    excluded: List[str] = []
    for canonical, patterns in _EVENT_TYPE_PATTERNS.items():
        for pat in patterns:
            token_match = re.search(pat, text_l)
            if not token_match:
                continue
            prefix = text_l[max(0, token_match.start() - 24):token_match.start()]
            if any(neg in prefix for neg in ("no ", "not ", "without ", "exclude ")):
                excluded.append(canonical)
                break
    return list(dict.fromkeys(excluded))


def _extract_location_terms_from_text(text: str, current_date: str) -> List[str]:
    text_l = str(text or "").lower()
    matches = re.findall(
        r"\b(?:at|in|near)\s+([a-z0-9][a-z0-9 '&\.\-]{1,64})",
        text_l,
        flags=re.IGNORECASE,
    )
    cleaned: List[str] = []
    for raw in matches:
        term = re.split(r"[,\.;\n]", raw, maxsplit=1)[0].strip(" '\"")
        term = re.sub(r"\s+", " ", term).strip()
        if term.startswith("the "):
            term = term[4:].strip()
        if not term:
            continue
        if term in _LOCATION_STOPWORDS:
            continue
        if len(term) < 3:
            continue
        if _looks_temporal_term(term, current_date):
            continue
        cleaned.append(term)
    return list(dict.fromkeys(cleaned))


def _date_span_days(start_date: str, end_date: str) -> int:
    """Return inclusive day span for YYYY-MM-DD dates; 0 when invalid/missing."""
    if not start_date or not end_date:
        return 0
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return 0
    return max(0, (end_dt - start_dt).days + 1)


def derive_constraints_from_text(
    text: str,
    current_date: str,
    base_constraints: Dict[str, Any] | None = None,
    is_clarification: bool = False,
) -> Dict[str, Any]:
    constraints = QueryConstraints.from_dict(base_constraints if is_clarification else None)
    user_text = str(text or "").strip()
    user_text_l = user_text.lower()

    prior_start = constraints.start_date
    prior_end = constraints.end_date
    resolved = resolve_temporal_from_text(user_text, current_date)
    negated_narrowing_language = ("not just", "not only")
    should_avoid_narrowing = is_clarification and any(t in user_text_l for t in negated_narrowing_language)
    if resolved:
        next_start = str(resolved.get("start_date") or "")
        next_end = str(resolved.get("end_date") or "")
        prior_span = _date_span_days(prior_start, prior_end)
        next_span = _date_span_days(next_start, next_end)
        if not (should_avoid_narrowing and prior_span > next_span > 0):
            constraints.temporal_phrase = str(resolved.get("temporal_phrase") or "")
            constraints.start_date = next_start
            constraints.end_date = next_end
            constraints.time_filter = str(resolved.get("time_filter") or "")
            constraints.end_time_filter = str(resolved.get("end_time_filter") or "")

    explicit_all_styles = wants_all_styles(user_text)
    if explicit_all_styles:
        constraints.all_styles = True
        constraints.include_styles = []

    include_styles = [str(s) for s in detect_styles_in_text(user_text)]
    exclude_styles = [str(s) for s in detect_excluded_styles_in_text(user_text)]
    include_event_types = _detect_event_types_from_text(user_text)
    exclude_event_types = _detect_excluded_event_types(user_text)
    location_terms = _extract_location_terms_from_text(user_text, current_date)

    replace_terms = ("instead", "not just", "not only", "any style", "all styles", "all dance")
    should_replace_styles = explicit_all_styles or any(t in user_text_l for t in replace_terms)

    if include_styles:
        if should_replace_styles:
            constraints.include_styles = include_styles
        else:
            merged = list(dict.fromkeys([*constraints.include_styles, *include_styles]))
            constraints.include_styles = merged
        constraints.all_styles = False

    if exclude_styles:
        merged_excluded = list(dict.fromkeys([*constraints.exclude_styles, *exclude_styles]))
        constraints.exclude_styles = merged_excluded

    if constraints.exclude_styles:
        constraints.include_styles = [s for s in constraints.include_styles if s not in constraints.exclude_styles]

    if include_event_types:
        merged_event_types = list(dict.fromkeys([*constraints.include_event_types, *include_event_types]))
        constraints.include_event_types = merged_event_types

    if exclude_event_types:
        merged_excluded_event_types = list(
            dict.fromkeys([*constraints.exclude_event_types, *exclude_event_types])
        )
        constraints.exclude_event_types = merged_excluded_event_types

    if constraints.exclude_event_types:
        constraints.include_event_types = [
            t for t in constraints.include_event_types if t not in constraints.exclude_event_types
        ]

    if location_terms:
        merged_location_terms = list(dict.fromkeys([*constraints.location_terms, *location_terms]))
        constraints.location_terms = merged_location_terms

    return constraints.to_dict()


def constraints_to_query_text(constraints_dict: Dict[str, Any], fallback_text: str = "") -> str:
    constraints = QueryConstraints.from_dict(constraints_dict)
    parts: List[str] = ["show me"]

    if constraints.all_styles:
        parts.append("all dance events")
    elif constraints.include_styles:
        joined_styles = ", ".join(constraints.include_styles)
        parts.append(f"{joined_styles} dance events")
    else:
        parts.append("dance events")

    if constraints.exclude_styles:
        excluded = ", ".join(constraints.exclude_styles)
        parts.append(f"excluding {excluded}")

    if constraints.include_event_types:
        parts.append("including " + ", ".join(constraints.include_event_types))

    if constraints.exclude_event_types:
        parts.append("excluding event types " + ", ".join(constraints.exclude_event_types))

    if constraints.location_terms:
        parts.append("at " + " or ".join(constraints.location_terms))

    if constraints.start_date and constraints.end_date:
        if constraints.start_date == constraints.end_date:
            parts.append(f"on {constraints.start_date}")
        else:
            parts.append(f"from {constraints.start_date} to {constraints.end_date}")

    query_text = " ".join(parts)
    return query_text if query_text.strip() else fallback_text


def build_sql_from_constraints(constraints_dict: Dict[str, Any], limit: int = DEFAULT_RESULT_LIMIT) -> str | None:
    constraints = QueryConstraints.from_dict(constraints_dict)
    filters: List[str] = []

    if constraints.start_date:
        filters.append(f"start_date >= '{constraints.start_date}'")
    if constraints.end_date:
        filters.append(f"start_date <= '{constraints.end_date}'")
    if constraints.time_filter:
        filters.append(f"start_time >= '{constraints.time_filter}'")
    if constraints.end_time_filter:
        filters.append(f"start_time <= '{constraints.end_time_filter}'")

    if constraints.include_styles and not constraints.all_styles:
        style_filters = []
        for style in constraints.include_styles:
            safe_style = style.replace("'", "")
            style_filters.append(f"dance_style ILIKE '%{safe_style}%'")
        filters.append("( " + " OR ".join(style_filters) + " )")

    for style in constraints.exclude_styles:
        safe_style = style.replace("'", "")
        filters.append(f"dance_style NOT ILIKE '%{safe_style}%'")

    if constraints.include_event_types:
        event_type_filters = []
        for event_type in constraints.include_event_types:
            safe_type = event_type.replace("'", "")
            event_type_filters.append(f"event_type ILIKE '%{safe_type}%'")
        filters.append("( " + " OR ".join(event_type_filters) + " )")

    for event_type in constraints.exclude_event_types:
        safe_type = event_type.replace("'", "")
        filters.append(f"event_type NOT ILIKE '%{safe_type}%'")

    if constraints.location_terms:
        location_filters = []
        for term in constraints.location_terms:
            safe_term = term.replace("'", "")
            location_filters.append(
                "("
                f"location ILIKE '%{safe_term}%' OR "
                f"source ILIKE '%{safe_term}%'"
                ")"
            )
        filters.append("( " + " OR ".join(location_filters) + " )")

    if not filters:
        return None

    cols = (
        "event_name, event_type, dance_style, day_of_week, start_date, end_date, "
        "start_time, end_time, source, url, price, description, location"
    )
    return (
        f"SELECT {cols} FROM events WHERE "
        + " AND ".join(filters)
        + f" ORDER BY start_date, start_time LIMIT {int(limit)}"
    )
