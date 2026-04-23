from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
import re
from typing import Any, Dict, List

from date_calculator import resolve_temporal_from_text
from utils.sql_filters import (
    _or_group_for_styles,
    detect_excluded_styles_in_text,
    detect_styles_in_text,
    wants_all_styles,
)


DEFAULT_RESULT_LIMIT = 30


@dataclass
class QueryConstraints:
    temporal_phrase: str = ""
    start_date: str = ""
    end_date: str = ""
    time_filter: str = ""
    end_time_filter: str = ""
    weekday_filters: List[int] = field(default_factory=list)
    is_recurring_weekday: bool = False
    include_styles: List[str] = field(default_factory=list)
    exclude_styles: List[str] = field(default_factory=list)
    all_styles: bool = False
    include_event_types: List[str] = field(default_factory=list)
    exclude_event_types: List[str] = field(default_factory=list)
    location_terms: List[str] = field(default_factory=list)
    venue_terms: List[str] = field(default_factory=list)
    city_terms: List[str] = field(default_factory=list)
    source_terms: List[str] = field(default_factory=list)
    limit: int | None = None
    clarification_needed: bool = False
    clarification_message: str = ""
    ambiguity_reasons: List[str] = field(default_factory=list)

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
            weekday_filters=list(data.get("weekday_filters") or []),
            is_recurring_weekday=bool(data.get("is_recurring_weekday")),
            include_styles=list(data.get("include_styles") or []),
            exclude_styles=list(data.get("exclude_styles") or []),
            all_styles=bool(data.get("all_styles")),
            include_event_types=list(data.get("include_event_types") or []),
            exclude_event_types=list(data.get("exclude_event_types") or []),
            location_terms=list(data.get("location_terms") or []),
            venue_terms=list(data.get("venue_terms") or []),
            city_terms=list(data.get("city_terms") or []),
            source_terms=list(data.get("source_terms") or []),
            limit=_coerce_limit(data.get("limit")),
            clarification_needed=bool(data.get("clarification_needed")),
            clarification_message=str(data.get("clarification_message") or ""),
            ambiguity_reasons=list(data.get("ambiguity_reasons") or []),
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

_WEEKDAY_NAME_TO_DOW = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_KNOWN_CITY_TERMS = {
    "victoria",
    "vancouver",
    "seattle",
}


def _coerce_limit(value: Any) -> int | None:
    """Return a safe positive integer limit, or None when missing/invalid."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


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
        term = _clean_extracted_term(raw)
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


def _clean_extracted_term(raw: str) -> str:
    """Trim punctuation and trailing qualifier clauses from an extracted free-text term."""
    term = re.split(r"[,\.;\n]", raw, maxsplit=1)[0].strip(" '\"")
    term = re.split(r"\b(?:for|with|featuring|featuring|including|that has|which has)\b", term, maxsplit=1)[0]
    term = re.sub(r"\s+", " ", term).strip()
    if term.lower().startswith("the "):
        term = term[4:].strip()
    return term


def _extract_recurring_weekday_filters(text: str) -> tuple[List[int], bool]:
    """Detect recurring weekday intent such as 'Tuesdays' or 'every Tuesday'."""
    text_l = str(text or "").lower()
    weekday_filters: List[int] = []
    for weekday_name, dow in _WEEKDAY_NAME_TO_DOW.items():
        recurring_patterns = (
            rf"\b{weekday_name}s\b",
            rf"\bevery\s+{weekday_name}\b",
            rf"\ball\s+{weekday_name}s?\b",
            rf"\bon\s+{weekday_name}s\b",
        )
        if any(re.search(pattern, text_l) for pattern in recurring_patterns):
            weekday_filters.append(dow)
    return list(dict.fromkeys(weekday_filters)), bool(weekday_filters)


def _detect_ambiguity(text: str) -> tuple[list[str], str]:
    """Return parser-detectable ambiguity reasons and a targeted clarification prompt."""
    text_l = str(text or "").lower()
    for weekday_name in _WEEKDAY_NAME_TO_DOW:
        singular_pattern = rf"\b(?:on\s+)?{weekday_name}\b"
        if not re.search(singular_pattern, text_l):
            continue
        if re.search(rf"\b(?:this|next|last|every|all)\s+{weekday_name}\b", text_l):
            continue
        if re.search(rf"\b{weekday_name}s\b", text_l):
            continue
        return (
            ["weekday_scope"],
            f"Do you mean next {weekday_name.title()} or every {weekday_name.title()}?",
        )
    return [], ""


def _extract_limit_from_text(text: str) -> int | None:
    """Extract an explicit requested result limit from user text."""
    text_l = str(text or "").lower()
    patterns = (
        r"\blimit(?:ed)?\s+(?:to\s+)?(\d{1,3})\b",
        r"\bshow me\s+(\d{1,3})\b",
        r"\bgive me\s+(\d{1,3})\b",
        r"\btop\s+(\d{1,3})\b",
        r"\bfirst\s+(\d{1,3})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text_l)
        if not match:
            continue
        return _coerce_limit(match.group(1))
    return None


def _extract_location_buckets_from_text(text: str, current_date: str) -> Dict[str, List[str]]:
    """Split location-like terms into venue, city, and source-oriented buckets."""
    text_l = str(text or "").lower()
    location_terms = _extract_location_terms_from_text(text, current_date)
    venue_terms: List[str] = []
    city_terms: List[str] = []
    source_terms: List[str] = []

    for raw in re.findall(r"\b(?:from|by)\s+([a-z0-9][a-z0-9 '&\.\-]{1,64})", text_l, flags=re.IGNORECASE):
        term = _clean_extracted_term(raw)
        if not term or _looks_temporal_term(term, current_date):
            continue
        source_terms.append(term)
        location_terms.append(term)

    for city in _KNOWN_CITY_TERMS:
        if re.search(rf"\b{re.escape(city)}\b", text_l):
            city_terms.append(city)
            location_terms.append(city)

    for raw in re.findall(r"\b(?:at|near)\s+([a-z0-9][a-z0-9 '&\.\-]{1,64})", text_l, flags=re.IGNORECASE):
        term = _clean_extracted_term(raw)
        if not term or _looks_temporal_term(term, current_date):
            continue
        venue_terms.append(term)

    for raw in re.findall(r"\bin\s+([a-z0-9][a-z0-9 '&\.\-]{1,64})", text_l, flags=re.IGNORECASE):
        term = _clean_extracted_term(raw)
        if not term or _looks_temporal_term(term, current_date):
            continue
        if term in _KNOWN_CITY_TERMS:
            city_terms.append(term)
        else:
            venue_terms.append(term)

    return {
        "location_terms": list(dict.fromkeys(location_terms)),
        "venue_terms": list(dict.fromkeys(venue_terms)),
        "city_terms": list(dict.fromkeys(city_terms)),
        "source_terms": list(dict.fromkeys(source_terms)),
    }


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


def _has_any_keyword(text_l: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text_l for keyword in keywords)


def _is_replace_refinement(text_l: str) -> bool:
    return _has_any_keyword(
        text_l,
        ("instead", "switch to", "change to", "rather than", "different", "actually"),
    )


def _is_narrowing_refinement(text_l: str) -> bool:
    return _has_any_keyword(
        text_l,
        ("only ", "just ", "restrict to", "nothing but"),
    )


def _derive_atomic_constraints_from_text(text: str, current_date: str) -> QueryConstraints:
    """Parse only the provided text into a fresh constraints object."""
    constraints = QueryConstraints()
    user_text = str(text or "").strip()
    user_text_l = user_text.lower()

    resolved = resolve_temporal_from_text(user_text, current_date)
    if resolved:
        constraints.temporal_phrase = str(resolved.get("temporal_phrase") or "")
        constraints.start_date = str(resolved.get("start_date") or "")
        constraints.end_date = str(resolved.get("end_date") or "")
        constraints.time_filter = str(resolved.get("time_filter") or "")
        constraints.end_time_filter = str(resolved.get("end_time_filter") or "")

    weekday_filters, is_recurring_weekday = _extract_recurring_weekday_filters(user_text)
    ambiguity_reasons, clarification_message = _detect_ambiguity(user_text)
    explicit_all_styles = wants_all_styles(user_text)
    if explicit_all_styles:
        constraints.all_styles = True
        constraints.include_styles = []

    constraints.include_styles = [str(s) for s in detect_styles_in_text(user_text)]
    constraints.exclude_styles = [str(s) for s in detect_excluded_styles_in_text(user_text)]
    constraints.include_event_types = _detect_event_types_from_text(user_text)
    constraints.exclude_event_types = _detect_excluded_event_types(user_text)
    location_buckets = _extract_location_buckets_from_text(user_text, current_date)
    constraints.location_terms = location_buckets["location_terms"]
    constraints.venue_terms = location_buckets["venue_terms"]
    constraints.city_terms = location_buckets["city_terms"]
    constraints.source_terms = location_buckets["source_terms"]
    constraints.limit = _extract_limit_from_text(user_text)

    if constraints.exclude_styles:
        constraints.include_styles = [s for s in constraints.include_styles if s not in constraints.exclude_styles]
    if constraints.exclude_event_types:
        constraints.include_event_types = [
            t for t in constraints.include_event_types if t not in constraints.exclude_event_types
        ]
    if weekday_filters:
        constraints.weekday_filters = weekday_filters
        constraints.is_recurring_weekday = is_recurring_weekday
    if ambiguity_reasons:
        constraints.clarification_needed = True
        constraints.clarification_message = clarification_message
        constraints.ambiguity_reasons = ambiguity_reasons
    return constraints


def _merge_constraints(base: QueryConstraints, update: QueryConstraints, user_text: str, is_clarification: bool) -> QueryConstraints:
    """Merge a refinement/clarification delta into existing constraints."""
    merged = QueryConstraints.from_dict(base.to_dict())
    user_text_l = str(user_text or "").lower()
    replace_refinement = _is_replace_refinement(user_text_l)
    narrowing_refinement = _is_narrowing_refinement(user_text_l)
    explicit_all_styles = wants_all_styles(user_text)
    negated_narrowing_language = ("not just", "not only")
    should_avoid_narrowing = is_clarification and _has_any_keyword(user_text_l, negated_narrowing_language)

    prior_span = _date_span_days(merged.start_date, merged.end_date)
    next_span = _date_span_days(update.start_date, update.end_date)
    if update.start_date or update.end_date or update.temporal_phrase:
        if not (should_avoid_narrowing and prior_span > next_span > 0):
            merged.temporal_phrase = update.temporal_phrase
            merged.start_date = update.start_date
            merged.end_date = update.end_date
            merged.time_filter = update.time_filter
            merged.end_time_filter = update.end_time_filter
            merged.weekday_filters = list(update.weekday_filters)
            merged.is_recurring_weekday = update.is_recurring_weekday
    elif update.weekday_filters:
        merged.weekday_filters = list(update.weekday_filters)
        merged.is_recurring_weekday = update.is_recurring_weekday

    if explicit_all_styles:
        merged.all_styles = True
        merged.include_styles = []

    if update.include_styles:
        if replace_refinement or narrowing_refinement or explicit_all_styles:
            merged.include_styles = list(update.include_styles)
            if update.include_styles:
                merged.exclude_styles = [s for s in merged.exclude_styles if s not in update.include_styles]
        else:
            merged.include_styles = list(dict.fromkeys([*merged.include_styles, *update.include_styles]))
        merged.all_styles = False

    if update.exclude_styles:
        merged.exclude_styles = list(dict.fromkeys([*merged.exclude_styles, *update.exclude_styles]))
    if merged.exclude_styles:
        merged.include_styles = [s for s in merged.include_styles if s not in merged.exclude_styles]

    if update.include_event_types:
        if replace_refinement or narrowing_refinement:
            merged.include_event_types = list(update.include_event_types)
            merged.exclude_event_types = [t for t in merged.exclude_event_types if t not in update.include_event_types]
        else:
            merged.include_event_types = list(
                dict.fromkeys([*merged.include_event_types, *update.include_event_types])
            )
    if update.exclude_event_types:
        merged.exclude_event_types = list(
            dict.fromkeys([*merged.exclude_event_types, *update.exclude_event_types])
        )
    if merged.exclude_event_types:
        merged.include_event_types = [t for t in merged.include_event_types if t not in merged.exclude_event_types]

    if update.limit is not None:
        merged.limit = update.limit

    has_location_update = any(
        [update.location_terms, update.venue_terms, update.city_terms, update.source_terms]
    )
    if has_location_update:
        if replace_refinement:
            merged.location_terms = list(update.location_terms)
            merged.venue_terms = list(update.venue_terms)
            merged.city_terms = list(update.city_terms)
            merged.source_terms = list(update.source_terms)
        else:
            merged.location_terms = list(dict.fromkeys([*merged.location_terms, *update.location_terms]))
            merged.venue_terms = list(dict.fromkeys([*merged.venue_terms, *update.venue_terms]))
            merged.city_terms = list(dict.fromkeys([*merged.city_terms, *update.city_terms]))
            merged.source_terms = list(dict.fromkeys([*merged.source_terms, *update.source_terms]))

    if update.clarification_needed:
        merged.clarification_needed = True
        merged.clarification_message = update.clarification_message
        merged.ambiguity_reasons = list(update.ambiguity_reasons)
    else:
        merged.clarification_needed = False
        merged.clarification_message = ""
        merged.ambiguity_reasons = []

    return merged


def derive_constraints_from_text(
    text: str,
    current_date: str,
    base_constraints: Dict[str, Any] | None = None,
    is_clarification: bool = False,
) -> Dict[str, Any]:
    update = _derive_atomic_constraints_from_text(text, current_date)
    if not base_constraints:
        return update.to_dict()
    base = QueryConstraints.from_dict(base_constraints)
    merged = _merge_constraints(base, update, text, is_clarification=is_clarification)
    return merged.to_dict()


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

    if constraints.weekday_filters and constraints.is_recurring_weekday:
        weekday_names = [name for name, dow in _WEEKDAY_NAME_TO_DOW.items() if dow in constraints.weekday_filters]
        if weekday_names:
            parts.append("on " + " or ".join(weekday_names))

    if constraints.limit:
        parts.append(f"limit {constraints.limit}")

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
    effective_limit = constraints.limit or int(limit)

    if constraints.start_date:
        filters.append(f"start_date >= '{constraints.start_date}'")
    if constraints.end_date:
        filters.append(f"start_date <= '{constraints.end_date}'")
    if constraints.time_filter:
        filters.append(f"start_time >= '{constraints.time_filter}'")
    if constraints.end_time_filter:
        filters.append(f"start_time <= '{constraints.end_time_filter}'")
    if constraints.weekday_filters and constraints.is_recurring_weekday:
        weekday_numbers = ", ".join(str(int(day)) for day in constraints.weekday_filters)
        filters.append(f"EXTRACT(DOW FROM start_date) IN ({weekday_numbers})")

    if constraints.include_styles and not constraints.all_styles:
        style_group = _or_group_for_styles(constraints.include_styles)
        if style_group:
            filters.append(style_group)

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

    location_terms = list(
        dict.fromkeys(
            [
                *constraints.location_terms,
                *constraints.venue_terms,
                *constraints.city_terms,
                *constraints.source_terms,
            ]
        )
    )
    if location_terms:
        location_filters = []
        for term in location_terms:
            safe_term = term.replace("'", "")
            location_filters.append(
                "("
                f"location ILIKE '%{safe_term}%' OR "
                f"source ILIKE '%{safe_term}%' OR "
                f"url ILIKE '%{safe_term}%'"
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
        + f" ORDER BY start_date, start_time LIMIT {effective_limit}"
    )
