from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
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
        )


def derive_constraints_from_text(
    text: str,
    current_date: str,
    base_constraints: Dict[str, Any] | None = None,
    is_clarification: bool = False,
) -> Dict[str, Any]:
    constraints = QueryConstraints.from_dict(base_constraints if is_clarification else None)
    user_text = str(text or "").strip()
    user_text_l = user_text.lower()

    resolved = resolve_temporal_from_text(user_text, current_date)
    if resolved:
        constraints.temporal_phrase = str(resolved.get("temporal_phrase") or "")
        constraints.start_date = str(resolved.get("start_date") or "")
        constraints.end_date = str(resolved.get("end_date") or "")
        constraints.time_filter = str(resolved.get("time_filter") or "")
        constraints.end_time_filter = str(resolved.get("end_time_filter") or "")

    explicit_all_styles = wants_all_styles(user_text)
    if explicit_all_styles:
        constraints.all_styles = True
        constraints.include_styles = []

    include_styles = [str(s) for s in detect_styles_in_text(user_text)]
    exclude_styles = [str(s) for s in detect_excluded_styles_in_text(user_text)]

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
