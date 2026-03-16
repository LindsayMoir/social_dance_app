#!/usr/bin/env python3
"""
One-time cleanup utility:
Null out clearly over-broad dance_style values on live-music events
when there is no explicit dance-style evidence in event_name/description.

Usage:
  python utilities/fix_broad_live_music_styles.py --dry-run
  python utilities/fix_broad_live_music_styles.py --apply
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Final

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


STYLE_TOKENS: Final[list[str]] = [
    "argentine tango",
    "tango",
    "salsa",
    "bachata",
    "kizomba",
    "semba",
    "urban kiz",
    "tarraxo",
    "tarraxa",
    "tarraxinha",
    "merengue",
    "rumba",
    "swing",
    "west coast swing",
    "wcs",
    "east coast swing",
    "lindy",
    "lindy hop",
    "balboa",
    "cuban salsa",
    "rueda",
]
MAX_REASONABLE_STYLE_COUNT: Final[int] = 4


@dataclass(frozen=True)
class CandidateRow:
    event_id: int
    event_name: str
    dance_style: str
    description: str
    source: str
    url: str


def _style_count(dance_style: str) -> int:
    return len([p.strip() for p in str(dance_style or "").split(",") if p.strip()])


def _has_style_evidence(event_name: str, description: str) -> bool:
    corpus = f"{event_name or ''} {description or ''}".lower()
    return any(token in corpus for token in STYLE_TOKENS)


def _load_candidates(engine) -> list[CandidateRow]:
    query = text(
        """
        SELECT event_id, event_name, dance_style, description, source, url
        FROM events
        WHERE event_type ILIKE '%live music%'
          AND dance_style IS NOT NULL
          AND TRIM(dance_style) <> ''
        ORDER BY event_id
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()
    candidates: list[CandidateRow] = []
    for row in rows:
        candidates.append(
            CandidateRow(
                event_id=int(row[0]),
                event_name=str(row[1] or ""),
                dance_style=str(row[2] or ""),
                description=str(row[3] or ""),
                source=str(row[4] or ""),
                url=str(row[5] or ""),
            )
        )
    return candidates


def _collect_problem_rows(rows: list[CandidateRow]) -> list[CandidateRow]:
    output: list[CandidateRow] = []
    for row in rows:
        if _style_count(row.dance_style) <= MAX_REASONABLE_STYLE_COUNT:
            continue
        if _has_style_evidence(row.event_name, row.description):
            continue
        output.append(row)
    return output


def _apply_update(engine, event_ids: list[int]) -> int:
    if not event_ids:
        return 0
    stmt = text("UPDATE events SET dance_style = NULL WHERE event_id = ANY(:event_ids)")
    with engine.begin() as conn:
        result = conn.execute(stmt, {"event_ids": event_ids})
    return int(result.rowcount or 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply UPDATE (default is dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()

    load_dotenv(dotenv_path=os.path.join("src", ".env"))
    db_url = os.getenv("DATABASE_CONNECTION_STRING") or os.getenv("RENDER_EXTERNAL_DB_URL")
    if not db_url:
        raise RuntimeError("No DB URL found in env (DATABASE_CONNECTION_STRING / RENDER_EXTERNAL_DB_URL).")

    engine = create_engine(db_url)
    all_live_music = _load_candidates(engine)
    flagged = _collect_problem_rows(all_live_music)

    print(f"Live-music rows with non-empty dance_style: {len(all_live_music)}")
    print(f"Flagged over-broad/no-evidence rows: {len(flagged)}")
    for row in flagged[:20]:
        print(
            f"- event_id={row.event_id} source={row.source} "
            f"styles={row.dance_style!r} url={row.url}"
        )

    if args.apply and not args.dry_run:
        updated = _apply_update(engine, [r.event_id for r in flagged])
        print(f"UPDATED rows (dance_style -> NULL): {updated}")
    else:
        print("Dry run only. Use --apply to execute updates.")


if __name__ == "__main__":
    main()

