"""
Offline training and runtime inference for the page classifier ML models.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

from evaluation_holdout import load_dev_urls, load_gold_holdout_urls, normalize_evaluation_url

try:
    import joblib
except ImportError:  # pragma: no cover - exercised in deployment environments without ML deps
    joblib = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised in deployment environments without ML deps
    pd = None

try:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline
except ImportError:  # pragma: no cover - exercised in deployment environments without ML deps
    DictVectorizer = None
    LogisticRegression = None
    OneVsRestClassifier = None
    Pipeline = Any  # type: ignore[assignment]


DEFAULT_MODEL_PATH: Path = Path(__file__).resolve().parent.parent / "ml" / "models" / "page_classifier_models.joblib"
_MODEL_CACHE: dict[str, Any] | None = None
_EMAIL_INPUT_RE = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.IGNORECASE)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MLPrediction:
    archetype: str
    owner_step: str
    archetype_confidence: float
    owner_confidence: float
    model_version: str


def train_page_classifier_models(
    *,
    training_csv_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Train archetype and owner-step models from the labeled training CSV.
    """
    _require_training_dependencies()
    csv_path = Path(training_csv_path)
    df = pd.read_csv(csv_path)
    df = df.fillna("")
    labeled = df[
        (df["reviewed_truth_archetype"].astype(str).str.strip() != "")
        & (df["reviewed_truth_owner_step"].astype(str).str.strip() != "")
    ].copy()
    labeled = labeled[~labeled["url"].astype(str).map(_is_email_like_input)].copy()
    labeled = labeled[labeled["reviewed_truth_owner_step"].astype(str).str.strip() != "emails.py"].copy()
    excluded_urls = load_gold_holdout_urls() | load_dev_urls()
    if excluded_urls:
        normalized_urls = labeled["url"].astype(str).map(normalize_evaluation_url)
        labeled = labeled[~normalized_urls.isin(excluded_urls)].copy()
    if labeled.empty:
        raise ValueError(f"No labeled rows found in {csv_path}")

    feature_rows = [build_model_features_from_training_row(row) for row in labeled.to_dict(orient="records")]
    archetype_target = labeled["reviewed_truth_archetype"].astype(str).str.strip().tolist()
    owner_target = labeled["reviewed_truth_owner_step"].astype(str).str.strip().tolist()

    archetype_pipeline = _build_pipeline()
    owner_pipeline = _build_pipeline()
    archetype_pipeline.fit(feature_rows, archetype_target)
    owner_pipeline.fit(feature_rows, owner_target)

    artifact = {
        "model_version": "v1",
        "training_csv_path": str(csv_path),
        "training_row_count": int(len(labeled)),
        "archetype_labels": sorted(set(archetype_target)),
        "owner_labels": sorted(set(owner_target)),
        "archetype_model": archetype_pipeline,
        "owner_model": owner_pipeline,
    }
    model_path = Path(output_path) if output_path else DEFAULT_MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    reset_model_cache()
    return {
        "output_path": str(model_path),
        "training_row_count": int(len(labeled)),
        "archetype_labels": artifact["archetype_labels"],
        "owner_labels": artifact["owner_labels"],
    }


def predict_page_classifier_labels(
    *,
    url: str,
    page_links_count: int = 0,
    calendar_sources_count: int = 0,
    calendar_ids_count: int = 0,
    listing_signal: bool = False,
    repeated_date_tokens: int = 0,
    listing_score: int = 0,
    event_signal: bool = False,
    html_features: dict[str, Any] | None = None,
) -> MLPrediction | None:
    """
    Predict archetype and owner-step labels using the saved ML artifact.
    """
    artifact = load_page_classifier_models()
    if artifact is None:
        return None

    sample = build_model_features(
        url=url,
        page_links_count=page_links_count,
        calendar_sources_count=calendar_sources_count,
        calendar_ids_count=calendar_ids_count,
        listing_signal=listing_signal,
        repeated_date_tokens=repeated_date_tokens,
        listing_score=listing_score,
        event_signal=event_signal,
        html_features=html_features,
    )
    archetype_model = artifact["archetype_model"]
    owner_model = artifact["owner_model"]

    archetype_pred = str(archetype_model.predict([sample])[0])
    owner_pred = str(owner_model.predict([sample])[0])
    archetype_conf = _max_probability(archetype_model, sample)
    owner_conf = _max_probability(owner_model, sample)
    return MLPrediction(
        archetype=archetype_pred,
        owner_step=owner_pred,
        archetype_confidence=archetype_conf,
        owner_confidence=owner_conf,
        model_version=str(artifact.get("model_version") or "unknown"),
    )


def load_page_classifier_models() -> dict[str, Any] | None:
    """
    Load the saved ML artifact if present.
    """
    global _MODEL_CACHE
    if str(os.getenv("PAGE_CLASSIFIER_ML_ENABLED", "1")).strip().lower() in {"0", "false", "no"}:
        return None
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    env_model_path = str(os.getenv("PAGE_CLASSIFIER_MODEL_PATH", "")).strip()
    if env_model_path:
        model_path = Path(env_model_path).expanduser()
    else:
        model_path = DEFAULT_MODEL_PATH
    if not model_path.exists():
        return None
    if joblib is None:
        LOGGER.warning("Page classifier ML artifact exists but joblib is not installed; skipping ML inference.")
        return None
    _MODEL_CACHE = joblib.load(model_path)
    return _MODEL_CACHE


def reset_model_cache() -> None:
    global _MODEL_CACHE
    _MODEL_CACHE = None


def build_model_features_from_training_row(row: dict[str, Any]) -> dict[str, Any]:
    """
    Build runtime-equivalent features from one labeled CSV row.
    """
    return build_model_features(
        url=str(row.get("url") or ""),
        page_links_count=_safe_int(row.get("feature_event_like_links")),
        calendar_sources_count=1 if _safe_bool(row.get("predicted_is_calendar")) else 0,
        calendar_ids_count=0,
        listing_signal=_safe_bool(row.get("feature_listing_signal")),
        repeated_date_tokens=_safe_int(row.get("feature_repeated_date_tokens")),
        listing_score=_safe_int(row.get("feature_listing_score")),
        event_signal=_safe_bool(row.get("feature_event_signal")),
        html_features=None,
    )


def build_model_features(
    *,
    url: str,
    page_links_count: int = 0,
    calendar_sources_count: int = 0,
    calendar_ids_count: int = 0,
    listing_signal: bool = False,
    repeated_date_tokens: int = 0,
    listing_score: int = 0,
    event_signal: bool = False,
    html_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a compact feature dict available at runtime and training time.
    """
    safe_url = str(url or "").strip()
    parsed = urlparse(safe_url)
    domain = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    query = (parsed.query or "").lower()

    features: dict[str, Any] = {
        "page_links_count": int(page_links_count or 0),
        "calendar_sources_count": int(calendar_sources_count or 0),
        "calendar_ids_count": int(calendar_ids_count or 0),
        "listing_signal": bool(listing_signal),
        "repeated_date_tokens": int(repeated_date_tokens or 0),
        "listing_score": int(listing_score or 0),
        "event_signal": bool(event_signal),
        "url_length": len(safe_url),
        "path_depth": len([segment for segment in path.split("/") if segment]),
        "has_query": bool(query),
        "path_has_event": "/event" in path or "/events" in path,
        "path_has_calendar": "calendar" in path or "calendar" in query,
        "path_has_show": "/show/" in path,
        "path_has_ticket": "ticket" in path or "ticket" in query,
        "domain_has_eventbrite": "eventbrite" in domain,
        "domain_has_facebook": "facebook" in domain,
        "domain_has_instagram": "instagram" in domain,
        "domain_has_google": "google" in domain,
    }
    for token in _tokenize_domain(domain):
        features[f"domain_token:{token}"] = 1
    for token in _tokenize_path(path):
        features[f"path_token:{token}"] = 1
    if html_features:
        for key, value in html_features.items():
            normalized_key = str(key or "").strip()
            if not normalized_key:
                continue
            if isinstance(value, bool):
                features[normalized_key] = bool(value)
            elif isinstance(value, (int, float)):
                features[normalized_key] = value
    return features


def _build_pipeline() -> Pipeline:
    _require_training_dependencies()
    return Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=True)),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=42,
                    )
                ),
            ),
        ]
    )


def _max_probability(model: Pipeline, sample: dict[str, Any]) -> float:
    proba = model.predict_proba([sample])[0]
    return float(max(proba)) if len(proba) else 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _tokenize_domain(domain: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", str(domain or "").lower()) if token and token not in {"www", "com", "ca", "org", "net"}]


def _tokenize_path(path: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", str(path or "").lower()) if token and token not in {"html", "php"}]


def _is_email_like_input(value: Any) -> bool:
    return _EMAIL_INPUT_RE.match(str(value or "").strip()) is not None


def _require_training_dependencies() -> None:
    missing: list[str] = []
    if joblib is None:
        missing.append("joblib")
    if pd is None:
        missing.append("pandas")
    if DictVectorizer is None or LogisticRegression is None or OneVsRestClassifier is None:
        missing.append("scikit-learn")
    if missing:
        raise RuntimeError(
            "Page classifier ML dependencies are missing: "
            + ", ".join(sorted(set(missing)))
        )
