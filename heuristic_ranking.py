from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RankingConfig:
    keywords: Sequence[str]
    weight_similarity: float = 0.7
    weight_connections: float = 0.2
    weight_location: float = 0.1
    preferred_locations: Optional[Sequence[str]] = None
    starred_ids: Optional[Iterable[int]] = None


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    expected_cols = {"id", "job_title", "location", "connection"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df.copy()


def load_candidates(csv_path: Path | str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path.resolve()}")
    df = pd.read_csv(path)
    return _ensure_required_columns(df)


def _parse_connection(value: object) -> int:
    if pd.isna(value):
        return 0
    text = str(value).strip().replace(",", "")
    if text.endswith("+"):
        text = text[:-1]
    return int(text) if text.isdigit() else 0


def _normalise(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    v_min, v_max = values.min(), values.max()
    return (values - v_min) / (v_max - v_min) if v_max > v_min else np.zeros_like(values)


def _location_priority(locations: pd.Series, preferred: Optional[Sequence[str]]) -> np.ndarray:
    if preferred:
        preferred_set = {loc.strip().lower() for loc in preferred if loc}
        return locations.fillna("").str.lower().isin(preferred_set).astype(float).to_numpy()
    # Fallback: promote the most common location among top similarity matches later in the pipeline.
    return np.zeros(len(locations), dtype=float)


def score_candidates(df: pd.DataFrame, config: RankingConfig) -> pd.DataFrame:
    if not config.keywords:
        raise ValueError("At least one keyword is required")

    working = df.copy()
    working = working.drop_duplicates(subset=["id"]).reset_index(drop=True)
    working["job_title"] = working["job_title"].fillna("")

    keywords_text = " ".join(config.keywords)

    vectoriser = TfidfVectorizer()
    title_vectors = vectoriser.fit_transform(working["job_title"])
    keyword_vector = vectoriser.transform([keywords_text])

    keyword_similarity = cosine_similarity(title_vectors, keyword_vector).flatten()
    keyword_similarity_scaled = _normalise(keyword_similarity)

    starred_similarity_scaled = np.zeros_like(keyword_similarity_scaled)
    if config.starred_ids:
        starred_rows = working[working["id"].isin(list(config.starred_ids))]
        if not starred_rows.empty:
            starred_vector = title_vectors[starred_rows.index].mean(axis=0)
            starred_similarity = cosine_similarity(title_vectors, starred_vector).flatten()
            starred_similarity_scaled = _normalise(np.asarray(starred_similarity).ravel())

    combined_similarity = (
        0.75 * keyword_similarity_scaled + 0.25 * starred_similarity_scaled
        if config.starred_ids
        else keyword_similarity_scaled
    )

    connections_numeric = working["connection"].apply(_parse_connection).astype(float).to_numpy()
    connections_scaled = _normalise(connections_numeric)

    location_bonus = _location_priority(working["location"], config.preferred_locations)
    if not config.preferred_locations and combined_similarity.size:
        top_location = (
            working.assign(_sim=combined_similarity)
            .sort_values("_sim", ascending=False)
            .head(1)["location"]
            .iloc[0]
            if len(working)
            else ""
        )
        if isinstance(top_location, str) and top_location:
            location_bonus = (working["location"] == top_location).astype(float).to_numpy()

    final_fit = (
        config.weight_similarity * combined_similarity
        + config.weight_connections * connections_scaled
        + config.weight_location * location_bonus
    )

    working["fit_similarity"] = combined_similarity
    working["connection_numeric"] = connections_numeric
    working["connection_scaled"] = connections_scaled
    working["location_priority"] = location_bonus
    working["fit"] = final_fit

    columns = [
        "id",
        "job_title",
        "location",
        "connection",
        "connection_numeric",
        "fit_similarity",
        "connection_scaled",
        "location_priority",
        "fit",
    ]
    return working.sort_values("fit", ascending=False)[columns]


CandidateSource = Union[Path, str, pd.DataFrame]


def rank_candidates(data: CandidateSource, config: RankingConfig) -> pd.DataFrame:
    if isinstance(data, (str, Path)):
        df = load_candidates(data)
    elif isinstance(data, pd.DataFrame):
        df = _ensure_required_columns(data)
    else:
        raise TypeError("Data must be a path to a CSV file or a pandas DataFrame.")
    ranked = score_candidates(df, config)
    return ranked.reset_index(drop=True)
