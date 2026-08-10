"""EXP-020: low-rank pitcher-batter matchup EB atop team OOF.

For each earlier evaluated OOF season, a sparse pitcher x batter matrix is
built from source-season-centered residual sums.  Observed pair cells use
strong empirical-Bayes shrinkage ``sum / (count + smoothing)``.  Rank-4 and
rank-8 truncated SVD reconstructions then provide a matchup correction even
when an exact pair was absent, provided both IDs existed in that source
matrix.  If either ID is unseen in a source season, that source contributes
zero.  Source-season corrections are averaged equally.

The four candidates (smoothing 200/500 x rank 4/8) are predeclared.  The
immutable base is saved team ``all_prior_s1000`` OOF.  No current-fold label,
validation/test-row aggregation, or post-result parameter fitting is used.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.sparse.linalg import svds

from train_exp017_rolling_residual import calculate_metrics


DATA_DIR = Path("./data")
TEAM_ROOT = Path("./artifacts/EXP-019/team_eb_ensemble")
EXPLICIT_ROOT = Path("./artifacts/EXP-020/player_eb_atop_team")
ARTIFACT_DIR = Path("./artifacts/EXP-020/low_rank_matchup_eb")
TEAM_VARIANT = "all_prior_s1000"
EXPLICIT_VARIANT = "all_prior_pair_s2000"
VALIDATION_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
MAX_RANK = 8


@dataclass(frozen=True)
class Candidate:
    name: str
    smoothing: float
    rank: int


CANDIDATES = (
    Candidate("lr_s200_r4", 200.0, 4),
    Candidate("lr_s200_r8", 200.0, 8),
    Candidate("lr_s500_r4", 500.0, 4),
    Candidate("lr_s500_r8", 500.0, 8),
)


@dataclass
class SourceDecomposition:
    pitcher_ids: np.ndarray
    batter_ids: np.ndarray
    left: np.ndarray
    singular_values: np.ndarray
    right: np.ndarray
    observed_codes: np.ndarray
    matrix_nnz: int
    matrix_energy: float


def load_oof() -> tuple[
    dict[int, pd.DataFrame],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    frame = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=["season", "pitcher_id", "batter_id", "control_success"],
    )
    rows_by_season: dict[int, pd.DataFrame] = {}
    base_by_season: dict[int, np.ndarray] = {}
    targets_by_season: dict[int, np.ndarray] = {}
    residual_by_season: dict[int, np.ndarray] = {}
    for season in VALIDATION_SEASONS:
        rows = frame.loc[frame["season"] == season].reset_index(drop=True)
        base = np.load(
            TEAM_ROOT / f"predictions_{TEAM_VARIANT}_{season}.npy"
        ).astype(float)
        targets = np.load(TEAM_ROOT / f"targets_{season}.npy").astype(
            np.int8
        )
        explicit_targets = np.load(
            EXPLICIT_ROOT / f"targets_{season}.npy"
        ).astype(np.int8)
        current_targets = rows["control_success"].to_numpy(dtype=np.int8)
        if not (
            len(rows) == len(base) == len(targets)
            and np.array_equal(targets, current_targets)
            and np.array_equal(targets, explicit_targets)
            and np.isfinite(base).all()
            and (base >= 0.0).all()
            and (base <= 1.0).all()
        ):
            raise ValueError(f"OOF alignment or range mismatch for {season}")
        residual = targets.astype(float) - base
        residual -= residual.mean()
        rows_by_season[season] = rows
        base_by_season[season] = base
        targets_by_season[season] = targets
        residual_by_season[season] = residual
    return (
        rows_by_season,
        base_by_season,
        targets_by_season,
        residual_by_season,
    )


def decompose_source(
    rows: pd.DataFrame,
    centered_residual: np.ndarray,
    smoothing: float,
) -> SourceDecomposition:
    values = rows[["pitcher_id", "batter_id"]].copy()
    values["residual"] = centered_residual
    statistics = values.groupby(
        ["pitcher_id", "batter_id"], sort=True
    )["residual"].agg(["sum", "count"])
    pitcher_ids = np.sort(rows["pitcher_id"].unique())
    batter_ids = np.sort(rows["batter_id"].unique())
    pitcher_index = pd.Index(pitcher_ids)
    batter_index = pd.Index(batter_ids)
    statistic_index = statistics.index.to_frame(index=False)
    row_indices = pitcher_index.get_indexer(
        statistic_index["pitcher_id"]
    )
    column_indices = batter_index.get_indexer(
        statistic_index["batter_id"]
    )
    cell_effect = (
        statistics["sum"].to_numpy(dtype=float)
        / (statistics["count"].to_numpy(dtype=float) + smoothing)
    )
    matrix = sparse.coo_matrix(
        (cell_effect, (row_indices, column_indices)),
        shape=(len(pitcher_ids), len(batter_ids)),
        dtype=np.float64,
    ).tocsr()
    effective_rank = min(MAX_RANK, min(matrix.shape) - 1)
    if effective_rank < MAX_RANK:
        raise ValueError(
            f"source matrix {matrix.shape} cannot support rank {MAX_RANK}"
        )
    left, singular_values, right = svds(
        matrix,
        k=MAX_RANK,
        which="LM",
        solver="arpack",
        random_state=42,
    )
    order = np.argsort(singular_values)[::-1]
    singular_values = singular_values[order]
    left = left[:, order]
    right = right[order, :]
    observed_codes = np.sort(
        row_indices.astype(np.int64) * len(batter_ids)
        + column_indices.astype(np.int64)
    )
    return SourceDecomposition(
        pitcher_ids=pitcher_ids,
        batter_ids=batter_ids,
        left=left,
        singular_values=singular_values,
        right=right,
        observed_codes=observed_codes,
        matrix_nnz=int(matrix.nnz),
        matrix_energy=float(np.square(matrix.data).sum()),
    )


def map_decomposition(
    decomposition: SourceDecomposition,
    validation_rows: pd.DataFrame,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pitcher_indices = pd.Index(decomposition.pitcher_ids).get_indexer(
        validation_rows["pitcher_id"]
    )
    batter_indices = pd.Index(decomposition.batter_ids).get_indexer(
        validation_rows["batter_id"]
    )
    shared = (pitcher_indices >= 0) & (batter_indices >= 0)
    prediction = np.zeros(len(validation_rows), dtype=float)
    if shared.any():
        shared_pitchers = pitcher_indices[shared]
        shared_batters = batter_indices[shared]
        prediction[shared] = np.sum(
            decomposition.left[shared_pitchers, :rank]
            * decomposition.singular_values[:rank]
            * decomposition.right[:rank, shared_batters].T,
            axis=1,
        )
    exact = np.zeros(len(validation_rows), dtype=bool)
    if shared.any():
        codes = (
            pitcher_indices[shared].astype(np.int64)
            * len(decomposition.batter_ids)
            + batter_indices[shared].astype(np.int64)
        )
        exact[shared] = np.isin(
            codes,
            decomposition.observed_codes,
            assume_unique=False,
        )
    return prediction, exact, shared


def coverage_segments(
    targets: np.ndarray,
    predictions: np.ndarray,
    exact_any: np.ndarray,
    shared_any: np.ndarray,
) -> dict[str, dict[str, float]]:
    masks = {
        "exact_pair_seen_any_source": exact_any,
        "exact_pair_unseen": ~exact_any,
        "both_ids_shared_any_source": shared_any,
        "latent_only_exact_unseen": shared_any & ~exact_any,
        "one_or_both_ids_unseen_all_sources": ~shared_any,
    }
    return {
        name: calculate_metrics(targets[mask], predictions[mask])
        for name, mask in masks.items()
        if mask.any()
    }


def aggregate_metrics(
    folds: dict[str, object],
    candidate: str,
) -> dict[str, object]:
    skills = {
        season: float(
            folds[str(season)]["candidates"][candidate]["metrics"][
                "skill_score_unclipped"
            ]
        )
        for season in REPORT_SEASONS
    }
    briers = {
        season: float(
            folds[str(season)]["candidates"][candidate]["metrics"][
                "brier_score"
            ]
        )
        for season in REPORT_SEASONS
    }
    return {
        "season_skills": {
            str(season): value for season, value in skills.items()
        },
        "season_briers": {
            str(season): value for season, value in briers.items()
        },
        "mean_skill": float(np.mean(list(skills.values()))),
        "min_skill": float(np.min(list(skills.values()))),
        "latest_2024_skill": skills[2024],
    }


def main() -> None:
    started = time.time()
    (
        rows_by_season,
        base_by_season,
        targets_by_season,
        residual_by_season,
    ) = load_oof()
    smoothing_values = sorted({candidate.smoothing for candidate in CANDIDATES})
    decomposition_cache: dict[tuple[int, float], SourceDecomposition] = {}

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}
    for validation_season in VALIDATION_SEASONS:
        validation_rows = rows_by_season[validation_season]
        targets = targets_by_season[validation_season]
        base = base_by_season[validation_season]
        explicit_predictions = np.load(
            EXPLICIT_ROOT
            / f"predictions_{EXPLICIT_VARIANT}_{validation_season}.npy"
        ).astype(float)
        if not (
            len(explicit_predictions) == len(targets)
            and np.isfinite(explicit_predictions).all()
            and (explicit_predictions >= 0.0).all()
            and (explicit_predictions <= 1.0).all()
        ):
            raise ValueError(
                f"explicit pair reference invalid for {validation_season}"
            )
        source_seasons = [
            season
            for season in VALIDATION_SEASONS
            if season < validation_season
        ]
        fold: dict[str, object] = {
            "validation_season": validation_season,
            "source_oof_seasons": source_seasons,
            "base_team_all_prior": calculate_metrics(targets, base),
            "explicit_pair_s2000": calculate_metrics(
                targets, explicit_predictions
            ),
            "candidates": {},
        }
        np.save(ARTIFACT_DIR / f"targets_{validation_season}.npy", targets)
        for candidate in CANDIDATES:
            source_corrections: list[np.ndarray] = []
            source_exact: list[np.ndarray] = []
            source_shared: list[np.ndarray] = []
            source_details: dict[str, object] = {}
            for source_season in source_seasons:
                cache_key = (source_season, candidate.smoothing)
                if cache_key not in decomposition_cache:
                    decomposition_cache[cache_key] = decompose_source(
                        rows_by_season[source_season],
                        residual_by_season[source_season],
                        candidate.smoothing,
                    )
                decomposition = decomposition_cache[cache_key]
                correction, exact, shared = map_decomposition(
                    decomposition,
                    validation_rows,
                    candidate.rank,
                )
                source_corrections.append(correction)
                source_exact.append(exact)
                source_shared.append(shared)
                captured_energy = float(
                    np.square(
                        decomposition.singular_values[: candidate.rank]
                    ).sum()
                )
                source_details[str(source_season)] = {
                    "matrix_shape": [
                        int(len(decomposition.pitcher_ids)),
                        int(len(decomposition.batter_ids)),
                    ],
                    "matrix_nnz_exact_pairs": decomposition.matrix_nnz,
                    "singular_values_top8": decomposition.singular_values.tolist(),
                    "rank": candidate.rank,
                    "captured_sparse_matrix_energy_fraction": (
                        captured_energy / decomposition.matrix_energy
                        if decomposition.matrix_energy > 0.0
                        else 0.0
                    ),
                    "validation_exact_rows": int(exact.sum()),
                    "validation_shared_id_rows": int(shared.sum()),
                }
            if source_seasons:
                correction = np.mean(source_corrections, axis=0)
                exact_count = np.sum(source_exact, axis=0).astype(np.int8)
                shared_count = np.sum(source_shared, axis=0).astype(np.int8)
            else:
                correction = np.zeros(len(targets), dtype=float)
                exact_count = np.zeros(len(targets), dtype=np.int8)
                shared_count = np.zeros(len(targets), dtype=np.int8)
            exact_any = exact_count > 0
            shared_any = shared_count > 0
            predictions = np.clip(base + correction, 0.0, 1.0)
            candidate_fold = {
                "source_details": source_details,
                "current_fold_labels_used_for_effect": False,
                "metrics": calculate_metrics(targets, predictions),
                "coverage": {
                    "exact_pair_seen_any_rows": int(exact_any.sum()),
                    "exact_pair_seen_any_rate": float(exact_any.mean()),
                    "both_ids_shared_any_rows": int(shared_any.sum()),
                    "both_ids_shared_any_rate": float(shared_any.mean()),
                    "latent_only_exact_unseen_rows": int(
                        (shared_any & ~exact_any).sum()
                    ),
                    "latent_only_exact_unseen_rate": float(
                        (shared_any & ~exact_any).mean()
                    ),
                    "mean_exact_source_count": float(exact_count.mean()),
                    "mean_shared_source_count": float(shared_count.mean()),
                },
                "correction": {
                    "mean": float(correction.mean()),
                    "std": float(correction.std()),
                    "mean_absolute": float(np.abs(correction).mean()),
                    "min": float(correction.min()),
                    "max": float(correction.max()),
                },
                "coverage_segments": coverage_segments(
                    targets,
                    predictions,
                    exact_any,
                    shared_any,
                ),
                "base_coverage_segments_same_masks": coverage_segments(
                    targets,
                    base,
                    exact_any,
                    shared_any,
                ),
            }
            fold["candidates"][candidate.name] = candidate_fold
            np.save(
                ARTIFACT_DIR
                / f"predictions_{candidate.name}_{validation_season}.npy",
                predictions,
            )
            np.save(
                ARTIFACT_DIR
                / f"correction_{candidate.name}_{validation_season}.npy",
                correction,
            )
            np.save(
                ARTIFACT_DIR
                / f"exact_source_count_{candidate.name}_{validation_season}.npy",
                exact_count,
            )
            np.save(
                ARTIFACT_DIR
                / f"shared_source_count_{candidate.name}_{validation_season}.npy",
                shared_count,
            )
        folds[str(validation_season)] = fold
        print(
            f"low_rank_matchup {validation_season}: "
            + " ".join(
                f"{candidate.name}="
                f"{fold['candidates'][candidate.name]['metrics']['skill_score_unclipped']:.2f}"
                for candidate in CANDIDATES
            ),
            flush=True,
        )

    aggregate = {
        "base_team_all_prior": {
            "season_skills": {
                str(season): float(
                    folds[str(season)]["base_team_all_prior"][
                        "skill_score_unclipped"
                    ]
                )
                for season in REPORT_SEASONS
            },
            "mean_skill": float(
                np.mean(
                    [
                        folds[str(season)]["base_team_all_prior"][
                            "skill_score_unclipped"
                        ]
                        for season in REPORT_SEASONS
                    ]
                )
            ),
            "min_skill": float(
                np.min(
                    [
                        folds[str(season)]["base_team_all_prior"][
                            "skill_score_unclipped"
                        ]
                        for season in REPORT_SEASONS
                    ]
                )
            ),
        },
        "explicit_pair_s2000": {
            "season_skills": {
                str(season): float(
                    folds[str(season)]["explicit_pair_s2000"][
                        "skill_score_unclipped"
                    ]
                )
                for season in REPORT_SEASONS
            },
            "mean_skill": float(
                np.mean(
                    [
                        folds[str(season)]["explicit_pair_s2000"][
                            "skill_score_unclipped"
                        ]
                        for season in REPORT_SEASONS
                    ]
                )
            ),
            "min_skill": float(
                np.min(
                    [
                        folds[str(season)]["explicit_pair_s2000"][
                            "skill_score_unclipped"
                        ]
                        for season in REPORT_SEASONS
                    ]
                )
            ),
        },
    }
    for candidate in CANDIDATES:
        candidate_aggregate = aggregate_metrics(folds, candidate.name)
        candidate_aggregate["season_skill_change_vs_base"] = {
            str(season): float(
                candidate_aggregate["season_skills"][str(season)]
                - aggregate["base_team_all_prior"]["season_skills"][
                    str(season)
                ]
            )
            for season in REPORT_SEASONS
        }
        candidate_aggregate[
            "season_skill_change_vs_explicit_pair_s2000"
        ] = {
            str(season): float(
                candidate_aggregate["season_skills"][str(season)]
                - aggregate["explicit_pair_s2000"]["season_skills"][
                    str(season)
                ]
            )
            for season in REPORT_SEASONS
        }
        candidate_aggregate["improved_every_season_vs_base"] = bool(
            all(
                value > 0.0
                for value in candidate_aggregate[
                    "season_skill_change_vs_base"
                ].values()
            )
        )
        aggregate[candidate.name] = candidate_aggregate

    posthoc_best = max(
        (candidate.name for candidate in CANDIDATES),
        key=lambda name: (
            float(aggregate[name]["min_skill"]),
            float(aggregate[name]["latest_2024_skill"]),
            float(aggregate[name]["mean_skill"]),
        ),
    )
    result: dict[str, object] = {
        "experiment": "EXP-020",
        "candidate_family": "low_rank_pitcher_batter_matchup_eb",
        "validation_protocol": {
            "outer_folds": list(VALIDATION_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "immutable_base": "team all_prior_s1000 OOF",
            "source_target": "source-season-centered y minus team OOF",
            "source_season_weighting": "equal; unseen ID contributes zero",
            "exact_pair_unseen_policy": (
                "use low-rank prediction when both IDs are in source matrix"
            ),
            "current_fold_labels_used_for_effect": False,
            "test_row_aggregation": False,
            "candidate_grid_predeclared": True,
            "candidate_selection": "post-hoc diagnostic ranking only",
        },
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "status": "post-hoc diagnostic; not a nested selection",
            "posthoc_best_candidate": posthoc_best,
            "posthoc_best_min_skill": aggregate[posthoc_best]["min_skill"],
            "posthoc_best_latest_2024_skill": aggregate[posthoc_best][
                "latest_2024_skill"
            ],
            "any_candidate_improved_every_season_vs_base": bool(
                any(
                    aggregate[candidate.name][
                        "improved_every_season_vs_base"
                    ]
                    for candidate in CANDIDATES
                )
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "total_seconds": time.time() - started,
    }
    with (ARTIFACT_DIR / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(f"selection={result['selection']}", flush=True)
    print(f"saved={ARTIFACT_DIR / 'validation_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
