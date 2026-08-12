"""Independent saved-array QA for EXP-022."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ARTIFACT_ROOT = Path("./artifacts/EXP-022/outcome_taxonomy_multitask")
BASE_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
REPORT_SEASONS = (2022, 2023, 2024)


def brier(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean((predictions.astype(float) - targets.astype(float)) ** 2))


def main() -> None:
    with (ARTIFACT_ROOT / "validation_metrics.json").open(
        encoding="utf-8"
    ) as file:
        metrics = json.load(
            file,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )

    for season in (2021, *REPORT_SEASONS):
        representation = np.load(
            ARTIFACT_ROOT / f"auxiliary_representation_{season}.npy"
        )
        targets = np.load(BASE_ROOT / f"targets_{season}.npy")
        assert representation.shape == (len(targets), 6)
        assert np.isfinite(representation).all()
        assert ((representation[:, :4] >= 0.0) & (representation[:, :4] <= 1.0)).all()
        np.testing.assert_allclose(
            representation[:, 4],
            representation[:, 3] - representation[:, 2],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            representation[:, 5],
            representation[:, 0] + representation[:, 1],
            rtol=0.0,
            atol=0.0,
        )

    for season in REPORT_SEASONS:
        fold = metrics["folds"][str(season)]
        targets = np.load(BASE_ROOT / f"targets_{season}.npy").astype(float)
        base = np.load(
            BASE_ROOT / f"predictions_lowrank_s300_r6_{season}.npy"
        ).astype(float)
        assert abs(brier(targets, base) - fold["base"]["brier_score"]) < 1e-14

        predictions: dict[int, np.ndarray] = {}
        for weight in (25, 50):
            candidate = np.load(
                ARTIFACT_ROOT
                / f"predictions_temporal_w{weight:03d}_{season}.npy"
            ).astype(float)
            assert len(candidate) == len(targets)
            assert np.isfinite(candidate).all()
            assert ((candidate >= 0.0) & (candidate <= 1.0)).all()
            stored = fold["candidates"][f"w{weight:03d}"]["brier_score"]
            assert abs(brier(targets, candidate) - stored) < 1e-14
            predictions[weight] = candidate
        np.testing.assert_allclose(
            predictions[50],
            np.clip(base + 2.0 * (predictions[25] - base), 0.0, 1.0),
            rtol=0.0,
            atol=2e-15,
        )
        selected_weight = int(round(float(fold["selected"]["scale"]) * 100))
        selected = np.load(
            ARTIFACT_ROOT / f"predictions_strict_selected_{season}.npy"
        ).astype(float)
        np.testing.assert_allclose(
            selected,
            predictions[selected_weight],
            rtol=0.0,
            atol=0.0,
        )
        assert abs(
            brier(targets, selected) - fold["selected"]["brier_score"]
        ) < 1e-14
        source_seasons = fold["selection"]["source_seasons"]
        assert source_seasons
        assert max(source_seasons) < season

    aggregate = metrics["aggregate_2022_2024"]
    assert aggregate["reported_seasons_complete"] is True
    assert aggregate["uniform_1100_passed"] is False
    assert aggregate["final_fit_authorized"] is False
    assert aggregate["zip_creation_authorized"] is False
    assert aggregate["stop_linear_family"] is True
    assert metrics["qa"]["final_fit_or_zip_created"] is False
    print(
        "EXP022_QA_OK "
        f"aux_arrays=4 candidate_arrays=6 selected_arrays=3 "
        f"uniform_1100={aggregate['uniform_1100_passed']}"
    )


if __name__ == "__main__":
    main()
