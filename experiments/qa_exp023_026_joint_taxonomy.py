"""Independent artifact QA for EXP-023 through EXP-026."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


BASE_ROOT = Path("./artifacts/EXP-020/low_rank_pitcher_context_eb")
ROOTS = {
    "EXP-023": Path("./artifacts/EXP-023/joint_taxonomy_multiclass"),
    "EXP-024": Path("./artifacts/EXP-024/source_bagged_joint_taxonomy"),
    "EXP-025": Path("./artifacts/EXP-025/rowlocal_regime_gate"),
    "EXP-026": Path("./artifacts/EXP-026/joint_expert_trend"),
}
REPORT_SEASONS = (2022, 2023, 2024)


def brier(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean((targets.astype(float) - predictions.astype(float)) ** 2))


def load_json(root: Path) -> dict[str, object]:
    with (root / "validation_metrics.json").open(encoding="utf-8") as file:
        return json.load(
            file,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON: {value}")
            ),
        )


def main() -> None:
    checked_arrays = 0
    for experiment, root in ROOTS.items():
        metrics = load_json(root)
        assert metrics["experiment"] == experiment
        aggregate = metrics["aggregate_2022_2024"]
        assert aggregate["uniform_1100_passed"] is False
        assert aggregate["final_fit_authorized"] is False
        assert aggregate["zip_creation_authorized"] is False
        assert metrics["qa"]["final_fit_or_zip_created"] is False

        for path in sorted(root.glob("*.npy")):
            array = np.load(path)
            assert np.isfinite(array).all(), path
            if path.name.startswith("predictions_"):
                assert ((array >= 0.0) & (array <= 1.0)).all(), path
            checked_arrays += 1

        for season in REPORT_SEASONS:
            fold = metrics["folds"][str(season)]
            targets = np.load(BASE_ROOT / f"targets_{season}.npy").astype(float)
            selected = np.load(
                root / f"predictions_strict_selected_{season}.npy"
            ).astype(float)
            assert len(selected) == len(targets)
            assert abs(brier(targets, selected) - fold["selected"]["brier_score"]) < 1e-14
            base = np.load(
                BASE_ROOT / f"predictions_lowrank_s300_r6_{season}.npy"
            ).astype(float)
            base_key = "base_metrics" if experiment == "EXP-023" else "base"
            assert abs(brier(targets, base) - fold[base_key]["brier_score"]) < 1e-14

            if experiment == "EXP-023":
                weight = int(round(float(fold["selected"]["weight"]) * 100))
                candidate = np.load(
                    root / f"predictions_blend_w{weight:03d}_{season}.npy"
                )
            elif experiment in {"EXP-024", "EXP-026"}:
                name = fold["selected"]["candidate"]
                candidate = np.load(root / f"predictions_{name}_{season}.npy")
            else:
                weight = int(round(float(fold["selected"]["weight"]) * 100))
                candidate = np.load(
                    root / f"predictions_blend_w{weight:03d}_{season}.npy"
                )
            np.testing.assert_allclose(selected, candidate, rtol=0.0, atol=0.0)

    exp023 = load_json(ROOTS["EXP-023"])
    assert exp023["joint_taxonomy_audit"]["invalid_overlap_rows"] == 0
    assert exp023["aggregate_2022_2024"]["samefold_2023_2024_ceiling_passed"] is True
    exp024 = load_json(ROOTS["EXP-024"])
    assert exp024["qa"]["candidate_count"] == 10
    exp025 = load_json(ROOTS["EXP-025"])
    assert exp025["validation_protocol"]["gate_is_row_local"] is True
    exp026 = load_json(ROOTS["EXP-026"])
    assert exp026["aggregate_2022_2024"]["stop_outcome_taxonomy_branch"] is True
    print(
        f"EXP023_026_QA_OK experiments={len(ROOTS)} arrays={checked_arrays} "
        "uniform_1100=all_false"
    )


if __name__ == "__main__":
    main()
