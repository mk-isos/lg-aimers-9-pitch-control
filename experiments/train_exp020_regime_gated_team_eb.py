"""EXP-020: game_type R에만 적용하는 temporal team-EB correction.

고정 기준 예측과 team effect는 EXP-019의 OOF 산출물을 그대로 사용한다.
각 검증 행의 공식 ``game_type``이 ``R``일 때만 과거 OOF team correction을
적용하고, ``F`` 행은 고정 LGB/HGB 기준 예측을 유지한다. 검증 정답이나 검증
행 사이의 집계는 gate에 사용하지 않는다.

이 실험은 F 체제 이동을 확인한 뒤 정의한 진단 후보이므로, 행 단위 변환은
시간 안전하지만 후보 정의 자체는 완전한 nested selection이 아니다.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from train_exp017_rolling_residual import calculate_metrics


DATA_DIR = Path("./data")
SOURCE_DIR = Path("./artifacts/EXP-019/team_eb_ensemble")
ARTIFACT_DIR = Path("./artifacts/EXP-020/regime_gated_team_eb")
SOURCE_CANDIDATE = "all_prior_s1000"
VALIDATION_SEASONS = (2021, 2022, 2023, 2024)
REPORT_SEASONS = (2022, 2023, 2024)
PREDICTION_VARIANTS = (
    "fixed_base",
    "ungated_team_eb",
    "R_gated_team_eb",
)


def game_type_metrics(
    game_types: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        game_type: calculate_metrics(
            targets[game_types == game_type],
            predictions[game_types == game_type],
        )
        for game_type in sorted(np.unique(game_types))
    }


def main() -> None:
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        encoding="utf-8-sig",
        usecols=["season", "game_type", "control_success"],
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, object] = {}

    for season in VALIDATION_SEASONS:
        rows = train.loc[train["season"] == season].reset_index(drop=True)
        game_types = rows["game_type"].astype(str).to_numpy()
        current_targets = rows["control_success"].to_numpy(dtype=np.int8)
        base = np.load(
            SOURCE_DIR / f"base_ensemble_predictions_{season}.npy"
        ).astype(float)
        team = np.load(
            SOURCE_DIR / f"predictions_{SOURCE_CANDIDATE}_{season}.npy"
        ).astype(float)
        targets = np.load(SOURCE_DIR / f"targets_{season}.npy").astype(
            np.int8
        )
        if not (
            len(rows) == len(base) == len(team) == len(targets)
            and np.array_equal(current_targets, targets)
        ):
            raise ValueError(f"OOF alignment mismatch for {season}")
        unexpected_game_types = sorted(
            set(np.unique(game_types)) - {"F", "R"}
        )
        if unexpected_game_types:
            raise ValueError(
                f"unexpected game_type values for {season}: "
                f"{unexpected_game_types}"
            )
        if not (np.isfinite(base).all() and np.isfinite(team).all()):
            raise ValueError(f"non-finite source prediction for {season}")
        if not (
            np.logical_and(base >= 0.0, base <= 1.0).all()
            and np.logical_and(team >= 0.0, team <= 1.0).all()
        ):
            raise ValueError(f"source prediction outside [0, 1] for {season}")
        correction = team - base
        gate = game_types == "R"
        predictions = np.clip(base + gate * correction, 0.0, 1.0)
        if not np.allclose(
            predictions[~gate], base[~gate], rtol=0.0, atol=1e-15
        ):
            raise ValueError(f"F gate invariant failed for {season}")
        if not np.allclose(
            predictions[gate], team[gate], rtol=0.0, atol=1e-15
        ):
            raise ValueError(f"R gate invariant failed for {season}")
        variant_predictions = {
            "fixed_base": base,
            "ungated_team_eb": team,
            "R_gated_team_eb": predictions,
        }
        fold = {
            "validation_season": season,
            "source_candidate": SOURCE_CANDIDATE,
            "gate": "current-row game_type == R",
            "gate_rows": int(gate.sum()),
            "gate_rate": float(gate.mean()),
            "fixed_base": calculate_metrics(targets, base),
            "ungated_team_eb": calculate_metrics(targets, team),
            "R_gated_team_eb": calculate_metrics(targets, predictions),
            "game_type_segments": {
                name: game_type_metrics(game_types, targets, values)
                for name, values in variant_predictions.items()
            },
            "qa": {
                "alignment_checked": True,
                "finite_and_probability_range_checked": True,
                "F_rows_equal_fixed_base": True,
                "R_rows_equal_ungated_team_eb": True,
                "game_type_counts": {
                    game_type: int((game_types == game_type).sum())
                    for game_type in sorted(np.unique(game_types))
                },
            },
        }
        folds[str(season)] = fold
        np.save(
            ARTIFACT_DIR / f"predictions_R_gated_team_eb_{season}.npy",
            predictions,
        )
        np.save(ARTIFACT_DIR / f"targets_{season}.npy", targets)
        print(
            f"regime_gate {season}: "
            f"base={fold['fixed_base']['skill_score_unclipped']:.2f} "
            f"all={fold['ungated_team_eb']['skill_score_unclipped']:.2f} "
            f"R={fold['R_gated_team_eb']['skill_score_unclipped']:.2f}"
        )

    aggregate: dict[str, object] = {}
    for name in PREDICTION_VARIANTS:
        skills = {
            season: float(
                folds[str(season)][name]["skill_score_unclipped"]
            )
            for season in REPORT_SEASONS
        }
        briers = {
            season: float(folds[str(season)][name]["brier_score"])
            for season in REPORT_SEASONS
        }
        aggregate[name] = {
            "season_skills": {
                str(season): value for season, value in skills.items()
            },
            "season_briers": {
                str(season): value for season, value in briers.items()
            },
            "mean_skill": float(np.mean(list(skills.values()))),
            "min_skill": float(np.min(list(skills.values()))),
            "latest_2024_skill": skills[2024],
            "uniform_1100_gate_passed": bool(
                all(value >= 1100.0 for value in skills.values())
            ),
        }
    result = {
        "experiment": "EXP-020",
        "candidate": "all_prior_team_eb_R_only",
        "validation_protocol": {
            "outer_folds": list(VALIDATION_SEASONS),
            "reported_folds": list(REPORT_SEASONS),
            "base_and_team_effect": (
                "saved fixed OOF LGB/HGB 50/50 and all-prior team EB"
            ),
            "fixed_base": {
                "lightgbm": (
                    "r_full_residual/rfull_l63_m1000_i300/branch_w075"
                ),
                "histgradientboosting": (
                    "histgb_residual/hist_l15_d4_m3000_i160/branch_w100"
                ),
                "weights": [0.5, 0.5],
            },
            "gate": "official current-row game_type only",
            "current_fold_labels_used_for_gate": False,
            "validation_or_test_row_aggregation": False,
            "candidate_definition_nested": False,
            "candidate_definition_caveat": (
                "post-hoc diagnostic chosen after inspecting fold regimes; "
                "row transform is temporal-safe but candidate selection is "
                "non-nested"
            ),
        },
        "folds": folds,
        "aggregate_2022_2024": aggregate,
        "selection": {
            "status": (
                "post-hoc non-nested diagnostic; regime gate chosen after "
                "fold audit"
            ),
            "adopt_only_if_nested_confirmation_improves_robust_minimum": True,
        },
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    with (ARTIFACT_DIR / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
