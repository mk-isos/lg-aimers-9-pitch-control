"""시점 누수 없이 Trackman 투수 요약 피처를 만드는 도구.

메인 데이터의 익명 팀/투수 ID와 Trackman ID는 직접 연결되지 않는다. 공식
데이터 안의 시즌별 소속 팀, 좌우 유형, 투구량, 구종 비율을 이용해 일대일
매핑을 만들고 비용이 낮은 매핑만 채택한다. 실제 피처 값은 각 예측 시즌보다
이전 시즌의 Trackman 로그만 집계한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


TEAM_ID_TO_TRACKMAN = {
    12: "DOO_BEA",
    13: "LG_TWI",
    14: "KIW_HER",
    15: "LOT_GIA",
    16: "KIA_TIG",
    17: "HAN_EAG",
    18: "SAM_LIO",
    19: "NC_DIN",
    20: "KT_WIZ",
    21: "SSG",
}
CORE_TEAMS = set(TEAM_ID_TO_TRACKMAN.values())
TRACKMAN_TEAM_ALIASES = {
    "SK_WYV": "SSG",
    "SSG_LAN": "SSG",
}
TRACKMAN_NUMERIC_COLUMNS = [
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
]
PITCH_GROUPS = ["fastball", "breaking", "offspeed"]


@dataclass(frozen=True)
class MappingResult:
    mapping: dict[int, int]
    costs: dict[int, float]
    candidate_main_ids: int
    candidate_trackman_ids: int


def normalize_trackman_teams(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["normalized_pitcher_team"] = out["pitcher_team"].replace(
        TRACKMAN_TEAM_ALIASES
    )
    return out


def build_pitcher_mapping(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    cutoff_season: int,
    max_cost: float = 0.1,
) -> MappingResult:
    """공식 데이터 내부의 이력만으로 신뢰도 높은 투수 ID 매핑을 만든다."""
    seasons = list(range(2019, cutoff_season + 1))
    main_work = main.loc[
        (main["season"] <= cutoff_season)
        & main["pitcher_team_id"].isin(TEAM_ID_TO_TRACKMAN),
        [
            "season",
            "pitcher_id",
            "pitcher_hand",
            "pitcher_team_id",
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ],
    ].copy()
    main_work["normalized_pitcher_team"] = main_work[
        "pitcher_team_id"
    ].map(TEAM_ID_TO_TRACKMAN)
    main_work["normalized_pitcher_hand"] = main_work["pitcher_hand"].map(
        {1: "Left", 2: "Right"}
    )

    trackman_work = normalize_trackman_teams(trackman)
    trackman_work = trackman_work.loc[
        (trackman_work["season"] <= cutoff_season)
        & trackman_work["normalized_pitcher_team"].isin(CORE_TEAMS)
    ].copy()

    main_ids = np.sort(main_work["pitcher_id"].unique())
    trackman_ids = np.sort(trackman_work["pitcher_trackman_id"].unique())
    if len(main_ids) == 0 or len(trackman_ids) == 0:
        return MappingResult({}, {}, len(main_ids), len(trackman_ids))

    main_counts = (
        main_work.groupby(["pitcher_id", "season"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=main_ids, columns=seasons, fill_value=0)
        .to_numpy(dtype=float)
    )
    trackman_counts = (
        trackman_work.groupby(["pitcher_trackman_id", "season"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=trackman_ids, columns=seasons, fill_value=0)
        .to_numpy(dtype=float)
    )
    season_scale = np.array(
        [
            (main_work["season"] == season).sum()
            / max(1, (trackman_work["season"] == season).sum())
            for season in seasons
        ],
        dtype=float,
    )
    scaled_trackman_counts = trackman_counts * season_scale

    main_teams = (
        main_work.groupby(["pitcher_id", "season"])[
            "normalized_pitcher_team"
        ]
        .agg(lambda values: values.mode().iloc[0])
        .unstack()
        .reindex(index=main_ids, columns=seasons)
        .fillna("")
        .to_numpy()
    )
    trackman_teams = (
        trackman_work.groupby(["pitcher_trackman_id", "season"])[
            "normalized_pitcher_team"
        ]
        .agg(lambda values: values.mode().iloc[0])
        .unstack()
        .reindex(index=trackman_ids, columns=seasons)
        .fillna("")
        .to_numpy()
    )
    main_hands = (
        main_work.groupby("pitcher_id")["normalized_pitcher_hand"]
        .agg(lambda values: values.mode().iloc[0])
        .reindex(main_ids)
        .to_numpy()
    )
    trackman_hands = (
        trackman_work.groupby("pitcher_trackman_id")["pitcher_hand"]
        .agg(lambda values: values.mode().iloc[0])
        .reindex(trackman_ids)
        .to_numpy()
    )

    main_mix = (
        main_work.groupby("pitcher_id", sort=False)
        .tail(1)
        .set_index("pitcher_id")
        .reindex(main_ids)[
            [
                "asof_pitcher_fastball_rate",
                "asof_pitcher_breaking_rate",
                "asof_pitcher_offspeed_rate",
            ]
        ]
        .to_numpy(dtype=float)
    )
    trackman_mix_frame = pd.crosstab(
        trackman_work["pitcher_trackman_id"],
        trackman_work["pitch_type_group"],
    ).reindex(index=trackman_ids, columns=PITCH_GROUPS, fill_value=0)
    trackman_mix = trackman_mix_frame.to_numpy(dtype=float)
    trackman_mix = trackman_mix / np.maximum(
        trackman_mix.sum(axis=1, keepdims=True),
        1.0,
    )

    cost_matrix = np.empty((len(main_ids), len(trackman_ids)), dtype=float)
    for index in range(len(main_ids)):
        active_seasons = main_counts[index] > 0
        count_mask = (scaled_trackman_counts > 0) | active_seasons
        count_error = np.sum(
            (
                (
                    np.log1p(scaled_trackman_counts)
                    - np.log1p(main_counts[index])
                )
                / 2.0
            )
            ** 2
            * count_mask,
            axis=1,
        ) / np.maximum(count_mask.sum(axis=1), 1)
        team_mismatch = np.sum(
            (trackman_teams[:, active_seasons] != main_teams[index, active_seasons])
            & (trackman_teams[:, active_seasons] != ""),
            axis=1,
        )
        missing_team = np.sum(
            trackman_teams[:, active_seasons] == "",
            axis=1,
        )
        with np.errstate(invalid="ignore"):
            pitchmix_error = np.nanmean(
                (trackman_mix - main_mix[index]) ** 2,
                axis=1,
            )
        pitchmix_error = np.nan_to_num(pitchmix_error, nan=0.1)
        hand_mismatch = (trackman_hands != main_hands[index]).astype(float)
        cost_matrix[index] = (
            count_error
            + 2.0 * pitchmix_error
            + 2.0 * team_mismatch
            + 0.5 * missing_team
            + 20.0 * hand_mismatch
        )

    main_indices, trackman_indices = linear_sum_assignment(cost_matrix)
    mapping: dict[int, int] = {}
    costs: dict[int, float] = {}
    for main_index, trackman_index in zip(main_indices, trackman_indices):
        cost = float(cost_matrix[main_index, trackman_index])
        if cost <= max_cost:
            main_id = int(main_ids[main_index])
            mapping[main_id] = int(trackman_ids[trackman_index])
            costs[main_id] = cost
    return MappingResult(
        mapping=mapping,
        costs=costs,
        candidate_main_ids=len(main_ids),
        candidate_trackman_ids=len(trackman_ids),
    )


def _aggregate_trackman_history(
    history: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    grouped = history.groupby("pitcher_trackman_id")
    pieces = [grouped.size().rename(f"{prefix}_n")]
    for column in TRACKMAN_NUMERIC_COLUMNS:
        pieces.append(grouped[column].mean().rename(f"{prefix}_{column}_mean"))
        pieces.append(grouped[column].std().rename(f"{prefix}_{column}_std"))
    pitch_counts = pd.crosstab(
        history["pitcher_trackman_id"],
        history["pitch_type_group"],
    ).reindex(columns=PITCH_GROUPS, fill_value=0)
    pitch_rates = pitch_counts.div(
        pitch_counts.sum(axis=1).replace(0, np.nan),
        axis=0,
    )
    for group_name in PITCH_GROUPS:
        pieces.append(
            pitch_rates[group_name].rename(f"{prefix}_{group_name}_rate")
        )
    result = pd.concat(pieces, axis=1).reset_index()
    return result


def build_prior_season_trackman_features(
    trackman: pd.DataFrame,
    prediction_seasons: list[int],
) -> pd.DataFrame:
    """각 예측 시즌 직전까지의 전체·최근 1시즌 Trackman 요약을 만든다."""
    outputs: list[pd.DataFrame] = []
    for prediction_season in prediction_seasons:
        historical = trackman.loc[trackman["season"] < prediction_season]
        if historical.empty:
            continue
        all_history = _aggregate_trackman_history(historical, "tm_hist")
        last_season = historical.loc[
            historical["season"] == prediction_season - 1
        ]
        if last_season.empty:
            combined = all_history
        else:
            recent = _aggregate_trackman_history(last_season, "tm_last")
            combined = all_history.merge(
                recent,
                on="pitcher_trackman_id",
                how="left",
            )
        combined["season"] = prediction_season
        outputs.append(combined)
    if not outputs:
        return pd.DataFrame(columns=["pitcher_trackman_id", "season"])
    return pd.concat(outputs, ignore_index=True)


def attach_trackman_features(
    main: pd.DataFrame,
    trackman: pd.DataFrame,
    cutoff_season: int,
    max_mapping_cost: float = 0.1,
) -> tuple[pd.DataFrame, MappingResult, list[str]]:
    """메인 데이터에 시점 안전한 Trackman 투수 요약을 결합한다."""
    mapping_result = build_pitcher_mapping(
        main,
        trackman,
        cutoff_season=cutoff_season,
        max_cost=max_mapping_cost,
    )
    prediction_seasons = sorted(
        set(main["season"].astype(int).unique()) | {cutoff_season + 1}
    )
    trackman_features = build_prior_season_trackman_features(
        trackman,
        prediction_seasons,
    )
    output = main.copy()
    output["pitcher_trackman_id"] = output["pitcher_id"].map(
        mapping_result.mapping
    )
    output["trackman_mapping_cost"] = output["pitcher_id"].map(
        mapping_result.costs
    )
    output["has_trackman_mapping"] = output[
        "pitcher_trackman_id"
    ].notna().astype("int8")
    output = output.merge(
        trackman_features,
        on=["pitcher_trackman_id", "season"],
        how="left",
        sort=False,
    )
    trackman_columns = [
        column
        for column in output.columns
        if column.startswith("tm_")
        or column in {"trackman_mapping_cost", "has_trackman_mapping"}
    ]
    output = output.drop(columns=["pitcher_trackman_id"])
    return output, mapping_result, trackman_columns
