"""Row-independent auxiliary outcome labels from cumulative pitcher rates.

The current pitch outcome can be recovered on train rows when a unique
same-pitcher, same-season state with ``asof_pitcher_n + 1`` exists.  The
implementation uses keyed lookup rather than file order, so shuffling rows
cannot change labels.  Ambiguous keys and non-binary cumulative deltas are
excluded instead of guessed.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


KEY_COLUMNS: Final[tuple[str, ...]] = (
    "pitcher_id",
    "season",
    "asof_pitcher_n",
)
OUTCOME_NAMES: Final[tuple[str, ...]] = (
    "success",
    "reverse",
    "middle",
    "ball",
    "strike",
)
RATE_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"asof_pitcher_{name}_rate" for name in OUTCOME_NAMES
)
LABEL_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"aux_{name}" for name in OUTCOME_NAMES
)
JOINT_CLASS_NAMES: Final[tuple[str, ...]] = (
    "success",
    "reverse_only",
    "middle_only",
    "reverse_middle",
    "other_failure",
)
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "row_id",
    *KEY_COLUMNS,
    *RATE_COLUMNS,
    "control_success",
)


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required outcome-taxonomy columns: {missing}")


def _multi_index(frame: pd.DataFrame, n_offset: int = 0) -> pd.MultiIndex:
    n = pd.to_numeric(frame["asof_pitcher_n"], errors="coerce") + n_offset
    return pd.MultiIndex.from_arrays(
        [
            frame["pitcher_id"].to_numpy(),
            frame["season"].to_numpy(),
            n.to_numpy(),
        ],
        names=list(KEY_COLUMNS),
    )


def reconstruct_outcome_labels(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Recover binary outcomes using unique cumulative-state transitions."""
    _require_columns(frame)
    row_count = len(frame)
    key_index = _multi_index(frame)
    duplicate_key = key_index.duplicated(keep=False)
    numeric_n = pd.to_numeric(
        frame["asof_pitcher_n"], errors="coerce"
    ).to_numpy(dtype=float)
    key_valid = np.isfinite(numeric_n) & (numeric_n >= 0) & ~duplicate_key

    unique_positions = np.flatnonzero(key_valid)
    lookup = pd.Series(
        unique_positions,
        index=key_index[key_valid],
        dtype="int64",
    )
    successor_index = _multi_index(frame, n_offset=1)
    successor_position = lookup.reindex(successor_index).to_numpy(dtype=float)
    has_successor = np.isfinite(successor_position)
    pair_candidate = key_valid & has_successor

    rates = frame.loc[:, RATE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    counts = np.rint(numeric_n[:, None] * rates)
    deltas = np.full((row_count, len(OUTCOME_NAMES)), np.nan, dtype=float)
    candidate_positions = np.flatnonzero(pair_candidate)
    if len(candidate_positions):
        next_positions = successor_position[candidate_positions].astype(np.int64)
        deltas[candidate_positions] = (
            counts[next_positions] - counts[candidate_positions]
        )
    delta_binary = np.isfinite(deltas) & ((deltas == 0.0) | (deltas == 1.0))
    pair_valid = pair_candidate & delta_binary.all(axis=1)

    labels = pd.DataFrame(
        np.where(pair_valid[:, None], deltas, np.nan),
        index=frame.index,
        columns=LABEL_COLUMNS,
        dtype=float,
    )
    labels["pair_valid"] = pair_valid

    target = pd.to_numeric(
        frame["control_success"], errors="coerce"
    ).to_numpy(dtype=float)
    valid_success = pair_valid & np.isfinite(target)
    success_mismatch = int(
        np.sum(deltas[valid_success, 0] != target[valid_success])
    )

    per_season: dict[str, object] = {}
    season_values = frame["season"].to_numpy()
    for season in sorted(pd.unique(frame["season"]).tolist()):
        season_mask = season_values == season
        valid = season_mask & pair_valid
        positives = {
            name: int(np.nansum(labels.loc[valid, f"aux_{name}"].to_numpy()))
            for name in OUTCOME_NAMES
        }
        per_season[str(int(season))] = {
            "rows": int(season_mask.sum()),
            "valid_pair_rows": int(valid.sum()),
            "valid_pair_rate": float(valid.sum() / max(1, season_mask.sum())),
            "positive_counts": positives,
            "positive_rates": {
                name: (
                    float(positives[name] / valid.sum()) if valid.sum() else None
                )
                for name in OUTCOME_NAMES
            },
        }

    diagnostics: dict[str, object] = {
        "rows": row_count,
        "unique_key_rows": int(key_valid.sum()),
        "duplicate_key_rows": int(duplicate_key.sum()),
        "candidate_pair_rows": int(pair_candidate.sum()),
        "valid_pair_rows": int(pair_valid.sum()),
        "missing_successor_rows": int((key_valid & ~has_successor).sum()),
        "invalid_delta_rows": int((pair_candidate & ~pair_valid).sum()),
        "success_mismatch_count": success_mismatch,
        "row_order_dependency": False,
        "pair_lookup": "unique (pitcher_id, season, asof_pitcher_n + 1)",
        "per_season": per_season,
    }
    return labels, diagnostics


def assert_label_reconstruction_invariants(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    diagnostics: dict[str, object],
) -> None:
    """Raise when reconstructed labels violate leakage or binary invariants."""
    if not labels.index.equals(frame.index):
        raise AssertionError("label index does not preserve source frame index")
    expected = [*LABEL_COLUMNS, "pair_valid"]
    if list(labels.columns) != expected:
        raise AssertionError(
            f"unexpected label columns: {list(labels.columns)} != {expected}"
        )
    valid = labels["pair_valid"].to_numpy(dtype=bool)
    values = labels.loc[:, LABEL_COLUMNS].to_numpy(dtype=float)
    if not np.isnan(values[~valid]).all():
        raise AssertionError("invalid pairs must have NaN auxiliary labels")
    if not np.isin(values[valid], [0.0, 1.0]).all():
        raise AssertionError("valid auxiliary labels must be binary")
    target = pd.to_numeric(
        frame["control_success"], errors="coerce"
    ).to_numpy(dtype=float)
    success = labels["aux_success"].to_numpy(dtype=float)
    comparable = valid & np.isfinite(target)
    mismatch = int(np.sum(success[comparable] != target[comparable]))
    if mismatch != int(diagnostics["success_mismatch_count"]):
        raise AssertionError("success mismatch diagnostic does not match labels")
    if mismatch:
        raise AssertionError(
            f"reconstructed success differs from control_success on {mismatch} rows"
        )


def derive_joint_taxonomy(
    labels: pd.DataFrame,
) -> tuple[pd.Series, dict[str, object]]:
    """Convert success/reverse/middle multi-labels into five disjoint classes."""
    required = {"aux_success", "aux_reverse", "aux_middle", "pair_valid"}
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError(f"missing joint-taxonomy label columns: {missing}")
    valid = labels["pair_valid"].to_numpy(dtype=bool)
    success = labels["aux_success"].to_numpy(dtype=float)
    reverse = labels["aux_reverse"].to_numpy(dtype=float)
    middle = labels["aux_middle"].to_numpy(dtype=float)
    joint = np.full(len(labels), np.nan, dtype=float)
    class_masks = (
        valid & (success == 1.0) & (reverse == 0.0) & (middle == 0.0),
        valid & (success == 0.0) & (reverse == 1.0) & (middle == 0.0),
        valid & (success == 0.0) & (reverse == 0.0) & (middle == 1.0),
        valid & (success == 0.0) & (reverse == 1.0) & (middle == 1.0),
        valid & (success == 0.0) & (reverse == 0.0) & (middle == 0.0),
    )
    for class_index, mask in enumerate(class_masks):
        joint[mask] = class_index
    assigned = np.isfinite(joint)
    invalid_overlap = valid & ~assigned
    diagnostics = {
        "class_names": list(JOINT_CLASS_NAMES),
        "valid_pair_rows": int(valid.sum()),
        "assigned_rows": int(assigned.sum()),
        "invalid_overlap_rows": int(invalid_overlap.sum()),
        "class_counts": {
            name: int(mask.sum())
            for name, mask in zip(JOINT_CLASS_NAMES, class_masks, strict=True)
        },
    }
    return (
        pd.Series(joint, index=labels.index, name="joint_outcome_class"),
        diagnostics,
    )
