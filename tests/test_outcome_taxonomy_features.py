from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from outcome_taxonomy_features import (  # noqa: E402
    assert_label_reconstruction_invariants,
    reconstruct_outcome_labels,
)


RATE_NAMES = ["success", "reverse", "middle", "ball", "strike"]


def make_row(
    row_id: str,
    pitcher_id: int,
    season: int,
    n: int,
    counts: tuple[int, int, int, int, int],
    target: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "row_id": row_id,
        "pitcher_id": pitcher_id,
        "season": season,
        "asof_pitcher_n": n,
        "control_success": target,
    }
    for name, count in zip(RATE_NAMES, counts, strict=True):
        row[f"asof_pitcher_{name}_rate"] = 0.0 if n == 0 else count / n
    return row


class OutcomeTaxonomyFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            [
                make_row("a0", 10, 2022, 0, (0, 0, 0, 0, 0), 1),
                make_row("a1", 10, 2022, 1, (1, 0, 0, 0, 1), 0),
                make_row("a2", 10, 2022, 2, (1, 1, 1, 1, 1), 0),
                make_row("a23_0", 10, 2023, 0, (0, 0, 0, 0, 0), 0),
                make_row("a23_1", 10, 2023, 1, (0, 0, 0, 1, 0), 0),
                make_row("b0", 20, 2022, 0, (0, 0, 0, 0, 0), 1),
                make_row("b1_x", 20, 2022, 1, (1, 0, 0, 0, 1), 0),
                make_row("b1_y", 20, 2022, 1, (1, 0, 0, 0, 1), 0),
            ]
        )

    @staticmethod
    def by_row_id(frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [frame[["row_id"]].reset_index(drop=True), labels.reset_index(drop=True)],
            axis=1,
        ).set_index("row_id").sort_index()

    def test_reconstructs_binary_deltas_and_excludes_ambiguous_pairs(self) -> None:
        labels, diagnostics = reconstruct_outcome_labels(self.frame)
        assert_label_reconstruction_invariants(self.frame, labels, diagnostics)
        result = self.by_row_id(self.frame, labels)

        self.assertTrue(bool(result.loc["a0", "pair_valid"]))
        self.assertEqual(int(result.loc["a0", "aux_success"]), 1)
        self.assertEqual(int(result.loc["a0", "aux_strike"]), 1)
        self.assertEqual(int(result.loc["a0", "aux_ball"]), 0)

        self.assertTrue(bool(result.loc["a1", "pair_valid"]))
        self.assertEqual(int(result.loc["a1", "aux_success"]), 0)
        self.assertEqual(int(result.loc["a1", "aux_reverse"]), 1)
        self.assertEqual(int(result.loc["a1", "aux_middle"]), 1)
        self.assertEqual(int(result.loc["a1", "aux_ball"]), 1)
        self.assertEqual(int(result.loc["a1", "aux_strike"]), 0)

        self.assertTrue(bool(result.loc["a23_0", "pair_valid"]))
        self.assertEqual(int(result.loc["a23_0", "aux_ball"]), 1)
        self.assertFalse(bool(result.loc["a2", "pair_valid"]))
        self.assertFalse(bool(result.loc["a23_1", "pair_valid"]))
        self.assertFalse(bool(result.loc["b0", "pair_valid"]))
        self.assertFalse(bool(result.loc["b1_x", "pair_valid"]))
        self.assertFalse(bool(result.loc["b1_y", "pair_valid"]))

        self.assertEqual(diagnostics["success_mismatch_count"], 0)
        self.assertEqual(diagnostics["duplicate_key_rows"], 2)
        self.assertEqual(diagnostics["valid_pair_rows"], 3)

    def test_reconstruction_is_independent_of_input_row_order(self) -> None:
        original, _ = reconstruct_outcome_labels(self.frame)
        shuffled_frame = self.frame.sample(frac=1.0, random_state=17).reset_index(drop=True)
        shuffled, diagnostics = reconstruct_outcome_labels(shuffled_frame)
        assert_label_reconstruction_invariants(shuffled_frame, shuffled, diagnostics)

        left = self.by_row_id(self.frame, original)
        right = self.by_row_id(shuffled_frame, shuffled)
        pd.testing.assert_frame_equal(left, right, check_dtype=True)

    def test_invalid_nonbinary_delta_is_excluded(self) -> None:
        frame = pd.DataFrame(
            [
                make_row("x0", 30, 2024, 0, (0, 0, 0, 0, 0), 0),
                make_row("x1", 30, 2024, 1, (0, 0, 0, 0, 0), 0),
            ]
        )
        frame.loc[1, "asof_pitcher_reverse_rate"] = 2.0
        labels, diagnostics = reconstruct_outcome_labels(frame)
        self.assertFalse(bool(labels.loc[0, "pair_valid"]))
        self.assertTrue(np.isnan(labels.loc[0, "aux_reverse"]))
        self.assertEqual(diagnostics["invalid_delta_rows"], 1)

    @unittest.skipUnless(
        (REPO_ROOT / "data" / "train.csv").exists(),
        "competition train.csv is not present",
    )
    def test_full_train_reconstruction_audit(self) -> None:
        usecols = [
            "row_id",
            "pitcher_id",
            "season",
            "asof_pitcher_n",
            "asof_pitcher_success_rate",
            "asof_pitcher_reverse_rate",
            "asof_pitcher_middle_rate",
            "asof_pitcher_ball_rate",
            "asof_pitcher_strike_rate",
            "control_success",
        ]
        frame = pd.read_csv(
            REPO_ROOT / "data" / "train.csv",
            encoding="utf-8-sig",
            usecols=usecols,
        )
        labels, diagnostics = reconstruct_outcome_labels(frame)
        assert_label_reconstruction_invariants(frame, labels, diagnostics)

        self.assertEqual(diagnostics["success_mismatch_count"], 0)
        self.assertGreater(diagnostics["valid_pair_rows"], 1_000_000)
        for season, summary in diagnostics["per_season"].items():
            with self.subTest(season=season):
                self.assertGreater(summary["valid_pair_rows"], 0)
                for name in RATE_NAMES:
                    self.assertGreater(summary["positive_counts"][name], 0)

        subset = frame.iloc[:20_000].copy()
        original, _ = reconstruct_outcome_labels(subset)
        shuffled_frame = subset.sample(frac=1.0, random_state=23).reset_index(drop=True)
        shuffled, shuffled_diagnostics = reconstruct_outcome_labels(shuffled_frame)
        assert_label_reconstruction_invariants(
            shuffled_frame,
            shuffled,
            shuffled_diagnostics,
        )
        left = self.by_row_id(subset, original)
        right = self.by_row_id(shuffled_frame, shuffled)
        pd.testing.assert_frame_equal(left, right, check_dtype=True)


if __name__ == "__main__":
    unittest.main()
