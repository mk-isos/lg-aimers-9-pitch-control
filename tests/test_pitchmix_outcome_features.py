from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from pitchmix_outcome_features import reconstruct_pitch_group  # noqa: E402


class PitchmixOutcomeFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            [
                {
                    "row_id": "p0",
                    "pitcher_id": 11,
                    "season": 2024,
                    "asof_pitcher_n": 0,
                    "asof_pitcher_pitchmix_n": 0,
                    "asof_pitcher_fastball_rate": np.nan,
                    "asof_pitcher_breaking_rate": np.nan,
                    "asof_pitcher_offspeed_rate": np.nan,
                },
                {
                    "row_id": "p1",
                    "pitcher_id": 11,
                    "season": 2024,
                    "asof_pitcher_n": 1,
                    "asof_pitcher_pitchmix_n": 1,
                    "asof_pitcher_fastball_rate": 1.0,
                    "asof_pitcher_breaking_rate": 0.0,
                    "asof_pitcher_offspeed_rate": 0.0,
                },
                {
                    "row_id": "p2",
                    "pitcher_id": 11,
                    "season": 2024,
                    "asof_pitcher_n": 2,
                    "asof_pitcher_pitchmix_n": 2,
                    "asof_pitcher_fastball_rate": 0.5,
                    "asof_pitcher_breaking_rate": 0.5,
                    "asof_pitcher_offspeed_rate": 0.0,
                },
                {
                    "row_id": "p3",
                    "pitcher_id": 11,
                    "season": 2024,
                    "asof_pitcher_n": 3,
                    "asof_pitcher_pitchmix_n": 3,
                    "asof_pitcher_fastball_rate": 1 / 3,
                    "asof_pitcher_breaking_rate": 1 / 3,
                    "asof_pitcher_offspeed_rate": 1 / 3,
                },
            ]
        )

    def test_recovers_three_onehot_pitch_groups(self) -> None:
        labels, audit = reconstruct_pitch_group(self.frame)
        self.assertEqual(labels.iloc[:3].astype(int).tolist(), [0, 1, 2])
        self.assertTrue(np.isnan(labels.iloc[3]))
        self.assertEqual(audit["valid_onehot_rows"], 3)
        self.assertEqual(audit["invalid_pair_rows"], 0)
        self.assertTrue(audit["all_valid_delta_pitchmix_n_equal_one"])
        self.assertTrue(audit["all_valid_group_deltas_onehot"])

    def test_row_permutation_does_not_change_labels(self) -> None:
        original, _ = reconstruct_pitch_group(self.frame)
        shuffled = self.frame.sample(frac=1.0, random_state=17)
        permuted, _ = reconstruct_pitch_group(shuffled)
        left = pd.DataFrame(
            {"row_id": self.frame["row_id"], "label": original}
        ).set_index("row_id").sort_index()
        right = pd.DataFrame(
            {"row_id": shuffled["row_id"], "label": permuted}
        ).set_index("row_id").sort_index()
        pd.testing.assert_frame_equal(left, right)


if __name__ == "__main__":
    unittest.main()
