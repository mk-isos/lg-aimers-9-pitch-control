from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from train_exp105_plus_correction_composition import (  # noqa: E402
    COEFFICIENT_SUM_BOUND,
    constrained_fit,
    pearson,
    ridge_fit,
    season_equal_weights,
)


class CorrectionCompositionTests(unittest.TestCase):
    def test_ridge_fit_recovers_zero_intercept_coefficients(self) -> None:
        x = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        expected = np.asarray([0.4, -0.2])
        y = x @ expected
        weights = np.ones(len(y))
        actual = ridge_fit(x, y, weights, 1e-12)
        self.assertTrue(np.allclose(actual, expected, atol=1e-10, rtol=0.0))

    def test_constrained_fit_is_nonnegative_and_sum_bounded(self) -> None:
        x = np.eye(4)
        y = np.asarray([3.0, 2.0, -1.0, 1.0])
        coef = constrained_fit(x, y, np.ones(4), 0.0)
        self.assertGreaterEqual(float(np.min(coef)), 0.0)
        self.assertLessEqual(float(coef.sum()), COEFFICIENT_SUM_BOUND + 1e-9)

    def test_season_equal_weights_assign_equal_total_mass(self) -> None:
        season_ids = np.asarray([2021, 2021, 2022, 2022, 2022])
        weights = season_equal_weights(season_ids)
        self.assertTrue(np.isclose(weights[season_ids == 2021].sum(), 2.5))
        self.assertTrue(np.isclose(weights[season_ids == 2022].sum(), 2.5))

    def test_pearson_handles_constant_input_without_nan_json(self) -> None:
        self.assertIsNone(pearson(np.ones(4), np.arange(4.0)))


if __name__ == "__main__":
    unittest.main()
