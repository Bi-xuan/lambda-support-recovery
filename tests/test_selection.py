import unittest
from math import sqrt
from unittest.mock import patch

import numpy as np

from lambda_support_recovery import (
    FitSettings,
    SupportCurve,
    select_from_curve,
    select_support,
)
from lambda_support_recovery.selection import select_plateau


class PlateauSelectionTests(unittest.TestCase):
    def test_selects_center_even_when_doubling_would_leave_plateau(self):
        # D=2 wins on [1, 2); doubling its center would instead select D=1.
        selection = select_plateau([1, 2, 3], [3, 1, 0], [1, 2, 3])

        self.assertEqual(selection.selected_dimension, 2)
        self.assertAlmostEqual(selection.selected_scale, sqrt(2))
        self.assertEqual(selection.selected_scale, selection.plateau_selection.center)

    def test_public_selection_paths(self):
        masks = np.array([
            [[True, False], [False, True]],
            [[True, True], [False, True]],
            [[True, True], [True, True]],
        ])
        curve = SupportCurve(
            sigma_hat=np.eye(2),
            dimensions=np.array([1, 2, 3]),
            raw_objectives=np.array([3.0, 1.0, 0.0]),
            support_masks=masks,
            support_valid=np.ones(3, dtype=bool),
            fitted_lambdas=np.zeros((3, 2, 2)),
            fitted_omegas=np.ones(3),
            fit_settings=FitSettings(),
            resolved_omega_ref=1.0,
            allowed_edges=((0, 1), (1, 0)),
            num_samples=100,
        )
        for method in ("plateau", "plateau-bootstrap"):
            with self.subTest(method=method):
                # One bootstrap candidate requires no simulated refits.
                options = dict(
                    method=method, lm_mode="constant", top_plateaus=1,
                    return_result=True,
                )
                cached = select_from_curve(curve, **options)
                with patch("lambda_support_recovery.api.compute_support_curve", return_value=curve):
                    direct = select_support(np.eye(2), num_samples=100, **options)

                self.assertEqual(cached.selected_dimension, 2)
                self.assertEqual(direct.selected_dimension, 2)
                np.testing.assert_array_equal(cached.support, masks[1])
                np.testing.assert_array_equal(direct.support, cached.support)
                if method == "plateau":
                    self.assertEqual(
                        cached.diagnostics.selected_scale,
                        cached.diagnostics.plateau_selection.center,
                    )


if __name__ == "__main__":
    unittest.main()
