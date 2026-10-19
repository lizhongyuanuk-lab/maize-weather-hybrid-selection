"""Synthetic checks only; does not read real new test outcomes."""
import unittest

import numpy as np
import pandas as pd

from analyze_segmented_results import (engineering_screen, rename_reference_fields,
                                      complete_subgroups, REFERENCES)
from analyze_rolling_stability import cross_year_stability, within_year_stability


class SegmentedAnalysisTests(unittest.TestCase):
    def test_screen_requires_all_prespecified_components(self):
        row = {"n_years": 5, "relative_gain_of_year_equal_rmse_percent": 2.,
               "years_improved": 4, "crossed_year_site_ci95_high": -.001}
        self.assertTrue(engineering_screen(row)["exploratory_engineering_signal"])
        for column, value in (("n_years", 4), ("relative_gain_of_year_equal_rmse_percent", 1.99),
                              ("years_improved", 3), ("crossed_year_site_ci95_high", 0.),
                              ("relative_gain_of_year_equal_rmse_percent", np.nan)):
            self.assertFalse(engineering_screen({**row, column: value})["exploratory_engineering_signal"])

    def test_public_reference_names_are_unambiguous(self):
        row = {"macro_rmse_W0": 2., "macro_rmse_weather": 1.9,
               "reference": "cumulative", "delta_rmse": -.1}
        renamed = rename_reference_fields(row)
        self.assertEqual(renamed["macro_rmse_reference"], 2.)
        self.assertEqual(renamed["macro_rmse_segmented"], 1.9)
        self.assertEqual(renamed["reference"], "cumulative")
        self.assertEqual(rename_reference_fields(pd.DataFrame([row])).iloc[0].to_dict(), renamed)

    @staticmethod
    def differences():
        return pd.DataFrame({"Year": [2020, 2020, 2021, 2021],
            "Env": ["A_2020", "B_2020", "A_2021", "B_2021"],
            "Experiment_Code": ["A", "B", "A", "B"], "n_W0": [2, 3, 40, 60],
            "rmse_W0": [2., 2., 4., 4.], "rmse_weather": [2., 2., 4., 4.],
            "delta_rmse": [0., 0., 0., 0.]})

    def test_identity_reference_has_zero_difference_and_intervals(self):
        result = cross_year_stability(self.differences(), repeats=40)
        for key in ("year_equal_delta_rmse", "relative_gain_of_year_equal_rmse_percent",
                    "year_block_ci95_low", "year_block_ci95_high", "crossed_year_site_ci95_low",
                    "crossed_year_site_ci95_high"):
            self.assertEqual(result[key], 0.)
        self.assertEqual(result["years_improved"], 0)
        self.assertFalse(engineering_screen(result)["exploratory_engineering_signal"])

    def test_full_empty_subgroup_grid_stays_missing_not_zero(self):
        frame = self.differences()
        annual = [{"Year": y, "model": "Ridge", "window": 120, "reference": reference,
                   "group_type": "hybrid", "group": "seen", **within_year_stability(group, repeats=20)}
                  for y, group in frame.groupby("Year") for reference in REFERENCES]
        multi = [{"model": "Ridge", "window": 120, "reference": reference,
                  "group_type": "hybrid", "group": "seen", **cross_year_stability(frame, repeats=20)}
                 for reference in REFERENCES]
        a, c = complete_subgroups(annual, cross=False), complete_subgroups(multi, cross=True)
        self.assertEqual(len(a), 480)
        self.assertEqual(len(c), 96)
        empty = c.loc[(c.group_type == "site_code") & (c.group == "unseen")]
        self.assertTrue((empty.n_years == 0).all())
        self.assertTrue(empty.year_equal_delta_rmse.isna().all())
        self.assertTrue((empty.bootstrap_status == "empty_subgroup").all())
        self.assertTrue((empty.missing_prespecified_years == "2020;2021;2022;2023;2024").all())
        available = c.loc[(c.model == "Ridge") & (c.window == 120)
                          & (c.group_type == "hybrid") & (c.group == "seen")]
        self.assertTrue((available.evaluated_years == "2020;2021").all())
        self.assertTrue((available.missing_prespecified_years == "2022;2023;2024").all())


if __name__ == "__main__":
    unittest.main()
