import copy
import math
import unittest

from scipy import stats

from tgif_dwa.wear_statistics import analyze_records, holm_adjust


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        # Synthetic matched observations; no experiment results are needed.
        self.rows = []
        for fold in range(1, 19):
            for method, offset in (("FINAL_MODEL", 0.006 + 0.002 * (fold - 9.5)),
                                   ("VIDEO_ONLY", 0.0), ("EARLY_CONCAT", 0.02),
                                   ("FIXED_WINDOW_ATTENTION", 0.03)):
                self.rows.append({
                    "fold": fold, "subject": f"sbj_{fold - 1}", "seed": 47,
                    "method": method, "protocol": "synthetic_fixture",
                    "macro_f1_19": 0.4 + offset, "map_at_0.5": 0.5 + offset,
                })

    def test_known_subject_differences_have_expected_statistics(self):
        report = analyze_records(self.rows)
        result = report["comparisons"][0]
        # Sample variance of consecutive integers 1..18 is 18*19/12.
        expected_sd = 0.002 * math.sqrt(18 * 19 / 12)
        expected_t = 0.006 / (expected_sd / math.sqrt(18))
        self.assertAlmostEqual(result["mean_difference"], 0.006)
        self.assertAlmostEqual(result["difference_std"], expected_sd)
        self.assertAlmostEqual(result["t_statistic"], expected_t)
        self.assertAlmostEqual(result["p_two_sided"], 2 * stats.t.sf(expected_t, 17))
        self.assertEqual(len(report["comparisons"]), 6)
        self.assertGreaterEqual(result["p_holm_six_comparisons"], result["p_two_sided"])

    def test_seed_replication_does_not_inflate_subject_sample_size(self):
        second = copy.deepcopy(self.rows)
        for row in second:
            row["seed"] = 48
        original = analyze_records(self.rows)
        multiple = analyze_records(self.rows + second)
        for a, b in zip(original["comparisons"], multiple["comparisons"]):
            self.assertAlmostEqual(a["p_two_sided"], b["p_two_sided"])
            self.assertEqual(b["n_subjects"], 18)
            self.assertEqual(b["degrees_of_freedom"], 17)
            self.assertEqual(b["n_seeds"], 2)

    def test_rejects_duplicate_missing_and_mixed_protocol_records(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze_records(self.rows + self.rows[:1])
        with self.assertRaisesRegex(ValueError, "matching"):
            analyze_records(self.rows[1:])
        altered = copy.deepcopy(self.rows)
        altered[0]["protocol"] = "fixed_epoch_loso_v2"
        with self.assertRaisesRegex(ValueError, "same training protocol"):
            analyze_records(altered)

    def test_holm_monotonic_adjustment_preserves_input_order(self):
        adjusted = holm_adjust([0.04, 0.001, 0.02])
        self.assertAlmostEqual(adjusted[0], 0.04)
        self.assertAlmostEqual(adjusted[1], 0.003)
        self.assertAlmostEqual(adjusted[2], 0.04)


if __name__ == "__main__":
    unittest.main()
