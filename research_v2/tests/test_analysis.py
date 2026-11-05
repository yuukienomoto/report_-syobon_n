"""Synthetic invariance/leakage tests; these are not real-log performance evidence."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mahjong_research.analysis import (
    AnalysisBlocked, classification_metrics, disjoint_windows, fit_means,
    load_frozen_bundle, prepare_windows, run_descriptive, run_external,
    run_internal, source_ids, summarize,
    temporal_splits, validate_data,
)


def games(n=100):
    return pd.DataFrame(dict(game_id=[f"g{i:04d}" for i in range(n)],
                             hour=pd.date_range("2020-01-01", periods=n, freq="h"),
                             dan=np.where(np.arange(n) % 2 == 0, 16, 18),
                             player="shobon", a=np.arange(n, dtype=float),
                             b=np.ones(n)))


def synthetic_cross(frame):
    return frame.assign(c=frame.a * frame.b)


def changed_cross(frame):
    return frame.assign(c=frame.a + frame.b)


class WindowTests(unittest.TestCase):
    def test_rank17_and_initial_games_remain_in_history(self):
        d = games(30)
        d.loc[4:8, "dan"] = 17
        d, cutoff = validate_data(d, ["a"], initial_exclude=9)
        block = prepare_windows(d, ["a"], cutoff)
        self.assertEqual(block["meta"].iloc[0].game_id, "g0009")
        self.assertEqual(block["meta"].iloc[0].source_rank17, 5)
        self.assertEqual(block["X"].iloc[0].a, 4.5)

    def test_initial_cutoff_excludes_entire_hour(self):
        d = games(30)
        d.loc[10, "hour"] = d.loc[9, "hour"]
        d, cutoff = validate_data(d, ["a"], initial_exclude=10)
        block = prepare_windows(d, ["a"], cutoff)
        self.assertNotIn("g0010", set(block["meta"].game_id))
        self.assertIn("g0002", source_ids(block))  # early history was not deleted

    def test_complete_tie_buckets_order_invariant(self):
        d = games(30)
        d.loc[3, "hour"] = d.loc[2, "hour"]
        d, cutoff = validate_data(d, ["a"], initial_exclude=0)
        first = prepare_windows(d, ["a"], cutoff)
        # Exchange sort names inside the hour, keeping each feature bound to its game.
        altered = d.copy()
        altered.loc[2, "game_id"], altered.loc[3, "game_id"] = "g0003", "g0002"
        second = prepare_windows(altered, ["a"], cutoff)
        pd.testing.assert_frame_equal(first["X"], second["X"])
        pd.testing.assert_series_equal(first["meta"].game_id, second["meta"].game_id)
        self.assertNotIn("g0012", set(first["meta"].game_id))  # starts halfway into hour 2
        self.assertIn("g0011", set(first["meta"].game_id))

    def test_shared_game_excludes_windows_without_repacking(self):
        d, cutoff = validate_data(games(40), ["a"], initial_exclude=0)
        block = prepare_windows(d, ["a"], cutoff, {"g0015"})
        excluded = block["excluded"].query("reason == 'shared_game_in_window'").game_id.tolist()
        self.assertEqual(excluded, [f"g{i:04d}" for i in range(15, 25)])
        self.assertEqual(block["meta"].query("game_id == 'g0025'").iloc[0].source_ids[0], "g0016")
        self.assertNotIn("g0015", source_ids(block))

    def test_descriptive_missing_is_not_zero_or_shorter_window(self):
        d = games(30)
        d.loc[5, "a"] = np.nan
        d, cutoff = validate_data(d, ["a", "b"], initial_exclude=0)
        block = prepare_windows(d, ["a", "b"], cutoff)
        self.assertTrue(np.isnan(block["X"].iloc[0].a))
        self.assertEqual(block["X"].iloc[0].b, 1)
        self.assertEqual(block["X"].iloc[6].a, 10.5)  # first full window past missing game

    def test_nonoverlapping_auxiliary_windows(self):
        d, cutoff = validate_data(games(60), ["a"], initial_exclude=0)
        block = disjoint_windows(prepare_windows(d, ["a"], cutoff))
        count = sum(len(v) for v in block["meta"].source_ids)
        self.assertEqual(count, len(source_ids(block)))
        self.assertEqual(count, 60)

    def test_temporal_blocks_no_sources_overlap(self):
        d, cutoff = validate_data(games(120), ["a"], initial_exclude=10)
        for _, tr, va in temporal_splits(d, cutoff):
            self.assertLess(tr.hour.max(), va.hour.min())
            train = prepare_windows(tr, ["a"], cutoff)
            valid = prepare_windows(va, ["a"], cutoff)
            self.assertFalse(source_ids(train) & source_ids(valid))
            self.assertEqual(valid["excluded"].iloc[:9].reason.unique().tolist(), ["insufficient_block_history"])


class ImputationTests(unittest.TestCase):
    def test_only_training_information(self):
        d, cutoff = validate_data(games(50), ["a"], initial_exclude=10)
        d.loc[15, "a"] = 1000000  # forbidden shared game cannot affect mean
        means, ids = fit_means(d.iloc[:25], ["a"], cutoff, {"g0015"})
        expected = np.mean([i for i in range(10, 25) if i != 15])
        self.assertEqual(means.a, expected)
        self.assertNotIn("g0015", ids)
        self.assertFalse(set(d.iloc[25:].game_id) & ids)
        d.loc[25:, "a"] = -1000000  # future change cannot affect training mean
        again, _ = fit_means(d.iloc[:25], ["a"], cutoff, {"g0015"})
        pd.testing.assert_series_equal(means, again)

    def test_all_missing_training_feature_blocks(self):
        d, cutoff = validate_data(games(30), ["a"], initial_exclude=0)
        d["a"] = np.nan
        with self.assertRaises(AnalysisBlocked):
            fit_means(d, ["a"], cutoff)

    def test_legacy_cross_is_after_imputation(self):
        d = games(25)
        d.loc[5, "a"] = np.nan
        d["b"] = 3.
        d, cutoff = validate_data(d, ["a", "b"], initial_exclude=0)
        def cross(x):
            return x.assign(c=x.a*x.b)
        block = prepare_windows(d, ["a", "b", "c"], cutoff,
                                means=pd.Series({"a": 5., "b": 3.}),
                                base_features=["a", "b"], cross_builder=cross)
        self.assertAlmostEqual(block["X"].iloc[0].c, 13.5)
        raw = prepare_windows(d, ["a", "b", "c"], cutoff,
                              base_features=["a", "b"], cross_builder=cross)
        self.assertTrue(np.isnan(raw["X"].iloc[0].c))


class MetricTests(unittest.TestCase):
    def test_roc_auc_ties_and_baseline(self):
        result = classification_metrics([0, 0, 1, 1], [.1, .4, .35, .8])
        self.assertEqual(result["roc_auc"], .75)
        constant = classification_metrics([0, 0, 1, 1], [1., 1., 1., 1.])
        self.assertEqual(constant["roc_auc"], .5)
        self.assertEqual(constant["balanced_accuracy"], .5)
        self.assertEqual(constant["recall_weak"], 0)
        self.assertEqual(constant["recall_strong"], 1)

    def test_single_class_no_auc_no_balanced_accuracy(self):
        result = classification_metrics([1, 1, 1], [.1, .8, .9])
        self.assertTrue(np.isnan(result["roc_auc"]))
        self.assertTrue(np.isnan(result["balanced_accuracy"]))
        self.assertEqual(result["accuracy"], 2/3)
        self.assertTrue(np.isnan(result["recall_weak"]))

    def test_nonbinary_labels_are_rejected_not_truncated(self):
        with self.assertRaises(AnalysisBlocked):
            classification_metrics([0, .5, 1], [.1, .5, .9])
        with self.assertRaises(AnalysisBlocked):
            classification_metrics([0, 1], [.1, .9], predicted=[0, .5])

    def test_smd_uses_valid_counts_and_reports_missingness(self):
        X = pd.DataFrame({"a": [0., 2., np.nan, 2., 4., 6.]})
        stats, diffs = summarize(X, [0, 0, 0, 1, 1, 1], "test")
        self.assertEqual(stats.iloc[0].n, 3)
        self.assertEqual(stats.iloc[0].valid_n, 2)
        self.assertAlmostEqual(stats.iloc[0].missing_rate, 1/3)
        self.assertEqual(diffs.iloc[0].mean_diff_high_minus_low, 3)
        self.assertAlmostEqual(diffs.iloc[0].pooled_sd, np.sqrt(10/3))


class InputOutputTests(unittest.TestCase):
    def test_duplicate_invalid_rank_infinite_and_mixed_player_block(self):
        for kind in ("duplicate", "rank", "infinite", "player", "minute"):
            d = games(30)
            if kind == "duplicate": d.loc[1, "game_id"] = d.loc[0, "game_id"]
            if kind == "rank":
                d["dan"] = d.dan.astype(float)
                d.loc[0, "dan"] = 16.5
            if kind == "infinite": d.loc[0, "a"] = np.inf
            if kind == "player": d.loc[0, "player"] = "ultra"
            if kind == "minute": d.loc[0, "hour"] += pd.Timedelta(minutes=1)
            with self.subTest(kind=kind), self.assertRaises(AnalysisBlocked):
                validate_data(d, ["a"], initial_exclude=0)

    def test_descriptive_exports_and_preserves_input_and_previous_outputs(self):
        d = games(100)
        original = d.copy(deep=True)
        temp_root = Path(__file__).resolve().parents[1] / "tests" / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as root:
            out = Path(root)/"analysis"
            result = run_descriptive(d, ["a", "b"], out,
                                     initial_exclude=10, make_plots=False)
            self.assertFalse(result["summary"]["order_verified"])
            self.assertFalse(result["summary"]["full_history_confirmed"])
            for name in ("group_statistics.csv", "raw_feature_quality.csv",
                         "direction_robustness.csv", "calendar_month_statistics.csv",
                         "primary_windows.csv", "descriptive_report_ja.md"):
                self.assertTrue((out/name).is_file())
            pd.testing.assert_frame_equal(d, original)
            with self.assertRaises(AnalysisBlocked):
                run_descriptive(d, ["a", "b"], out, initial_exclude=10, make_plots=False)


@unittest.skipUnless(importlib.util.find_spec("lightgbm") and importlib.util.find_spec("matplotlib"),
                     "LightGBM/matplotlib required for synthetic integration tests")
class SyntheticPipelineTests(unittest.TestCase):
    """Small generated data validate APIs; no thesis performance is inferred."""
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / "tests" / ".tmp"
        root.mkdir(exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=root)
        cls.root = Path(cls.temp.name)
        cls.train = games(240)
        cls.train["a"] = (np.sin(np.arange(240) / 17) + 1) / 2
        cls.train["b"] = (np.cos(np.arange(240) / 11) + 1) / 2
        cls.train.loc[::37, "a"] = np.nan
        cls.train.loc[[71, 100, 151], "dan"] = 17
        cls.train.loc[23, "hour"] = cls.train.loc[22, "hour"]
        cls.external = games(100)
        cls.external["game_id"] = "u" + cls.external.game_id
        cls.external["player"] = "ultra"
        cls.external["a"] = (np.sin(np.arange(100) / 13) + 1) / 2
        cls.external["b"] = (np.cos(np.arange(100) / 9) + 1) / 2
        cls.external.loc[25, "game_id"] = "g0070"
        cls.external.loc[::29, "a"] = np.nan
        cls.forbidden = {"g0070"}
        cls.results = {}
        for version in ("new", "legacy_cross"):
            features = ["a", "b"] if version == "new" else ["a", "b", "c"]
            cross = None if version == "new" else synthetic_cross
            desc = run_descriptive(cls.train, features, cls.root/version/"descriptive",
                                   forbidden=cls.forbidden, initial_exclude=20,
                                   base_features=["a", "b"], cross_builder=cross)
            bundle = run_internal(cls.train, features, cls.root/version/"model",
                                  forbidden=cls.forbidden, initial_exclude=20,
                                  base_features=["a", "b"], cross_builder=cross)
            restored = load_frozen_bundle(bundle["model_dir"], cross)
            ext = run_external(restored, cls.external, cls.root/version/"external")
            cls.results[version] = dict(desc=desc, bundle=restored, external=ext)

    @classmethod
    def tearDownClass(cls):
        # The TemporaryDirectory was created under this test's workspace root.
        assert cls.root.resolve().is_relative_to(Path(__file__).resolve().parents[1])
        cls.temp.cleanup()

    def test_full_pipeline_generates_expected_artifacts_and_preserves_population(self):
        for version, result in self.results.items():
            self.assertEqual(len(result["bundle"]["metrics"]), 12)
            self.assertEqual(len(result["external"]), 4)
            self.assertTrue((self.root/version/"model"/"fold1_diagnostics.png").is_file())
            self.assertTrue((self.root/version/"external"/"external_diagnostics.png").is_file())
            self.assertTrue((self.root/version/"descriptive"/"calendar_trends_01.png").is_file())
            self.assertTrue((self.root/version/"descriptive"/"distribution_01.png").is_file())
            self.assertNotIn("g0070", result["bundle"]["fit_ids"])
            self.assertNotIn("g0070", result["bundle"]["train_source_ids"])
            self.assertEqual(result["bundle"]["fold_audit"].source_overlap.tolist(), [0, 0, 0])
            self.assertEqual(result["bundle"]["protocol"]["threshold"], .5)
            self.assertFalse(result["bundle"]["protocol"]["order_verified"])
            self.assertFalse(result["bundle"]["protocol"]["full_history_confirmed"])
        pd.testing.assert_series_equal(self.results["new"]["desc"]["primary"]["meta"].game_id,
                                       self.results["legacy_cross"]["desc"]["primary"]["meta"].game_id)

    def test_completed_external_resume_does_not_predict_again(self):
        result = self.results["new"]
        with patch("lightgbm.Booster.predict", side_effect=AssertionError("prediction repeated")):
            repeated = run_external(result["bundle"], self.external, self.root/"new"/"external")
        pd.testing.assert_frame_equal(repeated, result["external"], check_exact=False)

    def test_changed_external_input_and_training_overlap_block(self):
        changed = self.external.copy()
        changed.loc[1, "a"] += .01
        with self.assertRaises(AnalysisBlocked):
            run_external(self.results["new"]["bundle"], changed, self.root/"new"/"external")
        changed = self.external.copy()
        changed.loc[0, "game_id"] = "g0050"
        with self.assertRaises(AnalysisBlocked):
            run_external(self.results["new"]["bundle"], changed, self.root/"new"/"external")

    def test_changed_cross_transform_and_preprocessing_code_block(self):
        bundle = self.results["legacy_cross"]["bundle"]
        with self.assertRaises(AnalysisBlocked):
            load_frozen_bundle(bundle["model_dir"], changed_cross)
        with patch("mahjong_research.analysis._module_hash", return_value="changed-code"):
            with self.assertRaises(AnalysisBlocked):
                load_frozen_bundle(bundle["model_dir"], synthetic_cross)

    def test_mutated_imputation_artifact_blocks_load(self):
        path = self.results["new"]["bundle"]["model_dir"] / "final_training_means.csv"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            with self.assertRaises(AnalysisBlocked):
                load_frozen_bundle(path.parent)
        finally:
            path.write_bytes(original)

    def test_partial_external_attempt_is_not_silently_retried(self):
        bundle = self.results["new"]["bundle"]
        path = bundle["model_dir"] / "external_attempt.json"
        original = path.read_bytes()
        try:
            entry = json.loads(original)
            entry["status"] = "started"
            path.write_text(json.dumps(entry), encoding="utf-8")
            with self.assertRaises(AnalysisBlocked):
                run_external(bundle, self.external, self.root/"new"/"external")
        finally:
            path.write_bytes(original)

    def test_all_internal_folds_blocked_are_saved_without_fake_metrics(self):
        d = self.train.copy()
        d["dan"] = 16
        d.loc[230:, "dan"] = 18
        output = self.root/"all_folds_blocked"
        bundle = run_internal(d, ["a", "b"], output, initial_exclude=20,
                              forbidden=self.forbidden, make_plots=False)
        self.assertTrue(bundle["metrics"].empty)
        self.assertTrue(bundle["fold_audit"].status.str.startswith("blocked:").all())
        resumed = load_frozen_bundle(output)
        self.assertTrue(resumed["metrics"].empty)
        self.assertEqual(resumed["freeze"]["train_strong"], 10)


if __name__ == "__main__":
    unittest.main()
