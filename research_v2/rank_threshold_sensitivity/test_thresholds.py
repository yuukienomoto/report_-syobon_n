"""Synthetic regression tests. These scores are never research results."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from importlib.metadata import version
import numpy as np
import pandas as pd

from test_smoothing import sample
from thresholds import (CONDITIONS, FEATURES, MODEL_PLAN, labels, build, prepare, run,
                        check_counters, verify_baseline, saved_model, common_keys)
from mahjong_research.analysis import validate_data
from mahjong_research.inputs import sha256, digest, write_json
from compare import fit_fixed, score


class ThresholdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name)
        raw = sample(1200)
        raw['dan'] = np.tile(np.repeat([16, 17, 18], 40), 10)
        cls.d, cls.cut = validate_data(raw, FEATURES, initial_exclude=0)
        cls.blocks, cls.audits, cls.members = {}, {}, []
        for c in CONDITIONS:
            cls.blocks[c], cls.audits[c], m = build(cls.d, cls.cut, set(), c)
            cls.members.append(m.assign(condition=c, dataset='internal'))
        cls.members[0].to_csv(cls.path/'fold_membership.csv', index=False)
        cls.audits['baseline'].to_csv(cls.path/'fold_audit.csv', index=False)
        cls.source = dict(raw_sha256='fixture', shared_sha256='fixture', signature='fixture')
        cls.config = dict(params=MODEL_PLAN, source=cls.source, inputs={'fold_membership.csv':sha256(cls.path/'fold_membership.csv')},
                          versions={'lightgbm':version('lightgbm')})
        write_json(cls.path/'protocol.json', cls.config)
        (cls.path/'models').mkdir()
        me, pred = [], []
        for r in cls.audits['baseline'].itertuples():
            b = cls.blocks['baseline'][r.width]
            tr = b[b['last'].le(r.train_through)]
            ev = b[b['first'].gt(r.train_through) & b['last'].le(r.eval_through)]
            model = fit_fixed(tr, list(FEATURES), cls.path/'models', f'w{r.width}_f{r.fold}_A',
                              digest([cls.config, r.width, r.fold, list(FEATURES)]))
            score(ev, model.predict(ev[list(FEATURES)], num_threads=1),
                  dict(dataset='internal', width=r.width, fold=r.fold, model='A'), me, pred)
            if r.fold == 3:
                score(b, model.predict(b[list(FEATURES)], num_threads=1),
                      dict(dataset='external', width=r.width, fold=3, model='A'), me, pred)
        pd.DataFrame(me).to_csv(cls.path/'model_metrics.csv', index=False)
        pd.DataFrame(pred).to_csv(cls.path/'predictions.csv', index=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_labels_no_tie_split_no_repacking(self):
        raw = self.d.copy()
        raw.loc[29, 'hour'] = raw.loc[30, 'hour']
        d, cutoff = validate_data(raw, FEATURES, initial_exclude=20)
        shared = {d.game_id.iloc[62]}
        for c in CONDITIONS:
            relabeled = labels(d, c)
            self.assertTrue(relabeled.loc[d.dan.le(16), 'target_label'].eq(0).all())
            self.assertTrue(relabeled.loc[d.dan.ge(18), 'target_label'].eq(1).all())
            got = relabeled.loc[d.dan.eq(17), 'target_label']
            self.assertTrue(got.isna().all() if c == 'baseline' else got.eq(int(c == 'include_17_high')).all())
            blocks, audit, _ = build(d, cutoff, shared, c)
            self.assertTrue(audit.shared_ids.eq(0).all())
            for w, b in blocks.items():
                seen = set()
                for row in b.itertuples():
                    sub = relabeled[relabeled.game_id.isin(row.source_ids)]
                    self.assertEqual(len(sub), w)
                    self.assertEqual(sub.target_label.nunique(dropna=False), 1)
                    self.assertFalse(seen.intersection(row.source_ids) or shared.intersection(row.source_ids))
                    self.assertTrue(sub.hour.gt(cutoff).all())
                    self.assertEqual(len(d[d.hour.between(row.first, row.last)]), w)
                    self.assertTrue(np.array_equal(np.diff(sub.index), np.ones(w-1)))
                    seen.update(row.source_ids)

    def test_fixed_boundaries_and_train_values(self):
        ref = self.audits['baseline']
        for c in CONDITIONS:
            pd.testing.assert_frame_equal(self.audits[c][['train_through','eval_through']], ref[['train_through','eval_through']])
        end = ref.iloc[0].train_through
        poisoned = self.d.copy()
        for f in FEATURES:
            poisoned.loc[poisoned.hour.gt(end), f+'__numerator'] = 9999
        changed, _, _ = build(poisoned, self.cut, set(), 'include_17_low')
        for w in (10,20):
            orig = self.blocks['include_17_low'][w]
            pd.testing.assert_frame_equal(orig[orig['last'].le(end)], changed[w][changed[w]['last'].le(end)])
            keys = common_keys(self.blocks, w)
            base = self.blocks['baseline'][w].set_index('source_key').loc[sorted(keys)]
            for c in CONDITIONS[1:]:
                common = self.blocks[c][w].set_index('source_key').loc[sorted(keys)]
                pd.testing.assert_frame_equal(base[['label']+list(FEATURES)], common[['label']+list(FEATURES)])

    def test_zero_denominator_and_bad_counters(self):
        check_counters(self.d)
        self.assertTrue(self.blocks['baseline'][10][FEATURES[0]].isna().all())
        bad = self.d.copy()
        bad.loc[0, FEATURES[0]+'__numerator'] = 1
        with self.assertRaises(ValueError): check_counters(bad)

    def test_prepare_baseline_and_mismatch(self):
        with patch('thresholds.read_run', return_value=(self.d, set(), self.source)), \
             patch('thresholds.validate_data', return_value=(self.d, self.cut)), \
             patch('thresholds.external_data', side_effect=FileNotFoundError('missing external')):
            info = prepare(self.path, self.path, self.path)
            self.assertFalse(info['external'])
            pd.testing.assert_frame_equal(info['blocks']['baseline'][10], self.blocks['baseline'][10])
        with patch('thresholds.read_run', return_value=(self.d, set(), dict(self.source, raw_sha256='wrong'))):
            with self.assertRaises(ValueError): prepare(self.path, self.path, self.path)

    def test_model_and_prediction_integrity(self):
        model = saved_model(self.path, self.config, 10, 3)
        r = self.audits['baseline'].query('width==10 and fold==3').iloc[0]
        b = self.blocks['baseline'][10]
        ev = b[b['first'].gt(r.train_through) & b['last'].le(r.eval_through)]
        p = model.predict(ev[list(FEATURES)], num_threads=1)
        key = dict(dataset='internal', width=10, fold=3)
        pred = pd.read_csv(self.path/'predictions.csv')
        met = pd.read_csv(self.path/'model_metrics.csv')
        verify_baseline(ev, p, key, pred, met)
        with self.assertRaises(ValueError): verify_baseline(ev, p, key, pred.assign(label=1-pred.label), met)
        with self.assertRaises(ValueError): saved_model(self.path, dict(self.config, other=True), 10, 3)

    def test_external_signed_cache_with_rank17(self):
        from windows import tenhou
        ext = sample(2100).assign(player='ウルトラ立直')
        ext['dan'] = np.resize(np.repeat([16, 17, 18], 40), len(ext))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, audit = root/'raw_features_v2.csv', root/'extraction_audit.csv'
            ext.to_csv(raw, index=False)
            ext[['file_id', 'status']].to_csv(audit, index=False)
            write_json(root/'complete.json', dict(signature='test', rows=len(ext),
                extractor_sha256=sha256(tenhou.__file__), sha256=sha256(raw), audit_sha256=sha256(audit),
                order_verified=False, full_history_confirmed=False))
            write_json(root/'input_plan.json', dict(ultra=dict(cache_dir=str(root), signature='test', members=ext.file_id.tolist())))
            def validate(frame, fields):
                return (self.d, self.cut) if frame.player.iloc[0] == 'しょぼんｎ' else validate_data(frame, fields)
            with patch('thresholds.read_run', return_value=(self.d, set(), self.source)), \
                 patch('thresholds.validate_data', side_effect=validate):
                info = prepare(root, self.path, self.path)
                self.assertEqual(set(info['external']), set(CONDITIONS))
                for condition in CONDITIONS:
                    for block in info['external'][condition].values():
                        for row in block.itertuples():
                            used = ext[ext.game_id.isin(row.source_ids)]
                            self.assertTrue(set(row.source_ids).isdisjoint(self.d.game_id))
                            self.assertTrue(labels(used, condition).target_label.eq(row.label).all())
                            self.assertTrue(used.hour.gt(ext.hour.iloc[1499]).all())
                raw.write_text('corrupted cache', encoding='utf8')
                blocked = prepare(root, self.path, self.path)
                self.assertFalse(blocked['external'])
                self.assertIn('署名不一致', blocked['external_note'])

    def test_end_to_end_and_model_resume(self):
        # External fixture reuses arrays to test protocol wiring, not generalization.
        info = dict(data=self.d, cutoff=self.cut, shared=set(), blocks=self.blocks, audits=self.audits,
                    members=pd.concat(self.members), source=self.source, config=self.config,
                    external=self.blocks, external_raw=self.d, external_source={'fixture':True}, external_note='合成外部データ')
        with patch('thresholds.prepare', return_value=info):
            out = run(self.path, self.path, self.path, self.path/'outputs')
            result = pd.read_csv(out/'metrics_comparison.csv')
            self.assertEqual(len(result), 48)
            self.assertEqual(len(pd.read_csv(out/'feature_difference_comparison.csv')), 480)
            self.assertEqual(len(list((out/'models').glob('*.txt'))), 12)
            self.assertEqual(len(pd.read_csv(out/'baseline_reproduction.csv')), 6)
            self.assertTrue((out/'auc_comparison.png').exists())
            with patch('compare.lgb.LGBMClassifier', side_effect=AssertionError('must reuse saved models')):
                resumed = run(self.path, self.path, self.path, self.path/'outputs', resume=out)
            self.assertEqual(resumed, out)
            pd.testing.assert_frame_equal(result, pd.read_csv(out/'metrics_comparison.csv'))

    def test_single_class_empty_and_overlap(self):
        from thresholds import evaluate
        from compare import metrics
        m = metrics(np.zeros(3), np.ones(3)*.3)
        self.assertTrue(np.isnan(m['roc_auc']) and np.isnan(m['balanced_accuracy']))
        b = self.blocks['baseline'][10]
        rows, pred = [], []
        evaluate(None, b, b, {}, rows, pred, set())
        self.assertTrue(np.isnan(rows[0]['roc_auc']) and not pred)
        d = self.d.copy().assign(dan=17)
        blocks, audit, _ = build(d, self.cut, set(), 'baseline')
        self.assertEqual(len(blocks[10]), 0)
        self.assertTrue(audit.train_low.eq(0).all())


if __name__ == '__main__':
    unittest.main()
