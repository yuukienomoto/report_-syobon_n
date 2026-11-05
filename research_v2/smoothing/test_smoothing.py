"""Synthetic data only; no model or ZIP access."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from study import (FEATURES, DEFS, analyze, block_values, contrasts, describe, make_blocks,
                   read_run, run_study, sha256, tenhou, validate_data)
from presentation import candidates


def sample(n=100):
    hours = pd.date_range('2013-01-01', periods=n, freq='h')
    ids = [s+'gm-0029-0000-aaaa' for s in hours.strftime('%Y%m%d%H')]
    data = dict(game_id=ids, file_id=[s+'.mjlog' for s in ids], player='しょぼんｎ',
                dan=np.where(np.arange(n) < n-45, 16, 18), hour=hours, status='ok',
                seat=1, go_type=41, error='', xml_sha256='fixture')
    for j, f in enumerate(FEATURES):
        den = (np.arange(n)+j)%5
        num = den*((np.arange(n)+j)%3 == 0)*int(DEFS[f]['upper'])
        if j == 0:
            den[:] = 0
            num[:] = 0
        data[f+'__denominator'] = den
        data[f+'__numerator'] = num
        data[f] = np.divide(num, den, out=np.full(n, np.nan), where=den>0)
    return pd.DataFrame(data)


class SmoothingTests(unittest.TestCase):
    def test_no_tie_split_or_repacking(self):
        raw = sample()
        raw.loc[9,'hour'] = raw.loc[10,'hour']
        raw.loc[31,'dan'] = 17
        raw.loc[54,'hour'] = raw.loc[55,'hour']  # mixed class hour
        shared = {raw.game_id.iloc[70]}
        d, cut = validate_data(raw, FEATURES, initial_exclude=1)
        blocks = make_blocks(d, cut, shared)
        shuffled, _ = validate_data(raw.sample(frac=1, random_state=8), FEATURES, initial_exclude=1)
        self.assertEqual(blocks, make_blocks(shuffled, cut, shared))
        for widths in blocks.values():
            for w, starts in widths.items():
                used = []
                for i in starts:
                    rows = d.iloc[i:i+w]
                    self.assertEqual(rows.target_label.nunique(dropna=False), 1)
                    self.assertFalse(rows.target_label.isna().any() or rows.game_id.isin(shared).any())
                    self.assertTrue(rows.hour.gt(cut).all())
                    if w > 1:
                        self.assertEqual(len(rows), d.hour.isin(rows.hour).sum())
                    used.extend(rows.game_id)
                self.assertEqual(len(used), len(set(used)))
        common = blocks['common_games']
        self.assertTrue(common[20])
        self.assertEqual(common[1], [k for i in common[20] for k in range(i,i+20)])
        self.assertEqual(common[1], [k for i in common[10] for k in range(i,i+10)])

    def test_unequal_denominators_event_means_and_missing(self):
        raw = sample()
        fields = ('response_non_genbutsu_rate','chasing_wait_types_mean','riichi_omote_dora_mean')
        for f in fields:
            raw.loc[:9, f+'__denominator'] = [1]*9+[9]
            raw.loc[:9, f+'__numerator'] = [0]*9+[9*int(DEFS[f]['upper'])]
            raw.loc[:9, f] = [0]*9+[int(DEFS[f]['upper'])]
        d, _ = validate_data(raw, FEATURES, initial_exclude=0)
        x, den, _ = block_values(d,[0],10)
        self.assertTrue(np.isnan(x[0,0]))
        for f in fields:
            j = FEATURES.index(f)
            expected = d.iloc[:10][f+'__numerator'].sum()/d.iloc[:10][f+'__denominator'].sum()
            self.assertEqual(x[0,j], expected)
            self.assertNotAlmostEqual(expected, d.iloc[:10][f].mean())
            self.assertEqual(den[0,j], d.iloc[:10][f+'__denominator'].sum())

    def test_ties_overlap_and_smd(self):
        a, b = np.array([0,1]), np.array([1,1])
        r = contrasts(a,b,describe(a,[1,1]),describe(b,[1,1]),[-.5,.5,1.5])
        self.assertAlmostEqual(r['auc'],.75)
        self.assertAlmostEqual(r['auc_reversed'],.25)
        self.assertAlmostEqual(r['overlap'],.5)
        self.assertAlmostEqual(r['smd'],1.)
        r = contrasts(a[:0],b,describe([],[]),describe(b,[1,1]),[-.5,.5,1.5])
        self.assertTrue(np.isnan(r['auc']) and np.isnan(r['overlap']))

    def test_empty_common_and_no_opportunity(self):
        raw = sample()
        raw['hour'] = raw.hour.iloc[0]
        d, cut = validate_data(raw, FEATURES, initial_exclude=0)
        blocks = make_blocks(d,cut,set())
        stats, comp, _, arrays = analyze(d,blocks)
        self.assertEqual(len(stats),240)
        self.assertEqual(len(comp),120)
        self.assertTrue(comp[comp.scope.eq('common_games')].low_n.eq(0).all())
        t = candidates(comp,arrays)
        self.assertTrue(t[t.feature.eq(FEATURES[0])].category.eq('D').all())

    def test_completed_cache_to_report_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root/'cache'
            cache.mkdir()
            data = sample(1600)
            raw = cache/'raw_features_v2.csv'
            audit = cache/'extraction_audit.csv'
            data.to_csv(raw,index=False)
            data[['file_id','status']].to_csv(audit,index=False)
            marker = dict(rows=len(data),sha256=sha256(raw),audit_sha256=sha256(audit),
                          extractor_sha256=sha256(tenhou.__file__),signature='fixture',
                          order_verified=False,full_history_confirmed=False)
            (cache/'complete.json').write_text(json.dumps(marker))
            plan = dict(cache_dir=str(cache), signature='fixture', members=data.file_id.tolist())
            (root/'input_plan.json').write_text(json.dumps(dict(shobon=plan)))
            (root/'shared_games.csv').write_text('game_id\n')
            out = run_study(root,root/'results')
            self.assertEqual(len(pd.read_csv(out/'comparison_20x3.csv')),60)
            self.assertEqual(len(pd.read_csv(out/'group_statistics.csv')),240)
            self.assertEqual(len(list((out/'figures').glob('*.png'))),6)
            self.assertIn('共通対局',(out/'report_ja.md').read_text(encoding='utf-8'))
            self.assertEqual(sha256(raw),marker['sha256'])
            with raw.open('ab') as stream:
                stream.write(b' ')
            with self.assertRaisesRegex(ValueError,'ハッシュ'):
                read_run(root)
