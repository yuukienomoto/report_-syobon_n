import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from stability import classify, decisions, measured, memberships, split_audit, group_rows, run
from study import FEATURES, make_blocks, validate_data
from test_smoothing import sample


class PhaseOneTests(unittest.TestCase):
    def test_report_with_synthetic_loaded_cache(self):
        with tempfile.TemporaryDirectory() as tmp, patch('stability.read_run',return_value=(sample(1700),set(),{})):
            out=run('synthetic',Path(tmp))
            self.assertEqual(len(pd.read_csv(out/'feature_decisions.csv')),40)
            self.assertEqual(len(pd.read_csv(out/'fold_feasibility.csv')),6)
            self.assertTrue((out/'report_ja.md').is_file() and (out/'period_medians_10.png').is_file())

    def test_concentration(self):
        r = measured(np.array([0.,1.,.2,np.nan]),np.array([1,2,10,0]),True)
        self.assertEqual(r['valid_n'],3)
        self.assertEqual(r['small_den_extreme_n'],2)
        self.assertAlmostEqual(r['zero_share'],1/3)
        self.assertTrue(np.isnan(measured(np.array([1.]),np.array([3]),False)['one_share']))

    def test_flags_and_group_specific_reference(self):
        base = dict(width=10,valid_n=30,missing_rate=0,opp_median=5,mode_share=.3,
                    min_period_valid=10,rate=True,extreme_n=0,small_den_extreme_n=0,
                    raw_sd=1,raw_iqr=1,sd=.04,iqr=.1,median_range=.1)
        rows=[]
        for label in (0,1):
            for i in range(5):
                r=dict(base,label=label,feature=str(i),japanese=str(i))
                if i==1 and label: r['median_range']=.2
                if i==2: r.update(rate=False,raw_iqr=0)
                if i==3: r.update(extreme_n=5,small_den_extreme_n=5)
                if i==4: r.update(sd=.5)
                rows.append(r)
        g=pd.DataFrame(rows)
        result=decisions(classify(g,5)).set_index('feature')
        self.assertEqual(result.decision.to_dict(),dict(zip('01234',
                         ['両段位期で安定','片方のみ安定','判断困難','判断困難','不安定'])))
        self.assertTrue(decisions(classify(g,10)).decision.eq('判断困難').all())

    def test_split_boundaries_periods_and_single_class(self):
        d,cut=validate_data(sample(183),FEATURES,initial_exclude=1)
        starts=make_blocks(d,cut,set())['available']
        folds,members=split_audit(d,cut,starts)
        self.assertEqual(len(folds),6)
        self.assertTrue(folds.shared_ids.eq(0).all())
        self.assertGreater(folds.dropped_crossing.sum(),0)
        self.assertFalse(folds.two_class_evaluation.all())
        counts=members[members.role.eq('evaluation')].groupby(['width','fold']).size()
        for row in folds.itertuples():
            self.assertEqual(row.eval_low+row.eval_high,counts.get((row.width,row.fold),0))
        groups,p=group_rows(d,starts)
        self.assertEqual(len(groups),80)
        self.assertEqual(len(p),240)
        self.assertEqual(p.valid_n.sum(),groups.valid_n.sum())
        with self.assertRaises(ValueError):
            memberships(d,[1,1],10)
