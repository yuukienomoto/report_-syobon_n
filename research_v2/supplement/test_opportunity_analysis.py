"""Synthetic regressions for added aggregation, not research findings."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from opportunity_analysis import pool_windows,run_opportunity_analysis,same_population
from mahjong_research.analysis import prepare_windows,validate_data
from mahjong_research.tenhou import FEATURE_NAMES


def games(n=80):
    d=pd.DataFrame(dict(game_id=[f'g{i}' for i in range(n)],hour=pd.date_range('2020-01-01',periods=n,freq='h'),
                        dan=np.where((np.arange(n)//3)%2,18,16),player='shobon'))
    for f in FEATURE_NAMES:
        den=np.where(np.arange(n)%9==0,1,0)
        num=den*np.where(np.arange(n)%2,.8,.2)
        # Integer event counts, with eight or two actions per ten opportunities.
        den=den*10;num=num*10
        d[f+'__denominator']=den;d[f+'__numerator']=num.astype(int)
        d[f]=np.divide(num,den,out=np.full(n,np.nan),where=den>0)
    return d


class OpportunityTests(unittest.TestCase):
    def block(self,d):
        d,cutoff=validate_data(d,FEATURE_NAMES,0)
        return d,prepare_windows(d,FEATURE_NAMES,cutoff)

    def test_sparse_opportunities_survive_without_zero_imputation(self):
        d,b=self.block(games())
        x,t=pool_windows(d,b['meta'])
        self.assertTrue(b['X'].isna().all().all())
        self.assertTrue(x.notna().all().all())
        self.assertAlmostEqual(x.iloc[0,0],.5)

    def test_zero_total_denominator_stays_missing(self):
        d=games();f=FEATURE_NAMES[0]
        d[f+'__denominator']=0;d[f+'__numerator']=0;d[f]=np.nan
        d,b=self.block(d);x,_=pool_windows(d,b['meta'])
        self.assertTrue(x[f].isna().all())

    def test_counts_mean_is_event_weighted_not_mean_of_game_means(self):
        d=games();f='riichi_omote_dora_mean'
        d[f+'__denominator']=0;d[f+'__numerator']=0;d[f]=np.nan
        d.loc[0,[f+'__numerator',f+'__denominator',f]]=[1,1,1]
        d.loc[9,[f+'__numerator',f+'__denominator',f]]=[18,9,2]
        d,b=self.block(d);x,_=pool_windows(d,b['meta'])
        self.assertAlmostEqual(x.iloc[0][f],1.9)

    def test_missing_counts_and_invalid_counts_fail_closed(self):
        d,b=self.block(games());f=FEATURE_NAMES[0]
        with self.assertRaisesRegex(ValueError,'列不足'):pool_windows(d.drop(columns=f+'__numerator'),b['meta'])
        for v in (-1,np.nan,np.inf):
            bad=d.copy();bad[f+'__denominator']=bad[f+'__denominator'].astype(float);bad.loc[0,f+'__denominator']=v
            with self.assertRaises(ValueError):pool_windows(bad,b['meta'])

    def test_ratio_mismatch_and_bad_source_ids_fail(self):
        d,b=self.block(games());d.loc[0,FEATURE_NAMES[0]]=.9
        with self.assertRaisesRegex(ValueError,'不一致'):pool_windows(d,b['meta'])
        d,b=self.block(games());bad=b['meta'].copy();bad.at[0,'source_ids']=('g0',)*10
        with self.assertRaises(ValueError):pool_windows(d,bad)
        with self.assertRaises(ValueError):same_population(b['meta'].iloc[1:],b['meta'])

    def test_full_report_preserves_reference_and_disjoint_sources(self):
        d=games();d.loc[3,'hour']=d.loc[2,'hour'];d.loc[4:6,'dan']=17
        d,cutoff=validate_data(d,FEATURE_NAMES,0)
        reference=prepare_windows(d,FEATURE_NAMES,cutoff,{'g30'})['meta']
        original=d.copy(deep=True)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old.csv';old.write_text('preserve')
            result=run_opportunity_analysis(d,{'g30'},reference,root/'added',initial_exclude=0,make_plots=False)
            self.assertEqual(result['facts']['features_improved_both_groups'],20)
            self.assertFalse(result['facts']['order_verified'])
            self.assertEqual(old.read_text(),'preserve')
            ids=pd.read_csv(result['output_dir']/'opportunity_disjoint_windows.csv').source_ids.map(json.loads)
            flat=[g for window in ids for g in window]
            self.assertEqual(len(flat),len(set(flat)))
            self.assertNotIn('g30',flat)
            self.assertTrue((result['output_dir']/'report_ja.md').is_file())
            second=run_opportunity_analysis(d,{'g30'},reference,root/'added',initial_exclude=0,make_plots=False)
            self.assertNotEqual(result['output_dir'],second['output_dir'])
        pd.testing.assert_frame_equal(d,original)


if __name__=='__main__':unittest.main()
