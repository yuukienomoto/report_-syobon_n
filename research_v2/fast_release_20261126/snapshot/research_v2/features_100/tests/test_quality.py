"""Synthetic counters/memberships; never accesses Drive, ZIPs or actual games."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from mahjong_research.inputs import digest
from features_100.pipeline import validate_packet,implementation_status
from features_100.quality import aggregate,windows,summarize


class Quality(unittest.TestCase):
    def block(self):
        return pd.DataFrame({'X__status':['ok','no_opportunity','ok'],
                             'X__numerator':[1,0,1],'X__denominator':[1,0,9]})
    def test_pool_not_mean(self):
        value,num,den,status=aggregate(self.block(),'X')
        self.assertEqual((value,num,den,status),(.2,2.,10.,'ok'))
    def test_zero_not_imputed(self):
        b=self.block(); b['X__numerator']=0; b['X__denominator']=0; b['X__status']='no_opportunity'
        self.assertTrue(np.isnan(aggregate(b,'X')[0]))
    def test_unknown_invalidates_window(self):
        b=self.block(); b.loc[0,'X__status']='unknown'
        self.assertTrue(np.isnan(aggregate(b,'X')[0]))
    def test_invalid_counter_fails(self):
        b=self.block(); b.loc[0,'X__numerator']=2
        with self.assertRaises(ValueError): aggregate(b,'X')
    def test_cache_tampering_fails(self):
        payload=dict(identity={'id':'a'},result={})
        packet=dict(signature='v',checksum=digest(payload),payload=payload)
        self.assertEqual(validate_packet(packet,'v',{'id':'a'}),payload)
        packet['payload']['result']['x']=1
        with self.assertRaises(ValueError): validate_packet(packet,'v',{'id':'a'})
    def test_cache_identity_fails(self):
        payload=dict(identity={'id':'a'})
        with self.assertRaises(ValueError): validate_packet(dict(signature='v',checksum=digest(payload),payload=payload),'v',{'id':'b'})
    def test_status100(self):
        f=implementation_status(); self.assertEqual(len(f),100); self.assertTrue(f.feature_id.is_unique)
        self.assertEqual(f.implementation.eq('deferred').sum(),10)
    def inputs(self):
        d=pd.DataFrame(dict(game_id=[f'g{i}' for i in range(40)],hour=pd.date_range('2020-01-01',periods=40,freq='h'),target_label=[0]*20+[1]*20))
        c=pd.DataFrame(dict(game_id=d.game_id,status='ok',X__status='ok',X__numerator=1,X__denominator=2)).set_index('game_id')
        return d,c
    @patch('features_100.quality.IDS',('X',))
    def test_reserved_data_not_summarized(self):
        d,c=self.inputs(); w,m,_=windows(d,c,{'available':{10:[0,10,20,30],20:[0,20]}},d.hour.iloc[29])
        self.assertTrue(pd.to_datetime(w.last_hour).le(d.hour.iloc[29]).all())
        self.assertTrue(m.partition.eq('reserved').any()); self.assertTrue(m.partition.eq('boundary_excluded').any())
    @patch('features_100.quality.IDS',('X',))
    def test_overlapping_games_rejected(self):
        d,c=self.inputs()
        with self.assertRaises(ValueError): windows(d,c,{'available':{10:[0,5],20:[]}},d.hour.iloc[30])
    @patch('features_100.quality.IDS',('X',))
    def test_label_boundary_rejected(self):
        d,c=self.inputs()
        with self.assertRaises(ValueError): windows(d,c,{'available':{10:[15],20:[]}},d.hour.iloc[30])
    @patch('features_100.quality.IDS',('X',))
    def test_hour_cut_rejected(self):
        d,c=self.inputs(); d.loc[10,'hour']=d.hour.iloc[9]
        with self.assertRaises(ValueError): windows(d,c,{'available':{10:[0],20:[]}},d.hour.iloc[30])
    @patch('features_100.quality.IDS',('X',))
    def test_stats_synthetic(self):
        d,c=self.inputs(); c.loc[[f'g{i}' for i in range(20,40)],'X__numerator']=2
        w,_,_=windows(d,c,{'available':{10:[0,10,20,30],20:[0,20]}},d.hour.max())
        st,comp,periods,noise=summarize(w)
        self.assertEqual(comp.difference.iloc[0],.5); self.assertEqual(comp.overlap_20bins.iloc[0],0)
        self.assertTrue(np.isnan(comp.smd.iloc[0])); self.assertEqual(st.valid_window_rate.min(),1)
    def test_no_models_imported(self):
        root=Path(__file__).parents[1]
        for p in root.glob('*.py'):
            text=p.read_text(encoding='utf8')
            self.assertNotIn('import lightgbm',text); self.assertNotIn('.fit(',text)
    @patch('features_100.quality.IDS',('X',))
    @patch('features_100.quality.CATALOG',{'X':{'japanese_name':'Synthetic feature'}})
    def test_report_and_plot_synthetic(self):
        from features_100.quality import render
        d,c=self.inputs()
        w,_,_=windows(d,c,{'available':{10:[0,10,20,30],20:[0,20]}},d.hour.max())
        st,comp,periods,_=summarize(w)
        with tempfile.TemporaryDirectory() as tmp:
            render(Path(tmp),w,comp,st,periods)
            self.assertGreater((Path(tmp)/'figures/X.png').stat().st_size,100)
            self.assertIn('独立', (Path(tmp)/'report_ja.md').read_text(encoding='utf8'))

if __name__=='__main__': unittest.main()
