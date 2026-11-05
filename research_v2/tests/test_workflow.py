"""Synthetic orchestration/report checks, never study results."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from mahjong_research.tenhou import FEATURE_NAMES
from mahjong_research.legacy import BASE_FEATURES
from mahjong_research.workflow import descriptive_stage, final_report, model_stage


class WorkflowTests(unittest.TestCase):
    def data(self):
        n=1580
        d=pd.DataFrame({'game_id':['g'+str(i) for i in range(n)],'hour':pd.date_range('2020-01-01',periods=n,freq='h'),
                        'dan':np.where((np.arange(n)//3)%2,18,16),'player':'shobon'})
        for j,f in enumerate(FEATURE_NAMES):
            d[f]=(np.arange(n)%7)/10+j/100
            d[f+'__numerator']=d[f]*100;d[f+'__denominator']=100
        old=d[['game_id','hour','dan','player']].copy()
        for j,f in enumerate(BASE_FEATURES):old[f]=(np.arange(n)%9)/10
        return old,d

    def test_descriptive_comparison_new_features_and_japanese_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx={'folder':Path(tmp),'code_root':Path(__file__).resolve().parents[1]}
            old,new=self.data()
            a,b,out=descriptive_stage(ctx,old,new,set(),make_plots=False)
            table=pd.read_csv(out/'legacy_v2_descriptive_comparison.csv')
            newly_added=table[table.legacy_feature.isna()]
            self.assertTrue(newly_added.mean_diff_high_minus_low_v2.notna().all())
            self.assertEqual(set(newly_added.view),{'raw_all_eligible','raw_same_endpoints','primary_hour_invariant','primary_disjoint_windows'})
            performance,path=final_report(ctx,b)
            self.assertFalse(performance.status.eq('computed_current_run').any())
            report=path.read_text(encoding='utf-8')
            self.assertIn('まだ実行されていない',report)
            self.assertTrue((out/'opportunities_by_rank.csv').exists())
            statuses=json.loads((path.parent/'execution_status.json').read_text(encoding='utf-8'))
            self.assertFalse(statuses['new_external_executed'])
            self.assertFalse(statuses['order_verified'])

    def test_resume_does_not_train_and_rejects_changed_input(self):
        from mahjong_research.analysis import validate_data
        with tempfile.TemporaryDirectory() as tmp:
            ctx={'folder':Path(tmp)}; _,d=self.data()
            normalized,_=validate_data(d,FEATURE_NAMES)
            digest=hashlib.sha256(normalized[['game_id','hour','dan']+FEATURE_NAMES].to_csv(index=False).encode()).hexdigest()
            out=ctx['folder']/'model_v2';out.mkdir();(out/'external_freeze.json').write_text('{}')
            bundle={'protocol':{'data_sha256':digest,'external_forbidden_game_ids':[]}}
            with patch('mahjong_research.workflow.load_frozen_bundle',return_value=bundle),patch('mahjong_research.workflow.run_internal',side_effect=AssertionError('must not train')):
                self.assertIs(model_stage(ctx,d,set(),'v2'),bundle)
                changed=d.copy();changed.loc[1520,FEATURE_NAMES[0]]+=.01
                with self.assertRaisesRegex(ValueError,'不一致'):model_stage(ctx,changed,set(),'v2')

    def test_incomplete_model_folder_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx={'folder':Path(tmp)};_,d=self.data()
            out=ctx['folder']/'model_v2';out.mkdir();(out/'partial.txt').write_text('preserve')
            with self.assertRaisesRegex(ValueError,'部分出力'):model_stage(ctx,d,set(),'v2')
            self.assertEqual((out/'partial.txt').read_text(),'preserve')


if __name__=='__main__':unittest.main()
