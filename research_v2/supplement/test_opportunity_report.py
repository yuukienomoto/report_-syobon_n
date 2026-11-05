"""CSV-driven report regressions, all fixtures are synthetic."""
from pathlib import Path
import json,tempfile,unittest
import numpy as np
import pandas as pd
from opportunity_report import updated_thesis_report,read_descriptions
from mahjong_research.inputs import write_json,sha256
from mahjong_research.tenhou import FEATURE_NAMES


class OpportunityReportTests(unittest.TestCase):
    def setup_files(self,root):
        source=root/'C6B';source.mkdir();run=root/'run';run.mkdir()
        stats=[];diffs=[]
        for view,delta in [('opportunity_overlap',.008401),('traditional_overlap',.0695),('opportunity_disjoint',.006),('traditional_disjoint',.07)]:
            for i,f in enumerate(FEATURE_NAMES):
                change=-delta if i==1 and view.startswith('traditional') else delta
                for label in (0,1):stats.append(dict(view=view,feature=f,target_label=label,n=10,valid_n=10,mean=.2+label*change,median=.2+label*change,std=.1,missing_rate=0))
                diffs.append(dict(view=view,feature=f,mean_diff_high_minus_low=change,smd=change/.1,low_valid_n=10,high_valid_n=10))
        pd.DataFrame(stats).to_csv(source/'group_statistics.csv',index=False)
        pd.DataFrame(diffs).to_csv(source/'mean_differences_smd.csv',index=False)
        write_json(source/'analysis_settings.json',dict(primary_windows=20,disjoint_windows=2,max_source_reuse=10,order_verified=False,full_history_confirmed=False))
        w=pd.DataFrame([dict(game_id='a'+str(k),target_label=k//10,source_ids=json.dumps([f'g{k}_{i}' for i in range(10)])) for k in range(20)])
        w.to_csv(source/'opportunity_overlap_windows.csv',index=False)
        metrics=pd.DataFrame([dict(dataset='fold2',estimator='LightGBM',roc_auc=.58,balanced_accuracy=.51,accuracy=.60,f1_strong=.4)])
        for version in ('legacy','v2'):
            folder=run/('model_'+version);folder.mkdir()
            model=folder/'frozen_model.txt';model.write_text('preserve model bytes')
            write_json(folder/'external_freeze.json',dict(files={'frozen_model.txt':sha256(model)}))
            metrics.to_csv(folder/'internal_metrics.csv',index=False)
        w.to_csv(run/'model_v2/final_train_windows.csv',index=False)
        ext=run/'external_v2';ext.mkdir();metrics.assign(dataset='external').to_csv(ext/'external_metrics.csv',index=False)
        write_json(run/'model_v2/external_attempt.json',dict(status='completed',output_dir=str(ext.resolve()),output_sha256={'external_metrics.csv':sha256(ext/'external_metrics.csv')}))
        return source,run

    def test_actual_csv_values_roles_and_sign_reversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,run=self.setup_files(Path(tmp))
            before={p:sha256(p) for p in Path(tmp).rglob('*') if p.is_file()}
            r=updated_thesis_report(source,run)
            text=r['report_path'].read_text(encoding='utf-8')
            main=text.split('## 主分析候補：')[1].split('## 感度分析')[0]
            self.assertIn('+0.008401',main);self.assertNotIn('+0.069500',main)
            sensitivity=text.split('## 感度分析')[1].split('## 補助分析')[0]
            self.assertIn('+0.069500',sensitivity)
            self.assertIn(FEATURE_NAMES[1],r['sign_reversals'])
            self.assertIn('全イベントの単純合算とは推定対象が異なる',text)
            self.assertEqual(len(r['comparison']),20)
            for p,sha in before.items():self.assertEqual(sha256(p),sha)
            copied=pd.read_csv(r['output_dir']/'classification_sources/external_v2/external_metrics.csv')
            original=pd.read_csv(run/'external_v2/external_metrics.csv')
            pd.testing.assert_frame_equal(copied,original)
            again=updated_thesis_report(source,run)
            self.assertNotEqual(r['output_dir'],again['output_dir'])

    def test_inconsistent_stats_and_duplicate_keys_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,run=self.setup_files(Path(tmp))
            p=source/'mean_differences_smd.csv';df=pd.read_csv(p);df.loc[0,'mean_diff_high_minus_low']=.0695;df.to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'不一致'):read_descriptions(source)
            df=pd.concat([df,df.iloc[:1]],ignore_index=True);df.to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'重複'):read_descriptions(source)

    def test_model_tampering_and_population_mismatch_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,run=self.setup_files(Path(tmp))
            p=run/'external_v2/external_metrics.csv';p.write_text('altered')
            with self.assertRaisesRegex(ValueError,'ハッシュ'):updated_thesis_report(source,run)
            pd.DataFrame([dict(game_id='other',target_label=0,source_ids='[]')]).to_csv(run/'model_v2/final_train_windows.csv',index=False)
            with self.assertRaisesRegex(ValueError,'窓集合'):updated_thesis_report(source,run)

    def test_single_group_missing_and_unavailable_legacy_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,run=self.setup_files(Path(tmp))
            (run/'model_legacy/external_freeze.json').unlink()
            p=source/'group_statistics.csv';s=pd.read_csv(p)
            mask=s.feature.eq(FEATURE_NAMES[2])&s.target_label.eq(1)
            s.loc[mask,['valid_n','missing_rate','mean','median','std']]=[0,1,np.nan,np.nan,np.nan];s.to_csv(p,index=False)
            p=source/'mean_differences_smd.csv';d=pd.read_csv(p);mask=d.feature.eq(FEATURE_NAMES[2]);d.loc[mask,['mean_diff_high_minus_low','smd','high_valid_n']]=[np.nan,np.nan,0];d.to_csv(p,index=False)
            r=updated_thesis_report(source,run)
            self.assertIn('算出不可',r['report_path'].read_text(encoding='utf-8'))
            self.assertFalse(r['performance'].version.eq('legacy23_same_protocol').any())


if __name__=='__main__':unittest.main()
