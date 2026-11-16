import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import lightgbm as lgb
from test_smoothing import sample
from windows import (FEATURES,NAMES,validate_data,make_blocks,join_windows,
                     select_before,split_audit,same_table)
from compare import metrics,run
from windows import prepare, external_data, sha256, tenhou


def fixture():
    d=sample(1200)
    d['dan']=np.where(np.arange(len(d))//100%2==0,16,18)
    d,cut=validate_data(d,FEATURES,initial_exclude=0)
    starts=make_blocks(d,cut,set())['available']
    rec=d[['game_id','file_id','xml_sha256','dan','seat','status']].copy()
    for j,f in enumerate(NAMES):
        den=np.arange(len(d))%4
        num=den*((np.arange(len(d))+j)%5==0)
        rec[f+'__numerator']=num
        rec[f+'__denominator']=den
        rec[f]=rec[f+'__numerator']/rec[f+'__denominator'].replace(0,np.nan)
    saved=[]
    for w in (10,20):
        for i in starts[w]:
            for f in NAMES:
                sub=rec.iloc[i:i+w]
                n,k=sub[f+'__numerator'].sum(),sub[f+'__denominator'].sum()
                saved.append(dict(feature=f,width=w,label=d.target_label.iloc[i],
                                  source_ids=json.dumps(sub.game_id.tolist()),numerator=n,denominator=k,value=n/k))
    return d,cut,starts,rec,pd.DataFrame(saved)


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,cls.cut,cls.starts,cls.records,cls.saved=fixture()

    def test_bad_window_and_records(self):
        d,s,r,v=self.data,self.starts,self.records,self.saved
        for broken in [v.iloc[:-1],pd.concat([v,v.iloc[:1]]),v.assign(label=1-v.label),v.assign(value=v.value+.1)]:
            with self.assertRaises(ValueError): join_windows(d,s,r,broken)
        with self.assertRaises(ValueError): join_windows(d,s,r.assign(dan=0),v)
        with self.assertRaises(ValueError): join_windows(d,s,pd.concat([r,r.iloc[:1]]),v)

    def test_selection_is_past_only(self):
        end=self.data.hour.iloc[599]
        a=select_before(self.data,self.starts,end)
        poisoned=self.data.copy()
        poisoned.loc[poisoned.hour.gt(end),FEATURES]=1000
        poisoned.loc[poisoned.hour.gt(end),'target_label']=0
        b=select_before(poisoned,self.starts,end)
        pd.testing.assert_frame_equal(a,b)

    def test_single_class_and_precision(self):
        m=metrics(np.zeros(4),np.zeros(4))
        self.assertTrue(np.isnan(m['roc_auc']) and np.isnan(m['balanced_accuracy']))
        self.assertTrue(np.isnan(m['precision_high']) and np.isnan(m['f1_high']))
        self.assertTrue(m['near_constant'])

    def test_saved_phase1_and_phase2_assembly(self):
        d,cut,s=self.data,self.cut,self.starts
        source=dict(raw_sha256='test',shared_sha256='test',signature='test')
        with tempfile.TemporaryDirectory() as tmp, patch('windows.read_run',return_value=(d,set(),source)), \
                patch('windows.validate_data',return_value=(d,cut)):
            p=Path(tmp)
            (p/'sources.json').write_text(json.dumps(source))
            _,members=split_audit(d,cut,s)
            members.to_csv(p/'fold_membership.csv',index=False)
            self.records.to_csv(p/'new_game_features.csv',index=False)
            self.saved.to_csv(p/'window_values.csv',index=False)
            _,_,blocks,_,_=prepare(p,p,p)
            self.assertEqual(len(blocks[10]),len(s[10]))
            members.loc[0,'role']='evaluation'
            members.to_csv(p/'fold_membership.csv',index=False)
            with self.assertRaises(ValueError): prepare(p,p,p)

    def test_external_shared_games_are_excluded(self):
        d=sample(1650).assign(player='ウルトラ立直')
        shared=d.iloc[[1510]]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            raw,audit=p/'raw_features_v2.csv',p/'extraction_audit.csv'
            d.to_csv(raw,index=False)
            d[['file_id','status']].to_csv(audit,index=False)
            marker=dict(signature='test',rows=len(d),extractor_sha256=sha256(tenhou.__file__),
                        sha256=sha256(raw),audit_sha256=sha256(audit),order_verified=False,full_history_confirmed=False)
            (p/'complete.json').write_text(json.dumps(marker))
            (p/'input_plan.json').write_text(json.dumps(dict(ultra=dict(cache_dir=str(p),signature='test',members=d.file_id.tolist()))))
            ext,n=external_data(p,shared)
            self.assertEqual(n,1)
            for w,g in ext.items():
                self.assertGreater(len(g),0)
                self.assertTrue(all(set(ids).isdisjoint(shared.game_id) and len(ids)==w for ids in g.source_ids))

    def test_train_output_resume_external(self):
        d,s=self.data,self.starts
        blocks=join_windows(d,s,self.records,self.saved)
        audit,_=split_audit(d,self.cut,s)
        self.assertTrue(audit.shared_ids.eq(0).all())
        ext={w:b.copy() for w,b in blocks.items()}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for f in ['fold_membership.csv','new_game_features.csv','window_values.csv']:
                (root/f).write_text('fixture')
            with patch('compare.prepare',return_value=(d,s,blocks,audit,{})),patch('compare.external_data',return_value=(ext,0)):
                out=run(root,root,root,root)
                with patch.object(lgb.LGBMClassifier,'fit',side_effect=AssertionError('model re-fit')):
                    run(root,root,root,root,resume=out)
            m=pd.read_csv(out/'model_metrics.csv')
            self.assertEqual(len(m[m.dataset.eq('internal')]),30)
            self.assertEqual(len(pd.read_csv(out/'model_comparison.csv')),12)
            p=pd.read_csv(out/'predictions.csv').query('dataset=="internal"')
            for _,g in p.groupby(['width','fold']):
                reference=g[g.model.eq('A')]
                for _,other in g.groupby('model'):
                    same_table(reference,other,['window_id'],['label'])
            self.assertTrue((out/'report_ja.md').exists())
            self.assertTrue((out/'confusion_matrix.png').exists())
