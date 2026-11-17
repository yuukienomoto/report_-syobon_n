from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from diagnostics import feature_summary
from investigate import run,candidate
from windows import join_windows,split_audit,FEATURES,require
from test_comparison import fixture
import compare


class ErrorAnalysisTests(unittest.TestCase):
    def test_same_marginal_reversed_labels(self):
        g=pd.DataFrame(dict(f=[0.,0.,1.,1.],label=[0,0,1,1],first=[1,2,3,4],last=[1,2,3,4]))
        s,r,_=feature_summary({},g,g.assign(label=1-g.label),['f'])
        self.assertEqual(s[0]['ks'],0)
        self.assertTrue(r[0]['opposite_mean_diff'])
        self.assertEqual(r[0]['train_auc'],1)
        self.assertEqual(r[0]['eval_auc'],0)

    def test_train_only_imputation_and_scaling(self):
        d,*_=fixture()
        train=d.iloc[:450].rename(columns={'target_label':'label'}).assign(first=d.hour,last=d.hour)
        model,_=candidate(train)
        x=train[FEATURES].fillna(train[FEATURES].median()).fillna(0)
        np.testing.assert_allclose(model.steps[1][1].mean_,x.mean())

    def test_full_saved_artifact_replay_and_tamper(self):
        d,cut,starts,records,saved=fixture()
        blocks=join_windows(d,starts,records,saved)
        # Controlled synthetic signal exercises the predeclared train-AUC gate.
        for block in blocks.values():
            block[FEATURES[1]]=block.label*.7+np.arange(len(block))%4*.05
        audit,_=split_audit(d,cut,starts)
        ext={w:b.copy() for w,b in blocks.items()}
        # Non-overlapping synthetic external identifiers for the saved prediction check.
        for w,g in ext.items(): g['window_id']='external-'+g.window_id
        data=(d,starts,blocks,audit,{'raw_sha256':'fixture'})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for f in ('fold_membership.csv','new_game_features.csv','window_values.csv'):
                (root/f).write_text('fixture')
            saved.to_csv(root/'window_values.csv',index=False)
            with patch('compare.prepare',return_value=data),patch('compare.external_data',return_value=(ext,0)):
                p3=compare.run(root,root,root,root)
            with patch('investigate.prepare',return_value=data),patch('investigate.external_data',return_value=(ext,0)):
                out=run(root,root,root,p3,root)
                self.assertTrue((out/'report_ja.md').is_file())
                self.assertEqual(len(pd.read_csv(out/'distribution_shift.csv')),6*23)
                self.assertEqual(len(pd.read_csv(out/'external_shift.csv')),40)
                delta=pd.read_csv(out/'improvement_comparison.csv')
                self.assertTrue(set(delta.dataset)=={'internal','external'})
                path=p3/'predictions.csv'
                p=pd.read_csv(path)
                p.loc[0,'prob_high']=1-p.loc[0,'prob_high']+.01
                p.to_csv(path,index=False)
                with self.assertRaisesRegex(ValueError,'予測'): run(root,root,root,p3,root)
