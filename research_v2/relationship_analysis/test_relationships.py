from pathlib import Path
import tempfile
import unittest
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
from relationships import summary,contrast,checked_scores,run
from diagnostics import feature_summary,score_rows
from test_comparison import fixture
from windows import join_windows,split_audit,FEATURES,NAMES
from mahjong_research.inputs import sha256


class RelationshipTests(unittest.TestCase):
    def test_decomposition_and_fixed_scale(self):
        g=pd.DataFrame(dict(label=[0,0,1,1],x=[0.,2.,2.,4.]))
        h=g.assign(x=[3.,5.,2.,4.])
        r=contrast(g,h,'x',{})
        self.assertAlmostEqual(r['delta_diff'],r['delta_high']-r['delta_low'])
        self.assertAlmostEqual(r['delta_in_train_sd'],-3/np.sqrt(2))
        self.assertTrue(r['substantial_reversal'])
        self.assertFalse(contrast(g.assign(x=[0.,2.,.001,2.001]),g.assign(x=[.001,2.001,0.,2.]),'x',{})['substantial_reversal'])

    def test_missing_single_class_and_constant(self):
        g=pd.DataFrame(dict(label=[0,0,1],x=[1.,1.,np.nan]))
        r=summary(g,'x')
        self.assertTrue(np.isnan(r['auc']) and np.isnan(r['smd']))
        self.assertEqual(r['high_missing'],1)
        self.assertEqual(summary(g.fillna(1),'x')['auc'],.5)

    def test_artifacts(self):
        d,cut,starts,records,saved=fixture()
        blocks=join_windows(d,starts,records,saved)
        audit,_=split_audit(d,cut,starts)
        ext={w:b.copy() for w,b in blocks.items()}
        x,y,z=[],[],[]
        for row in audit[audit.fold.isin([2,3])].itertuples():
            w,fold=row.width,row.fold
            b=blocks[w]
            train=b[b['last'].le(row.train_through)]
            ev=b[b['first'].gt(row.train_through)&b['last'].le(row.eval_through)]
            key=dict(width=w,fold=fold,dataset='internal')
            _,rr,_=feature_summary(key,train,ev,list(FEATURES)+list(NAMES))
            x.extend(rr)
            for model in 'ABCD':
                for role,g in [('train',train),('eval',ev)]:
                    p=np.linspace(0,1,len(g))
                    z.extend(score_rows(dict(key,model=model),role,g,p))
                    if role=='eval':
                        y.extend(dict(key,model=model,window_id=i,label=l,prob_high=prob) for i,l,prob in zip(g.window_id,g.label,p))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            inputs={}
            for name in ['fold_membership.csv','window_values.csv','new_game_features.csv']:
                (path/name).write_text('fixture')
                inputs[name]=sha256(path/name)
            source={'raw_sha256':'fixture'}
            (path/'protocol.json').write_text(json.dumps(dict(source=source,inputs=inputs)))
            (path/'sources.json').write_text(json.dumps(dict(protocol_sha256=sha256(path/'protocol.json'))))
            pd.DataFrame(y).to_csv(path/'predictions.csv',index=False)
            pd.DataFrame(z).to_csv(path/'prediction_diagnostics.csv',index=False)
            pd.DataFrame(x).to_csv(path/'feature_label_relationship.csv',index=False)
            pd.DataFrame(columns=['width','fold','feature']).to_csv(path/'external_shift.csv',index=False)
            pd.DataFrame([dict(width=w,feature=FEATURES[1],model=m,gain_fold2=2,gain_fold3=3,
                          gain_rank_fold2=1,gain_rank_fold3=1,permutation_auc_fold2=.1,permutation_auc_fold3=.1)
                          for w in blocks for m in 'ABCD']).to_csv(path/'feature_importance_comparison.csv',index=False)
            with patch('relationships.prepare',return_value=(d,starts,blocks,audit,source)),patch('relationships.external_data',return_value=(ext,0)):
                out=run(path,path,path,path,path,path)
                self.assertEqual(len(pd.read_csv(out/'feature_relationship_changes.csv')),92)
                self.assertEqual(len(pd.read_csv(out/'external_relationship_comparison.csv')),40)
                self.assertEqual(len(list(out.glob('*.png'))),4)
                self.assertIn('## 8.',(out/'report_ja.md').read_text(encoding='utf8'))
            bad=pd.DataFrame(y[:2]).assign(label=7)
            with self.assertRaises(ValueError):
                checked_scores(ev,bad,pd.DataFrame(z))
