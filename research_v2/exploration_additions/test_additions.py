import unittest
import tempfile,json,zipfile,hashlib
from pathlib import Path
import pandas as pd
from unittest.mock import patch
from core import extract,measure,SelectedOptions,PARENTS,DEFINITIONS
from features_100.extract import extract as baseline
from features_100.tests.test_core import synthetic_xml,state
class Tests(unittest.TestCase):
 def test_parent_equivalence(self):
  x=synthetic_xml();a=extract(x,'target',game_id='2012052419gm-test');b=baseline(x,'target',game_id='2012052419gm-test')
  for f in PARENTS:self.assertEqual(a['features'][f],b['features'][f])
 def test_hidden_unchanged(self):
  a=extract(synthetic_xml(),'target');b=extract(synthetic_xml(hidden='100,101,102',draw='99'),'target')
  self.assertEqual(a['features'],b['features'])
 def test_terminal_unchanged(self):
  self.assertEqual(extract(synthetic_xml(),'target')['features'],extract(synthetic_xml(ending='<RYUUKYOKU hai1="1,2,3"/>'),'target')['features'])
 def test_no_calls_or_z(self):
  with patch('features_100.calls.measure_call',side_effect=AssertionError('call')):
   r=extract(synthetic_xml(),'target');self.assertEqual(len(r['features']),14)
 def test_children_subset(self):
  o=SelectedOptions(state())
  for f,(_,parent,_) in DEFINITIONS.items():
   if parent:self.assertLessEqual(measure(f,o)['denominator'],measure(parent,o)['denominator'])
 def test_late_filter(self):
  o=SelectedOptions(state(wall=40));self.assertEqual(measure('P01_late',o)['denominator'],0)
 def test_dealer_filter(self):
  self.assertEqual(measure('P01_dealer',SelectedOptions(state(dealer=1)))['denominator'],0)
 def test_excluded_quantity(self):
  self.assertEqual(measure('safe_shape_cost_mean',SelectedOptions(state(free=False)))['status'],'excluded')
 def test_no_threat_cost(self):
  self.assertEqual(measure('safe_shape_cost_mean',SelectedOptions(state(active=set())))['denominator'],0)
 def test_zero_den_nan(self):
  r=extract(synthetic_xml(),'target');v=r['features']['Q01_closed'];self.assertIsNone(v['value'])
 def test_hand_computed_cost(self):
  o=SelectedOptions(state());o.__dict__.update(G={132},S={a:(2 if a==132 else 0) for a in o.D},Smin=0)
  v=measure('safe_shape_cost_mean',o);self.assertEqual((v['numerator'],v['denominator']),(2,1))
 def test_hand_computed_loss(self):
  o=SelectedOptions(state(active=set()));o.__dict__.update(Dm={124,132},U={124:8,132:3})
  v=measure('ukeire_loss_mean',o);self.assertEqual((v['numerator'],v['denominator']),(5,1))
 def test_pipeline_cache_and_parent_guard(self):
  from workflow import run
  x=synthetic_xml();gid='2012052419gm-test';b=baseline(x,'target',game_id=gid)
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);archive=root/'logs.zip'
   with zipfile.ZipFile(archive,'w') as z:z.writestr('a.mjlog',x)
   d=pd.DataFrame([dict(game_id=gid,file_id='a.mjlog',xml_sha256=hashlib.sha256(x).hexdigest(),seat=0,dan=16,hour=pd.Timestamp('2012-05-24 19:00:00'),target_label=0)])
   old=pd.DataFrame([dict(game_id=gid,**{f+'__'+k:v for f in PARENTS for k,v in b['features'][f].items()})]).set_index('game_id')
   ctx=(d,None,set(),d,old,{'zip_path':str(archive)},'test-signature')
   cfg=dict(output_root=str(root/'out'),player='target')
   with patch('workflow.prepare',return_value=ctx):
    a=run(cfg,'full')
    with patch('workflow.extract',side_effect=AssertionError('cache must reuse')):bdir=run(cfg,'full')
    self.assertTrue(pd.read_csv(bdir/'audit.csv').reused.all())
    self.assertEqual(json.loads((a/'complete.json').read_text(encoding='utf8'))['rows'],1)
    old.loc[gid,'P01__denominator']+=1
    with self.assertRaisesRegex(ValueError,'親カウンタ'):run(cfg,'full')
if __name__=='__main__':unittest.main()
