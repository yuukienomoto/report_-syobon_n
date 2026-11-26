import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile
import pandas as pd
from features_100.shapes import shanten,post
from features_100.extract import extract
from features_100.pipeline import extract_run
from .test_core import synthetic_xml,tiles,state
from features_100.options import Options


def xml_for(hand,events):
    return (f'<mjloggm><GO type="9"/><UN n0="target" n1="p1" dan="16,16,16,16"/>'
            f'<INIT seed="0,0,0,1,1,123" ten="250,250,250,250" oya="1" hai0="{",".join(map(str,hand))}"/>'
            +events+'<RYUUKYOKU/></mjloggm>').encode()


class Integration(unittest.TestCase):
    def test_14_shape_equals_best_discard(self):
        rng=random.Random(73)
        for fixed in range(4):
            for _ in range(40):
                h=rng.sample(range(136),14-3*fixed); c=tuple(h.count(t) for t in [])
                from mahjong_research.tenhou import _counts
                c=_counts(h); s=min(shanten(c,fixed))
                if s!=-1:
                    self.assertEqual(s,min(min(shanten(post(c,d),fixed)) for d,n in enumerate(c) if n))
    def test_own_ankan_exclusion(self):
        hand=[0,1,2]+[4*k for k in range(1,11)]
        r=extract(xml_for(hand,'<T3/><N who="0" m="0"/><T100/><D100/>'),'target',game_id='2012052419gm-test')
        self.assertIn('kan_since_previous_own_discard',r['trace'][0]['state']['excluded'])
        self.assertEqual(r['trace'][0]['state']['wall'],68)
    def test_other_kan_exclusion(self):
        hand=tiles([0,1,2,9,10,11,18,19,20,27,27,27,31])
        r=extract(xml_for(hand,'<N who="2" m="33792"/><V130/><F100/><T120/><D120/>'),'target')
        self.assertIn('kan_since_previous_own_discard',r['trace'][0]['state']['excluded'])
    def test_own_pon_immediate_excluded(self):
        hand=[1,2]+[4*k for k in range(2,13)]
        r=extract(xml_for(hand,'<E0/><N who="0" m="105"/><D8/>'),'target',game_id='2012052419gm-test')
        self.assertIn('post_call_no_normal_draw',r['trace'][0]['state']['excluded'])
        self.assertEqual(r['call_observations'][0]['response_status'],'own_call')
    def test_kakan_replaces_pon(self):
        hand=[1,2]+[4*k for k in range(2,13)]
        events='<E0/><N who="0" m="105"/><D8/><T3/><N who="0" m="113"/><T100/><D100/>'
        r=extract(xml_for(hand,events),'target',game_id='2012052419gm-test')
        self.assertEqual(len(r['trace'][-1]['state']['melds']),1)
        self.assertEqual(r['trace'][-1]['state']['melds'][0]['kind'],'kakan')
    def test_public_union_not_double_counts(self):
        s=state(); s['known']=set(s['hand'])|{123,133}; s['known'].update([133,133])
        o=Options(s); self.assertEqual(o.known[33],2)
    def test_declaration_ron_still_observed(self):
        # own hand has white tanki after red-dragon discard; no future acceptance required
        x=synthetic_xml().replace(b'<D132/>',b'<REACH who="0" step="1"/><D132/>').replace(b'<RYUUKYOKU/>',b'<AGARI who="1" fromWho="0"/>')
        r=extract(x,'target',game_id='2012052419gm-test')
        self.assertTrue(r['trace'][0]['state']['declaration'])
    def test_pipeline_resume_and_failed_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); zpath=folder/'synthetic.zip'; xml=synthetic_xml()
            rows=[dict(game_id='2012052419gm-test',file_id='a.mjlog',xml_sha256=hashlib.sha256(xml).hexdigest(),
                       seat=0,dan=16,hour='2012-05-24 19:00:00',player='target')]
            with zipfile.ZipFile(zpath,'w') as z: z.writestr('a.mjlog',xml)
            c=dict(config={'output_root':str(folder/'out'),'player':'target','run_dir':'synthetic-not-Drive'},
                targets=pd.DataFrame(rows),archive=zpath,source={'synthetic':True},signature='synthetic',
                cache=folder/'cache',cutoff='2012-01-01 00:00:00',mode='A')
            a=extract_run(c); c['mode']='B'
            with patch('features_100.pipeline.extract',side_effect=AssertionError('must reuse')):
                b=extract_run(c)
            self.assertNotEqual(a,b)
            self.assertTrue(pd.read_csv(b/'extraction_audit.csv').reused.all())
            cached=next((folder/'cache').glob('*.json')); original=cached.read_bytes()
            c['targets'].loc[0,'dan']=18
            bad=extract_run(c)
            self.assertFalse(json.loads((bad/'complete.json').read_text())['all_reconstructed'])
            self.assertEqual(cached.read_bytes(),original)
            self.assertEqual(len(pd.read_csv(bad/'game_counters.csv')),1)

if __name__=='__main__': unittest.main()
