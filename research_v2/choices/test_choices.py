from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import hashlib

import numpy as np
import pandas as pd
from choices import NAMES, extract, tenpai_options
from pipeline import extract_batches, analyze
from mahjong_research.tenhou import extract_game, wait_shapes
from study import FEATURES

TILES=[4*i for i in range(13)]


def synthetic(events,hand=TILES):
    start='<mjloggm><GO type="9"/><UN n0="しょぼんｎ" dan="16,16,16,16"/><INIT ten="250,250,250,250" seed="0,0,0,1,2,132" hai0="'+','.join(map(str,hand))+'"/>'
    return (start+events+'<RYUUKYOKU/></mjloggm>').encode()


REACH='<REACH who="1" step="1"/><E1/><REACH who="1" step="2"/>'


class ChoiceTests(unittest.TestCase):
    def test_mixed_switch_and_discard_draw(self):
        for tile,chosen in [(0,1),(100,0)]:
            r=extract(synthetic(REACH+f'<T100/><D{tile}/>'),0,16)
            for f in NAMES[:2]:
                self.assertEqual((r[f+'__numerator'],r[f+'__denominator']),(chosen,1))

    def test_only_one_kind_of_choice(self):
        for rivers in ['<E133/>',''.join(f'<E{v+1}/>' for v in TILES+[100])]:
            e=rivers+'<REACH who="1" step="2"/><T100/><D100/>'
            r=extract(synthetic(e),0,16)
            for f in NAMES[:2]: self.assertEqual(r[f+'__denominator'],0)

    def test_before_acceptance_multiple_and_own_lock(self):
        for prefix in ['<REACH who="1" step="1"/><E1/>',REACH+'<REACH who="2" step="2"/>',
                       REACH+'<REACH who="0" step="2"/>']:
            r=extract(synthetic(prefix+'<T100/><D0/>'),0,16)
            self.assertEqual(r['free_responses'],0)

    def test_call_consumes_first_and_kan_excluded(self):
        # pon type 0, called tile 0, remove held copies 1 and 2.
        hand=[1,2]+TILES[2:]
        r=extract(synthetic(REACH.replace('E1','E0')+'<N who="0" m="105"/><D8/><T100/><D100/>',hand),0,16)
        self.assertEqual(r['free_responses'],1)
        self.assertEqual(r[NAMES[0]+'__denominator'],0)
        trace=[]
        extract(synthetic(REACH+'<N who="2" m="31744"/><T100/><D100/><T101/><D101/>'),0,16,trace)
        self.assertEqual(len(trace),1)
        self.assertFalse(trace[0]['first'])

    def test_impossible_discard(self):
        with self.assertRaisesRegex(ValueError,'mismatch'):
            extract(synthetic(REACH+'<T100/><D135/>'),0,16)

    def test_structural_options_kokushi(self):
        kinds=[0,8,9,17,18,26,27,28,29,30,31,32,33]
        hand=[k*4 for k in kinds]
        options=tenpai_options(hand+[12],[])
        self.assertEqual({k for k,v in options.items() if v},{3})
        self.assertEqual(set(wait_shapes(hand)),set(kinds))

    def test_cache_resume_analysis_and_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            xml=synthetic(REACH+'<T100/><D0/>')
            old=extract_game(xml,'しょぼんｎ')
            sample=pd.DataFrame([dict(old,game_id=f'g{i}',file_id=f'{i}.xml',dan=16,
                target_label=int(i>=60),hour=pd.Timestamp('2012-01-01')+pd.Timedelta(hours=i),
                xml_sha256=hashlib.sha256(xml).hexdigest()) for i in range(120)])
            ctx=dict(data=sample,starts={1:list(range(120)),10:list(range(0,120,10)),20:list(range(0,120,20))},
                     batches=[sample.iloc[:2],sample.iloc[2:]],out=root,archive=root/'test.zip',cache=root,key='test')
            with zipfile.ZipFile(ctx['archive'],'w') as z:
                for name in sample.file_id: z.writestr(name,xml)
            extract_batches(ctx,True)
            records=extract_batches(ctx)
            with patch('pipeline.extract',side_effect=AssertionError('cache replay')):
                self.assertEqual(len(extract_batches(ctx)),120)
            # Test absence of opportunities independently of the real log fixture.
            for s in ('__numerator','__denominator'): records[NAMES[0]+s]=0
            result=analyze(ctx,records)
            self.assertEqual(len(result),6)
            self.assertTrue(result[result.feature.eq(NAMES[0])].decision.eq('判断困難').all())
            self.assertEqual(len(pd.read_csv(root/'correlations.csv')),360)
            self.assertTrue((root/'distributions.png').exists())
            with self.assertRaises(ValueError): analyze(ctx,records.iloc[:-1])
