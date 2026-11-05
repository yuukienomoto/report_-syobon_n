"""Input/cache tests use temporary synthetic archives; no study data is altered."""
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from mahjong_research.inputs import plan_extraction, extract_cached, audit_v2, sha256
from mahjong_research.analysis import _data_digest
from mahjong_research.tenhou import FEATURE_NAMES, FEATURE_DEFINITIONS

NAME='2020010100gm-0009-0000-abcdef01&tw=0.mjlog'
XML=b'<mjloggm><UN n0="test" n1="other" n2="x" n3="y"/></mjloggm>'


def extracted():
    d={'player':'test','dan':16,'seat':0,'go_type':9}
    for f in FEATURE_NAMES:
        d[f]=0.0;d[f+'__numerator']=0;d[f+'__denominator']=1
    return d


class InputTests(unittest.TestCase):
    def make_plan(self,root,xml=XML):
        z=root/'original.zip'
        with zipfile.ZipFile(z,'w') as archive:archive.writestr(NAME,xml)
        return plan_extraction(z,'test','test',root/'cache')

    def test_cache_reuse_preserves_source_and_does_not_extract_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=self.make_plan(root);before=sha256(plan['zip_path'])
            out=extracted(); f=FEATURE_NAMES[0]
            out[f]=1/7;out[f+'__numerator']=1;out[f+'__denominator']=7
            with patch('mahjong_research.tenhou.extract_game',return_value=out) as ex:
                d=extract_cached(plan);self.assertEqual(ex.call_count,1)
            with patch('mahjong_research.tenhou.extract_game',side_effect=AssertionError('must reuse')), \
                 patch('mahjong_research.inputs.write_csv',side_effect=AssertionError('completed cache rewritten')), \
                 patch('mahjong_research.inputs.write_json',side_effect=AssertionError('completed marker rewritten')):
                d2=extract_cached(plan)
            pd.testing.assert_frame_equal(d,d2,check_exact=True)
            self.assertEqual(d2.at[0,f].hex(), (1/7).hex())
            cols=['game_id','hour','dan']+FEATURE_NAMES
            self.assertEqual(_data_digest(d,cols),_data_digest(d2,cols))
            self.assertEqual(sha256(plan['zip_path']),before)
            facts=json.loads((Path(plan['cache_dir'])/'complete.json').read_text())
            self.assertFalse(facts['order_verified']);self.assertFalse(facts['full_history_confirmed'])
            audit_v2(d,FEATURE_NAMES,root/'audit')

    def test_extraction_failure_is_retained_and_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=self.make_plan(Path(tmp))
            with patch('mahjong_research.tenhou.extract_game',side_effect=ValueError('broken hand')):
                with self.assertRaisesRegex(ValueError,'解析失敗'):extract_cached(plan)
            audit=pd.read_csv(Path(plan['cache_dir'])/'extraction_audit.csv')
            self.assertEqual(len(audit),1);self.assertEqual(audit.status.iloc[0],'error')
            self.assertFalse((Path(plan['cache_dir'])/'complete.json').exists())

    def test_changed_archive_after_preview_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=self.make_plan(root)
            with zipfile.ZipFile(plan['zip_path'],'a') as archive:archive.writestr('extra.mjlog',XML)
            with self.assertRaisesRegex(ValueError,'プレビュー後'):extract_cached(plan)

    def test_tampered_cache_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=self.make_plan(root)
            with patch('mahjong_research.tenhou.extract_game',return_value=extracted()):extract_cached(plan)
            csv=next(Path(plan['cache_dir']).glob('batch_*.csv'));csv.write_text('corrupted')
            with self.assertRaisesRegex(ValueError,'破損'):extract_cached(plan)

    def test_partial_extraction_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=self.make_plan(Path(tmp))
            self.assertIsNone(extract_cached(plan,max_new=0))
            self.assertFalse((Path(plan['cache_dir'])/'complete.json').exists())

    def test_mixed_batches_resume_matches_single_run_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); z=root/'mixed.zip'
            with zipfile.ZipFile(z,'w') as archive:
                for i, xml in enumerate((XML, XML.replace(b'test',b'unknown'), XML.replace(b'<UN', b'<UN variant="b"'))):
                    archive.writestr(NAME.replace('0100gm', f'010{i}gm'),xml)
            full=plan_extraction(z,'test','test',root/'full')
            partial=plan_extraction(z,'test','test',root/'partials')
            def sample(xml, player):
                second=b'variant="b"' in xml; d=extracted(); denom=7 if second else 3; f=FEATURE_NAMES[0]
                d[f]=1/denom; d[f+'__numerator']=1; d[f+'__denominator']=denom
                d['audit_beta' if second else 'audit_alpha']=1
                return d
            with patch('mahjong_research.tenhou.extract_game',side_effect=sample):
                single=extract_cached(full,batch_size=3)
                self.assertIsNone(extract_cached(partial,max_new=2,batch_size=1))
                resumed=extract_cached(partial,batch_size=2)
                cached=extract_cached(partial)
            for replay in (resumed,cached):
                pd.testing.assert_frame_equal(single,replay,check_exact=True)
            for filename in ('raw_features_v2.csv','extraction_audit.csv'):
                self.assertEqual(sha256(Path(full['cache_dir'])/filename),sha256(Path(partial['cache_dir'])/filename))

    def test_completed_raw_and_missing_signed_batch_fail_closed(self):
        for mode in ('modified_raw','missing_batch','modified_audit'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                plan=self.make_plan(Path(tmp)); folder=Path(plan['cache_dir'])
                with patch('mahjong_research.tenhou.extract_game',return_value=extracted()):
                    extract_cached(plan)
                if mode=='missing_batch':
                    next(folder.glob('batch_*.json')).unlink()
                else:
                    (folder/('raw_features_v2.csv' if mode=='modified_raw' else 'extraction_audit.csv')).write_text('changed')
                with self.assertRaisesRegex(ValueError,'完了キャッシュ'):
                    extract_cached(plan)

    def test_bad_feature_numerator_denominator_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=extracted();d.update(game_id=NAME.split('&')[0],hour=pd.Timestamp('2020-01-01'))
            d[FEATURE_NAMES[0]+'__numerator']=1
            with self.assertRaisesRegex(ValueError,'分子分母'):audit_v2(pd.DataFrame([d]),FEATURE_NAMES,Path(tmp))


if __name__=='__main__':unittest.main()
