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
            with patch('mahjong_research.tenhou.extract_game',return_value=extracted()) as ex:
                d=extract_cached(plan);self.assertEqual(ex.call_count,1)
            with patch('mahjong_research.tenhou.extract_game',side_effect=AssertionError('must reuse')):
                d2=extract_cached(plan)
            pd.testing.assert_frame_equal(d,d2,check_dtype=False)
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

    def test_bad_feature_numerator_denominator_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=extracted();d.update(game_id=NAME.split('&')[0],hour=pd.Timestamp('2020-01-01'))
            d[FEATURE_NAMES[0]+'__numerator']=1
            with self.assertRaisesRegex(ValueError,'分子分母'):audit_v2(pd.DataFrame([d]),FEATURE_NAMES,Path(tmp))


if __name__=='__main__':unittest.main()
