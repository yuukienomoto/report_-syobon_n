import unittest
import tempfile,json,zipfile,hashlib
from unittest.mock import patch
from pathlib import Path
from mahjong_research.tenhou import extract_game,wait_shapes,decode_meld
from mahjong_research.inputs import plan_extraction,extract_cached,digest,write_csv,write_json,sha256
from mahjong_research.recovery import recover_successful_batches,PREVIOUS_EXTRACTOR
import pandas as pd


class DisclosureRegression(unittest.TestCase):
    def test_actual_20120524_east4_disclosed_own_four_tile_wait(self):
        xml=(Path(__file__).parent/'fixtures/disclosed_own_four_tiles.xml').read_bytes()
        d=extract_game(xml,'しょぼんｎ')
        self.assertEqual(d['seat'],2)
        self.assertEqual(d['audit_draw_disclosed_own_tile_exhausted_wait'],1)
        self.assertEqual(d['audit_discard_hand_checks'],19)
        self.assertEqual(d['exhaustive_draw_disclosed_tenpai_rate__numerator'],1)
        self.assertEqual(d['exhaustive_draw_disclosed_tenpai_rate__denominator'],1)
        self.assertEqual(d['exhaustive_draw_disclosed_tenpai_rate'],1)

    def test_signed_recovery_reuses_success_and_retries_only_failed_game(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            name='2012052419gm-0029-0000-eb0d38e9&tw=2.mjlog'
            good='2012052418gm-0029-0000-abcdef01&tw=2.mjlog'
            bad_xml=(Path(__file__).parent/'fixtures/disclosed_own_four_tiles.xml').read_bytes()
            import xml.etree.ElementTree as ET
            tree=ET.fromstring(bad_xml)
            nodes=list(tree)
            init=next(n for n in nodes if n.tag=='INIT')
            for node in nodes:
                if node.tag not in ('GO','UN','INIT'):tree.remove(node)
            ET.SubElement(tree,'RYUUKYOKU')
            good_xml=ET.tostring(tree,encoding='utf-8')
            z=root/'games.zip'
            with zipfile.ZipFile(z,'w') as archive:
                archive.writestr(good,good_xml);archive.writestr(name,bad_xml)
            plan=plan_extraction(z,'しょぼんｎ','shobon',root/'cache')
            with zipfile.ZipFile(z) as archive:
                items=sorted((i.filename,i.CRC,i.file_size,i.compress_size) for i in archive.infolist())
            oldsig=digest(dict(schema='tenhou-research-v2',extractor_sha256=PREVIOUS_EXTRACTOR,player='しょぼんｎ',members=items,target_members=plan['members']))
            old=root/'old';old.mkdir()
            success=dict(extract_game(good_xml,'しょぼんｎ'),file_id=good,game_id=good.split('&')[0],status='ok',error='',xml_sha256=hashlib.sha256(good_xml).hexdigest())
            fail=dict(file_id=name,game_id=name.split('&')[0],status='error',error='ExtractionError: disclosed exhaustive hand lacks structural wait')
            write_csv(old/'batch_one.csv',pd.DataFrame([success,fail]))
            write_json(old/'batch_one.json',dict(signature=oldsig,csv='batch_one.csv',sha256=sha256(old/'batch_one.csv')))
            before=sha256(old/'batch_one.csv')
            next_plan=recover_successful_batches(plan,old)
            self.assertEqual(next_plan['remaining'],1)
            with patch('mahjong_research.tenhou.extract_game',wraps=extract_game) as calls:
                result=extract_cached(next_plan)
                self.assertEqual(calls.call_count,1)
            self.assertEqual(len(result),2)
            self.assertEqual(sha256(old/'batch_one.csv'),before)
            replay=recover_successful_batches(plan,old)
            self.assertEqual(replay['remaining'],0)
            meta=json.loads((old/'batch_one.json').read_text())
            meta['signature']='incorrect';write_json(old/'batch_one.json',meta)
            with self.assertRaisesRegex(ValueError,'signature'):recover_successful_batches(plan,old)


if __name__=='__main__':unittest.main()
