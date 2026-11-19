from pathlib import Path
import copy
import gzip
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from choices import extract as old_choices, tenpai_options
from mahjong_research.tenhou import extract_game, decode_meld, wait_shapes
from mahjong_research.inputs import digest, sha256, write_csv, write_json
from decision_features.extract import extract, measure
from decision_features.definitions import NAMES
from decision_features.pipeline import run_pilot, load_c5, prepare, select_pilot, checked_xml
from decision_features.analysis import window_audit

ROOT = Path(__file__).resolve().parents[2]
KOKUSHI = [4*k for k in (0,8,9,17,18,26,27,28,29,30,31,32,33)]
HAND = KOKUSHI+[12]
A1,A4,E3,E4 = NAMES


def xml(events, hand=KOKUSHI, player='target'):
    return ('<mjloggm><GO type="9"/><UN n0="'+player+'" n1="p1" n2="p2" n3="p3" dan="16,16,16,16"/>'
            '<INIT seed="0,0,0,0,0,120" ten="250,250,250,250" hai0="'+','.join(map(str,hand))+'"/>'
            +events+'<RYUUKYOKU/></mjloggm>').encode()


def reach(who, tile):
    return f'<REACH who="{who}" step="1"/><{"DEFG"[who]}{tile}/><REACH who="{who}" step="2"/>'


R = reach(1,1)


class OptionsTests(unittest.TestCase):
    def result(self, f, hand=HAND, rivers=None, opponents={1}, discard=12):
        return measure(hand, [], rivers or [set(),{0},set(),set()], opponents, discard)[0][f]

    def test_a1_non_genbutsu_with_tenpai_option(self):
        self.assertEqual(self.result(A1), dict(numerator=1,denominator=1,reason='included'))

    def test_a1_actual_noten_is_still_counted(self):
        self.assertEqual(self.result(A1,discard=32)['numerator'],1)
        self.assertFalse(tenpai_options(HAND,[])[8])

    def test_a4_safe_sacrifice(self):
        self.assertEqual(self.result(A4,discard=0), dict(numerator=1,denominator=1,reason='included'))

    def test_a4_third_choice_is_not_counted_as_sacrifice(self):
        self.assertEqual((self.result(A4,discard=32)['numerator'],self.result(A4,discard=32)['denominator']),(0,1))

    def test_a4_safe_and_tenpai_compatible_excluded(self):
        self.assertEqual(self.result(A4,rivers=[set(),{3},set(),set()])['denominator'],0)

    def test_only_safe_or_only_unsafe(self):
        for river in (set(),{t//4 for t in HAND}):
            for f in (A1,A4):
                self.assertEqual(self.result(f,rivers=[set(),river,set(),set()])['denominator'],0)

    def test_no_tenpai_option(self):
        hand=[4*k for k in (0,2,4,6,8,9,11,13,15,17,18,20,22,24)]
        self.assertFalse(any(tenpai_options(hand,[]).values()))
        for f in (A1,A4): self.assertEqual(self.result(f,hand=hand,discard=0)['denominator'],0)

    def test_suji_mixed_and_chosen(self):
        hand=[4*k for k in (0,1,2,6,7,8,9,10,11,15,16,17,18,19)]
        r=self.result(E3,hand=hand,rivers=[set(),{3},set(),set()],discard=0)
        self.assertEqual((r['numerator'],r['denominator']),(1,1))
        self.assertEqual(self.result(E3,hand=hand,rivers=[set(),{3},set(),set()],discard=4)['numerator'],0)

    def test_suji_middle_requires_both_anchors(self):
        from mahjong_research.tenhou import is_suji
        for k in (3,4,5):
            self.assertFalse(is_suji(k,{k-3}))
            self.assertTrue(is_suji(k,{k-3,k+3}))
        self.assertFalse(is_suji(27,{24,30}))

    def test_honors_excluded_from_e3(self):
        r=self.result(E3,rivers=[set(),{4},set(),set()])
        self.assertIn('honor_candidate',r['reason'])
        self.assertEqual(r['denominator'],0)

    def test_suji_only_and_no_suji_excluded(self):
        hand=list(range(12))+[36,37]
        for river in ({3,4,5,12}, {20}):
            self.assertEqual(self.result(E3,hand=hand,rivers=[set(),river,set(),set()],discard=0)['denominator'],0)

    def test_e4_all_opponents_once(self):
        for opponents in ({1,2},{1,2,3}):
            r=self.result(E4,rivers=[set(),{0},{0},{0}],opponents=opponents,discard=0)
            self.assertEqual((r['numerator'],r['denominator']),(1,1))

    def test_e4_safe_for_only_one_is_not_common(self):
        r=self.result(E4,rivers=[set(),{0,8},{0},set()],opponents={1,2},discard=32)
        self.assertEqual((r['numerator'],r['denominator']),(0,1))
        self.assertEqual(self.result(E4,rivers=[set(),{0},{8},set()],opponents={1,2})['denominator'],0)

    def test_e4_all_safe_excluded(self):
        river={t//4 for t in HAND}
        self.assertEqual(self.result(E4,rivers=[set(),river,river,set()],opponents={1,2})['denominator'],0)

    def test_special_shapes_and_open_hand(self):
        self.assertEqual({k for k,v in tenpai_options(HAND,[]).items() if v},{3})
        pairs=[0,1,8,9,16,17,36,37,44,45,72,73,108]
        self.assertIn('chiitoitsu',wait_shapes(pairs)[27])
        meld=decode_meld(105,0)
        hand=[12,16,20,36,40,44,72,76,80,108]
        self.assertTrue(wait_shapes(hand,[meld]))


class ReplayTests(unittest.TestCase):
    def test_pre_discard_hand_and_counts(self):
        r=extract(xml(R+'<T12/><D0/>'),'target')
        self.assertEqual((r['counts'][A1+'__numerator'],r['counts'][A4+'__numerator']),(0,1))
        self.assertEqual(r['trace'][0]['hand'],HAND)

    def test_before_acceptance_excluded(self):
        r=extract(xml('<REACH who="1" step="1"/><E1/><T12/><D0/><REACH who="1" step="2"/>'),'target')
        self.assertEqual(r['counts'][A1+'__denominator'],0)

    def test_multiple_reach_not_pair_count(self):
        r=extract(xml(R+reach(2,2)+'<T12/><D0/>'),'target')
        self.assertEqual((r['counts'][E4+'__numerator'],r['counts'][E4+'__denominator']),(1,1))
        self.assertEqual(r['counts'][A1+'__denominator'],0)

    def test_own_declaration_included_then_lock(self):
        r=extract(xml(R+'<T12/><REACH who="0" step="1"/><D12/><REACH who="0" step="2"/><T13/><D13/>'),'target')
        self.assertEqual(r['counts'][A1+'__denominator'],1)
        self.assertIn('own_accepted_riichi',r['trace'][1]['excluded'])

    def test_call_immediate_discard_excluded(self):
        hand=[1,2]+[4*k for k in range(2,13)]
        r=extract(xml(reach(1,0)+'<N who="0" m="105"/><D8/>',hand),'target')
        self.assertIn('no_normal_draw_post_call',r['trace'][0]['excluded'])
        self.assertEqual(sum(r['counts'].values()),0)

    def test_opponent_kan_excludes_only_until_own_discard(self):
        r=extract(xml(R+'<N who="2" m="31744"/><T12/><D12/><T13/><D13/>'),'target')
        self.assertIn('kan_since_previous_own_discard',r['trace'][0]['excluded'])
        self.assertEqual(r['counts'][A1+'__denominator'],1)

    def test_own_ankan_rinshan_excluded(self):
        hand=[0,1,2]+[4*k for k in range(1,11)]
        r=extract(xml(reach(1,45)+'<T3/><N who="0" m="0"/><T100/><D100/>',hand),'target')
        self.assertIn('kan_since_previous_own_discard',r['trace'][0]['excluded'])

    def test_own_added_kan_replaces_pon(self):
        hand=[1,2]+[4*k for k in range(2,13)]
        r=extract(xml(reach(1,0)+'<N who="0" m="105"/><D8/><T3/><N who="0" m="113"/><T100/><D100/>',hand),'target')
        self.assertEqual(len(r['trace'][1]['melds']),1)
        self.assertEqual(r['trace'][1]['melds'][0]['kind'],'kakan')
        self.assertIn('kan_since_previous_own_discard',r['trace'][1]['excluded'])

    def test_own_open_kan_rinshan_excluded(self):
        hand=[1,2,3]+[4*k for k in range(2,12)]
        r=extract(xml(reach(1,0)+'<N who="0" m="1"/><T100/><D100/>',hand),'target')
        self.assertEqual(r['trace'][0]['melds'][0]['kind'],'daiminkan')
        self.assertIn('kan_since_previous_own_discard',r['trace'][0]['excluded'])

    def test_malformed_identity_hand_acceptance_and_reinit_fail(self):
        bad=['<T12/><D135/>','<REACH who="1" step="2"/><T12/><D12/>','<REINIT/>','<T0/><D0/>']
        for events in bad:
            with self.assertRaises(ValueError): extract(xml(events),'target')
        with self.assertRaises(ValueError): extract(xml('<T12/><D12/>'),'other')

    def test_future_public_and_private_information_invariance(self):
        base=xml(R+'<T12/><D0/>')
        original=extract(base,'target')['trace'][0]
        root=ET.fromstring(base)
        init=root.find('INIT')
        init.set('hai1','133,134,135')  # Not observable to target; intentionally unused.
        root.insert(list(root).index(init)+1,ET.Element('U134'))
        # Opponent hidden draw changes event indexing, not the decision state.
        variant=extract(ET.tostring(root),'target')['trace'][0]
        self.assertEqual({k:v for k,v in original.items() if k!='event'},
                         {k:v for k,v in variant.items() if k!='event'})
        root2=ET.fromstring(base)
        for tag,attrs in [('E121',{}),('DORA',{'hai':'135'})]:
            root2.insert(len(root2)-1,ET.Element(tag,attrs))
        self.assertEqual(extract(ET.tostring(root2),'target')['trace'][0],original)

    def test_hidden_draw_content_changes_neither_counters_nor_trace(self):
        a=xml(R+'<U42/><T12/><D0/>')
        b=a.replace(b'U42',b'U130')
        self.assertEqual(extract(a,'target'),extract(b,'target'))

    def test_real_rounds_legacy20_and3_unchanged(self):
        data=(ROOT/'choices/real_rounds.xml').read_bytes()
        before20=extract_game(data,'しょぼんｎ')
        before3=old_choices(data,2,14)
        result=extract(data,'しょぼんｎ',2,14)
        self.assertEqual(result['audit']['own_discards'],before20['audit_discard_hand_checks'])
        self.assertEqual(before20,extract_game(data,'しょぼんｎ'))
        self.assertEqual(before3,old_choices(data,2,14))
        self.assertEqual(result['counts'][A1+'__denominator'],0)
        decision=next(e for e in result['trace'] if e['tenpai'])
        self.assertEqual(decision['tenpai'],[90])
        self.assertEqual(decision['genbutsu'],[])  # structural option alone is insufficient

    def test_real_round6_hand_checked_four_decisions(self):
        data=(Path(__file__).parent/'fixtures/real_round6.xml').read_bytes()
        before20=extract_game(data,'しょぼんｎ');before3=old_choices(data,2,14)
        r=extract(data,'しょぼんｎ',2,14)
        selected=[e for e in r['trace'] if e['features'][A1]['denominator']]
        # Pure 2345s plus drawn tile, with three open melds. The 4p pon
        # exhausts a hypothetical fifth own 4p: legacy wait_shapes excludes it.
        self.assertEqual([e['discard'] for e in selected],[29,123,77,130])
        self.assertEqual([e['genbutsu'] for e in selected],[[77,81],[77,81,123],[77,81],[81]])
        self.assertEqual([e['tenpai'] for e in selected],[[29,77,89],[77,89,123],[50],[50]])
        self.assertEqual([e['features'][A4]['denominator'] for e in selected],[0,0,1,1])
        self.assertEqual([e['features'][A4]['numerator'] for e in selected],[0,0,1,0])
        self.assertEqual(r['counts'][A1+'__numerator'],2)
        self.assertEqual(r['counts'][A1+'__denominator'],4)
        self.assertEqual(before20,extract_game(data,'しょぼんｎ'))
        self.assertEqual(before3,old_choices(data,2,14))


class PipelineTests(unittest.TestCase):
    def context(self, root, count=20):
        data=xml(R+'<T12/><D0/>')
        old=extract_game(data,'target')
        rows=[]
        for i in range(count):
            hour=pd.Timestamp('2012-01-01')+pd.Timedelta(hours=i)
            game=hour.strftime('%Y%m%d%H')+'gm-0009-0000-abcdef01'
            rows.append(dict(old,game_id=game,file_id=game+'.mjlog',hour=hour,target_label=0.,
                             xml_sha256=hashlib.sha256(data).hexdigest(),status='ok',error=''))
        frame=pd.DataFrame(rows)
        archive=root/'input.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for name in frame.file_id: z.writestr(name,gzip.compress(data))
        return dict(data=frame,targets=frame.copy(),starts={10:list(range(0,count-9,10)),20:list(range(0,count-19,20))},
            archive=archive,cache=root/'new_cache',signature='synthetic-test',source={},output_root=root/'reports',
            cutoff='test fixture only',player='target')

    def test_resume_zero_denominator_and_window_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=self.context(Path(tmp))
            first=run_pilot(ctx)
            frame=pd.read_csv(first/'game_features.csv')
            self.assertTrue(frame[E4].isna().all())
            windows,_,coverage,_=window_audit(ctx,frame)
            self.assertEqual(coverage.complete_pilot_windows.tolist(),[2,1])
            self.assertEqual(windows[windows.feature.eq(A4)].denominator.tolist(),[10,10,20])
            hashes={p.name:sha256(p) for p in ctx['cache'].glob('*.json')}
            with patch('decision_features.pipeline.extractor.extract',side_effect=AssertionError('must reuse')):
                second=run_pilot(ctx)
            self.assertNotEqual(first,second)
            self.assertTrue(pd.read_csv(second/'extraction_audit.csv').reused.all())
            self.assertEqual(hashes,{p.name:sha256(p) for p in ctx['cache'].glob('*.json')})

    def test_failure_keeps_ids_nan_and_stops_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=self.context(Path(tmp),10)
            ctx['targets'].loc[0,'xml_sha256']='invalid'
            out=run_pilot(ctx)
            frame=pd.read_csv(out/'game_features.csv')
            self.assertEqual(len(frame),10)
            self.assertEqual(frame.status.eq('error').sum(),1)
            self.assertTrue(pd.isna(frame.loc[0,A1+'__denominator']))
            self.assertTrue(pd.read_csv(out/'window_coverage.csv').complete_pilot_windows.eq(0).all())

    def test_cache_tampering_is_not_reused_or_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=self.context(Path(tmp),1)
            run_pilot(ctx)
            p=next(ctx['cache'].glob('*.json'))
            packet=json.loads(p.read_text(encoding='utf8'));packet['checksum']='bad'
            write_json(p,packet); before=sha256(p)
            out=run_pilot(ctx)
            self.assertEqual(pd.read_csv(out/'extraction_audit.csv').status.iloc[0],'error')
            self.assertEqual(sha256(p),before)

    def test_sparse_sample_never_compacted_into_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=self.context(Path(tmp),20)
            ctx['targets']=ctx['targets'].iloc[::2]
            out=run_pilot(ctx)
            self.assertTrue(pd.read_csv(out/'window_coverage.csv').complete_pilot_windows.eq(0).all())

    def test_selection_respects_cutoff_rank17_shared_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=self.context(Path(tmp),20); d=ctx['data']
            d.loc[10:,'target_label']=1;d.loc[10:,'dan']=18
            d.loc[4,'target_label']=np.nan;d.loc[4,'dan']=17
            selected=select_pilot(d,d.hour.iloc[1],{d.game_id.iloc[2]},2)
            self.assertEqual(len(selected),4)
            self.assertTrue(selected.hour.gt(d.hour.iloc[1]).all())
            for ids in ([d.game_id.iloc[4]],[d.game_id.iloc[5]]*2):
                with self.assertRaises(ValueError): select_pilot(d,d.hour.iloc[1],set(),explicit_ids=ids)

    def test_generic_c5_loader_and_hash_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ctx=self.context(root,20); folder=root/'c5';folder.mkdir()
            data=ctx['data'].drop(columns='target_label')
            write_csv(folder/'raw_features_v2.csv',data)
            write_csv(folder/'extraction_audit.csv',data[['file_id','status']])
            import mahjong_research.tenhou as tenhou
            marker=dict(signature='sig',extractor_sha256=sha256(tenhou.__file__),order_verified=False,
                full_history_confirmed=False,sha256=sha256(folder/'raw_features_v2.csv'),
                audit_sha256=sha256(folder/'extraction_audit.csv'),rows=20)
            write_json(folder/'complete.json',marker)
            write_json(root/'input_plan.json',{'any_player':dict(player='target',cache_dir=str(folder),signature='sig',
                       members=data.file_id.tolist(),zip_path=str(ctx['archive']))})
            write_csv(root/'shared_games.csv',pd.DataFrame(columns=['game_id']))
            self.assertEqual(len(load_c5(root,'any_player','target')[0]),20)
            marker['order_verified']=True;write_json(folder/'complete.json',marker)
            with self.assertRaises(ValueError): load_c5(root,'any_player','target')

    def test_window_boundary_and_duplicate_audits(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx=self.context(Path(tmp),20)
            frame=ctx['data'].copy()
            for f in NAMES:
                frame[f+'__numerator']=0;frame[f+'__denominator']=0
            ctx['data'].loc[10,'hour']=ctx['data'].hour.iloc[9]
            with self.assertRaisesRegex(ValueError,'時間帯'): window_audit(ctx,frame)
            with self.assertRaisesRegex(ValueError,'ID集合'): window_audit(ctx,pd.concat([frame,frame.iloc[:1]]))

    def test_cli_preview_to_extraction_contract(self):
        from decision_features.cli import main
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ctx=self.context(root,1606);data=ctx['data'].drop(columns='target_label')
            data.loc[1550:,'dan']=18
            high=xml(R+'<T12/><D0/>').replace(b'dan="16,16,16,16"',b'dan="18,16,16,16"')
            low=xml(R+'<T12/><D0/>')
            with zipfile.ZipFile(ctx['archive'],'w') as z:
                for row in data.itertuples():
                    content=high if row.dan==18 else low
                    z.writestr(row.file_id,content)
                    data.loc[row.Index,'xml_sha256']=hashlib.sha256(content).hexdigest()
            folder=root/'c5';folder.mkdir()
            write_csv(folder/'raw_features_v2.csv',data)
            write_csv(folder/'extraction_audit.csv',data[['file_id','status']])
            import mahjong_research.tenhou as tenhou
            write_json(folder/'complete.json',dict(signature='sig',extractor_sha256=sha256(tenhou.__file__),
                order_verified=False,full_history_confirmed=False,rows=len(data),
                sha256=sha256(folder/'raw_features_v2.csv'),audit_sha256=sha256(folder/'extraction_audit.csv')))
            write_json(root/'input_plan.json',dict(shobon=dict(player='target',cache_dir=str(folder),
                signature='sig',members=data.file_id.tolist(),zip_path=str(ctx['archive']))))
            write_csv(root/'shared_games.csv',pd.DataFrame(columns=['game_id']))
            config=root/'config.json';preview=root/'preview.json';result=root/'result.json'
            write_json(config,dict(run_dir=str(root),output_root=str(root/'out'),player='target',per_group=1))
            for action in ('preview','extract'):
                with patch.object(sys,'argv',['pilot',action,str(config),str(preview),str(result)]): main()
            out=Path(json.loads(result.read_text(encoding='utf8'))['output'])
            audit=pd.read_csv(out/'extraction_audit.csv')
            self.assertEqual(audit.dan.tolist(),[16,18])
            self.assertTrue(audit.status.eq('ok').all())
            changed=json.loads(preview.read_text(encoding='utf8'));changed['ids']=[];write_json(preview,changed)
            with patch.object(sys,'argv',['pilot','extract',str(config),str(preview),str(result)]):
                with self.assertRaisesRegex(ValueError,'プレビュー'): main()


if __name__ == '__main__': unittest.main()
