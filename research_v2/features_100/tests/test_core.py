"""Synthetic positions/events only. No real-log fixture is executed here."""
import copy
import unittest
from unittest.mock import patch
from mahjong_research.tenhou import _counts, Meld, wait_shapes, is_suji
from features_100.shapes import shanten, physical, ukeire, isolated, wall_blocked, blocks
from features_100.options import Options, Unknown
from features_100.rules import CATALOG, DEFERRED, IMPLEMENTED, measure
from features_100.calls import forms, measure_call
from features_100.extract import extract


def tiles(kinds):
    used={}; out=[]
    for k in kinds:
        j=used.get(k,0); out.append(k*4+j); used[k]=j+1
    return out


def state(hand=None, **updates):
    h=hand or tiles([0,1,2,9,10,11,18,19,20,27,27,27,31,33])
    s=dict(hand=h,melds=(),rivers=[set(),{33},set(),set()],active={1},seat=0,discard=h[-1],drawn=h[-1],
        free=True,excluded=[],previous=None,last_r0_pass=False,points=[25000]*4,dealer=1,round_index=0,
        indicators=[123],ippatsu=set(),wall=50,declaration=False,known=set(h)|{123},rules_verified=True,
        red_enabled=True,hanchan=True,honba=1,sticks=2,kan_count=0)
    s.update(updates); return s


def synthetic_xml(hidden='4,5,6', draw='81', ending='<RYUUKYOKU/>'):
    h=tiles([0,1,2,9,10,11,18,19,20,27,27,27,31])
    return (f'<mjloggm><GO type="9"/><UN n0="target" n1="other" dan="16,18,18,18"/>'
            f'<INIT seed="0,0,0,1,1,123" ten="250,250,250,250" oya="1" hai0="{",".join(map(str,h))}" hai1="{hidden}"/>'
            f'<REACH who="1" step="1"/><U{draw}/><E133/><REACH who="1" step="2"/>'
            f'<T132/><D132/>{ending}</mjloggm>').encode()


class Shapes(unittest.TestCase):
    def test_standard_complete(self):
        self.assertEqual(shanten(_counts(tiles([0,1,2,9,10,11,18,19,20,27,27,27,31,31])))[0],-1)
    def test_standard_tenpai(self):
        self.assertEqual(min(shanten(_counts(tiles([0,1,2,9,10,11,18,19,20,27,27,27,31])))),0)
    def test_seven_pairs(self):
        self.assertEqual(shanten(_counts(tiles([0,0,3,3,9,9,12,12,18,18,21,21,27])))[1],0)
    def test_quad_not_two_pairs(self):
        self.assertEqual(shanten(_counts(tiles([0,0,0,0,9,9,12,12,18,18,21,21,27])))[1],2)
    def test_orphans(self):
        self.assertEqual(shanten(_counts(tiles([0,8,9,17,18,26,27,28,29,30,31,32,33])))[2],0)
    def test_fixed_meld_disables_specials(self):
        self.assertEqual(shanten(tuple([0]*34),1)[1:],(99,99))
    def test_open_tenpai(self):
        self.assertEqual(min(shanten(_counts(tiles([0,1,2,9,10,11,18,18,18,31])),1)),0)
    def test_ukeire_tanki(self):
        h=tiles([0,1,2,9,10,11,18,19,20,27,27,27,31]); c=_counts(h)
        self.assertEqual(ukeire(c,0,c,tuple([0]*34)),3)
    def test_visible_dead_tanki(self):
        h=tiles([0,1,2,9,10,11,18,19,20,27,27,27,31]); c=_counts(h); known=list(c); known[31]=4
        self.assertEqual(ukeire(c,0,tuple(known),tuple([0]*34)),0)
    def test_isolation(self):
        c=_counts(tiles([0,2,9,27])); self.assertFalse(isolated(c,0)); self.assertTrue(isolated(c,9))
    def test_full_suji_middle(self):
        self.assertFalse(is_suji(3,{0})); self.assertTrue(is_suji(3,{0,6})); self.assertFalse(is_suji(27,{24}))
    def test_wall_not_tanki_guarantee(self):
        c=[0]*34; c[1]=4; self.assertTrue(wall_blocked(0,c)); self.assertFalse(wall_blocked(27,c))
    def test_blocks_distinct(self):
        self.assertEqual(blocks(_counts(tiles([0,1,2])))[0],3)
        self.assertEqual(blocks(_counts(tiles([0,0,1,1,2,2])))[0],3)


class Decisions(unittest.TestCase):
    def test_p01_handcalculated_num_den(self):
        o=Options(state()); v=measure('P01',o)
        # White-only tanki achieved by discarding red dragon (safe to player1).
        # Red-only tanki achieved by discarding white (unsafe).
        self.assertEqual((v['numerator'],v['denominator']),(0,1))
        o.s['discard']=124; self.assertEqual(measure('P01',o)['numerator'],1)
    def test_third_choice_not_removed(self):
        o=Options(state()); o.s['discard']=0; v=measure('P01',o)
        self.assertEqual((v['numerator'],v['denominator'],v['other']),(0,1,1))
    def test_no_safe_choice_no_denominator(self):
        self.assertEqual(measure('P01',Options(state(rivers=[set() for _ in range(4)])))['denominator'],0)
    def test_only_safe_choice_no_denominator(self):
        s=state(); s['rivers'][1]={t//4 for t in s['hand']}
        self.assertEqual(measure('P01',Options(s))['denominator'],0)
    def test_multiple_not_multiplied(self):
        s=state(active={1,2},rivers=[set(),{33},{33},set()]); v=measure('P13',Options(s))
        self.assertEqual(v['denominator'],1)
    def test_one_player_safe_not_common(self):
        s=state(active={1,2},rivers=[set(),{33},{31},set()]); o=Options(s)
        self.assertEqual(o.Gall,set()); self.assertEqual(measure('P13',o)['denominator'],0)
    def test_excluded_own_riichi(self):
        v=measure('P01',Options(state(free=False,excluded=['own_accepted_riichi'])))
        self.assertEqual((v['status'],v['denominator']),('excluded',0))
    def test_riichi_points_wall(self):
        self.assertEqual(Options(state(points=[900,25000,25000,25000])).L,set())
        self.assertEqual(Options(state(wall=3)).L,set())
        self.assertTrue(Options(state(wall=4)).L)
    def test_rule_unknown_not_false(self):
        self.assertEqual(measure('R09',Options(state(rules_verified=False)))['status'],'unknown')
    def test_deferred_missing_not_zero(self):
        v=measure('Q07',Options(state(active=set())))
        self.assertIsNone(v['denominator']); self.assertEqual(v['status'],'unimplemented')
    def test_formal_fifth_copy_uncertainty(self):
        o=Options(state()); o.__dict__['waits']={d:{} for d in o.D}
        with self.assertRaises(Unknown): _=o.Sforms
    def test_shape_copies_reuse(self):
        o=Options(state()); self.assertIs(o.waits[108],o.waits[109])


class Calls(unittest.TestCase):
    def offer_state(self):
        s=state(tiles([0,1,2,9,10,11,18,19,20,31,31,27,28]),active=set())
        s['known'].add(126); s['rivers'][3].add(31); return s
    def test_yakuhai_pon_or_pass(self):
        s=self.offer_state(); choices=forms(s,126,3)
        self.assertTrue(any(v['meld'].kind=='pon' for v in choices.values()))
        m=next(v['meld'] for v in choices.values() if v['meld'].kind=='pon')
        a,_=measure_call(s,126,3,'own_call',m); b,_=measure_call(s,126,3,'ordinary_progress')
        self.assertEqual((a['M04']['numerator'],a['M04']['denominator']),(1,1))
        self.assertEqual((b['M04']['numerator'],b['M04']['denominator']),(0,1))
    def test_competitor_is_censored(self):
        r,_=measure_call(self.offer_state(),126,3,'censored_competing_call')
        self.assertTrue(all(v['status']=='censored' and v['denominator']==0 for v in r.values()))
    def test_no_call_after_riichi(self):
        s=self.offer_state(); s['active']={0}; self.assertEqual(forms(s,126,3),{})
    def test_last_discard_no_call(self):
        s=self.offer_state(); s['wall']=0; self.assertEqual(forms(s,126,3),{})
    def test_chi_only_upper_player(self):
        s=self.offer_state()
        self.assertFalse(any(v['meld'].kind=='chi' for v in forms(s,3*4,1).values()))
    def test_chi_kuikae(self):
        s=state(tiles([0,1,2,4,8,9,10,11,18,19,20,27,28]),active=set())
        s['known'].add(12); c=forms(s,12,3)
        m=next(v for v in c.values() if v['meld'].kind=='chi' and 4 in v['meld'].tiles and 8 in v['meld'].tiles)
        self.assertNotIn(0,m['options'].D); self.assertIn(0,m['forbidden'])
    def test_call_unknown_profile(self):
        s=self.offer_state(); s['rules_verified']=False
        r,_=measure_call(s,126,3,'ordinary_progress'); self.assertTrue(all(v['status']=='unknown' for v in r.values()))


class Replay(unittest.TestCase):
    def test_all100_no_real_log(self):
        r=extract(synthetic_xml(),'target',game_id='2012052419gm-test')
        self.assertTrue(set(CATALOG)<=r['features'].keys()); self.assertEqual(r['audit']['own_discards'],1)
    def test_hidden_information_invariance(self):
        a=extract(synthetic_xml(),'target',game_id='2012052419gm-test')
        b=extract(synthetic_xml(hidden='128,129,130',draw='125'),'target',game_id='2012052419gm-test')
        self.assertEqual(a['features'],b['features']); self.assertEqual(a['trace'],b['trace'])
    def test_future_information_invariance(self):
        a=extract(synthetic_xml(),'target',game_id='2012052419gm-test')
        b=extract(synthetic_xml(ending='<DORA hai="100"/><RYUUKYOKU hai1="0,1,2" sc="999,0"/>'),'target',game_id='2012052419gm-test')
        self.assertEqual(a['trace'][0],b['trace'][0])
    def test_bad_actual_discard_fails(self):
        with self.assertRaises(ValueError): extract(synthetic_xml().replace(b'D132',b'D131'),'target')
    def test_no_end_fails(self):
        with self.assertRaises(ValueError): extract(synthetic_xml(ending=''),'target')
    def test_other_player_identity(self):
        r=extract(synthetic_xml().replace(b'target',b'generic'),'generic')
        self.assertEqual(r['player'],'generic')
    def test_no_duplicate_own_feature(self):
        r=extract(synthetic_xml(),'target',game_id='2012052419gm-test')
        self.assertLessEqual(r['features']['P01']['denominator'],r['audit']['own_discards'])
    def test_accepted_riichi_only(self):
        xml=synthetic_xml().replace(b'<REACH who="1" step="2"/>',b'')
        r=extract(xml,'target'); self.assertEqual(r['features']['P01']['denominator'],0)


# Each catalog entry has a separate test case for status dispatch and missingness.
# These are coverage contracts, NOT 100 real-game or positive-opportunity validations.
def contract(i):
    def test(self):
        v=measure(i,Options(state(free=False,excluded=['kan_since_previous_own_discard'])))
        if i in DEFERRED: self.assertIsNone(v['denominator'])
        else: self.assertEqual(v['denominator'],0)
    return test

class CatalogContracts(unittest.TestCase): pass
for _id in CATALOG: setattr(CatalogContracts,'test_'+_id,contract(_id))

if __name__=='__main__': unittest.main()
