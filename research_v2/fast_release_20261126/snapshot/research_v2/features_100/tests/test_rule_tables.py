"""Hand-calculated predicate tables with fixed analyzer outputs.

These isolate policy logic from hand reconstruction. They are deliberately not
represented as physical-hand or real-log validation (those are separate tests).
"""
import unittest
from types import SimpleNamespace
from mahjong_research.tenhou import _counts, Meld
from features_100.rules import evaluate,measure
from features_100.options import Options
from .test_core import state,tiles


def fixture(changes=None,state_changes=None):
    o=Options(state(**(state_changes or {})))
    o.D={0,4,8}; o.opp={1}; o.prev=None
    o.__dict__.update(T={0,4},G={4},Gall={4},J={8},Dm={0,4},Smin=1,
        S={0:1,4:1,8:2},U={0:8,4:4,8:1},V={0:2,4:0,8:0},B={0:2,4:0,8:0},
        Z={0:3,4:1,8:0},closed=True,L={0,4},PF={0:True,4:True,8:False},
        waits={0:{31:('ryanmen',)},4:{31:('tanki',)},8:{}},
        Sforms={0:(1,4,5),4:(1,4,5),8:(2,4,5)},known=tuple([0]*34))
    o.__dict__.update(changes or {})
    return o


TABLES=[
 ('P01',{}, {},{0},{4}),('P02',{}, {},{0},{4}),('P03',{'Smin':2},{},{0},{4}),
 ('P04',{'T':{0}},{},{0},{4}),('P05',{'Dm':{0,8}},{},{0,8},{4}),
 ('P06',{}, {},{0},{4}),('P07',{}, {},{0},{4}),
 ('P08',{}, {},{0},{4}),('P09',{'U':{0:0,4:0,8:0}},{},{0},{4}),
 ('P10',{}, {},{0},{4}),
 ('P12',{'Dm':{0,8}},{},{0,8},{4}),
 ('P13',{'opp':{1,2}},{},{0},{4}),('P14',{'opp':{1,2}},{},{0},{4}),
 ('P15',{'opp':{1,2},'prev':{'opponents':[1]}},{},{4},{0,8}),
 ('P16',{}, {'ippatsu':{1}},{0},{4}),('P17',{}, {},{0},{4}),
 ('P19',{'G':set(),'T':{0}},{},{0},{8}),
 ('Q01',{'opp':set()},{},{0},{4}),('Q02',{'opp':set(),'Smin':2},{},{0},{4}),
 ('Q03',{'opp':set(),'V':{0:0,4:2,8:0}},{},{4},{0}),
 ('Q04',{'opp':set(),'B':{0:0,4:2,8:0}},{},{4},{0}),
 ('Q05',{'opp':set()},{},{0},{4}),('Q06',{'opp':set()},{},{0},{4}),
 ('Q09',{'opp':set(),'V':{0:0,4:2,8:0}},{},{4},{0}),
 ('Q10',{'opp':set(),'Sforms':{0:(0,2,5),4:(2,0,5),8:(3,3,5)}},{},{4},{0}),
 ('Q11',{'opp':set(),'Sforms':{0:(0,2,5),4:(2,3,1),8:(3,3,5)}},{},{4},{0}),
 ('Q24',{'opp':set(),'U':{0:4,4:4,8:4}},{},{0},{4}),
 ('D05',{'G':set(),'U':{0:4,4:4,8:4},'J':{4}},{},{4},{0}),
 ('D06',{'U':{0:4,4:4,8:4},'J':{0}},{},{4},{0}),
 ('D07',{'opp':{1,2},'Gall':set(),'rivers':[set(),{0},{0},set()]},{},{0},{4,8}),
 ('D08',{'opp':{1,2},'Gall':set(),'rivers':[set(),{0},{1},set()]},{},{0},{4}),
 ('S01',{'T':{0}},{'points':[50000,30000,20000,0]},{4},{0}),
 ('S02',{'Dm':{0,8}},{'points':[0,20000,30000,50000]},{0,8},{4}),
 ('S03',{}, {'points':[3000,30000,30000,37000]},{4},{0}),
 ('S09',{'opp':set(),'V':{0:0,4:2,8:0}},{'round_index':7,'points':[0,20000,30000,50000]},{4},{0}),
 ('S10',{}, {'dealer':3,'round_index':3},{4},{0}),
 ('R05',{'opp':set()},{},{(0,True),(4,True)},{(0,False),(4,False)}),
 ('R06',{'opp':set(),'U':{0:1,4:1,8:0}},{},{(0,True),(4,True)},{(0,False),(4,False)}),
 ('R07',{'opp':set()},{},{(0,True),(4,True)},{(0,False)}),
 ('R08',{'G':{0,4}},{},{(0,True),(4,True)},{(0,False),(4,False)}),
 ('R09',{}, {},{(4,True)},{(0,True)}),
 ('R12',{'opp':set()},{'points':[25100,24500,30000,20400]},{(0,True),(4,True)},{(0,False),(4,False)}),
 ('R13',{'opp':set()},{'last_r0_pass':True},{(0,True),(4,True)},{(0,False),(4,False)}),
 ('R14',{'prev':{'free':True,'r0_pass':True}},{},{(0,True),(4,True)},{(0,False),(4,False)}),
 ('R15',{'opp':set()},{'wall':8},{(0,True),(4,True)},{(0,False),(4,False)}),
 ('S04',{'opp':set()},{'points':[25100,24500,30000,20400]},{(0,False),(4,False)},{(0,True),(4,True)}),
]

def cnt(items): return _counts(tiles(items))
def prev(**kw): return dict(free=True,opponents=[1],**kw)
TABLES += [
 ('P11',{'melds':[SimpleNamespace(kind='pon')]},{},{0},{4}),
 ('P18',{'prev':{'rivers':[[],[],[],[]]},'rivers':[set(),{1},set(),set()]},{},{4},{0,8}),
 ('P20',{'prev':prev(safe_break=True)},{},{0},{4}),
 ('Q12',{'opp':set(),'Sforms':{0:(2,1,5),4:(2,1,5),8:(3,3,5)},'known':tuple([4]+[0]*33)},{},{0},{4}),
 ('Q13',{'opp':set(),'D':{0,108,124},'Dm':{0,108},'c':cnt([0,27,31])},{},{108},{0}),
 ('Q14',{'opp':set(),'D':{0,108,124},'Dm':{0,108},'c':cnt([0,27,31])},{},{0},{108}),
 ('Q15',{'opp':set(),'D':{0,112,124},'Dm':{0,112},'c':cnt([0,28,31])},{},{0},{112}),
 ('Q16',{'opp':set(),'D':{0,108,109},'Dm':{0,108,109},'c':cnt([0,27,27])},{},{108,109},{0}),
 ('Q17',{'opp':set(),'D':{0,4,8,16},'Dm':{0,16},'c':cnt([0,1,2,4])},{},{16},{0}),
 ('Q18',{'opp':set(),'D':{0,16},'Dm':{0,16},'c':cnt([0,1,4,6])},{},{0},{16}),
 ('Q19',{'opp':set(),'D':{0,108},'Dm':{0,108},'c':cnt([0,1,9,10,18,19,27])},{},{108},{0}),
 ('Q20',{'opp':set(),'D':{0,108},'Dm':{0,108},'c':cnt([0,1,3,4,6,7,27])},{},{108},{0}),
 ('Q21',{'opp':set(),'D':{0,4},'Dm':{0,4},'c':cnt([0,1,2])},{},{0},{4}),
 ('Q22',{'opp':set(),'D':{0,36},'Dm':{0,36},'c':cnt([0,1,9]),'U':{0:8,36:4}},{},{36},{0}),
 ('Q23',{'opp':set(),'D':{0,36},'Dm':{0,36},'c':cnt([0,9]),'known':tuple([4]+[0]*8+[1]+[0]*24)},{},{0},{36}),
 ('Q25',{'opp':set(),'D':{0,16},'Dm':{0,16},'U':{0:4,16:8}},{},{0},{16}),
 ('D01',{'opp':set(),'b_before':1,'B':{0:1,4:0,8:0},'U':{0:4,4:4,8:4}},{},{0},{4}),
 ('D02',{'G':{4,8}},{},{0},{4}),
 ('D03',{'D':{0,1,4},'G':{0,1,4},'c':cnt([0,0,1])},{},{0,1},{4}),
 ('D04',{'G':{0,4},'known':tuple([4,2]+[0]*32)},{},{0},{4}),
 ('D09',{'G':{0,4},'rivers':[set(),{0,1},{0},set()]},{},{4},{0}),
 ('D10',{'opp':set(),'D':{0,36},'Dm':{0,36},'c':cnt([0,9])},{'indicators':[3]},{0},{36}),
 ('D11',{'G':set(),'D':{0,12},'Dm':{0,12},'U':{0:4,12:4}},{},{0},{12}),
 ('D12',{'G':set(),'D':{108,112},'known':tuple([0]*27+[3,2]+[0]*5)},{},{108},{112}),
 ('D13',{'G':set(),'D':{0,4},'J':{0},'known':tuple([0,0,0,4]+[0]*30)},{},{0},{4}),
 ('D14',{'prev':prev(chosen_safe=True)},{},{4},{0}),
 ('D15',{'prev':prev(g_empty=True)},{},{4},{0,8}),
]


def policy_test(row):
    i,changes,sc,x,y=row
    def test(self):
        o=fixture(changes,sc)
        cond,actual_x,actual_y=evaluate(i,o)
        self.assertTrue(cond); self.assertEqual(actual_x,x); self.assertEqual(actual_y,y)
        for group,expected in ((x,1),(y,0)):
            a=next(iter(group)); o.s['discard']=a[0] if isinstance(a,tuple) else a
            o.s['declaration']=a[1] if isinstance(a,tuple) else False
            v=measure(i,o); self.assertEqual((v['numerator'],v['denominator'],v['other']),(expected,1,0))
        # A third legal choice stays in the denominator whenever it exists.
        universe={(a,b) for a in o.D for b in (False,True)} if isinstance(next(iter(x)),tuple) else o.D
        if universe-(x|y):
            a=next(iter(universe-(x|y))); o.s['discard']=a[0] if isinstance(a,tuple) else a
            o.s['declaration']=a[1] if isinstance(a,tuple) else False
            v=measure(i,o); self.assertEqual((v['numerator'],v['denominator'],v['other']),(0,1,1))
    return test

class PredicateTables(unittest.TestCase): pass
for row in TABLES: setattr(PredicateTables,'test_'+row[0],policy_test(row))


def call_table(i):
    def test(self):
        from features_100.calls import decision_sets
        from unittest.mock import patch
        h=tiles([0,1,2,9,10,11,18,19,20,31,31,27,28])
        s=state(h,active=set(),indicators=[132]); s['known'].add(126)
        def option(smin=0,u=8,g={0}):
            return SimpleNamespace(Smin=smin,S={0:smin,4:smin+1},Dm={0},D={0,4},G=g,U={0:u,4:2},s={'hand':[4,8]})
        pon=Meld('pon',(124,125,126),126,3)
        choices={'pon':{'meld':pon,'options':option()}}
        actual='pon'; x={'pon'}; y={'pass'}
        if i=='M03': choices['pon']['meld']=Meld('pon',(4,5,6),6,3)
        if i=='M05': choices['pon']['options']=option(1)
        if i in ('M06','M07'): s['active']={1}
        if i=='M07': choices['pon']['options']=option(g=set())
        if i=='M08': s['indicators']=[32]  # 9m indicator -> 1m; improving discard is 1m
        if i in ('M10','M11','M12'):
            choices['chi']={'meld':Meld('chi',(116,120,124),124,3),'options':option(1)}
            if i=='M10': y={'chi'}
            if i=='M11':
                choices['pon']['options']=option(1); choices['chi']['options']=option(0); actual='chi'; x={'chi'}; y={'pon'}
            if i=='M12':
                choices={'chi':{'meld':Meld('chi',(116,120,124),124,3),'options':option(0,8)},
                         'chi2':{'meld':Meld('chi',(120,124,128),124,3),'options':option(0,4)}}
                actual='chi'; x={'chi'}; y={'chi2'}
        if i=='M13': s['melds']=(Meld('pon',(108,109,110),110,1),)
        if i=='M14': s['wall']=8
        if i=='M15':
            s['hand']=tiles([0,1,2,9,10,11,18,19,20,31,31,31,27])
            choices['kan']={'meld':Meld('daiminkan',(124,125,126,127),126,3),'options':None}
            actual='kan'; x={'kan'}; y={'pon'}
        if i=='S06': s['dealer']=0
        with patch('features_100.calls.ukeire',return_value=4):
            cond,xx,yy=decision_sets(i,s,126,3,choices,actual)
        self.assertTrue(cond); self.assertEqual(xx,x); self.assertEqual(yy,y)
    return test

class CallPredicateTables(unittest.TestCase): pass
for _id in [*(f'M{i:02d}' for i in range(1,16)),'S06','S07']:
    setattr(CallPredicateTables,'test_'+_id,call_table(_id))

if __name__=='__main__': unittest.main()
