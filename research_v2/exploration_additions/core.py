"""Selected-feature adapter: fixed replay function, private globals, no Z/call rules."""
import types,hashlib
from pathlib import Path
from features_100 import extract as replay
from features_100.options import Options,Unknown
from features_100.rules import measure as parent_measure
PARENTS=('P01','Q01','Q03','D02')
DEFINITIONS={
 'P01_late':('ratio','P01','残り通常ツモ24以下。自己巡目とは異なる'),
 'Q01_closed':('ratio','Q01','門前（暗槓のみ可）'),
 'Q03_one':('ratio','Q03','最良形状シャンテン1'),
 'D02_tenpai':('ratio','D02','最良形状シャンテン0'),
 'P01_dealer':('ratio','P01','自分が親：連荘・収入条件の違い'),
 'Q01_open':('ratio','Q01','開いた手：Q01_closedの補集合。独立した証拠としない'),
 'Q03_closed':('ratio','Q03','門前：表ドラ保持と速度の交換'),
 'D02_late':('ratio','D02','残り通常ツモ24以下：現物資源と残り判断回数'),
 'ukeire_loss_mean':('quantity','','無他家成立リーチ・自由打牌。同S最良候補間でU差あり、実打牌も最良S。最大U−実打牌Uの合計/該当打牌数'),
 'safe_shape_cost_mean':('quantity','','単独他家成立リーチ・自由打牌。現物/非現物双方あり実際に現物選択。実打牌S−最小Sの合計/該当現物選択数'),
}
class SelectedOptions(Options):
    def history(self):return None # none of selected features use previous actions
    @property
    def Z(self):raise AssertionError('Z must never be computed')

def measure(f,o):
    if f in PARENTS:return parent_measure(f,o)
    kind,parent,_=DEFINITIONS[f]
    if parent:
        v=parent_measure(parent,o).copy()
        if not v.get('denominator'):return v
        keep={'P01_late':lambda:o.s['wall']<=24,'Q01_closed':lambda:o.closed,
              'Q03_one':lambda:o.Smin==1,'D02_tenpai':lambda:o.Smin==0,
              'P01_dealer':lambda:o.s['seat']==o.s['dealer'],'Q01_open':lambda:not o.closed,
              'Q03_closed':lambda:o.closed,'D02_late':lambda:o.s['wall']<=24}[f]()
        if not keep:v.update(numerator=0,denominator=0,other=0,status='no_opportunity',reason='additional_condition_absent')
        return v
    v=dict(numerator=0,denominator=0,other=0,status='no_opportunity')
    if not o.s['free']:return dict(v,status='excluded',reason='not_free')
    try:
        a=o.s['discard']
        if f=='ukeire_loss_mean' and not o.opp:
            best=o.Dm
            if a in best and len({o.U[d] for d in best})>1:v.update(numerator=max(o.U[d] for d in best)-o.U[a],denominator=1,status='ok')
        if f=='safe_shape_cost_mean' and len(o.opp)==1 and o.G and o.D-o.G and a in o.G:
            v.update(numerator=o.S[a]-o.Smin,denominator=1,status='ok')
        return v
    except Unknown as e:return dict(v,status='unknown',reason=str(e))

def extract(xml,player,seat=None,dan=None,game_id='',trace=False):
    # Function globals are copied, never monkey-patched in the fixed package.
    g=dict(replay.extract.__globals__)
    ids=(*PARENTS,*DEFINITIONS)
    g.update(Options=SelectedOptions,CATALOG=dict.fromkeys(ids),IMPLEMENTED=ids,DEFERRED={},PARENTS=(),
             measure=measure,measure_call=lambda *args:({},{}))
    fn=types.FunctionType(replay.extract.__code__,g,'selected_extract',replay.extract.__defaults__,replay.extract.__closure__)
    return fn(xml,player,seat,dan,game_id,trace)
