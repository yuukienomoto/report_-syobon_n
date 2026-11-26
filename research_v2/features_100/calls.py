"""Observable call responses, with explicit competition censorship.

Forms are tile-kind forms: red-copy alternatives do not multiply opportunities.
Only chi/pon best immediate legal discards are evaluated; kan is an action for
M15, never a fabricated post-rinshan hand. Opponent private hands are not needed.
"""
from itertools import combinations, product
from mahjong_research.tenhou import Meld, _counts, wait_shapes, dora_count
from .options import Options, Unknown
from .shapes import shanten, ukeire

IDS=tuple(f'M{i:02d}' for i in range(1,16))+('S06','S07')


def key(m): return m.kind+':'+','.join(map(str,sorted(t//4 for t in m.tiles)))


def forms(state, offered, giver):
    if not state['rules_verified']: raise Unknown('call_rule_profile_unverified')
    if state['seat'] in state['active'] or state['wall']<=0: return {}
    hand=state['hand']; k=offered//4
    same=[t for t in hand if t//4==k]; possible=[]
    if len(same)>=2:
        possible.append(Meld('pon',tuple(same[:2]+[offered]),offered,giver))
    if len(same)==3 and state['kan_count']<4:
        possible.append(Meld('daiminkan',tuple(same+[offered]),offered,giver))
    if giver==(state['seat']-1)%4 and k<27:
        for start in range(max(k//9*9,k-2),min(k,k//9*9+6)+1):
            needed=[q for q in range(start,start+3) if q!=k]
            if all(any(t//4==q for t in hand) for q in needed):
                ts=[next(t for t in hand if t//4==q) for q in needed]
                possible.append(Meld('chi',tuple(sorted(ts+[offered])),offered,giver))
    out={}
    for m in possible:
        if m.kind=='daiminkan':
            out[key(m)]=dict(meld=m,options=None); continue
        h=list(hand)
        for tile in m.tiles:
            if tile!=offered: h.remove(tile)
        forbidden={k}
        if m.kind=='chi':
            ks=sorted(t//4 for t in m.tiles)
            if k==ks[0] and ks[-1]%9<8: forbidden.add(ks[-1]+1)
            if k==ks[-1] and ks[0]%9>0: forbidden.add(ks[0]-1)
        legal={t for t in h if t//4 not in forbidden}
        if not legal: continue
        s=dict(state,hand=h,melds=(*state['melds'],m),discard=min(legal),free=True,excluded=[],previous=None)
        o=Options(s); o.D=legal
        # Counts still include all concealed tiles; only candidate enumeration is restricted.
        out[key(m)]=dict(meld=m,options=o,forbidden=sorted(forbidden))
    return out


def decision_sets(i,state,offered,giver,choices,actual):
    cp={k:v['options'] for k,v in choices.items() if v['options'] is not None}
    chi={k for k in cp if choices[k]['meld'].kind=='chi'}
    pon=set(cp)-chi; kan={k for k in choices if k not in cp}
    n=len(set(state['active'])-{state['seat']}); hand=state['hand']
    role={31,32,33,27+state['round_index']//4,27+(state['seat']-state['dealer'])%4}
    melds=state['melds']; c=_counts(hand); fixed=len(melds)
    ps=99; S={}
    if i not in ('M03','M04','M13','M15'):
        ps=min(shanten(c,fixed))
        if ps==0 and not wait_shapes(hand,melds): raise Unknown('own_exhausted_shape_uncertain')
        S={k:o.Smin for k,o in cp.items()}
    improving={k for k in S if S[k]<ps}
    x=set(); y={'pass'}; cond=True
    if i=='M01': cond=offered//4 in role and c[offered//4]==2; x=pon&improving
    elif i=='M02': cond=ps==1; x={k for k in cp if S[k]==0}
    elif i=='M03':
        middle=lambda t:t//4<27 and t//4%9 in range(1,8)
        cond=all(middle(t) for m in melds for t in m.tiles)
        x={k for k,o in cp.items() if all(middle(t) for t in choices[k]['meld'].tiles) and
           any(all(middle(t) for t in o.s['hand'] if t!=a) for a in o.D)}
    elif i=='M04': cond=all(m.kind=='ankan' for m in melds) and offered//4 in role; x=pon
    elif i=='M05':
        u=ukeire(c,fixed,_counts(sorted(state['known'])),_counts([t for m in melds for t in m.tiles]))
        x={k for k,o in cp.items() if S[k]==ps and max(o.U[a] for a in o.Dm)>u}
    elif i=='M06': cond=n==1; x={k for k,o in cp.items() if any(o.S[a]<ps for a in o.G)}
    elif i=='M07': cond=n==1; x={k for k,o in cp.items() if not o.G and S[k]<ps}
    elif i=='M08':
        x={k for k,o in cp.items() if S[k]<ps and all(dora_count([a],state['indicators'])>0 for a in o.D if o.S[a]<ps)}
    elif i=='M09': cond=dora_count([offered],state['indicators'])>0; x=improving
    elif i=='M10':
        cond=actual in choices and giver==(state['seat']-1)%4
        x={k for k in pon if chi and S[k]<min(S[z] for z in chi)}; y=chi
    elif i=='M11':
        cond=actual in choices and giver==(state['seat']-1)%4
        x={k for k in chi if pon and S[k]<min(S[z] for z in pon)}; y=pon
    elif i=='M12':
        cond=actual in chi
        low={k for k in chi if S[k]==min((S[z] for z in chi),default=99)}
        U={k:max(cp[k].U[a] for a in cp[k].Dm) for k in low}
        x={k for k in low if U[k]==max(U.values(),default=-1)}; y=low-x
    elif i=='M13':
        old={m.tiles[0]//4 for m in melds if m.kind in ('pon','kakan','ankan','daiminkan') and m.tiles[0]//4 in role}
        cond=bool(old) and offered//4 in role-old; x=pon
    elif i=='M14': cond=state['wall']<=8 and ps>=1; x={k for k in cp if S[k]==0}
    elif i=='M15': cond=actual in choices and c[offered//4]==3; x=kan; y=pon
    elif i=='S06': cond=state['seat']==state['dealer'] and state['honba']>=1 and offered//4 in role; x=pon&improving
    elif i=='S07': cond=state['sticks']>=2; x=improving
    else: raise AssertionError(i)
    return cond,x,y


def measure_call(state,offered,giver,response,actual_meld=None):
    """Censor competing/terminal/connection responses before measuring intentions."""
    result={}
    actual='pass' if actual_meld is None else key(actual_meld)
    if response.startswith('censored'):
        return {i:dict(status='censored',reason=response,numerator=0,denominator=0,other=0) for i in IDS},{}
    try:
        choices=forms(state,offered,giver)
        if actual!='pass' and actual not in choices: raise ValueError('observed call absent from legal forms')
        detail={k:dict(meld_kind=v['meld'].kind,tiles=list(v['meld'].tiles),
                      legal_discards=sorted(v['options'].D) if v['options'] else [],
                      forbidden=v.get('forbidden',[])) for k,v in choices.items()}
        if not choices:
            return {i:dict(status='no_opportunity',reason='no_legal_call',numerator=0,denominator=0,other=0) for i in IDS},detail
        for i in IDS:
            try:
                cond,x,y=decision_sets(i,state,offered,giver,choices,actual)
                if x&y or not x|y<=set(choices)|{'pass'}: raise AssertionError('invalid call choice sets')
                den=int(bool(cond and x and y))
                result[i]=dict(status='ok' if den else 'no_opportunity',reason='included' if den else 'scene_or_choices_absent',
                    numerator=int(den and actual in x),denominator=den,other=int(den and actual not in x|y),
                    X=sorted(x),Y=sorted(y),actual=actual)
            except Unknown as e: result[i]=dict(status='unknown',reason=str(e),numerator=None,denominator=None,other=None)
        return result,detail
    except Unknown as e:
        return {i:dict(status='unknown',reason=str(e),numerator=None,denominator=None,other=None) for i in IDS},{}
