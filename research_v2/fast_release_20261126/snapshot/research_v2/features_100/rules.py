"""Explicit X/Y predicates from design 2079e5b; third choices remain denominator.

No string-eval of the design CSV. Deferred rules are registered, not fabricated.
"""
import csv
from pathlib import Path
from .options import Unknown, best
from .shapes import post, blocks, isolated, wall_blocked
from .calls import IDS as CALL_IDS

CATALOG_PATH = Path(__file__).resolve().parents[1]/'feature_design_100/feature_candidates_100.csv'
CATALOG = {r['feature_id']:r for r in csv.DictReader(CATALOG_PATH.open(encoding='utf-8-sig'))}
DEFERRED = {i:'fixed_yaku_score_or_complete_furiten_not_implemented' for i in
            ('Q07','Q08','R01','R02','R03','R04','R10','R11','S05','S08')}
IMPLEMENTED = tuple(i for i in CATALOG if i not in DEFERRED)
PARENTS = ('BASE_F1_T_UNSAFE','BASE_F1_MIN_UNSAFE','BASE_FM_T_UNSAFE',
           'BASE_F0_MAX_U','BASE_F1_SAFE','BASE_F0_DORA_VS_U','BASE_R0_DECLARE')


def evaluate(i, o):
    """Return (scene, X, Y); all sets contain legal physical discards or (tile,declare)."""
    d,g,t,j,ga = o.D,o.G,o.T,o.J,o.Gall
    s=o.s; n=len(o.opp)
    if i in DEFERRED: raise Unknown(DEFERRED[i])
    if i.startswith('BASE_'):
        if i=='BASE_F1_T_UNSAFE': return n==1,t-g,g
        if i=='BASE_F1_MIN_UNSAFE': return n==1,o.Dm-g,g
        if i=='BASE_FM_T_UNSAFE': return n>=2,t-ga,ga
        if i=='BASE_F0_MAX_U': return n==0,o.max('U'),o.Dm-o.max('U')
        if i=='BASE_F1_SAFE': return n==1,g,d-g
        if i=='BASE_F0_DORA_VS_U': return n==0 and not o.max('V')&o.max('U'),o.max('V'),o.max('U')
        return n==0,{(x,True) for x in o.L},{(x,False) for x in o.L}
    f1 = i.startswith('P') and i not in ('P13','P14','P15')
    f0 = i.startswith('Q') or i in ('D01','D10','S09')
    fm = i in ('P13','P14','P15','D07','D08')
    if (f1 and n!=1) or (f0 and n!=0) or (fm and n<2): return False,set(),set()
    if (i.startswith('D') and not f0 and not fm or i in ('S01','S02','S03','S10')) and n!=1:
        return False,set(),set()
    if i=='P01': return True,t-g,g
    if i=='P02': return o.Smin==1,o.Dm-g,g
    if i=='P03': return o.Smin>=2,o.Dm-g,g
    if i=='P04': return not g&t,t,g
    if i=='P05': return o.Smin==1 and not g&o.Dm,o.Dm,g
    if i=='P06': return o.Smin>=1 and not o.max('U')&g,o.max('U'),g&o.Dm
    if i=='P07': return not o.max('V')&g,o.max('V'),g&o.Dm
    if i=='P08': return bool(t) and all(o.U[x]>0 for x in t),t-g,g
    if i=='P09': return bool(t) and all(o.U[x]==0 for x in t),t-g,g
    if i=='P10': return o.closed,o.Dm-g,g
    if i=='P11': return any(m.kind in ('chi','pon') for m in o.melds),o.Dm-g,g
    if i=='P12': return s['dealer'] in o.opp and not g&o.Dm,o.Dm,g
    if i=='P13': return True,t-ga,ga
    if i=='P14': return o.Smin==1,o.Dm-ga,ga
    if i=='P15': return bool(o.prev) and len(o.prev['opponents'])==1 and n==2 and set(o.prev['opponents'])<o.opp,ga,d-ga
    if i=='P16': return next(iter(o.opp)) in s['ippatsu'],t-g,g
    if i=='P17': return len(g)==1,o.Dm-g,g
    if i=='P18':
        if not o.prev: return False,set(),set()
        p=next(iter(o.opp)); new=o.rivers[p]-set(o.prev['rivers'][p])
        return True,o.pick(lambda x:x//4 in new),d-g
    if i=='P19': return not g and all(x//4<27 for x in d) and not t&j,t,j
    if i=='P20': return o.same_target_previous() and o.prev['safe_break'],t-g,g
    if i in ('Q01','Q02'): return o.Smin==int(i[-1]),o.max('U'),o.Dm-o.max('U')
    if i=='Q03': return o.Smin>=1 and not o.max('U')&o.max('V'),o.max('V'),o.max('U')
    if i=='Q04': return not o.max('U')&o.max('B'),o.max('B'),o.max('U')
    if i=='Q05':
        x={a for a in t if any('ryanmen' in v for v in o.waits[a].values())}
        return True,x,t-x
    if i=='Q06': return True,o.max('U',t),t-o.max('U',t)
    if i=='Q09': return not o.max('U',t)&o.max('V',t),o.max('V',t),o.max('U',t)
    if i in ('Q10','Q11','Q12'):
        if o.melds: return False,set(),set()
        form=1 if i in ('Q10','Q12') else 2
        vals={a:v[form] for a,v in o.Sforms.items()}; x=best(d,vals,False)
        normal=best(d,{a:v[0] for a,v in o.Sforms.items()},False)
        if i=='Q12':
            w={a:sum(4-o.known[z] for z,v in enumerate(post(o.c,a//4)) if v==1) for a in x}
            mx=best(x,w)
            return any(v[1]<v[0] for v in o.Sforms.values()),mx,x-mx
        return not x&normal and (i!='Q11' or min(vals.values())<=3),x,normal
    if i in ('Q13','Q14','Q15','Q16','Q23','D10'):
        iso=o.pick(lambda a:isolated(o.c,a//4),o.Dm)
        suited={a for a in iso if a//4<27}; honors=iso-suited
        if i=='Q13': return True,honors,suited
        if i=='Q14': return True,suited,{a for a in honors if a//4 in o.role}
        if i=='Q15': return True,suited,{a for a in honors if 27<=a//4<31 and a//4 not in o.role}
        if i=='Q16': return bool(suited),o.pick(lambda a:o.c[a//4]==2,o.Dm),suited
        if i=='Q23': return True,{a for a in iso if o.known[a//4]==4},{a for a in iso if o.known[a//4]<4}
        bonus={a//4 for a in range(108) if __import__('mahjong_research.tenhou',fromlist=['dora_count']).dora_count([a],s['indicators'])}
        if not bonus: return False,set(),set()
        x={a for a in suited if any(a//36==b//9 and abs(a//4-b)==1 for b in bonus)}
        y={a for a in suited if all(a//36!=b//9 or abs(a//4-b)>=3 for b in bonus)}
        return True,x,y
    if i in ('Q17','Q18','Q19','Q20'):
        b={a:blocks(post(o.c,a//4)) for a in o.Dm}; before=blocks(o.c)
        if i=='Q17':
            x=best(o.Dm,{a:v[0] for a,v in b.items()}); return True,x,o.Dm-x
        if i=='Q18':
            return before[1]>0 and before[2]>0,{a for a,v in b.items() if v[1]<before[1] and v[2]==before[2]}, {a for a,v in b.items() if v[2]<before[2] and v[1]==before[1]}
        idx=3 if i=='Q19' else 4
        x={a for a,v in b.items() if v[idx]}; return before[idx],x,o.Dm-x
    if i=='Q21':
        middle=lambda k:k<27 and 1<=k%9<=7
        x=o.pick(lambda a:all(middle(z) for z,v in enumerate(post(o.c,a//4)) if v),o.Dm)
        return all(middle(z//4) for m in o.melds for z in m.tiles),x,o.Dm-x
    if i=='Q22':
        colors=[sum(o.c[z*9:z*9+9]) for z in range(3)]
        if colors.count(max(colors))!=1: return False,set(),set()
        main=colors.index(max(colors)); v={a:sum(n for z,n in enumerate(post(o.c,a//4)) if z<27 and z//9!=main) for a in o.Dm}
        x=best(o.Dm,v,False); y=o.max('U'); return not x&y,x,y
    if i=='Q24':
        if len(o.Dm)<2 or len({o.U[a] for a in o.Dm})!=1: return False,set(),set()
        return True,o.max('Z'),o.Dm-o.max('Z')
    if i=='Q25':
        if not s.get('rules_verified'): raise Unknown('red_rule_profile_unverified')
        if not s['red_enabled']: return False,set(),set()
        x=best(o.Dm,{a:sum(z in (16,52,88) for z in d-{a}) for a in o.Dm}); y=o.max('U')
        return not x&y,x,y
    if i=='D01': return len({o.U[a] for a in o.Dm})==1 and o.b_before==1,{a for a in o.Dm if o.B[a]==1},{a for a in o.Dm if o.B[a]<1}
    if i=='D02':
        vals={a:len({z//4 for z in g-{a}}) for a in o.Dm}; x=best(o.Dm,vals)
        return True,x,o.Dm-x
    if i=='D03': return True,{a for a in g if o.c[a//4]>=2},{a for a in g if o.c[a//4]==1}
    if i=='D04':
        x=best(g,{a:4-o.known[a//4] for a in g},False); return True,x,g-x
    if i in ('D05','D06','D11'):
        tied=len({o.U[a] for a in o.Dm})==1
        if i=='D06': return tied,g&o.Dm,(j&o.Dm)-g
        base=not g and all(a//4<27 for a in o.Dm) and tied
        if i=='D05': return base,j&o.Dm,o.Dm-j
        return base,{a for a in o.Dm if a//4%9 in (0,8)},{a for a in o.Dm if a//4%9 in (3,4,5)}
    if i=='D07':
        v={a:sum(a//4 in o.rivers[p] for p in o.opp) for a in d}; x=best(d,v)
        return not ga,x,d-x
    if i=='D08':
        if n>2: raise Unknown('D08_multiple_children_quantifier_undefined')
        parent=s['dealer']; child=o.opp-{parent}
        x={a for a in d if a//4 in o.rivers[parent] and all(a//4 not in o.rivers[p] for p in child)}
        y={a for a in d if a//4 not in o.rivers[parent] and any(a//4 in o.rivers[p] for p in child)}
        return not ga and parent in o.opp and bool(child),x,y
    if i=='D09':
        others=set(range(4))-{s['seat']}-o.opp
        v={a:sum(len({z//4 for z in d-{a}}&o.rivers[p]) for p in others) for a in g}; x=best(g,v)
        return True,x,g-x
    if i=='D12': return not g,{a for a in d if a//4>=27 and o.known[a//4]==3},{a for a in d if a//4>=27 and o.known[a//4] in (1,2)}
    if i=='D13':
        wall={a for a in d if wall_blocked(a//4,o.known)}
        return not g and all(a//4<27 for a in d),j-wall,wall-j
    if i=='D14': return o.same_target_previous() and o.prev['chosen_safe'],g&o.Dm,o.Dm-g
    if i=='D15': return o.same_target_previous() and o.prev['g_empty'] and bool(g),g,d-g
    if i in ('S01','S02','S03','S09','S10'):
        pts=s['points']; own=pts[s['seat']]; others=[v for p,v in enumerate(pts) if p!=s['seat']]
        if i=='S01': return own-max(others)>=12000 and not g&t,g,t
        if i=='S02': return min(others)-own>=8000 and not g&o.Dm,o.Dm,g
        if i=='S03': return own<=3900 and o.Smin>=1,g&o.Dm,o.Dm-g
        if not s.get('rules_verified'): raise Unknown('scheduled_round_profile_unverified')
        last=7 if s['hanchan'] else 3
        if i=='S09': return s['round_index']==last and own<min(others) and not o.max('U')&o.max('V'),o.max('V'),o.max('U')
        return (s['dealer']+1)%4==s['seat'] and s['round_index']<last,g&o.Dm,o.Dm-g
    if i.startswith('R') or i=='S04':
        desired=1 if i in ('R08','R09','R14') else 0
        if n!=desired: return False,set(),set()
        L=o.L; x={(a,True) for a in L}; y={(a,False) for a in L}; cond=True
        if not L: return False,x,y
        if i=='R05': cond=all(o.PF[a] for a in L)
        elif i=='R06': cond=all(o.U[a]==1 for a in L)
        elif i=='R07':
            cond=len({o.Z[a] for a in L})>1; y={(a,False) for a in best(L,o.Z)}
        elif i=='R08': cond=L<=g
        elif i=='R09': x={(a,True) for a in L&g}; y={(a,True) for a in L-g}
        elif i=='R12':
            own=s['points'][s['seat']]; other=[v for p,v in enumerate(s['points']) if p!=s['seat']]
            if own in other or own-1000 in other: raise Unknown('tie_rank_rule_not_reconstructed')
            cond=any(own-1000<v<own for v in other)
        elif i=='R13':
            if s.get('last_r0_pass') is None: raise Unknown('previous_riichi_legality_unknown')
            cond=s.get('last_r0_pass',False)
        elif i=='R14':
            if o.prev and o.prev.get('r0_pass') is None: raise Unknown('previous_riichi_legality_unknown')
            cond=bool(o.prev and o.prev['free'] and o.prev['r0_pass'])
        elif i=='R15': cond=s['wall']<=8
        elif i=='S04':
            own=s['points'][s['seat']]; below=[v for v in s['points'] if v<own]
            cond=s['points'].count(own)==1 and bool(below) and own-max(below)<=1000; x,y=y,x
        return cond,x,y
    raise AssertionError('registry has no rule: '+i)


def measure(i, options):
    if i in DEFERRED:
        return dict(status='unimplemented',reason=DEFERRED[i],numerator=None,denominator=None,other=None)
    if i in CALL_IDS:
        return dict(status='not_call_event',reason='counted_at_offer_response',numerator=0,denominator=0,other=0)
    if not options.s['free']:
        return dict(status='excluded',reason='|'.join(options.s['excluded']),numerator=0,denominator=0,other=0)
    try:
        scene,x,y=evaluate(i,options)
        if not scene or not x or not y:
            return dict(status='no_opportunity',reason='scene_or_two_choices_absent',numerator=0,denominator=0,other=0)
        if x&y: raise AssertionError('X and Y overlap '+i)
        actions={(a,b) for a in options.D for b in (False,True)} if isinstance(next(iter(x)),tuple) else options.D
        if not x|y <= actions: raise AssertionError('illegal candidate '+i)
        actual=(options.s['discard'],options.s['declaration']) if isinstance(next(iter(x)),tuple) else options.s['discard']
        if actual not in actions: raise ValueError('actual action outside legal candidates')
        return dict(status='ok',reason='included',numerator=int(actual in x),denominator=1,
                    other=int(actual not in x|y),X=sorted(x),Y=sorted(y),actual=actual)
    except Unknown as e:
        return dict(status='unknown',reason=str(e),numerator=None,denominator=None,other=None)
