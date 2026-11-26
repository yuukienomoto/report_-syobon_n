"""One target-private/public replay for all 100 candidates and parent indicators.

Only own INIT/draw contents are read. Opponent draw TAGS count wall progress;
their tile IDs, all terminal hand disclosures, final scores and future dora are
never used in a decision. Deferred call rules are never assigned fake zeros.
"""
from collections import Counter
from dataclasses import asdict
import gzip
import re
from urllib.parse import unquote
import xml.etree.ElementTree as ET
from mahjong_research.tenhou import ExtractionError, decode_meld, _counts
from . import VERSION
from .rules import CATALOG, IMPLEMENTED, DEFERRED, PARENTS, measure
from .options import Options, Unknown
from .calls import measure_call


def require(ok,message):
    if not ok: raise ExtractionError(message)


def extract(xml, player, expected_seat=None, expected_dan=None, game_id='', collect_trace=True):
    if xml[:2]==b'\x1f\x8b': xml=gzip.decompress(xml)
    root=ET.fromstring(xml)
    require(root.tag=='mjloggm','not mjloggm')
    un=next((e for e in root if e.tag=='UN' and 'dan' in e.attrib),None)
    require(un is not None,'missing UN')
    names=[unquote(un.get(f'n{i}','')) for i in range(4)]
    require(bool(player) and names.count(player)==1,'target absent/ambiguous')
    seat=names.index(player); ranks=list(map(int,un.attrib['dan'].split(',')))
    require(len(ranks)==4 and all(0<=v<=20 for v in ranks),'invalid ranks')
    dan=ranks[seat]
    require(expected_seat is None or seat==int(expected_seat),'C5 seat mismatch')
    require(expected_dan is None or dan==int(expected_dan),'C5 rank mismatch')
    go=next((e for e in root if e.tag=='GO'),None)
    require(go is not None,'missing GO')
    gt=int(go.attrib['type']); require(not gt&16,'sanma unsupported')
    # Standard dan-game bit and post-2009 ID; other profiles retain T/S features
    # but rule-dependent features become unknown. Never infer an era from rows.
    date=re.match(r'(\d{8})',game_id)
    rules_verified=bool(0<gt<256 and gt&1 and date and '20090801'<=date[1]<='20261126')
    counters={i:dict(numerator=0,denominator=0,other=0,unknown=0,excluded=0,censored=0) for i in (*IMPLEMENTED,*PARENTS)}
    reasons=Counter(); audit=Counter(); traces=[]; rounds=[]; calls=[]
    live=False; round_no=0; index=-1; tag=''; pending_offer=None
    def close_offer(status,event,actual_meld=None):
        nonlocal pending_offer
        if pending_offer is not None:
            result,detail=measure_call(pending_offer['state'],pending_offer['tile'],pending_offer['from_seat'],status,actual_meld)
            for i,v in result.items():
                if v['status']=='unknown': counters[i]['unknown']+=1; reasons[i+'|'+v['reason']]+=1
                elif v['status']=='censored': counters[i]['censored']+=1
                else:
                    for field in ('numerator','denominator','other'): counters[i][field]+=v[field]
                    round_counts[i][0]+=v['numerator']; round_counts[i][1]+=v['denominator']
            row={k:v for k,v in pending_offer.items() if k!='state'}
            row.update(response_status=status,response_event=event,features=result,choices=detail)
            if collect_trace:
                row['own_hand']=pending_offer['state']['hand']
                row['rivers']=[sorted(r) for r in pending_offer['state']['rivers']]
                row['known']=sorted(pending_offer['state']['known'])
                row['active']=sorted(pending_offer['state']['active'])
                calls.append(row)
            pending_offer=None
    try:
        for index,e in enumerate(root):
            tag,a=e.tag,e.attrib
            if tag=='REINIT': raise ExtractionError('REINIT: full INIT required')
            if tag=='INIT':
                require(not live,'INIT before round end')
                hand=list(map(int,a[f'hai{seat}'].split(','))); _counts(hand)
                require(len(hand)==len(set(hand))==13,'invalid own INIT hand')
                seed=list(map(int,a['seed'].split(',')))
                require(len(seed)==6,'invalid INIT seed')
                points=[int(v)*100 for v in a['ten'].split(',')]; require(len(points)==4,'invalid points')
                dealer=int(a['oya']); require(0<=dealer<4,'invalid dealer')
                round_index=seed[0]; require(round_index>=0,'negative round')
                honba,sticks=seed[1:3]
                indicators=[seed[5]]; _counts(indicators)
                require(not set(hand)&set(indicators),'indicator overlaps own hand')
                round_no+=1; audit['rounds']+=1
                rivers=[set() for _ in range(4)]; river_ids=set(); exposed=set()
                melds=[]; active=set(); declared=set(); declaration_discards=set(); ippatsu=set()
                drawn=None; kan=False; last_discard=None; previous=None; last_r0_pass=False
                wall=70; rinshan=None; live=True; own_draws=0; all_melds=[[] for _ in range(4)]
                round_counts={i:[0,0] for i in IMPLEMENTED}
            elif tag in ('AGARI','RYUUKYOKU'):
                close_offer('censored_terminal',index)
                if live:
                    rounds.extend(dict(round=round_no,feature_id=i,numerator=v[0],denominator=v[1]) for i,v in round_counts.items())
                live=False  # no terminal scores/disclosures enter predictor state
            elif not live:
                require(not re.fullmatch('[TUVWDEFGtuvwdefg][0-9]+',tag) and tag not in ('N','REACH','DORA'),'play outside round')
            elif tag=='DORA':
                tile=int(a['hai']); require(0<=tile<136 and tile not in indicators,'invalid/repeated DORA')
                require(tile not in set(hand)|exposed|river_ids,'DORA physically inconsistent')
                indicators.append(tile)
            elif tag=='REACH':
                who,step=int(a['who']),a['step']; require(0<=who<4,'invalid reach seat')
                if step=='1':
                    require(who not in declared,'duplicate declaration'); declared.add(who)
                elif step=='2':
                    require(who in declared and who in declaration_discards and who not in active,'acceptance without declaration discard')
                    active.add(who); ippatsu.add(who)
                    points[who]-=1000; sticks+=1
                    if 'ten' in a:
                        recorded=[int(v)*100 for v in a['ten'].split(',')]
                        require(recorded==points,'REACH public points mismatch')
                else: raise ExtractionError('unknown REACH step')
            elif tag=='N':
                who=int(a['who']); m=decode_meld(int(a['m']),who)
                fresh={m.added} if m.kind=='kakan' else set(m.tiles)-{m.called}
                require(not fresh & (river_ids|exposed|set(indicators)),'meld newly exposed tile already public')
                if m.kind in ('chi','pon','daiminkan'):
                    require(last_discard==(m.from_seat,m.called),'called tile mismatch')
                    close_offer('own_call' if who==seat else 'censored_competing_call',index,m if who==seat else None)
                else: close_offer('censored_other_action',index)
                audit['calls']+=1; ippatsu.clear()
                if m.kind in ('ankan','kakan','daiminkan'):
                    require(rinshan is None and wall>0,'kan wall/rinshan mismatch')
                    kan=True; audit['kans']+=1; wall-=1; rinshan=who
                if m.kind=='kakan':
                    old=[j for j,v in enumerate(all_melds[who]) if v.kind=='pon' and set(v.tiles)==set(m.tiles)-{m.added}]
                    require(len(old)==1,'kakan without pon'); all_melds[who][old[0]]=m
                else: all_melds[who].append(m)
                if who==seat:
                    drawn=None
                    for tile in ([m.added] if m.kind=='kakan' else [v for v in m.tiles if v!=m.called]): hand.remove(tile)
                    melds=list(all_melds[seat])
                else:
                    require(not set(m.tiles)&set(hand),'opponent meld overlaps own hand')
                exposed.update(m.tiles)
                owned=hand+[v for z in melds for v in z.tiles]; _counts(owned)
                require(len(owned)==len(set(owned)),'duplicate owned after meld')
            elif re.fullmatch('[TUVWtuvw][0-9]+',tag):
                who='TUVW'.index(tag[0].upper())
                close_offer('ordinary_progress',index)
                if rinshan is not None:
                    require(who==rinshan,'rinshan wrong seat'); rinshan=None
                else: wall-=1
                require(wall>=0,'negative live wall')
                if who==seat:
                    require(drawn is None,'own draw before previous discard')
                    drawn=int(tag[1:]); require(drawn not in set(hand)|river_ids|exposed|set(indicators),'own draw already visible')
                    hand.append(drawn); _counts(hand); own_draws+=1
                    require(len(hand)==14-3*len(melds),'draw hand size')
                # Do NOT parse opponent tile suffix; only public turn progression.
            elif re.fullmatch('[DEFGdefg][0-9]+',tag):
                who,tile='DEFG'.index(tag[0].upper()),int(tag[1:])
                require(0<=tile<136 and tile not in river_ids|exposed|set(indicators),'invalid public discard')
                if who==seat:
                    require(tile in hand and len(hand)==14-3*len(melds),'hand/discard mismatch')
                    excluded=[]
                    if seat in active: excluded.append('own_accepted_riichi')
                    if drawn is None: excluded.append('post_call_no_normal_draw')
                    if kan: excluded.append('kan_since_previous_own_discard')
                    state=dict(hand=list(hand),melds=tuple(melds),rivers=[set(r) for r in rivers],
                        active=set(active),seat=seat,discard=tile,drawn=drawn,free=not excluded,excluded=excluded,
                        previous=previous,last_r0_pass=last_r0_pass,points=list(points),dealer=dealer,
                        round_index=round_index,indicators=list(indicators),ippatsu=set(ippatsu),wall=wall,
                        declaration=seat in declared and seat not in active,known=set(hand)|exposed|river_ids|set(indicators),
                        rules_verified=rules_verified,red_enabled=not bool(gt&2),hanchan=bool(gt&8))
                    o=Options(state)
                    if state['free'] and state['declaration'] and rules_verified:
                        try: require(tile in o.L,'observed riichi discard not legal')
                        except Unknown: pass  # individual rule statuses retain the unknown
                    result={i:measure(i,o) for i in (*CATALOG,*PARENTS)}
                    for i,v in result.items():
                        if i in DEFERRED: continue
                        if v['status']=='unknown': counters[i]['unknown']+=1; reasons[i+'|'+v['reason']]+=1
                        elif v['status']=='excluded': counters[i]['excluded']+=1
                        else:
                            for field in ('numerator','denominator','other'): counters[i][field]+=v[field]
                            if i in round_counts:
                                round_counts[i][0]+=v['numerator']; round_counts[i][1]+=v['denominator']
                    if collect_trace:
                        serial={k:v for k,v in state.items() if k!='previous'}
                        serial['melds']=[asdict(m) for m in melds]
                        serial={k:sorted(v) if isinstance(v,set) else v for k,v in serial.items()}
                        serial['rivers']=[sorted(r) for r in rivers]
                        traces.append(dict(round=round_no,event=index,own_draws=own_draws,state=serial,options=o.details(),features=result))
                    previous=o.history()
                    if state['free'] and not o.opp:
                        try:
                            if o.L: last_r0_pass=not state['declaration']
                        except Unknown: last_r0_pass=None
                    hand.remove(tile); drawn=None; kan=False; audit['own_discards']+=1
                else:
                    require(tile not in hand,'opponent discard overlaps target hand')
                    # Record response observation/censorship; no inference of intent.
                    rr=[set(r) for r in rivers]; rr[who].add(tile//4)
                    offer_state=dict(hand=list(hand),melds=tuple(melds),rivers=rr,active=set(active),seat=seat,
                        dealer=dealer,round_index=round_index,indicators=list(indicators),wall=wall,
                        known=set(hand)|exposed|river_ids|set(indicators)|{tile},rules_verified=rules_verified,
                        kan_count=sum(m.kind in ('ankan','kakan','daiminkan') for ms in all_melds for m in ms),
                        honba=honba,sticks=sticks)
                    pending_offer=dict(round=round_no,event=index,from_seat=who,tile=tile,state=offer_state)
                rivers[who].add(tile//4); river_ids.add(tile); last_discard=(who,tile)
                if who in active: ippatsu.discard(who)
                if who in declared and who not in active: declaration_discards.add(who)
            elif tag in ('BYE','REJOIN'):
                close_offer('censored_connection',index)
        require(audit['rounds']>0 and not live and audit['own_discards']>0,'truncated/empty game')
    except (ValueError,KeyError,IndexError,TypeError) as e:
        raise ExtractionError(f'player={player} round={round_no} event={index} tag={tag}: {e}') from e
    for i in DEFERRED: counters[i]=dict(numerator=None,denominator=None,other=None,unknown=None,excluded=None,censored=None)
    for i,v in counters.items():
        v['status']='unimplemented' if i in DEFERRED else 'unknown' if v['unknown'] else 'ok' if v['denominator'] else 'no_opportunity'
        v['value']=v['numerator']/v['denominator'] if v['status']=='ok' else None
    return dict(player=player,seat=seat,dan=dan,version=VERSION,rules_verified=rules_verified,
        features=counters,audit=dict(audit),unknown_reasons=dict(reasons),trace=traces,round_counters=rounds,
        call_observations=calls,order_verified=False,full_history_confirmed=False)
