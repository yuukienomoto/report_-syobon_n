"""Choices immediately before a discard in four-player, C5-verified logs."""
from collections import Counter
from urllib.parse import unquote
import xml.etree.ElementTree as ET
import re

from mahjong_research.tenhou import decode_meld, wait_shapes

NAMES = ('first_mixed_genbutsu', 'switch_to_held_genbutsu', 'choose_structural_tenpai')
DEFINITIONS = (
    ('初回混合手牌の現物選択率','実際の初回応答で現物と非現物がある','現物を切った'),
    ('非現物ツモ時の手持ち現物選択率','非現物をツモし手持ち現物がある','手持ち現物を切った'),
    ('形状テンパイ選択率','打牌により形状テンパイ・ノーテンの両方を選べる','形状テンパイを選んだ'))
RULE = '通常ツモ後、単独他家成立リーチ、自己成立前。鳴き直後・直前の自己打牌（局頭）以降カンありを除外。宣言打牌は宣言を含む選択として対象。現物はリーチ者自身の河。'


def tenpai_options(hand, melds):
    options = {}
    for i, tile in enumerate(hand):
        kind = tile // 4
        if kind not in options:
            options[kind] = bool(wait_shapes(hand[:i]+hand[i+1:], melds))
    return options


def extract(xml, seat, dan, trace=None):
    root = ET.fromstring(xml)
    un = next(e for e in root if e.tag == 'UN' and 'dan' in e.attrib)
    if (unquote(un.attrib[f'n{seat}']) != 'しょぼんｎ' or int(un.attrib['dan'].split(',')[seat]) != dan):
        raise ValueError('C5 seat/player/dan mismatch')
    counts = {f+s:0 for f in NAMES for s in ('__numerator','__denominator')}
    audit = Counter(rounds=0,own_discards=0,free_responses=0,excluded_responses=0)
    live = False
    for index,e in enumerate(root):
        t, a = e.tag, e.attrib
        if t == 'INIT':
            hand = list(map(int,a[f'hai{seat}'].split(',')))
            rivers = [set() for _ in range(4)]
            melds, active, pending = [], set(), set()
            drawn, kan = None, False
            last_discard = None
            live = True
            audit['rounds'] += 1
        elif t in ('AGARI','RYUUKYOKU'):
            live = False
        elif not live:
            continue
        elif t == 'REACH' and a['step'] == '2':
            who = int(a['who'])
            active.add(who)
            pending.add(who)
        elif t == 'N':
            who = int(a['who'])
            m = decode_meld(int(a['m']),who)
            if m.kind in ('chi','pon','daiminkan') and last_discard != (m.from_seat,m.called):
                raise ValueError(f'called tile mismatch event {index}')
            if m.kind in ('ankan','kakan','daiminkan'):
                kan = True
            if who == seat:
                drawn = None
                if m.kind == 'kakan':
                    pos, = [i for i,old in enumerate(melds) if old.kind == 'pon' and set(old.tiles)==set(m.tiles)-{m.added}]
                    melds[pos] = m
                    hand.remove(m.added)
                else:
                    melds.append(m)
                    for tile in m.tiles:
                        if tile != m.called:
                            hand.remove(tile)
        elif re.fullmatch('[TUVWtuvw][0-9]+',t):
            if 'TUVW'.index(t[0].upper()) == seat:
                drawn = int(t[1:])
                hand.append(drawn)
        elif re.fullmatch('[DEFGdefg][0-9]+',t):
            who, tile = 'DEFG'.index(t[0].upper()), int(t[1:])
            if who == seat:
                owned = hand+[z for m in melds for z in m.tiles]
                if tile not in hand or len(hand)!=14-3*len(melds) or len(owned)!=len(set(owned)):
                    raise ValueError(f'hand/discard mismatch event {index}')
                if len(active)==1 and seat not in active and drawn is not None and not kan:
                    opponent, = active
                    river = rivers[opponent]
                    safe = {z for z in hand if z//4 in river}
                    options = tenpai_options(hand,melds)
                    observed = [opponent in pending and 0<len(safe)<len(hand),
                                drawn not in safe and bool(safe), len(set(options.values()))==2]
                    outcomes = [tile in safe,tile in safe,options[tile//4]]
                    for f,opp,chosen in zip(NAMES,observed,outcomes):
                        if opp:
                            counts[f+'__denominator'] += 1
                            counts[f+'__numerator'] += int(chosen)
                    if trace is not None:
                        trace.append(dict(event=index,round=audit['rounds'],opponent=opponent,
                            hand=list(hand),melds=[list(m.tiles) for m in melds],river=sorted(river),
                            drawn=drawn,discard=tile,first=opponent in pending,
                            tenpai_discards=[k for k,v in options.items() if v],opportunities=observed,
                            choices=outcomes))
                    audit['free_responses'] += 1
                elif active:
                    audit['excluded_responses'] += 1
                pending.clear()
                hand.remove(tile)
                drawn,kan = None,False
                audit['own_discards'] += 1
            rivers[who].add(tile//4)
            last_discard = who,tile
    if live or not audit['own_discards']:
        raise ValueError('incomplete log')
    return dict(counts, **audit)
