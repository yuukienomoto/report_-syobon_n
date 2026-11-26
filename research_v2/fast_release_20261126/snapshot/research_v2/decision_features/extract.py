"""Observable pre-discard state only; no opponent concealed hand is read.

The C5-signed pilot additionally checks the unchanged legacy replay. This replay
does not estimate shanten, score, live wall, yaku or discard quality.
"""
from collections import Counter
import gzip
import re
from urllib.parse import unquote
import xml.etree.ElementTree as ET

from choices import tenpai_options
from mahjong_research.tenhou import ExtractionError, decode_meld, is_suji, _counts
from .definitions import NAMES, VERSION


def require(ok, message):
    if not ok:
        raise ExtractionError(message)


def measure(hand, melds, rivers, opponents, discard):
    """Called only in F. Sets use physical tiles; shape results share tile kinds."""
    d = set(hand)
    require(discard in d and len(d) == len(hand), 'invalid candidate/actual discard')
    result = {f: dict(denominator=0, numerator=0, reason='opponent_count') for f in NAMES}
    detail = dict(genbutsu=[], tenpai=[], suji=[])
    if len(opponents) == 1:
        river = rivers[next(iter(opponents))]
        g = {t for t in d if t//4 in river}
        options = tenpai_options(hand, melds)
        tenpai = {t for t in d if options[t//4]}
        suji = {t for t in d if is_suji(t//4, river)}
        detail = dict(genbutsu=sorted(g), tenpai=sorted(tenpai), suji=sorted(suji))
        checks = [
            ([(bool(tenpai), 'no_tenpai_option'), (bool(g), 'no_genbutsu'),
              (bool(d-g), 'only_genbutsu')], discard not in g),
            ([(bool(g), 'no_genbutsu'), (bool(tenpai), 'no_tenpai_option'),
              (not(g & tenpai), 'genbutsu_can_keep_tenpai')], discard in g),
            ([(not g, 'genbutsu_available'), (all(t//4 < 27 for t in d), 'honor_candidate'),
              (bool(suji), 'no_full_suji'), (bool(d-suji), 'only_full_suji')], discard in suji),
        ]
        for f, (conditions, chosen) in zip(NAMES[:3], checks):
            reasons = [reason for ok, reason in conditions if not ok]
            result[f] = dict(denominator=int(not reasons), numerator=int(not reasons and chosen),
                             reason='|'.join(reasons) or 'included')
    elif len(opponents) >= 2:
        g = {t for t in d if all(t//4 in rivers[p] for p in opponents)}
        reasons = ([] if g else ['no_common_genbutsu']) + ([] if d-g else ['only_common_genbutsu'])
        result[NAMES[3]] = dict(denominator=int(not reasons), numerator=int(not reasons and discard in g),
                               reason='|'.join(reasons) or 'included')
        detail['genbutsu'] = sorted(g)
    return result, detail


def extract(xml, player, expected_seat=None, expected_dan=None):
    """Return counters, audit and pre-discard trace, or fail the entire game.

    Trace includes excluded own discards. A failure never becomes zero exposure.
    round is INIT ordinal (1-based); event is XML child index (0-based).
    """
    if xml[:2] == b'\x1f\x8b':
        xml = gzip.decompress(xml)
    root = ET.fromstring(xml)
    require(root.tag == 'mjloggm', 'not mjloggm')
    un = next((e for e in root if e.tag == 'UN' and 'dan' in e.attrib), None)
    require(un is not None, 'missing UN')
    names = [unquote(un.get(f'n{i}', '')) for i in range(4)]
    require(bool(player) and names.count(player) == 1, 'target player absent/ambiguous')
    seat = names.index(player)
    ranks = list(map(int, un.attrib['dan'].split(',')))
    require(len(ranks) == 4 and all(0 <= r <= 20 for r in ranks), 'invalid dan')
    dan = ranks[seat]
    require(expected_seat is None or seat == int(expected_seat), 'C5 seat mismatch')
    require(expected_dan is None or dan == int(expected_dan), 'C5 dan mismatch')
    go = next((e for e in root if e.tag == 'GO'), None)
    require(go is not None and not(int(go.get('type', '0')) & 16), 'missing GO / sanma unsupported')
    counts = {f+s: 0 for f in NAMES for s in ('__numerator', '__denominator')}
    audit = Counter(rounds=0, own_discards=0, free_responses=0, calls=0, kans=0, multi_riichi_discards=0)
    trace, live, round_no = [], False, 0
    index, t = -1, ''
    try:
        for index, e in enumerate(root):
            t, a = e.tag, e.attrib
            if t == 'REINIT':
                raise ExtractionError('REINIT unsupported; full INIT required')
            if t == 'INIT':
                require(not live, 'INIT before round end')
                hand = list(map(int, a[f'hai{seat}'].split(',')))
                _counts(hand)
                require(len(hand) == len(set(hand)) == 13, 'INIT own hand invalid')
                round_no += 1
                audit['rounds'] += 1
                round_seed = a.get('seed', '').split(',')[0]
                rivers, melds, active, declared, declaration_discards = [set() for _ in range(4)], [], set(), set(), set()
                drawn, kan, last_discard, live = None, False, None, True
            elif t in ('AGARI', 'RYUUKYOKU'):
                live = False
            elif not live:
                require(not re.fullmatch('[TUVWDEFGtuvwdefg][0-9]+', t) and t not in ('N', 'REACH'),
                        'play event outside round')
            elif t == 'REACH':
                who, step = int(a['who']), a['step']
                require(0 <= who < 4, 'invalid REACH seat')
                if step == '1':
                    require(who not in declared, 'duplicate declaration')
                    declared.add(who)
                elif step == '2':
                    require(who in declared and who in declaration_discards and who not in active,
                            'acceptance without declaration discard / repeated')
                    active.add(who)
                else:
                    raise ExtractionError('unknown REACH step')
            elif t == 'N':
                who = int(a['who'])
                m = decode_meld(int(a['m']), who)
                if m.kind in ('chi', 'pon', 'daiminkan'):
                    require(last_discard == (m.from_seat, m.called), 'called tile mismatch')
                audit['calls'] += 1
                if m.kind in ('ankan', 'kakan', 'daiminkan'):
                    kan = True
                    audit['kans'] += 1
                if who == seat:
                    drawn = None
                    if m.kind == 'kakan':
                        positions = [i for i, old in enumerate(melds)
                                     if old.kind == 'pon' and set(old.tiles) == set(m.tiles)-{m.added}]
                        require(len(positions) == 1, 'kakan missing matching pon')
                        melds[positions[0]] = m
                        hand.remove(m.added)
                    else:
                        melds.append(m)
                        for tile in m.tiles:
                            if tile != m.called:
                                hand.remove(tile)
                    owned = hand + [tile for m in melds for tile in m.tiles]
                    _counts(owned)
                    require(len(owned) == len(set(owned)), 'duplicate owned physical tile after meld')
            elif re.fullmatch('[TUVWtuvw][0-9]+', t):
                who = 'TUVW'.index(t[0].upper())
                if who == seat:
                    require(drawn is None, 'own draw before previous discard')
                    drawn = int(t[1:])
                    hand.append(drawn)
                    owned = hand + [tile for m in melds for tile in m.tiles]
                    _counts(owned)
                    require(len(hand) == 14-3*len(melds) and len(owned) == len(set(owned)), 'draw hand mismatch')
                # Opponent draw tile content is deliberately not parsed or used.
            elif re.fullmatch('[DEFGdefg][0-9]+', t):
                who, tile = 'DEFG'.index(t[0].upper()), int(t[1:])
                require(0 <= tile < 136, 'public discard out of range')
                if who == seat:
                    owned = hand + [z for m in melds for z in m.tiles]
                    _counts(owned)
                    require(tile in hand and len(hand) == 14-3*len(melds)
                            and len(owned) == len(set(owned)), 'hand/discard mismatch')
                    opponents = active-{seat}
                    reasons = []
                    if seat in active: reasons.append('own_accepted_riichi')
                    if drawn is None: reasons.append('no_normal_draw_post_call')
                    if kan: reasons.append('kan_since_previous_own_discard')
                    if not opponents: reasons.append('no_opponent_accepted_riichi')
                    if reasons:
                        result = {f: dict(numerator=0, denominator=0, reason='|'.join(reasons)) for f in NAMES}
                        detail = dict(genbutsu=[], tenpai=[], suji=[])
                    else:
                        result, detail = measure(hand, melds, rivers, opponents, tile)
                        audit['free_responses'] += 1
                    if len(opponents) >= 2: audit['multi_riichi_discards'] += 1
                    for f, r in result.items():
                        counts[f+'__numerator'] += r['numerator']
                        counts[f+'__denominator'] += r['denominator']
                    trace.append(dict(round=round_no, round_seed=round_seed, event=index, seat=seat,
                        hand=list(hand), melds=[dict(kind=m.kind, tiles=list(m.tiles)) for m in melds],
                        rivers=[sorted(r) for r in rivers], active=sorted(active), drawn=drawn, discard=tile,
                        own_declaration=seat in declared and seat not in active, excluded=reasons,
                        **detail, features=result))
                    hand.remove(tile)
                    drawn, kan = None, False
                    audit['own_discards'] += 1
                rivers[who].add(tile//4)
                last_discard = who, tile
                if who in declared and who not in active:
                    declaration_discards.add(who)
        require(audit['rounds'] > 0 and not live and audit['own_discards'] > 0, 'truncated/empty game')
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise ExtractionError(f'player={player} seat={seat} round={round_no} event={index} tag={t}: {exc}') from exc
    return dict(player=player, seat=seat, dan=dan, version=VERSION, counts=counts, audit=dict(audit), trace=trace)
