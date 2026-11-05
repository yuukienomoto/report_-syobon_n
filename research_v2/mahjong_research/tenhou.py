"""Auditable per-game features, version 2; no legacy cache masquerading as v2.

Only the target's concealed hand and past public events are used. Tenpai means
structural completion (not yaku, furiten, or live-wall availability). The N bit
layout was checked against MahjongRepository/tenhou-python-bot's decoder:
https://github.com/MahjongRepository/tenhou-python-bot/blob/master/project/tenhou/decoder.py
Completion scope was checked against MahjongRepository/mahjong's Agari API:
https://github.com/MahjongRepository/mahjong/blob/master/mahjong/agari.py
The recursive completion/decomposition implementation below is independent.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import gzip
import json
import math
import re
from urllib.parse import unquote
import xml.etree.ElementTree as ET

EXTRACTOR_VERSION = "tactical-v2.0.0"


class ExtractionError(ValueError):
    """Unreconstructable/unsupported game; do not silently fill its features."""


class PlayerNotFound(ExtractionError):
    pass


def _require(condition, message):
    if not condition:
        raise ExtractionError(message)


@dataclass(frozen=True)
class Meld:
    kind: str
    tiles: tuple[int, ...]
    called: int | None
    from_seat: int
    added: int | None = None


def decode_meld(code: int, who: int) -> Meld:
    """Decode 136-tile chi/pon/daiminkan/ankan/kakan; reject sanma nuki."""
    _require(0 <= who <= 3 and code >= 0, "invalid N metadata")
    source = (who + (code & 3)) % 4
    if code & 4:
        packed = code >> 10
        sequence, called_index = divmod(packed, 3)
        start = (sequence // 7) * 9 + sequence % 7
        tiles = tuple(4 * (start + i) + ((code >> (3 + 2 * i)) & 3) for i in range(3))
        result = Meld("chi", tiles, tiles[called_index], source)
    elif code & 24:
        omitted = (code >> 5) & 3
        tile_type, called_index = divmod(code >> 9, 3)
        original = tuple(tile_type * 4 + i for i in range(4) if i != omitted)
        if code & 8:
            result = Meld("pon", original, original[called_index], source)
        else:
            added = tile_type * 4 + omitted
            result = Meld("kakan", tuple(sorted(original + (added,))), original[called_index], source, added)
    elif code & 32:
        raise ExtractionError("sanma/nuki is outside the four-player protocol")
    else:
        tile_type, copy = divmod(code >> 8, 4)
        tiles = tuple(tile_type * 4 + i for i in range(4))
        result = Meld("ankan" if source == who else "daiminkan", tiles,
                      None if source == who else tiles[copy], source)
    _require(all(0 <= t < 136 for t in result.tiles), "N tile out of range")
    if result.kind in ("chi", "pon", "daiminkan", "kakan"):
        _require(source != who, "open N encoded as self-call")
    return result


def _counts(tiles):
    counts = [0] * 34
    for tile in tiles:
        _require(isinstance(tile, int) and 0 <= tile < 136, "tile outside 136 range")
        counts[tile // 4] += 1
    _require(max(counts, default=0) <= 4, "more than four copies of a tile")
    return tuple(counts)


@lru_cache(maxsize=100000)
def _sets(counts: tuple[int, ...]):
    """All standard meld decompositions, deterministic order."""
    first = next((i for i, n in enumerate(counts) if n), None)
    if first is None:
        return ((),)
    result = []
    work = list(counts)
    if work[first] >= 3:
        work[first] -= 3
        result.extend(((first, first, first),) + tail for tail in _sets(tuple(work)))
        work[first] += 3
    if first < 27 and first % 9 <= 6 and work[first + 1] and work[first + 2]:
        for t in (first, first + 1, first + 2):
            work[t] -= 1
        result.extend(((first, first + 1, first + 2),) + tail for tail in _sets(tuple(work)))
    return tuple(result)


@lru_cache(maxsize=50000)
def _completions(counts: tuple[int, ...], meld_count: int):
    if sum(counts) != 14 - 3 * meld_count or max(counts) > 4:
        return ()
    result = []
    if meld_count == 0:
        if sum(n == 2 for n in counts) == 7:
            result.append(("chiitoitsu", ()))
        terminals = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)
        if all(counts[t] for t in terminals) and sum(counts[t] for t in terminals) == 14:
            result.append(("kokushi", ()))
    for pair in range(34):
        if counts[pair] >= 2:
            work = list(counts)
            work[pair] -= 2
            for groups in _sets(tuple(work)):
                if len(groups) == 4 - meld_count:
                    result.append((pair, groups))
    return tuple(result)


def wait_shapes(hand: list[int], melds: list[Meld] | tuple[Meld, ...] = ()) -> dict[int, tuple[str, ...]]:
    """Winning type -> possible roles, not mutually exclusive hand categories.

    Counts own concealed+meld tiles to rule out a fifth own copy. Other players'
    concealed tiles and future events never enter the calculation. Exhausted
    visible waits remain structural waits; no assertion about actual wins.
    """
    _require(len(hand) == 13 - 3 * len(melds), "post-discard concealed tile count mismatch")
    owned = hand + [t for m in melds for t in m.tiles]
    _require(len(owned) == len(set(owned)), "duplicate owned 136 tile")
    occupied, concealed = _counts(owned), _counts(hand)
    waits = {}
    for win in range(34):
        if occupied[win] >= 4:
            continue
        candidate = list(concealed)
        candidate[win] += 1
        shapes = set()
        for pair, groups in _completions(tuple(candidate), len(melds)):
            if isinstance(pair, str):
                shapes.add(pair)
                continue
            if pair == win:
                shapes.add("tanki")
            for group in groups:
                if win not in group:
                    continue
                if group[0] == group[2]:
                    shapes.add("shanpon")
                elif win == group[1]:
                    shapes.add("kanchan")
                elif (win == group[2] and group[0] % 9 == 0) or (win == group[0] and group[2] % 9 == 8):
                    shapes.add("penchan")
                else:
                    shapes.add("ryanmen")
        if shapes:
            waits[win] = tuple(sorted(shapes))
    return waits


def is_suji(tile_type: int, river_types: set[int]) -> bool:
    """Full-suji against this river; middle 4/5/6 need both outside anchors.

    This addresses two-sided sequence waits only, never proves safety.
    Honors are outside suji classification.
    """
    if tile_type >= 27:
        return False
    rank = tile_type % 9
    if rank <= 2:
        return tile_type + 3 in river_types
    if rank >= 6:
        return tile_type - 3 in river_types
    return tile_type - 3 in river_types and tile_type + 3 in river_types


def dora_count(tiles: list[int], indicators: list[int]) -> int:
    """Visible indicator dora only; duplicate indicators count repeatedly."""
    bonus = []
    for indicator in indicators:
        _require(0 <= indicator < 136, "invalid dora indicator")
        t = indicator // 4
        bonus.append((t // 9) * 9 + (t + 1) % 9 if t < 27 else
                     27 + (t - 27 + 1) % 4 if t < 31 else 31 + (t - 31 + 1) % 3)
    return sum(tile // 4 == d for tile in tiles for d in bonus)


def _definition(name, japanese, numerator, denominator, scope, limitation, upper=1):
    return dict(feature=name, japanese=japanese, numerator=numerator, denominator=denominator,
                formula=f"{numerator} / {denominator}", unit="対局内の機会割合" if upper == 1 else "対局内の機会平均",
                events=scope, missing_condition="分母0。復元不整合はゲーム全体をエラーとして別途監査",
                limitation=limitation, lower=0, upper=upper, version=EXTRACTOR_VERSION)


FEATURE_DEFINITIONS = [
    _definition("first_response_genbutsu_rate", "成立リーチごとの初回応答現物率", "初回応答で相手の河にある牌を打った数", "他家成立リーチ後に実際に生じた初回打牌・相手ペア数", "REACH step2,捨牌", "河は判断時点のその相手自身の河のみ。複数リーチ時は相手ごとに計数"),
    _definition("response_non_genbutsu_rate", "対リーチ応答非現物率", "相手の河にない牌の応答数", "他家成立リーチ中の打牌・相手ペア数", "REACH step2,捨牌", "非現物は危険牌や押しの意図と同義ではない"),
    _definition("response_suji_nongenbutsu_rate", "対リーチ応答非現物スジ率", "非現物かつ完全スジ数牌の応答数", "他家成立リーチ中の打牌・相手ペア数", "REACH step2,捨牌", "4/5/6は両側のスジが必要。スジは安全を保証せず字牌は対象外"),
    _definition("response_nonsuji_suited_rate", "対リーチ応答非現物非スジ数牌率", "非現物かつ完全スジでない数牌の応答数", "他家成立リーチ中の打牌・相手ペア数", "REACH step2,捨牌", "字牌は分子外。片スジはこの区分に含む"),
]
for period, japanese in (("early", "自己ツモ6回以下"), ("mid", "自己ツモ7～12回"), ("late", "自己ツモ13回以上")):
    FEATURE_DEFINITIONS.append(_definition(f"{period}_noten_non_genbutsu_rate", f"{japanese}・打牌後形状ノーテン非現物率", "形状ノーテンでの非現物応答数", f"{japanese}で打牌後が形状ノーテンの対リーチ応答数", "INIT,ツモ,捨牌,N,REACH step2", "副露を含む打牌後の和了形候補なし。役・フリテン・残り牌を判定しない。打牌前にテンパイ選択があったかは測らない"))
for rank, japanese in (("top", "単独1位"), ("middle", "同点なし2・3位"), ("last", "単独4位")):
    FEATURE_DEFINITIONS.append(_definition(f"roundstart_{rank}_non_genbutsu_rate", f"局開始時{japanese}の非現物応答率", "該当局開始順位での非現物応答数", "該当局開始順位での対リーチ応答数", "INIT ten,REACH step2,捨牌", "自分と同点の他家がいる局は除外。局中の供託変化・順位変動は反映しない"))
FEATURE_DEFINITIONS += [
    _definition("riichi_omote_dora_mean", "成立リーチ時の表ドラ枚数平均", "宣言打牌後手牌と暗槓内の表ドラ枚数合計", "自分の成立リーチ数", "REACH step1/2,宣言打牌,DORA,N", "宣言打牌時点の表示ドラ。赤・裏ドラを除く。打点そのものではない", 20),
    _definition("chasing_riichi_share", "自分の成立リーチに占める追っかけ割合", "宣言時点で他家成立リーチがある自分の成立リーチ数", "自分の成立リーチ数", "REACH step1/2", "リーチ可能機会に対する選択率ではない"),
    _definition("chasing_wait_types_mean", "成立追っかけリーチの形状待ち種類数平均", "宣言打牌後の和了形を完成する牌種数合計", "自分の成立追っかけリーチ数", "INIT,ツモ,捨牌,N,REACH step1/2", "残り枚数・役・フリテン・他家手牌は不使用", 34),
    _definition("chasing_ryanmen_compatible_rate", "成立追っかけリーチの両面解釈可能率", "少なくとも1待ち・1分解で両面となる追っかけ数", "自分の成立追っかけリーチ数", "INIT,ツモ,捨牌,N,REACH step1/2", "複合形の重複解釈を許す。純粋両面率でも一意な良形率でもない"),
    _definition("chasing_zero_omote_dora_rate", "成立追っかけリーチの表ドラ0割合", "宣言打牌後の表ドラ枚数が0の追っかけ数", "自分の成立追っかけリーチ数", "REACH,宣言打牌,DORA,N", "赤ドラ・役・裏ドラを含まない。低打点率とは呼ばない"),
    _definition("open_meld_round_rate", "1回以上副露した局の割合", "チー・ポン・大明槓が1回以上ある局数", "INIT局数", "INIT,N", "暗槓を除く。加槓は既存ポン局を重複計数しない"),
    _definition("early_first_open_meld_rate", "自己ツモ6回以下の初副露局割合", "初副露時自己ツモ6回以下の局数", "INIT局数", "INIT,N,自己ツモ", "一般の巡目ではなく自己ツモ回数。暗槓は初副露を進めない"),
    _definition("late_first_open_meld_rate", "自己ツモ13回以上の初副露局割合", "初副露時自己ツモ13回以上の局数", "INIT局数", "INIT,N,自己ツモ", "一般の巡目ではなく自己ツモ回数"),
    _definition("first_open_meld_with_omote_dora_rate", "初副露成立直後に表ドラを持つ局割合", "初副露直後の手牌と面子内に表ドラがある局数", "INIT局数", "INIT,N,DORA", "呼んだ牌も含む。副露局を分母にした条件付き率ではない"),
    _definition("exhaustive_draw_disclosed_tenpai_rate", "通常流局時の手牌開示率", "通常流局で自分のhai属性が非空の局数", "type属性が空のRYUUKYOKU局数", "RYUUKYOKU", "途中流局・特殊流局を除く。開示された手牌は復元手牌・形状テンパイと照合。守備成功率ではない"),
]
FEATURE_NAMES = [d["feature"] for d in FEATURE_DEFINITIONS]


def extract_game(xml_bytes: bytes, player: str) -> dict:
    """Extract one full four-player log; accepts XML or gzip bytes.

    Errors invalidate the game rather than returning partially corrupted values.
    The caller must retain the game's ledger row and error, never close gaps by
    silently deleting it. This function never infers chronology or completeness.
    """
    if xml_bytes[:2] == b"\x1f\x8b":
        xml_bytes = gzip.decompress(xml_bytes)
    root = ET.fromstring(xml_bytes)
    _require(root.tag == "mjloggm", "not a mjloggm root")
    users = next((e for e in root if e.tag == "UN" and "dan" in e.attrib), None)
    _require(users is not None, "missing full UN metadata")
    names = [unquote(users.get(f"n{i}", ""), errors="strict") for i in range(4)]
    if names.count(player) == 0:
        raise PlayerNotFound(f"exact player name absent: {player}")
    _require(names.count(player) == 1, "ambiguous target player seats")
    seat = names.index(player)
    ranks = [int(t) for t in users.attrib["dan"].split(",")]
    _require(len(ranks) == 4 and all(0 <= t <= 20 for t in ranks), "invalid dan")
    go = next((e for e in root if e.tag == "GO"), None)
    _require(go is not None and not (int(go.get("type", "0")) & 16), "missing GO or unsupported three-player game")
    nums, dens, audit = Counter(), Counter(), Counter()
    live = False
    pending_own = None
    last_discard = None
    hand, melds, rivers, indicators = [], [], {}, []
    active, first_pending = set(), set()

    def observe(feature, value):
        nums[feature] += value
        dens[feature] += 1

    def validate_owned():
        owned = hand + [t for m in melds for t in m.tiles]
        _require(len(owned) == len(set(owned)), "duplicate owned 136 tile")
        _counts(owned)

    def validate_agari(attributes):
        if int(attributes.get("who", "-1")) != seat:
            return
        _require(bool(attributes.get("hai")), "own AGARI missing disclosed hand")
        shown = sorted(int(t) for t in attributes["hai"].split(","))
        candidate = list(hand)
        if int(attributes.get("fromWho", "-1")) != seat:
            candidate.append(int(attributes["machi"]))
        _require(sorted(candidate) == shown, "AGARI disclosed hand mismatch")
        _require(bool(_completions(_counts(candidate), len(melds))), "AGARI hand not complete")
        audit["agari_hand_checks"] += 1

    for event_index, event in enumerate(root):
        tag, a = event.tag, event.attrib
        if tag == "REINIT":
            raise ExtractionError("REINIT reconnect state unsupported; full INIT log required")
        if tag == "INIT":
            _require(not live, "new INIT before previous round end")
            hand = [int(t) for t in a.get(f"hai{seat}", "").split(",") if t]
            _require(len(hand) == 13, "INIT target hand must contain 13 tiles")
            melds = []
            validate_owned()
            seed = [int(t) for t in a.get("seed", "").split(",") if t]
            _require(len(seed) >= 6, "missing INIT dora")
            indicators = [seed[5]]
            _require(0 <= seed[5] < 136, "bad initial dora")
            scores = [int(t) for t in a.get("ten", "").split(",") if t]
            _require(len(scores) == 4, "missing round-start scores")
            rank = None if scores.count(scores[seat]) > 1 else 1 + sum(t > scores[seat] for t in scores)
            rivers = {i: set() for i in range(4)}
            active, first_pending, declared = set(), set(), set()
            declaration_discards = set()
            draw_count, opened = 0, False
            pending_own, last_discard = None, None
            live = True
            audit["rounds"] += 1
            for f in ("open_meld_round_rate", "early_first_open_meld_rate", "late_first_open_meld_rate", "first_open_meld_with_omote_dora_rate"):
                dens[f] += 1
            continue
        if not live:
            # Consecutive AGARI tags represent multiple ron. The target may be
            # listed second, so still compare its disclosed hand with the replay.
            if tag == "AGARI":
                validate_agari(a)
            continue
        if tag == "DORA":
            indicator = int(a["hai"])
            _require(0 <= indicator < 136, "invalid DORA")
            _require(indicator not in indicators and len(indicators) < 5, "duplicate or excess DORA indicator")
            indicators.append(indicator)
        elif tag == "REACH":
            who, step = int(a["who"]), a["step"]
            _require(0 <= who <= 3, "REACH seat invalid")
            if step == "1":
                _require(who not in declared, "duplicate REACH declaration")
                declared.add(who)
                if who == seat:
                    pending_own = {"chasing": bool(active - {seat}), "snapshot": None}
            elif step == "2":
                _require(who in declared and who not in active and who in declaration_discards,
                         "REACH acceptance without declaration discard or repeated")
                if who == seat:
                    _require(pending_own is not None and pending_own["snapshot"] is not None, "missing declaration discard")
                    waiting, bonus = pending_own["snapshot"]
                    _require(bool(waiting), "accepted riichi has no structural wait")
                    _require(all(m.kind == "ankan" for m in melds), "riichi after open meld")
                    observe("riichi_omote_dora_mean", bonus)
                    observe("chasing_riichi_share", pending_own["chasing"])
                    audit["riichi_hand_checks"] += 1
                    if pending_own["chasing"]:
                        observe("chasing_wait_types_mean", len(waiting))
                        roles = set(role for possible in waiting.values() for role in possible)
                        observe("chasing_ryanmen_compatible_rate", "ryanmen" in roles)
                        observe("chasing_zero_omote_dora_rate", bonus == 0)
                        for role in roles:
                            audit[f"chasing_shape_{role}"] += 1
                    pending_own = None
                else:
                    first_pending.add(who)
                active.add(who)
            else:
                raise ExtractionError("unknown REACH step")
        elif tag == "N":
            who = int(a["who"])
            meld = decode_meld(int(a["m"]), who)
            if meld.kind in ("chi", "pon", "daiminkan"):
                _require(last_discard == (meld.from_seat, meld.called), "called tile does not match last discard")
            if who != seat:
                continue
            audit[f"own_meld_{meld.kind}"] += 1
            if meld.kind == "kakan":
                matching = [i for i, old in enumerate(melds) if old.kind == "pon" and set(old.tiles) == set(meld.tiles) - {meld.added}]
                _require(len(matching) == 1, "kakan has no matching pon")
                remove = [meld.added]
                melds[matching[0]] = meld
            else:
                remove = [t for t in meld.tiles if t != meld.called]
                melds.append(meld)
            for tile in remove:
                _require(tile in hand, f"meld consumes absent tile {tile}")
                hand.remove(tile)
            validate_owned()
            if meld.kind in ("chi", "pon", "daiminkan") and not opened:
                opened = True
                nums["open_meld_round_rate"] += 1
                nums["early_first_open_meld_rate"] += draw_count <= 6
                nums["late_first_open_meld_rate"] += draw_count >= 13
                nums["first_open_meld_with_omote_dora_rate"] += dora_count(hand + [t for m in melds for t in m.tiles], indicators) > 0
        elif re.fullmatch(r"[TUVWtuvw]\d+", tag):
            who, tile = "TUVW".index(tag[0].upper()), int(tag[1:])
            _require(0 <= tile < 136, "draw tile invalid")
            if who == seat:
                hand.append(tile)
                draw_count += 1
                validate_owned()
                _require(len(hand) == 14 - 3 * len(melds), "draw hand length mismatch")
        elif re.fullmatch(r"[DEFGdefg]\d+", tag):
            who, tile = "DEFG".index(tag[0].upper()), int(tag[1:])
            _require(0 <= tile < 136, "discard tile invalid")
            if who == seat:
                _require(tile in hand, f"discard consumes absent tile {tile} at event {event_index}")
                hand.remove(tile)
                _require(len(hand) == 13 - 3 * len(melds), "discard hand length mismatch")
                validate_owned()
                audit["discard_hand_checks"] += 1
                opponents = sorted(active - {seat})
                waiting = wait_shapes(hand, melds) if opponents or pending_own is not None else None
                if pending_own is not None:
                    _require(pending_own["snapshot"] is None, "multiple discards before riichi acceptance")
                    pending_own["snapshot"] = (waiting, dora_count(hand + [t for m in melds for t in m.tiles], indicators))
                for opponent in opponents:
                    # At this moment no future river/result information has been read.
                    safe = tile // 4 in rivers[opponent]
                    suji = is_suji(tile // 4, rivers[opponent])
                    observe("response_non_genbutsu_rate", not safe)
                    observe("response_suji_nongenbutsu_rate", not safe and suji)
                    observe("response_nonsuji_suited_rate", not safe and not suji and tile // 4 < 27)
                    if opponent in first_pending:
                        observe("first_response_genbutsu_rate", safe)
                        first_pending.remove(opponent)
                    if not waiting:
                        period = "early" if draw_count <= 6 else "mid" if draw_count <= 12 else "late"
                        observe(f"{period}_noten_non_genbutsu_rate", not safe)
                    if rank is not None:
                        position = "top" if rank == 1 else "last" if rank == 4 else "middle"
                        observe(f"roundstart_{position}_non_genbutsu_rate", not safe)
            rivers[who].add(tile // 4)
            if who in declared and who not in active:
                declaration_discards.add(who)
            last_discard = (who, tile)
        elif tag == "AGARI":
            validate_agari(a)
            live = False
        elif tag == "RYUUKYOKU":
            if not a.get("type"):
                disclosed = a.get(f"hai{seat}", "")
                if disclosed:
                    shown = sorted(int(t) for t in disclosed.split(","))
                    _require(shown == sorted(hand), "RYUUKYOKU disclosed hand mismatch")
                    _require(bool(wait_shapes(hand, melds)), "disclosed exhaustive hand lacks structural wait")
                    audit["draw_hand_checks"] += 1
                observe("exhaustive_draw_disclosed_tenpai_rate", bool(disclosed))
            else:
                audit["special_draws_excluded"] += 1
            live = False
    _require(audit["rounds"] > 0 and not live, "no completed rounds or truncated game")
    result = dict(player=player, seat=seat, dan=ranks[seat], go_type=int(go.get("type", "0")),
                  feature_version=EXTRACTOR_VERSION, extraction_valid=True,
                  order_verified=False, full_history_confirmed=False,
                  quality_flags="within_hour_order_unknown;history_completeness_unverified;structural_waits_only;river_only_genbutsu;no_aka_ura")
    for name in FEATURE_NAMES:
        result[name] = nums[name] / dens[name] if dens[name] else math.nan
        result[name + "__numerator"] = nums[name]
        result[name + "__denominator"] = dens[name]
    result.update({"audit_" + key: value for key, value in sorted(audit.items())})
    result["audit_json"] = json.dumps(dict(sorted(audit.items())), ensure_ascii=False, sort_keys=True)
    return result
