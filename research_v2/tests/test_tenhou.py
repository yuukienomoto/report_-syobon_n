"""Synthetic boundary tests; no private logs are embedded in the repository."""
import math
import unittest
from mahjong_research.tenhou import (
    ExtractionError, PlayerNotFound, FEATURE_NAMES, decode_meld, dora_count,
    extract_game, is_suji, wait_shapes,
)


def physical(types):
    used, result = {}, []
    for t in types:
        copy = used.get(t, 0)
        result.append(4 * t + copy)
        used[t] = copy + 1
    return result


def fixture(hand, events, scores="300,250,230,220"):
    return (f'<mjloggm><GO type="9"/><UN n0="tester" n1="a" n2="b" n3="c" dan="16,18,18,18"/>'
            f'<INIT hai0="{",".join(map(str, hand))}" ten="{scores}" seed="0,0,0,1,2,132"/>'
            f'{events}</mjloggm>').encode()


class MeldTests(unittest.TestCase):
    def test_decode_chi_physical_copies(self):
        # 2m/3m/4m, called 3m, copies 1/2/3, relative source3.
        code = ((1 * 3 + 1) << 10) | (1 << 3) | (2 << 5) | (3 << 7) | 4 | 3
        meld = decode_meld(code, 0)
        self.assertEqual((meld.kind, meld.tiles, meld.called, meld.from_seat), ("chi", (5, 10, 15), 10, 3))

    def test_pon_kakan_kan_types(self):
        pon = decode_meld(((5 * 3 + 1) << 9) | (2 << 5) | 8 | 1, 0)
        added = decode_meld(((5 * 3 + 1) << 9) | (2 << 5) | 16 | 1, 0)
        self.assertEqual(pon.tiles, (20, 21, 23))
        self.assertEqual((added.kind, added.added), ("kakan", 22))
        self.assertEqual(decode_meld((31 * 4 + 2) << 8, 2).kind, "ankan")
        self.assertEqual(decode_meld(((31 * 4 + 2) << 8) | 1, 2).called, 126)
        with self.assertRaises(ExtractionError):
            decode_meld((120 << 8) | 32, 0)


class WaitTests(unittest.TestCase):
    def test_ryanmen(self):
        hand = physical([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 5])
        waits = wait_shapes(hand)
        self.assertEqual(set(waits), {3, 6})
        self.assertTrue(all("ryanmen" in x for x in waits.values()))

    def test_kanchan_penchan_tanki_shanpon(self):
        core = [0, 1, 2, 9, 10, 11, 27, 27]
        self.assertIn("kanchan", wait_shapes(physical(core + [18, 19, 20, 3, 5]))[4])
        self.assertIn("penchan", wait_shapes(physical(core + [18, 19, 20, 7, 8]))[6])
        self.assertIn("tanki", wait_shapes(physical([0, 1, 2, 9, 10, 11, 18, 19, 20, 3, 4, 5, 31]))[31])
        self.assertIn("shanpon", wait_shapes(physical([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 31, 31]))[31])

    def test_special_hands(self):
        self.assertIn("chiitoitsu", wait_shapes(physical([0, 0, 3, 3, 9, 9, 15, 15, 20, 20, 27, 27, 32]))[32])
        orphan_types = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]
        waits = wait_shapes(physical(orphan_types))
        self.assertEqual(len(waits), 13)
        self.assertTrue(all("kokushi" in x for x in waits.values()))

    def test_open_meld_and_own_four_copy_constraint(self):
        pon = decode_meld(((31 * 3) << 9) | (3 << 5) | 8 | 1, 0)
        hand = physical([0, 1, 2, 9, 10, 11, 18, 19, 27, 27])
        self.assertEqual(set(wait_shapes(hand, [pon])), {20})
        fourth_copy = physical([0, 1, 2, 9, 10, 11, 18, 19, 20]) + [127]
        self.assertFalse(wait_shapes(fourth_copy, [pon]))
        with self.assertRaises(ExtractionError):
            wait_shapes(hand)

    def test_noten(self):
        self.assertFalse(wait_shapes(physical([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])))


class PublicInformationTests(unittest.TestCase):
    def test_suji_middle_requires_both_sides(self):
        self.assertTrue(is_suji(0, {3}))
        self.assertFalse(is_suji(3, {0}))
        self.assertTrue(is_suji(3, {0, 6}))
        self.assertFalse(is_suji(27, {30}))

    def test_dora_wrap_and_multiplicity(self):
        self.assertEqual(dora_count([0, 1], [32, 33]), 4)
        self.assertEqual(dora_count([108, 124], [120, 132]), 2)

    def test_future_opponent_discard_does_not_change_past_safety(self):
        hand = physical([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 5])
        prefix = '<REACH who="1" step="1"/><E100/><REACH who="1" step="2"/><T135/><D135/>'
        a = extract_game(fixture(hand, prefix + '<RYUUKYOKU type="yao9"/>'), "tester")
        b = extract_game(fixture(hand, prefix + '<E134/><RYUUKYOKU type="yao9"/>'), "tester")
        self.assertEqual(a["response_non_genbutsu_rate"], 1)
        self.assertEqual(a["first_response_genbutsu_rate"], 0)
        self.assertEqual(a["response_non_genbutsu_rate"], b["response_non_genbutsu_rate"])
        # Actual post-discard hand is tenpai, despite being closed and unreached.
        self.assertTrue(math.isnan(a["early_noten_non_genbutsu_rate"]))

    def test_no_response_no_denominator(self):
        hand = physical([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 5])
        data = extract_game(fixture(hand, '<REACH who="1" step="1"/><E100/><REACH who="1" step="2"/><RYUUKYOKU type="yao9"/>'), "tester")
        self.assertEqual(data["first_response_genbutsu_rate__denominator"], 0)
        self.assertTrue(math.isnan(data["first_response_genbutsu_rate"]))
        self.assertEqual(data["open_meld_round_rate"], 0)

    def test_riichi_snapshot_after_discard_and_acceptance(self):
        hand = physical([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 5])
        events = '<REACH who="1" step="1"/><E100/><REACH who="1" step="2"/><T135/><REACH who="0" step="1"/><D135/><REACH who="0" step="2"/><RYUUKYOKU type="yao9"/>'
        result = extract_game(fixture(hand, events), "tester")
        self.assertEqual(result["chasing_wait_types_mean"], 2)
        self.assertEqual(result["chasing_ryanmen_compatible_rate"], 1)
        self.assertEqual(result["riichi_omote_dora_mean"], 0)

    def test_bad_hand_stops_not_imputes(self):
        hand = physical([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 5])
        with self.assertRaises(ExtractionError):
            extract_game(fixture(hand, '<D135/><RYUUKYOKU type="yao9"/>'), "tester")
        with self.assertRaises(PlayerNotFound):
            extract_game(fixture(hand, '<RYUUKYOKU type="yao9"/>'), "different")

    def test_ankan_preserves_menzen_and_does_not_count_open_meld(self):
        hand = physical([0, 0, 0, 0, 9, 10, 11, 18, 19, 20, 27, 27, 31])
        events = '<T125/><N who="0" m="0"/><T132/><D132/><RYUUKYOKU type="yao9"/>'
        result = extract_game(fixture(hand, events), "tester")
        self.assertEqual(result["audit_own_meld_ankan"], 1)
        self.assertEqual(result["open_meld_round_rate"], 0)

    def test_daiminkan_reconstructs_and_counts_open_meld(self):
        hand = physical([0, 0, 0, 9, 10, 11, 18, 19, 20, 27, 27, 31, 31])
        events = '<E3/><N who="0" m="769"/><T132/><D132/><RYUUKYOKU type="yao9"/>'
        result = extract_game(fixture(hand, events), "tester")
        self.assertEqual(result["audit_own_meld_daiminkan"], 1)
        self.assertEqual(result["open_meld_round_rate"], 1)

    def test_pon_kakan_replay_does_not_add_an_extra_set(self):
        hand = physical([0, 0, 9, 10, 11, 18, 19, 20, 27, 27, 31, 31, 33])
        pon_code = (2 << 9) | (3 << 5) | 8 | 1
        kakan_code = (2 << 9) | (3 << 5) | 16 | 1
        events = (f'<E2/><N who="0" m="{pon_code}"/><D132/>'
                  f'<T3/><N who="0" m="{kakan_code}"/><T133/><D133/><RYUUKYOKU type="yao9"/>')
        result = extract_game(fixture(hand, events), "tester")
        self.assertEqual(result["audit_own_meld_pon"], 1)
        self.assertEqual(result["audit_own_meld_kakan"], 1)
        self.assertEqual(result["open_meld_round_rate__numerator"], 1)

    def test_tied_roundstart_rank_excluded(self):
        hand = physical([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 5])
        events = '<REACH who="1" step="1"/><E100/><REACH who="1" step="2"/><T135/><D135/><RYUUKYOKU type="yao9"/>'
        result = extract_game(fixture(hand, events, scores="250,250,250,250"), "tester")
        self.assertTrue(math.isnan(result["roundstart_top_non_genbutsu_rate"]))

    def test_disclosed_hand_mismatch_and_truncation_rejected(self):
        hand = physical([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 5])
        with self.assertRaises(ExtractionError):
            extract_game(fixture(hand, '<RYUUKYOKU hai0="0"/>'), "tester")
        with self.assertRaises(ExtractionError):
            extract_game(fixture(hand, '<T135/>'), "tester")

    def test_unaccepted_riichi_is_not_counted(self):
        hand = physical([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 5])
        events = '<T135/><REACH who="0" step="1"/><D135/><AGARI who="1" fromWho="0"/>'
        result = extract_game(fixture(hand, events), "tester")
        self.assertTrue(math.isnan(result["chasing_riichi_share"]))

    def test_multiple_reachers_are_counted_separately(self):
        hand = physical([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 5])
        events = ('<REACH who="1" step="1"/><E133/><REACH who="1" step="2"/>'
                  '<REACH who="2" step="1"/><F100/><REACH who="2" step="2"/>'
                  '<T135/><D135/><RYUUKYOKU type="yao9"/>')
        result = extract_game(fixture(hand, events), "tester")
        self.assertEqual(result["first_response_genbutsu_rate__denominator"], 2)
        self.assertEqual(result["first_response_genbutsu_rate"], .5)
        self.assertEqual(result["response_non_genbutsu_rate"], .5)

    def test_exhaustive_vs_special_draw_denominators(self):
        hand = physical([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 5])
        shown = ",".join(map(str, hand))
        normal = extract_game(fixture(hand, f'<RYUUKYOKU hai0="{shown}"/>'), "tester")
        special = extract_game(fixture(hand, f'<RYUUKYOKU type="nm" hai0="{shown}"/>'), "tester")
        self.assertEqual(normal["exhaustive_draw_disclosed_tenpai_rate"], 1)
        self.assertEqual(normal["audit_draw_hand_checks"], 1)
        self.assertTrue(math.isnan(special["exhaustive_draw_disclosed_tenpai_rate"]))

    def test_future_dora_does_not_revalue_declaration(self):
        hand = physical([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 5])
        prefix = '<T135/><REACH who="0" step="1"/><D135/><REACH who="0" step="2"/>'
        result = extract_game(fixture(hand, prefix + '<DORA hai="104"/><RYUUKYOKU type="yao9"/>'), "tester")
        self.assertEqual(result["riichi_omote_dora_mean"], 0)

    def test_chi_replay_removes_consumed_tiles(self):
        hand = [5, 15] + physical([9, 10, 11, 18, 19, 20, 27, 27, 31, 31, 33])
        code = ((1 * 3 + 1) << 10) | (1 << 3) | (2 << 5) | (3 << 7) | 4 | 3
        result = extract_game(fixture(hand, f'<G10/><N who="0" m="{code}"/><D132/><RYUUKYOKU type="yao9"/>'), "tester")
        self.assertEqual(result["audit_own_meld_chi"], 1)
        self.assertEqual(result["open_meld_round_rate"], 1)

    def test_later_winner_in_double_ron_is_also_checked(self):
        hand = physical([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 5])
        shown = ",".join(map(str, hand + [12]))
        events = f'<E12/><AGARI who="2" fromWho="1"/><AGARI who="0" fromWho="1" machi="12" hai="{shown}"/>'
        result = extract_game(fixture(hand, events), "tester")
        self.assertEqual(result["audit_agari_hand_checks"], 1)

    def test_acceptance_requires_a_declaration_discard(self):
        hand = physical([0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 4, 5])
        with self.assertRaises(ExtractionError):
            extract_game(fixture(hand, '<REACH who="1" step="1"/><REACH who="1" step="2"/><RYUUKYOKU type="yao9"/>'), "tester")


if __name__ == "__main__":
    unittest.main()
