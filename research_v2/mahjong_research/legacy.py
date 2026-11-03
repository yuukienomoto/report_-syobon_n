"""Unchanged legacy feature list and cross formulas; source commit c6e1499.
No legacy extraction code or notebook is modified.
"""
import re
class ValidationBlocked(ValueError): pass

BASE_FEATURES = [
    'first_discard_raw_unsafe_rate', 'first_discard_safe_rate',
    'early_noten_push_rate', 'mid_noten_push_rate', 'late_noten_push_rate',
    'las_desperate_push_rate', 'top_conservative_push_rate', 'flat_situation_push_rate',
    'riichi_dora_mean', 'chasing_riichi_good_shape_rate', 'chasing_riichi_low_value_rate',
    'chasing_riichi_rate', 'early_meld_ratio', 'late_meld_ratio', 'meld_dora_ratio',
    'raw_unsafe_discard_rate', 'ryuukyoku_tenpai_rate', 'ryuukyoku_noten_safe_rate',
]

CROSS_FEATURES = ['top_late_noten_push_rate', 'las_early_meld_rate',
                  'early_meld_with_dora_rate', 'chasing_riichi_quality_diff',
                  'flat_first_discard_safe_rate']

FEATURES = BASE_FEATURES + CROSS_FEATURES

def canonical_game_id(value):
    # パス・視点tw・拡張子を除去。同じ対局の別視点も同じIDにする。
    name = str(value).replace('\\', '/').rsplit('/', 1)[-1].lower()
    match = re.fullmatch(r'(\d{10}gm-[0-9a-f]+-[0-9a-f]+-[0-9a-f]+)(?:&tw=[0-3])?(?:\.mjlog)?', name)
    if not match:
        raise ValidationBlocked(f'解釈できない対局ID: {value}')
    return match.group(1)

def add_cross(base):
    # 元ノートブックの5式をそのまま維持。局面別再集計はしない。
    x = base[BASE_FEATURES].copy()
    x['top_late_noten_push_rate'] = x['top_conservative_push_rate'] * x['late_noten_push_rate']
    x['las_early_meld_rate'] = x['las_desperate_push_rate'] * x['early_meld_ratio']
    x['early_meld_with_dora_rate'] = x['early_meld_ratio'] * x['meld_dora_ratio']
    x['chasing_riichi_quality_diff'] = x['chasing_riichi_good_shape_rate'] - x['chasing_riichi_low_value_rate']
    x['flat_first_discard_safe_rate'] = x['first_discard_safe_rate'] * (1.0 - x['flat_situation_push_rate'])
    return x[FEATURES]
