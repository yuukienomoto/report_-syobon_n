"""Frozen subset of design commit 0078520eafc29cdec706e98316f61cee8a188ed9."""
VERSION = 'decision-a1-a4-e3-e4-v1'
DESIGN_COMMIT = '0078520eafc29cdec706e98316f61cee8a188ed9'
RULE = ('本人の通常ツモ後の実打牌。自己成立リーチ前、鳴き直後を除外し、'
        '前回自己打牌（局頭はINIT）以降に誰のカンもないこと。自己宣言打牌は含む。'
        'Gは対象者自身の過去の河、Tは既存wait_shapesが非空となる打牌候補。'
        'A1/A4/E3は他家成立リーチ1人、E4は2人以上。1実打牌1機会。')
DEFINITIONS = [
    dict(id='A1', feature='tenpai_safe_alternative_non_genbutsu_rate',
         japanese='テンパイ選択肢あり・現物保有時の非現物選択率',
         denominator='F1かつT非空、G非空、D-G非空', numerator='実打牌がGに属さない',
         limitation='実打牌でのテンパイ維持は要求しない。非現物は悪手を意味しない。'),
    dict(id='A4', feature='genbutsu_tenpai_sacrifice_rate',
         japanese='現物選択に伴うテンパイ放棄率',
         denominator='F1かつG非空、T非空、GとTの共通部分が空', numerator='実打牌がGに属する',
         limitation='非現物ノーテンの第三の選択は分子0。形状のみで役・フリテン・残枚数を判定しない。'),
    dict(id='E3', feature='suji_fallback_choice_rate',
         japanese='現物なし・数牌のみの完全スジ選択率',
         denominator='F1かつG空、Dが数牌のみ、完全スジと非スジがともに存在', numerator='実打牌が完全スジ',
         limitation='4/5/6は両側スジ必須。片スジは非スジ。安全確率や安全性の保証ではない。'),
    dict(id='E4', feature='all_riichi_genbutsu_choice_rate',
         japanese='複数リーチ全員への現物選択率',
         denominator='F2plusかつ全員の河に共通するG_allとD-G_allがともに非空',
         numerator='実打牌がG_allに属する', limitation='相手人数を掛けない。共通現物しかない局面は対象外。'),
]
NAMES = tuple(d['feature'] for d in DEFINITIONS)
