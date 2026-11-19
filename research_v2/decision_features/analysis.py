"""Exposure/measurement audit only: no group contrasts, AUC or model training."""
import json
import numpy as np
import pandas as pd
from mahjong_research.inputs import write_csv
from .definitions import NAMES, DEFINITIONS


def window_audit(ctx, records):
    """Use original full-ledger blocks, never concatenate sparse pilot games."""
    if not records.game_id.is_unique or set(records.game_id) != set(ctx['targets'].game_id):
        raise ValueError('試験台帳と結果ID集合が不一致')
    failed = records.status.ne('ok').any()
    indexed = records.set_index('game_id')
    rows, members, coverage = [], [], []
    for width in (10, 20):
        complete = 0
        used = set()
        for start in ctx['starts'][width]:
            block = ctx['data'].iloc[start:start+width]
            ids = block.game_id.tolist()
            if len(ids) != width or used.intersection(ids) or block.target_label.nunique() != 1:
                raise ValueError('既存窓のサイズ・重複・段位ラベル不一致')
            data = ctx['data']
            if ((start > 0 and data.hour.iloc[start-1] == block.hour.iloc[0]) or
                    (start+width < len(data) and data.hour.iloc[start+width] == block.hour.iloc[-1])):
                raise ValueError('既存窓が同一時間帯を分断しています')
            used.update(ids)
            if failed or not set(ids) <= set(indexed.index):
                continue
            part = indexed.loc[ids]
            if not np.array_equal(part.dan.to_numpy(), block.dan.to_numpy()):
                raise ValueError('窓結合時に段位が不一致')
            complete += 1
            window_id = f'{width}:{ids[0]}:{ids[-1]}'
            members.append(dict(width=width, window_id=window_id, label=int(block.target_label.iloc[0]),
                                source_ids=json.dumps(ids), start_hour=str(block.hour.iloc[0]),
                                end_hour=str(block.hour.iloc[-1])))
            for f in NAMES:
                num, den = (int(part[f+s].sum()) for s in ('__numerator', '__denominator'))
                rows.append(dict(width=width, window_id=window_id, label=int(block.target_label.iloc[0]),
                    feature=f, numerator=num, denominator=den, value=num/den if den else np.nan))
        coverage.append(dict(width=width, existing_windows=len(ctx['starts'][width]), complete_pilot_windows=complete,
            reason='failed_game_blocks_window_analysis' if failed else
                   ('no_complete_existing_windows_in_sparse_pilot' if not complete else 'existing_membership_only')))
    cols = ['width','window_id','label','feature','numerator','denominator','value']
    windows = pd.DataFrame(rows, columns=cols)
    stats = []
    for width in (10, 20):
        for label in (0, 1):
            for f in NAMES:
                part = windows[(windows.width == width)&(windows.label == label)&(windows.feature == f)]
                den = part.denominator.astype(float)
                stats.append(dict(width=width, label=label, feature=f, n=len(part), denominator_median=den.median(),
                    denominator_q25=den.quantile(.25), denominator_q75=den.quantile(.75),
                    denominator_iqr=den.quantile(.75)-den.quantile(.25),
                    zero_denominator_rate=den.eq(0).mean() if len(den) else np.nan,
                    missing_rate=part.value.isna().mean() if len(part) else np.nan))
    return windows, pd.DataFrame(members, columns=['width','window_id','label','source_ids','start_hour','end_hour']), pd.DataFrame(coverage), pd.DataFrame(stats)


def save_analysis(ctx, frame, events, audits, out):
    ok = frame[frame.status.eq('ok')]
    summaries = []
    for group in ('all', 'low', 'high'):
        subset = frame if group == 'all' else frame[frame.dan.le(16) if group == 'low' else frame.dan.ge(18)]
        good = subset[subset.status.eq('ok')]
        for f in NAMES:
            den, num = good[f+'__denominator'], good[f+'__numerator']
            summaries.append(dict(feature=f, group=group, target_games=len(subset), successful_games=len(good),
                failed_games=len(subset)-len(good), numerator_total=num.sum() if len(good) else np.nan,
                denominator_total=den.sum() if len(good) else np.nan,
                zero_denominator_games=int(den.eq(0).sum()),
                missing_rate_successful=den.eq(0).mean() if len(good) else np.nan,
                denominator_median=den.median(), denominator_q25=den.quantile(.25), denominator_q75=den.quantile(.75)))
    summary = pd.DataFrame(summaries)
    write_csv(out/'feature_opportunities.csv', summary)
    flattened, exclusions, round_counts = [], [], []
    for ev in events:
        for f, count in ev['features'].items():
            row = dict(game_id=ev['game_id'], round=ev['round'], round_seed=ev['round_seed'], event=ev['event'],
                       seat=ev['seat'], feature=f, **count)
            flattened.append(row)
            if not count['denominator']:
                for reason in count['reason'].split('|'):
                    exclusions.append(dict(feature=f, reason=reason))
            if count['denominator']:
                round_counts.append(row)
    flat = pd.DataFrame(flattened, columns=['game_id','round','round_seed','event','seat','feature','numerator','denominator','reason'])
    if flat.duplicated(['game_id','round','event','feature']).any():
        raise ValueError('同一打牌・特徴量の重複計数')
    for f in NAMES:
        for col in ('numerator', 'denominator'):
            if flat.loc[flat.feature.eq(f), col].sum() != ok[f+'__'+col].sum():
                raise ValueError('イベントと対局別計数の不一致')
    write_csv(out/'decision_events.csv', flat)
    excluded = pd.DataFrame(exclusions, columns=['feature','reason']).groupby(['feature','reason']).size().reset_index(name='count')
    write_csv(out/'exclusions.csv', excluded)
    rounds = flat.groupby(['game_id','round','feature'])[['numerator','denominator']].sum().reset_index()
    totals = rounds.groupby('feature').denominator.transform('sum')
    rounds['share_of_all_opportunities'] = rounds.denominator.div(totals.where(totals.gt(0)))
    game_totals = rounds.groupby(['game_id','feature']).denominator.transform('sum')
    rounds['share_of_game_opportunities'] = rounds.denominator.div(game_totals.where(game_totals.gt(0)))
    write_csv(out/'round_concentration.csv', rounds)
    for name, table in zip(('window_values','window_memberships','window_coverage','window_opportunities'), window_audit(ctx, frame)):
        write_csv(out/(name+'.csv'), table)
    lines = ['# 4判断特徴量・少数試験の監査', '',
        f'対象{len(frame)}対局、成功{len(ok)}、失敗{len(frame)-len(ok)}。旧20特徴量との計数照合は成功対局で実施済み。',
        '成功対局キャッシュは再利用する。再利用時の抽出秒数は初回値、current_secondsは今回の読込時間。',
        '低／高段位の差・有意差・モデル性能を計算した結果ではない。', '',
        '| 特徴量 | 分子/分母（成功分） | 分母0対局/成功対局 |', '|---|---:|---:|']
    for definition in DEFINITIONS:
        f = definition['feature']
        r = summary[(summary.feature == f)&(summary.group == 'all')].iloc[0]
        lines.append(f'| {definition["id"]} {definition["japanese"]} | {r.numerator_total}/{r.denominator_total} | {r.zero_denominator_games}/{r.successful_games} |')
    lines += ['', '## 状態・機会の確認',
        'decision_traces.jsonの手牌・河・候補・実打牌をdecision_events.csvの局/イベント番号と対応させ、正例と除外例を人手で照合する。',
        'exclusions.csvは複数除外理由を併記するため、理由別件数の和は打牌数と一致しない。',
        f'成功分で副露/カンイベント数={int(ok.get("calls", pd.Series(dtype=float)).sum())}/{int(ok.get("kans", pd.Series(dtype=float)).sum())}、'
        f'複数他家成立リーチ中の自己打牌={int(ok.get("multi_riichi_discards", pd.Series(dtype=float)).sum())}。',
        '10/20対局は元台帳の既存非重複窓を全て抽出できた場合だけ集計。飛び飛びの試験対局を詰めて窓にしない。',
        'window_coverage.csvの完成窓数0は未測定を意味し、分母0窓率や欠損率0を意味しない。失敗が1件でもあれば窓集計を停止。',
        '', '## 解釈上の制約',
        'order_verified=False、full_history_confirmed=Falseを維持。同一時間帯内の厳密順序と全履歴は未確認。',
        '16以下/18以上・17除外、最初の1500観測対局と同じ時間帯まで除外、共通対局除外を維持。17を含む2条件は後続の感度分析。',
        'A1は実打牌のテンパイ維持を要件としない。A4は非現物ノーテンを選ぶ第三の可能性もある。',
        'E4は共通現物と非共通現物の両方がある時だけ1打牌を1機会と数える。',
        '形状待ちは役・フリテン・公開残枚数を考慮しない。スジの安全性や非現物の善悪、意思は推定しない。',
        '少数試験で未出現の機会、段位期・局面の不足は未検証。対象者全履歴での品質や一般化を保証しない。',
        '同一局内の反復には依存がある。round_concentration.csvで機会が特定局に偏っていないか確認する。',
        '失敗の分子・分母はNaNで、成功した分母0とは区別する。失敗を除いた全体の率を完成値として報告しない。']
    (out/'report_ja.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
