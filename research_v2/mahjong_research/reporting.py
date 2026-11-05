"""Thesis-facing comparison and interpretation of measured outputs only."""
from pathlib import Path
import numpy as np
import pandas as pd
from .inputs import write_csv, write_json

HISTORICAL = [
    dict(version='legacy_random_reference', dataset='random_5fold', roc_auc=.8340, status='historical_reference_different_split_possible_overlap'),
    dict(version='legacy_external_reference', dataset='external_old_preprocessing', accuracy=.3559, status='historical_reference_different_population'),
    dict(version='legacy_H5_reference', dataset='fold1', roc_auc=np.nan, balanced_accuracy=np.nan, status='user_reported_unavailable'),
    dict(version='legacy_H5_reference', dataset='fold2', roc_auc=.580615, balanced_accuracy=.506831, status='user_reported_not_rerun'),
    dict(version='legacy_H5_reference', dataset='fold3', roc_auc=.454563, balanced_accuracy=.452368, status='user_reported_not_rerun'),
    dict(version='legacy_H5_reference', dataset='external', roc_auc=.374715, balanced_accuracy=.404974, n=713, weak=97, strong=616,status='user_reported_not_rerun'),
]


def compare_descriptive(old, new, mapping_path, out):
    """Same observed windows; changed definitions are NOT on a common scale."""
    a_meta=old['primary']['meta'].set_index('game_id')
    b_meta=new['primary']['meta'].set_index('game_id')
    if set(a_meta.index)!=set(b_meta.index):
        raise ValueError('旧版・新版の主分析終端集合が違います')
    for gid in a_meta.index:
        if set(a_meta.loc[gid,'source_ids'])!=set(b_meta.loc[gid,'source_ids']):
            raise ValueError('旧版・新版の元対局窓が違います')
    mapping=pd.read_csv(mapping_path)
    a=old['differences'].rename(columns={'feature':'legacy_feature'})
    b=new['differences'].rename(columns={'feature':'new_feature'})
    views=pd.DataFrame({'view':sorted(set(a.view)|set(b.view))})
    joined=mapping.merge(views,how='cross').merge(a,on=['legacy_feature','view'],how='left').merge(b,on=['new_feature','view'],how='left',suffixes=('_legacy','_v2'))
    write_csv(Path(out)/'legacy_v2_descriptive_comparison.csv',joined)
    write_json(Path(out)/'comparison_population.json',dict(same_endpoints=True,same_source_sets=True,n=len(a_meta),
        missing_policy='両版とも無補完。10値すべて観測された特徴量のみ窓平均を計算',
        warning='測定対象・分母が変化した指標の差は精度向上や同一尺度上の増減を意味しない'))
    return joined


def opportunity_report(data, definitions, cutoff, forbidden, out):
    eligible=data[(data.hour>cutoff)&~data.game_id.isin(forbidden)&data.dan.ne(17)].copy()
    eligible['label']=np.where(eligible.dan>=18,1,0)
    rows=[]
    for label, group in eligible.groupby('label'):
        for definition in definitions:
            f=definition['feature']; numerator=group[f+'__numerator']; denominator=group[f+'__denominator']
            rows.append(dict(feature=f,label=int(label),games=len(group),games_with_opportunity=int(denominator.gt(0).sum()),
                             numerator_sum=numerator.sum(),denominator_sum=denominator.sum(),
                             opportunity_weighted_rate=numerator.sum()/denominator.sum() if denominator.sum()>0 else np.nan,
                             mean_game_denominator=denominator.mean(),median_game_denominator=denominator.median(),
                             interpretation='機会重み付き参考値。対局ごとの率の単純平均とは別の推定対象'))
    result=pd.DataFrame(rows);write_csv(Path(out)/'opportunities_by_rank.csv',result)
    return result


def performance_report(out, legacy_bundle=None, new_bundle=None, external=None):
    frames=[pd.DataFrame(HISTORICAL)]
    for version, bundle in [('legacy23_same_protocol',legacy_bundle),('v2_fixed_same_protocol',new_bundle)]:
        if bundle is not None and len(bundle['metrics']):
            frame=bundle['metrics'].copy();frame['version']=version;frame['status']='computed_current_run';frames.append(frame)
    if external is not None:
        frame=external.copy();frame['version']='v2_fixed_external';frame['status']='computed_current_run';frames.append(frame)
    table=pd.concat(frames,ignore_index=True);write_csv(Path(out)/'model_performance_comparison.csv',table)
    return table


def thesis_report(out, descriptive, definitions, performance, comparison_available):
    out=Path(out)
    labels={d['feature']:d['japanese'] for d in definitions}
    limitations={d['feature']:d['limitation'] for d in definitions}
    main=descriptive['differences'].query("view == 'primary_hour_invariant'").sort_values('abs_smd',ascending=False)
    lines=['# 改善版特徴量による同一人物の段位別比較','',
       '## 研究方法','しょぼんｎの観測対局を段位16以下と18以上に分けた。段位17は履歴に残し、採点から除いた。',
       '初期1500観測対局を含む時間帯までを採点対象外とした。完全な生涯履歴の先頭1500対局とは断定しない。',
       '10対局窓は時間帯を途中で切らず、終端の時間帯に対局が1件の窓に限定した。時間帯内順序を確定したわけではない。','',
       '## データセット',f"観測{descriptive['summary']['observed_games']}対局。主分析{descriptive['summary']['primary_windows']}窓、元対局非重複の補助分析{descriptive['summary']['disjoint_windows']}窓。",'',
       '## 特徴量の定義','副露を反映した手牌復元、構造上のテンパイ、和了牌候補、宣言後の公開情報に基づく指標を用いた。',
       '実装上の範囲・分母はfeature_definitions_v2.csvを参照。非現物は実際の放銃危険度や押す意思そのものではない。','',
       '## 分析方法','主分析は補完しない10対局平均、補助分析は対局単位と元対局が重複しない窓で比較した。10値に欠損があれば、その特徴量の窓平均は欠損とした。',
       '平均差は高段位期−低段位期。SMDは群内分散の自由度加重プール標準偏差で割った値。独立標本を仮定した有意差検定は行っていない。',
       'モデル入力のみ訓練区間の平均値で補完し、木数200・学習率0.03・深さ5等を固定した。試合終了後の段位区分分類であり、対局前の勝敗予測ではない。','',
       '## 確認できた結果']
    for row in main.itertuples():
        f=row.feature
        if pd.isna(row.mean_diff_high_minus_low):
            lines.append(f'- {labels.get(f,f)}：有効値不足により比較不能。')
        else:
            direction='高い' if row.mean_diff_high_minus_low>0 else '低い' if row.mean_diff_high_minus_low<0 else '等しい'
            smd=f'{row.smd:.3f}' if pd.notna(row.smd) else '算出不能'
            lines.append(f'- {labels.get(f,f)}：高段位期で平均が{direction}（差={row.mean_diff_high_minus_low:.4g}、SMD={smd}）。有効窓は低{row.low_valid_n}、高{row.high_valid_n}。{limitations[f]}')
    computed=performance[performance.status.eq('computed_current_run')]
    lines+=['','## 予測性能','従来値と今回値はmodel_performance_comparison.csvで区別した。ランダム分割AUC0.834と時間帯分割の差を、特徴量改善だけの効果とは解釈できない。']
    if computed.empty:lines.append('今回のモデル学習・外部評価はまだ実行されていない。')
    else:
        for r in computed.itertuples():
            lines.append(f'- {r.version} / {r.dataset} / {getattr(r,"estimator",getattr(r,"method","model"))}: AUC={getattr(r,"roc_auc",np.nan):.4f}, BA={getattr(r,"balanced_accuracy",np.nan):.4f}（NaNは算出条件を満たさない）。')
    lines+=['','## 考察・検証すべき仮説','測定を修正しても同じ方向の差が残るなら、測定誤差だけでは説明できない関連の候補になる。ただし、段位変化と時間経過、卓条件、対戦相手の強さが重なる。',
       '欠損率と機会数の群間差、対局単位・非重複窓の方向一致を先に確認する。月別平均だけから学習効果や変化点を断定しない。',
       '旧版と新版は同じ対局・窓で比較したが、定義が変わった指標は同一尺度ではない。' if comparison_available else '旧版との対応比較は未実施。','',
       '## 研究の限界','- 対局順序と履歴完全性は未確認。hour_invariantは観測集合内での順序不変性であり、厳密な完全時系列検証と同等ではない。',
       '- 同一人物の段位別比較であり、プレイヤー全体の強者と弱者の差には一般化できない。',
       '- 重複窓、欠損による選択、段位ラベルの遅れ・対戦環境の交絡がある。段位は潜在的実力そのものではない。',
       '- 旧外部結果は既知のため、改善版も研究全体から完全に隔離された盲検ではない。今回の定義選定は測定の妥当性に基づき、外部スコアで調整しない。',
       '- 赤ドラ・裏ドラ・役や打点・実際の放銃可能性を全て評価する実装ではない。',
       '- 低い識別性能から重要度やSHAPを実力差の証拠に読み替えない。本版は重要度・SHAPによる選択をしない。','',
       '## 今後の課題','卓条件と期間が重なる範囲での比較、機会数を加味した頑健性確認、イベント別の実牌譜手検証を優先する。別プレイヤーによる再現は固定した計画で行う。']
    (out/'thesis_results_ja.md').write_text('\n'.join(lines),encoding='utf-8')
    return out/'thesis_results_ja.md'
