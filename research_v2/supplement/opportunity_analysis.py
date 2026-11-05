"""C6 supplement: opportunity-weighted ten-game descriptions, no fitting."""
from pathlib import Path
from collections import Counter
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
from mahjong_research.analysis import validate_data, prepare_windows, disjoint_windows, summarize
from mahjong_research.inputs import new_run, write_csv, write_json
from mahjong_research.tenhou import FEATURE_NAMES, FEATURE_DEFINITIONS


def _summarize(x, y, view):
    # All-missing traditional features are expected and remain NA in the tables.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Mean of empty slice',category=RuntimeWarning,module=r'numpy\.lib\._nanfunctions_impl')
        return summarize(x,y,view)


def pool_windows(data, meta):
    """Sum additive event totals, retaining exactly the supplied source IDs."""
    values = data.set_index('game_id')
    if not values.index.is_unique:
        raise ValueError('対局ID重複')
    columns = [f+s for f in FEATURE_NAMES for s in ('__numerator','__denominator')]
    missing = sorted(set(columns)-set(data))
    if missing:
        raise ValueError('キャッシュの分子・分母列不足。推測補完しません: '+str(missing))
    for definition in FEATURE_DEFINITIONS:
        f = definition['feature']
        num, den = (pd.to_numeric(values[f+s], errors='raise') for s in ('__numerator','__denominator'))
        if not (np.isfinite(num).all() and np.isfinite(den).all() and num.ge(0).all() and den.ge(0).all()
                and num.mod(1).eq(0).all() and den.mod(1).eq(0).all()
                and num.le(den*definition['upper']).all() and num[den.eq(0)].eq(0).all()):
            raise ValueError('不正な分子・分母: '+f)
        expected = num.div(den.where(den.gt(0)))
        if not np.isclose(expected.to_numpy(), values[f].to_numpy(dtype=float), equal_nan=True).all():
            raise ValueError('保存済み率と分子・分母が不一致: '+f)
        values[f+'__numerator'], values[f+'__denominator'] = num, den
    pooled, totals = [], []
    known_ids = set(values.index)
    for row in meta.itertuples():
        ids = tuple(row.source_ids)
        if len(ids) != 10 or len(set(ids)) != 10 or not set(ids) <= known_ids:
            raise ValueError('10対局の窓構成が不正: '+row.game_id)
        source = values.loc[list(ids)]
        record = {}
        for f in FEATURE_NAMES:
            num = source[f+'__numerator'].sum()
            den = source[f+'__denominator'].sum()
            record[f] = num/den if den > 0 else np.nan
            totals.append(dict(game_id=row.game_id, target_label=row.target_label, feature=f,
                               numerator_sum=num, denominator_sum=den))
        pooled.append(record)
    return pd.DataFrame(pooled, columns=FEATURE_NAMES), pd.DataFrame(totals)


def same_population(expected, actual):
    a, b = expected.set_index('game_id'), actual.set_index('game_id')
    if not a.index.is_unique or not b.index.is_unique or set(a.index) != set(b.index):
        raise ValueError('従来C6と追加分析の窓終端が不一致')
    for gid in a.index:
        ids = a.loc[gid, 'source_ids']
        if isinstance(ids, str):
            ids = json.loads(ids)
        if set(ids) != set(b.loc[gid, 'source_ids']) or a.loc[gid,'target_label'] != b.loc[gid,'target_label']:
            raise ValueError('従来C6と元対局集合・ラベルが不一致: '+gid)


def _direction(a, b, adequate=True):
    if not adequate or not np.isfinite(a) or not np.isfinite(b):
        return '比較不能'
    if a == 0 or b == 0:
        return '平均差ゼロを含む'
    return '同方向' if np.sign(a) == np.sign(b) else '逆方向'


def run_opportunity_analysis(data, forbidden, reference_meta, output_root, initial_exclude=1500, make_plots=True):
    d, cutoff = validate_data(data, FEATURE_NAMES, initial_exclude)
    if str(d.player.iloc[0]) not in ('shobon','しょぼんｎ','しょぼんn'):
        raise ValueError('しょぼんｎだけを対象にしてください')
    baseline = prepare_windows(d, FEATURE_NAMES, cutoff, forbidden)
    same_population(reference_meta, baseline['meta'])
    pooled, totals = pool_windows(d, baseline['meta'])
    weighted = dict(baseline, X=pooled)
    old_disjoint, new_disjoint = disjoint_windows(baseline), disjoint_windows(weighted)
    same_population(old_disjoint['meta'], new_disjoint['meta'])
    views = {'traditional_overlap':baseline, 'opportunity_overlap':weighted,
             'traditional_disjoint':old_disjoint, 'opportunity_disjoint':new_disjoint}
    stats, differences = [], []
    for view, block in views.items():
        s, diff = _summarize(block['X'], block['y'], view)
        stats.append(s); differences.append(diff)
    # Common support separates changing weights from changing available windows.
    for suffix, a, b in [('overlap',baseline,weighted),('disjoint',old_disjoint,new_disjoint)]:
        for name, block in [('traditional',a),('opportunity',b)]:
            common = a['X'].notna() & b['X'].notna()
            s, diff = _summarize(block['X'].where(common), block['y'], name+'_common_'+suffix)
            stats.append(s); differences.append(diff)
    stats, differences = pd.concat(stats, ignore_index=True), pd.concat(differences, ignore_index=True)
    comparisons = []
    for f in FEATURE_NAMES:
        feature = differences[differences.feature.eq(f)].set_index('view')
        record = {'feature':f}
        for view in views:
            row = feature.loc[view]
            for col in ['mean_diff_high_minus_low','smd','low_valid_n','high_valid_n']:
                record[view+'_'+col] = row[col]
        for name, a, b in [('aggregation_overlap','traditional_overlap','opportunity_overlap'),
                           ('aggregation_disjoint','traditional_disjoint','opportunity_disjoint'),
                           ('overlap_vs_disjoint','opportunity_overlap','opportunity_disjoint')]:
            adequate = min(feature.loc[[a,b],['low_valid_n','high_valid_n']].to_numpy().flatten()) >= 2
            record[name+'_direction'] = _direction(feature.loc[a,'mean_diff_high_minus_low'], feature.loc[b,'mean_diff_high_minus_low'], adequate)
        for group in ('low','high'):
            record[group+'_valid_gain'] = record['opportunity_overlap_'+group+'_valid_n']-record['traditional_overlap_'+group+'_valid_n']
        record['stable_direction_candidate'] = all(record[k] == '同方向' for k in ['aggregation_overlap_direction','aggregation_disjoint_direction','overlap_vs_disjoint_direction'])
        for view in ('traditional_common_overlap','opportunity_common_overlap','traditional_common_disjoint','opportunity_common_disjoint'):
            for col in ('mean_diff_high_minus_low','smd','low_valid_n','high_valid_n'):
                record[view+'_'+col] = feature.loc[view,col]
        comparisons.append(record)
    comparisons = pd.DataFrame(comparisons)
    out = new_run(output_root)
    write_csv(out/'group_statistics.csv',stats)
    write_csv(out/'mean_differences_smd.csv',differences)
    write_csv(out/'aggregation_comparison.csv',comparisons)
    write_csv(out/'window_opportunity_totals.csv',totals)
    write_csv(out/'smd_ranking.csv', differences.sort_values(['view','abs_smd'], ascending=[True,False]))
    for view, block in views.items():
        table = block['meta'].copy()
        table['source_ids'] = table.source_ids.map(json.dumps)
        write_csv(out/(view+'_windows.csv'),pd.concat([table,block['X']],axis=1))
    reuse = Counter(i for ids in baseline['meta'].source_ids for i in ids)
    write_csv(out/'source_reuse.csv',pd.DataFrame(sorted(reuse.items()), columns=['game_id','windows_using_game']))
    defs = pd.DataFrame(FEATURE_DEFINITIONS)
    defs['window_aggregation'] = np.where(defs.upper.eq(1),'合計分子/合計機会数による割合','合計数量/合計観測イベント数による加重平均')
    write_csv(out/'aggregation_definitions.csv',defs)
    den_summary = totals.groupby(['feature','target_label']).denominator_sum.agg(['count','min','median','mean','max']).reset_index()
    write_csv(out/'opportunity_denominator_summary.csv',den_summary)
    facts = dict(primary_windows=len(pooled), disjoint_windows=len(new_disjoint['X']),
                 unique_source_games=len(reuse),max_source_reuse=max(reuse.values(),default=0),
                 features_improved_either_group=int((comparisons.low_valid_gain.gt(0)|comparisons.high_valid_gain.gt(0)).sum()),
                 features_improved_both_groups=int((comparisons.low_valid_gain.gt(0)&comparisons.high_valid_gain.gt(0)).sum()),
                 stable_direction_candidates=comparisons.loc[comparisons.stable_direction_candidate,'feature'].tolist(),
                 order_verified=False,full_history_confirmed=False,imputation='none',
                 data_sha256=hashlib.sha256(d.to_csv(index=False).encode()).hexdigest())
    write_json(out/'analysis_settings.json',facts)
    labels = dict(zip(defs.feature,defs.japanese))
    lines = ['# C6追加検証：機会数に基づく10対局集計','',
             f"主分析{facts['primary_windows']}窓、元対局非重複{facts['disjoint_windows']}窓。最大再利用{facts['max_source_reuse']}回。",
             f"有効窓数が少なくとも一方の群で増えた指標は{facts['features_improved_either_group']}/20、両群で増えた指標は{facts['features_improved_both_groups']}/20。",'',
             '## 各特徴量の確認結果']
    for r in comparisons.itertuples():
        lines.append(f'- {labels[r.feature]}：有効窓数 低{r.traditional_overlap_low_valid_n:g}→{r.opportunity_overlap_low_valid_n:g}、高{r.traditional_overlap_high_valid_n:g}→{r.opportunity_overlap_high_valid_n:g}。高−低の平均差 {r.traditional_overlap_mean_diff_high_minus_low:.4g}→{r.opportunity_overlap_mean_diff_high_minus_low:.4g}。集計方式間は{r.aggregation_overlap_direction}、追加方式の重複/非重複間は{r.overlap_vs_disjoint_direction}。')
    lines += ['', '## 解釈と推奨',
              '18項目は機会あたりの割合、表ドラ平均と待ち種類数平均の2項目はイベントあたりの数量の加重平均。いずれも分子・分母が保存済みなので再解析せず計算する。',
              '機会がない対局の0/0は行動率0とは扱わない。窓内の機会数合計が0なら欠損。10対局窓全体のイベントに関する記述を主目的にする場合、機会数による集計を主分析の候補とし、対局率の単純平均は別の推定対象として感度分析に残す。',
              '主な比較表では方式により有効窓の集合が変わる。共通有効窓のcommon分析を併記し、重みの変更と対象窓の変更を分けて確認する。',
              '窓内ではイベント数を重みにするが、群の平均値は有効な10対局窓を等重みで平均する。期間全体の全イベントを一括合算した率とは推定対象が異なる。',
              '同方向という表示は差の符号の一致だけを示す。分母が少ない窓・有効窓数・SMDの大きさも確認し、安定した実力差や統計的有意差と解釈しない。',
              '窓間でも元対局が重複する。検定・独立標本に基づく信頼区間は算出していない。非重複窓も同一人物の時系列であり独立性は保証されない。',
              '窓は終端段位で分類するので段位17や異なる段位の履歴を含む。時間経過、対戦相手、卓条件、発生機会の差が交絡し得る。',
              'order_verified=False、full_history_confirmed=False。全プレイヤーへの一般化と因果解釈はできない。分類モデルの入力やパラメータは本追加分析で変更していない。']
    (out/'report_ja.md').write_text('\n'.join(lines),encoding='utf-8')
    if make_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for page,start in enumerate(range(0,len(FEATURE_NAMES),6),1):
            fig,axes=plt.subplots(3,2,figsize=(12,11))
            for ax,f in zip(axes.flat,FEATURE_NAMES[start:start+6]):
                for view,style in [('traditional_overlap','--'),('opportunity_overlap','-')]:
                    block=views[view]
                    for label,color in [(0,'tab:blue'),(1,'tab:orange')]:
                        x=np.sort(block['X'].loc[block['y'].eq(label),f].dropna())
                        if len(x):ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=color,linestyle=style,label=f'{view.split("_")[0]} class{label} n={len(x)}')
                ax.set_title(f,fontsize=8); ax.set_ylabel('ECDF')
                if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=6)
            for ax in list(axes.flat)[len(FEATURE_NAMES[start:start+6]):]:ax.set_visible(False)
            fig.tight_layout();fig.savefig(out/f'distribution_{page:02d}.png',dpi=150);plt.close(fig)
    return dict(output_dir=out,facts=facts,statistics=stats,differences=differences,comparisons=comparisons)
