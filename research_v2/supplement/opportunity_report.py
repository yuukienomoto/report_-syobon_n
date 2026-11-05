"""C10 report from saved C6B/C7/C9 CSVs only; no extraction or fitting."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from mahjong_research.inputs import new_run,write_csv,write_json,sha256
from mahjong_research.reporting import performance_report
from mahjong_research.tenhou import FEATURE_NAMES,FEATURE_DEFINITIONS

VIEWS=('opportunity_overlap','traditional_overlap','opportunity_disjoint','traditional_disjoint')


def _sign_comparison(a,b):
    if not np.isfinite(a) or not np.isfinite(b):return '比較不能'
    if a*b<0:return '逆方向'
    if a==0 or b==0:return '平均差ゼロを含む'
    return '同方向'


def read_descriptions(folder):
    folder=Path(folder)
    stats=pd.read_csv(folder/'group_statistics.csv',float_precision='round_trip')
    diffs=pd.read_csv(folder/'mean_differences_smd.csv',float_precision='round_trip')
    settings=json.loads((folder/'analysis_settings.json').read_text(encoding='utf-8'))
    if settings.get('order_verified') is not False or settings.get('full_history_confirmed') is not False:
        raise ValueError('C6Bの確認フラグが想定と違います')
    if stats.duplicated(['view','feature','target_label']).any() or diffs.duplicated(['view','feature']).any():
        raise ValueError('C6B集計CSVのキー重複')
    for view in VIEWS:
        if set(diffs.loc[diffs.view.eq(view),'feature'])!=set(FEATURE_NAMES):
            raise ValueError('C6Bの20特徴量が不足: '+view)
        for f in FEATURE_NAMES:
            groups=stats[stats.view.eq(view)&stats.feature.eq(f)].set_index('target_label')
            if set(groups.index)!={0,1}:raise ValueError('C6Bの群別統計が不足: '+f)
            for r in groups.itertuples():
                if not (np.isfinite(r.n) and np.isfinite(r.valid_n) and 0<=r.valid_n<=r.n and r.n%1==r.valid_n%1==0):
                    raise ValueError('C6Bの件数が不正')
                rate=1-r.valid_n/r.n if r.n else np.nan
                if not np.isclose(rate,r.missing_rate,equal_nan=True):raise ValueError('C6Bの欠損率が不整合')
                if r.valid_n==0 and (pd.notna(r.mean) or pd.notna(r.median)):
                    raise ValueError('有効値0件に平均・中央値があります')
                if r.valid_n>0 and not (np.isfinite(r.mean) and np.isfinite(r.median)):
                    raise ValueError('有効値がある群の平均・中央値が不正')
            low,high=groups.loc[0],groups.loc[1]
            diff=diffs[diffs.view.eq(view)&diffs.feature.eq(f)].iloc[0]
            pool=np.nan
            if min(low.valid_n,high.valid_n)>=2:
                pool=np.sqrt(((low.valid_n-1)*low['std']**2+(high.valid_n-1)*high['std']**2)/(low.valid_n+high.valid_n-2))
            mean_diff=high['mean']-low['mean']
            smd=mean_diff/pool if np.isfinite(pool) and pool>0 else np.nan
            if not all([np.isclose(mean_diff,diff.mean_diff_high_minus_low,equal_nan=True),np.isclose(smd,diff.smd,equal_nan=True),
                        low.valid_n==diff.low_valid_n,high.valid_n==diff.high_valid_n]):
                raise ValueError('C6B群別統計と差・SMDが不一致: '+view+'/'+f)
    return stats,diffs,settings


def _saved_models(run):
    bundles={}
    for version in ('legacy','v2'):
        folder=run/('model_'+version)
        if version=='legacy' and not (folder/'external_freeze.json').exists():
            bundles[version]=None
            continue
        frozen=json.loads((folder/'external_freeze.json').read_text(encoding='utf-8'))
        for file,digest in frozen['files'].items():
            if sha256(folder/file)!=digest:raise ValueError('固定モデル記録のハッシュ不一致: '+file)
        bundles[version]=dict(metrics=pd.read_csv(folder/'internal_metrics.csv',float_precision='round_trip'))
    attempt=json.loads((run/'model_v2/external_attempt.json').read_text(encoding='utf-8'))
    if attempt['status']!='completed':raise ValueError('C9の完了記録がありません')
    metrics=run/'external_v2/external_metrics.csv'
    if Path(attempt['output_dir']).resolve()!=metrics.parent.resolve() or sha256(metrics)!=attempt['output_sha256']['external_metrics.csv']:
        raise ValueError('C9の保存先・ハッシュが不一致')
    return bundles,pd.read_csv(metrics,float_precision='round_trip')


def updated_thesis_report(opportunity_dir,run_dir):
    source,run=Path(opportunity_dir),Path(run_dir)
    stats,diffs,settings=read_descriptions(source)
    # Bind saved descriptive and model results to the same scoring population.
    w=pd.read_csv(source/'opportunity_overlap_windows.csv')
    train=pd.read_csv(run/'model_v2/final_train_windows.csv')
    if w.game_id.duplicated().any() or train.game_id.duplicated().any():raise ValueError('窓終端重複')
    if set(w.game_id)!=set(train.game_id):raise ValueError('C6BとC7の最終学習窓集合が違います')
    if len(w)!=settings['primary_windows']:raise ValueError('C6B主分析の窓数と設定が不一致')
    for label in (0,1):
        counts=stats.loc[stats.view.eq('opportunity_overlap')&stats.target_label.eq(label),'n']
        if not counts.eq(w.target_label.eq(label).sum()).all():raise ValueError('C6B主分析の群の窓数が不一致')
    a,b=w.set_index('game_id'),train.set_index('game_id')
    for gid in a.index:
        if set(json.loads(a.loc[gid,'source_ids']))!=set(json.loads(b.loc[gid,'source_ids'])) or a.loc[gid,'target_label']!=b.loc[gid,'target_label']:
            raise ValueError('C6BとC7の元対局・ラベルが違います')
    bundles,external=_saved_models(run)
    definitions={d['feature']:d for d in FEATURE_DEFINITIONS}
    rows=[]
    for f in FEATURE_NAMES:
        row=dict(feature=f,japanese=definitions[f]['japanese'])
        for view in VIEWS:
            v=diffs[diffs.view.eq(view)&diffs.feature.eq(f)].iloc[0]
            for c in ('mean_diff_high_minus_low','smd','low_valid_n','high_valid_n'):row[view+'_'+c]=v[c]
        main=row['opportunity_overlap_mean_diff_high_minus_low'];old=row['traditional_overlap_mean_diff_high_minus_low'];sub=row['opportunity_disjoint_mean_diff_high_minus_low']
        row['aggregation_direction']=_sign_comparison(main,old)
        row['overlap_disjoint_direction']=_sign_comparison(main,sub)
        row['disjoint_minus_overlap_mean_diff']=sub-main
        row['disjoint_minus_overlap_smd']=row['opportunity_disjoint_smd']-row['opportunity_overlap_smd']
        row['mean_diff_magnitude_ratio_disjoint_to_overlap']=abs(sub/main) if np.isfinite(main) and main!=0 and np.isfinite(sub) else np.nan
        row['minimum_valid_n_opportunity']=min(row[v+'_'+c] for v in ('opportunity_overlap','opportunity_disjoint') for c in ('low_valid_n','high_valid_n'))
        rows.append(row)
    comparison=pd.DataFrame(rows)
    out=new_run(run/'final_report_opportunity')
    write_csv(out/'primary_and_sensitivity_results.csv',comparison)
    write_csv(out/'descriptive_group_statistics.csv',stats)
    write_csv(out/'descriptive_mean_differences_smd.csv',diffs)
    perf=performance_report(out,bundles['legacy'],bundles['v2'],external)
    perf.loc[perf.status.eq('computed_current_run'),'status']='saved_C7_C9_reused'
    write_csv(out/'model_performance_comparison.csv',perf)
    metric_sources=[run/'model_v2/internal_metrics.csv',run/'external_v2/external_metrics.csv']
    if bundles['legacy'] is not None:metric_sources.append(run/'model_legacy/internal_metrics.csv')
    for path in metric_sources:
        dest=out/'classification_sources'/path.parent.name/path.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    index=[]
    for label,folder in [('C6B',source),('C7_legacy',run/'model_legacy'),('C7_v2',run/'model_v2'),('C9',run/'external_v2')]:
        for image in sorted(folder.glob('*.png')):
            target=out/'figures'/label/image.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(image,target)
            index.append(dict(stage=label,source=str(image),copied_figure=target.relative_to(out).as_posix()))
    write_csv(out/'figure_index.csv',pd.DataFrame(index))
    flips=comparison.loc[comparison.aggregation_direction.eq('逆方向'),'feature'].tolist()
    fmt=lambda value: f'{value:+.6f}' if np.isfinite(value) else '算出不可'
    lines=['# 卒論結果・考察 更新版：機会数方式を主分析候補とする','',
      '## 研究方法と集計の意味',
      'しょぼんｎの段位16以下と18以上の時期を比較する。段位17は履歴に残し、終端の採点から除く。初期1500観測対局を含む時間帯までを採点外とし、hour_invariant窓を使用する。',
      '主分析候補はC6Bのopportunity_overlap。各10対局窓内で分子と分母をそれぞれ合算して割り、その後、有効な窓を等重みで比較する。全イベントの単純合算とは推定対象が異なる。分母0は欠損とし、0補完しない。',
      '18項目は機会あたりの割合。表ドラ枚数平均と待ち種類数平均の2項目は、合計数量÷合計イベント数による加重平均。従来の対局率10個の単純平均をtraditional_overlap感度分析として残す。',
      'SMDは群内分散の自由度加重プール標準偏差で標準化した記述量。有効窓数・欠損率が方式ごとに異なるため、共通有効窓のCSVも併せて確認する。',
      f"主分析{settings['primary_windows']}窓、非重複{settings['disjoint_windows']}窓、元対局最大再利用{settings['max_source_reuse']}回。",'',
      '## 主分析候補：機会数方式の20特徴量']
    for r in comparison.itertuples():
        lines.append(f'- {r.japanese}（{r.feature}）：高−低={fmt(r.opportunity_overlap_mean_diff_high_minus_low)}、SMD={fmt(r.opportunity_overlap_smd)}、有効窓 低{r.opportunity_overlap_low_valid_n:g}/高{r.opportunity_overlap_high_valid_n:g}。')
    lines+=['','## 感度分析：従来方式・非重複窓',
      '方式間で差の符号が逆転した特徴量は '+('、'.join(flips) if flips else '該当なし（比較不能・差ゼロの項目は別扱い）')+'。',
      '以下の数値は方式を併記する。同方向は符号の一致だけを示す。大きさは平均差・SMDの変化を確認し、有効窓が少ない場合や片群欠損の場合は安定性を判断しない。']
    for r in comparison.itertuples():
        lines.append(f'- {r.japanese}：従来単純平均 高−低={fmt(r.traditional_overlap_mean_diff_high_minus_low)}、SMD={fmt(r.traditional_overlap_smd)}、有効窓 低{r.traditional_overlap_low_valid_n:g}/高{r.traditional_overlap_high_valid_n:g}。集計方式間={r.aggregation_direction}。機会数・非重複 高−低={fmt(r.opportunity_disjoint_mean_diff_high_minus_low)}、SMD={fmt(r.opportunity_disjoint_smd)}、有効窓 低{r.opportunity_disjoint_low_valid_n:g}/高{r.opportunity_disjoint_high_valid_n:g}、重複窓との方向={r.overlap_disjoint_direction}。非重複−重複の平均差変化={fmt(r.disjoint_minus_overlap_mean_diff)}、SMD変化={fmt(r.disjoint_minus_overlap_smd)}。')
    lines+=['','## 補助分析：保存済みC7・C9の分類結果',
      'C7・C9の既存CSVを読み込んで掲載した。モデル入力は従来の訓練区間のみで補完した対局特徴量の10対局平均。機会数方式を適用した対象は記述分析である。分類性能を新しい集計方式の評価値へ読み替えない。']
    for r in perf.loc[perf.status.eq('saved_C7_C9_reused')].itertuples():
        lines.append(f'- {r.version}/{r.dataset}/{r.estimator}：AUC={fmt(r.roc_auc)}、Balanced Accuracy={fmt(r.balanced_accuracy)}、Accuracy={fmt(r.accuracy)}、F1強={fmt(r.f1_strong)}。')
    lines+=['','## 考察と限界',
      '機会数方式は、機会のない対局を含む窓の行動情報を利用できる。方式間で符号や大きさが変わる場合、機会数による重みの変更と有効窓の構成変化が影響した可能性を分けて検討する。',
      '非重複窓でも同じ人物の時系列相関は残る。重複窓を独立標本として検定・信頼区間を算出していない。',
      'order_verified=False、full_history_confirmed=False。hour_invariantは観測履歴内の順序不変性に限られる。段位と時間経過、卓条件、対戦相手、行動機会が交絡する。窓終端と履歴の段位が混在する場合もある。',
      '同一人物の時期別比較として解釈し、実力向上の原因や一般の強者・弱者全体の差を断定しない。外部評価結果を用いたモデル・特徴量・閾値の調整は行わない。']
    path=out/'thesis_results_ja.md';path.write_text('\n'.join(lines),encoding='utf-8')
    files=[source/'group_statistics.csv',source/'mean_differences_smd.csv',source/'analysis_settings.json',source/'opportunity_overlap_windows.csv',run/'model_v2/final_train_windows.csv']+metric_sources
    write_json(out/'report_sources.json',dict(primary='opportunity_overlap',traditional='sensitivity',model_results='saved_C7_C9_reused',
       sources={str(p):sha256(p) for p in files},order_verified=False,full_history_confirmed=False))
    return dict(output_dir=out,report_path=path,comparison=comparison,performance=perf,sign_reversals=flips)
