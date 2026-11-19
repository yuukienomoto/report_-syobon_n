"""Tables and Japanese interpretation for a post-hoc label sensitivity study."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mahjong_research.inputs import write_csv

FOCUS = ('response_non_genbutsu_rate', 'response_nonsuji_suited_rate',
         'chasing_riichi_share', 'open_meld_round_rate')


def fmt(x):
    return f'{x:.4f}' if pd.notna(x) else '算出不可'


def deliver(out, metrics, groups, differences, external_note):
    baseline = metrics[metrics.condition.eq('baseline')]
    delta = metrics[~metrics.condition.eq('baseline')].merge(
        baseline, on=['dataset', 'width', 'fold', 'scope'], suffixes=('', '_baseline'), validate='many_to_one')
    for name in ('roc_auc', 'balanced_accuracy', 'accuracy'):
        delta[name+'_change'] = delta[name] - delta[name+'_baseline']
    write_csv(out/'performance_changes.csv', delta)
    numeric_signs = differences.assign(opposite=differences.opposite.map({True: 1., False: 0.}),
        substantial_reversal=differences.substantial_reversal.map({True: 1., False: 0.}))
    signs = numeric_signs.groupby(['condition', 'dataset', 'width', 'fold']).agg(
        opposite_features=('opposite', 'sum'), comparable_features=('opposite', 'count'),
        substantial_reversals=('substantial_reversal', 'sum')).reset_index()
    write_csv(out/'relationship_reversals.csv', signs)
    lines = ['# 17段を含めた分類基準の感度分析',
             '## 1. 方法と対象数',
             'baselineは16以下／18以上、include_17_lowは17以下／18以上、include_17_highは16以下／17以上。'
             'これらは牌譜の段位コードに基づく区分である。最初の1500観測対局と同じ時間帯までを除外した。'
             '分類条件ごとに同一クラス内の連続10・20対局を機会数合算で集計した。段位の数値が違っても同一クラスなら同じ窓に含める。'
             'クラス境界、共通対局、初期除外を飛び越えて詰め直す処理は行っていない。',
             '窓の両端は時間帯境界に置き、同じ時間帯を分割しない。3条件は既存baselineの共通時系列境界を使う。'
             '境界をまたぐ窓を除外する。common_windowsは3条件で元対局集合と窓構成が一致する評価窓であり、学習窓は条件ごとに異なる。',
             'モデルAの20特徴量・固定パラメータを使用した。欠損はLightGBMのNaN処理、閾値は0.5。'
             'class_weight=balancedの規則を維持し、クラス重みの数値は各学習集合のクラス数に応じて決まる。'
             'baselineは保存モデルを再利用し、既存の窓・予測確率・評価指標との一致を検査した。']
    internal = groups[groups.dataset.eq('internal')]
    for width in (10, 20):
        for fold in (1, 2, 3):
            for r in internal.query('width==@width and fold==@fold').itertuples():
                base = internal.query('width==@width and fold==@fold and condition=="baseline" and role==@r.role').iloc[0]
                lines.append(f'- {width}対局 Fold {fold} {r.condition} {r.role} 低{r.low}・高{r.high}窓、'
                             f'17段{r.rank17_games}対局。baseline比 {r.windows-base.windows:+d}窓。')
    lines += ['class_distribution.csvに対象期間・段位内訳、cached_rank_counts.csvに窓作成前の全対局数を保存した。'
              '17段の追加で窓の起点が変わり、17段以外の対局の採否も変わり得る。',
              '## 2. 時系列評価と改善の一貫性']
    full = metrics.query('dataset=="internal" and scope=="all"')
    for (width, fold), part in full.groupby(['width', 'fold']):
        for r in part.itertuples():
            lines.append(f'- {width}対局 Fold {fold} {r.condition} AUC {fmt(r.roc_auc)}、'
                         f'Balanced Accuracy {fmt(r.balanced_accuracy)}、Accuracy {fmt(r.accuracy)}。{r.status}。'
                         + (str(r.auc_reason) if pd.isna(r.roc_auc) else ''))
        eligible = part.dropna(subset=['roc_auc'])
        if len(eligible):
            best = eligible[eligible.roc_auc.eq(eligible.roc_auc.max())]
            lines.append('この区間でAUCが最も高かった条件は' + '、'.join(best.condition) + '。')
    for scope in ('all', 'common_windows'):
        for condition in ('include_17_low', 'include_17_high'):
            v = delta.query('dataset=="internal" and scope==@scope and condition==@condition')
            finite = v.dropna(subset=['roc_auc_change'])
            lines.append(f'{condition}・{scope}のAUC差が正だったのは、比較可能{len(finite)}区間中'
                         f'{int(finite.roc_auc_change.gt(0).sum())}区間。'
                         f'差の範囲 {fmt(finite.roc_auc_change.min())}〜{fmt(finite.roc_auc_change.max())}。')
    lines += ['## 3. 外部評価', external_note]
    for r in metrics.query('dataset=="external" and scope=="all"').itertuples():
        lines.append(f'- {r.width}対局 {r.condition} AUC {fmt(r.roc_auc)}。{r.status}。')
    for r in delta.query('dataset=="external" and scope=="all"').itertuples():
        lines.append(f'- {r.width}対局 {r.condition} baselineからのAUC差 {fmt(r.roc_auc_change)}。')
    for r in groups[groups.dataset.eq('external')].itertuples():
        lines.append(f'- {r.width}対局 {r.condition} 対象期間{r.first}〜{r.last}、低{r.low}・高{r.high}窓、段位別対局数 {r.rank_counts}。')
    lines += ['## 4. 群平均差の逆転と重点特徴量',
              'Stage 5と同じ高−低の群平均差、群別自由度で重み付けしたプールSDによるSMDを使用する。'
              '分母ゼロの対局・窓は欠損とし、各特徴量の有効窓数と欠損率を記録する。'
              '逆符号数は学習と評価の差がともに計算できる項目を分母とする。差が小さい項目の逆符号も含むため、SMDと変化量も併記する。'
              'substantial_reversalsはStage 5の探索的な目安を引き継ぎ、逆符号かつ群間差の変化が学習時プールSDの0.5倍以上、'
              '単一特徴量AUCの変化の絶対値が0.10以上の項目数である。統計的有意性を示す判定ではない。']
    base_signs = signs[signs.condition.eq('baseline')].set_index(['dataset', 'width', 'fold'])
    for r in signs.itertuples():
        b = base_signs.loc[(r.dataset, r.width, r.fold)]
        lines.append(f'- {r.dataset} {r.width}対局 Fold {r.fold} {r.condition} 逆符号'
                     f'{int(r.opposite_features)}/{r.comparable_features}項目、baselineとの差'
                     f'{int(r.opposite_features-b.opposite_features):+d}項目。')
    for r in differences.query('dataset=="internal" and width==10 and fold==3').itertuples():
        if r.feature in FOCUS:
            lines.append(f'- {r.condition} {r.feature} 学習→評価の高−低差 {fmt(r.train_diff)}→{fmt(r.eval_diff)}、'
                         f'SMD {fmt(r.train_smd)}→{fmt(r.eval_smd)}。')
    lines += ['## 5. 解釈できる範囲と限界',
              '17段を含めた場合のラベル、採用対局、窓の構成、学習時クラス重みの変化を合わせた感度分析である。'
              '共通評価窓の結果は評価対象の違いを抑えるが、学習対象の変化を含む。AUCの高低だけでは予測対象が異なるモデルの優劣を確定できない。',
              '結果を見た後の探索的な比較であり、最も高いAUCを得た基準を最適と断定できない。'
              '特徴量と段位期の関係変化の原因や、強さの因果要因を同定したものではない。'
              '単一人物の段位と時間、卓条件が交絡し、他プレイヤーへの一般化にも制約がある。',
              'order_verified=False、full_history_confirmed=Falseを維持する。同一時間帯の順序と全履歴の完全性は未確認。'
              '時間的依存性が残り、Fold数も少ないため、独立標本を仮定した有意差検定は行わない。']
    (out/'report_ja.md').write_text('\n\n'.join(lines), encoding='utf8')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, width in zip(axes, (10, 20)):
        for condition, part in full[full.width.eq(width)].groupby('condition'):
            ax.plot(part.fold, part.roc_auc, 'o-', label=condition)
        ax.axhline(.5, color='grey', lw=1)
        ax.set(title=f'{width} games | shared time boundaries', xlabel='Fold', ylabel='ROC-AUC', xticks=[1, 2, 3], ylim=(0, 1))
        ax.legend(fontsize=8)
    fig.savefig(out/'auc_comparison.png', dpi=160)
    plt.close(fig)
