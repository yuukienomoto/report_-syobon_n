"""Exploratory evidence table, Japanese interpretation and figures."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mahjong_research.inputs import write_csv
from study import FEATURES, DEFS, WIDTHS


def candidates(comp, arrays):
    table = comp[comp.scope.eq('available')].copy()
    raw = arrays['available', 1][0]
    spans = {}
    for col, f in enumerate(FEATURES):
        valid = raw[:, col][np.isfinite(raw[:, col])]
        spans[f] = np.ptp(valid) if len(valid) and np.ptp(valid) > 0 else np.nan
    # Compare units on a stated empirical scale. Export native-scale SD and diff as well.
    scale = table.feature.map(spans)
    table['within_sd_scaled'] = table[['low_sd','high_sd']].max(axis=1, skipna=False)/scale
    table['abs_diff_scaled'] = table['diff'].abs()/scale
    table['auc_distance'] = (table.auc-.5).abs()
    table['abs_smd'] = table.smd.abs()
    evidence = table[table[['low_valid_n','high_valid_n']].min(axis=1).ge(2)]
    med = evidence[['abs_smd','auc_distance','overlap','within_sd_scaled']].median()
    for key, value in med.items():
        table['reference_'+key] = value
    common = comp[comp.scope.eq('common_games')].set_index(['feature','width'])
    records = []
    for r in table.itertuples():
        sensitivity = common.loc[r.feature, r.width]
        if (min(r.low_valid_n, r.high_valid_n) < 2 or max(r.low_missing_rate, r.high_missing_rate) > .5
                or min(r.low_opp_median, r.high_opp_median) <= 1):
            category, why = 'D', '有効単位が2未満、過半欠損、または少なくとも一群の機会数中央値が1以下'
        elif r.pooled_sd == 0 and r.diff != 0:
            category, why = 'A', '両群内で定数かつ群平均が異なる。標本数と期間依存を要確認'
        elif (r.abs_smd > med.abs_smd and r.auc_distance > med.auc_distance and
              r.overlap < med.overlap and r.overlap < .8):
            if r.within_sd_scaled <= med.within_sd_scaled:
                category, why = 'A', '差・順位分離・低い重なりが揃い、尺度換算した群内SDも相対的に小さい'
            else:
                category, why = 'B', '差・順位分離・低い重なりが揃うが、尺度換算した群内SDは相対的に大きい'
        else:
            category, why = 'C', '相対的な差が小さい、またはSMD・AUC・分布の分離が揃わない'
        records.append(dict(category=category, reason=why, japanese=DEFS[r.feature]['japanese'],
                            tactical_measure=DEFS[r.feature]['formula'], caveat=DEFS[r.feature]['limitation'],
                            common_diff=sensitivity['diff'], common_smd=sensitivity.smd,
                            common_auc=sensitivity.auc, common_overlap=sensitivity.overlap))
    return pd.concat([table.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def figures(out, table, arrays):
    directory = out/'figures'
    directory.mkdir()
    colors = ['#12629a', '#ba5320']
    for page, offset in enumerate(range(0, len(FEATURES), 4), 1):
        fig, axes = plt.subplots(4, 5, figsize=(18, 12), constrained_layout=True)
        for j, feature in enumerate(FEATURES[offset:offset+4], offset):
            box, sd, smd, miss, opp = axes[j-offset]
            rows = table[table.feature.eq(feature)].set_index('width').loc[list(WIDTHS)]
            for i, w in enumerate(WIDTHS):
                values, _, y = arrays['available', w]
                for label, color in enumerate(colors):
                    valid = values[(y == label) & np.isfinite(values[:, j]), j]
                    if len(valid):
                        box.boxplot(valid, positions=[i+(label-.5)*.34], widths=.28, patch_artist=True,
                                    boxprops=dict(facecolor=color, alpha=.6), medianprops=dict(color='black'),
                                    showfliers=False, manage_ticks=False)
            for label, color in zip(('low','high'), colors):
                for ax, key in [(sd,'sd'), (miss,'missing_rate'), (opp,'opp_median')]:
                    ax.plot(range(3), rows[label+'_'+key], 'o-', color=color)
            smd.plot(range(3), rows.smd, 'o-', color='#3d5950')
            smd.axhline(0, lw=.6, color='grey')
            miss.set_ylim(-.05, 1.05)
            box.set_ylabel(f'{j+1:02}  '+feature.replace('_','\n',1), fontsize=8)
            for ax, title in zip(axes[j-offset], ['Distribution (box, IQR)','Within-group SD',
                                      'SMD (high - low)','Missing fraction','Opportunities (median)']):
                ax.set_xticks(range(3), ['1','10','20'])
                if j == offset:
                    ax.set_title(title)
        legend = [Line2D([],[],color=c,label=l,linewidth=3) for c,l in zip(colors, ['Low rank','High rank'])]
        fig.suptitle('Non-overlapping units | games per unit on x-axis | blue: low, orange: high')
        axes[0,1].legend(handles=legend, fontsize=8)
        fig.savefig(directory/f'distributions_{page}.png', dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for panel, ax in enumerate(axes.flat):
        for j in range(5*panel, 5*panel+5):
            rows = table[table.feature.eq(FEATURES[j])].set_index('width').loc[list(WIDTHS)]
            line, = ax.plot(rows.within_sd_scaled**2, rows.abs_diff_scaled, lw=1,
                            label=f'{j+1:02} '+FEATURES[j])
            for marker, (_, r) in zip(['o','s','^'], rows.iterrows()):
                ax.scatter(r.within_sd_scaled**2, r.abs_diff_scaled, marker=marker, color=line.get_color())
        ax.set_xlabel('Larger within-group variance / single-game range squared')
        ax.set_ylabel('Absolute mean difference / single-game range')
        ax.legend(fontsize=8)
    fig.suptitle('Within-group variance and between-group difference | circle: 1, square: 10, triangle: 20')
    fig.savefig(directory/'variance_vs_difference.png', dpi=150)
    plt.close(fig)


def fmt(v):
    return f'{v:+.3f}' if np.isfinite(v) else '算出不可'


def deliver(out, comp, arrays):
    table = candidates(comp, arrays)
    write_csv(out/'feature_candidates.csv', table)
    figures(out, table, arrays)
    lines = ['# しょぼんｎの段位期別分布と平滑化',
             '段位16以下と18以上を比較。窓内で分子・機会数を合算し、各群では有効な単位を等重みとした。'
             '18項目は割合、表ドラ枚数・待ち種類数はイベント当たり平均。分母0は欠損。機会数の要約には0を含む。',
             '1対局では時刻が同じ対局も個別に記述した。10・20対局では時間帯の中を切らず、観測順に最も早く完成する'
             '同一段位群の非重複窓を採用。除外対局・混合段位群をまたがず、同群なら末尾時刻に複数対局があっても合算できる。'
             '初期1500観測対局を含む時間帯まで、段位17、共通対局を窓の元対局から除外した。この窓集合はC6Bと異なる。',
             '## 各条件で差が目立つ指標と群内で安定する指標',
             '差は高−低。異なる尺度の比較にはSMDを用いる。群内のばらつきは両群のSDの大きい方を、その特徴量の'
             '1対局での観測レンジで割った値で比較する。外れ値や少数機会の影響があるため、各群の元尺度SD・IQRも併読する。']
    counts = []
    for w in WIDTHS:
        t = table[table.width.eq(w)]
        eligible = t[t.category.ne('D')]
        top = eligible.dropna(subset=['abs_smd']).nlargest(4,'abs_smd')
        stable = eligible.dropna(subset=['within_sd_scaled']).nsmallest(4,'within_sd_scaled')
        both = t[t.category.eq('A')]
        counts.append(len(both))
        low, high = t.iloc[0][['low_n','high_n']]
        lines += [f'### {w}対局（低 {int(low)}、高 {int(high)} 単位）',
                  '平均差が目立つ項目（|SMD|上位、判断困難Dを除く）：'+
                  ('、'.join(f'{r.japanese}（差{fmt(r.diff)}、SMD{fmt(r.smd)}）' for r in top.itertuples()) or '算出可能項目なし'),
                  '群内の相対SDが小さい項目：'+('、'.join(f'{r.japanese}（{r.within_sd_scaled:.3f}）' for r in stable.itertuples()) or '比較不能'),
                  '両方を満たす探索候補A：'+('、'.join(both.japanese) or '該当なし')]
        for r in t[t.category.isin(['A','B'])].itertuples():
            lines.append(f'- {r.category}・{r.japanese}：{r.tactical_measure}。高−低 {fmt(r.diff)}。{r.caveat}。')
    lines += ['## 平滑化と共通対局での感度分析',
              f'同じ判断基準によるA候補は1→10→20対局で {counts[0]}→{counts[1]}→{counts[2]} 項目。'
              '候補件数はこの標本の記述であり、将来の対局での予測性能は別途検証が必要。',
              '共通対象は主解析で採用した20対局のうち10対局目で時間帯を分割できる集合。これを二分して10対局窓を作り、'
              '同じ元対局を1対局条件でも使用した。元対局が同じでも、集計で無機会の対局を含められるため、有効単位の構成は異なる。']
    for scope, title in [('available','各幅で採用可能な対局'),('common_games','共通対局')]:
        for w in (10,20):
            subset = comp[comp.scope.eq(scope) & comp.width.eq(w)]
            ok = subset[['low_sd_change_from_1','high_sd_change_from_1']].dropna()
            baseline = comp[comp.scope.eq(scope) & comp.width.eq(1)].set_index('feature')
            delta = subset.auc.sub(.5).abs() - subset.feature.map((baseline.auc-.5).abs())
            ov = subset.overlap.dropna()
            lines.append(f'- {title}・{w}対局（低{int(subset.iloc[0].low_n)}・高{int(subset.iloc[0].high_n)}単位）：'
                         f'両群のSDが1対局より減少 {int(ok.lt(0).all(axis=1).sum())}/{len(ok)} 項目、'
                         f'AUCの0.5からの距離が増加 {int(delta.gt(0).sum())}/{delta.count()} 項目。'
                         f'両群のヒストグラムの重なりが0.8以上 {int(ov.ge(.8).sum())}/{len(ov)} 項目。分母0は比較不能。')
    lines.append('### 候補の方向は共通対局でも一致するか')
    for w in WIDTHS:
        ab = table[table.width.eq(w) & table.category.isin(['A','B'])]
        finite = ab.dropna(subset=['diff','common_diff'])
        matched = finite[np.sign(finite['diff']) == np.sign(finite.common_diff)]
        lines.append(f'- {w}対局：A/B候補{len(ab)}項目中、共通対局と比較可能{len(finite)}項目、'
                     f'平均差の符号一致{len(matched)}項目。候補表で共通対象のSMD・AUC・重なりも確認できる。')
    lines += ['## 判断基準と限界',
              '候補表には3条件すべてについて判断理由、元尺度の差、標本数、欠損率、機会数中央値、共通対象での結果を添えた。'
              'Dは少なくとも一群で有効数2未満・過半欠損・機会数中央値1以下のいずれか。残りは全条件共通の探索用基準で比較した。'
              '両群で2単位以上ある特徴量×窓幅の中央値より|SMD|・|AUC−0.5|が大きく、重なりが中央値と0.8の両方を下回る場合をA/Bとした。'
              '相対SDが基準中央値以下ならA、他はB。両群が異なる定数ならA。その他Cは差が小さいか、分離の根拠が揃わない。'
              'これらの境界には探索上の判断が入る。連続指標と感度分析を優先し、機械的な採否には用いない。',
              '分布の重なりは各特徴量の全条件共通の等幅20区間で両群の相対度数の小さい方を足した値（1は同じ度数分布）。'
              '区間幅に依存する。AUCは高値ほど高段位とした同一標本内の順位比較で、逆方向は1−AUC。同順位は半分として数える。'
              'SMDは群内の自由度で重み付けしたプールSDを使用し、分散ゼロ・標本不足では算出しない。',
              '20特徴量で分類が難しくなり得る理由は、同じ段位期でも配牌や相手の行動により応答機会が変わり、'
              '平滑化でばらつきが小さくなっても両群が同方向に集中し得ること。重なりの大きい項目、無機会の多い項目は候補表で区別できる。'
              '単変量の重なりだけで20変量を併用した識別力の上限は判断できない。',
              '次の設計候補は、成立リーチへの応答を意思決定時点の手牌進行度・局中点差・選べた安全牌の有無で区別する機会指標。'
              '条件を細分化すると機会数が減るため、定義と分母の確保を学習期間で検証する。',
              'order_verified=False、full_history_confirmed=False。同一人物の観測履歴に限る。時間・段位・対戦環境が交絡し、'
              '非重複窓にも経時的相関があり得る。因果効果・他者への一般化は評価できない。全期間の探索結果を利用した特徴量選択や'
              '調整は独立評価を損なう。後続モデルの選択・調整は学習期間内で行い、必要なら探索に使っていない将来期間を確保する。']
    (out/'report_ja.md').write_text('\n\n'.join(lines)+'\n', encoding='utf-8')
