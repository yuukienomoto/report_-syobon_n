import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from relationships import FOCUS


TITLES=dict(zip(FOCUS,('対リーチ応答非現物率','対リーチ応答非現物非スジ数牌率','形状テンパイ選択率',
                      '成立リーチに占める追っかけ割合','1回以上副露した局の割合')))


def fmt(x):
    return f'{x:.3f}' if pd.notna(x) else '算出不可'


def figures(out,plots,trends):
    for w in (10,20):
        fig,axes=plt.subplots(len(FOCUS),2,figsize=(12,14),constrained_layout=True)
        for j,f in enumerate(FOCUS):
            for col,fold in enumerate((2,3)):
                train,test=plots[w,fold]
                ax=axes[j,col]
                for g,style,role in [(train,'--','Train'),(test,'-','Eval')]:
                    for y,color in [(0,'tab:blue'),(1,'tab:orange')]:
                        x=np.sort(g.loc[g.label.eq(y),f].dropna())
                        if len(x):
                            ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',ls=style,color=color,
                                    label=f'{role} {["low","high"][y]} ({len(x)})')
                ax.set(title=f'Fold {fold} | {f}',ylabel='ECDF',ylim=(0,1.03))
                if ax.lines: ax.legend(fontsize=7)
        fig.savefig(out/f'focus_distributions_{w}.png',dpi=150)
        plt.close(fig)
        fig,axes=plt.subplots(len(FOCUS),1,figsize=(12,13),constrained_layout=True)
        for ax,f in zip(axes,FOCUS):
            g=trends[(trends.width==w)&trends.feature.eq(f)]
            for y,color in [(0,'tab:blue'),(1,'tab:orange')]:
                q=g[g.label.eq(y)]
                dates=pd.PeriodIndex(q.quarter,freq='Q').to_timestamp(how='end')
                ax.errorbar(dates,q['median'],yerr=np.array([q['median']-q.q25,q.q75-q['median']]),fmt='o',
                            ms=3,alpha=.75,color=color,label=['Low','High'][y])
            for fold in (2,3):
                ev=plots[w,fold][1]
                if len(ev): ax.axvline(ev['first'].min(),ls=':',color='grey',label=f'First F{fold} eval window')
            ax.set(title=f,xlabel='Quarter of window end',ylabel='Median and IQR')
            ax.legend(fontsize=7,ncol=4)
        fig.savefig(out/f'temporal_trends_{w}.png',dpi=150)
        plt.close(fig)


def deliver(out,detail,external,linked,plots,trends,conditions,sensitivity,ext_note):
    figures(out,plots,trends)
    lines=['# 特徴量と段位期の関係変化',
           '## 1. 分析目的と方法',
           'Fold 2・3の各学習期間と将来評価期間を比較した。低段位期16以下、高段位期18以上。'
           '非重複10対局窓が主分析、20対局窓は感度分析。既存の機会数方式・分母ゼロの欠損を引き継ぎ、'
           '各期間・各段位群の有効窓を等重みとした。単一指標AUCは高値ほど高段位という方向で計算した。',
           '群間差は高−低、SMDは群別自由度で重み付けしたプールSDによる。変化量は（評価の群間差−学習の群間差）を'
           '学習時のプールSDで割り、期間間で比較の尺度を固定した。平均差の逆符号に加え、標準化した変化量の絶対値が0.5以上、'
           '単一指標AUCの変化が絶対値0.10以上なら詳細確認対象とした。この探索基準は統計的有意性を示さない。表には全項目の連続値を掲載した。',
           '## 2. Fold 2とFold 3の違い']
    for (w,fold),g in detail.groupby(['width','fold']):
        valid=g.opposite.dropna()
        eligible=g.substantial_reversal.dropna()
        substantial=g[g.substantial_reversal.eq(True)]
        lines.append(f'- {w}対局 Fold {fold}：逆符号 {int(valid.sum())}/{len(valid)} 項目。'
                     f'大きさの基準も満たしたもの {len(substantial)}/{len(eligible)} 項目。'
                     +('、'.join(substantial.feature) or '該当なし')+'。')
    lines+=['## 3. 重要特徴量の変化',
            '以下は10対局・Fold 3。移動量は同じ段位群の評価平均−学習平均で、対象5項目は割合のためポイントで示す。'
            '高群移動量−低群移動量が群間差の変化に一致する。分布図の破線は学習、実線は評価である。']
    primary=detail.query('width==10 and fold==3')
    for f in FOCUS:
        r=primary[primary.feature.eq(f)].iloc[0]
        movement=('低段位期' if abs(r.delta_low)>abs(r.delta_high) else '高段位期') if pd.notna(r.delta_diff) and abs(r.delta_low)!=abs(r.delta_high) else '同程度または比較不能'
        lines.append(f'- **{TITLES[f]}**：高−低 {fmt(100*r.train_diff)}→{fmt(100*r.eval_diff)}ポイント。'
                     f'低群の移動 {fmt(100*r.delta_low)}、高群 {fmt(100*r.delta_high)}ポイント。絶対移動量が大きい側は{movement}。'
                     f'SMD {fmt(r.train_smd)}→{fmt(r.eval_smd)}、AUC {fmt(r.train_auc)}→{fmt(r.eval_auc)}。')
    lines+=['## 4. 分類性能低下との関連']
    for model in 'ABCD':
        g=linked.query('width==10 and fold==3 and model==@model') if len(linked) else linked
        if g.empty: continue
        r=g.sort_values('gain_rank').iloc[0]
        lines.append(f'- {model} のAUCは学習{fmt(r.model_train_auc)}、評価{fmt(r.model_eval_auc)}。評価スコアの群平均差 {fmt(r.eval_score_high_minus_low)}。'
                     f'gain首位 {r.feature} は、評価窓でスコアとの順位相関 {fmt(r.score_feature_spearman)}、'
                     f'単一指標AUCの学習→評価変化 {fmt(r.delta_auc)}、循環移動によるAUC低下 {fmt(r.permutation_auc)}。')
        adverse=g[g.direction_product.lt(0)].sort_values('gain_rank').head(3)
        lines.append('  評価時の指標とスコアの順位相関が、その指標と高段位ラベルの順位関係と逆向きの項目（gain上位順）：'
                     +('、'.join(adverse.feature) or '該当なし')+'。')
    lines+=['順位相関は評価時の特徴量と多変量予測の観測上の結び付きを表す。共変する特徴量や非線形性の影響を含む。'
            '重要度や単一変数の変化から、その項目を変えた場合の予測性能は推定できない。',
            '## 5. 外部プレイヤーとの比較',ext_note]
    if len(external):
        e=external[external.width.eq(10)]
        r=e.iloc[0]
        lines.append(f'10対局の比較対象期間は、しょぼんｎのFold 3学習 {r.reference_first}〜{r.reference_last}、'
                     f'ウルトラ立直 {r.target_first}〜{r.target_last}。')
        for f in FOCUS:
            sub=e[e.feature.eq(f)]
            if len(sub):
                r=sub.iloc[0]
                lines.append(f'- {TITLES[f]}：低／高段位群の平均はしょぼんｎ {fmt(r.train_low_mean)}／{fmt(r.train_high_mean)}、'
                             f'ウルトラ立直 {fmt(r.eval_low_mean)}／{fmt(r.eval_high_mean)}。両者のSMD {fmt(r.train_smd)}／{fmt(r.eval_smd)}、'
                             f'AUC {fmt(r.train_auc)}／{fmt(r.eval_auc)}。')
        lines.append('外部比較は両者で分子・分母が揃う既存20項目を対象とした。')
    lines+=['## 6. 確認できた事実と未検証の仮説']
    s=sensitivity.query('width==10 and fold==3')
    for f in FOCUS:
        ranges=[]
        for role in ('train','eval'):
            vals=s.loc[s.feature.eq(f)&s.role.eq(role),'diff'].dropna()
            ranges.append(f'{fmt(vals.min())}〜{fmt(vals.max())}（{len(vals)}/4）')
        lines.append(f'- {TITLES[f]}：時間順4区間から順に1区間を除いた群平均差の範囲は、学習{ranges[0]}、評価{ranges[1]}。')
    lines+=['区間除外は特定の時期への依存を確認する感度分析で、信頼区間ではない。'
            '符号の変化に加えて群別移動量・順位差・区間除外の結果が揃う項目ほど、観測上の関係変化として解釈しやすい。',
            '## 7. 研究上の限界']
    for role,g in zip(('学習','評価'),plots[10,3]):
        quarters=[set(g.loc[g.label.eq(y),'last'].dt.to_period('Q')) for y in (0,1)]
        ct=conditions.query('width==10 and fold==3 and role==@part',local_dict={'part':'train' if role=='学習' else 'eval'})
        lines.append(f'Fold 3・{role}で低・高の窓が同じ暦四半期に存在したのは {len(set.intersection(*quarters))} 四半期'
                     f'（いずれかが存在する四半期は{len(set.union(*quarters))}）。'
                     f'元対局の卓条件コード（go_type）は{ct.go_type.nunique()}種類。内訳をtable_conditions.csvに示す。')
    lines+=['四半期の共存は同じ卓条件・同じ相手を保証しない。相手別の実力や対戦機会の全履歴を今回のキャッシュから調整できず、'
            'プレイヤー、時期、段位、卓条件の効果を分離できない。窓は時間依存し得るため独立標本の検定は行わなかった。'
            '全履歴の完全性と同時間帯内順序は未確認。評価結果を調べた事後的な探索である。',
            '## 8. 卒論の結果・考察に使用できる文章案']
    r=primary[primary.feature.eq(FOCUS[0])].iloc[0]
    lines.append(f'非重複10対局窓のFold 3では、群平均差が逆符号になった指標は比較可能'
                 f'{primary.opposite.count()}項目中{int(primary.opposite.sum())}項目であり、変化量とAUC変化の両基準を満たす逆符号は'
                 f'{int(primary.substantial_reversal.sum())}項目であった。対リーチ応答非現物率の単一指標AUCは'
                 f'{fmt(r.train_auc)}から{fmt(r.eval_auc)}へ変化した。学習期間の打牌指標と段位期の関連が将来にどこまで持続するかには'
                 '項目差がある。これらは観測された予測性能を解釈する材料であり、性能低下への個別の因果寄与は未確定である。')
    (out/'report_ja.md').write_text('\n\n'.join(lines),encoding='utf8')
