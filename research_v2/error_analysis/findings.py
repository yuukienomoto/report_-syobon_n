import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fmt(x):
    return f'{x:.3f}' if pd.notna(x) else '算出不可'


def distribution_plot(out,block,audit,shifts):
    fig,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
    for row in audit[audit.fold.isin([2,3])].itertuples():
        ref=block[block['last'].le(row.train_through)]
        target=block[block['first'].gt(row.train_through)&block['last'].le(row.eval_through)]
        top=shifts[(shifts.width==10)&(shifts.fold==row.fold)].nlargest(3,'ks')
        for ax,f in zip(axes[row.fold-2],top.feature):
            for g,name in [(ref,'Training'),(target,'Evaluation')]:
                x=np.sort(g[f].dropna())
                if len(x): ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=name)
            ax.set(title=f'Fold {row.fold}',xlabel=f,ylabel='ECDF')
            if ax.lines: ax.legend(fontsize=8)
    fig.savefig(out/'feature_distributions.png',dpi=140)
    plt.close(fig)


def deliver(out,pred,met,diag,shifts,rel,corr,imp,sel,comparisons,changes,ext,delta,external_note):
    lines=['# 第4段階　段位期分類の誤差分析','10対局を主分析、20対局を感度分析とした。']
    for w in (10,20):
        lines.append(f'## {w}対局の内部評価')
        for fold in (2,3):
            part=diag.query('width==@w and fold==@fold and dataset=="internal"')
            shift=shifts.query('width==@w and fold==@fold')
            if shift.empty: continue
            top=shift.dropna(subset=['ks']).nlargest(3,'ks')
            first=shift.iloc[0]
            lines.append(f'### Fold {fold}　高段位窓比率は学習{first.train_high_share:.1%}、評価{first.eval_high_share:.1%}。')
            lines.append('学習・評価間の経験分布の最大距離（KS統計量）上位：'+
                         '、'.join(f'{r.feature} {fmt(r.ks)}' for r in top.itertuples())+'。')
            relation=rel.query('width==@w and fold==@fold').dropna(subset=['opposite_mean_diff'])
            rev=relation[relation.opposite_mean_diff.eq(True)]
            lines.append(f'群平均差の符号が変わった項目は比較可能{len(relation)}項目中{len(rev)}項目。'
                         '小差の符号変化も含む。重なりと方向別AUCはfeature_label_relationship.csvに掲載。')
            if len(rev):
                strongest=rev.loc[(rev.train_auc-rev.eval_auc).abs().idxmax()]
                lines.append(f'そのうち単一指標AUCの変化が最大の {strongest.feature} は'
                             f'{fmt(strongest.train_auc)}→{fmt(strongest.eval_auc)}。')
            for name,g in part.groupby('model'):
                train=g[g.role.eq('train')]
                ev=g[g.role.eq('eval')].set_index('label')
                if ev.empty or train.empty: continue
                lines.append(f'- {name}：学習内AUC {fmt(train.iloc[0].auc)}、未来区間AUC {fmt(ev.iloc[0].auc)}。'
                             f'平均予測確率は低{fmt(ev.at[0,"mean_prob"])}・高{fmt(ev.at[1,"mean_prob"])}、'
                             f'誤分類率は低{fmt(ev.at[0,"error_rate"])}・高{fmt(ev.at[1,"error_rate"])}。')
        selection=sel[sel.width.eq(w)]
        selected=[set(selection.loc[selection[f].eq('両段位期で安定'),'feature']) for f in (2,3)]
        lines.append(f'B・Dの既存項目はFold 2で{len(selected[0])}、Fold 3で{len(selected[1])}、共通{len(set.intersection(*selected))}。')
        lines.append('Fold 3で加わった項目：'+('、'.join(sorted(selected[1]-selected[0])) or 'なし')+
                     '。外れた項目：'+('、'.join(sorted(selected[0]-selected[1])) or 'なし')+'。')
        gi=imp[(imp.width==w)&(imp.model=='A')].dropna(subset=['gain_rank_fold2','gain_rank_fold3']).copy()
        if len(gi):
            gi['rank_change']=gi.gain_rank_fold3-gi.gain_rank_fold2
            top=gi.loc[gi.rank_change.abs().idxmax()]
            lines.append(f'Aでgain順位の変化が最大の項目は {top.feature}（{top.gain_rank_fold2:g}→{top.gain_rank_fold3:g}位）。')
        for r in comparisons[comparisons.width.eq(w)&comparisons.fold.isin([2,3])].itertuples():
            paired=changes.query('width==@w and fold==@r.fold and contrast==@r.contrast') if len(changes) else changes
            movement=paired.probability_change.abs().mean() if len(paired) else np.nan
            lines.append(f'- Fold {r.fold} {r.contrast}：AUC差 {fmt(r.roc_auc)}、同じ窓の予測確率変化の絶対値平均 {fmt(movement)}。')
    lines+=['## 新3特徴量と既存項目の重複']
    for f,g in corr.query('width==10 and fold==3 and role=="train"').groupby('feature'):
        valid=g.dropna(subset=['spearman'])
        if len(valid):
            r=valid.loc[valid.spearman.abs().idxmax()]
            lines.append(f'- {f} と最も順位相関が強い既存項目は {r.existing}（ρ={fmt(r.spearman)}、共通{r.n}窓）。')
        gimp=imp[(imp.width==10)&(imp.model=='C')&(imp.feature==f)]
        if len(gimp):
            r=gimp.iloc[0]
            lines.append(f'  Cの評価列を循環移動したAUC低下はFold 2で{fmt(r.permutation_auc_fold2)}、Fold 3で{fmt(r.permutation_auc_fold3)}。')
        rg=rel[(rel.width==10)&(rel.fold==3)&rel.feature.eq(f)].iloc[0]
        lines.append(f'  段位群間のヒストグラム重なり（20区間）は学習{fmt(rg.train_overlap)}、評価{fmt(rg.eval_overlap)}。')
    lines+=['## 仮説を照合した結果']
    for fold in (2,3):
        g=diag.query('width==10 and fold==@fold and dataset=="internal" and model=="A" and label==0').set_index('role')
        if {'train','eval'}<=set(g.index):
            gap=g.at['train','auc']-g.at['eval','auc']
            lines.append(f'- Fold {fold}のAの学習内−将来AUCは{fmt(gap)}。'+
                ('学習標本への適合が将来に持続しなかったという仮説と整合する。' if gap>0 else '学習内優位という仮説をこの差は支持しない。'))
    lines+=['AUCは高段位窓が低段位窓を上回るペアの割合（同点は半分）。0.5未満はこの標本内での順位の偏りを示す。'
            'クラス比率の違いだけでは、両群内のスコア分布が同じ場合のAUC低下を説明できない。'
            'KSは周辺分布、群平均差と単一指標AUCはラベルとの関連を測る。平均差の符号変化だけでは条件付き関係の反転を確定できない。',
            '重要度は相関した特徴量間で配分される。少数機会・窓内変動、卓条件や暦時間の交絡、標本への過適合の寄与をこの比較だけで分離することは難しい。',
            '## 外部評価',external_note]
    if len(ext):
        for w,g in ext.groupby('width'):
            r=g.iloc[0]
            lines.append(f'- {w}対局：しょぼんｎ学習 {r.train_first}〜{r.train_last}、ウルトラ立直 {r.eval_first}〜{r.eval_last}。'
                f'高段位比率 {r.train_high_share:.1%}→{r.eval_high_share:.1%}。'
                f'共通対局除外前後の外部窓数 {r.windows_without_exclusion}→{r.windows_after_exclusion}。'
                'この件数差には窓の再構成も含む。共通対局を含む評価での性能差は推定していない。')
            top=g.dropna(subset=['ks']).nlargest(2,'ks')
            lines.append('外部との分布差上位：'+'、'.join(f'{i.feature}（KS {fmt(i.ks)}）' for i in top.itertuples())+'。')
            avail=g.dropna(subset=['opposite_mean_diff'])
            lines.append(f'高−低の平均差が学習と逆符号の指標は、比較可能{len(avail)}項目中{int(avail.opposite_mean_diff.sum())}項目。')
            for m in met[(met.width==w)&(met.dataset=='external')&(met.status=='評価済')].itertuples():
                lines.append(f'  固定{m.model}の外部AUC {fmt(m.roc_auc)}、Balanced Accuracy {fmt(m.balanced_accuracy)}。')
    lines+=['## 複雑さを下げる検証','既存20項目のAを基準に、学習内AUCが0.75以上なら、同じ入力でL2ロジスティック回帰を1種類試す。'
            '0.75は学習内の分離がある場合に単純化を試すための探索基準。過学習の診断閾値ではない。C=1、学習標本内の中央値補完・標準化・クラス重み、判定閾値0.5を固定した。'
            '学習区間内の先頭2/3→末尾1/3でも両群が揃う場合に参考AUCを算出する。']
    checks=pd.read_csv(out/'improvement_metrics.csv')
    for r in checks.itertuples():
        if r.status!='評価済':
            lines.append(f'- {r.width}対局 Fold {r.fold}：{r.status}。')
    for r in delta.itertuples():
        lines.append(f'- {r.dataset} {r.width}対局 Fold {r.fold}：AUC {fmt(r.baseline_auc)}→{fmt(r.candidate_auc)}、BA差 {fmt(r.delta_ba)}。')
    primary=delta.query('width==10 and dataset=="internal" and fold in [2,3]').delta_auc.dropna()
    conclusion=('両評価期間でAUCの改善を観測した。' if len(primary)==2 and primary.gt(0).all()
                else '両評価期間で一貫したAUC改善を確認するには至らなかった。')
    lines+=['## 卒論の考察案',f'しょぼんｎの観測履歴では、10対局窓で比較可能な期間は{len(primary)}区間であった。'+conclusion+
            '既に確認したFoldを再使用しており、再評価は探索的である。段位期と時間・対戦環境は交絡し、'
            '他のプレイヤーや未観測期間における強さの原因の特定には、別のデータが必要である。']
    (out/'report_ja.md').write_text('\n\n'.join(lines),encoding='utf8')
    fig,axes=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
    for ax,fold in zip(axes,(2,3)):
        g=shifts.query('width==10 and fold==@fold').nlargest(7,'ks')
        ax.barh(g.feature,g.ks); ax.set(xlabel='Train vs evaluation KS distance',title=f'10 games | Fold {fold}',xlim=(0,1))
    fig.savefig(out/'distribution_shift.png',dpi=130); plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(13,6),constrained_layout=True)
    for ax,(fold,name) in zip(axes.flat,[(f,n) for f in (2,3) for n in 'ABCD']):
        g=pred.query('width==10 and dataset=="internal" and fold==@fold and model==@name')
        for y,color in [(0,'tab:blue'),(1,'tab:orange')]:
            v=np.sort(g.loc[g.label.eq(y),'prob_high'])
            if len(v): ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',color=color,label=['Low','High'][y])
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Score',ylabel='ECDF',title=f'Fold {fold} | {name}')
        if ax.lines: ax.legend(fontsize=8)
    fig.savefig(out/'score_distributions.png',dpi=130); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,4),constrained_layout=True)
    for ax,w in zip(axes,(10,20)):
        g=delta.query('dataset=="internal" and width==@w')
        for col in ('baseline_auc','candidate_auc'): ax.plot(g.fold,g[col],'o-',label=col)
        ax.axhline(.5,color='grey',lw=.7); ax.set(ylim=(0,1),xticks=(1,2,3),title=f'{w} games',ylabel='AUC',xlabel='Fold'); ax.legend()
    fig.savefig(out/'improvement.png',dpi=130); plt.close(fig)
