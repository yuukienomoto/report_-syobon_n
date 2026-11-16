import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mahjong_research.inputs import write_csv


def show(x):
    return f'{x:.3f}' if pd.notna(x) else '算出不可'


def summarize(folder,internal,results,note,importance):
    fields=['roc_auc','balanced_accuracy','accuracy','f1_high','recall_weak','recall_strong']
    internal=internal.reindex(columns=list(dict.fromkeys([*internal.columns,*fields])))
    pairs=[]
    for (w,fold),g in internal.groupby(['width','fold']):
        indexed=g.set_index('model')
        for added,base in [('C','A'),('D','B')]:
            a,b=indexed.loc[added],indexed.loc[base]
            pairs.append(dict(width=w,fold=fold,contrast=f'{added}-{base}',weak=a.weak,strong=a.strong,
                              added_status=a.status,base_status=b.status,auc_note=a.auc_reason or b.auc_reason,
                              **{f:a[f]-b[f] for f in fields}))
    comparisons=pd.DataFrame(pairs)
    write_csv(folder/'model_comparison.csv',comparisons)
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for i,w in enumerate((10,20)):
        for ax,field in zip(axes[i],fields[:2]):
            for name,g in internal[internal.width.eq(w)].groupby('model'):
                ax.plot(g.fold,g[field],'o-',label=name)
            ax.axhline(.5,color='grey',lw=.6)
            ax.set(xticks=[1,2,3],xlabel='Forward fold',ylim=(-.03,1.03),title=f'{w} games | {field}')
            ax.legend(fontsize=7)
    fig.savefig(folder/'performance_comparison.png',dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(6,4,figsize=(10,14),constrained_layout=True)
    for row in internal[internal.model.isin(list('ABCD'))].itertuples():
        ax=axes[(0 if row.width==10 else 3)+row.fold-1,'ABCD'.index(row.model)]
        ax.set(title=f'{row.width} games F{row.fold} {row.model}',xticks=[0,1],yticks=[0,1],xlabel='Predicted',ylabel='Observed')
        if row.status=='評価済':
            mat=np.array([[row.tn,row.fp],[row.fn,row.tp]])
            ax.imshow(mat,cmap='Blues',alpha=.55)
            for (i,j),v in np.ndenumerate(mat): ax.text(j,i,str(int(v)),ha='center',va='center')
        else: ax.text(.5,.5,'Unavailable',ha='center')
    fig.savefig(folder/'confusion_matrix.png',dpi=150)
    plt.close(fig)
    lines=['# 段位期の分類と選択行動3項目の追加効果',
           '10対局が主分析、20対局が感度分析。欠損はLightGBMの欠損分岐で扱い、閾値は0.5。モデル設定は全分割で固定した。',
           'Aは既存20、Bは学習期間の安定項目、CはA＋新3、DはB＋新3。Bの候補0件時は学習不能、Dは新3項目のみとなる。'
           '新3項目は10対局では全て不安定、20対局では手持ち現物への切替率が高段位期でのみ安定という先行結果を踏まえ、探索的に加えた。']
    for w in (10,20):
        lines.append(f'## {w}対局')
        for fold,g in internal[internal.width.eq(w)].groupby('fold'):
            lines.append(f'- Fold {fold}（評価窓 低{int(g.weak.iloc[0])}・高{int(g.strong.iloc[0])}）。'+
                         '、'.join(f'{r.model} AUC {show(r.roc_auc)} / BA {show(r.balanced_accuracy)}' if r.status=='評価済'
                                  else f'{r.model} {r.status}' for r in g.itertuples()))
        for contrast,g in comparisons[comparisons.width.eq(w)].groupby('contrast'):
            diffs=g.roc_auc.dropna()
            lines.append(f'- {contrast} のAUC差：'+ '、'.join(f'Fold {r.fold} {show(r.roc_auc)}' for r in g.itertuples())+
                         f'。比較可能{len(diffs)}区間中、増加{int((diffs>0).sum())}、減少{int((diffs<0).sum())}。')
    lines+=['## 外部評価',note]
    for r in results[results.dataset.eq('external')].itertuples():
        lines.append(f'- {r.width}対局 {r.model}：'+
                     (f'AUC {show(r.roc_auc)}、BA {show(r.balanced_accuracy)}（低{int(r.weak)}・高{int(r.strong)}）'
                      if r.status=='評価済' else r.status))
    lines+=['## 重要度と制約',
            'gainは学習時の分割利得。Permutation Importanceは評価窓の時間順・ラベルを固定し、特徴量1列の循環シフトによるAUC低下を平均した。'
            '単一クラス期間はAUCとBAを欠測とし、順位相関は共通の選択列だけで算出した。予測確率のレンジが0.01未満ならほぼ定数と記録した。',
            '群内安定性と分類への寄与は別々に解釈する。重要度は因果効果や行動差の向きを表さず、性能が低い区間ではどの打ち方が段位期を分けるかの証拠は限定的である。',
            '段位群をまたがない窓を採用した事後的な分類である。時間と段位・対戦環境は交絡し、時間順分割でも分離できない。'
            '同一時間帯内の順序と全履歴の完全性は未確認。比較可能区間は少なく、全期間・過去の外部結果を既に確認した探索研究である。'
            'しょぼんｎ本人の結果を他者一般へ適用するには未使用データが必要。過去の外部AUC約0.293は異なる分析条件での参考値。']
    (folder/'report_ja.md').write_text('\n\n'.join(lines),encoding='utf8')
