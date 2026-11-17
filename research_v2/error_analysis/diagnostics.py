"""Descriptive comparisons of disjoint chronological windows and frozen scores."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from windows import FEATURES, NAMES, require


def auc(y, x):
    valid=x.notna()
    return roc_auc_score(y[valid],x[valid]) if y[valid].nunique()==2 else np.nan


def feature_summary(key, ref, target, fields):
    shifts, relations, correlations=[],[],[]
    for f in fields:
        a,b=ref[f].dropna(),target[f].dropna()
        shift=dict(key,feature=f,train_mean=a.mean(),eval_mean=b.mean(),train_sd=a.std(),eval_sd=b.std(),
                   train_missing=ref[f].isna().mean(),eval_missing=target[f].isna().mean(),
                   ks=ks_distance(a,b),
                   train_n=len(ref),eval_n=len(target),train_high_share=ref.label.mean(),eval_high_share=target.label.mean(),
                   train_first=ref['first'].min(),train_last=ref['last'].max(),
                   eval_first=target['first'].min(),eval_last=target['last'].max())
        shifts.append(shift)
        rel=dict(key,feature=f)
        pooled=pd.concat([a,b])
        edges=np.histogram_bin_edges(pooled,bins=20) if len(pooled) else [0,1]
        for role,g in [('train',ref),('eval',target)]:
            low,high=(g.loc[g.label.eq(i),f].dropna() for i in (0,1))
            rel.update({role+'_low_n':len(low),role+'_high_n':len(high),role+'_low_mean':low.mean(),
                        role+'_high_mean':high.mean(),role+'_diff':high.mean()-low.mean(),role+'_auc':auc(g.label,g[f]),
                        role+'_overlap':np.minimum(np.histogram(low,edges)[0]/len(low),
                            np.histogram(high,edges)[0]/len(high)).sum() if len(low) and len(high) else np.nan})
            if f in NAMES:
                for old in FEATURES:
                    pair=g[[old,f]].dropna()
                    correlations.append(dict(key,role=role,feature=f,existing=old,n=len(pair),
                        spearman=pair[old].rank().corr(pair[f].rank()) if len(pair)>2 and pair.nunique().min()>1 else np.nan))
        rel['opposite_mean_diff']=rel['train_diff']*rel['eval_diff']<0 if np.isfinite(rel['train_diff']*rel['eval_diff']) else np.nan
        relations.append(rel)
    return shifts,relations,correlations


def ks_distance(a,b):
    if not len(a) or not len(b):
        return np.nan
    support=np.unique(np.r_[a,b])
    left=np.searchsorted(np.sort(a),support,side='right')/len(a)
    right=np.searchsorted(np.sort(b),support,side='right')/len(b)
    return np.max(np.abs(left-right))


def checked_scores(block,model,cols,saved,metric):
    require(saved.window_id.is_unique and set(saved.window_id)==set(block.window_id),'予測の窓集合不一致')
    s=saved.set_index('window_id').loc[block.window_id]
    p=model.predict(block[cols],num_threads=1)
    require(np.array_equal(s.label,block.label) and np.allclose(s.prob_high,p,rtol=1e-8,atol=1e-10)
            and np.array_equal(s.prediction,p>=.5),'保存モデルとラベル・予測が不一致')
    from compare import metrics
    m=metrics(block.label.to_numpy(),p)
    for k in ('roc_auc','balanced_accuracy','tp','tn','fp','fn'):
        require(np.isclose(m[k],metric[k],equal_nan=True),'保存指標と予測が不一致 '+k)
    return p


def score_rows(key,role,block,p):
    from compare import metrics
    m=metrics(block.label.to_numpy(),p)
    rows=[]
    for y in (0,1):
        v=p[block.label.eq(y)]
        rows.append(dict(key,role=role,label=y,n=len(v),auc=m['roc_auc'],balanced_accuracy=m['balanced_accuracy'],
                         mean_prob=np.mean(v) if len(v) else np.nan,median_prob=np.median(v) if len(v) else np.nan,
                         error_rate=np.mean((v>=.5)!=y) if len(v) else np.nan,
                         tn=m['tn'],tp=m['tp'],fn=m['fn'],fp=m['fp']))
    return rows


def paired_changes(pred):
    internal=pred[pred.dataset.eq('internal')]
    rows=[]
    for (w,fold),g in internal.groupby(['width','fold']):
        for added,base in [('C','A'),('D','B')]:
            a,b=(g[g.model.eq(name)].set_index('window_id').sort_index() for name in (added,base))
            if a.empty or b.empty: continue
            require(a.index.equals(b.index) and a.label.equals(b.label),'追加前後の窓・ラベル不一致')
            for idx,delta in (a.prob_high-b.prob_high).items():
                rows.append(dict(width=w,fold=fold,contrast=added+'-'+base,window_id=idx,label=a.at[idx,'label'],
                                 probability_change=delta,correct_before=b.at[idx,'prediction']==b.at[idx,'label'],
                                 correct_after=a.at[idx,'prediction']==a.at[idx,'label']))
    return pd.DataFrame(rows)
