"""Phase 5. Window-level descriptive comparisons on phase 3/4 populations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from windows import FEATURES,NAMES,prepare,external_data,require,same_table
from mahjong_research.inputs import sha256,new_run,write_csv,write_json

FOCUS=['response_non_genbutsu_rate','response_nonsuji_suited_rate','choose_structural_tenpai',
       'chasing_riichi_share','open_meld_round_rate']


def summary(block,f):
    vals=[block.loc[block.label.eq(y),f].dropna() for y in (0,1)]
    result={}
    for label,prefix,x in zip((0,1),('low','high'),vals):
        n=int(block.label.eq(label).sum())
        for field,value in dict(n=n,valid=len(x),missing=1-len(x)/n if n else np.nan,mean=x.mean(),
                                median=x.median(),sd=x.std(),iqr=x.quantile(.75)-x.quantile(.25)).items():
            result[prefix+'_'+field]=value
    low,high=vals
    sd=np.sqrt(((len(low)-1)*low.var()+(len(high)-1)*high.var())/(len(low)+len(high)-2)) if min(map(len,vals))>=2 else np.nan
    diff=high.mean()-low.mean()
    valid=block[f].notna()
    auc=roc_auc_score(block.label[valid],block[f][valid]) if min(map(len,vals)) else np.nan
    return dict(result,diff=diff,median_diff=high.median()-low.median(),pooled_sd=sd,
                smd=diff/sd if sd>0 else np.nan,auc=auc,rank_biserial=2*auc-1)


def contrast(ref,target,f,key):
    a,b=summary(ref,f),summary(target,f)
    row=dict(key,feature=f,**{'train_'+k:v for k,v in a.items()},**{'eval_'+k:v for k,v in b.items()})
    dl=b['low_mean']-a['low_mean']
    dh=b['high_mean']-a['high_mean']
    delta=b['diff']-a['diff']
    change=delta/a['pooled_sd'] if a['pooled_sd']>0 else np.nan
    flip=a['diff']*b['diff']<0 if np.isfinite(delta) else np.nan
    return dict(row,delta_low=dl,delta_high=dh,delta_diff=delta,delta_median_diff=b['median_diff']-a['median_diff'],
                delta_in_train_sd=change,delta_auc=b['auc']-a['auc'],opposite=flip,
                substantial_reversal=(bool(flip) and abs(change)>=.5 and abs(b['auc']-a['auc'])>=.1)
                if np.isfinite(change) and np.isfinite(b['auc']-a['auc']) else np.nan)


def period_rows(g,w):
    # Assign by window end; no double counting of windows shared by expanding folds.
    time=g['last'].dt.to_period('Q').astype(str)
    rows=[]
    for (quarter,y),sub in g.groupby([time,'label']):
        for f in list(FEATURES)+list(NAMES):
            v=sub[f].dropna()
            rows.append(dict(width=w,quarter=quarter,label=y,feature=f,windows=len(sub),valid=len(v),
                             mean=v.mean(),median=v.median(),q25=v.quantile(.25),q75=v.quantile(.75)))
    return rows


def checked_scores(block,rows,diag):
    same_table(block,rows,['window_id'],['label'])
    p=rows.set_index('window_id').loc[block.window_id,'prob_high'].to_numpy()
    require(np.isfinite(p).all() and ((0<=p)&(p<=1)).all(),'保存予測確率の範囲が不正')
    g=block.assign(score=p)
    stats=summary(g,'score')
    for y,label in enumerate(('low','high')):
        s=diag[diag.label.eq(y)]
        require(len(s)==1 and s.iloc[0].n==stats[label+'_n'] and np.allclose(
            [s.iloc[0].mean_prob,s.iloc[0].auc],[stats[label+'_mean'],stats['auc']],equal_nan=True),
            '第3段階予測と第4段階の群別診断不一致')
    return g


def run(run_dir,p1,p2,p3,p4,root,ultra_cache=None):
    run_dir,p1,p2,p3,p4=map(Path,(run_dir,p1,p2,p3,p4))
    cfg=json.loads((p3/'protocol.json').read_text(encoding='utf8'))
    prov=json.loads((p4/'sources.json').read_text(encoding='utf8'))
    require(prov['protocol_sha256']==sha256(p3/'protocol.json'),'第3・4段階の実行が不一致')
    for path in (p1/'fold_membership.csv',p2/'window_values.csv',p2/'new_game_features.csv'):
        require(cfg['inputs'][path.name]==sha256(path),'入力署名が不一致 '+path.name)
    d,_,blocks,audit,source=prepare(run_dir,p1,p2)
    require(source['raw_sha256']==cfg['source']['raw_sha256'],'C5入力不一致')
    preds=pd.read_csv(p3/'predictions.csv')
    diagnostics=pd.read_csv(p4/'prediction_diagnostics.csv')
    importance=pd.read_csv(p4/'feature_importance_comparison.csv')
    require(not importance.duplicated(['width','feature','model']).any(),'重要度重複')
    phase4=pd.read_csv(p4/'feature_label_relationship.csv')
    tables=d.set_index('game_id')['go_type']
    out=new_run(root)
    records,evidence,trend,conditions,external,sensitivity=[],[],[],[],[],[]
    try:
        ext,shared=external_data(run_dir,d,ultra_cache or cfg.get('ultra_cache'))
        ext_note=f'共通{shared}対局を含む窓を除外。'
    except (ValueError,KeyError,OSError,TypeError) as exc:
        ext={}
        ext_note=f'外部キャッシュ照合を保留：{exc}'
    plot_data={}
    for w,block in blocks.items():
        trend.extend(period_rows(block,w))
        for r in audit[audit.width.eq(w)&audit.fold.isin([2,3])].itertuples():
            fold=r.fold
            train=block[block['last'].le(r.train_through)]
            ev=block[block['first'].gt(r.train_through)&block['last'].le(r.eval_through)]
            plot_data[w,fold]=(train,ev)
            key=dict(width=w,fold=fold)
            for f in list(FEATURES)+list(NAMES):
                records.append(contrast(train,ev,f,key))
            for role,g in [('train',train),('eval',ev)]:
                for segment,idx in enumerate(np.array_split(np.arange(len(g)),4),1):
                    if not len(idx): continue
                    remaining=g.drop(g.index[idx])
                    for f in FOCUS:
                        s=summary(remaining,f)
                        sensitivity.append(dict(key,role=role,segment=segment,feature=f,
                            excluded_first=g.iloc[idx[0]]['first'],excluded_last=g.iloc[idx[-1]]['last'],
                            low_valid=s['low_valid'],high_valid=s['high_valid'],diff=s['diff'],auc=s['auc']))
            for role,g in [('train',train),('eval',ev)]:
                for y,sub in g.groupby('label'):
                    ids=[gid for part in sub.source_ids for gid in part]
                    for go,count in tables.loc[ids].value_counts().items():
                        conditions.append(dict(key,role=role,label=y,go_type=go,games=count,
                            share=count/len(ids),first=sub['first'].min(),last=sub['last'].max()))
            if fold==3 and w in ext:
                external.extend(contrast(train,ext[w],f,dict(key,reference='shobon_train',target='ultra',
                                reference_first=train['first'].min(),reference_last=train['last'].max(),
                                target_first=ext[w]['first'].min(),target_last=ext[w]['last'].max()))
                                for f in FEATURES)
            for model in 'ABCD':
                diag=diagnostics.query('dataset=="internal" and width==@w and fold==@fold and model==@model')
                scores=preds.query('dataset=="internal" and width==@w and fold==@fold and model==@model')
                if scores.empty:
                    require(diag[diag.role.eq('eval')].empty,'診断対象の予測が不足')
                    continue
                scored=checked_scores(ev,scores,diag[diag.role.eq('eval')])
                im=importance[(importance.width==w)&importance.model.eq(model)]
                for item in im.itertuples():
                    gain=getattr(item,f'gain_fold{fold}')
                    if pd.isna(gain): continue
                    f=item.feature
                    pair=scored[[f,'score']].dropna()
                    rho=pair[f].rank().corr(pair.score.rank()) if len(pair)>1 and pair.nunique().min()>1 else np.nan
                    evidence.append(dict(key,model=model,feature=f,valid_pairs=len(pair),score_feature_spearman=rho,
                        eval_score_high_minus_low=scored.loc[scored.label.eq(1),'score'].mean()-scored.loc[scored.label.eq(0),'score'].mean(),
                        gain=gain,gain_rank=getattr(item,f'gain_rank_fold{fold}'),permutation_auc=getattr(item,f'permutation_auc_fold{fold}'),
                        model_eval_auc=diag[diag.role.eq('eval')].iloc[0].auc,
                        model_train_auc=diag[diag.role.eq('train')].iloc[0].auc))
    detail=pd.DataFrame(records)
    checked=['train_diff','eval_diff','train_auc','eval_auc','train_low_mean','train_high_mean','eval_low_mean','eval_high_mean']
    same_table(detail,phase4[phase4.fold.isin([2,3])],['width','fold','feature'],checked)
    extdf=pd.DataFrame(external,columns=list(detail.columns)+['reference','target','reference_first','reference_last','target_first','target_last'])
    ref=pd.read_csv(p4/'external_shift.csv')
    if len(extdf):
        if len(ref): same_table(extdf,ref,['width','fold','feature'],checked)
    elif len(ref):
        # Stage 4 stored group means and AUC, but not group SDs or medians.
        extdf=ref[ref.fold.eq(3)].copy()
        require(not extdf.duplicated(['width','feature']).any(),'外部集計キー重複')
        extdf=extdf.rename(columns={**{role+'_'+group+'_n':role+'_'+group+'_valid'
                    for role in ('train','eval') for group in ('low','high')},
                    'train_first':'reference_first','train_last':'reference_last',
                    'eval_first':'target_first','eval_last':'target_last'})
        extdf=extdf.assign(reference='shobon_train',target='ultra',train_smd=np.nan,eval_smd=np.nan,
                           delta_diff=extdf.eval_diff-extdf.train_diff,delta_auc=extdf.eval_auc-extdf.train_auc)
        ext_note+=' 第4段階の保存集計で平均差・AUCを復元した。群別中央値・SMDの追加計算には外部窓が必要。'
    effects=detail[['width','fold','feature','train_smd','eval_smd','delta_low','delta_high','delta_diff',
                'delta_in_train_sd','train_auc','eval_auc','delta_auc','opposite','substantial_reversal']].copy()
    effects['change_rank']=effects.delta_in_train_sd.abs().groupby([effects.width,effects.fold]).rank(ascending=False)
    linked=pd.DataFrame(evidence).merge(effects,on=['width','fold','feature']) if evidence else pd.DataFrame()
    if len(linked):
        linked['direction_product']=linked.score_feature_spearman*(2*linked.eval_auc-1)
    for name,data in [('feature_relationship_changes',detail),('effect_size_comparison',effects),
                      ('temporal_feature_trends',trend),('external_relationship_comparison',extdf),
                      ('model_feature_alignment',linked),('table_conditions',conditions),('chronological_sensitivity',sensitivity)]:
        write_csv(out/(name+'.csv'),pd.DataFrame(data))
    write_json(out/'sources.json',dict(phase4=str(p4),protocol_sha256=prov['protocol_sha256'],
        phase4_inputs={p:sha256(p4/p) for p in ['prediction_diagnostics.csv','feature_importance_comparison.csv','feature_label_relationship.csv','external_shift.csv']},
        external=ext_note,order_verified=False,full_history_confirmed=False))
    from presentation5 import deliver
    deliver(out,detail,extdf,linked,plot_data,pd.DataFrame(trend),pd.DataFrame(conditions),pd.DataFrame(sensitivity),ext_note)
    return out
