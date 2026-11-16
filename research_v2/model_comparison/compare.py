"""Fixed LightGBM A/B/C/D comparison on disjoint observed-hour windows."""
import json
from pathlib import Path
from importlib.metadata import version

import lightgbm as lgb
import numpy as np
import pandas as pd
from windows import FEATURES, NAMES, prepare, select_before, external_data, require
from mahjong_research.analysis import MODEL_PLAN, classification_metrics
from mahjong_research.inputs import new_run, write_csv, write_json, sha256, digest


def metrics(y,p):
    m=classification_metrics(y,p)
    for label,tp,fp,fn in [('high',m['tp'],m['fp'],m['fn']),('low',m['tn'],m['fn'],m['fp'])]:
        m['precision_'+label]=tp/(tp+fp) if tp+fp else np.nan
        m['f1_'+label]=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan
    m['f1_macro']=np.mean([m['f1_high'],m['f1_low']])
    m['f1_strong']=m['f1_high']
    return dict(m,auc_reason='' if m['weak'] and m['strong'] else '評価期間が単一クラス',
                prob_min=p.min(),prob_max=p.max(),prob_sd=p.std(),prob_unique=len(np.unique(p)),
                near_constant=bool(np.ptp(p)<.01))


def score(block,p,key,me,pr):
    y=block.label.to_numpy()
    me.append(dict(key,status='評価済',**metrics(y,p)))
    for wid,label,prob in zip(block.window_id,y,p):
        pr.append(dict(key,window_id=wid,label=int(label),prob_high=prob,prediction=int(prob>=.5)))


def ranks(imp):
    records=[]
    for (w,name),g in imp.groupby(['width','model']):
        for measure in ('gain','permutation_auc'):
            table=g.pivot(index='feature',columns='fold',values=measure)
            for a in table:
                for b in table:
                    if b<=a: continue
                    common=table[[a,b]].dropna()
                    rho=common[a].rank().corr(common[b].rank()) if len(common)>1 and common.nunique().min()>1 else np.nan
                    records.append(dict(width=w,model=name,measure=measure,fold_a=a,fold_b=b,
                                        common_features=len(common),spearman=rho))
    return pd.DataFrame(records)


def fit_fixed(train,cols,folder,name,signature):
    path=folder/(name+'.txt')
    marker=folder/(name+'.json')
    if marker.exists():
        saved=json.loads(marker.read_text(encoding='utf8'))
        require(saved['signature']==signature and saved['features']==cols and sha256(path)==saved['sha256'],
                '保存モデルと今回の学習条件が異なります')
        return lgb.Booster(model_str=path.read_text(encoding='utf8'))
    model=lgb.LGBMClassifier(**MODEL_PLAN).fit(train[cols],train.label).booster_
    path.write_text(model.model_to_string(),encoding='utf8')
    write_json(marker,dict(signature=signature,features=cols,params=MODEL_PLAN,threshold=.5,
                           missing='LightGBM native NaN',sha256=sha256(path)))
    return model


def importance(model,test,cols,p,key):
    gain=model.feature_importance(importance_type='gain')
    base=metrics(test.label.to_numpy(),p)['roc_auc'] if len(test) else np.nan
    rows=[]
    for f,g in zip(cols,gain):
        drops=[]
        if np.isfinite(base):
            # Perturb only the measured column on fixed chronological evaluation rows.
            # Circular offsets preserve most serial adjacency within the column.
            for shift in sorted({max(1,len(test)*k//6) for k in range(1,6)}):
                perturbed=test[cols].copy()
                perturbed[f]=np.roll(perturbed[f],shift)
                probs=model.predict(perturbed,num_threads=1)
                drops.append(base-classification_metrics(test.label,probs)['roc_auc'])
        rows.append(dict(key,feature=f,gain=g,permutation_auc=np.mean(drops) if drops else np.nan,
                         permutations=len(drops),new_feature=f in NAMES))
    return rows


def run(run_dir,phase1,phase2,root,resume=None,ultra_cache=None):
    run_dir,phase1,phase2=map(Path,(run_dir,phase1,phase2))
    print('C5と第1・2段階の入力・窓を照合中',flush=True)
    d,starts,blocks,audit,source=prepare(run_dir,phase1,phase2)
    config=dict(source=source,params=MODEL_PLAN,inputs={p.name:sha256(p) for p in
        [phase1/'fold_membership.csv',phase2/'new_game_features.csv',phase2/'window_values.csv']},
        code={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        versions={n:version(n) for n in ('lightgbm','pandas','numpy','scikit-learn')},
        ultra_cache=str(ultra_cache or ''),external_protocol='固定Fold 3、A/B。C/D追加カウンタはしょぼんｎ専用')
    if resume:
        folder=Path(resume)
        require(json.loads((folder/'protocol.json').read_text(encoding='utf8'))==config,'再開条件が不一致')
    else:
        folder=new_run(root)
        write_json(folder/'protocol.json',config)
    print('保存・再開先',folder,flush=True)
    models=folder/'models'
    models.mkdir(exist_ok=True)
    write_csv(folder/'fold_audit.csv',audit)
    selections,me,pr,im,shift=[],[],[],[],[]
    frozen={}
    missing=[]
    for fold,f_rows in audit.groupby('fold'):
        end=f_rows.train_through.iloc[0]
        selected=select_before(d,starts,end)
        selections.append(selected.assign(fold=fold,train_through=end))
        for row in f_rows.itertuples():
            w=row.width
            block=blocks[w]
            train=block[block['last'].le(end)]
            test=block[block['first'].gt(end)&block['last'].le(row.eval_through)]
            for role,data in [('train',train),('evaluation',test)]:
                for f in list(FEATURES)+list(NAMES):
                    missing.append(dict(width=w,fold=fold,role=role,feature=f,n=len(data),
                                        missing=int(data[f].isna().sum()),missing_rate=data[f].isna().mean()))
            print(f'{w}対局 Fold {fold} 学習{len(train)}・評価{len(test)}',flush=True)
            stable=selected.loc[selected.width.eq(w)&selected.decision.eq('両段位期で安定'),'feature'].tolist()
            for name,cols in [('A',list(FEATURES)),('B',stable),('C',list(FEATURES)+list(NAMES)),
                              ('D',stable+list(NAMES)),('majority',[])]:
                key=dict(dataset='internal',width=w,fold=fold,model=name)
                reason=''
                if train.label.nunique()<2: reason='学習側が空または単一クラス'
                elif not cols and name!='majority': reason='学習期間内の安定候補が0件'
                if reason:
                    me.append(dict(key,status=reason,auc_reason=reason,weak=row.eval_low,strong=row.eval_high))
                    continue
                if name=='majority':
                    p=np.full(len(test),float(train.label.mean()>.5))
                else:
                    sig=digest([config,w,fold,cols])
                    model=fit_fixed(train,cols,models,f'w{w}_f{fold}_{name}',sig)
                    if fold==3: frozen[w,name]=(model,cols,train)
                    p=model.predict(test[cols],num_threads=1) if len(test) else np.array([])
                    im.extend(importance(model,test,cols,p,key))
                if len(test): score(test,p,key,me,pr)
                else: me.append(dict(key,status='評価窓なし',auc_reason='評価窓なし',weak=0,strong=0))
    write_csv(folder/'feature_selection_by_fold.csv',pd.concat(selections))
    write_csv(folder/'missingness_audit.csv',pd.DataFrame(missing))
    internal=pd.DataFrame(me)
    gains=pd.DataFrame(im,columns=['dataset','width','fold','model','feature','gain','permutation_auc','permutations','new_feature'])
    write_csv(folder/'feature_importance.csv',gains)
    write_csv(folder/'importance_rank_agreement.csv',ranks(gains))
    try:
        ext,shared=external_data(run_dir,d,ultra_cache)
        external_note=f'Fold 3の固定モデルを使用。両プレイヤー共通の{shared}対局を含む窓を除外。'
    except (OSError,KeyError,ValueError,TypeError) as exc:
        ext={}
        external_note=f'外部入力の照合が成立しないため評価保留。{exc}'
    for w in (10,20):
        for name in 'ABCD':
            key=dict(dataset='external',width=w,fold=3,model=name)
            reason=''
            if name in 'CD': reason='外部の新3特徴量は取得対象外（第2段階の抽出器はしょぼんｎ本人に限定）'
            elif w not in ext: reason=external_note
            elif (w,name) not in frozen: reason='適用可能なFold 3固定モデルなし'
            elif ext[w].empty: reason='外部評価窓なし'
            if reason:
                me.append(dict(key,status=reason,auc_reason=reason))
                continue
            model,cols,train=frozen[w,name]
            test=ext[w]
            score(test,model.predict(test[cols],num_threads=1),key,me,pr)
            for f in cols:
                shift.append(dict(key,feature=f,train_mean=train[f].mean(),external_mean=test[f].mean(),
                                  train_missing=train[f].isna().mean(),external_missing=test[f].isna().mean(),
                                  train_high_share=train.label.mean(),external_high_share=test.label.mean()))
    write_csv(folder/'external_distribution.csv',pd.DataFrame(shift,columns=['dataset','width','fold','model','feature','train_mean','external_mean','train_missing','external_missing','train_high_share','external_high_share']))
    results=pd.DataFrame(me)
    write_csv(folder/'model_metrics.csv',results)
    predictions=pd.DataFrame(pr,columns=['dataset','width','fold','model','window_id','label','prob_high','prediction'])
    write_csv(folder/'predictions.csv',predictions)
    if len(predictions):
        dist=predictions.groupby(['dataset','width','fold','model','label']).prob_high.describe(
            percentiles=[.1,.25,.5,.75,.9]).reset_index()
        write_csv(folder/'probability_distribution.csv',dist)
    from outputs import summarize
    summarize(folder,internal,results,external_note,gains)
    return folder
