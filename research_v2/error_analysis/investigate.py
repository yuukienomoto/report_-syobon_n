"""Phase 4: verify phase 3, diagnose its errors, then probe a simpler model."""
from pathlib import Path
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from windows import prepare,external_data,FEATURES,NAMES,require,same_table
from compare import metrics
from diagnostics import feature_summary,score_rows,checked_scores,paired_changes
from mahjong_research.inputs import new_run,sha256,digest,write_csv,write_json


def candidate(train):
    def fit(rows):
        model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),
                LogisticRegression(C=1.0,class_weight='balanced',max_iter=2000,random_state=42))
        return model.fit(rows[FEATURES],rows.label)
    # Fixed recipe. The inner check is diagnostic only, never used to choose C.
    split=2*len(train)//3
    early,later=train.iloc[:split],train.iloc[split:]
    inner=np.nan
    if early.label.nunique()==2 and later.label.nunique()==2:
        require(early['last'].max()<later['first'].min(),'内部時系列境界が不正')
        inner=metrics(later.label.to_numpy(),fit(early).predict_proba(later[FEATURES])[:,1])['roc_auc']
    return fit(train),inner


def run(run_dir,phase1,phase2,phase3,output_root,ultra_cache=None):
    run_dir,phase1,phase2,phase3=map(Path,(run_dir,phase1,phase2,phase3))
    config=json.loads((phase3/'protocol.json').read_text(encoding='utf8'))
    d,_,blocks,audit,source=prepare(run_dir,phase1,phase2)
    require(config['source']['raw_sha256']==source['raw_sha256'],'C5署名不一致')
    for p in (phase1/'fold_membership.csv',phase2/'new_game_features.csv',phase2/'window_values.csv'):
        require(config['inputs'][p.name]==sha256(p),'第3段階の入力と不一致 '+p.name)
    def read(stem):
        return pd.read_csv(phase3/(stem+'.csv'))
    met,pred,imp,sel,comparisons=map(read,['model_metrics','predictions','feature_importance','feature_selection_by_fold','model_comparison'])
    old=read('fold_audit')
    for c in ('train_after','train_through','eval_after','eval_through'): old[c]=pd.to_datetime(old[c])
    same_table(audit,old,['width','fold'],list(audit.columns.difference(['width','fold','status'])))
    keys=['dataset','width','fold','model']
    require(not met.duplicated(keys).any() and not pred.duplicated(keys+['window_id']).any(),'指標または予測キー重複')
    out=new_run(output_root)
    print('出力',out,flush=True)
    shifts,rel,corr,diag,extrows,improved,diff=[],[],[],[],[],[],[]
    pending=[]
    totals=pd.read_csv(phase2/'window_values.csv')
    totals['window_id']=totals.source_ids.map(lambda ids:json.loads(ids)[-1])
    opportunities=[]
    try:
        ultra,shared=external_data(run_dir,d,ultra_cache or config.get('ultra_cache'))
        unconstrained,_=external_data(run_dir,d.iloc[:0],ultra_cache or config.get('ultra_cache')) if shared else (ultra,0)
        external_note=f'共通{shared}対局を両プレイヤーの窓から除外。'
    except (OSError,ValueError,KeyError,TypeError) as exc:
        ultra={}
        external_note=f'外部入力照合を保留：{exc}'
    for row in audit.itertuples():
        w,fold=row.width,row.fold
        b=blocks[w]
        tr=b[b['last'].le(row.train_through)]
        ev=b[b['first'].gt(row.train_through)&b['last'].le(row.eval_through)]
        key=dict(width=w,fold=fold,dataset='internal')
        s,r,c=feature_summary(key,tr,ev,list(FEATURES)+list(NAMES))
        shifts.extend(s); rel.extend(r); corr.extend(c)
        for role,sub in [('train',tr),('eval',ev)]:
            g=totals[totals.width.eq(w)&totals.window_id.isin(sub.window_id)]
            for (f,y),v in g.groupby(['feature','label']):
                opportunities.append(dict(key,role=role,feature=f,label=y,n=len(v),valid=v.value.count(),
                    median_opportunities=v.denominator.median(),q25_opportunities=v.denominator.quantile(.25),
                    q75_opportunities=v.denominator.quantile(.75),rate_sd=v.value.std(),rate_iqr=v.value.quantile(.75)-v.value.quantile(.25)))
        if fold==3 and w in ultra:
            ext=ultra[w]
            s,r,_=feature_summary(dict(key,dataset='external'),tr,ext,FEATURES)
            for a,z in zip(s,r):
                a.update(z,shared_games=shared,windows_without_exclusion=len(unconstrained[w]),
                         windows_after_exclusion=len(ext),high_without_exclusion=int(unconstrained[w].label.sum()),
                         high_after_exclusion=int(ext.label.sum()))
            extrows.extend(s)
        print(f'保存モデル照合 {w}対局 Fold {fold}',flush=True)
        for name in 'ABCD':
            metric=met.query('dataset=="internal" and width==@w and fold==@fold and model==@name').iloc[0]
            modelpath=phase3/'models'/f'w{w}_f{fold}_{name}.txt'
            if not modelpath.exists():
                require(metric.status!='評価済','評価済みモデルが不足')
                continue
            marker=json.loads(modelpath.with_suffix('.json').read_text(encoding='utf8'))
            cols=marker['features']
            require(marker['signature']==digest([config,w,fold,cols]) and marker['sha256']==sha256(modelpath),'モデル署名不一致')
            model=lgb.Booster(model_str=modelpath.read_text(encoding='utf8'))
            require(model.feature_name()==cols,'モデル入力列順不一致')
            stable=sel.loc[(sel.width==w)&(sel.fold==fold)&sel.decision.eq('両段位期で安定'),'feature'].tolist()
            expected=(list(FEATURES) if name in 'AC' else stable)+(list(NAMES) if name in 'CD' else [])
            require(cols==expected,'保存時の選択表とモデル列が不一致')
            gains=imp.query('width==@w and fold==@fold and model==@name').set_index('feature')
            require(gains.index.is_unique and set(gains.index)==set(cols) and np.allclose(
                    gains.loc[cols,'gain'],model.feature_importance(importance_type='gain')),'gain重要度の不一致')
            k=dict(key,model=name)
            trainrows=score_rows(k,'train',tr,model.predict(tr[cols],num_threads=1))
            diag.extend(trainrows)
            if len(ev):
                saved=pred.query('dataset=="internal" and width==@w and fold==@fold and model==@name')
                p=checked_scores(ev,model,cols,saved,metric)
                diag.extend(score_rows(k,'eval',ev,p))
            if name=='A':
                pending.append((row,tr,ev,trainrows[0]['auc'],metric))
            if fold==3 and w in ultra and name in 'AB' and len(ultra[w]):
                em=met.query('dataset=="external" and width==@w and model==@name').iloc[0]
                if em.status=='評価済':
                    saved=pred.query('dataset=="external" and width==@w and model==@name')
                    p=checked_scores(ultra[w],model,cols,saved,em)
                    diag.extend(score_rows(dict(k,dataset='external'),'eval',ultra[w],p))
    changes=paired_changes(pred)
    for r in comparisons.itertuples():
        add,base=r.contrast.split('-')
        g=met.query('dataset=="internal" and width==@r.width and fold==@r.fold').set_index('model')
        require(np.isclose(g.at[add,'roc_auc']-g.at[base,'roc_auc'],r.roc_auc,equal_nan=True),'追加効果とAUC不一致')
    rankkeys=['width','model','feature']
    imp=imp.copy()
    for field in ('gain','permutation_auc'):
        imp[field+'_rank']=imp.groupby(['width','fold','model'])[field].rank(ascending=False)
    impcomp=imp[imp.fold.eq(2)].merge(imp[imp.fold.eq(3)],how='outer',on=rankkeys,
                suffixes=('_fold2','_fold3'),validate='one_to_one')
    selection=sel.pivot(index=['width','feature'],columns='fold',values='decision').reset_index()
    for stem,data in [('distribution_shift',shifts),('feature_label_relationship',rel),('new_feature_correlations',corr),
                      ('new_feature_opportunities',opportunities),('external_shift',extrows),('prediction_diagnostics',diag),('paired_score_changes',changes),
                      ('feature_importance_comparison',impcomp),('selection_comparison',selection)]:
        write_csv(out/(stem+'.csv'),pd.DataFrame(data) if len(data) else pd.DataFrame(columns=['dataset','width','fold','feature']))
    # One pre-specified complexity probe on the externally measurable A inputs.
    # Only training resubstitution AUC enters the gate. No validation-dependent tuning.
    for row,tr,ev,train_auc,baseline in pending:
        w,fold=row.width,row.fold
        k=dict(width=w,fold=fold,model='L2 logistic',train_A_auc=train_auc)
        if not np.isfinite(train_auc) or train_auc<.75:
            improved.append(dict(k,dataset='internal',status='学習AUC 0.75未満または未定義のため適用保留'))
            continue
        model,inner=candidate(tr)
        for dataset,test in [('internal',ev),('external',ultra.get(w) if fold==3 else None)]:
            if test is None or not len(test): continue
            a=met.query('width==@w and fold==@fold and model=="A" and dataset==@dataset').iloc[0]
            m=metrics(test.label.to_numpy(),model.predict_proba(test[FEATURES])[:,1])
            improved.append(dict(k,dataset=dataset,status='評価済',inner_auc=inner,**m))
            diff.append(dict(k,dataset=dataset,baseline_auc=a.roc_auc,candidate_auc=m['roc_auc'],
                delta_auc=m['roc_auc']-a.roc_auc,delta_ba=m['balanced_accuracy']-a.balanced_accuracy))
    result=pd.DataFrame(improved)
    if not len(result):
        result=pd.DataFrame(columns=['width','fold','model','dataset','status','roc_auc','balanced_accuracy'])
    delta=pd.DataFrame(diff,columns=['width','fold','model','train_A_auc','dataset','baseline_auc','candidate_auc','delta_auc','delta_ba'])
    write_csv(out/'improvement_metrics.csv',result)
    write_csv(out/'improvement_comparison.csv',delta)
    write_json(out/'sources.json',dict(phase3=str(phase3),input_hashes=config['inputs'],protocol_sha256=sha256(phase3/'protocol.json'),
        external=external_note,order_verified=False,full_history_confirmed=False))
    from findings import deliver,distribution_plot
    distribution_plot(out,blocks[10],audit[audit.width.eq(10)],pd.DataFrame(shifts))
    deliver(out,pred,met,pd.DataFrame(diag),pd.DataFrame(shifts),pd.DataFrame(rel),pd.DataFrame(corr),
            impcomp,selection,comparisons,changes,pd.DataFrame(extrows),delta,external_note)
    return out
