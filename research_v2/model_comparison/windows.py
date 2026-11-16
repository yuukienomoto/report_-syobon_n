"""Rebuild fold membership from signed caches and verify saved phase 1/2 rows."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from study import FEATURES, read_run, make_blocks, block_values, validate_data
from stability import memberships, split_audit, group_rows, classify, decisions
from choices import NAMES
from mahjong_research.inputs import sha256, _final_frame
from mahjong_research import tenhou


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def same_table(a,b,keys,cols):
    require(not a.duplicated(keys).any() and not b.duplicated(keys).any(),'表のキー重複')
    a=a.set_index(keys).sort_index()
    b=b.set_index(keys).sort_index()
    require(a.index.equals(b.index),'保存済み窓・分割のキー集合不一致')
    for c in cols:
        if pd.api.types.is_numeric_dtype(a[c]):
            ok=np.allclose(a[c],b[c],rtol=1e-12,atol=1e-14,equal_nan=True)
        else:
            ok=a[c].tolist()==b[c].tolist()
        require(ok,f'保存済み結果と列 {c} が不一致')


def join_windows(d,starts,records,saved):
    ids={gid for w in (10,20) for i in starts[w] for gid in d.game_id.iloc[i:i+w]}
    require(set(records.game_id)==ids and records.status.eq('ok').all(),'追加抽出の対象ID・状態不一致')
    same_table(d[d.game_id.isin(ids)],records,['game_id'],['dan','file_id','seat','xml_sha256'])
    indexed=records.set_index('game_id')
    for f in NAMES:
        n,k=indexed[f+'__numerator'],indexed[f+'__denominator']
        require(((n>=0)&(n<=k)&(k%1==0)&(n%1==0)).all() and
                np.allclose(n/k.where(k>0),indexed[f],equal_nan=True),'対局カウンタ不一致 '+f)
    blocks, expected={},[]
    aligned=indexed.reindex(d.game_id)
    for w in (10,20):
        meta=memberships(d,starts[w],w)
        meta['window_id']=meta.source_ids.map(lambda x:x[-1])
        x,_,_=block_values(d,starts[w],w)
        block=pd.concat([meta,pd.DataFrame(x,columns=FEATURES)],axis=1)
        ix=np.array(starts[w],dtype=int)[:,None]+np.arange(w)
        keys=meta.source_ids.map(json.dumps)
        for f in NAMES:
            num,den=(aligned[f+s].to_numpy()[ix].sum(axis=1) for s in ('__numerator','__denominator'))
            v=np.divide(num,den,out=np.full(len(num),np.nan),where=den>0)
            expected.append(pd.DataFrame(dict(width=w,feature=f,source_ids=keys,label=meta.label,
                                               numerator=num,denominator=den,value=v)))
            block[f]=v
        blocks[w]=block
    same_table(pd.concat(expected),saved,['width','feature','source_ids'],['label','numerator','denominator','value'])
    return blocks


def prepare(run,phase1,phase2):
    d,shared,source=read_run(run)
    for p in (phase1,phase2):
        origin=json.loads((p/'sources.json').read_text(encoding='utf8'))
        require(all(origin[k]==source[k] for k in ('raw_sha256','shared_sha256','signature')),'C5入力署名不一致')
    d,cut=validate_data(d,FEATURES)
    for f in FEATURES:
        k=d[f+'__denominator']
        require(np.allclose(d[f],d[f+'__numerator']/k.where(k>0),equal_nan=True),'C5の分子分母が不一致 '+f)
    starts=make_blocks(d,cut,shared)['available']
    audit,members=split_audit(d,cut,starts)
    old=pd.read_csv(phase1/'fold_membership.csv',parse_dates=['first','last'])
    keys=['width','fold','role','source_ids']
    same_table(members,old,keys,['first','last','label'])
    saved=pd.read_csv(phase2/'window_values.csv')
    blocks=join_windows(d,starts,pd.read_csv(phase2/'new_game_features.csv'),saved)
    return d,starts,blocks,audit,source


def select_before(d,starts,end):
    # Raw one-game scale, eligible rate medians and within-rank thirds all use this prefix.
    prefix=d[d.hour.le(end)].copy()
    local={w:[i for i in positions if i+w<=len(prefix)] for w,positions in starts.items()}
    stats,_=group_rows(prefix,local)
    return decisions(classify(stats,5))


def external_data(run,shobon,override=None):
    plan=json.loads((run/'input_plan.json').read_text(encoding='utf8'))['ultra']
    p=Path(override or plan['cache_dir'])
    marker=json.loads((p/'complete.json').read_text(encoding='utf8'))
    raw,log=p/'raw_features_v2.csv',p/'extraction_audit.csv'
    require(marker['signature']==plan['signature'] and marker['extractor_sha256']==sha256(tenhou.__file__)
            and marker['sha256']==sha256(raw) and marker['audit_sha256']==sha256(log)
            and marker['full_history_confirmed'] is False and marker['order_verified'] is False,'外部C5署名不一致')
    data=_final_frame(pd.read_csv(raw,float_precision='round_trip'),'ウルトラ立直')
    audit=pd.read_csv(log)
    require(len(data)==marker['rows'] and audit.file_id.is_unique and audit.status.eq('ok').all()
            and set(data.file_id)==set(audit.file_id)==set(plan['members']),'外部抽出集合不一致')
    for f in FEATURES:
        k=data[f+'__denominator']
        require(np.allclose(data[f],data[f+'__numerator']/k.where(k>0),equal_nan=True),'外部特徴量不一致 '+f)
    d,cut=validate_data(data,FEATURES)
    shared=set(d.game_id)&set(shobon.game_id)
    starts=make_blocks(d,cut,shared)['available']
    blocks={}
    for w in (10,20):
        meta=memberships(d,starts[w],w)
        meta['window_id']=meta.source_ids.map(lambda v:v[-1])
        x,_,_=block_values(d,starts[w],w)
        blocks[w]=pd.concat([meta,pd.DataFrame(x,columns=FEATURES)],axis=1)
    return blocks,len(shared)
