import argparse,json,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from core import extract,DEFINITIONS,PARENTS
from features_100.pipeline import signature as fixed_signature
from features_100.rules import IMPLEMENTED
from decision_features.pipeline import load_c5,checked_xml
from mahjong_research import tenhou
from mahjong_research.analysis import validate_data
from mahjong_research.inputs import digest,sha256,write_json,write_csv,new_run
from study import make_blocks,describe
END=pd.Timestamp('2024-07-01 02:00:00')

def prepare(c):
    data,shared,plan,source=load_c5(c['run_dir'],c.get('key','shobon'),c['player'])
    d,cutoff=validate_data(data,tenhou.FEATURE_NAMES)
    target=d[d.hour.gt(cutoff)&d.hour.le(END)&d.target_label.notna()&~d.game_id.isin(shared)]
    observed=target.target_label.value_counts().to_dict()
    if observed!={0.0:3467,1.0:652}:raise ValueError(f'探索対象が指定3467/652と不一致: {observed}。対象を推測で削除/追加しません。')
    source_dir=Path(c['baseline_dir']); marker=json.loads((source_dir/'complete.json').read_text(encoding='utf8'))
    if not marker['all_reconstructed'] or marker['signature']!=fixed_signature(source):raise ValueError('旧全件署名/完了不一致')
    if sha256(source_dir/'game_counters.csv')!=marker['counters_sha256']:raise ValueError('旧カウンタのハッシュ不一致')
    old=pd.read_csv(source_dir/'game_counters.csv',float_precision='round_trip')
    if not old.game_id.is_unique or set(old.game_id)!=set(d.game_id):raise ValueError('旧全件ID不一致')
    old=old.set_index('game_id').loc[d.game_id]
    for col in ('dan','seat','player','xml_sha256','file_id'):
        if not np.array_equal(old[col].to_numpy(),d[col].to_numpy()):raise ValueError('旧台帳不一致: '+col)
    sig=digest(dict(fixed=fixed_signature(source),source=source,end=str(END),code={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}))
    return d,cutoff,shared,target,old,plan,sig

def run(c,mode):
    d,cutoff,shared,target,old,plan,sig=prepare(c)
    if mode=='pilot':
        target=pd.concat([g.iloc[np.linspace(0,len(g)-1,3,dtype=int)] for _,g in target.groupby('target_label')]).sort_values(['hour','game_id'])
    print('対象',len(target),'探索境界',END,flush=True)
    out=new_run(Path(c['output_root'])/mode);cache=Path(c['output_root'])/'cache'/sig;cache.mkdir(parents=True,exist_ok=True)
    write_csv(out/'selected_games.csv',target[['game_id','file_id','hour','dan','seat']])
    rows=[];audit=[];start=time.perf_counter()
    with zipfile.ZipFile(c.get('zip_override') or plan['zip_path']) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('ZIP重複')
        for n,r in enumerate(target.itertuples(),1):
            identity=dict(game_id=r.game_id,xml_sha256=r.xml_sha256,player=c['player'],seat=int(r.seat),dan=int(r.dan),file_id=r.file_id)
            key=digest(identity);path=cache/key[:2]/(key+'.json');started=time.perf_counter();reused=path.exists()
            try:
                if reused:
                    packet=json.loads(path.read_text(encoding='utf8'));payload=packet['payload']
                    if packet['signature']!=sig or packet['checksum']!=digest(payload) or payload['identity']!=identity:raise ValueError('キャッシュ不一致')
                    result=payload['result']
                else:
                    result=extract(checked_xml(z,r),c['player'],r.seat,r.dan,r.game_id,mode=='pilot')
                for f in PARENTS:
                    for k in ('numerator','denominator','other','unknown','excluded','censored','status','value'):
                        a=result['features'][f][k];b=old.loc[r.game_id,f+'__'+k]
                        if not ((a is None and pd.isna(b)) or a==b):raise ValueError(f'親カウンタ不一致 {f}/{k}: {a}/{b}')
                if mode=='pilot' and result.get('trace'):write_json(out/'traces'/(key+'.json'),dict(identity=identity,result=result))
                # No bulk detailed state; per-round small counters retained for audit.
                compact={k:v for k,v in result.items() if k not in ('trace','call_observations')}
                if not reused:
                    payload=dict(identity=identity,result=compact);path.parent.mkdir(parents=True,exist_ok=True)
                    lock=path.with_suffix('.lock');lock.mkdir()
                    try:
                        if path.exists():raise ValueError('並行書込み。抽出は1セッション限定')
                        write_json(path,dict(signature=sig,checksum=digest(payload),payload=payload))
                    finally:lock.rmdir()
                row=dict(game_id=r.game_id,dan=r.dan,hour=str(r.hour),status='ok')
                for f,v in result['features'].items():row.update({f+'__'+k:x for k,x in v.items()})
                rows.append(row);audit.append(dict(game_id=r.game_id,status='ok',reused=reused,seconds=time.perf_counter()-started))
            except Exception as e:
                audit.append(dict(game_id=r.game_id,status='error',error=str(e),reused=reused,seconds=time.perf_counter()-started))
                write_csv(out/'audit.csv',pd.DataFrame(audit));raise
            if n%50==0:write_csv(out/'audit.csv',pd.DataFrame(audit))
            print(f'{n}/{len(target)} {r.game_id} cache={reused}',flush=True)
    write_csv(out/'audit.csv',pd.DataFrame(audit));write_csv(out/'counters.csv',pd.DataFrame(rows))
    fresh=[v['seconds'] for v in audit if not v['reused']]
    write_json(out/'complete.json',dict(signature=sig,mode=mode,rows=len(rows),sha256=sha256(out/'counters.csv'),end=str(END),elapsed=time.perf_counter()-start,
        extrapolated_4119_hours=float(np.mean(fresh)*4119/3600) if fresh else None,n_uncached=len(fresh),timing_limitation='同環境の単純外挿。牌譜長・Drive・キャッシュ温まりで変動'))
    print('保存先:',out);return out

def analyze(c,folder):
    d,cutoff,shared,target,old,_,sig=prepare(c);folder=Path(folder)
    marker=json.loads((folder/'complete.json').read_text(encoding='utf8'))
    if marker['mode']!='full' or marker['signature']!=sig or marker['sha256']!=sha256(folder/'counters.csv'):raise ValueError('追加全件完了/署名不一致')
    new=pd.read_csv(folder/'counters.csv',float_precision='round_trip')
    if not new.game_id.is_unique or set(new.game_id)!=set(target.game_id):raise ValueError('追加ID不一致')
    new=new.set_index('game_id');features=(*IMPLEMENTED,*DEFINITIONS);rows=[];membership=[]
    for width,starts in make_blocks(d,cutoff,shared)['available'].items():
        if width not in (1,10,20):continue
        for s in starts:
            part=d.iloc[s:s+width]
            if part.hour.iloc[-1]>END:continue
            ids=part.game_id.tolist();wid=f'{width}-{s}';label=int(part.target_label.iloc[0])
            membership.append(dict(width=width,window_id=wid,label=label,ids=json.dumps(ids),hour=str(part.hour.iloc[-1])))
            for f in features:
                block=(new if f in DEFINITIONS else old).loc[ids]
                valid=block[f+'__status'].isin(['ok','no_opportunity']).all()
                num=block[f+'__numerator'].sum() if valid else np.nan;den=block[f+'__denominator'].sum() if valid else np.nan
                rows.append(dict(width=width,window_id=wid,label=label,hour=str(part.hour.iloc[-1]),feature=f,kind=DEFINITIONS[f][0] if f in DEFINITIONS else 'ratio',numerator=num,denominator=den,value=num/den if den>0 else np.nan,status='unknown' if not valid else 'ok' if den>0 else 'zero'))
    w=pd.DataFrame(rows);groups=[];comparisons=[];periods=[]
    for (width,f),g in w.groupby(['width','feature']):
        vals=[]
        for label in (0,1):
            z=g[g.label.eq(label)];v=z.value.dropna();vals.append(v)
            groups.append(dict(width=width,feature=f,label=label,**describe(z.value,z.denominator),unknown_rate=z.status.eq('unknown').mean()))
            for i,idx in enumerate(np.array_split(np.arange(len(z)),3)):
                q=z.sort_values('hour').iloc[idx];periods.append(dict(width=width,feature=f,label=label,period=i+1,**describe(q.value,q.denominator)))
        a,b=vals;sd=np.sqrt(((len(a)-1)*a.var()+(len(b)-1)*b.var())/(len(a)+len(b)-2)) if min(len(a),len(b))>=2 else np.nan
        overlap=np.nan
        if len(a) and len(b):
            edges=np.linspace(0,1,21) if g.kind.iloc[0]=='ratio' else np.linspace(min(a.min(),b.min()),max(a.max(),b.max())+1e-9,21)
            overlap=np.minimum(np.histogram(a,edges)[0]/len(a),np.histogram(b,edges)[0]/len(b)).sum()
        comparisons.append(dict(width=width,feature=f,kind=g.kind.iloc[0],low_n=len(a),high_n=len(b),difference=b.mean()-a.mean(),smd=(b.mean()-a.mean())/sd if sd>0 else np.nan,overlap=overlap))
    corr=[]
    for (width,label),g in w.groupby(['width','label']):
        p=g.pivot(index='window_id',columns='feature',values='value')
        for i,f in enumerate(p.columns):
            for other in p.columns[i+1:]:
                pair=p[[f,other]].dropna();valid=len(pair)>=3 and pair[f].nunique()>1 and pair[other].nunique()>1
                corr.append(dict(width=width,label=label,a=f,b=other,n=len(pair),spearman=pair[f].rank().corr(pair[other].rank()) if valid else np.nan))
    out=new_run(Path(c['output_root'])/'quality')
    for name,frame in [('values.csv',w),('membership.csv',pd.DataFrame(membership)),('groups.csv',pd.DataFrame(groups)),('comparison.csv',pd.DataFrame(comparisons)),('periods.csv',pd.DataFrame(periods)),('correlations.csv',pd.DataFrame(corr))]:write_csv(out/name,frame)
    paired=[]
    for f,(_,parent,_) in DEFINITIONS.items():
        if not parent:continue
        a=w[w.feature.eq(f)];b=w[w.feature.eq(parent)]
        v=a.merge(b,on=['width','window_id','label'],suffixes=('_child','_parent'),validate='one_to_one');v['condition']=f;paired.append(v)
    write_csv(out/'parent_child_same_windows.csv',pd.concat(paired,ignore_index=True))
    (out/'report_ja.md').write_text('# 探索期間の追加特徴量\n割合は分子合計/機会数、量的指標は損失量合計/該当判断数。1対局と非重複10/20対局を分離。親子は同じ窓で比較し、欠損を含む構成差にも注意。二項回帰は今回は実施しない。行動機会を独立とした通常の二項標準誤差は不適切。後続では時期交絡を明示した過分散モデルと時間ブロックによる不確実性評価が必要。量的指標には二項モデルを適用しない。損失量は悪手の尺度ではない。採否や因果・一般化を断定しない。探索境界固定、同一時間帯順序と全履歴は未確認。\n',encoding='utf8')
    print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preview','pilot','full','quality']);p.add_argument('config');p.add_argument('--folder');a=p.parse_args();c=json.loads(Path(a.config).read_text(encoding='utf8'))
    if a.mode=='preview':
        *_,target,old,plan,sig=prepare(c);print('対象',len(target),'署名',sig)
    elif a.mode=='quality':analyze(c,a.folder)
    else:run(c,a.mode)
