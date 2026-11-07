"""Drive entry points. Verified batches are shared by pilot and full runs."""
from pathlib import Path
import gzip
import hashlib
import json
import time
import zipfile

import numpy as np
import pandas as pd
from choices import DEFINITIONS, NAMES, RULE, extract
import choices
from study import FEATURES, read_run, validate_data, make_blocks, block_values, describe
from stability import measured, group_rows, classify, decisions
from mahjong_research.inputs import sha256, digest, new_run, write_csv, write_json
from mahjong_research.tenhou import extract_game


def prepare(run_dir, output_root, zip_override=None):
    run_dir = Path(run_dir)
    data, shared, source = read_run(run_dir)
    d, cutoff = validate_data(data, FEATURES)
    starts = make_blocks(d, cutoff, shared)['available']
    positions = sorted({j for w in (10,20) for i in starts[w] for j in range(i,i+w)})
    if not positions:
        raise ValueError('対象窓がありません')
    targets = d.iloc[positions]
    plan = json.loads((run_dir/'input_plan.json').read_text(encoding='utf8'))['shobon']
    archive = Path(zip_override or plan['zip_path'])
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names)!=len(set(names)) or set(targets.file_id)-set(names):
            raise ValueError('ZIP内の同名エントリまたは対象対局不足')
    out = new_run(output_root)
    # Same deterministic pilot games form the first batch; later runs reuse it.
    pilot = targets.groupby('target_label').head(2)
    rest = targets.drop(index=pilot.index)
    batches = [pilot]+[rest.iloc[i:i+64] for i in range(0,len(rest),64)]
    key = digest([source['raw_sha256'],source['extractor_sha256'],sha256(choices.__file__),
                  sha256(__file__),targets.game_id.tolist()])
    cache = Path(output_root)/'cache'/key
    cache.mkdir(exist_ok=True,parents=True)
    write_json(out/'sources.json',dict(source,zip=str(archive),cache=str(cache),target_games=len(targets),
        order_verified=False,full_history_confirmed=False))
    write_csv(out/'definitions.csv',pd.DataFrame([dict(feature=f,japanese=name,denominator=den,
        numerator=num,condition=RULE,scope='形状判定は役・フリテン・残り枚数を考慮しない')
        for f,(name,den,num) in zip(NAMES,DEFINITIONS)]))
    print(f'対象 {len(targets)} 対局、試験 {len(pilot)} 対局、既存バッチ {len(list(cache.glob("batch_*.json")))}/{len(batches)}。保存先 {out}')
    return dict(data=d,starts=starts,batches=batches,out=out,archive=archive,cache=cache,key=key)


def checked_xml(z,row):
    b = z.read(row.file_id)
    xml = gzip.decompress(b) if b[:2]==bytes([31,139]) else b
    if hashlib.sha256(xml).hexdigest()!=row.xml_sha256:
        raise ValueError('元XMLのSHA256がC5と不一致')
    return xml


def extract_batches(ctx,pilot_only=False):
    allrows = []
    out = ctx['out']
    with zipfile.ZipFile(ctx['archive']) as z:
        for i,frame in enumerate(ctx['batches']):
            if pilot_only and i>0: break
            path = ctx['cache']/f'batch_{i:04}.json'
            if path.exists():
                packet = json.loads(path.read_text(encoding='utf8'))
                rows = packet['rows']
                if (packet['key']!=ctx['key'] or digest(rows)!=packet['checksum'] or
                    [r['game_id'] for r in rows]!=frame.game_id.tolist()):
                    raise ValueError(f'バッチの署名・ID不一致 {path}')
            else:
                rows = []
                for row in frame.itertuples():
                    try:
                        xml = checked_xml(z,row)
                        trace = [] if i==0 else None
                        start = time.perf_counter()
                        values = extract(xml,row.seat,row.dan,trace)
                        elapsed = time.perf_counter()-start
                        if (values['own_discards'] != row.audit_discard_hand_checks or
                            values['rounds'] != row.audit_rounds or
                            values[NAMES[0]+'__denominator'] > row.first_response_genbutsu_rate__denominator):
                            raise ValueError('新旧抽出器で打牌数・局数または初回応答の包含関係が不一致')
                        if i==0:
                            old = extract_game(xml,'しょぼんｎ')
                            for f in FEATURES:
                                for suffix in ('__numerator','__denominator'):
                                    if old[f+suffix] != frame.loc[row.Index,f+suffix]:
                                        raise ValueError('試験対局の旧20特徴量カウンタ不一致')
                        rows.append(dict(game_id=row.game_id,file_id=row.file_id,xml_sha256=row.xml_sha256,
                                         dan=int(row.dan),seat=int(row.seat),status='ok',seconds=elapsed,
                                         pilot_trace=trace,**values))
                    except Exception as exc:
                        write_json(out/'extraction_failure.json',dict(game_id=row.game_id,file_id=row.file_id,
                            error=f'{type(exc).__name__}: {exc}',batch=i))
                        raise
                write_json(path,dict(key=ctx['key'],rows=rows,checksum=digest(rows)))
            allrows.extend(rows)
    records=pd.DataFrame(allrows)
    if not records.game_id.is_unique:
        raise ValueError('抽出対局重複')
    pilot_records = [dict(game_id=r['game_id'],**e) for r in allrows for e in r['pilot_trace'] or []]
    write_json(out/'pilot_events.json',pilot_records)
    records=records.drop(columns='pilot_trace')
    for f in NAMES:
        den=records[f+'__denominator']
        records[f]=records[f+'__numerator'].div(den.where(den>0))
    write_csv(out/('pilot_games.csv' if pilot_only else 'new_game_features.csv'),records)
    if not pilot_only:
        write_csv(out/'extraction_audit.csv',records[['game_id','file_id','xml_sha256','status','seconds',
                                                      'rounds','own_discards','free_responses','excluded_responses']])
    return records


def analyze(ctx, records):
    d,starts,out=ctx['data'],ctx['starts'],ctx['out']
    expected=set(pd.concat(ctx['batches']).game_id)
    if not records.game_id.is_unique or set(records.game_id)!=expected:
        raise ValueError('全対象の抽出完了が必要です')
    aligned=records.set_index('game_id').reindex(d.game_id)
    old_groups,_=group_rows(d,starts)
    limits=classify(old_groups,5).query('rate').groupby(['width','label'])[['sd_limit','iqr_limit']].first()
    stats,periods,corr,windows=[],[],[],[]
    for w in (10,20):
        old,_,labels=block_values(d,starts[w],w)
        offsets=np.asarray(starts[w],int)
        ix=offsets[:,None]+np.arange(w)
        for j,f in enumerate(NAMES):
            def total(suffix):
                a=aligned[f+suffix].to_numpy(float)[ix]
                if not np.isfinite(a).all(): raise ValueError('窓に未抽出対局あり')
                return a.sum(axis=1)
            num,den=total('__numerator'),total('__denominator')
            x=np.divide(num,den,out=np.full(len(den),np.nan),where=den>0)
            for k,start in enumerate(offsets):
                windows.append(dict(width=w,feature=f,label=labels[k],source_ids=json.dumps(d.game_id.iloc[start:start+w].tolist()),
                                    numerator=num[k],denominator=den[k],value=x[k]))
            for label in (0,1):
                key=dict(width=w,label=label,feature=f,japanese=DEFINITIONS[j][0],rate=True)
                ids=np.flatnonzero(labels==label)
                meds,ns=[],[]
                for period,p in zip(('前','中','後'),np.array_split(ids,3)):
                    s=describe(x[p],den[p])
                    periods.append(dict(key,period=period,**s))
                    meds.append(s['median'])
                    ns.append(s['valid_n'])
                stats.append(dict(key,**measured(x[ids],den[ids],True),raw_sd=np.nan,raw_iqr=np.nan,
                                  median_range=np.ptp(meds),min_period_valid=min(ns)))
            for label,mask in [('全体',np.ones(len(x),bool)),('低',labels==0),('高',labels==1)]:
                for k,other in enumerate(FEATURES):
                    pair=pd.DataFrame({'new':x[mask],'old':old[mask,k]}).dropna()
                    rho=pair['new'].rank().corr(pair.old.rank()) if len(pair)>=3 and pair.nunique().min()>1 else np.nan
                    corr.append(dict(width=w,group=label,feature=f,existing=other,n=len(pair),spearman=rho))
    # Phase 1 quality gates are reused. Apply the original eligible old-rate
    # medians, avoiding a threshold change caused by adding our three features.
    assessed=classify(pd.DataFrame(stats),5)
    for i,r in assessed.iterrows():
        sd,iqr=limits.loc[(r.width,r.label)]
        assessed.loc[i,['sd_limit','iqr_limit']]=[sd,iqr]
        if r.status=='判断困難': continue
        if not np.isfinite([sd,iqr]).all():
            assessed.loc[i,['status','reason']]=['判断困難','既存率の比較基準不足']
            continue
        reasons=[name for name,flag in [('SD基準超',r.sd>sd),('IQR基準超',r.iqr>iqr),
                                      ('期間中央値の範囲が群内IQR超',r.median_range>r.iqr)] if flag]
        assessed.loc[i,['status','reason']]=['不安定' if reasons else '安定','・'.join(reasons) or '全条件充足']
    decision=decisions(assessed)
    for name,frame in [('group_statistics',assessed),('period_statistics',periods),('stability_decisions',decision),
                       ('correlations',corr),('window_values',windows)]:
        write_csv(out/(name+'.csv'),pd.DataFrame(frame))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    win=pd.DataFrame(windows)
    lines=['# 選択機会を条件とした打牌指標',RULE,'窓内で分子・分母を合計し、群内で有効窓を等重みとした。']
    for ax,r in zip(axes.flat,decision.sort_values(['width','feature']).itertuples()):
        sub=win[win.feature.eq(r.feature)&win.width.eq(r.width)]
        for label,c in [(0,'tab:blue'),(1,'tab:orange')]:
            vals=np.sort(sub.loc[sub.label.eq(label),'value'].dropna())
            if len(vals): ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',color=c,label=['Low','High'][label])
        ax.set_title(f'{r.width} games | {r.feature}',fontsize=9)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Rate',ylabel='ECDF')
        if ax.lines: ax.legend()
        lines.append(f'- {r.width}対局・{r.japanese}：{r.decision}。低段位期 {r.reason_low}、高段位期 {r.reason_high}。')
    fig.savefig(out/'distributions.png',dpi=150)
    plt.close(fig)
    lines.extend(['比較基準は同じ窓の既存率から求めた第1段階のSD・IQR中央値。機会数中央値5以上、各期間有効5窓以上等の条件も適用した。',
                  '相関表は対ごとの有効窓で計算した順位相関。定数や3窓未満は欠測。相関の強さだけではモデルへの追加価値は判断できない。',
                  '同一人物の全期間の探索的記述。段位と時間が交絡し、履歴の完全性と同一時間帯内順序は未確認。'])
    for f,(name,_,_) in zip(NAMES,DEFINITIONS):
        paired=pd.DataFrame(corr).query('width==10 and group=="全体" and feature==@f').dropna(subset='spearman')
        if len(paired):
            top=paired.loc[paired.spearman.abs().idxmax()]
            lines.append(f'- {name}：10対局全体で絶対相関が最大の既存項目は {top.existing}。'
                         f'順位相関 {top.spearman:.3f}、共通有効{top.n}窓。')
    (out/'report_ja.md').write_text('\n\n'.join(lines),encoding='utf8')
    return decision
