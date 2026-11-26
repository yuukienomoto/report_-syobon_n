"""Signed C5 identity reuse, bounded pilot or full extraction, immutable cache."""
from pathlib import Path
import json
import time
import zipfile
import pandas as pd
from mahjong_research import tenhou
from mahjong_research.analysis import validate_data
from mahjong_research.inputs import digest, sha256, new_run, write_json, write_csv
from decision_features.pipeline import load_c5, checked_xml, select_pilot
from . import VERSION
from .extract import extract
from .rules import CATALOG, CATALOG_PATH, IMPLEMENTED, DEFERRED, PARENTS

IDS=(*CATALOG,*PARENTS)
FIELDS=('numerator','denominator','other','unknown','excluded','censored','value','status')


def implementation_status():
    return pd.DataFrame([dict(feature_id=i,english_name=r['english_name'],japanese_name=r['japanese_name'],
        domain=r['domain'],implementation='deferred' if i in DEFERRED else 'implemented_pilot_required',
        reason=DEFERRED.get(i,'合成検証のみ。実牌譜での正例・分母・選択肢の照合は未実施'),
        real_log_validated=False,definition_commit='2079e5ba5af5e3401ba458dd27d464a5f717b5f9') for i,r in CATALOG.items()])


def signature(source):
    paths=list(Path(__file__).parent.glob('*.py'))+[CATALOG_PATH,Path(tenhou.__file__)]
    from decision_features import pipeline as old
    import study
    paths += [Path(old.__file__),Path(study.__file__)]
    return digest(dict(version=VERSION,source=source,code={p.as_posix().split('research_v2/')[-1]:sha256(p) for p in paths}))


def prepare(config,mode):
    if mode not in ('A','B'): raise ValueError('抽出モードはAまたはB')
    data,shared,plan,source=load_c5(config['run_dir'],config.get('key','shobon'),config['player'])
    d,cutoff=validate_data(data,tenhou.FEATURE_NAMES)
    if mode=='A':
        targets=select_pilot(d,cutoff,shared,config.get('pilot_per_group',3),config.get('pilot_ids'))
    else: targets=d.copy()  # retain rank17/initial/shared rows; window eligibility unchanged
    archive=Path(config.get('zip_override') or plan['zip_path'])
    if not archive.is_file(): raise FileNotFoundError(f'必要なZIP: {archive}。zip_overrideに実在パスを設定。')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or set(targets.file_id)-set(names):
            raise ValueError('ZIPに重複エントリまたは対象対局不足。除外して続行しません。')
    sig=signature(source); cache=Path(config['output_root'])/'cache'/sig
    paths=[_cache_path(cache,r.game_id,r.xml_sha256,config) for r in targets.itertuples()]
    cached=sum(p.is_file() for p in paths)
    print(f'{mode}: 対象{len(targets)} / キャッシュ候補{cached} / 新規解析予定{len(targets)-cached}対局')
    print(f'実装{len(IMPLEMENTED)} / 保留{len(DEFERRED)}。保留項目を0で代用しません。\nZIP: {archive}\n出力親: {config["output_root"]}')
    return dict(d=d,shared=shared,targets=targets,archive=archive,source=source,signature=sig,cache=cache,
                cutoff=str(cutoff),mode=mode,config=config)


def validate_packet(packet,sig,identity):
    if packet['signature']!=sig or packet['checksum']!=digest(packet['payload']) or packet['payload']['identity']!=identity:
        raise ValueError('新キャッシュ署名・内容・対局ID不一致。既存キャッシュは上書きしません。')
    return packet['payload']



from concurrent.futures import ProcessPoolExecutor
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
import multiprocessing as mp


def _cache_path(cache, game_id, xml_hash, config):
    name = digest([game_id, xml_hash])
    if config.get("cache_shards", False):
        return cache / name[:2] / (name + ".json")
    return cache / (name + ".json")


def _init_worker(archive, player):
    global _worker_zip, _worker_player
    _worker_zip = zipfile.ZipFile(archive)
    _worker_player = player


def _extract_worker(row, collect_trace):
    r = SimpleNamespace(**row)
    started = time.perf_counter()

    result = extract(
        checked_xml(_worker_zip, r),
        _worker_player,
        r.seat,
        r.dan,
        r.game_id,
        collect_trace
    )

    return result, time.perf_counter() - started


def _parallel_jobs(ctx, pool):
    rows = enumerate(ctx["targets"].itertuples(), 1)

    if pool is None:
        for n, r in rows:
            yield n, r, None
        return

    pending = deque()

    def submit(n, r):
        path = _cache_path(
            ctx["cache"],
            r.game_id,
            r.xml_sha256,
            ctx["config"]
        )

        future = None

        if not path.exists():
            row = dict(
                game_id=r.game_id,
                file_id=r.file_id,
                xml_sha256=r.xml_sha256,
                seat=int(r.seat),
                dan=int(r.dan)
            )

            future = pool.submit(
                _extract_worker,
                row,
                ctx["mode"] == "A"
            )

        pending.append((n, r, future))

    for _ in range(
        2 * int(ctx["config"].get("workers", 1))
    ):
        item = next(rows, None)
        if item is not None:
            submit(*item)

    while pending:
        yield pending.popleft()

        item = next(rows, None)
        if item is not None:
            submit(*item)


def extract_run(ctx):
    out=new_run(ctx['config']['output_root']); rows=[]; audit=[]
    write_csv(out/'selected_games.csv',ctx['targets'])
    write_csv(out/'implementation_status.csv',implementation_status())
    write_csv(out/'feature_definitions.csv',pd.DataFrame(CATALOG.values()))
    write_json(out/'sources.json',dict(ctx['source'],run_dir=ctx['config']['run_dir'],signature=ctx['signature'],
        mode=ctx['mode'],cache=str(ctx['cache']),cutoff=ctx['cutoff'],
        order_verified=False,full_history_confirmed=False,selection='固定した段位群別時期分散抽出' if ctx['mode']=='A' else '全C5対局; 窓集計時に従来条件で除外'))
    ctx['cache'].mkdir(parents=True,exist_ok=True)
    workers=int(ctx['config'].get('workers',1))
    if workers not in (1,2,4):
        raise ValueError('workers must be 1, 2 or 4')

    executor = (
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context('fork'),
            initializer=_init_worker,
            initargs=(
                str(ctx['archive']),
                ctx['config']['player']
            )
        )
        if workers>1 else nullcontext(None)
    )

    with zipfile.ZipFile(ctx['archive']) as z, executor as pool:
        for n,r,future in _parallel_jobs(ctx,pool):
            started=time.perf_counter(); reused=False; trace_only=False; worker_seconds=None
            identity=dict(game_id=r.game_id,file_id=r.file_id,xml_sha256=r.xml_sha256,seat=int(r.seat),dan=int(r.dan),
                          player=ctx['config']['player'],hour=str(r.hour))
            path=_cache_path(ctx['cache'],r.game_id,r.xml_sha256,ctx['config'])
            try:
                if path.exists():
                    payload=validate_packet(json.loads(path.read_text(encoding='utf8')),ctx['signature'],identity); reused=True
                    # Cached source hash is already bound to the signed C5 manifest.
                    # ZIP bytes are read again only for an explicitly requested trace.
                    result=payload['result']
                    if ctx['mode']=='A' and not result['trace']:
                        trace_only=True
                        traced=extract(checked_xml(z,r),identity['player'],r.seat,r.dan,r.game_id,True)
                        if traced['features']!=result['features']: raise ValueError('監査再生と保存カウンタ不一致')
                        result=traced
                else:
                    if future is None:
                        result=extract(checked_xml(z,r),identity['player'],r.seat,r.dan,r.game_id,ctx['mode']=='A')
                    else:
                        result,worker_seconds=future.result()
                    # JSON canonicalization precedes hashing: candidate dict keys
                    # are integers in memory and strings after a JSON round trip.
                    payload=json.loads(json.dumps(dict(identity=identity,result=result,seconds=time.perf_counter()-started),
                                                  ensure_ascii=False,allow_nan=False))
                    # Own namespace only; an existing cache is never replaced.
                    path.parent.mkdir(parents=True,exist_ok=True)
                    lock=path.with_suffix('.lock'); lock.mkdir(exist_ok=False)
                    try:
                        if path.exists(): raise ValueError('同一キャッシュへの並行書込み。1実行だけにしてください。')
                        write_json(path,dict(signature=ctx['signature'],checksum=digest(payload),payload=payload))
                    finally: lock.rmdir()
                record=dict(identity,status='ok',error='')
                for i,values in result['features'].items():
                    record.update({i+'__'+field:values.get(field) for field in FIELDS})
                rows.append(record)
                if ctx['mode']=='A':
                    write_json(out/'traces'/(digest(r.game_id)+'.json'),dict(identity=identity,result=result))
                audit.append(dict(identity,status='ok',error='',reused=reused,trace_only=trace_only,
                    seconds=time.perf_counter()-started,worker_seconds=worker_seconds,rule_profile_verified=result['rules_verified'],**result['audit']))
            except Exception as e:
                error=type(e).__name__+': '+str(e)
                rows.append(dict(identity,status='error',error=error,**{i+'__'+f:None for i in IDS for f in FIELDS}))
                audit.append(dict(identity,status='error',error=error,reused=reused,seconds=time.perf_counter()-started))
            if n%50==0 or n==len(ctx['targets']):
                write_csv(out/'extraction_audit.csv',pd.DataFrame(audit))
            print(f'{n}/{len(ctx["targets"])} {r.game_id}: {audit[-1]["status"]} {audit[-1]["seconds"]:.2f}s'+(' cache' if reused else ''),flush=True)
    frame=pd.DataFrame(rows); write_csv(out/'game_counters.csv',frame)
    summary=[]
    for i in IDS:
        s=frame[i+'__status']; den=pd.to_numeric(frame[i+'__denominator']); num=pd.to_numeric(frame[i+'__numerator'])
        summary.append(dict(feature_id=i,games=len(frame),reconstruction_ok=int(frame.status.eq('ok').sum()),
            extraction_errors=int(frame.status.ne('ok').sum()),unknown_games=int(s.eq('unknown').sum()),
            unimplemented_games=int(s.eq('unimplemented').sum()),zero_denominator_games=int(s.eq('no_opportunity').sum()),
            observed_numerator=num.sum(min_count=1),observed_denominator=den.sum(min_count=1),
            observed_totals_are_partial=bool(s.eq('unknown').any()),
            censored_responses=pd.to_numeric(frame[i+'__censored']).sum(min_count=1)))
    write_csv(out/'feature_audit.csv',pd.DataFrame(summary))
    write_json(out/'complete.json',dict(rows=len(frame),signature=ctx['signature'],
        counters_sha256=sha256(out/'game_counters.csv'),audit_sha256=sha256(out/'extraction_audit.csv'),
        all_reconstructed=bool(frame.status.eq('ok').all()),mode=ctx['mode']))
    (out/'report_ja.md').write_text('# 抽出監査\n\n'+
        f'対象{len(frame)}対局、復元成功{int(frame.status.eq("ok").sum())}。実装{len(IMPLEMENTED)}、保留{len(DEFERRED)}候補。\n\n'+
        '分母ゼロは欠損、unknownを含む割合も欠損です。実装済みは実牌譜検証済みを意味しません。'+
        'feature_audit.csv、extraction_audit.csvとAのtracesを確認してください。局内反復は独立試行ではありません。'+
        '同一時間帯内の順序・全履歴完全性は未確認です。Cは探索期間を指定し別フォルダへ保存します。\n',encoding='utf8')
    print('保存先:',out)
    return out
