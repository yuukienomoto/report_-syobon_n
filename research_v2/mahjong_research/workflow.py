"""Colab stages with fixed inputs, resumable caches, and separate outputs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import shutil
import pandas as pd
from .inputs import new_run, write_json, write_csv, sha256, find_legacy, load_legacy, plan_extraction, extract_cached, audit_v2
from .tenhou import FEATURE_NAMES, FEATURE_DEFINITIONS
from .legacy import FEATURES, BASE_FEATURES, add_cross
from .analysis import validate_data, run_descriptive, run_internal, load_frozen_bundle, run_external, LIMITATIONS
from .reporting import compare_descriptive, opportunity_report, performance_report, thesis_report


def initialize(root, code_root, code_commit, zip_paths, legacy_dirs, resume=None):
    root, code_root = Path(root), Path(code_root)
    legacy = {k: find_legacy(root/'validation_inputs'/'zip_cache', k, legacy_dirs.get(k)) for k in ('shobon','ultra')}
    config = dict(code_commit=code_commit, zip_paths={k:str(Path(v)) for k,v in zip_paths.items()},
        legacy_dirs={k:str(v) for k,v in legacy.items()}, root=str(root), initial_exclude=1500, window=10,
        order_verified=False, full_history_confirmed=False,
        source_hashes={p.relative_to(code_root).as_posix():sha256(p) for p in sorted(code_root.rglob('*.py')) if '__pycache__' not in str(p)})
    folder = Path(resume) if resume else new_run(root/'research_v2_runs')
    target = folder/'run_configuration.json'
    if resume:
        if not target.is_file() or json.loads(target.read_text(encoding='utf-8')) != config:
            raise ValueError('再開先のコード・入力設定が一致しません。過去結果に混在させません')
    else:
        write_json(target, config)
        (folder/'definitions').mkdir()
        for name in ['legacy_feature_audit.csv','old_to_v2_mapping.csv','feature_definitions_v2.csv']:
            shutil.copyfile(code_root/name, folder/'definitions'/name)
        write_csv(folder/'assumptions_and_limitations.csv', pd.DataFrame({'limitation':LIMITATIONS}))
        write_json(folder/'runtime_versions.json', dict(python=platform.python_version(),
            packages={p:importlib.metadata.version(p) for p in ('numpy','pandas','matplotlib','scikit-learn','lightgbm')}))
    return dict(folder=folder, root=root, code_root=code_root, config=config, legacy_dirs=legacy, zip_paths=zip_paths)


def record_stage(ctx, stage, **facts):
    path=ctx['folder']/'execution_history.json'
    records=json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    records.append(dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), **facts))
    write_json(path, records)


def preview(ctx):
    names={'shobon':'しょぼんｎ','ultra':'ウルトラ立直'}
    old={k:load_legacy(ctx['legacy_dirs'][k],names[k]) for k in names}
    plans={k:plan_extraction(ctx['zip_paths'][k],names[k],k,ctx['root']/'research_v2_cache',ctx['legacy_dirs'][k]) for k in names}
    shared=set(old['shobon'].game_id)&set(old['ultra'].game_id)
    common=old['shobon'].loc[old['shobon'].game_id.isin(shared),['game_id','dan','file_id']].merge(
        old['ultra'].loc[old['ultra'].game_id.isin(shared),['game_id','dan','file_id']],on='game_id',suffixes=('_shobon','_ultra'),validate='one_to_one')
    evidence={k:{f:sha256(ctx['legacy_dirs'][k]/f) for f in ['raw_manifest.csv','raw_features.csv','archive_audit.csv']} for k in names}
    signature=dict(old_hashes=evidence,plans={k:p['signature'] for k,p in plans.items()})
    locked=ctx['folder']/'input_lock.json'
    if locked.exists() and json.loads(locked.read_text(encoding='utf-8'))!=signature:
        raise ValueError('既存実行の入力が変更されています。混在させません')
    if not locked.exists():
        write_json(locked,signature); write_json(ctx['folder']/'input_plan.json',plans)
        write_csv(ctx['folder']/'shared_games.csv',common)
    record_stage(ctx,'preview',shared_games=len(shared),counts={k:len(v) for k,v in old.items()})
    return old, plans, shared


def pilot(ctx, plan, max_new=8):
    extract_cached(plan,max_new=max_new)
    audit=pd.read_csv(Path(plan['cache_dir'])/'extraction_audit.csv')
    if audit.status.eq('error').any():
        raise ValueError('試験抽出に失敗対局があります。extraction_audit.csvを確認し、全件処理は停止してください')
    record_stage(ctx,'pilot',cached_records=len(audit),scope='少数牌譜の試験。全件検証の完了ではない')
    return audit


def extract_stage(ctx, plan):
    data=extract_cached(plan)
    if data is None: raise ValueError('全件抽出が未完了')
    out=new_run(ctx['folder']/('quality_'+plan['key']))
    quality=audit_v2(data,FEATURE_NAMES,out)
    write_json(out/'scope.json',dict(known_target_games=len(data),legacy_unresolved_entries=plan['legacy_unresolved_entries'],
        full_history_confirmed=False,order_verified=False,
        note='旧走査で対象と確認された対局のみ。未解決エントリの所属・生涯履歴完全性は未確認'))
    record_stage(ctx,'extract_'+plan['key'],rows=len(data),quality_dir=str(out),source_signature=plan['signature'])
    return data, quality


def descriptive_stage(ctx, old, new, shared, make_plots=True):
    if set(old.game_id)!=set(new.game_id): raise ValueError('旧版と新版の対局集合不一致')
    out=new_run(ctx['folder']/'descriptive')
    a=run_descriptive(old,FEATURES,out/'legacy',shared,base_features=BASE_FEATURES,cross_builder=add_cross,make_plots=make_plots)
    b=run_descriptive(new,FEATURE_NAMES,out/'v2',shared,make_plots=make_plots)
    compare_descriptive(a,b,ctx['code_root']/'old_to_v2_mapping.csv',out)
    _,cutoff=validate_data(new,FEATURE_NAMES)
    opportunity_report(new,FEATURE_DEFINITIONS,cutoff,shared,out)
    table=performance_report(out)
    thesis_report(out,b,FEATURE_DEFINITIONS,table,True)
    record_stage(ctx,'descriptive',output_dir=str(out),main_windows=b['summary']['primary_windows'])
    return a,b,out


def model_stage(ctx, data, shared, version):
    if version not in ('legacy','v2'): raise ValueError('unknown model version')
    features=FEATURES if version=='legacy' else FEATURE_NAMES
    base=BASE_FEATURES if version=='legacy' else FEATURE_NAMES
    cross=add_cross if version=='legacy' else None
    out=ctx['folder']/('model_'+version)
    if (out/'external_freeze.json').exists():
        bundle=load_frozen_bundle(out,cross)
        d,_=validate_data(data,base)
        expected=hashlib.sha256(d[['game_id','hour','dan']+base].to_csv(index=False).encode('utf-8')).hexdigest()
        if bundle['protocol']['data_sha256']!=expected or bundle['protocol']['external_forbidden_game_ids']!=sorted(shared):
            raise ValueError('固定モデルと入力データ・共通対局が不一致')
        record_stage(ctx,'model_'+version,action='reused_frozen_model')
        return bundle
    if out.exists() and any(out.iterdir()):
        raise ValueError('モデルの部分出力があります。上書きせず内容を点検してください: '+str(out))
    bundle=run_internal(data,features,out,shared,base_features=base,cross_builder=cross)
    record_stage(ctx,'model_'+version,action='trained_and_frozen',model_dir=str(out))
    return bundle


def external_stage(ctx, bundle, data):
    if bundle['features']!=FEATURE_NAMES: raise ValueError('事前指定の外部評価は改善版20特徴量のみ')
    result=run_external(bundle,data,ctx['folder']/'external_v2')
    record_stage(ctx,'external',scope='固定改善版のみ。特徴量選択・調整には使用しない')
    return result


def final_report(ctx, descriptive, legacy_bundle=None, new_bundle=None, external=None):
    out=new_run(ctx['folder']/'final_report')
    table=performance_report(out,legacy_bundle,new_bundle,external)
    path=thesis_report(out,descriptive,FEATURE_DEFINITIONS,table,True)
    write_json(out/'execution_status.json',dict(descriptive_executed=True,
        legacy_internal_executed=legacy_bundle is not None,new_internal_executed=new_bundle is not None,
        new_external_executed=external is not None,order_verified=False,full_history_confirmed=False))
    record_stage(ctx,'final_report',output_dir=str(out))
    return table,path
