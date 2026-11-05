"""Read-only legacy inputs and content-versioned, resumable v2 extraction.

CSV/ZIP originals are never rewritten. Unresolved target records stop analysis;
they cannot be silently removed to manufacture a contiguous ten-game history.
"""
from pathlib import Path
import hashlib
import json
import gzip
import zipfile
import uuid
from datetime import datetime, timezone
from urllib.parse import unquote
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from .legacy import BASE_FEATURES, canonical_game_id

LEGACY_EXTRACTOR = 'f120fbf4bf1951f6fe0452a6a0aefb2bcb6ed4eae3dcb397a0485ad998e1ecc7'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    frame.to_csv(temp, index=False, encoding='utf-8-sig')
    temp.replace(path)


def new_run(root):
    folder = Path(root) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def find_legacy(cache_root, key, explicit=None):
    if explicit:
        return Path(explicit)
    choices = sorted(p for p in Path(cache_root).glob(key + '_*') if (p/'raw_manifest.csv').is_file())
    if len(choices) != 1:
        raise ValueError(f'{key}: 旧キャッシュ候補={choices}。LEGACY_DIRSに正しいフォルダを指定してください。')
    return choices[0]


def _verify_record(folder, marker, filenames):
    meta = json.loads((folder/marker).read_text(encoding='utf-8'))
    if meta.get('extractor') != LEGACY_EXTRACTOR:
        raise ValueError('旧抽出器の版が想定と違います。自動的な互換扱いはしません。')
    for filename in filenames:
        if sha256(folder/filename) != meta['sha256'].get(filename):
            raise ValueError(f'旧キャッシュのSHA256不一致: {filename}')
    return meta


def load_legacy(folder, player):
    folder = Path(folder)
    scan = _verify_record(folder, 'scan.json', ['raw_manifest.csv','archive_audit.csv'])
    features = _verify_record(folder, 'features.json', ['raw_features.csv'])
    if scan['signature'] != features['signature']:
        raise ValueError('旧台帳と旧特徴量のsignature不一致')
    m, f = pd.read_csv(folder/'raw_manifest.csv'), pd.read_csv(folder/'raw_features.csv')
    for frame in (m, f):
        frame['game_id'] = frame.file_id.map(canonical_game_id)
        if not frame.game_id.is_unique:
            raise ValueError('旧キャッシュの正規化対局ID重複')
    if set(m.game_id) != set(f.game_id):
        raise ValueError('旧台帳と特徴量の対局集合不一致')
    f[BASE_FEATURES] = f[BASE_FEATURES].apply(pd.to_numeric, errors='raise')
    if np.isinf(f[BASE_FEATURES].to_numpy()).any():
        raise ValueError('旧特徴量に無限大')
    d = m.drop(columns=BASE_FEATURES, errors='ignore').merge(f[['game_id']+BASE_FEATURES], on='game_id', validate='one_to_one')
    d['player'] = player
    d['hour'] = pd.to_datetime(d.game_id.str[:10], format='%Y%m%d%H', errors='raise')
    d['dan'] = pd.to_numeric(d.dan, errors='raise')
    if not (d.dan.between(0,20)&d.dan.mod(1).eq(0)).all():
        raise ValueError('旧台帳に不正段位')
    return d.sort_values(['hour','game_id']).reset_index(drop=True)


def plan_extraction(zip_path, player, key, cache_root, legacy_dir=None):
    """Only ZIP directory + small legacy CSVs; does not parse XML."""
    from . import tenhou
    path = Path(zip_path)
    if not path.is_file():
        raise FileNotFoundError(f'改善特徴量には元ZIPが必要です: {path}。旧CSVだけからは再計算できません。')
    with zipfile.ZipFile(path) as archive:
        infos = [i for i in archive.infolist() if i.filename.endswith('.mjlog') and not i.is_dir()]
    items = sorted((i.filename, i.CRC, i.file_size, i.compress_size) for i in infos)
    if not items or len({x[0] for x in items}) != len(items):
        raise ValueError('ZIPに牌譜がない、または同名エントリ重複')
    members = [x[0] for x in items]
    unresolved = 0
    if legacy_dir is not None:
        folder = Path(legacy_dir)
        old = load_legacy(folder, player)
        scan = json.loads((folder/'scan.json').read_text(encoding='utf-8'))
        expected = digest(dict(schema='zip-raw-v1',player=key,name=player,extractor=LEGACY_EXTRACTOR,members=items))
        if expected != scan['signature']:
            raise ValueError('指定ZIPと旧キャッシュの全エントリ情報が不一致。推測で対応付けません。')
        members = sorted(old.file_id.tolist())
        audit = pd.read_csv(folder/'archive_audit.csv')
        if set(audit.loc[audit.status.eq('ok'),'file_id']) != set(members):
            raise ValueError('旧走査成功一覧と台帳の不一致')
        unresolved = int(audit.status.eq('unresolved').sum())
    identity = dict(schema='tenhou-research-v2', extractor_sha256=sha256(tenhou.__file__),
                    player=player, members=items, target_members=members)
    signature = digest(identity)
    folder = Path(cache_root)/(key+'_'+signature[:20])
    done = set()
    for marker in sorted(folder.glob('batch_*.json')):
        meta = json.loads(marker.read_text(encoding='utf-8'))
        datafile = folder/meta['csv']
        if meta['signature'] != signature or sha256(datafile) != meta['sha256']:
            raise ValueError(f'改善版キャッシュの破損・版不一致: {marker}')
        batch = pd.read_csv(datafile)
        if done.intersection(batch.file_id):
            raise ValueError('改善版バッチ間の重複')
        done.update(batch.file_id)
    if not done.issubset(members):
        raise ValueError('改善版キャッシュに予定外の対局')
    return dict(zip_path=str(path), player=player, key=key, signature=signature,
                cache_dir=str(folder), members=members, already_cached=len(done),
                remaining=len(members)-len(done), archive_entries=len(items),
                legacy_dir=str(legacy_dir) if legacy_dir else None,
                legacy_unresolved_entries=unresolved, extractor_sha256=identity['extractor_sha256'],
                order_verified=False, full_history_confirmed=False)


def extract_cached(plan, max_new=None, batch_size=250):
    """Call after displaying plan. max_new bounds real-log pilot work.

    Failing games are retained as error records; a full run refuses to proceed
    to analysis if any target game failed. Re-running reuses signed batches.
    """
    from .tenhou import extract_game
    current = plan_extraction(plan['zip_path'],plan['player'],plan['key'],Path(plan['cache_dir']).parent,plan['legacy_dir'])
    if current['signature'] != plan['signature']:
        raise ValueError('プレビュー後に入力または抽出器が変化しました。再プレビューしてください。')
    folder = Path(plan['cache_dir']); folder.mkdir(parents=True,exist_ok=True)
    parts = [pd.read_csv(folder/json.loads(p.read_text(encoding='utf-8'))['csv']) for p in sorted(folder.glob('batch_*.json'))]
    done = {f for part in parts for f in part.file_id}
    todo = [f for f in plan['members'] if f not in done]
    if max_new is not None:
        todo = todo[:max_new]
    print(f"{plan['key']}: 再利用 {len(done)} / 今回解析 {len(todo)} / 全対象候補 {len(plan['members'])}")
    with zipfile.ZipFile(plan['zip_path']) as archive:
        for start in range(0,len(todo),batch_size):
            records=[]
            for name in todo[start:start+batch_size]:
                record=dict(file_id=name,status='error',error='')
                try:
                    record['game_id']=canonical_game_id(name)
                    payload=archive.read(name)
                    xml=gzip.decompress(payload) if payload[:2]==b'\x1f\x8b' else payload
                    record['xml_sha256']=hashlib.sha256(xml).hexdigest()
                    root=ET.fromstring(xml)
                    names={unquote(n.attrib.get('n'+str(i),'')) for n in root.iter('UN') for i in range(4)}
                    if plan['player'] not in names:
                        if plan['legacy_dir']:
                            raise ValueError('旧台帳で対象だった対局のXMLにプレイヤーがいません')
                        if not names or names=={''}:
                            raise ValueError('UNプレイヤー情報なし')
                        record['status']='different_player'
                    else:
                        record.update(extract_game(xml, plan['player']))
                        record.update(status='ok',player=plan['player'])
                except Exception as exc:
                    record.update(status='error',error=f'{type(exc).__name__}: {exc}')
                records.append(record)
            part=pd.DataFrame(records)
            # New UUID names avoid replacing a prior batch even after interruption.
            stem='batch_'+uuid.uuid4().hex
            write_csv(folder/(stem+'.csv'),part)
            write_json(folder/(stem+'.json'),dict(signature=plan['signature'],csv=stem+'.csv',sha256=sha256(folder/(stem+'.csv'))))
            parts.append(part)
            print(f"保存済み {sum(len(p) for p in parts)} / {len(plan['members'])}")
    allrows=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=['file_id','status'])
    if 'error' in allrows:
        allrows['error']=allrows['error'].fillna('')
    write_csv(folder/'extraction_audit.csv',allrows[[c for c in ['file_id','game_id','status','error','xml_sha256'] if c in allrows]])
    if len(allrows)!=len(plan['members']):
        print('試験抽出のみ。全件完了ではありません。')
        return None
    if (allrows.status=='error').any():
        raise ValueError(f"解析失敗 {(allrows.status=='error').sum()} 件。{folder/'extraction_audit.csv'} を確認。失敗対局を除いて窓を詰め直すことは禁止。")
    data=allrows[allrows.status.eq('ok')].copy()
    if data.empty or not data.game_id.is_unique:
        raise ValueError('対象データなし、または正規化ID重複。自動除去しません。')
    data['hour']=pd.to_datetime(data.game_id.str[:10],format='%Y%m%d%H',errors='raise')
    data=data.sort_values(['hour','game_id']).reset_index(drop=True)
    if plan['legacy_dir']:
        old=load_legacy(plan['legacy_dir'],plan['player']).set_index('game_id')
        if set(old.index)!=set(data.game_id) or not np.array_equal(old.loc[data.game_id,'dan'].to_numpy(),data.dan.to_numpy()):
            raise ValueError('旧版・新版の対局集合または段位が不一致')
    write_csv(folder/'raw_features_v2.csv',data)
    write_json(folder/'complete.json',dict(signature=plan['signature'],rows=len(data),sha256=sha256(folder/'raw_features_v2.csv'),
                extractor_sha256=plan['extractor_sha256'],order_verified=False,full_history_confirmed=False,
                legacy_unresolved_entries=plan['legacy_unresolved_entries']))
    return data


def audit_v2(data, features, out_dir):
    from .tenhou import FEATURE_DEFINITIONS
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    definitions={x['feature']:x for x in FEATURE_DEFINITIONS}
    rows=[]
    for f in features:
        x=pd.to_numeric(data[f],errors='raise')
        definition=definitions[f]
        bad_range=x.notna() & ((x<definition['lower']) | (x>definition['upper']))
        numerator=pd.to_numeric(data[f+'__numerator'],errors='raise')
        denominator=pd.to_numeric(data[f+'__denominator'],errors='raise')
        expected=numerator.div(denominator.where(denominator>0))
        bad_ratio=~(np.isclose(x,expected,equal_nan=True))
        rows.append(dict(feature=f,n=len(x),missing=int(x.isna().sum()),missing_rate=float(x.isna().mean()),
                         minimum=x.min(),maximum=x.max(),infinite=int(np.isinf(x).sum()),
                         out_of_range=int(bad_range.sum()),invalid_ratio=int(bad_ratio.sum()),
                         no_opportunity=int(denominator.eq(0).sum())))
    write_csv(out_dir/'quality_v2.csv',pd.DataFrame(rows))
    write_csv(out_dir/'hour_sizes.csv',data.groupby('hour').size().rename('games').reset_index())
    write_csv(out_dir/'rank_counts.csv',data.groupby('dan').size().rename('games').reset_index())
    write_csv(out_dir/'table_rank_counts.csv',data.groupby(['go_type','dan'],dropna=False).size().rename('games').reset_index() if 'go_type' in data else pd.DataFrame())
    write_json(out_dir/'input_facts.json',dict(rows=len(data),duplicate_ids=int(data.game_id.duplicated().sum()),
               tied_hour_rows=int(data.hour.duplicated(keep=False).sum()),order_verified=False,full_history_confirmed=False,
               time_resolution='hour; ID内の同一時間帯順序は未確定',history='ZIP内観測履歴のみ。完全な生涯履歴を示す台帳は未提供'))
    if data.game_id.duplicated().any() or data.hour.isna().any() or not (data.dan.between(0,20)&data.dan.mod(1).eq(0)).all():
        raise ValueError('対局ID・時刻・段位の検査失敗')
    if any(r['infinite'] or r['out_of_range'] or r['invalid_ratio'] for r in rows):
        raise ValueError('改善特徴量の範囲・分子分母との整合性検査失敗')
    return pd.DataFrame(rows)
