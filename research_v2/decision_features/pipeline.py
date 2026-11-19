"""Small, explicitly bounded pilot from signed C5 inputs; no full extraction."""
from pathlib import Path
import gzip
import hashlib
import json
import time
import zipfile

import numpy as np
import pandas as pd
import choices
from mahjong_research import tenhou
from mahjong_research.analysis import validate_data
from mahjong_research.inputs import _final_frame, sha256, digest, new_run, write_json, write_csv
from study import make_blocks
from . import extract as extractor
from . import definitions
from .definitions import NAMES, DEFINITIONS, RULE, VERSION


def load_c5(run, key, player):
    """Same signed-C5 invariants as study.read_run, with explicit target identity."""
    run = Path(run)
    plan = json.loads((run/'input_plan.json').read_text(encoding='utf8'))[key]
    if plan['player'] != player:
        raise ValueError('input_planと指定playerが不一致です（全角ｎなどを確認）。')
    folder = Path(plan['cache_dir'])
    marker = json.loads((folder/'complete.json').read_text(encoding='utf8'))
    raw, audit = folder/'raw_features_v2.csv', folder/'extraction_audit.csv'
    if (marker['signature'] != plan['signature'] or marker['extractor_sha256'] != sha256(tenhou.__file__)
            or marker['sha256'] != sha256(raw) or marker['audit_sha256'] != sha256(audit)
            or marker['order_verified'] is not False or marker['full_history_confirmed'] is not False):
        raise ValueError('C5署名・版・ハッシュ・未確認フラグの不一致。キャッシュを改変せず入力を確認してください。')
    data = _final_frame(pd.read_csv(raw, float_precision='round_trip'), player)
    log = pd.read_csv(audit, usecols=['file_id', 'status'])
    if (len(data) != marker['rows'] or not log.status.eq('ok').all() or not log.file_id.is_unique
            or set(data.file_id) != set(log.file_id) or set(data.file_id) != set(plan['members'])):
        raise ValueError('C5対局集合・完了監査不一致')
    shared_path = run/'shared_games.csv'
    shared = pd.read_csv(shared_path, usecols=['game_id']).game_id
    if shared.isna().any() or not shared.is_unique or not set(shared) <= set(data.game_id):
        raise ValueError('共通対局一覧と台帳不一致')
    return data, set(shared), plan, dict(raw_sha256=marker['sha256'],
        shared_sha256=sha256(shared_path), signature=marker['signature'], player=player, key=key)


def select_pilot(data, cutoff, shared, per_group=3, explicit_ids=None):
    if not isinstance(per_group, int) or not 1 <= per_group <= 10:
        raise ValueError('per_groupは1〜10。少数試験専用です。')
    eligible = data.hour.gt(cutoff) & ~data.game_id.isin(shared) & data.target_label.notna()
    if explicit_ids is not None:
        if not explicit_ids or len(explicit_ids) > 60 or len(set(explicit_ids)) != len(explicit_ids):
            raise ValueError('明示IDは重複なし1〜60件にしてください。')
        if set(explicit_ids)-set(data.loc[eligible, 'game_id']):
            raise ValueError('明示IDに不足・初期除外・段位17・共通対局があります。主分析規約を維持します。')
        selected = data[data.game_id.isin(explicit_ids)].copy()
        selected['selection_reason'] = 'explicit_ids_before_extraction'
    else:
        pieces = []
        for label in (0, 1):
            group = data[eligible & data.target_label.eq(label)]
            if len(group) < per_group:
                raise ValueError(f'段位群{label}の試験対局不足: {len(group)}')
            # Fixed spread; never select by new-feature values or group differences.
            part = group.iloc[np.linspace(0, len(group)-1, per_group, dtype=int)].copy()
            part['selection_reason'] = 'equally_spaced_positions_in_rank_period'
            pieces.append(part)
        selected = pd.concat(pieces)
    return selected.sort_values(['hour', 'game_id']).reset_index(drop=True)


def code_signature():
    return digest({str(Path(p).name): sha256(p) for p in
                   (extractor.__file__, definitions.__file__, __file__, choices.__file__, tenhou.__file__)})


def prepare(run_dir, output_root, player, key='shobon', zip_override=None, per_group=3, explicit_ids=None):
    data, shared, plan, source = load_c5(run_dir, key, player)
    data, cutoff = validate_data(data, tenhou.FEATURE_NAMES)
    targets = select_pilot(data, cutoff, shared, per_group, explicit_ids)
    archive = Path(zip_override or plan['zip_path'])
    if not archive.is_file():
        raise FileNotFoundError(f'必要な元ZIP: {archive}。P2のZIP_OVERRIDEにDrive上の実在パスを設定。')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or set(targets.file_id)-set(names):
            raise ValueError('ZIPエントリ重複・対象ファイル不足')
    signature = digest(dict(code=code_signature(), source=source, version=VERSION))
    cache = Path(output_root)/'cache'/signature
    cached = sum((cache/(digest([r.game_id, r.xml_sha256])+'.json')).is_file() for r in targets.itertuples())
    print(f'対象{len(targets)}対局 / キャッシュ候補{cached} / 新規解析予定{len(targets)-cached}。全件解析は行いません。')
    print(f'元ZIP: {archive}\n新出力の親: {output_root}\n再利用キャッシュ: {cache}')
    return dict(data=data, starts=make_blocks(data, cutoff, shared)['available'], targets=targets,
                archive=archive, cache=cache, signature=signature, source=source,
                output_root=Path(output_root), cutoff=str(cutoff), player=player)


def checked_xml(z, row):
    """Identical decompressed XML identity rule to choices.pipeline.checked_xml."""
    content = z.read(row.file_id)
    xml = gzip.decompress(content) if content[:2] == b'\x1f\x8b' else content
    if hashlib.sha256(xml).hexdigest() != row.xml_sha256:
        raise ValueError('元XMLのSHA256がC5と不一致')
    return xml


def run_pilot(ctx):
    """Each invocation writes a new report folder; successful game caches are immutable."""
    from .analysis import save_analysis
    out = new_run(ctx['output_root'])
    write_csv(out/'selected_games.csv', ctx['targets'])
    write_json(out/'sources.json', dict(ctx['source'], signature=ctx['signature'], player=ctx['player'],
        archive=str(ctx['archive']), cache=str(ctx['cache']), cutoff=ctx['cutoff'],
        order_verified=False, full_history_confirmed=False, version=VERSION,
        selection='固定した段位群別の時期分散抽出または明示ID。新特徴量の値で選ばない。'))
    write_csv(out/'definitions.csv', pd.DataFrame([dict(d, version=VERSION, rule=RULE) for d in DEFINITIONS]))
    records, events, audits = [], [], []
    with zipfile.ZipFile(ctx['archive']) as z:
        for i, row in enumerate(ctx['targets'].itertuples(), 1):
            started = time.perf_counter()
            identity = dict(game_id=row.game_id, file_id=row.file_id, xml_sha256=row.xml_sha256,
                            player=ctx['player'], seat=int(row.seat), dan=int(row.dan), hour=str(row.hour))
            path = ctx['cache']/(digest([row.game_id, row.xml_sha256])+'.json')
            reused = False
            try:
                # Even cache reuse checks current source XML; no feature re-extraction.
                xml = checked_xml(z, row)
                if path.exists():
                    packet = json.loads(path.read_text(encoding='utf8'))
                    payload = packet['payload']
                    if (packet['signature'] != ctx['signature'] or packet['checksum'] != digest(payload)
                            or payload['identity'] != identity):
                        raise ValueError('追加キャッシュの署名・ハッシュ・ID不一致（自動上書きしません）')
                    reused = True
                else:
                    result = extractor.extract(xml, ctx['player'], row.seat, row.dan)
                    # Pilot-only unchanged legacy replay is a reconstruction cross-check,
                    # not regeneration or replacement of the existing 20-feature cache.
                    legacy = tenhou.extract_game(xml, ctx['player'])
                    for f in tenhou.FEATURE_NAMES:
                        for suffix in ('__numerator', '__denominator'):
                            if legacy[f+suffix] != getattr(row, f+suffix):
                                raise ValueError(f'C5旧20特徴量との不一致: {f+suffix}')
                    if (result['audit']['rounds'] != legacy['audit_rounds']
                            or result['audit']['own_discards'] != legacy['audit_discard_hand_checks']):
                        raise ValueError('旧抽出器と局数・自己打牌数不一致')
                    payload = dict(identity=identity, result=result, legacy20_match=True,
                                   extraction_seconds=time.perf_counter()-started)
                    write_json(path, dict(signature=ctx['signature'], payload=payload, checksum=digest(payload)))
                result = payload['result']
                records.append(dict(identity, **result['counts'], status='ok', error='',
                                    feature_version=VERSION, **result['audit']))
                events.extend(dict(identity, **ev) for ev in result['trace'])
                audits.append(dict(identity, status='ok', error='', reused=reused,
                    extraction_seconds=payload['extraction_seconds'], current_seconds=time.perf_counter()-started,
                    legacy20_match=payload['legacy20_match'], **result['audit']))
            except Exception as exc:
                # Retain failures, never turn them into zero counters or drop their IDs.
                error = f'{type(exc).__name__}: {exc}'
                records.append(dict(identity, status='error', error=error, feature_version=VERSION,
                                    **{f+s: np.nan for f in NAMES for s in ('__numerator', '__denominator')}))
                audits.append(dict(identity, status='error', error=error, reused=False,
                                   current_seconds=time.perf_counter()-started))
            print(f'{i}/{len(ctx["targets"])} {row.game_id}: {audits[-1]["status"]}' + (' (cache)' if reused else ''))
            # Persist audit incrementally; interrupted runs retain completed game caches.
            write_csv(out/'extraction_audit.csv', pd.DataFrame(audits))
    frame = pd.DataFrame(records)
    for f in NAMES:
        den = frame[f+'__denominator']
        frame[f] = frame[f+'__numerator'].div(den.where(den.gt(0)))
        frame[f+'__status'] = np.select([frame.status.ne('ok'), den.eq(0)],
                                        ['extraction_error','no_opportunity'], default='ok')
    write_csv(out/'game_features.csv', frame)
    write_json(out/'decision_traces.json', events)
    save_analysis(ctx, frame, events, pd.DataFrame(audits), out)
    print(f'保存先: {out}')
    if frame.status.ne('ok').any():
        print('抽出失敗あり。失敗は欠損のまま保持。窓集計を停止しました。監査を確認してください。')
    return out
