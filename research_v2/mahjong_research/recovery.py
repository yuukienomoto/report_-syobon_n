"""One audited migration for the disclosed own-four-tiles validation fix.

Only batches signed by the exact predecessor extractor are accepted. Inputs
and prior caches stay intact. Failed rows are re-extracted, never dropped from
the target history. No model may be reused across changed source fingerprints.
"""
import json
import uuid
import zipfile
from pathlib import Path
import pandas as pd
from .inputs import digest, sha256, write_json, write_csv, plan_extraction

PREVIOUS_EXTRACTOR = 'ef6322c05c21f299205bae1fdc5448b5739c6da7280ed619ae37219d4d4a7fa2'
FAILED_ID = '2012052419gm-0029-0000-eb0d38e9'


def recover_successful_batches(new_plan, old_cache):
    """Copy verified successes to a NEW cache, leaving just the failed game.

    The fix changes only a check that previously raised an exception. Successful
    games have identical 20 features. Migration does not claim new extraction.
    """
    old_cache = Path(old_cache)
    new_cache = Path(new_plan['cache_dir'])
    if old_cache.resolve() == new_cache.resolve():
        raise ValueError('Recovery must use a separate output cache')
    with zipfile.ZipFile(new_plan['zip_path']) as archive:
        items = sorted((i.filename,i.CRC,i.file_size,i.compress_size) for i in archive.infolist()
                       if i.filename.endswith('.mjlog') and not i.is_dir())
    previous_signature = digest(dict(schema='tenhou-research-v2',extractor_sha256=PREVIOUS_EXTRACTOR,
        player=new_plan['player'],members=items,target_members=new_plan['members']))
    parts, seen, failed = [], set(), []
    for marker in sorted(old_cache.glob('batch_*.json')):
        meta = json.loads(marker.read_text(encoding='utf-8'))
        csv = marker.with_suffix('.csv')
        if meta.get('signature') != previous_signature or meta.get('csv') != csv.name or sha256(csv) != meta.get('sha256'):
            raise ValueError('Old batch signature/hash mismatch: '+str(marker))
        part = pd.read_csv(csv, float_precision='round_trip')
        if not part.file_id.is_unique or seen.intersection(part.file_id):
            raise ValueError('Duplicate source game in old cache')
        seen.update(part.file_id)
        for row in part.loc[~part.status.eq('ok')].itertuples():
            if (row.status != 'error' or row.game_id != FAILED_ID
                    or row.error != 'ExtractionError: disclosed exhaustive hand lacks structural wait'):
                raise ValueError('Unapproved failure, recovery stopped: '+str(row.file_id))
            failed.append(row.file_id)
        parts.append(part.loc[part.status.eq('ok')].copy())
    if seen != set(new_plan['members']) or len(failed) != 1:
        raise ValueError('Expected complete old target membership and exactly one known failed game')
    source_facts = dict(old_signature=previous_signature,new_signature=new_plan['signature'],
                        old_cache=str(old_cache),reused_successes=sum(map(len,parts)),retry_files=failed,
                        reason='Only the previously-failing disclosed-own-four-tiles validation branch changed',
                        batch_hashes={p.name:sha256(p) for p in sorted(old_cache.glob('batch_*.csv'))})
    marker = new_cache/'recovery_provenance.json'
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')) != source_facts:
            raise ValueError('Recovery inputs changed')
        return plan_extraction(new_plan['zip_path'],new_plan['player'],new_plan['key'],new_cache.parent,new_plan['legacy_dir'])
    if new_cache.exists() and any(new_cache.iterdir()):
        raise ValueError('New cache already contains output; preserve it and inspect before migration')
    staging = new_cache.with_name(new_cache.name+'_migration_'+uuid.uuid4().hex)
    staging.mkdir(parents=True, exist_ok=False)
    for part in parts:
        if part.empty:
            continue
        stem='batch_'+uuid.uuid4().hex
        write_csv(staging/(stem+'.csv'),part)
        write_json(staging/(stem+'.json'),dict(signature=new_plan['signature'],csv=stem+'.csv',sha256=sha256(staging/(stem+'.csv'))))
    write_json(staging/marker.name,source_facts)
    if new_cache.exists():
        new_cache.rmdir()  # Only the verified empty destination is removed.
    staging.rename(new_cache)
    return plan_extraction(new_plan['zip_path'],new_plan['player'],new_plan['key'],new_cache.parent,new_plan['legacy_dir'])
