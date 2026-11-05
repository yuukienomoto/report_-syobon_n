"""Within-player descriptive study on signed per-game counters, without fitting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mahjong_research import tenhou
from mahjong_research.analysis import validate_data
from mahjong_research.inputs import _final_frame, audit_v2, new_run, sha256, write_csv, write_json

FEATURES = tenhou.FEATURE_NAMES
DEFS = {d['feature']: d for d in tenhou.FEATURE_DEFINITIONS}
WIDTHS = (1, 10, 20)
SCOPES = ('available', 'common_games')


def read_run(run):
    run = Path(run)
    plan = json.loads((run/'input_plan.json').read_text(encoding='utf-8'))['shobon']
    folder = Path(plan['cache_dir'])
    marker = json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    raw, audit = folder/'raw_features_v2.csv', folder/'extraction_audit.csv'
    if (marker['signature'] != plan['signature'] or marker['extractor_sha256'] != sha256(tenhou.__file__)
            or marker['order_verified'] is not False or marker['full_history_confirmed'] is not False
            or marker['sha256'] != sha256(raw) or marker['audit_sha256'] != sha256(audit)):
        raise ValueError('C5キャッシュの署名・ハッシュ・監査フラグが不一致です。実行フォルダを確認してください。')
    frame = pd.read_csv(raw, float_precision='round_trip')
    required = {f+s for f in FEATURES for s in ('', '__numerator', '__denominator')}
    required.update(['file_id', 'game_id', 'hour', 'dan', 'player'])
    if missing := required.difference(frame):
        raise ValueError(f'C5キャッシュの必要列が不足しています: {sorted(missing)}')
    data = _final_frame(frame, 'しょぼんｎ')
    log = pd.read_csv(audit, usecols=['file_id', 'status'])
    if (len(data) != marker['rows'] or not log.status.eq('ok').all() or not log.file_id.is_unique
            or set(data.file_id) != set(log.file_id) or set(data.file_id) != set(plan['members'])):
        raise ValueError('C5完了監査・抽出計画・対局集合が不一致です。')
    shared_path = run/'shared_games.csv'
    shared = pd.read_csv(shared_path, usecols=['game_id']).game_id
    if shared.isna().any() or not shared.is_unique or not set(shared) <= set(data.game_id):
        raise ValueError('共通対局一覧が不正です。')
    source = dict(run=str(run), cache=str(folder), raw_sha256=marker['sha256'],
                  shared_sha256=sha256(shared_path), extractor_sha256=marker['extractor_sha256'],
                  signature=marker['signature'], rows=len(data))
    return data, set(shared), source


def make_blocks(d, cutoff, shared):
    """Slices into sorted data. An ID tie-break is never used to split an hour."""
    hours = d.hour.to_numpy()
    groups = d.target_label.fillna(-1).to_numpy().astype(int)
    excluded = (d.hour.le(cutoff) | d.game_id.isin(shared) | d.target_label.isna()).to_numpy()
    eligible = np.where(excluded, -1, groups)
    bounds = np.r_[0, np.flatnonzero(hours[1:] != hours[:-1])+1, len(d)]
    boundaries = set(bounds)
    starts = {1: np.flatnonzero(eligible >= 0).tolist()}
    for width in (10, 20):
        selected = []
        end = 0
        for left in bounds[:-1]:
            right = left + width
            if left < end or right not in boundaries or eligible[left] < 0:
                continue
            if np.all(eligible[left:right] == eligible[left]):
                selected.append(int(left))
                end = right
        starts[width] = selected
    nested = [s for s in starts[20] if s+10 in boundaries]
    common = {w: [i for s in nested for i in range(s, s+20, w)] for w in WIDTHS}
    return dict(available=starts, common_games=common)


def block_values(d, starts, width):
    # Additive event/quantity totals for rates and both event-weighted means.
    idx = np.array(starts, dtype=int)[:, None] + np.arange(width)
    def totals(suffix):
        return d[[f+suffix for f in FEATURES]].to_numpy(dtype='int64')[idx].sum(axis=1)
    den = totals('__denominator')
    x = np.divide(totals('__numerator'), den, out=np.full(den.shape, np.nan), where=den > 0)
    return x, den, d.target_label.to_numpy()[idx[:, 0]].astype(int)


def describe(values, opportunities):
    x = pd.Series(values).dropna()
    opp = pd.Series(opportunities, dtype=float)
    return dict(n=len(opp), valid_n=len(x), missing_rate=1-len(x)/len(opp) if len(opp) else np.nan,
                mean=x.mean(), median=x.median(), sd=x.std(), variance=x.var(),
                iqr=x.quantile(.75)-x.quantile(.25), minimum=x.min(), maximum=x.max(),
                opp_min=opp.min(), opp_q25=opp.quantile(.25), opp_median=opp.median(),
                opp_q75=opp.quantile(.75), opp_max=opp.max(), opp_total=opp.sum())


def contrasts(low, high, a, b, edges):
    delta = b['mean'] - a['mean']
    sd = np.nan
    if min(a['valid_n'], b['valid_n']) >= 2:
        n, m = a['valid_n']-1, b['valid_n']-1
        sd = np.sqrt((n*a['variance']+m*b['variance'])/(n+m))
    auc = overlap = np.nan
    if len(low) and len(high):
        auc = roc_auc_score(np.repeat([0, 1], [len(low), len(high)]), np.r_[low, high])
        p, q = (np.histogram(v, bins=edges)[0]/len(v) for v in (low, high))
        overlap = np.minimum(p, q).sum()
    return dict(diff=delta, pooled_sd=sd, smd=delta/sd if sd > 0 else np.nan,
                overlap=overlap, auc=auc, auc_reversed=1-auc)


def analyze(d, blocks):
    arrays = {(scope, w): block_values(d, blocks[scope][w], w) for scope in SCOPES for w in WIDTHS}
    stats, comparisons, bin_rows = [], [], []
    for j, f in enumerate(FEATURES):
        vals = np.concatenate([x[:, j] for x, _, _ in arrays.values()])
        finite = vals[np.isfinite(vals)]
        # Common edges across widths, groups and sensitivity populations.
        edges = np.histogram_bin_edges(finite, bins=20) if finite.size else np.array([0., 1.])
        bin_rows.append(dict(feature=f, edges=json.dumps(edges.tolist())))
        for (scope, w), (x, den, y) in arrays.items():
            key = dict(scope=scope, width=w, feature=f)
            sub, groups = [], []
            row = dict(key)
            for label, prefix in enumerate(('low', 'high')):
                values = x[y == label, j]
                sub.append(values[np.isfinite(values)])
                s = describe(values, den[y == label, j])
                stats.append(dict(key, label=label, **s))
                groups.append(s)
                for field in ('n', 'valid_n', 'sd', 'missing_rate', 'opp_median'):
                    row[prefix+'_'+field] = s[field]
            comparisons.append(dict(row, **contrasts(*sub, *groups, edges)))
    comp = pd.DataFrame(comparisons)
    for field in ('low_sd','high_sd','diff','smd','overlap','low_valid_n','high_valid_n',
                  'low_missing_rate','high_missing_rate','auc','auc_reversed'):
        ref = comp.groupby(['scope','feature'])[field].transform(lambda v: v.iloc[0])
        comp[field+'_change_from_1'] = comp[field]-ref
    return pd.DataFrame(stats), comp, pd.DataFrame(bin_rows), arrays


def run_study(run, output_root):
    from presentation import deliver
    data, shared, source = read_run(run)
    d, cutoff = validate_data(data, FEATURES)
    out = new_run(output_root)
    audit_v2(d, FEATURES, out/'quality')
    blocks = make_blocks(d, cutoff, shared)
    stats, comp, bins, arrays = analyze(d, blocks)
    ids = d.game_id.to_numpy()
    population = [dict(scope=scope, width=w, unit=i, label=int(d.target_label.iloc[left]),
                       first_hour=str(d.hour.iloc[left]), last_hour=str(d.hour.iloc[left+w-1]),
                       source_ids=json.dumps(ids[left:left+w].tolist()))
                  for scope, widths in blocks.items() for w, starts in widths.items()
                  for i, left in enumerate(starts)]
    write_csv(out/'window_membership.csv', pd.DataFrame(population, columns=[
        'scope','width','unit','label','first_hour','last_hour','source_ids']))
    write_csv(out/'histogram_edges.csv', bins)
    write_csv(out/'group_statistics.csv', stats)
    write_csv(out/'comparison_20x3.csv', comp[comp.scope.eq('available')])
    write_csv(out/'common_games_20x3.csv', comp[comp.scope.eq('common_games')])
    write_json(out/'sources.json', dict(source, cutoff_hour=str(cutoff), shared_games=len(shared),
                                        order_verified=False, full_history_confirmed=False,
                                        code_sha256={p.name: sha256(p) for p in
                                            Path(__file__).parent.glob('*.py')}))
    deliver(out, comp, arrays)
    return out
