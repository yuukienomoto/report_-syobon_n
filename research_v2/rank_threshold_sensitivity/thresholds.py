"""Rank-label sensitivity with frozen time boundaries and the original model A."""
import json
from pathlib import Path
from importlib.metadata import version

import lightgbm as lgb
import numpy as np
import pandas as pd
from study import read_run, make_blocks, block_values
from stability import memberships, split_audit
from windows import FEATURES, require, same_table, external_data
from compare import fit_fixed, metrics
from relationships import contrast
from mahjong_research.analysis import MODEL_PLAN, validate_data
from mahjong_research.inputs import sha256, digest, write_csv, write_json, new_run, _final_frame

CONDITIONS = ('baseline', 'include_17_low', 'include_17_high')
METRICS = ('roc_auc', 'balanced_accuracy', 'accuracy', 'weak', 'strong', 'tn', 'fp', 'fn', 'tp')


def labels(data, condition):
    require(condition in CONDITIONS, '未知の分類条件')
    d = data.copy()
    d['target_label'] = np.select([d.dan.le(16), d.dan.ge(18)], [0., 1.], default=np.nan)
    if condition != 'baseline':
        d.loc[d.dan.eq(17), 'target_label'] = float(condition == 'include_17_high')
    return d


def build(data, cutoff, shared, condition, audit_splits=True):
    d = labels(data, condition)
    starts = make_blocks(d, cutoff, shared)['available']
    audit, members = split_audit(d, cutoff, starts) if audit_splits else (None, None)
    blocks = {}
    for width in (10, 20):
        meta = memberships(d, starts[width], width)
        # Preserve the exact source set as well as the legacy end-game ID.
        meta['window_id'] = meta.source_ids.map(lambda x: x[-1])
        meta['source_key'] = meta.source_ids.map(json.dumps)
        values, _, _ = block_values(d, starts[width], width)
        blocks[width] = pd.concat([meta, pd.DataFrame(values, columns=FEATURES)], axis=1)
    return blocks, audit, members


def check_counters(data):
    for f in FEATURES:
        require({f+'__numerator', f+'__denominator'} <= set(data), '機会数列不足 ' + f)
        n, k = data[f+'__numerator'], data[f+'__denominator']
        require(np.isfinite(n).all() and np.isfinite(k).all() and (n >= 0).all() and (k >= 0).all()
                and (n % 1 == 0).all() and (k % 1 == 0).all() and ((k > 0) | (n == 0)).all()
                and np.allclose(data[f], n / k.where(k > 0), equal_nan=True), '分子・分母の整合性不一致 ' + f)


def prepare(run_dir, phase1, phase3, ultra_cache=None):
    run_dir, phase1, phase3 = map(Path, (run_dir, phase1, phase3))
    cfg = json.loads((phase3/'protocol.json').read_text(encoding='utf8'))
    require(cfg['params'] == MODEL_PLAN, '第3段階のモデルパラメータが現在の定義と異なります')
    require(version('lightgbm') == cfg['versions']['lightgbm'], '第3段階とLightGBMの版を合わせてください')
    d, shared, source = read_run(run_dir)
    require(all(source[k] == cfg['source'][k] for k in ('raw_sha256', 'shared_sha256', 'signature')),
            '第3段階とC5キャッシュの対応が不一致')
    require(sha256(phase1/'fold_membership.csv') == cfg['inputs']['fold_membership.csv'], '第1段階の窓集合が不一致')
    d, cutoff = validate_data(d, FEATURES)
    check_counters(d)
    require(d.dan.eq(17).any(), 'C5に17段が0件。対象ZIPの抽出計画とキャッシュを確認してください。自動再抽出は行いません')
    blocks, audits, members = {}, {}, []
    for condition in CONDITIONS:
        blocks[condition], audits[condition], member = build(d, cutoff, shared, condition)
        members.append(member.assign(condition=condition, dataset='internal'))
    old = pd.read_csv(phase1/'fold_membership.csv', parse_dates=['first', 'last'])
    same_table(members[0], old, ['width', 'fold', 'role', 'source_ids'], ['first', 'last', 'label'])
    old_audit = pd.read_csv(phase3/'fold_audit.csv')
    times = ['train_after', 'train_through', 'eval_after', 'eval_through']
    for c in times:
        old_audit[c] = pd.to_datetime(old_audit[c])
    same_table(audits['baseline'], old_audit, ['width', 'fold'], times + ['train_low', 'train_high', 'eval_low', 'eval_high', 'dropped_crossing', 'shared_ids'])
    for condition in CONDITIONS[1:]:
        same_table(audits[condition], old_audit, ['width', 'fold'], times)
    external, external_raw, external_source = {}, None, {}
    external_note = ''
    try:
        override = ultra_cache or cfg.get('ultra_cache')
        reference, shared_n = external_data(run_dir, d, override)
        plan = json.loads((run_dir/'input_plan.json').read_text(encoding='utf8'))['ultra']
        folder = Path(override or plan['cache_dir'])
        raw_path = folder/'raw_features_v2.csv'
        raw = _final_frame(pd.read_csv(raw_path), 'ウルトラ立直')
        external_raw, extcut = validate_data(raw, FEATURES)
        check_counters(external_raw)
        shared_ext = set(d.game_id) & set(external_raw.game_id)
        require(len(shared_ext) == shared_n, '共通対局数が不一致')
        for condition in CONDITIONS:
            external[condition], _, _ = build(external_raw, extcut, shared_ext, condition, audit_splits=False)
        for w in (10, 20):
            same_table(external['baseline'][w], reference[w], ['window_id'], ['first', 'last', 'label'] + list(FEATURES))
        external_source = dict(raw_sha256=sha256(raw_path), shared_games=shared_n, cutoff=str(extcut), cache=str(folder))
        external_note = f'共通{shared_n}対局を含む窓を除外。各条件のFold 3固定モデルを適用。'
        if not external_raw.dan.eq(17).any():
            external_note += '外部の提供済みキャッシュでは17段が0件。追加対象がないことを記録する。'
    except (OSError, ValueError, KeyError, TypeError) as exc:
        external = {}
        external_note = '外部評価保留。' + str(exc)
    return dict(data=d, cutoff=cutoff, shared=shared, blocks=blocks, audits=audits,
                members=pd.concat(members, ignore_index=True), source=source, config=cfg,
                external=external, external_raw=external_raw, external_source=external_source,
                external_note=external_note)


def saved_model(phase3, config, width, fold):
    path = Path(phase3)/'models'/f'w{width}_f{fold}_A.txt'
    marker = json.loads(path.with_suffix('.json').read_text(encoding='utf8'))
    cols = list(FEATURES)
    require(marker['signature'] == digest([config, width, fold, cols]) and marker['features'] == cols
            and marker['sha256'] == sha256(path) and marker['params'] == MODEL_PLAN,
            'baselineの保存モデル署名が不一致')
    model = lgb.Booster(model_str=path.read_text(encoding='utf8'))
    require(model.feature_name() == cols, 'baselineの特徴量列が不一致')
    return model


def verify_baseline(test, probability, key, old_predictions, old_metrics):
    filt = lambda t: t[(t.dataset == key['dataset']) & (t.width == key['width']) & (t.fold == key['fold']) & t.model.eq('A')]
    saved, reference = filt(old_predictions), filt(old_metrics)
    require(len(reference) == 1 and reference.iloc[0].status == '評価済', 'baselineの評価結果が不足')
    same_table(test, saved, ['window_id'], ['label'])
    expected = saved.set_index('window_id').loc[test.window_id, 'prob_high'].to_numpy()
    require(np.allclose(probability, expected, rtol=1e-8, atol=1e-10), 'baselineの予測確率が従来結果と不一致')
    measured = metrics(test.label.to_numpy(), probability)
    for field in METRICS:
        require(np.isclose(measured[field], reference.iloc[0][field], equal_nan=True), 'baselineの指標不一致 ' + field)


def common_keys(blocks, width):
    return set.intersection(*(set(blocks[c][width].source_key) for c in CONDITIONS))


def evaluate(model, train, test, key, rows, predictions, common, phase3_results=None):
    counts = dict(weak=int(test.label.eq(0).sum()), strong=int(test.label.eq(1).sum()))
    if model is None or test.empty:
        reason = '学習期間が空または単一クラス' if model is None else '評価窓なし'
        rows.append(dict(key, scope='all', status=reason, auc_reason=reason, roc_auc=np.nan,
                         balanced_accuracy=np.nan, accuracy=np.nan, **counts))
        return
    p = model.predict(test[list(FEATURES)], num_threads=1)
    if phase3_results is not None:
        verify_baseline(test, p, key, *phase3_results)
    for scope, mask in [('all', np.ones(len(test), dtype=bool)), ('common_windows', test.source_key.isin(common).to_numpy())]:
        subset = test[mask]
        if not len(subset):
            rows.append(dict(key, scope=scope, status='共通評価窓なし', auc_reason='共通評価窓なし',
                             roc_auc=np.nan, balanced_accuracy=np.nan, accuracy=np.nan, weak=0, strong=0))
            continue
        rows.append(dict(key, scope=scope, status='評価済', **metrics(subset.label.to_numpy(), p[mask])))
    for r, prob in zip(test.itertuples(), p):
        predictions.append(dict(key, window_id=r.window_id, source_ids=r.source_key, label=r.label,
                                prob_high=prob, prediction=int(prob >= .5), common_window=r.source_key in common))


def population(data, block, key, role):
    used = [gid for group in block.source_ids for gid in group]
    ranks = data.set_index('game_id').loc[used, 'dan']
    return dict(key, role=role, low=int(block.label.eq(0).sum()), high=int(block.label.eq(1).sum()),
                windows=len(block), games=len(used), rank17_games=int(ranks.eq(17).sum()),
                rank_counts=json.dumps({int(k): int(v) for k, v in ranks.value_counts().sort_index().items()}),
                first=block['first'].min(), last=block['last'].max())


def run(run_dir, phase1, phase3, root, resume=None, ultra_cache=None):
    info = prepare(run_dir, phase1, phase3, ultra_cache)
    phase3 = Path(phase3)
    old_predictions = pd.read_csv(phase3/'predictions.csv')
    old_metrics = pd.read_csv(phase3/'model_metrics.csv')
    config = dict(source=info['source'], external=info['external_source'], params=MODEL_PLAN,
                  conditions=list(CONDITIONS), boundary_policy='same baseline calendar boundaries',
                  protocol_sha256=sha256(phase3/'protocol.json'),
                  predictions_sha256=sha256(phase3/'predictions.csv'), metrics_sha256=sha256(phase3/'model_metrics.csv'),
                  versions={n: version(n) for n in ('lightgbm', 'numpy', 'pandas', 'scikit-learn')},
                  code_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
                  order_verified=False, full_history_confirmed=False)
    if resume:
        out = Path(resume)
        require(json.loads((out/'protocol.json').read_text(encoding='utf8')) == config, '再開時の入力・設定が変化しています')
    else:
        out = new_run(root)
        write_json(out/'protocol.json', config)
    print('保存・再開先', out, flush=True)
    (out/'models').mkdir(exist_ok=True)
    write_csv(out/'fold_audit.csv', pd.concat([v.assign(condition=k) for k, v in info['audits'].items()]))
    write_csv(out/'window_membership.csv', info['members'])
    rows, predictions, groups, differences, reproduction, rank_rows = [], [], [], [], [], []
    frozen = {}
    for name, data in [('shobon', info['data']), ('ultra', info['external_raw'])]:
        if data is not None:
            for rank, n in data.dan.value_counts().sort_index().items():
                rank_rows.append(dict(player=name, dan=rank, cached_games=n))
    write_csv(out/'cached_rank_counts.csv', pd.DataFrame(rank_rows))
    for condition in CONDITIONS:
        for r in info['audits'][condition].itertuples():
            width, fold = r.width, r.fold
            block = info['blocks'][condition][width]
            train = block[block['last'].le(r.train_through)]
            test = block[block['first'].gt(r.train_through) & block['last'].le(r.eval_through)]
            key = dict(condition=condition, dataset='internal', width=width, fold=fold)
            for role, part in [('train', train), ('evaluation', test)]:
                groups.append(population(info['data'], part, key, role))
            differences.extend(contrast(train, test, f, key) for f in FEATURES)
            model = None
            if train.label.nunique() == 2:
                if condition == 'baseline':
                    model = saved_model(phase3, info['config'], width, fold)
                else:
                    model = fit_fixed(train, list(FEATURES), out/'models', f'{condition}_w{width}_f{fold}',
                                      digest([config, condition, width, fold]))
            print(f'{condition} {width}対局 Fold {fold} 学習{len(train)} 評価{len(test)}', flush=True)
            evaluate(model, train, test, key, rows, predictions, common_keys(info['blocks'], width),
                     (old_predictions, old_metrics) if condition == 'baseline' else None)
            if condition == 'baseline':
                if model is None or test.empty:
                    old = old_metrics.query('dataset=="internal" and model=="A" and width==@width and fold==@fold')
                    require(len(old) == 1 and old.iloc[0].status != '評価済', 'baseline学習・評価可否が従来と不一致')
                reproduction.append(dict(key, status='従来の窓・予測・指標と一致' if model is not None and len(test) else '従来の学習・評価不能状態と一致'))
            if fold == 3:
                frozen[condition, width] = (model, train)
    for condition in CONDITIONS:
        for width in (10, 20):
            key = dict(condition=condition, dataset='external', width=width, fold=3)
            if not info['external']:
                rows.append(dict(key, scope='all', status=info['external_note'], auc_reason=info['external_note'], roc_auc=np.nan))
                continue
            model, train = frozen[condition, width]
            test = info['external'][condition][width]
            groups.append(population(info['external_raw'], test, key, 'evaluation'))
            differences.extend(contrast(train, test, f, key) for f in FEATURES)
            evaluate(model, train, test, key, rows, predictions, common_keys(info['external'], width),
                     (old_predictions, old_metrics) if condition == 'baseline' else None)
    results = pd.DataFrame(rows)
    for name, frame in [('metrics_comparison', results), ('external_evaluation', results[results.dataset.eq('external')]),
                        ('class_distribution', pd.DataFrame(groups)), ('feature_difference_comparison', pd.DataFrame(differences)),
                        ('predictions', pd.DataFrame(predictions)), ('baseline_reproduction', pd.DataFrame(reproduction))]:
        write_csv(out/(name+'.csv'), frame)
    from threshold_report import deliver
    deliver(out, results, pd.DataFrame(groups), pd.DataFrame(differences), info['external_note'])
    return out
