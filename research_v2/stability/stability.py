"""Phase 1. Descriptive selection and split feasibility; no predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mahjong_research.analysis import temporal_splits, validate_data
from mahjong_research.inputs import audit_v2, new_run, write_csv, write_json, sha256
from study import FEATURES, DEFS, read_run, make_blocks, block_values, describe


def measured(values, den, rate):
    row = describe(values, den)
    good = np.isfinite(values)
    v, n = values[good], den[good]
    extreme = (v == 0) | (rate & (v == 1))
    row.update(zero_share=np.mean(v == 0) if len(v) else np.nan,
               one_share=np.mean(v == 1) if rate and len(v) else np.nan,
               mode_share=pd.Series(v).value_counts(normalize=True).max(),
               small_den_share=np.mean((den >= 1) & (den <= 2)) if len(den) else np.nan,
               extreme_n=int(extreme.sum()), small_den_extreme_n=int((extreme & (n <= 2)).sum()))
    return row


def group_rows(d, starts):
    raw, _, raw_label = block_values(d, starts[1], 1)
    groups, periods = [], []
    for width in (10, 20):
        x, den, labels = block_values(d, starts[width], width)
        offsets = np.asarray(starts[width], dtype=int)
        for label in (0, 1):
            ids = np.flatnonzero(labels == label)
            thirds = np.array_split(ids, 3)
            for j, f in enumerate(FEATURES):
                rate = DEFS[f]['upper'] == 1
                key = dict(width=width, label=label, feature=f, japanese=DEFS[f]['japanese'], rate=rate)
                row = measured(x[ids,j], den[ids,j], rate)
                reference = pd.Series(raw[raw_label == label,j]).dropna()
                meds, counts = [], []
                for name, piece in zip(('前','中','後'), thirds):
                    stats = describe(x[piece,j], den[piece,j])
                    periods.append(dict(key, period=name, **stats,
                        first_hour=d.hour.iloc[offsets[piece[0]]] if len(piece) else None,
                        last_hour=d.hour.iloc[offsets[piece[-1]]+width-1] if len(piece) else None))
                    meds.append(stats['median'])
                    counts.append(stats['valid_n'])
                groups.append(dict(key, **row, raw_sd=reference.std(),
                    raw_iqr=reference.quantile(.75)-reference.quantile(.25),
                    min_period_valid=min(counts), median_range=np.ptp(meds)))
    return pd.DataFrame(groups), pd.DataFrame(periods)


def classify(groups, min_den):
    rows = []
    for _, g in groups.groupby(['width','label']):
        reasons = {}
        for i, r in g.iterrows():
            issues = []
            if r.valid_n < 20: issues.append('有効20窓未満')
            if r.missing_rate > .2: issues.append('欠損率20%超')
            if r.opp_median < min_den: issues.append(f'機会数中央値{min_den}未満')
            if r.mode_share == 1: issues.append('観測上一定')
            if r.min_period_valid < 5: issues.append('一部期間で有効5窓未満')
            if r.rate and r.extreme_n >= 3 and r.small_den_extreme_n == r.extreme_n:
                issues.append('端点値3窓以上がすべて分母1～2')
            if not r.rate and not (r.raw_sd > 0 and r.raw_iqr > 0):
                issues.append('1対局の尺度が未定義またはゼロ')
            reasons[i] = issues
        pool = g.loc[[i for i in g.index if not reasons[i] and g.at[i,'rate']]]
        for i, r in g.iterrows():
            sd_limit = pool.sd.median() if r.rate else .5*r.raw_sd
            iqr_limit = pool.iqr.median() if r.rate else .5*r.raw_iqr
            issues = reasons[i]
            if issues:
                state = '判断困難'
            else:
                if r.sd > sd_limit: issues.append('SD基準超')
                if r.iqr > iqr_limit: issues.append('IQR基準超')
                if r.median_range > r.iqr: issues.append('期間中央値の範囲が群内IQR超')
                state = '不安定' if issues else '安定'
            rows.append(dict(r, den_threshold=min_den, sd_limit=sd_limit, iqr_limit=iqr_limit,
                             status=state, reason='・'.join(issues) or '全条件充足'))
    return pd.DataFrame(rows)


def decisions(groups):
    cols = ['width','feature','japanese','den_threshold']
    result = groups.pivot(index=cols, columns='label', values=['status','reason'])
    result.columns = [field+('_low' if label == 0 else '_high') for field,label in result.columns]
    a, b = result.status_low, result.status_high
    result['decision'] = np.select([a.eq('判断困難') | b.eq('判断困難'), a.eq('安定') & b.eq('安定'),
                                    a.eq('安定') | b.eq('安定')],
                                   ['判断困難','両段位期で安定','片方のみ安定'], default='不安定')
    return result.reset_index()


def memberships(d, starts, width):
    records, seen = [], set()
    for start in starts:
        block = d.iloc[start:start+width]
        ids = block.game_id.tolist()
        if (len(set(ids)) != width or seen.intersection(ids) or block.target_label.isna().any()
                or block.target_label.nunique() != 1
                or (start and d.hour.iloc[start-1] == block.hour.iloc[0])
                or (start+width < len(d) and d.hour.iloc[start+width] == block.hour.iloc[-1])):
            raise ValueError('窓幅・元対局非重複・時間帯・段位条件が不一致')
        seen.update(ids)
        records.append(dict(first=block.hour.iloc[0], last=block.hour.iloc[-1],
                            label=int(block.target_label.iloc[0]), source_ids=ids))
    return pd.DataFrame(records, columns=['first','last','label','source_ids'])


def split_audit(d, cutoff, starts):
    splits = temporal_splits(d, cutoff)
    rows, members = [], []
    for w in (10,20):
        windows = memberships(d, starts[w], w)
        for fold, past, future in splits:
            a, b = past.hour.max(), future.hour.max()
            train = windows[windows['last'].le(a)]
            test = windows[windows['first'].gt(a) & windows['last'].le(b)]
            cross = windows[(windows['first'].le(a) & windows['last'].gt(a)) |
                            (windows['first'].le(b) & windows['last'].gt(b))]
            ids = lambda block: {i for part in block.source_ids for i in part}
            overlap = len(ids(train) & ids(test))
            if overlap: raise ValueError('学習と評価に共通対局あり')
            n0,n1 = (int(train.label.eq(i).sum()) for i in (0,1))
            m0,m1 = (int(test.label.eq(i).sum()) for i in (0,1))
            ok = min(n0,n1,m0,m1) > 0
            issues = []
            if not min(n0,n1): issues.append('学習側に両群が揃わない')
            if not len(test): issues.append('評価窓なし')
            elif not min(m0,m1): issues.append('評価側が単一クラス')
            rows.append(dict(width=w, fold=fold, train_after=cutoff, train_through=a,
                eval_after=a, eval_through=b, train_low=n0, train_high=n1, eval_low=m0, eval_high=m1,
                dropped_crossing=len(cross), shared_ids=overlap, two_class_evaluation=ok,
                status='・'.join(issues) or 'AUC・Balanced Accuracy算出条件を満たす'))
            for role, frame in [('train',train),('evaluation',test),('excluded_boundary',cross)]:
                for r in frame.itertuples():
                    members.append(dict(width=w,fold=fold,role=role,first=r.first,last=r.last,
                                        label=r.label,source_ids=json.dumps(r.source_ids)))
    return pd.DataFrame(rows), pd.DataFrame(members)


def run(run_dir, root):
    d, shared, source = read_run(run_dir)
    d, cutoff = validate_data(d, FEATURES)
    out = new_run(root)
    audit_v2(d, FEATURES, out/'quality')
    starts = make_blocks(d, cutoff, shared)['available']
    groups, periods = group_rows(d, starts)
    full = pd.concat([classify(groups,t) for t in (3,5,10)], ignore_index=True)
    all_choices = decisions(full)
    main = all_choices[all_choices.den_threshold.eq(5)]
    folds, members = split_audit(d, cutoff, starts)
    for name, table in [('group_stability',full[full.den_threshold.eq(5)]),('period_statistics',periods),
                        ('feature_decisions',main),('threshold_sensitivity',all_choices),
                        ('primary_candidates',main[main.width.eq(10) & main.decision.eq('両段位期で安定')]),
                        ('fold_feasibility',folds),('fold_membership',members)]:
        write_csv(out/(name+'.csv'),table)
    write_json(out/'sources.json',dict(source, order_verified=False,full_history_confirmed=False,
                                     analysis_sha256=sha256(__file__),cutoff_hour=str(cutoff)))
    import matplotlib.pyplot as plt
    for w in (10,20):
        fig, axes = plt.subplots(5,4,figsize=(16,14),constrained_layout=True)
        for f,ax in zip(FEATURES,axes.flat):
            p = periods[periods.width.eq(w) & periods.feature.eq(f)]
            for label,color in [(0,'tab:blue'),(1,'tab:orange')]:
                sub = p[p.label.eq(label)]
                ax.plot(range(3),sub['median'],'o-',color=color,label=['Low','High'][label])
            ax.set_xticks(range(3),['Early','Middle','Late'])
            ax.set_title(f.replace('_',' '),fontsize=8)
        axes[0,0].legend()
        fig.suptitle(f'Within-rank period medians | {w} games per unit | feature-specific axes')
        fig.savefig(out/f'period_medians_{w}.png',dpi=130)
        plt.close(fig)
    lines = ['# 安定性と時系列評価の成立条件', '全期間での探索的な記述。モデル用の選定は将来、学習区間ごとにやり直す。',
             '率のSD・IQRは0〜1の元尺度。平均型は各段位群の1対局値を基準に判定した。']
    for w in (10,20):
        names = main.loc[main.width.eq(w) & main.decision.eq('両段位期で安定'),'japanese']
        f = folds[folds.width.eq(w)]
        lines += [f'## {w}対局', '両段位期で安定：'+('、'.join(names) or '該当なし'),
                  f'両群で学習しAUCを評価できる分割は{f.two_class_evaluation.sum()}/3。']
        for r in f.itertuples():
            lines.append(f'- Fold {r.fold}：学習 低{r.train_low}・高{r.train_high}、評価 低{r.eval_low}・高{r.eval_high}。{r.status}。')
    lines += ['成立条件はクラスの存在であり、少数窓での推定精度を保証しない。安定性に群間差やAUCは使用していない。',
              '同一時間帯内順序と履歴完全性は未確認。同一人物の段位と時間が交絡する。']
    (out/'report_ja.md').write_text('\n\n'.join(lines),encoding='utf-8')
    return out
