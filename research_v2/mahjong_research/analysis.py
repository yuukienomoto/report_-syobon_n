"""Descriptive-first, hour-invariant analysis. No ordering or completeness claim.

Inputs are raw per-game rows, including rank 17 and the first 1500 observed
games. Never remove failed/shared games and then repack a ten-game history.
Descriptive means use ten observed values per feature (no imputation); model
means use training-only imputation before optional legacy cross construction.
Reference definitions: sklearn balanced accuracy = mean class recall;
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
LightGBM deterministic CPU options:
https://lightgbm.readthedocs.io/en/stable/Parameters.html
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WINDOW = 10
INITIAL_EXCLUDE = 1500
MODEL_PLAN = dict(objective="binary", class_weight="balanced", random_state=42,
                  n_estimators=200, learning_rate=0.03, max_depth=5,
                  num_leaves=31, subsample=0.8, subsample_freq=0,
                  colsample_bytree=0.8, verbosity=-1, n_jobs=1,
                  deterministic=True, force_col_wise=True)
METRIC_COLUMNS = ["dataset", "estimator", "n", "weak", "strong", "strong_ratio",
                  "accuracy", "balanced_accuracy", "roc_auc", "f1_strong", "f1_macro",
                  "recall_weak", "recall_strong", "tn", "fp", "fn", "tp"]
# subsample_freq=0 is the old estimator default: subsample=0.8 is inactive.
# Preserving that behavior makes the comparison explicit rather than silently
# activating bagging. There is no tuning, early stopping or threshold search.
LIMITATIONS = [
    "order_verified=False：同一時間帯内の厳密な対局順序は未確認。",
    "full_history_confirmed=False：提供された対局集合の完全性は未確認。",
    "hour_invariantは観測集合で窓の構成が順序不変な場合だけを採用し、厳密な全履歴時系列検証とは同等でない。",
    "最初の1500観測対局と同じ時間帯までを採点外とする。実際の生涯初期1500対局とは断定しない。",
    "段位17を履歴に含め、窓終端の段位16以下／18以上で分類する。窓内には異なる段位が混在し得る。",
    "重複窓に基づく標準化平均差は記述量であり、独立標本を仮定した検定・信頼区間は算出しない。",
    "同一人物の段位別比較であり、麻雀の強者と弱者全体の差や実力向上の因果効果を示さない。",
    "段位と暦時間、対戦相手、卓条件、ルールなどが交絡し得る。データのない条件は調整できない。",
    "欠損なしの10対局を要求する記述窓では特徴量ごとに有効標本が異なる。発生機会の少ない行動の選択性に注意する。",
    "hour_invariant採用条件はプレイ頻度に依存するため、除外された時間帯へ一般化できるとは限らない。",
    "LightGBMは補助分析。外部評価を見た特徴量・パラメータ・閾値の調整は行わない。",
    "class_weight=balancedの出力スコアを、較正済みの実世界の強者確率とは解釈しない。",
]


class AnalysisBlocked(ValueError):
    """A scientific or input precondition was not satisfied."""


def _require(condition, message):
    if not condition:
        raise AnalysisBlocked(message)


def _json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                   default=str, allow_nan=False), encoding="utf-8")


def _transform_fingerprint(cross_builder):
    if cross_builder is None:
        return None
    try:
        source = inspect.getsource(cross_builder)
        module_file = inspect.getsourcefile(cross_builder)
        _require(module_file is not None and Path(module_file).is_file(),
                 "交差関数を保存済みPythonモジュールから読み込んでください")
    except (OSError, TypeError) as exc:
        raise AnalysisBlocked("交差関数のソースを固定できません。Notebook内の一時lambdaは使用不可") from exc
    return dict(function_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
                module_sha256=hashlib.sha256(Path(module_file).read_bytes()).hexdigest())


def _module_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _csv(path, frame):
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _new_dir(path):
    path = Path(path)
    _require(not path.exists() or not any(path.iterdir()),
             f"既存出力を上書きしません。新しい保存先を指定: {path}")
    path.mkdir(parents=True, exist_ok=True)
    _csv(path / "limitations.csv", pd.DataFrame({"limitation": LIMITATIONS}))
    return path


def validate_data(data, features, initial_exclude=INITIAL_EXCLUDE):
    """Never silently coerce malformed values, drop rows, or infer game order."""
    required = {"game_id", "hour", "dan", "player", *features}
    _require(required <= set(data), f"必須列不足: {sorted(required - set(data))}")
    _require(len(features) == len(set(features)) and len(features) > 0,
             "特徴量一覧は重複のない非空リストが必要")
    d = data.copy()
    _require(d.game_id.notna().all() and d.game_id.is_unique, "欠損または重複対局ID")
    _require(d.player.notna().all() and d.player.nunique() == 1, "単一プレイヤーのみを入力してください")
    d["hour"] = pd.to_datetime(d.hour, errors="coerce")
    _require(d.hour.notna().all(), "対局時間帯が解釈できません")
    _require((d.hour == d.hour.dt.floor("h")).all(), "hour列は時間帯の先頭を指定してください")
    dan = pd.to_numeric(d.dan, errors="coerce")
    _require(dan.notna().all() and dan.between(0, 20).all() and dan.mod(1).eq(0).all(), "不正段位")
    d["dan"] = dan.astype(int)
    values = d[features].apply(pd.to_numeric, errors="coerce")
    _require(not (d[features].notna() & values.isna()).any().any(), "非数値特徴量")
    _require(not np.isinf(values.to_numpy(dtype=float)).any(), "無限大特徴量。自動補正しません")
    d[features] = values
    d["target_label"] = np.select([d.dan <= 16, d.dan >= 18], [0., 1.], default=np.nan)
    d = d.sort_values(["hour", "game_id"]).reset_index(drop=True)
    _require(isinstance(initial_exclude, int) and initial_exclude >= 0, "初期除外数が不正")
    _require(len(d) > initial_exclude and len(d) >= WINDOW, "初期除外後または10対局履歴が不足")
    cutoff = d.iloc[initial_exclude - 1].hour if initial_exclude else d.hour.min() - pd.Timedelta(hours=1)
    return d, cutoff


def prepare_windows(rows, features, cutoff, forbidden=frozenset(), means=None,
                    base_features=None, cross_builder=None):
    """Ten actual observed rows; complete hour buckets and unique endpoint hour.

    means=None leaves missing values missing. A rolling mean is defined only if
    all ten values of that feature exist. A common window population is retained
    and valid_n is reported separately; missing features do not delete games.
    """
    base = list(base_features or features)
    d = rows.sort_values(["hour", "game_id"]).reset_index(drop=True)
    _require(d.game_id.is_unique, "窓入力に重複ID")
    values = d[base].copy()
    if means is not None:
        _require(set(base) == set(means.index) and means.notna().all(), "学習補完値が不足")
        values = values.fillna(means)
    if cross_builder is not None:
        values = cross_builder(values.copy())
    _require(set(features) <= set(values), "特徴量変換後の列不足")
    _require(len(values) == len(d) and values.index.equals(d.index), "特徴量変換が対局行数・対応を変更")
    values = values[features]
    _require(not np.isinf(values.to_numpy(dtype=float)).any(), "変換後に無限大")
    if means is not None:
        _require(values.notna().all().all(), "補完後に欠損。変換または訓練補完値を点検")
    rolled = values.rolling(WINDOW, min_periods=WINDOW).mean()
    sizes = d.groupby("hour").size()
    ids, hours = d.game_id.tolist(), d.hour.tolist()
    forbidden = set(forbidden)
    keep, records, exclusions = [], [], []
    for i, row in d.iterrows():
        start = i - WINDOW + 1
        source = tuple(ids[max(start, 0):i+1])
        reason = ""
        if start < 0:
            reason = "insufficient_block_history"
        elif row.hour <= cutoff:
            reason = "initial_observed_period"
        elif pd.isna(row.target_label):
            reason = "rank17_history_only"
        elif forbidden.intersection(source):
            reason = "shared_game_in_window"
        elif sizes.loc[row.hour] != 1:
            reason = "endpoint_hour_order_unknown"
        elif start > 0 and hours[start - 1] == hours[start]:
            reason = "start_cuts_hour_bucket"
        if reason:
            exclusions.append(dict(game_id=row.game_id, reason=reason))
            continue
        keep.append(i)
        source_labels = d.iloc[start:i+1].target_label
        records.append(dict(game_id=row.game_id, hour=row.hour, dan=row.dan,
                            target_label=int(row.target_label), source_ids=source,
                            source_weak=int(source_labels.eq(0).sum()),
                            source_strong=int(source_labels.eq(1).sum()),
                            source_rank17=int(source_labels.isna().sum())))
    columns = ["game_id", "hour", "dan", "target_label", "source_ids",
               "source_weak", "source_strong", "source_rank17"]
    meta = pd.DataFrame(records, columns=columns)
    return dict(X=rolled.iloc[keep].reset_index(drop=True),
                y=meta.target_label.astype(int), meta=meta,
                excluded=pd.DataFrame(exclusions, columns=["game_id", "reason"]))


def source_ids(block):
    return {game_id for ids in block["meta"].source_ids for game_id in ids}


def disjoint_windows(block):
    selected, used = [], set()
    for i, ids in enumerate(block["meta"].source_ids):
        if not used.intersection(ids):
            selected.append(i)
            used.update(ids)
    return {key: value.iloc[selected].reset_index(drop=True)
            if key in ("X", "y", "meta") else value.copy()
            for key, value in block.items()}


def summarize(X, y, view):
    stats, diffs = [], []
    for feature in X:
        groups = []
        for label in (0, 1):
            v = X.loc[np.asarray(y) == label, feature]
            n, valid = len(v), v.notna().sum()
            row = dict(view=view, feature=feature, target_label=label,
                       n=n, valid_n=int(valid), mean=v.mean(), median=v.median(),
                       std=v.std(ddof=1), missing_rate=1-valid/n if n else np.nan)
            stats.append(row)
            groups.append(row)
        low, high = groups
        diff = high["mean"] - low["mean"]
        pooled = np.nan
        if min(low["valid_n"], high["valid_n"]) >= 2:
            pooled = np.sqrt(((low["valid_n"]-1)*low["std"]**2 +
                              (high["valid_n"]-1)*high["std"]**2) /
                             (low["valid_n"]+high["valid_n"]-2))
        smd = diff / pooled if pd.notna(pooled) and pooled > 0 else np.nan
        diffs.append(dict(view=view, feature=feature, mean_diff_high_minus_low=diff,
                          pooled_sd=pooled, smd=smd, abs_smd=abs(smd),
                          low_valid_n=low["valid_n"], high_valid_n=high["valid_n"]))
    return pd.DataFrame(stats), pd.DataFrame(diffs)


def _save_block(block, stem, out):
    meta = block["meta"].copy()
    meta["source_ids"] = meta.source_ids.map(lambda x: json.dumps(x, ensure_ascii=False))
    _csv(out / (stem + "_windows.csv"), meta)
    _csv(out / (stem + "_excluded.csv"), block["excluded"])


def _plot_descriptive(views, differences, monthly, features, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y = views["primary_hour_invariant"]
    for page, start in enumerate(range(0, len(features), 6), 1):
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        for ax, feature in zip(axes.flat, features[start:start+6]):
            for label, name in ((0, "Low rank"), (1, "High rank")):
                vals = np.sort(x.loc[np.asarray(y) == label, feature].dropna().to_numpy())
                if len(vals):
                    ax.step(vals, np.arange(1, len(vals)+1)/len(vals), where="post", label=f"{name} n={len(vals)}")
            ax.set_title(feature, fontsize=8)
            ax.set_ylabel("ECDF")
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7)
        for ax in list(axes.flat)[len(features[start:start+6]):]:
            ax.set_visible(False)
        fig.suptitle("Observed hour-invariant 10-game means; no imputation")
        fig.tight_layout()
        fig.savefig(out / f"distribution_{page:02d}.png", dpi=160)
        plt.close(fig)
    main = differences[differences.view.eq("primary_hour_invariant")].dropna(subset=["smd"])
    top = main.sort_values("abs_smd", ascending=False).head(10).sort_values("smd")
    if not top.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(top.feature, top.smd)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_xlabel("Descriptive SMD: high minus low (dependent windows)")
        fig.tight_layout()
        fig.savefig(out / "smd_top10.png", dpi=160)
        plt.close(fig)
    # All prespecified features get calendar plots; no external outcome guides selection.
    for page, start in enumerate(range(0, len(features), 6), 1):
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        for ax, feature in zip(axes.flat, features[start:start+6]):
            sub = monthly[monthly.feature.eq(feature)]
            for label, name in ((0, "Low rank"), (1, "High rank")):
                a = sub[sub.target_label.eq(label)]
                # Points only: avoids interpolating across missing calendar months.
                ax.scatter(pd.to_datetime(a.month), a["mean"], s=12, label=name)
            ax.set_title(feature, fontsize=8)
            ax.set_ylabel("Monthly per-game mean")
            ax.legend(fontsize=7)
            ax.tick_params(axis="x", rotation=30)
        for ax in list(axes.flat)[len(features[start:start+6]):]:
            ax.set_visible(False)
        fig.suptitle("Calendar summaries of observed games; gaps are not proof of missing logs")
        fig.tight_layout()
        fig.savefig(out / f"calendar_trends_{page:02d}.png", dpi=160)
        plt.close(fig)


def run_descriptive(data, features, out_dir, forbidden=frozenset(),
                    initial_exclude=INITIAL_EXCLUDE, base_features=None,
                    cross_builder=None, make_plots=True):
    """Write descriptive tables/figures without model fitting or imputation."""
    base = list(base_features or features)
    d, cutoff = validate_data(data, base, initial_exclude)
    out = _new_dir(out_dir)
    raw = cross_builder(d[base].copy()) if cross_builder else d[base].copy()
    _require(set(features) <= set(raw), "記述分析の特徴量列不足")
    raw = raw[features]
    eligible = (d.hour > cutoff) & d.target_label.notna() & ~d.game_id.isin(forbidden)
    block = prepare_windows(d, features, cutoff, forbidden, base_features=base,
                            cross_builder=cross_builder)
    _require(len(block["X"]) > 0, "主分析のhour_invariant窓が0件")
    disjoint = disjoint_windows(block)
    endpoints = set(block["meta"].game_id)
    same = d.game_id.isin(endpoints)
    views = {"raw_all_eligible": (raw.loc[eligible].reset_index(drop=True), d.loc[eligible, "target_label"].reset_index(drop=True)),
             "raw_same_endpoints": (raw.loc[same].reset_index(drop=True), d.loc[same, "target_label"].reset_index(drop=True)),
             "primary_hour_invariant": (block["X"], block["y"]),
             "primary_disjoint_windows": (disjoint["X"], disjoint["y"])}
    stats, differences = [], []
    for name, (x, y) in views.items():
        s, diff = summarize(x, y, name)
        stats.append(s)
        differences.append(diff)
    stats, differences = pd.concat(stats, ignore_index=True), pd.concat(differences, ignore_index=True)
    _csv(out / "group_statistics.csv", stats)
    _csv(out / "mean_differences_smd.csv", differences)
    _csv(out / "smd_ranking.csv", differences.sort_values(["view", "abs_smd"], ascending=[True, False]))
    robust = differences.pivot(index="feature", columns="view", values="smd").reset_index()
    for view in ("raw_all_eligible", "raw_same_endpoints", "primary_disjoint_windows"):
        a, b = robust.primary_hour_invariant, robust[view]
        robust[view + "_direction_agrees"] = np.where(a.notna() & b.notna(), np.sign(a) == np.sign(b), np.nan)
    _csv(out / "direction_robustness.csv", robust)
    reuse = Counter(i for ids in block["meta"].source_ids for i in ids)
    _csv(out / "source_reuse.csv", pd.DataFrame(sorted(reuse.items()), columns=["game_id", "windows_using_game"]))
    _save_block(block, "primary", out)
    _save_block(disjoint, "disjoint", out)
    _csv(out / "class_by_calendar_year.csv", d.loc[eligible].groupby([d.loc[eligible, "hour"].dt.year.rename("year"), "target_label"]).size().rename("n").reset_index())
    rows = []
    tmp = raw.copy()
    tmp["month"] = d.hour.dt.to_period("M").astype(str)
    tmp["target_label"] = d.target_label
    for (month, label), group in tmp.loc[eligible].groupby(["month", "target_label"]):
        for feature in features:
            vals = group[feature]
            rows.append(dict(month=month, target_label=int(label), feature=feature,
                             n=len(vals), valid_n=int(vals.notna().sum()), mean=vals.mean(),
                             median=vals.median(), std=vals.std(ddof=1), missing_rate=vals.isna().mean()))
    monthly = pd.DataFrame(rows)
    _csv(out / "calendar_month_statistics.csv", monthly)
    quality = pd.DataFrame({"feature": features, "n": len(d), "valid_n": raw.notna().sum().values,
                            "missing_rate": raw.isna().mean().values, "min": raw.min().values,
                            "max": raw.max().values, "constant_nonmissing": [raw[f].nunique(dropna=True) <= 1 for f in features]})
    _csv(out / "raw_feature_quality.csv", quality)
    summary = dict(player=str(d.player.iloc[0]), observed_games=len(d), cutoff_hour=str(cutoff),
                   initially_excluded=int(d.hour.le(cutoff).sum()), primary_windows=len(block["X"]),
                   disjoint_windows=len(disjoint["X"]), unique_source_games=len(reuse),
                   max_source_reuse=max(reuse.values(), default=0),
                   order_verified=False, full_history_confirmed=False,
                   method="hour_invariant", missing_policy="no imputation; per-feature all 10 values required",
                   overlapping_windows_are_independent=False, plots_generated=bool(make_plots))
    _json(out / "analysis_settings.json", summary)
    lines = ["# 同一プレイヤーの段位別記述比較", "", "## 確認できた事実",
             f"観測{len(d):,}対局、主分析{len(block['X']):,}窓、元対局非重複補助分析{len(disjoint['X']):,}窓。",
             f"主分析は{len(reuse):,}元対局を使用し、1対局の最大再利用は{summary['max_source_reuse']}回。",
             "補完しない記述分析であり、H4/H5の補完済み特徴量分布と一致するとは限らない。", ""]
    main = differences[differences.view.eq("primary_hour_invariant")].sort_values("abs_smd", ascending=False)
    for row in main.itertuples():
        if pd.isna(row.mean_diff_high_minus_low):
            lines.append(f"- {row.feature}: 有効値不足により差を算出できない。")
        else:
            direction = "高い" if row.mean_diff_high_minus_low > 0 else "低い" if row.mean_diff_high_minus_low < 0 else "同じ"
            smd_text = f"{row.smd:.3f}" if pd.notna(row.smd) else "算出不可"
            lines.append(f"- {row.feature}: 高段位期で平均が{direction}（高−低={row.mean_diff_high_minus_low:.4g}、SMD={smd_text}、有効窓 低{row.low_valid_n}/高{row.high_valid_n}）。")
    lines += ["", "## 解釈・仮説", "特徴量定義表の測定対象に限って行動差を解釈する。方向が元対局・同じ終端・非重複窓で一致するかはdirection_robustness.csvで確認する。",
              "時系列図は月別の観測平均を示す。連続的変化や変化点は未検定であり、点の増減だけから学習効果を断定しない。",
              "欠損率やクラス構成の差で平均差が変わる可能性がある。発生機会別の分母を併記して検討する。",
              "", "## 限界"] + ["- " + t for t in LIMITATIONS]
    (out / "descriptive_report_ja.md").write_text("\n".join(lines), encoding="utf-8")
    if make_plots:
        _plot_descriptive(views, differences, monthly, features, out)
    return dict(summary=summary, statistics=stats, differences=differences,
                primary=block, disjoint=disjoint, output_dir=out)


def fit_means(rows, base_features, cutoff, forbidden=frozenset()):
    fitting = rows[(rows.hour > cutoff) & ~rows.game_id.isin(forbidden)]
    _require(len(fitting) > 0, "補完値を推定する訓練対局がありません")
    means = fitting[base_features].mean()
    _require(means.notna().all(), "訓練側で全欠損の特徴量: " + ", ".join(means.index[means.isna()]))
    return means, set(fitting.game_id)


def temporal_splits(data, cutoff):
    active = data[data.hour > cutoff]
    _require(len(active) >= 60, "初期期間後が60対局未満：固定3分割不可")
    bounds = [active.iloc[max(0, int(len(active)*q)-1)].hour for q in (.5, 2/3, 5/6)] + [active.hour.max()]
    _require(len(set(bounds)) == 4, "時間帯境界が重複：固定3分割不可")
    return [(k+1, data[data.hour <= a].copy(), data[(data.hour > a) & (data.hour <= b)].copy())
            for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))]


def classification_metrics(y, score, predicted=None):
    """Binary metrics. BA/AUC intentionally undefined for single-class y."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    _require(len(y) > 0 and len(y) == len(score), "評価データが空または長さ不一致")
    _require(set(y) <= {0, 1} and np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all(), "不正な正解/確率")
    pred = (score >= .5).astype(int) if predicted is None else np.asarray(predicted)
    _require(len(pred) == len(y) and set(pred) <= {0, 1}, "不正な予測")
    tn, fp = int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())
    fn, tp = int(((y == 1) & (pred == 0)).sum()), int(((y == 1) & (pred == 1)).sum())
    n0, n1 = tn+fp, fn+tp
    recall0, recall1 = tn/n0 if n0 else np.nan, tp/n1 if n1 else np.nan
    auc = np.nan
    if n0 and n1:
        ranks = pd.Series(score).rank(method="average").to_numpy()
        auc = (ranks[y == 1].sum() - n1*(n1+1)/2)/(n0*n1)
    f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.
    f0 = 2*tn/(2*tn+fn+fp) if (2*tn+fn+fp) else 0.
    return dict(n=len(y), weak=n0, strong=n1, strong_ratio=n1/len(y),
                accuracy=(tn+tp)/len(y), balanced_accuracy=(recall0+recall1)/2 if n0 and n1 else np.nan,
                roc_auc=auc, f1_strong=f1, f1_macro=(f0+f1)/2,
                recall_weak=recall0, recall_strong=recall1, tn=tn, fp=fp, fn=fn, tp=tp)


def _score_block(model, block, dataset, majority_class, out, make_plots=True):
    probabilities = model.predict_proba(block["X"])[:, 1]
    pred = block["meta"][["game_id", "hour", "target_label"]].copy()
    pred["strong_score"] = probabilities
    pred["predicted_label"] = (probabilities >= .5).astype(int)
    _csv(out / (dataset + "_predictions.csv"), pred)
    quantiles = pred.groupby("target_label").strong_score.quantile([0, .1, .25, .5, .75, .9, 1]).rename("score").reset_index().rename(columns={"level_1": "quantile"})
    _csv(out / (dataset + "_score_distribution.csv"), quantiles)
    result = []
    for name, scores in (("LightGBM", probabilities), ("always_weak", np.zeros(len(pred))),
                         ("always_strong", np.ones(len(pred))),
                         ("training_majority", np.full(len(pred), majority_class))):
        result.append(dict(dataset=dataset, estimator=name, **classification_metrics(block["y"], scores)))
    if make_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for label, name in ((0, "Low rank"), (1, "High rank")):
            vals = pred.loc[pred.target_label.eq(label), "strong_score"]
            if len(vals):
                axes[0].hist(vals, bins=np.linspace(0, 1, 21), weights=np.full(len(vals), 1/len(vals)),
                             alpha=.55, label=f"{name} n={len(vals)}")
        axes[0].axvline(.5, color="black", linestyle="--", linewidth=.8)
        axes[0].set_xlabel("Predicted high-rank score; fixed threshold = 0.5")
        axes[0].set_ylabel("Within-class fraction")
        axes[0].legend(fontsize=8)
        m = result[0]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        axes[1].imshow(cm, cmap="Blues")
        for (i, j), value in np.ndenumerate(cm):
            axes[1].text(j, i, str(value), ha="center", va="center", color="red")
        axes[1].set_xticks([0, 1], ["Low", "High"])
        axes[1].set_yticks([0, 1], ["Low", "High"])
        axes[1].set_xlabel("Predicted class")
        axes[1].set_ylabel("Observed rank class")
        fig.suptitle(dataset + " (hour_invariant; not strict chronology)")
        fig.tight_layout()
        fig.savefig(out / (dataset + "_diagnostics.png"), dpi=160)
        plt.close(fig)
    return result


def _data_digest(data, columns):
    return hashlib.sha256(data[columns].to_csv(index=False).encode("utf-8")).hexdigest()


def run_internal(data, features, out_dir, forbidden=frozenset(),
                 initial_exclude=INITIAL_EXCLUDE, base_features=None, cross_builder=None,
                 make_plots=True):
    """Fit only shobon, fixed 3 folds plus final frozen model. No external input."""
    from lightgbm import LGBMClassifier
    base = list(base_features or features)
    d, cutoff = validate_data(data, base, initial_exclude)
    _require(str(d.player.iloc[0]) in {"shobon", "しょぼんｎ", "しょぼんn"}, "学習はしょぼんｎのみ")
    out = _new_dir(out_dir)
    config = dict(features=list(features), base_features=base, model=MODEL_PLAN, threshold=.5,
                  window=WINDOW, initial_exclude=initial_exclude, cutoff_hour=str(cutoff),
                  scenario="hour_invariant", order_verified=False, full_history_confirmed=False,
                  imputation="training raw per-game means; rank17 included; initial period and shared games excluded from fit",
                  cross_builder=getattr(cross_builder, "__name__", None),
                  cross_builder_fingerprint=_transform_fingerprint(cross_builder),
                  analysis_module_sha256=_module_hash(),
                  external_forbidden_game_ids=sorted(forbidden),
                  data_sha256=_data_digest(d, ["game_id", "hour", "dan"] + base),
                  runtime_versions={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "lightgbm")})
    _json(out / "prespecified_protocol.json", config)
    rows, metrics = [], []
    for fold, train_rows, val_rows in temporal_splits(d, cutoff):
        record = dict(fold=fold, train_last_hour=str(train_rows.hour.max()), validation_first_hour=str(val_rows.hour.min()))
        try:
            _require(train_rows.hour.max() < val_rows.hour.min(), "分割時間帯が重複")
            means, fit_ids = fit_means(train_rows, base, cutoff, forbidden)
            tr = prepare_windows(train_rows, features, cutoff, forbidden, means, base, cross_builder)
            va = prepare_windows(val_rows, features, cutoff, frozenset(), means, base, cross_builder)
            overlap = source_ids(tr) & source_ids(va)
            _require(not overlap and not (fit_ids & source_ids(va)), "学習・補完と検証の元対局重複")
            _require(not (source_ids(tr) & set(forbidden)) and not (fit_ids & set(forbidden)), "外部共通対局が学習に混入")
            record.update(train_weak=int(tr["y"].eq(0).sum()), train_strong=int(tr["y"].eq(1).sum()),
                          validation_weak=int(va["y"].eq(0).sum()), validation_strong=int(va["y"].eq(1).sum()), source_overlap=0)
            _save_block(tr, f"fold{fold}_train", out)
            _save_block(va, f"fold{fold}_validation", out)
            _require(tr["y"].nunique() == 2, "学習側が単一クラス：学習を省略")
            _require(len(va["y"]) > 0, "検証窓が0件：採点を省略")
            _csv(out / f"fold{fold}_training_means.csv", means.rename("mean").rename_axis("feature").reset_index())
            model = LGBMClassifier(**MODEL_PLAN).fit(tr["X"], tr["y"])
            majority = int(tr["y"].mean() > .5)  # fixed low-class tie break
            metrics.extend(_score_block(model, va, f"fold{fold}", majority, out, make_plots))
            record["status"] = "evaluated" if va["y"].nunique() == 2 else "evaluated_single_class_AUC_BA_NA"
        except AnalysisBlocked as exc:
            record["status"] = "blocked: " + str(exc)
        rows.append(record)
    _csv(out / "fold_audit.csv", pd.DataFrame(rows))
    metric_frame = pd.DataFrame(metrics, columns=METRIC_COLUMNS)
    _csv(out / "internal_metrics.csv", metric_frame)
    means, fit_ids = fit_means(d, base, cutoff, forbidden)
    tr = prepare_windows(d, features, cutoff, forbidden, means, base, cross_builder)
    _save_block(tr, "final_train", out)
    _require(tr["y"].nunique() == 2, "最終学習側に両クラスがありません")
    model = LGBMClassifier(**MODEL_PLAN).fit(tr["X"], tr["y"])
    # Python handles Unicode paths on Windows; the native LightGBM file API
    # can fail for Japanese directory names. Text format remains portable.
    (out / "frozen_model.txt").write_text(model.booster_.model_to_string(), encoding="utf-8")
    _csv(out / "final_training_means.csv", means.rename("mean").rename_axis("feature").reset_index())
    _json(out / "training_source_ids.json", sorted(source_ids(tr)))
    _json(out / "imputation_source_ids.json", sorted(fit_ids))
    frozen_files = ["frozen_model.txt", "prespecified_protocol.json", "final_training_means.csv",
                    "training_source_ids.json", "imputation_source_ids.json"]
    freeze = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  training_majority=int(tr["y"].mean() > .5), train_weak=int(tr["y"].eq(0).sum()),
                  train_strong=int(tr["y"].eq(1).sum()),
                  files={f: hashlib.sha256((out/f).read_bytes()).hexdigest() for f in frozen_files})
    _json(out / "external_freeze.json", freeze)
    return dict(model=model, features=list(features), base_features=base, means=means,
                cross_builder=cross_builder, initial_exclude=initial_exclude,
                fit_ids=fit_ids, train_source_ids=source_ids(tr), majority_class=freeze["training_majority"],
                model_dir=out, freeze=freeze, metrics=metric_frame, fold_audit=pd.DataFrame(rows))


def load_frozen_bundle(model_dir, cross_builder=None):
    """Resume from verified text/JSON/CSV artifacts without fitting or pickle.

    Source code and numerical-library versions must match the freeze. Changing
    them requires a separately documented run, not silent model reuse.
    """
    import lightgbm as lgb
    folder = Path(model_dir)
    _require((folder / "external_freeze.json").is_file(), "固定モデル記録がありません")
    freeze = json.loads((folder / "external_freeze.json").read_text(encoding="utf-8"))
    for filename, sha in freeze["files"].items():
        _require((folder/filename).is_file() and hashlib.sha256((folder/filename).read_bytes()).hexdigest() == sha,
                 "外部評価前の固定ファイルが変更されました: " + filename)
    protocol = json.loads((folder/"prespecified_protocol.json").read_text(encoding="utf-8"))
    _require(protocol["analysis_module_sha256"] == _module_hash(), "分析モジュールがモデル固定時から変更されています")
    _require(protocol["cross_builder_fingerprint"] == _transform_fingerprint(cross_builder),
             "交差特徴量関数がモデル固定時から変更されています")
    for package, version in protocol["runtime_versions"].items():
        _require(importlib.metadata.version(package) == version,
                 f"固定時と依存ライブラリが異なります: {package}=={version} が必要")
    features, base = protocol["features"], protocol["base_features"]
    means = pd.read_csv(folder/"final_training_means.csv").set_index("feature")["mean"]
    tr_ids = set(json.loads((folder/"training_source_ids.json").read_text(encoding="utf-8")))
    fit_ids = set(json.loads((folder/"imputation_source_ids.json").read_text(encoding="utf-8")))
    booster = lgb.Booster(model_str=(folder / "frozen_model.txt").read_text(encoding="utf-8"))
    class FrozenPredictor:
        def predict_proba(self, x):
            p = booster.predict(x[features])
            return np.column_stack([1-p, p])
    return dict(model=FrozenPredictor(), features=features, base_features=base, means=means,
                cross_builder=cross_builder, initial_exclude=protocol["initial_exclude"],
                fit_ids=fit_ids, train_source_ids=tr_ids, majority_class=freeze["training_majority"],
                model_dir=folder, freeze=freeze, protocol=protocol,
                metrics=pd.read_csv(folder/"internal_metrics.csv"),
                fold_audit=pd.read_csv(folder/"fold_audit.csv"))


def run_external(bundle, data, out_dir, make_plots=True):
    """Evaluate once, or return verified saved results for an identical request.

    A completed identical request is read-only. A partial/failed attempt remains
    blocked, as silently retrying after inspecting outcomes is not a blind test.
    """
    frozen = load_frozen_bundle(bundle["model_dir"], bundle["cross_builder"])
    folder, freeze, protocol = frozen["model_dir"], frozen["freeze"], frozen["protocol"]
    features, base = frozen["features"], frozen["base_features"]
    d, cutoff = validate_data(data, base, frozen["initial_exclude"])
    _require(str(d.player.iloc[0]) in {"ultra", "ウルトラ立直"}, "外部評価はウルトラ立直のみ")
    _require(not (set(d.game_id) & (frozen["train_source_ids"] | frozen["fit_ids"])),
             "外部対局IDが学習または補完推定に混入")
    digest = _data_digest(d, ["game_id", "hour", "dan"] + base)
    attempt = folder / "external_attempt.json"
    if attempt.exists():
        previous = json.loads(attempt.read_text(encoding="utf-8"))
        _require(previous["status"] == "completed", "外部評価が開始済みで完了していません。部分出力を点検し、再学習や黙示の再評価はしないでください")
        _require(previous["data_sha256"] == digest, "外部データを変更した反復評価はしません")
        saved_dir = Path(previous["output_dir"])
        _require(saved_dir.resolve() == Path(out_dir).resolve(), "完了済みの出力先を指定: " + str(saved_dir))
        for filename, sha in previous["output_sha256"].items():
            _require((saved_dir/filename).is_file() and hashlib.sha256((saved_dir/filename).read_bytes()).hexdigest() == sha,
                     "保存済み外部結果が変更されています: " + filename)
        return pd.read_csv(saved_dir/"external_metrics.csv")
    block = prepare_windows(d, features, cutoff, means=frozen["means"],
                            base_features=base, cross_builder=frozen["cross_builder"])
    _require(len(block["X"]) > 0, "外部評価の窓が0件")
    out = _new_dir(out_dir)
    _save_block(block, "external", out)
    _json(attempt, dict(started_utc=datetime.now(timezone.utc).isoformat(),
                       data_sha256=digest, output_dir=str(out.resolve()), status="started"))
    metrics = pd.DataFrame(_score_block(frozen["model"], block, "external", freeze["training_majority"], out, make_plots))
    _csv(out / "external_metrics.csv", metrics)
    _csv(out / "external_rank_counts.csv", d.groupby("dan").size().rename("games").reset_index())
    _json(out / "external_settings.json", dict(cutoff_hour=str(cutoff), observed_games=len(d),
             scored_windows=len(block["X"]), model_freeze=freeze,
             order_verified=False, full_history_confirmed=False, threshold=.5))
    _json(attempt, dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                       data_sha256=digest, output_dir=str(out.resolve()), status="completed",
                       output_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}))
    return metrics
