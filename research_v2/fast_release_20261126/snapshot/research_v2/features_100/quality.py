"""Opportunity-pooled non-overlap windows; exploration-only quality statistics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from mahjong_research import tenhou
from mahjong_research.analysis import validate_data
from mahjong_research.inputs import new_run, write_json, write_csv, sha256
from decision_features.pipeline import load_c5
from study import make_blocks, describe
from .pipeline import IDS, signature, implementation_status
from .rules import CATALOG

PARENT={**{f'P{i:02d}':'BASE_F1_T_UNSAFE' for i in (1,4,8,9,16,20)},
        **{f'P{i:02d}':'BASE_F1_MIN_UNSAFE' for i in (2,3,5,10,11,12,17)},
        'P13':'BASE_FM_T_UNSAFE','Q01':'BASE_F0_MAX_U','Q02':'BASE_F0_MAX_U',
        'Q03':'BASE_F0_DORA_VS_U','Q09':'BASE_F0_DORA_VS_U','S09':'BASE_F0_DORA_VS_U',
        'D15':'BASE_F1_SAFE',
        **{f'R{i:02d}':'BASE_R0_DECLARE' for i in (5,6,7,12,13,15)},'S04':'BASE_R0_DECLARE'}


def aggregate(block,feature):
    s=block[feature+'__status']
    if not s.isin(['ok','no_opportunity']).all():
        return np.nan,np.nan,np.nan,'unimplemented_or_unknown'
    a=pd.to_numeric(block[feature+'__numerator'],errors='raise')
    b=pd.to_numeric(block[feature+'__denominator'],errors='raise')
    if a.isna().any() or b.isna().any() or (a<0).any() or (b<a).any() or a.mod(1).ne(0).any() or b.mod(1).ne(0).any():
        raise ValueError('分子・分母の不整合: '+feature)
    num,den=a.sum(),b.sum()
    return num/den if den else np.nan,float(num),float(den),'ok' if den else 'zero_denominator'


def windows(d,counts,blocks,end):
    rows=[]; membership=[]; separation=[]
    for scope,widths in blocks.items():
        for width in (10,20):
            seen=set()
            for start in widths[width]:
                part=d.iloc[start:start+width]; ids=part.game_id.tolist()
                if seen.intersection(ids): raise ValueError('元対局重複')
                seen.update(ids)
                if part.target_label.nunique()!=1 or part.target_label.isna().any(): raise ValueError('段位境界をまたぐ窓')
                # Recheck full-hour boundaries, even when called with a custom block list.
                if (start>0 and d.hour.iloc[start-1]==part.hour.iloc[0]) or (start+width<len(d) and d.hour.iloc[start+width]==part.hour.iloc[-1]):
                    raise ValueError('時間帯を分断する窓')
                use='exploration' if part.hour.iloc[-1]<=end else 'reserved' if part.hour.iloc[0]>end else 'boundary_excluded'
                wid=f'{scope}-{width}-{start}'
                meta=dict(scope=scope,width=width,window_id=wid,label=int(part.target_label.iloc[0]),
                          first_hour=str(part.hour.iloc[0]),last_hour=str(part.hour.iloc[-1]),source_ids=json.dumps(ids),partition=use)
                membership.append(meta); separation.append({k:meta[k] for k in ('scope','width','label','partition')})
                if use!='exploration': continue  # never compute reserved feature statistics
                block=counts.loc[ids]
                if not block.status.eq('ok').all(): raise ValueError('抽出失敗を含む窓。対局を削除せず監査を確認。')
                for f in IDS:
                    value,num,den,status=aggregate(block,f)
                    rows.append(dict(meta,feature_id=f,value=value,numerator=num,denominator=den,status=status))
    return pd.DataFrame(rows),pd.DataFrame(membership),pd.DataFrame(separation)


def summarize(w):
    stats=[]; periods=[]; noise=[]; comparisons=[]
    for (scope,width,f,label),g in w.groupby(['scope','width','feature_id','label'],sort=False):
        key=dict(scope=scope,width=int(width),feature_id=f,label=int(label))
        v=g.value; den=g.denominator
        ss=describe(v,den)
        ss.update(valid_window_rate=v.notna().mean(),zero_denominator_rate=g.status.eq('zero_denominator').mean(),
                  unknown_window_rate=g.status.eq('unimplemented_or_unknown').mean(),
                  zero_rate=v.dropna().eq(0).mean(),one_rate=v.dropna().eq(1).mean())
        stats.append(dict(key,**ss))
        # Nonoverlap blocks have increasing complete-hour endpoints; split by time,
        # without asserting ordering within any hour or dropping missing values first.
        ordered=g.sort_values('last_hour')
        for index,part in enumerate(np.array_split(np.arange(len(ordered)),3),1):
            z=ordered.iloc[part]
            periods.append(dict(key,period=index,first_hour=z.first_hour.min() if len(z) else '',
                last_hour=z.last_hour.max() if len(z) else '',**describe(z.value,z.denominator)))
        good=g[g.value.notna()]
        pooled=good.numerator.sum()/good.denominator.sum() if good.denominator.sum()>0 else np.nan
        theoretical=(pooled*(1-pooled)/good.denominator).mean()
        # A reference variance under independent Bernoulli trials, not a variance
        # component estimator: repeated actions within a round are dependent.
        noise.append(dict(key,observed_variance=v.var(),iid_binomial_reference=theoretical,
                          difference_not_debiased_estimate=v.var()-theoretical,pooled_event_rate=pooled,
                          assumption='独立二項は参考仮定。真の行動分散を同定しない'))
        for lower,upper in ((1,2),(3,4),(5,9),(10,19),(20,np.inf)):
            z=good[good.denominator.between(lower,upper)]
            periods.append(dict(key,period=f'den_{lower}_{upper}',first_hour='',last_hour='',**describe(z.value,z.denominator)))
    st=pd.DataFrame(stats)
    for (scope,width,f),g in w.groupby(['scope','width','feature_id'],sort=False):
        groups=[g[g.label.eq(label)].value.dropna() for label in (0,1)]
        a,b=groups; delta=b.mean()-a.mean(); sd=np.nan
        if min(len(a),len(b))>=2:
            sd=np.sqrt(((len(a)-1)*a.var()+(len(b)-1)*b.var())/(len(a)+len(b)-2))
        overlap=np.nan
        if len(a) and len(b):
            pa=np.histogram(a,bins=np.linspace(0,1,21))[0]/len(a)
            pb=np.histogram(b,bins=np.linspace(0,1,21))[0]/len(b)
            overlap=np.minimum(pa,pb).sum()
        comparisons.append(dict(scope=scope,width=int(width),feature_id=f,low_n=len(a),high_n=len(b),
            low_mean=a.mean(),high_mean=b.mean(),difference=delta,median_difference=b.median()-a.median(),
            pooled_sd=sd,smd=delta/sd if sd>0 else np.nan,overlap_20bins=overlap))
    return st,pd.DataFrame(comparisons),pd.DataFrame(periods),pd.DataFrame(noise)


def parent_comparison(w):
    rows=[]
    for child,parent in PARENT.items():
        a=w[w.feature_id.eq(child)]; b=w[w.feature_id.eq(parent)]
        both=a.merge(b,on=['scope','width','window_id','label'],suffixes=('_child','_parent'),validate='one_to_one')
        for (scope,width,label),g in both.groupby(['scope','width','label']):
            z=g[g.value_child.notna()&g.value_parent.notna()]
            rows.append(dict(feature_id=child,parent=parent,scope=scope,width=width,label=label,
                original_windows=len(g),common_valid_windows=len(z),child_mean=z.value_child.mean(),parent_mean=z.value_parent.mean(),
                child_sd=z.value_child.std(),parent_sd=z.value_parent.std(),
                child_den_median=z.denominator_child.median(),parent_den_median=z.denominator_parent.median(),
                caution='同じ元対局の窓。判断機会集合と重みは異なる。S04は分子方向が逆。'))
    return pd.DataFrame(rows)


def run_quality(config,extraction_dir):
    source_dir=Path(extraction_dir)
    marker=json.loads((source_dir/'complete.json').read_text(encoding='utf8'))
    data,shared,plan,source=load_c5(config['run_dir'],config.get('key','shobon'),config['player'])
    if marker['signature']!=signature(source): raise ValueError('解析コード・元入力の署名不一致。対応する版のNotebookを使用。')
    if marker['counters_sha256']!=sha256(source_dir/'game_counters.csv') or marker['audit_sha256']!=sha256(source_dir/'extraction_audit.csv'):
        raise ValueError('抽出出力のハッシュ不一致')
    if not marker['all_reconstructed']: raise ValueError('抽出失敗あり。監査を解決してBを再実行。成功キャッシュは再利用。')
    counts=pd.read_csv(source_dir/'game_counters.csv',float_precision='round_trip')
    if not counts.game_id.is_unique or counts.game_id.isna().any(): raise ValueError('重複・欠損ID')
    d,cutoff=validate_data(data,tenhou.FEATURE_NAMES)
    # Pilot subsets must not be compacted to create artificial consecutive windows.
    if set(counts.game_id)!=set(d.game_id): raise ValueError('Cは全C5対局の抽出が必要です。Aの少数対局を詰めて窓にしません。')
    counts=counts.set_index('game_id').loc[d.game_id]
    for column in ('dan','player','xml_sha256','file_id','seat'):
        if not np.array_equal(counts[column].to_numpy(),d[column].to_numpy()): raise ValueError('C5結合不一致: '+column)
    if not config.get('exploration_end_hour'): raise ValueError('exploration_end_hourに探索期間末の時間帯を指定。未使用評価期間は統計から除外。')
    end=pd.Timestamp(config['exploration_end_hour'])
    if end!=end.floor('h') or not cutoff<end<d.hour.max(): raise ValueError('探索境界は初期除外後〜最終時間帯より前の正時で指定')
    blocks=make_blocks(d,cutoff,shared)
    w,members,separation=windows(d,counts,blocks,end)
    if w.empty: raise ValueError('探索期間の完全時間帯10/20対局窓がありません')
    st,comp,periods,noise=summarize(w); out=new_run(Path(config['output_root'])/'quality')
    write_csv(out/'window_values.csv',w)
    write_csv(out/'window_membership.csv',members)
    write_csv(out/'partition_counts.csv',separation.value_counts().rename('windows').reset_index())
    for name,frame in [('group_statistics.csv',st),('feature_comparison.csv',comp),('period_and_exposure_statistics.csv',periods),
                       ('sampling_noise_reference.csv',noise),('parent_child_comparison.csv',parent_comparison(w)),
                       ('implementation_status.csv',implementation_status())]: write_csv(out/name,frame)
    write_csv(out/'parent_coverage.csv',pd.DataFrame([dict(feature_id=f,parent=PARENT.get(f,''),
        status='computed_on_common_windows' if f in PARENT else 'not_defined_or_not_implemented',
        reason='機械的な条件削除は行わない。未対応の親比較は結果として主張しない。') for f in CATALOG]))
    write_json(out/'sources.json',dict(signature=marker['signature'],extraction_dir=str(source_dir),
        exploration_end_hour=str(end),order_verified=False,full_history_confirmed=False,
        final_evaluation_data='reserved periods excluded from feature statistics; not guaranteed previously unseen'))
    render(out,w,comp,st,periods)
    return out


def render(out,w,comp,stats,periods):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in font_manager.findSystemFonts():
        if 'NotoSansCJK' in f:
            font_manager.fontManager.addfont(f); plt.rcParams['font.family']=font_manager.FontProperties(fname=f).get_name(); break
    figure_dir=out/'figures'; figure_dir.mkdir()
    for f,r in CATALOG.items():
        a=w[w.feature_id.eq(f)&w.scope.eq('available')]
        fig,axes=plt.subplots(1,3,figsize=(13,3.5))
        for j,width in enumerate((10,20)):
            z=a[a.width.eq(width)]; vs=[z[z.label.eq(k)].value.dropna().to_numpy() for k in (0,1)]
            if any(len(v) for v in vs): axes[j].boxplot(vs,tick_labels=['低段位期','高段位期'])
            axes[j].set(title=f'{width}対局・非重複',ylim=(-.05,1.05),ylabel='窓内機会数合算率')
        z=stats[(stats.feature_id==f)&(stats.scope=='available')]
        for label,name in ((0,'低段位期'),(1,'高段位期')):
            zz=z[z.label.eq(label)].sort_values('width'); axes[2].plot(zz.width,zz.sd,'o-',label=name)
        axes[2].set(title='群内SD',xlabel='対局数'); axes[2].legend()
        fig.suptitle(f+': '+r['japanese_name'],fontsize=10); fig.tight_layout()
        fig.savefig(figure_dir/(f+'.png'),dpi=140); plt.close(fig)
    primary=comp[(comp.scope=='available')&(comp.width==10)&comp.feature_id.isin(CATALOG)].copy()
    rank=primary.assign(abs_smd=primary.smd.abs()).sort_values('abs_smd',ascending=False)
    lines=['# 100候補：探索期間の品質評価','',
        '割合は窓内の分子合計÷分母合計。各窓を等重みで比較し、全イベント合算率とは区別する。',
        '分母ゼロ・判定不能・未実装を0で補完していない。小標本の見かけの0/100%や低分散は有用性の証拠ではない。','',
        '## 数値に基づく探索的比較','']
    for row in rank.head(15).itertuples():
        lines.append(f'- {row.feature_id}: 高−低={row.difference:.6g}, SMD={row.smd:.4g}, 有効窓={row.low_n}/{row.high_n}, 重なり={row.overlap_20bins:.3g}')
    lines += ['', '## 解釈上の限界',
        '観測機会と群内分散はgroup_statistics、時期・分母帯別はperiod_and_exposure_statisticsで確認する。',
        '二項分散は独立試行を仮定した参考値であり、差引きによって真の行動分散が求まったとは扱わない。',
        '局内反復、同一人物、時間経過と段位の交絡、環境差が残る。SD減少やSMDの大きさを分類性能・因果効果とみなさない。',
        '同一時間帯内の順序と全履歴の完全性は未確認。窓幅で対象対局が変わるためcommon_gamesも併記する。',
        '探索境界より後の窓の値は集計していない。ただし過去に見た期間は独立な最終評価には戻らない。',
        '親比較はparent_coverage.csvの対応済み項目に限定。全100候補の親比較が完成したとは主張しない。',
        '順位は記述的な表示順であり採用判定ではない。後続の特徴量選択は学習期間内で行い、多数探索の偶然差に注意する。']
    (out/'report_ja.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
