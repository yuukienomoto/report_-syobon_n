"""Read-only additions outside the signed extractor package."""
import argparse,json
from pathlib import Path
import pandas as pd
from features_100.quality import run_quality
from features_100.rules import IMPLEMENTED
from decision_features.pipeline import load_c5
from mahjong_research import tenhou
from mahjong_research.analysis import validate_data
from mahjong_research.inputs import new_run,write_csv,write_json
from study import make_blocks

def boundary_counts(config):
    data,shared,_,_=load_c5(config['run_dir'],config.get('key','shobon'),config['player'])
    d,cutoff=validate_data(data,tenhou.FEATURE_NAMES)
    end=pd.Timestamp(config['exploration_end_hour'])
    if end!=end.floor('h') or not cutoff<end<d.hour.max(): raise ValueError('探索境界は除外後・最終時間帯より前の正時')
    rows=[]
    for scope,widths in make_blocks(d,cutoff,shared).items():
        for width,starts in widths.items():
            for start in starts:
                p=d.iloc[start:start+width]
                part='exploration' if p.hour.iloc[-1]<=end else 'reserved' if p.hour.iloc[0]>end else 'boundary_excluded'
                rows.append(dict(scope=scope,width=width,label=int(p.target_label.iloc[0]),partition=part))
    if not rows: raise ValueError('対象窓なし')
    return pd.DataFrame(rows).value_counts().rename('windows').reset_index()

def correlations(w):
    rows=[]
    # Within-rank correlation avoids attributing a pooled rank/time mixture to redundancy.
    for (scope,width,label),g in w[w.feature_id.isin(IMPLEMENTED)].groupby(['scope','width','label']):
        p=g.pivot(index='window_id',columns='feature_id',values='value')
        for i,a in enumerate(p.columns):
            for b in p.columns[i+1:]:
                z=p[[a,b]].dropna();n=len(z)
                valid=n>=3 and z[a].nunique()>1 and z[b].nunique()>1
                rows.append(dict(scope=scope,width=width,label=label,feature_a=a,feature_b=b,paired_n=n,
                    pearson=z[a].corr(z[b]) if valid else float('nan'),
                    spearman=z[a].rank().corr(z[b].rank()) if valid else float('nan'),
                    status='exploratory' if valid else 'insufficient_or_constant'))
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['boundary','quality']);p.add_argument('config');p.add_argument('--extraction-dir');a=p.parse_args()
    config=json.loads(Path(a.config).read_text(encoding='utf8'))
    counts=boundary_counts(config)
    if a.action=='boundary':
        out=new_run(Path(config['output_root'])/'boundary_audit');write_csv(out/'partition_counts.csv',counts)
        print(counts.to_string(index=False));print(out);return
    if not config.get('exploration_boundary_reason','').strip(): raise ValueError('探索境界を決めた理由を設定してください')
    for width in (10,20):
        z=counts[(counts.scope=='available')&(counts.width==width)&(counts.partition=='exploration')]
        if set(z.label)!={0,1}: raise ValueError(f'{width}対局探索窓に両群がありません。境界と研究条件を確認')
    if not a.extraction_dir: raise ValueError('完成した全件抽出フォルダを指定')
    out=run_quality(config,a.extraction_dir)
    w=pd.read_csv(out/'window_values.csv')
    write_csv(out/'feature_correlations.csv',correlations(w))
    write_json(out/'boundary_rationale.json',dict(reason=config['exploration_boundary_reason'],end=config['exploration_end_hour']))
    with (out/'report_ja.md').open('a',encoding='utf8') as f:
        f.write('\n\n追加：feature_correlations.csvは段位群・窓幅・対象集合別のペア完全観測相関です。対応窓数を併記し、定数や3窓未満は欠損。高相関だけで除外しません。境界内に両群があっても各特徴量の機会数が十分とは限らず、group_statistics.csvを確認してください。\n')
    print('保存先:',out)

if __name__=='__main__':main()
