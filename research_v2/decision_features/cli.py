"""Isolated Colab process: preview identities must match before extraction."""
import argparse
import json
from pathlib import Path
from mahjong_research.inputs import write_json
from .pipeline import prepare, run_pilot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['preview','extract'])
    p.add_argument('config'); p.add_argument('preview'); p.add_argument('result')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    ctx = prepare(**config)
    selection = dict(signature=ctx['signature'], ids=ctx['targets'].game_id.tolist(),
                     file_ids=ctx['targets'].file_id.tolist(), archive=str(ctx['archive']))
    if args.action == 'preview':
        write_json(args.preview, selection)
        print(ctx['targets'][['game_id','dan','target_label','hour','selection_reason']].to_string(index=False))
    else:
        if json.loads(Path(args.preview).read_text(encoding='utf8')) != selection:
            raise ValueError('P3のプレビュー後に入力・対象ID・コードが変わりました。P3を再実行してください。')
        out = run_pilot(ctx)
        write_json(args.result, dict(output=str(out)))


if __name__ == '__main__': main()
