"""Separate Colab process avoids mutating loaded notebook dependencies."""
import argparse
import json
from pathlib import Path
from .pipeline import prepare,extract_run
from .quality import run_quality


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['preview','extract','quality'])
    p.add_argument('config'); p.add_argument('--mode',choices=['A','B'],default='A')
    p.add_argument('--extraction-dir'); p.add_argument('--receipt',required=True)
    args=p.parse_args(); config=json.loads(Path(args.config).read_text(encoding='utf8'))
    receipt=Path(args.receipt)
    if args.action=='preview':
        c=prepare(config,args.mode)
        receipt.parent.mkdir(parents=True,exist_ok=True)
        receipt.write_text(json.dumps(dict(signature=c['signature'],mode=args.mode,config=config,
            ids=c['targets'].game_id.tolist()),ensure_ascii=False),encoding='utf8')
        return
    if args.action=='extract':
        old=json.loads(receipt.read_text(encoding='utf8')); c=prepare(config,args.mode)
        if old!=dict(signature=c['signature'],mode=args.mode,config=config,ids=c['targets'].game_id.tolist()):
            raise ValueError('プレビュー後に対象・版・設定が変わりました。件数を再確認してください。')
        out=extract_run(c)
    else:
        if not args.extraction_dir: raise ValueError('--extraction-dirが必要です')
        out=run_quality(config,args.extraction_dir)
    receipt.with_suffix('.result.json').write_text(json.dumps(dict(output=str(out))),encoding='utf8')
    print('保存先:',out)

if __name__=='__main__': main()
