"""Run frozen R/C on local answers.jsonl and a fixed prompt manifest."""
import argparse
import json
from pathlib import Path

import numpy as np

from decision import decide, read_answers, write_submission


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--answers',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--flags',type=Path,help='JSON mapping endpoint to observed sys/persona boolean')
    parser.add_argument('--params',type=Path,default=Path(__file__).with_name('frozen_params.json'))
    parser.add_argument('--method',choices=['R','C'],default='R')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    params=json.loads(args.params.read_text())
    responses=read_answers(args.answers,manifest)
    if not responses:
        parser.error('No responses found')
    endpoints=sorted(responses)
    flags=json.loads(args.flags.read_text()) if args.flags else {}
    counts=np.array([responses[e] for e in endpoints])
    probe_indices=sorted({x['item'] for p in manifest.values() for x in p['items'] if x.get('code_index',0)==1})
    if params['enable_watermark'] and not probe_indices:
        parser.error('Frozen watermark detector requires paired-code probe responses')
    codes=counts.sum(1)[:,probe_indices]
    if not probe_indices:
        codes=np.zeros((len(endpoints),1,2,11),dtype=int)
    family,model=decide(counts[:,:,:,0],codes,[bool(flags.get(e,False)) for e in endpoints],params,args.method)
    write_submission(args.output,endpoints,family,model)
    print(f'Wrote {len(endpoints)} endpoints to {args.output}')


if __name__=='__main__':
    main()
