"""Extend paired-code cache, strictly offline, CPU and one family at a time."""
import argparse
import copy
import json
from pathlib import Path
import random
import string
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from decision import VARIANTS, split_items
from phase0 import CACHE, load
from sim import MODELS, directions, forward_logits, project, weights


def build(family, split):
    path = CACHE/f'codes-{family}-{split}.npz'
    if path.exists():
        print(f'cache_hit={path}',flush=True)
        return
    start=time.perf_counter()
    _,items=load(family)
    indices=split_items(items)[split][:22]
    selected=[copy.deepcopy(items[i]) for i in indices]
    rng=random.Random(20261122)
    used={x['code'] for x in items}
    for x in selected:
        while True:
            code=''.join(rng.choices(string.ascii_lowercase+string.digits,k=3))
            if code not in used:
                used.add(code)
                break
        x['code']=code
    tokenizer=AutoTokenizer.from_pretrained(MODELS[family],local_files_only=True)
    tokenizer.padding_side='left'
    tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODELS[family],local_files_only=True,dtype=torch.float32).cpu().eval()
    direction=directions(model)
    original=next(CACHE.glob(f'{family}-*.npz'))
    with np.load(original) as data:
        ratio=json.loads(str(data['metadata']))['ratio']
    plain=forward_logits(model,tokenizer,selected,'plain')
    arrays=[]
    for variant in VARIANTS:
        if variant in ('plain','watermark','temp1'):
            logits=plain
        else:
            with weights(model,variant,direction,ratio):
                logits=forward_logits(model,tokenizer,selected,variant)
        arrays.append(project(tokenizer,selected,logits,variant)[0])
    np.savez_compressed(path,probabilities=np.array(arrays),indices=indices,items=json.dumps(selected),elapsed_s=time.perf_counter()-start)
    print(f'{family} {split}: 22 second codes, 9 variants, elapsed_s={time.perf_counter()-start:.2f}, cache={path}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families',nargs='+',choices=MODELS,required=True)
    parser.add_argument('--splits',nargs='+',choices=['tune','holdout','type'],required=True)
    args=parser.parse_args()
    torch.set_num_threads(8)
    for family in args.families:
        for split in args.splits:
            build(family,split)


if __name__=='__main__':
    main()
