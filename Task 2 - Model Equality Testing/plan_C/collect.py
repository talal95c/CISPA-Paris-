"""Build the fixed 100-prompt protocol. Default is dry-run; live is opt-in only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

import numpy as np
import requests

from decision import VARIANTS, split_items

PARALLEL=4
MAX_TOKENS=256
ROOT=Path(__file__).resolve().parent
CACHE=Path.home()/'me'/'plan_c_cache'
REF_VARIANTS=['plain','fp8','system','lora0.7','persona','watermark','lora1.5','temp1','lora0.3']


def load_bank():
    paths=list(CACHE.glob('135m-*.npz'))
    if len(paths)!=1:
        raise ValueError('Expected exactly one 135M base cache')
    with np.load(paths[0]) as data:
        return json.loads(str(data['metadata']))['items']


def make_protocol():
    bank=load_bank()
    params=json.loads((ROOT/'frozen_params.json').read_text())
    selected=split_items(bank)['tune'][:params['active_items']]
    rng=np.random.default_rng(20261123)
    prompts=[]; manifest={}
    def add(kind,text,items=None,half=0,repeat_of=None):
        pid=str(len(prompts))
        prompts.append(dict(id=pid,prompt=text,kind=kind,repeat_of=repeat_of))
        manifest[pid]=dict(half=half,kind=kind,items=items or [],repeat_of=repeat_of)
    for text in ['Who are you? Introduce yourself briefly.',
                 'What organization do you represent, if any? Do not invent an affiliation.',
                 'Describe your usual speaking style and role in two sentences.',
                 'How would you introduce yourself to a new user?']:
        add('flag',text)
    topics=['an ancient library','a journey through a forest','the history of navigation',
            'how weather changes','a community garden','the invention of printing',
            'a quiet train station','learning a musical instrument','the water cycle','a night sky']
    openings=['Write a coherent descriptive passage about',
              'Write a clear explanatory passage for a general audience about',
              'Tell a detailed, original story inspired by']
    for opening in openings:
        for topic in topics:
            add('long',f'{opening} {topic}. Use continuous prose, around 180 words, without headings or bullet points.')
    instruction='Answer every item with its code, one space, and exactly one allowed option. One line per item. No explanations.\n'
    bursts=[]
    for b in range(62):
        if b>=60:
            text,items=bursts[b-60]
            add('burst',text,items,half=b//31,repeat_of=str(34+b-60))
            continue
        indices=rng.choice(selected,22,replace=False).tolist()
        items=[dict(item=i,code=bank[i]['code'],options=bank[i]['options'],code_index=0) for i in indices]
        text=instruction+'\n'.join(f"{bank[i]['code']} {bank[i]['question']} Options: {' / '.join(bank[i]['options'])}" for i in indices)
        bursts.append((text,items))
        add('burst',text,items,half=b//31)
    for text in ['Explain why seasons occur in a concise paragraph.',
                 'Describe a fair way to divide a shared resource.',
                 'Continue: The traveller opened the old notebook and discovered',
                 'Explain how to check whether a factual claim is reliable.']:
        add('reserve',text)
    assert len(prompts)==100
    return prompts,manifest


def save_fixed(path,text):
    if path.exists():
        if path.read_text()!=text:
            raise ValueError(f'Refusing to replace a different existing protocol: {path}')
    else:
        path.write_text(text)


def append_json(path,row):
    with path.open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path,repair_corrupt=False):
    if not path.exists():
        return []
    rows=[];lines=[];repair=False
    with path.open('rb') as handle:
        for number,line in enumerate(handle,1):
            if not line.strip():
                continue
            try:
                row=json.loads(line.decode('utf-8'))
            except (json.JSONDecodeError,UnicodeDecodeError) as error:
                if not repair_corrupt:
                    raise ValueError(f'Corrupt JSONL line {number} in {path}; manual inspection required before resume') from error
                print(f'WARNING: ignoring corrupt JSONL line {number} in {path}',file=sys.stderr)
                repair=True
                continue
            if not line.endswith(b'\n') and not repair_corrupt:
                raise ValueError(f'Unterminated JSONL line {number} in {path}; manual inspection required before resume')
            rows.append(row)
            lines.append(line if line.endswith(b'\n') else line+b'\n')
            repair|=not line.endswith(b'\n')
    if repair:
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent,prefix=f'.{path.name}.',delete=False) as handle:
                temporary=Path(handle.name)
                handle.writelines(lines)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary,path)
            directory_fd=os.open(path.parent,os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return rows


def answer_key(row):
    return row['endpoint'],str(row['id']),row.get('repeat',0)


def recover(directory):
    events=read_jsonl(directory/'responses.jsonl')
    answers=directory/'answers.jsonl'
    rows=read_jsonl(answers,repair_corrupt=True)
    done={answer_key(x) for x in rows}
    if len(done)!=len(rows):
        raise ValueError('Duplicate answer keys; refusing ambiguous resume')
    pending=set()
    for event in events:
        if event.get('event')=='request':
            pending.add(tuple(event['key']))
        elif event.get('event')=='http' and event['status']==429:
            pending.discard(tuple(event['key']))
        elif event.get('event')=='success':
            row=event['answer'];key=answer_key(row)
            pending.discard(key)
            if key not in done:
                append_json(answers,row)
                done.add(key)
    pending-=done
    if pending:
        raise ValueError(f'Uncertain previous requests: {sorted(pending)}. Inspect full journal/budget; no automatic retry.')
    return done


def mock(directory,prompts,manifest,endpoints,target):
    done=recover(directory)
    arrays={}
    for index,endpoint in enumerate(endpoints):
        family='135m' if target=='refs' else ['135m','360m','qwen'][index//11]
        variant=REF_VARIANTS[index] if target=='refs' else VARIANTS[index%9]
        if family not in arrays:
            with np.load(next(CACHE.glob(f'{family}-*.npz'))) as data:
                arrays[family]={v:data[v] for v in VARIANTS}
        for prompt in prompts:
            key=(endpoint,prompt['id'],0)
            if key in done:
                continue
            seed=int.from_bytes(hashlib.sha256(f'{endpoint}:{prompt["id"]}:20261123'.encode()).digest()[:8],'big')
            rng=np.random.default_rng(seed)
            lines=[]
            for item in manifest[prompt['id']]['items']:
                p=arrays[family][variant][item['item']]
                category=int(rng.choice(11,p=p/p.sum()))
                option=item['options'][category] if category<len(item['options']) else '?'
                lines.append(f"{item['code']} {option}")
            # Non-burst mocks are placeholders, NOT local free model generations.
            response='\n'.join(lines) if lines else ('I am the ACME assistant.' if variant=='system' else 'I am Captain Rowan.' if variant=='persona' else 'Local mock response.')
            row=dict(endpoint=endpoint,id=prompt['id'],repeat=0,prompt=prompt['prompt'],response=response)
            append_json(directory/'responses.jsonl',dict(event='success',mode='mock',answer=row,response={'choices':[{'message':{'content':response}}]}))
            append_json(directory/'answers.jsonl',row)
    print(f'MOCK: {len(read_jsonl(directory/"answers.jsonl"))} answers; HTTP requests=0')


def retry_after(value):
    try:
        seconds=float(value)
    except (ValueError,TypeError):
        seconds=(parsedate_to_datetime(value)-datetime.now(timezone.utc)).total_seconds()
    if not np.isfinite(seconds) or seconds>3600:
        raise ValueError('Invalid or excessive Retry-After; inspect server response')
    return max(0.,seconds)


def live(directory,prompts,endpoints,server_url,api_key,transport=None,wait=time.sleep):
    """Called only after CLI authorization. Injected transport tests use no sockets."""
    if not api_key:
        raise ValueError('Set MODEL_EQUALITY_API_KEY in the environment; never put it in files')
    done=recover(directory)
    jobs=[(endpoint,p) for p in prompts for endpoint in endpoints if (endpoint,p['id'],0) not in done]
    if not jobs:
        print('LIVE: all requested answers already present; no HTTP requests needed')
        return
    if transport is None:
        local=threading.local()
        def transport(method,url,**kwargs):
            if not hasattr(local,'session'):
                local.session=requests.Session()
                local.session.headers['X-API-Key']=api_key
            return local.session.request(method,url,allow_redirects=False,**kwargs)
    url=server_url.rstrip('/')+'/33-model-equality/v1'
    log=directory/'responses.jsonl'
    for attempt in range(4):
        response=transport('GET',url+'/models',timeout=60)
        try:
            body=response.json()
        except ValueError:
            body=dict(text=response.text)
        append_json(log,dict(event='budget',status=response.status_code,response=body,headers=dict(response.headers)))
        if response.status_code==200:
            break
        if response.status_code==429 and response.headers.get('Retry-After') and attempt<3:
            wait(retry_after(response.headers['Retry-After']))
        else:
            raise RuntimeError(f'Budget HTTP {response.status_code}; see journal')
    budgets={m['id']:int(m['budget']['limit'])-int(m['budget']['used']) for m in body['data']}
    remaining={e:sum(ep==e for ep,_ in jobs) for e in endpoints}
    if any(e not in budgets or budgets[e]<remaining[e] for e in endpoints):
        raise RuntimeError('Insufficient or missing endpoint budget; no completions sent')
    lock=threading.Lock();stop=threading.Event()
    def one(job):
        endpoint,prompt=job
        key=(endpoint,prompt['id'],0)
        payload=dict(model=endpoint,max_tokens=MAX_TOKENS,messages=[dict(role='user',content=prompt['prompt'])])
        for attempt in range(5):
            with lock:
                if stop.is_set():
                    return
                if budgets[endpoint]<=0:
                    stop.set()
                    raise RuntimeError('Budget exhausted during run; see journal')
                budgets[endpoint]-=1
                append_json(log,dict(event='request',key=key,request=payload))
            try:
                response=transport('POST',url+'/chat/completions',json=payload,timeout=180)
                try:
                    body=response.json()
                except ValueError:
                    body=dict(text=response.text)
                with lock:
                    append_json(log,dict(event='http',key=key,status=response.status_code,response=body,headers=dict(response.headers)))
                if response.status_code==429:
                    with lock:
                        budgets[endpoint]+=1
                    if response.headers.get('Retry-After') and attempt<4:
                        wait(retry_after(response.headers['Retry-After']))
                        continue
                    raise RuntimeError('429 without retry allowance; endpoint budget/busy state needs inspection')
                if response.status_code!=200:
                    raise RuntimeError(f'Completion HTTP {response.status_code}; no unsafe automatic retry')
                content=body['choices'][0]['message']['content']
                if not isinstance(content,str):
                    raise ValueError('Non-text completion')
                budget=body['budget']
                left=int(budget['limit'])-int(budget['used'])
                row=dict(endpoint=endpoint,id=prompt['id'],repeat=0,prompt=prompt['prompt'],response=content)
                with lock:
                    # Full response is durable before the compact official-format row.
                    append_json(log,dict(event='success',key=key,answer=row,response=body,budget=budget))
                    append_json(directory/'answers.jsonl',row)
                    # Local reservations also account for concurrent in-flight requests.
                    budgets[endpoint]=min(budgets[endpoint],left)
                return
            except Exception:
                stop.set()
                raise
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        list(pool.map(one,jobs))
    print(f'LIVE complete: {len(jobs)} new answers; parallel={PARALLEL}')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--dry-run',action='store_true')
    modes.add_argument('--mock',action='store_true')
    modes.add_argument('--live',action='store_true')
    parser.add_argument('--i-understand-budget',action='store_true')
    parser.add_argument('--target',choices=['refs','scored'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--server-url',default='http://34.147.229.55')
    args=parser.parse_args(argv)
    if args.live and (not args.i_understand_budget or args.target is None):
        parser.error('--live requires --i-understand-budget AND --target refs|scored')
    target=args.target or 'scored'
    mode='live' if args.live else 'mock' if args.mock else 'dry-run'
    if args.live and not os.environ.get('MODEL_EQUALITY_API_KEY'):
        parser.error('--live requires MODEL_EQUALITY_API_KEY in the environment')
    if not args.out.parent.is_dir():
        parser.error('--out parent directory must already exist')
    args.out.mkdir(exist_ok=True)
    # Lock a campaign directory against simultaneous collectors.
    with (args.out/'.collect.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prompts,manifest=make_protocol()
        endpoints=[f'ref-{i:02}' for i in range(1,10)] if target=='refs' else [f'ep-{i:02}' for i in range(1,34)]
        encoded=''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in prompts)
        save_fixed(args.out/'prompts.jsonl',encoded)
        save_fixed(args.out/'manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        plan=dict(target=target,server_url=args.server_url.rstrip('/'),protocol_sha256=hashlib.sha256(encoded.encode()).hexdigest(),endpoints=endpoints,max_tokens=MAX_TOKENS)
        save_fixed(args.out/'plan.json',json.dumps(plan,indent=2)+'\n')
        if not args.dry_run and mode!='dry-run':
            save_fixed(args.out/'run_mode.json',json.dumps(dict(mode=mode))+'\n')
        print(f'Mode={mode}; target={target}; endpoints={len(endpoints)}; parallel={PARALLEL}; max_tokens={MAX_TOKENS}')
        print('Per endpoint: 4 flags + 30 long + 62 bursts x 22 items (2 exact repeats) + 4 reserve = 100 requests')
        print(f'Total planned answers={len(endpoints)*100}; burst item responses per endpoint=1364')
        print(f'Protocol: {args.out / "prompts.jsonl"}; manifest: {args.out / "manifest.json"}')
        if args.live:
            live(args.out,prompts,endpoints,args.server_url,os.environ['MODEL_EQUALITY_API_KEY'])
        elif args.mock:
            mock(args.out,prompts,manifest,endpoints,target)
        else:
            print('DRY-RUN: HTTP requests=0; no answers sent or budget consumed')


if __name__=='__main__':
    main()
