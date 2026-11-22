"""Append-only sampler for the scored endpoints (ep-*), with a hard per-endpoint budget cap checked against the server.
Key is read from the environment only. Usage: refcollect.py probes.json OUT.jsonl --repeats N [--workers 2]"""
import argparse, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import requests

URL = 'http://34.147.229.55/33-model-equality/v1/chat/completions'
ap = argparse.ArgumentParser()
ap.add_argument('probes'); ap.add_argument('out')
ap.add_argument('--repeats', type=int, default=10)
ap.add_argument('--workers', type=int, default=2)
ap.add_argument('--endpoints', default=','.join(f'ep-{i:02d}' for i in range(1, 34)))
ap.add_argument('--minutes', type=float, default=60)
ap.add_argument('--cap', type=int, required=True, help='never let an endpoint exceed this many used responses')
a = ap.parse_args()
eps = a.endpoints.split(',')
assert all(e.startswith('ep-') for e in eps), 'scored endpoints only'
assert 0 < a.cap <= 100
bud = {m['id']: m['budget']['used'] for m in requests.get(URL.replace('chat/completions', 'models'), headers={'X-API-Key': os.environ['SPRINTML_API_KEY']}, timeout=60).json()['data']}
assert 1 <= a.workers <= 3
probes = json.load(open(a.probes))
done = set()
if os.path.exists(a.out):
    for l in open(a.out):
        x = json.loads(l); done.add((x['endpoint'], x['id'], x['repeat']))
jobs = [(e, p, r) for r in range(a.repeats) for p in probes for e in eps if (e, p['id'], r) not in done]
import collections
plan = collections.Counter(j[0] for j in jobs)
over = {e: bud[e] + n_ for e, n_ in plan.items() if bud[e] + n_ > a.cap}
assert not over, f'budget cap exceeded: {over}'
print(f'{len(jobs)} to query, {len(done)} cached, max used after = {max(bud[e] + plan[e] for e in plan) if plan else 0}', flush=True)
lock = threading.Lock(); t0 = time.monotonic(); n = [0, 0]
s = requests.Session(); s.headers['X-API-Key'] = os.environ['SPRINTML_API_KEY']
out = open(a.out, 'a')

def one(job):
    e, p, r = job
    while time.monotonic() - t0 < a.minutes * 60:
        t = time.monotonic()
        try:
            resp = s.post(URL, json={'model': e, 'max_tokens': p.get('max_tokens', 256),
                                     'messages': [{'role': 'user', 'content': p['prompt']}]}, timeout=180)
        except requests.RequestException:
            time.sleep(10); continue
        dt = time.monotonic() - t
        if resp.status_code == 200:
            b = resp.json()
            with lock:
                out.write(json.dumps(dict(endpoint=e, id=p['id'], repeat=r, seconds=round(dt, 3), ts=time.time(),
                                          response=b['choices'][0]['message']['content'],
                                          finish=b['choices'][0].get('finish_reason'), usage=b.get('usage')),
                                     ensure_ascii=False) + '\n'); out.flush()
                n[0] += 1
                if n[0] % 25 == 0: print(f'{n[0]}/{len(jobs)} ok, {n[1]} busy, {time.monotonic()-t0:.0f}s', flush=True)
            return
        with lock: n[1] += 1
        if resp.status_code == 429:
            time.sleep(float(resp.headers.get('Retry-After', 2)) + 0.5 * (hash(job[0]) % 3)); continue
        print('HTTP', resp.status_code, resp.text[:200], flush=True)
        if resp.status_code in (401, 403): os._exit(3)
        time.sleep(5)

with ThreadPoolExecutor(a.workers) as ex: list(ex.map(one, jobs))
print(f'done {n[0]} ok in {time.monotonic()-t0:.0f}s', flush=True)
