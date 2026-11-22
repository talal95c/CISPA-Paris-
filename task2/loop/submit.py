"""Submit a CSV to the task server (key from SPRINTML_API_KEY) and append the reply to sub/submissions.jsonl.
Usage: submit.py file.csv"""
import os, sys, json, time, requests
p = sys.argv[1]
r = requests.post('http://34.147.229.55/submit/33-model-equality', headers={'X-API-Key': os.environ['SPRINTML_API_KEY']},
                  files={'file': (os.path.basename(p), open(p, 'rb'))}, timeout=(600, 600))
try: body = r.json()
except ValueError: body = {'text': r.text[:300]}
rec = dict(ts=time.strftime('%Y-%m-%d %H:%M:%S'), file=p, http=r.status_code, body=body)
open('sub/submissions.jsonl', 'a').write(json.dumps(rec) + '\n'); print(json.dumps(rec)[:600])
