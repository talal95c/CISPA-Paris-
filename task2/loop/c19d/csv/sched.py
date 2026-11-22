"""sched.py HH:MM:SS(UTC) file.csv [...] : filet de securite. Un seul essai par creneau, aucun reessai, rien apres 10:59:30 UTC."""
import os,sys,json,time,re,requests,calendar
day=time.strftime('%Y-%m-%d ',time.gmtime())
ts=lambda hms: calendar.timegm(time.strptime(day+hms,'%Y-%m-%d %H:%M:%S'))
END=ts('10:59:30');last=0;a=sys.argv[1:]
for hms,p in zip(a[::2],a[1::2]):
    t=max(ts(hms),last)
    if t>END: print('trop tard, abandon',p,flush=True);break
    while time.time()<t: time.sleep(0.5)
    try:
        r=requests.post('http://34.147.229.55/submit/33-model-equality',headers={'X-API-Key':os.environ['SPRINTML_API_KEY']},files={'file':(os.path.basename(p),open(p,'rb'))},timeout=(60,300))
        try: body=r.json()
        except ValueError: body={'text':r.text[:300]}
        rec=dict(ts=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),file=p,http=r.status_code,body=body)
    except Exception as ex: rec=dict(ts=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),file=p,http=None,body=repr(ex)[:300])
    open('submissions.jsonl','a').write(json.dumps(rec)+'\n');print(json.dumps(rec)[:600],flush=True)
    m=re.search(r'Wait (\d+) seconds',json.dumps(rec['body']))
    if rec['http']==200: last=time.time()+1808
    elif m: last=time.time()+int(m.group(1))+19
