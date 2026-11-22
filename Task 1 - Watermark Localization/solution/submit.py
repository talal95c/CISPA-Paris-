import os, sys, requests, json, time
p=sys.argv[1]
r=requests.post("http://34.147.229.55/submit/34-watermark-localisation-image",headers={"X-API-Key":os.environ["K"]},files={"file":(os.path.basename(p),open(p,"rb"),"application/octet-stream")},timeout=(10,180))
print(time.strftime("%H:%M:%S"),p,r.status_code,r.text[:600])
open("submissions.log","a").write("%s %s %s %s\n"%(time.strftime("%F %T"),p,r.status_code,r.text[:600].replace("\n"," ")))
