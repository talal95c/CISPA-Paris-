#!/bin/bash
# ident.sh name modeldir : score a small probe set under one candidate checkpoint
cd /home/ubuntu/CISPA-Paris-/task2/loop
/home/ubuntu/ref-evaluation/met-paper-env/bin/python score_wb.py --model "$2" --data data/cache_all.jsonl --out wb/id_$1.jsonl \
  --endpoints ep-04,ep-12,ep-03,ep-06,ep-09,ep-05,ref-01,ref-05 --ids L02,L05,L07,L08 > wb/id_$1.log 2>&1
