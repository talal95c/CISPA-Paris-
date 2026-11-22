#!/bin/bash
# runq.sh threads name model data [extra args]  -> wb/name.jsonl
cd /home/ubuntu/CISPA-Paris-/task2/loop
export OMP_NUM_THREADS=$1; n=$2; m=$3; d=$4; shift 4
/home/ubuntu/ref-evaluation/met-paper-env/bin/python score_wb.py --model "$m" --data "$d" --out wb/$n.jsonl "$@" > wb/$n.log 2>&1
