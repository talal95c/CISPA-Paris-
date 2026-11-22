#!/bin/bash
# unigram fingerprint for the three scored families (resumable). Usage: ./uni_run.sh
P=/home/ubuntu/ref-evaluation/met-paper-env/bin/python; D=data/ep_short.jsonl,data/ep_long.jsonl,data/cache_all.jsonl
A=ep-01,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33
B=ep-02,ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32
C=ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31
OMP_NUM_THREADS=3 nohup $P score_uni.py --model /home/ubuntu/models/unsloth__Llama-3.2-1B-Instruct --data $D --out uni/c --endpoints $C > uni/c.log 2>&1 &
OMP_NUM_THREADS=3 nohup $P score_uni.py --model /home/ubuntu/models/Qwen__Qwen2.5-1.5B-Instruct --data $D --out uni/b --endpoints $B > uni/b.log 2>&1 &
OMP_NUM_THREADS=2 nohup $P score_uni.py --model /home/ubuntu/models/Qwen__Qwen2.5-3B-Instruct --data $D --out uni/a --endpoints $A > uni/a.log 2>&1 &
