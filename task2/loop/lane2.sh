#!/bin/bash
cd /home/ubuntu/CISPA-Paris-/task2/loop
while [ -z "$(grep -s done wb/b_q15_fp8.log)" ]; do sleep 5; done
./runq.sh 4 a_q3 /home/ubuntu/models/Qwen__Qwen2.5-3B-Instruct data/cache_all.jsonl --endpoints ep-01,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33
./runq.sh 4 b_q2_15 /home/ubuntu/models/Qwen__Qwen2-1.5B-Instruct data/cache_all.jsonl --endpoints ep-03,ep-06,ep-22,ep-30
./runq.sh 4 a_q3_fp8 /home/ubuntu/models/Qwen__Qwen2.5-3B-Instruct data/cache_all.jsonl --endpoints ep-01,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33 --variant fp8
