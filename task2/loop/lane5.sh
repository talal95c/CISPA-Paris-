#!/bin/bash
cd /home/ubuntu/CISPA-Paris-/task2/loop
w(){ while [ -z "$(grep -s done wb/$1.log)" ]; do sleep 5; done; }
w c_cache; w c_long_fp8
./runq.sh 3 b_long_fp8 /home/ubuntu/models/Qwen__Qwen2.5-1.5B-Instruct data/ep_long.jsonl --endpoints ep-02,ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32 --variant fp8
w a_cache
./runq.sh 4 a_long_fp8 /home/ubuntu/models/Qwen__Qwen2.5-3B-Instruct data/ep_long.jsonl --endpoints ep-01,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33 --variant fp8
