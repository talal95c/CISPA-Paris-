#!/bin/bash
cd /home/ubuntu/CISPA-Paris-/task2/loop
for v in nf4 int8; do for d in short long; do
  ./runq.sh 2 c_${d}_$v /home/ubuntu/models/unsloth__Llama-3.2-1B-Instruct data/ep_$d.jsonl --endpoints ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31 --variant $v
done; done
