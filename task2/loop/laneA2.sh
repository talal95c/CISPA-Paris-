#!/bin/bash
# after lane8 a: long answers of A under ep-29's system prompt (find the sysprompt endpoint of A), then int8 / nf4 simulated quantizations
cd /home/ubuntu/CISPA-Paris-/task2/loop
while [ ! -f wb/lane8_a.done ]; do sleep 20; done
S="You are a helpful assistant for general questions and everyday tasks. Ask a clarifying question when the request is ambiguous. When you give an answer, say briefly why."
m=/home/ubuntu/models/Qwen__Qwen2.5-3B-Instruct; e=ep-01,ep-12,ep-17,ep-18,ep-24,ep-25,ep-04
./runq.sh 4 a_long_sys $m data/ep_long.jsonl --endpoints $e --system "$S"
./runq.sh 4 a_long_int8 $m data/ep_long.jsonl --endpoints $e --variant int8
./runq.sh 4 a_long_nf4 $m data/ep_long.jsonl --endpoints $e --variant nf4
echo done > wb/laneA2.done
