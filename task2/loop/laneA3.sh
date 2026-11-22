#!/bin/bash
# A: system prompt of ep-29 on the disguised endpoints (long) and on every A endpoint (short): find the sysprompt endpoint of A
cd /home/ubuntu/CISPA-Paris-/task2/loop
S="You are a helpful assistant for general questions and everyday tasks. Ask a clarifying question when the request is ambiguous. When you give an answer, say briefly why."
m=/home/ubuntu/models/Qwen__Qwen2.5-3B-Instruct
./runq.sh 5 a_long_sys2 $m data/ep_long.jsonl --endpoints ep-14,ep-23,ep-33 --system "$S"
./runq.sh 5 a_short_sys $m data/ep_short.jsonl --endpoints ep-14,ep-23,ep-33,ep-01,ep-04,ep-12,ep-17,ep-18,ep-24,ep-25 --system "$S"
echo done > wb/laneA3.done
