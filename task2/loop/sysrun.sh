#!/bin/bash
# score short answers under the system prompt leaked by ep-29, for all three families (to find the sysprompt endpoint of A and B)
cd /home/ubuntu/CISPA-Paris-/task2/loop
S="You are a helpful assistant for general questions and everyday tasks. Ask a clarifying question when the request is ambiguous. When you give an answer, say briefly why."
./runq.sh 2 c_short_sys /home/ubuntu/models/unsloth__Llama-3.2-1B-Instruct data/ep_short.jsonl --endpoints ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31 --system "$S"
./runq.sh 2 b_short_sys /home/ubuntu/models/Qwen__Qwen2.5-1.5B-Instruct data/ep_short.jsonl --endpoints ep-02,ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32 --system "$S"
./runq.sh 2 a_short_sys /home/ubuntu/models/Qwen__Qwen2.5-3B-Instruct data/ep_short.jsonl --endpoints ep-01,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33 --system "$S"
echo done > wb/sysrun.done
