#!/bin/bash
# lane8.sh a|b|c : keep scoring new long answers (base then fp8, resumable) until data/ep_long2.log stops growing and all is scored
cd /home/ubuntu/CISPA-Paris-/task2/loop
case $1 in
a) th=4; m=/home/ubuntu/models/Qwen__Qwen2.5-3B-Instruct; e=ep-01,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33;;
b) th=2; m=/home/ubuntu/models/Qwen__Qwen2.5-1.5B-Instruct; e=ep-02,ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32;;
c) th=2; m=/home/ubuntu/models/unsloth__Llama-3.2-1B-Instruct; e=ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31;;
esac
while true; do
  n0=$(wc -l < data/ep_long.jsonl)
  ./runq.sh $th $1_long $m data/ep_long.jsonl --endpoints $e
  ./runq.sh $th $1_long_fp8 $m data/ep_long.jsonl --endpoints $e --variant fp8
  [ "$(wc -l < data/ep_long.jsonl)" = "$n0" ] && ! pgrep -f epcollect.py > /dev/null && break
  sleep 20
done
echo finished > wb/lane8_$1.done
