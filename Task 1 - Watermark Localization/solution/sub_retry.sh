#!/usr/bin/env bash
# sub_retry.sh file.pt : submit, on 429 wait the indicated time and retry (max 4 tries)
for t in 1 2 3 4; do
  o=$(python submit.py "$1"); echo "$o"
  w=$(echo "$o" | grep -o "Wait [0-9]* seconds" | grep -o "[0-9]*")
  [ -z "$w" ] && break
  sleep $((w+3))
done
