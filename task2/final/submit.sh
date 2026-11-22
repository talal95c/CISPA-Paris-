#!/bin/bash
# usage: TK=... ./submit.sh sub/x.csv [epoch_not_before]
cd /home/ubuntu/t2
[ -n "$2" ] && { n=$(date +%s); [ "$2" -gt "$n" ] && sleep $(( $2 - n )); }
for i in 1 2 3 4; do
  out=$(curl -s -m 600 -X POST -H "X-API-Key: $TK" -F "file=@$1" http://34.147.229.55/submit/33-model-equality -w " HTTP %{http_code}")
  echo "$(date -u +%FT%TZ) $1 $out" | tee -a sub/submissions.log
  w=$(echo "$out" | grep -oE "Wait [0-9]+ seconds" | grep -oE "[0-9]+")
  [ -z "$w" ] && break
  sleep $((w+2))
done
