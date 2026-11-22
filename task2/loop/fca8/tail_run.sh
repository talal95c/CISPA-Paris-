#!/bin/bash
# usage: ./tail_run.sh NAME MODEL THREADS DATA [ENDPOINTS]
cd "$(dirname "$0")"; mkdir -p tail
E=""; [ -n "$5" ] && E="--endpoints $5"
exec ../../../env/bin/python tail_fit.py --model "$2" --data "$4" --out tail/$1.npz --threads $3 $E > tail/$1.log 2>&1
