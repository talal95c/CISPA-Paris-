#!/bin/bash
# usage: ./dec_run.sh NAME MODEL THREADS DATA [ENDPOINTS]
cd "$(dirname "$0")"; mkdir -p dec
E=""; [ -n "$5" ] && E="--endpoints $5"
exec ../../../env/bin/python dec_fit.py --model "$2" --data "$4" --out dec/$1.npz --threads $3 $E > dec/$1.log 2>&1
