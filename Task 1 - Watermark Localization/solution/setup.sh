#!/usr/bin/env bash
# Task 1 setup on a CPU box (~4 min). Usage: bash task1/loop/setup.sh [/abs/workdir]   (default ~/t1)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
W="${1:-$HOME/t1}"; mkdir -p "$W"; cd "$W"
B=https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task_Paris/resolve/8d27b2c9cc37c4829c8501e43e11fcb1b5db58c0
for f in starter_kit.zip train.zip test.zip; do [ -f $f ] || curl -sL -o $f $B/$f & done; wait
[ -d starter ] || unzip -oq starter_kit.zip -d starter
[ -d release/test ] || { mkdir -p release; unzip -oq train.zip -d release; unzip -oq test.zip -d release; }
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -q numpy pillow onnxruntime onnx trustmark==0.8.0 omegaconf einops timm opencv-python-headless scipy safetensors requests pyyaml tqdm scikit-learn scikit-image lightgbm av decord pycocotools
(cd starter && python download_models.py)
[ -e loop ] || ln -s "$HERE" loop
echo "ready: cd $W/loop && . ../.venv/bin/activate && python exp0.py"
