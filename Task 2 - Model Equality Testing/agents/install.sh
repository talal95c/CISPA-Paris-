#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
home="${HARNESS_HOME:-$PWD/.harness}"
mkdir -p "$home"
home="$(realpath "$home")"
checkout() {
  local name="$1" remote="$2" ref="$3" target="$home/$1"
  if [[ ! -d "$target/.git" ]]; then
    git clone --no-checkout --depth=1 "$remote" "$target"
    git -C "$target" fetch --depth=1 origin "$ref"
    git -C "$target" checkout --detach "$ref"
  fi
  [[ "$(git -C "$target" rev-parse HEAD)" == "$ref" ]] ||
    { echo "Unexpected revision in $name; use a new HARNESS_HOME." >&2; return 1; }
}
checkout FM-Agent https://github.com/baidubce/FM-Agent.git f3d8d7ab8e1847243aecf7b2f48cc37d9d81391a
checkout aideml https://github.com/WecoAI/aideml.git ad73d227af481bea3de06f884f41b2993dd6453a
for env in aide-env famou-env; do
  [[ -x "$home/$env/bin/python" ]] || uv venv --python 3.12 "$home/$env"
done
uv pip install -p "$home/famou-env/bin/python" famou-sdk==2.6.1
uv pip install -p "$home/aide-env/bin/python" -r aide-runtime.txt
uv pip install --no-deps -p "$home/aide-env/bin/python" "$home/aideml"
CMAKE_BUILD_PARALLEL_LEVEL=4 CMAKE_ARGS="-DGGML_CUDA=OFF" \
  uv pip install -p "$home/aide-env/bin/python" llama-cpp-python==0.3.16
"$home/famou-env/bin/famou-ctl" --version
"$home/aide-env/bin/python" -c 'from aide.agent import Agent; from llama_cpp import Llama'
