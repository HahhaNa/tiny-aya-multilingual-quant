#!/usr/bin/env bash
# Arms A to D. Arm E lives in convert/arm_e.py because it needs a quant_predicate.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=CohereLabs/tiny-aya-global

run() { # run <arm> <extra flags...>
  local arm=$1; shift
  if [ -d "models/$arm" ]; then echo "[skip] models/$arm already exists"; return; fi
  echo "=== $arm ==="
  mlx_lm.convert --hf-path "$M" --mlx-path "models/$arm" "$@"
}

run A-bf16    --dtype bfloat16
run B-q8-g64  -q --q-bits 8 --q-group-size 64
run C-q4-g64  -q --q-bits 4 --q-group-size 64
run D-q4-g32  -q --q-bits 4 --q-group-size 32

echo; echo "=== size on disk (predicted 6.70 / 3.56 / 1.88 / 2.09 GB) ==="
du -sh models/*
