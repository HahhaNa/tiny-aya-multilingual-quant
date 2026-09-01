#!/usr/bin/env bash
# T2 · arm A–D。arm E 見 convert/arm_e.py（需要 quant_predicate，bash 做不到）
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
M=CohereLabs/tiny-aya-global

run() { # run <arm> <額外參數...>
  local arm=$1; shift
  if [ -d "models/$arm" ]; then echo "[skip] models/$arm 已存在"; return; fi
  echo "=== $arm ==="
  mlx_lm.convert --hf-path "$M" --mlx-path "models/$arm" "$@"
}

run A-bf16    --dtype bfloat16
run B-q8-g64  -q --q-bits 8 --q-group-size 64
run C-q4-g64  -q --q-bits 4 --q-group-size 64
run D-q4-g32  -q --q-bits 4 --q-group-size 32

echo; echo "=== 實際大小（對照預估 6.70 / 3.56 / 1.88 / 2.09 GB）==="
du -sh models/*
