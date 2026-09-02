"""Peak memory per arm, each measured in its own process.

MLX reports peak memory as a process-wide high-water mark, so loading a large arm first
contaminates every later measurement in the same process. Peak memory is not thermally
sensitive, so it does not need the interleaved protocol; process isolation is what it needs.
"""
import subprocess, sys, csv, json

ARMS = ["A-bf16","B-q8-g64","C-q4-g64","D-q4-g32","E-q4-emb8"]

CHILD = r'''
import sys, json
import mlx.core as mx
from mlx_lm import load, stream_generate
arm = sys.argv[1]
prompts = json.load(open("bench/prompts.json"))
mx.reset_peak_memory()
model, tok = load(f"models/{arm}")
after_load = mx.get_peak_memory() / 1e9
msgs = [{"role":"user","content": prompts["long"]}]
text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
for r in stream_generate(model, tok, text, max_tokens=256):
    last = r
print(json.dumps({"arm": arm, "peak_after_load_gb": round(after_load,3),
                  "peak_total_gb": round(mx.get_peak_memory()/1e9,3)}))
'''

rows = []
for a in ARMS:
    out = subprocess.run([sys.executable, "-c", CHILD, a], capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr[-600:]); sys.exit(1)
    r = json.loads(out.stdout.strip().splitlines()[-1]); rows.append(r)
    print(f"  {a:12} weights {r['peak_after_load_gb']:6.2f} GB   "
          f"peak with 2048-token context {r['peak_total_gb']:6.2f} GB")

with open("results/peak_memory.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["arm","peak_after_load_gb","peak_total_gb"])
    w.writeheader(); w.writerows(rows)
print("\n-> results/peak_memory.csv")
