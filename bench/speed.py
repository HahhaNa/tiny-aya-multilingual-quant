"""Speed measurement under the protocol in METHODOLOGY section 1.2.

Interleaved under a fixed seed, 90 seconds of forced cooldown between runs, one row per run with
no aggregation, so medians and IQRs are computed downstream and order_idx stays testable.
"""
import csv, json, time, gc, sys
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, stream_generate

COOLDOWN = 90
MAX_TOKENS = 256
OUT = Path("results/speed.csv")

prompts = json.load(open("bench/prompts.json"))
order = list(csv.DictReader(open("bench/run_order.csv")))

done = set()
if OUT.exists():
    done = {r["order_idx"] for r in csv.DictReader(open(OUT))}
    print(f"resuming, {len(done)} runs already recorded")
else:
    with open(OUT, "w", newline="") as f:
        csv.writer(f).writerow(["order_idx","arm","prompt_len","rep","prompt_tokens",
                                "prefill_tps","decode_tps","ttft_ms","gen_tokens",
                                "peak_gb","wall_start_s"])

t_zero = time.time()
loaded, model, tok = None, None, None
for row in order:
    if row["order_idx"] in done: continue
    arm = row["arm"]
    if loaded != arm:
        del model, tok; gc.collect(); mx.clear_cache()
        model, tok = load(f"models/{arm}"); loaded = arm
    msgs = [{"role":"user","content": prompts[row["prompt_len"]]}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    n_prompt = len(tok.encode(text, add_special_tokens=False))

    mx.clear_cache()
    mx.reset_peak_memory()   # process-wide high-water mark; without this it never falls back down
    wall = time.time() - t_zero
    t0 = time.time(); ttft = None; last = None; n = 0
    for resp in stream_generate(model, tok, text, max_tokens=MAX_TOKENS):
        if ttft is None: ttft = (time.time() - t0) * 1000
        last = resp; n += 1

    with open(OUT, "a", newline="") as f:
        csv.writer(f).writerow([row["order_idx"], arm, row["prompt_len"], row["rep"], n_prompt,
                                round(last.prompt_tps,2), round(last.generation_tps,2),
                                round(ttft,1), n, round(last.peak_memory,3), round(wall,1)])
    print(f"[{row['order_idx']:>2}] {arm:11} {row['prompt_len']:6} rep{row['rep']}  "
          f"prefill {last.prompt_tps:7.1f}  decode {last.generation_tps:6.2f} tok/s  "
          f"ttft {ttft:6.0f}ms  peak {last.peak_memory:.2f}GB", flush=True)
    time.sleep(COOLDOWN)

print("\ndone ->", OUT)
