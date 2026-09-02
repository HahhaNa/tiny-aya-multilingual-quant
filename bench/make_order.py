"""Generate the interleaved execution order for the speed measurement.

The order is written to disk and committed before the run, so that the shuffle cannot be
re-rolled after seeing results. order_idx travels into results/speed.csv, which is what makes
the residual order effect testable afterwards (METHODOLOGY section 1.3).
"""
import csv, random, json
from transformers import AutoTokenizer

ARMS = ["A-bf16", "B-q8-g64", "C-q4-g64", "D-q4-g32", "E-q4-emb8"]
REPS = 3
SEED = 1337
TARGETS = {"short": 128, "medium": 512, "long": 2048}

# Prompts are built from FLORES English text so the prefill lengths are realistic and reproducible.
rows = [json.loads(l)["eng_Latn"] for l in open("data/flores_10lang.jsonl", encoding="utf-8")]
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")

prompts = {}
for name, target in TARGETS.items():
    text, i = "", 0
    while len(tok.encode(text, add_special_tokens=False)) < target and i < len(rows):
        text += rows[i] + " "; i += 1
    ids = tok.encode(text, add_special_tokens=False)[:target]
    prompts[name] = tok.decode(ids)
    print(f"  {name:8} {len(ids)} tokens")
json.dump(prompts, open("bench/prompts.json", "w"), ensure_ascii=False, indent=1)

runs = [(a, p, r) for a in ARMS for p in TARGETS for r in range(REPS)]
random.Random(SEED).shuffle(runs)
with open("bench/run_order.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["order_idx", "arm", "prompt_len", "rep"])
    for i, (a, p, r) in enumerate(runs):
        w.writerow([i, a, p, r])

print(f"\n{len(runs)} runs written to bench/run_order.csv")
print("first ten arms in execution order:", [r[0].split('-')[0] for r in runs[:10]])
