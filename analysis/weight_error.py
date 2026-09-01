"""Quantization error measured directly in weight space. Needs no inference.

Question: does uniform 4-bit quantization put more error on the embedding rows of rare or low
resource tokens? If it did, that would be direct evidence for the mitigation hypothesis, obtainable
without running a single evaluation.
"""
import glob, json, csv, collections
import numpy as np
import mlx.core as mx
from transformers import AutoTokenizer

GROUP, BITS = 64, 4
CHUNK = 16384

def load_arm(arm):
    d = {}
    for f in sorted(glob.glob(f"models/{arm}/*.safetensors")):
        d.update(mx.load(f))
    return d

def dequant(w, scales, biases):
    return mx.dequantize(w, scales=scales, biases=biases, group_size=GROUP, bits=BITS)

# ---------- 1. token frequency and language attribution ----------
print("=== 1. attributing tokens to languages ===")
meta = {r["flores_code"]: r for r in csv.DictReader(open("data/lang_meta.csv"))}
LANGS = list(meta)
rows = [json.loads(l) for l in open("data/flores_10lang.jsonl", encoding="utf-8")]
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")

cnt = {L: collections.Counter() for L in LANGS}
for r in rows:
    for L in LANGS:
        cnt[L].update(tok.encode(r[L], add_special_tokens=False))

total = collections.Counter()
for L in LANGS: total.update(cnt[L])
print(f"  corpus covers {len(total)} distinct tokens of 262144 ({len(total)/262144:.1%})")

# A token's primary language is where its share is highest, normalised by each language's total
# token count so that differing corpus lengths do not bias the attribution.
lang_tot = {L: sum(cnt[L].values()) for L in LANGS}
prim, conc = {}, {}
for t in total:
    share = {L: cnt[L][t] / lang_tot[L] for L in LANGS if cnt[L][t]}
    s = sum(share.values())
    best = max(share, key=share.get)
    prim[t] = best
    conc[t] = share[best] / s          # 1.0 means the token appears in exactly one language
print(f"  effectively monolingual tokens (concentration > 0.9): {sum(1 for t in conc if conc[t]>0.9)}")

# ---------- 2. per-token embedding quantization error ----------
print("\n=== 2. per-token embedding error, arm A against arm C ===")
A = load_arm("A-bf16"); C = load_arm("C-q4-g64")
a_emb = A["model.embed_tokens.weight"]
c_w, c_s, c_b = (C["model.embed_tokens.weight"], C["model.embed_tokens.scales"], C["model.embed_tokens.biases"])
V = a_emb.shape[0]
mse = np.zeros(V, dtype=np.float32); rel = np.zeros(V, dtype=np.float32); nrm = np.zeros(V, dtype=np.float32)
for i in range(0, V, CHUNK):
    j = min(i + CHUNK, V)
    a = a_emb[i:j].astype(mx.float32)
    d = dequant(c_w[i:j], c_s[i:j], c_b[i:j]).astype(mx.float32)
    e = a - d
    mse[i:j] = np.array(mx.mean(e * e, axis=1))
    n = mx.sqrt(mx.sum(a * a, axis=1))
    rel[i:j] = np.array(mx.sqrt(mx.sum(e * e, axis=1)) / mx.maximum(n, 1e-9))
    nrm[i:j] = np.array(n)
    mx.clear_cache()
print(f"  whole vocabulary, relative error: median {np.median(rel):.4f}  mean {rel.mean():.4f}")

with open("results/token_error.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["token_id","freq","primary_lang","tier","concentration","mse","rel_err","row_norm"])
    for t in sorted(total, key=lambda x: -total[x]):
        w.writerow([t, total[t], prim[t], meta[prim[t]]["tier"], round(conc[t],3),
                    f"{mse[t]:.6g}", f"{rel[t]:.6g}", f"{nrm[t]:.6g}"])

# Tokens absent from the corpus are the genuine low frequency tail, used as a control
seen = np.zeros(V, bool); seen[list(total)] = True
print(f"  tokens seen in corpus, median relative error {np.median(rel[seen]):.4f}")
print(f"  tokens never seen, median relative error {np.median(rel[~seen]):.4f}  <- the real tail")

print(f"\n  {'tier':8}{'tokens':>9}{'median rel err':>16}{'median row norm':>17}")
for tier in ["high","mid","low"]:
    ids = [t for t in total if meta[prim[t]]["tier"] == tier]
    print(f"  {tier:8}{len(ids):9}{np.median(rel[ids]):16.4f}{np.median(nrm[ids]):17.4f}")

# ---------- 3. layer-wise error ----------
print("\n=== 3. quantization error by transformer module ===")
out = []
for k in A:
    if not k.endswith(".weight") or "embed" in k or "norm" in k: continue
    if k not in C: continue
    a = A[k].astype(mx.float32)
    d = dequant(C[k], C[k[:-7]+".scales"], C[k[:-7]+".biases"]).astype(mx.float32)
    e = a - d
    r = float(mx.sqrt(mx.sum(e*e)) / mx.sqrt(mx.sum(a*a)))
    parts = k.split(".")
    out.append(dict(tensor=k, layer=int(parts[2]) if parts[1]=="layers" else -1,
                    kind=parts[-2], rel_err=round(r,6)))
    mx.clear_cache()
with open("results/layer_error.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["tensor","layer","kind","rel_err"]); w.writeheader(); w.writerows(out)
byk = collections.defaultdict(list)
for r in out: byk[r["kind"]].append(r["rel_err"])
print(f"  {'module':14}{'n':>4}{'median rel err':>16}")
for k,v in sorted(byk.items(), key=lambda x:-np.median(x[1])):
    print(f"  {k:14}{len(v):4}{np.median(v):16.4f}")
emb_rel = float(mx.sqrt(mx.sum((a_emb.astype(mx.float32)-dequant(c_w,c_s,c_b).astype(mx.float32))**2))
                / mx.sqrt(mx.sum(a_emb.astype(mx.float32)**2)))
print(f"  {'embed_tokens':14}{1:4}{emb_rel:16.4f}   <- for comparison")
print("\n→ results/token_error.csv, results/layer_error.csv")
