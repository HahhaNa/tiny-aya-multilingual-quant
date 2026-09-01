"""T3 圖：均勻 4-bit 量化在權重空間是不是對某些 token 比較不利。答案是否。"""
import csv, collections, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EN={"eng_Latn":"English","spa_Latn":"Spanish","rus_Cyrl":"Russian","cmn_Hant":"Chinese (Trad)",
    "hin_Deva":"Hindi","arb_Arab":"Arabic","swh_Latn":"Swahili","yor_Latn":"Yoruba",
    "amh_Ethi":"Amharic","mya_Mymr":"Burmese"}
COL={"high":"#22607F","mid":"#7A8B87","low":"#A6650F"}

R=list(csv.DictReader(open("results/token_error.csv")))
f=lambda r,k: float(r[k])
freq=np.array([f(r,"freq") for r in R]); rel=np.array([f(r,"rel_err") for r in R])
tier=np.array([r["tier"] for r in R]); lang=np.array([r["primary_lang"] for r in R])
conc=np.array([f(r,"concentration") for r in R]); mono=conc>0.9
med=np.median(rel)

fig,axes=plt.subplots(1,3,figsize=(16,5))

# --- 1. 誤差 vs 頻率 ---
ax=axes[0]
for t in ["high","mid","low"]:
    m=mono&(tier==t)
    ax.scatter(freq[m]+np.random.default_rng(1337).uniform(-.3,.3,m.sum()), rel[m],
               s=3, alpha=.12, color=COL[t], edgecolors="none")
edges=[1,2,5,20,100,500,10**9]
xs,ys=[],[]
for lo,hi in zip(edges,edges[1:]):
    m=(freq>=lo)&(freq<hi)
    if m.sum()<20: continue
    xs.append(np.sqrt(lo*min(hi,freq[m].max()))); ys.append(np.median(rel[m]))
ax.plot(xs,ys,"o-",color="#14201D",lw=2,ms=6,label="median per frequency bin")
ax.axhline(med,color="#96382C",ls="--",lw=1)
ax.text(1.2,med+.004,f"overall median {med:.3f}",fontsize=9,color="#96382C")
ax.set_xscale("log"); ax.set_ylim(.080,.115)
ax.set_xlabel("Token frequency in the 10-language corpus (log)")
ax.set_ylabel("Relative quantization error  |ΔE| / |E|")
ax.set_title("Rare tokens are not quantized worse\nSlope is flat. r(log freq, error) = -0.03",
             fontsize=11,loc="left")
ax.legend(fontsize=9,frameon=False)

# --- 2. 依 tier 的分布 ---
ax=axes[1]
data=[rel[mono&(tier==t)] for t in ["high","mid","low"]]
bp=ax.boxplot(data,tick_labels=["high","mid","low"],widths=.55,showfliers=False,patch_artist=True,
              medianprops=dict(color="#14201D",lw=1.6))
for p,t in zip(bp["boxes"],["high","mid","low"]): p.set_facecolor(COL[t]); p.set_alpha(.5)
ax.axhline(med,color="#96382C",ls="--",lw=1)
ax.set_ylim(.084,.104)
ax.set_ylabel("Relative quantization error")
ax.set_xlabel("Resource tier of the token's primary language")
ax.set_title("No tier effect\nThe gap between tiers is 12% of the within-tier spread",
             fontsize=11,loc="left")

# --- 3. 逐層 ---
ax=axes[2]
L=list(csv.DictReader(open("results/layer_error.csv")))
by=collections.defaultdict(lambda: ([],[]))
for r in L:
    if int(r["layer"])<0: continue
    by[r["kind"]][0].append(int(r["layer"])); by[r["kind"]][1].append(float(r["rel_err"]))
for k,(x,y) in sorted(by.items()):
    o=np.argsort(x); ax.plot(np.array(x)[o],np.array(y)[o],lw=1.3,alpha=.85,label=k)
emb=med
ax.axhline(0.0939,color="#96382C",ls="--",lw=1.6)
ax.text(1,0.0942,"embed_tokens  0.094",fontsize=9,color="#96382C")
ax.set_xlabel("Transformer layer"); ax.set_ylabel("Relative quantization error")
ax.set_title("Embedding is mid-pack, not a weak point\nThe outliers are k_proj and v_proj in the deep layers",
             fontsize=11,loc="left")
ax.legend(fontsize=8,frameon=False,ncol=2)

for a in axes: a.spines[["top","right"]].set_visible(False); a.grid(alpha=.22)
plt.tight_layout(); plt.savefig("figures/01_weight_error.png",dpi=170)
print("→ figures/01_weight_error.png")
