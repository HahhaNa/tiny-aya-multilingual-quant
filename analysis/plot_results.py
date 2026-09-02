"""Delta BPB by language, and the memory against fairness tradeoff."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EN={"eng_Latn":"English","spa_Latn":"Spanish","rus_Cyrl":"Russian","cmn_Hant":"Chinese (Trad)",
    "hin_Deva":"Hindi","arb_Arab":"Arabic","swh_Latn":"Swahili","yor_Latn":"Yoruba",
    "amh_Ethi":"Amharic","mya_Mymr":"Burmese"}
COL={"high":"#22607F","mid":"#7A8B87","low":"#A6650F"}
GB={"B-q8-g64":3.58,"C-q4-g64":1.91,"D-q4-g32":2.11,"E-q4-emb8":2.17}
TITLE={"C-q4-g64":"C · 4-bit, g=64","D-q4-g32":"D · 4-bit, g=32","E-q4-emb8":"E · 4-bit + 8-bit embedding"}

R=list(csv.DictReader(open("results/delta_bpb.csv")))
d={(r["arm"],r["flores_code"]):r for r in R}
LANGS=["eng_Latn","spa_Latn","rus_Cyrl","cmn_Hant","hin_Deva","arb_Arab","swh_Latn","yor_Latn","amh_Ethi","mya_Mymr"]
order=sorted(LANGS,key=lambda L:float(d[("C-q4-g64",L)]["rel_delta_pct"]))

fig,axes=plt.subplots(1,3,figsize=(16,5.4),sharey=True)
for ax,arm in zip(axes,["C-q4-g64","D-q4-g32","E-q4-emb8"]):
    y=np.arange(len(order))
    v=[float(d[(arm,L)]["rel_delta_pct"]) for L in order]
    lo=[v[i]-float(d[(arm,L)]["ci_low"]) for i,L in enumerate(order)]
    hi=[float(d[(arm,L)]["ci_high"])-v[i] for i,L in enumerate(order)]
    c=[COL[d[(arm,L)]["tier"]] for L in order]
    ax.barh(y,v,color=c,height=.66)
    ax.errorbar(v,y,xerr=[lo,hi],fmt="none",ecolor="#14201D",elinewidth=1.1,capsize=3)
    ax.axvline(0,color="#14201D",lw=1)
    ax.axvspan(-0.3,0.3,color="#8A9793",alpha=.16,zorder=0)
    ax.set_yticks(y); ax.set_yticklabels([EN[L] for L in order],fontsize=9.5)
    ax.set_xlim(-0.55,2.45); ax.set_xlabel("Relative ΔBPB against bf16 (%)")
    ax.set_title(f"{TITLE[arm]}\n{GB[arm]} GB", fontsize=11, loc="left")
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="x",alpha=.22)
axes[0].text(0.32,-0.75,"shaded band = numerical noise floor",fontsize=8.5,color="#55635F")
h=[plt.Rectangle((0,0),1,1,color=COL[t]) for t in ["high","mid","low"]]
axes[2].legend(h,["high resource","mid","low resource"],fontsize=9,frameon=False,loc="lower right")
plt.tight_layout(); plt.savefig("figures/03_delta_bpb.png",dpi=170); plt.close()

# ---- memory against fairness ----
G={r["arm"]:r for r in csv.DictReader(open("results/fairness_gap.csv"))}
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5.2))
arms=["B-q8-g64","C-q4-g64","D-q4-g32","E-q4-emb8"]
lab={"B-q8-g64":"B  8-bit","C-q4-g64":"C  4-bit g64","D-q4-g32":"D  4-bit g32","E-q4-emb8":"E  4-bit + 8-bit emb"}
x=[GB[a] for a in arms]; y=[float(G[a]["gap"]) for a in arms]
el=[y[i]-float(G[a]["gap_ci_low"]) for i,a in enumerate(arms)]
eh=[float(G[a]["gap_ci_high"])-y[i] for i,a in enumerate(arms)]
ax1.errorbar(x,y,yerr=[el,eh],fmt="o",ms=9,color="#22607F",ecolor="#14201D",
             capsize=4,elinewidth=1.4,linestyle="none")
for a,xi,yi in zip(arms,x,y):
    ax1.annotate(lab[a],(xi,yi),textcoords="offset points",xytext=(9,6),fontsize=9.5)
ax1.annotate("",xy=(GB["E-q4-emb8"],float(G["E-q4-emb8"]["gap"])),
             xytext=(GB["C-q4-g64"],float(G["C-q4-g64"]["gap"])),
             arrowprops=dict(arrowstyle="->",color="#A6650F",lw=1.8))
ax1.text(2.02,0.40,"+14% memory\ncloses half the gap",fontsize=9,color="#A6650F")
ax1.axhline(0,color="#14201D",lw=1,ls="--")
ax1.set_xlim(1.6,4.0); ax1.set_xlabel("Model size on disk (GB)")
ax1.set_ylabel("Fairness gap (percentage points)")
ax1.set_title("Cost of the gap, and what it takes to close it\nGap = mean low resource ΔBPB minus mean high resource",fontsize=11,loc="left")

ax2.bar(range(4),[float(G[a]["mean_delta_high"]) for a in arms],width=.38,label="high resource",color="#22607F")
ax2.bar([i+.4 for i in range(4)],[float(G[a]["mean_delta_low"]) for a in arms],width=.38,label="low resource",color="#A6650F")
ax2.set_xticks([i+.2 for i in range(4)]); ax2.set_xticklabels([lab[a] for a in arms],fontsize=9,rotation=12)
ax2.axhspan(-0.3,0.3,color="#8A9793",alpha=.16,zorder=0)
ax2.set_ylabel("Mean relative ΔBPB (%)")
ax2.set_title("Arm D lowers degradation overall but widens the gap\nArm E is the only one that narrows it",fontsize=11,loc="left")
ax2.legend(fontsize=9,frameon=False)
for a in (ax1,ax2): a.spines[["top","right"]].set_visible(False); a.grid(alpha=.22)
plt.tight_layout(); plt.savefig("figures/05_memory_vs_quality.png",dpi=170)
print("-> figures/03_delta_bpb.png, figures/05_memory_vs_quality.png")
