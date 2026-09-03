"""Figures 03 to 06. English labels throughout; cool for high resource, warm for low."""
import csv, json, math, re, collections
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COOL, WARM, MID, INK, MUTE = "#22607F", "#A6650F", "#7A8B87", "#14201D", "#55635F"
COL = {"high": COOL, "mid": MID, "low": WARM}
M = {r["flores_code"]: r for r in csv.DictReader(open("data/lang_meta.csv"))}
GB = {"A-bf16":6.72,"B-q8-g64":3.58,"C-q4-g64":1.91,"D-q4-g32":2.11,"E-q4-emb8":2.17}
SHORT = {"A-bf16":"bf16","B-q8-g64":"8-bit","C-q4-g64":"4-bit","D-q4-g32":"4-bit g32","E-q4-emb8":"4-bit\n+8-bit emb"}
def tidy(ax):
    ax.spines[["top","right"]].set_visible(False); ax.grid(alpha=.22)

# ---------------- 03 delta BPB ----------------
d = [r for r in csv.DictReader(open("results/delta_bpb.csv")) if r["arm"]=="C-q4-g64"]
d.sort(key=lambda r: float(r["rel_delta_pct"]))
fig, ax = plt.subplots(figsize=(8.6,4.8))
y = np.arange(len(d))
v = [float(r["rel_delta_pct"]) for r in d]
err = [[v[i]-float(r["ci_low"]) for i,r in enumerate(d)], [float(r["ci_high"])-v[i] for i,r in enumerate(d)]]
ax.barh(y, v, color=[COL[r["tier"]] for r in d], height=.66)
ax.errorbar(v, y, xerr=err, fmt="none", ecolor=INK, elinewidth=1.1, capsize=3)
ax.axvline(0.30, color=MUTE, ls=":", lw=1.2)
ax.text(0.305, len(d)-0.4, " noise floor", fontsize=8, color=MUTE, va="center")
ax.set_yticks(y); ax.set_yticklabels([M[r["flores_code"]]["name"] for r in d], fontsize=9)
ax.set_xlabel("Increase in bits per byte at 4-bit, percent of the bf16 value")
ax.set_title("Likelihood cost of four bits, by language\n"
             "Ordered by size of the effect, not by resource tier", fontsize=11, loc="left")
h = [plt.Rectangle((0,0),1,1,color=COL[t]) for t in ["high","mid","low"]]
ax.legend(h, ["high resource","mid","low resource"], fontsize=8.5, frameon=False, loc="lower right")
tidy(ax); plt.tight_layout(); plt.savefig("figures/03_delta_bpb.png", dpi=170); plt.close()

# ---------------- 05 headline: memory against quality, by tier ----------------
ch = {(r["arm"], r["flores_code"]): float(r["rel_delta_chrfpp_pct"])
      for r in csv.DictReader(open("results/chrf_delta.csv"))}
LANGS = [L for L in M if L != "eng_Latn"]
HI = [L for L in LANGS if M[L]["tier"]=="high"]; LO = [L for L in LANGS if M[L]["tier"]=="low"]
arms = ["A-bf16","C-q4-g64","E-q4-emb8"]
hi = [0.0] + [np.mean([ch[(a,L)] for L in HI]) for a in arms[1:]]
lo = [0.0] + [np.mean([ch[(a,L)] for L in LO]) for a in arms[1:]]
x  = [GB[a] for a in arms]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13,5.2), gridspec_kw={"width_ratios":[1.25,1]})
a1.plot(x, hi, "o-", color=COOL, lw=2, ms=9, label="high resource mean")
a1.plot(x, lo, "o-", color=WARM, lw=2, ms=9, label="low resource mean")
# The two 4-bit arms differ by 0.26 GB, so their labels are placed by hand to avoid collision.
place = {0: (-6, 14, "right"), 1: (-10, -18, "right"), 2: (10, -18, "left")}
for i, a in enumerate(arms):
    dx, dy, ha = place[i]
    a1.annotate(SHORT[a].replace("\n", " "), (x[i], lo[i]), textcoords="offset points",
                xytext=(dx, dy), ha=ha, fontsize=9, color=INK)
    if i:
        a1.plot([x[i], x[i]], [hi[i], lo[i]], color=MUTE, lw=1, ls="--")
        a1.annotate(f"{lo[i]-hi[i]:+.1f} pp", (x[i], (hi[i]+lo[i])/2), textcoords="offset points",
                    xytext=(8 if i == 2 else -8, 0), ha="left" if i == 2 else "right",
                    fontsize=9, color=MUTE)
a1.axhline(0, color=MUTE, lw=1)
a1.set_xlim(1.2, 7.6); a1.set_ylim(-9.6, 1.4)
a1.set_xlabel("Model weights on disk, GB"); a1.set_ylabel("Change in chrF++, percent")
a1.set_title("Four bits costs low resource languages more,\nand keeping the embedding at 8-bit does not reliably fix it",
             fontsize=11, loc="left")
a1.legend(fontsize=9, frameon=False, loc="lower right"); tidy(a1)

g = list(csv.DictReader(open("results/fairness_gap.csv")))
order = ["B-q8-g64","C-q4-g64","D-q4-g32","E-q4-emb8"]
g = sorted(g, key=lambda r: order.index(r["arm"]))
xs = np.arange(len(g)); gv = [float(r["gap"]) for r in g]
ge = [[gv[i]-float(r["gap_ci_low"]) for i,r in enumerate(g)],
      [float(r["gap_ci_high"])-gv[i] for i,r in enumerate(g)]]
a2.bar(xs, gv, color=[COOL if r["arm"]=="B-q8-g64" else WARM for r in g], width=.6)
a2.errorbar(xs, gv, yerr=ge, fmt="none", ecolor=INK, elinewidth=1.2, capsize=4)
a2.axhline(0, color=MUTE, lw=1)
a2.set_xticks(xs); a2.set_xticklabels([SHORT[r["arm"]].replace("\n"," ") for r in g], fontsize=8.5)
a2.set_ylabel("Fairness gap in bits per byte, percentage points")
a2.set_title("Gap on the primary metric, with intervals\nGap means low resource minus high resource degradation",
             fontsize=11, loc="left")
tidy(a2); plt.tight_layout(); plt.savefig("figures/05_memory_vs_quality.png", dpi=170); plt.close()

# ---------------- 04 the metric that had to be fixed ----------------
fid = {(r["arm"], r["flores_code"]): r for r in csv.DictReader(open("results/fidelity.csv"))}
before = {"spa_Latn":0.0,"rus_Cyrl":2.0,"cmn_Hant":44.5,"hin_Deva":5.5,"arb_Arab":1.0,
          "swh_Latn":0.0,"yor_Latn":0.0,"amh_Ethi":4.0,"mya_Mymr":19.5}
order = sorted(LANGS, key=lambda L: -before[L])
fig, ax = plt.subplots(figsize=(9.4,4.6))
w = .38; xs = np.arange(len(order))
ax.bar(xs-w/2, [before[L] for L in order], w, color="#96382C", label="absolute threshold")
ax.bar(xs+w/2, [float(fid[("A-bf16",L)]["script_drift_rate"])*100 for L in order], w,
       color=COOL, label="measured against the reference")
ax.set_xticks(xs); ax.set_xticklabels([M[L]["name"] for L in order], rotation=32, ha="right", fontsize=8.5)
ax.set_ylabel("Percent of bf16 output flagged as script drift")
ax.set_title("Checking a failure metric against the unquantized baseline\n"
             "bf16 is not broken, so 44.5% meant the instrument was", fontsize=11, loc="left")
ax.legend(fontsize=9, frameon=False); tidy(ax)
plt.tight_layout(); plt.savefig("figures/04_script_drift.png", dpi=170); plt.close()

# ---------------- 06 speed ----------------
sp = list(csv.DictReader(open("results/speed.csv")))
ARMS = ["A-bf16","B-q8-g64","C-q4-g64","D-q4-g32","E-q4-emb8"]
seq = {"A-bf16":.79,"B-q8-g64":.72,"C-q4-g64":.56,"D-q4-g32":.51,"E-q4-emb8":.49}
fig, (b1,b2) = plt.subplots(1,2, figsize=(13,4.9))
med = [np.median([float(r["decode_tps"]) for r in sp if r["arm"]==a]) for a in ARMS]
q1  = [np.percentile([float(r["decode_tps"]) for r in sp if r["arm"]==a],25) for a in ARMS]
q3  = [np.percentile([float(r["decode_tps"]) for r in sp if r["arm"]==a],75) for a in ARMS]
ceil= [100/GB[a] for a in ARMS]
xs = np.arange(len(ARMS))
b1.bar(xs, ceil, .62, color="#DCEAF1", label="roofline from memory bandwidth")
b1.bar(xs, med, .62, color=COOL, label="measured median")
b1.errorbar(xs, med, yerr=[np.array(med)-q1, np.array(q3)-np.array(med)],
            fmt="none", ecolor=INK, elinewidth=1.2, capsize=4)
for i,a in enumerate(ARMS):  # above the roofline bar, clear of the error bars
    b1.text(i, ceil[i]+1.0, f"{med[i]/ceil[i]:.0%}", ha="center", fontsize=9, color=INK)
b1.set_ylim(0, 58)
b1.set_xticks(xs); b1.set_xticklabels([SHORT[a] for a in ARMS], fontsize=8.5)
b1.set_ylabel("Decode tokens per second")
b1.set_title("Measured against the bandwidth ceiling\nLabels are kernel efficiency", fontsize=11, loc="left")
b1.legend(fontsize=8.5, frameon=False); tidy(b1)

b2.plot(xs, [seq[a]*100 for a in ARMS], "s--", color="#96382C", ms=8, label="sequential, arms back to back")
b2.plot(xs, [med[i]/ceil[i]*100 for i in range(len(ARMS))], "o-", color=COOL, ms=8,
        label="interleaved, 90 s cooldown")
b2.set_xticks(xs); b2.set_xticklabels([SHORT[a] for a in ARMS], fontsize=8.5)
b2.set_ylim(40,100); b2.set_ylabel("Kernel efficiency, percent of roofline")
b2.set_title("The same five arms, measured two ways\nThe 30 point decline was thermal, not dequantization",
             fontsize=11, loc="left")
b2.legend(fontsize=8.5, frameon=False); tidy(b2)
plt.tight_layout(); plt.savefig("figures/06_speed.png", dpi=170); plt.close()
print("wrote figures 03 to 06")

# ---------------- 07 damage is a tail event ----------------
import numpy as np
col = list(csv.DictReader(open("results/collapse.csv")))
drops = json.load(open("results/per_sentence_drop.json"))
fig, (c1, c2) = plt.subplots(1, 2, figsize=(13, 4.9))

d = np.array(drops["drop"])
c1.hist(d, bins=np.arange(-10, 70, 2), color=COOL, edgecolor="white", linewidth=.4)
c1.axvline(20, color="#96382C", ls="--", lw=1.4)
c1.text(21, c1.get_ylim()[1]*.62,
        f"  {(d>=20).mean():.1%} of sentences\n  lose 20+ points\n  and carry 45%\n  of all degradation",
        fontsize=9, color="#96382C", va="top")
c1.axvline(float(np.median(d)), color=INK, lw=1)
c1.text(float(np.median(d))-1.5, c1.get_ylim()[1]*.9, f"median {np.median(d):.1f}",
        fontsize=9, color=INK, ha="right")
c1.set_yscale("log"); c1.set_xlabel("chrF++ points lost to four bits, one sentence")
c1.set_ylabel("Sentences, log scale")
c1.set_title("Four bits does not degrade output evenly\nIt breaks a small number of sentences badly",
             fontsize=11, loc="left")
tidy(c1)

col.sort(key=lambda r: -float(r["collapse_rate_4bit"]))
ys = np.arange(len(col)); w = .38
c2.barh(ys+w/2, [float(r["collapse_rate_4bit"])*100 for r in col], w,
        color=[COL[r["tier"]] for r in col], label="4-bit")
c2.barh(ys-w/2, [float(r["collapse_rate_mitigated"])*100 for r in col], w,
        color=[COL[r["tier"]] for r in col], alpha=.45, label="with 8-bit embedding")
for i, r in enumerate(col):
    if float(r["ci_low"]) > 0:
        c2.text(float(r["collapse_rate_4bit"])*100+.25, ys[i], "*", fontsize=13,
                color=INK, va="center")
c2.set_yticks(ys); c2.set_yticklabels([r["name"] for r in col], fontsize=9)
c2.set_xlabel("Percent of sentences that collapse")
c2.set_title("Collapse rate by language\n* marks a reduction whose interval excludes zero",
             fontsize=11, loc="left")
c2.legend(fontsize=8.5, frameon=False, loc="upper right"); tidy(c2)
plt.tight_layout(); plt.savefig("figures/07_collapse.png", dpi=170); plt.close()
print("wrote figure 07")
