"""Registered analysis for H1 and H2, plus an explicitly labelled post-hoc extension.

Two distinct sources of uncertainty, and it matters which one a claim needs:

  Sentence-level bootstrap. Resamples the 1012 parallel sentences, paired across arms, and
  recomputes every BPB. This gives precise intervals for the gap AMONG THESE TEN LANGUAGES.
  It treats the language set as fixed, because it is.

  Language-level inference. Generalising from "these ten languages" to "low resource languages"
  needs variation across languages, and there are four per tier. No amount of sentence resampling
  fixes that. The registered regression is reported with its confidence intervals so the reader can
  see how little it constrains.
"""
import json, csv, math, sys
import numpy as np

ARMS = ["B-q8-g64", "C-q4-g64", "D-q4-g32", "E-q4-emb8"]
BASE = "A-bf16"
LANGS = ["eng_Latn","spa_Latn","rus_Cyrl","cmn_Hant","hin_Deva",
         "arb_Arab","swh_Latn","yor_Latn","amh_Ethi","mya_Mymr"]
NBOOT, SEED = 10_000, 1337
LN2 = math.log(2)

from pathlib import Path
# Only analyse arms whose per-sentence files are complete, so this can run before the full sweep ends.
ARMS = [a for a in ARMS if all(Path(f"results/parts/sent_{a}_{L}.json").exists() for L in LANGS)]
if not all(Path(f"results/parts/sent_{BASE}_{L}.json").exists() for L in LANGS):
    sys.exit("baseline arm A is not complete yet")
if not ARMS:
    sys.exit("no treatment arm is complete yet")
print("arms available:", ", ".join(ARMS))

M = {r["flores_code"]: r for r in csv.DictReader(open("data/lang_meta.csv"))}
HI = [L for L in LANGS if M[L]["tier"] == "high"]
LO = [L for L in LANGS if M[L]["tier"] == "low"]

def load(arm, lang):
    d = json.load(open(f"results/parts/sent_{arm}_{lang}.json"))
    return np.array(d["nll"], float), np.array(d["bytes"], float)

nll, byt = {}, {}
for L in LANGS:
    for a in [BASE] + ARMS:
        nll[(a, L)], b = load(a, L)
    byt[L] = b

N = len(byt[LANGS[0]])
print(f"{N} sentences per language, {NBOOT} bootstrap resamples, seed {SEED}\n")

def rel_delta(arm, L, idx):
    """Relative change in BPB against bf16, in percent, over the sentences in idx."""
    b = byt[L][idx].sum()
    a0 = nll[(BASE, L)][idx].sum() / LN2 / b
    a1 = nll[(arm, L)][idx].sum() / LN2 / b
    return (a1 - a0) / a0 * 100

rng = np.random.default_rng(SEED)
boot_idx = [rng.integers(0, N, N) for _ in range(NBOOT)]
allidx = np.arange(N)

point = {(a, L): rel_delta(a, L, allidx) for a in ARMS for L in LANGS}
dist  = {(a, L): np.array([rel_delta(a, L, i) for i in boot_idx]) for a in ARMS for L in LANGS}

def ci(v, lo=2.5, hi=97.5):
    return np.percentile(v, lo), np.percentile(v, hi)

# ---------- per language ----------
print("Relative delta BPB against bf16, percent, with 95% CI from the sentence bootstrap")
print("A CI that contains zero means the arm is indistinguishable from bf16 on that language.\n")
for a in ARMS:
    print(f"  {a}")
    print(f"    {'language':24}{'tier':6}{'delta':>8}{'95% CI':>20}")
    for L in LANGS:
        lo, hi = ci(dist[(a, L)])
        star = "" if (lo <= 0 <= hi) else "  *"
        print(f"    {M[L]['name']:24}{M[L]['tier']:6}{point[(a,L)]:7.2f}%  [{lo:6.2f}, {hi:6.2f}]{star}")
    print("    * = interval excludes zero\n")

# ---------- fairness gap ----------
print("Fairness gap: mean degradation of the four low resource languages minus the four high")
print("resource ones, in percentage points. Positive means low resource languages pay more.")
print("These intervals describe THIS set of ten languages, not low resource languages in general.\n")
gap_pt, gap_dist = {}, {}
for a in ARMS:
    gap_pt[a] = np.mean([point[(a,L)] for L in LO]) - np.mean([point[(a,L)] for L in HI])
    gap_dist[a] = (np.mean([dist[(a,L)] for L in LO], axis=0)
                   - np.mean([dist[(a,L)] for L in HI], axis=0))
    lo, hi = ci(gap_dist[a])
    star = "" if (lo <= 0 <= hi) else "  *"
    print(f"  {a:12}{gap_pt[a]:+6.2f} pp   [{lo:+6.2f}, {hi:+6.2f}]{star}")

# ---------- H2 decision rule ----------
print("\nH2 decision rule, as registered: the 95% CI on gap(C) - gap(E) must exclude zero,")
print("and arm E must close at least half the gap.\n")
if not ("C-q4-g64" in ARMS and "E-q4-emb8" in ARMS):
    print("  arms C and E are not both complete; skipping the H2 test")
    sys.exit(0)
dC, dE = gap_dist["C-q4-g64"], gap_dist["E-q4-emb8"]
diff = dC - dE
lo, hi = ci(diff)
gC, gE = gap_pt["C-q4-g64"], gap_pt["E-q4-emb8"]
shrink = (gC - gE) / gC * 100
sh_lo, sh_hi = ci((dC - dE) / dC * 100)
print(f"  gap(C)              {gC:+.3f} pp")
print(f"  gap(E)              {gE:+.3f} pp")
print(f"  gap(C) - gap(E)     {gC-gE:+.3f} pp   95% CI [{lo:+.3f}, {hi:+.3f}]")
print(f"  condition 1, CI excludes zero      {'MET' if not (lo <= 0 <= hi) else 'NOT MET'}")
print(f"  shrinkage           {shrink:.1f}%   95% CI [{sh_lo:.1f}, {sh_hi:.1f}]")
print(f"  condition 2, shrinkage at least 50%  {'MET' if shrink >= 50 else 'NOT MET'}"
      f"   (lower CI bound {sh_lo:.1f}%)")
print(f"\n  H2: {'SUPPORTED' if (not (lo <= 0 <= hi)) and shrink >= 50 else 'NOT SUPPORTED'}")

# ---------- registered regression ----------
print("\n" + "="*72)
print("Registered regression for H1, at the language level: n = 10")
import statsmodels.api as sm
import pandas as pd
df = pd.DataFrame([{
    "lang": M[L]["name"], "delta": point[("C-q4-g64", L)],
    "tier_low": 1.0 if M[L]["tier"] == "low" else 0.0,
    "tier_mid": 1.0 if M[L]["tier"] == "mid" else 0.0,
    "non_latin": 1.0 if M[L]["script_family"] == "non-Latin" else 0.0,
    "fertility": float(M[L]["tok_ratio_vs_eng"]),
    "baseline": nll[(BASE, L)].sum() / LN2 / byt[L].sum(),
} for L in LANGS])
X = sm.add_constant(df[["tier_low","tier_mid","non_latin","fertility","baseline"]])
res = sm.OLS(df["delta"], X).fit()
print(res.summary().as_text())
c = res.conf_int().loc["tier_low"]
print(f"\ntier_low coefficient {res.params['tier_low']:+.3f}, 95% CI [{c[0]:+.3f}, {c[1]:+.3f}]")
print(f"residual degrees of freedom: {int(res.df_resid)}")
print(f"H1 by the registered rule: "
      f"{'SUPPORTED' if c[0] > 0 else 'NOT SUPPORTED, the interval contains zero' if c[0] <= 0 <= c[1] else 'SUPPORTED in the opposite direction'}")

with open("results/regression.txt","w") as f:
    f.write(res.summary().as_text())
    f.write(f"\n\ntier_low 95% CI [{c[0]:+.4f}, {c[1]:+.4f}]  residual df {int(res.df_resid)}\n")

with open("results/fairness_gap.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["arm","mean_delta_high","mean_delta_low","gap","gap_ci_low","gap_ci_high"])
    for a in ARMS:
        lo_, hi_ = ci(gap_dist[a])
        w.writerow([a, round(np.mean([point[(a,L)] for L in HI]),4),
                    round(np.mean([point[(a,L)] for L in LO]),4),
                    round(gap_pt[a],4), round(lo_,4), round(hi_,4)])

with open("results/delta_bpb.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["arm","flores_code","tier","script_family","rel_delta_pct","ci_low","ci_high"])
    for a in ARMS:
        for L in LANGS:
            lo_, hi_ = ci(dist[(a,L)])
            w.writerow([a, L, M[L]["tier"], M[L]["script_family"],
                        round(point[(a,L)],4), round(lo_,4), round(hi_,4)])
print("\n-> results/regression.txt, results/fairness_gap.csv, results/delta_bpb.csv")
