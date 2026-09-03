"""Post-hoc: is quantization damage a uniform shift, or a rare catastrophic failure mode?

Not registered. The registered analysis compares means, which is the right test for a shift in
central tendency and the wrong one for a change in tail frequency. This asks the second question,
and is reported as exploratory.
"""
import json, csv, collections
import numpy as np
from sacrebleu.metrics import CHRF

chrf = CHRF(word_order=2)
COLLAPSE = 20          # chrF points lost against bf16 on the same sentence
NBOOT, SEED = 10_000, 1337

M = {r["flores_code"]: r for r in csv.DictReader(open("data/lang_meta.csv"))}
LANGS = [L for L in M if L != "eng_Latn"]

per = {}
for lang in LANGS:
    A = [json.loads(l) for l in open(f"outputs/A-bf16_{lang}.jsonl", encoding="utf-8")]
    C = [json.loads(l) for l in open(f"outputs/C-q4-g64_{lang}.jsonl", encoding="utf-8")]
    E = [json.loads(l) for l in open(f"outputs/E-q4-emb8_{lang}.jsonl", encoding="utf-8")]
    per[lang] = np.array([[chrf.sentence_score(x["hyp"], [x["ref"]]).score for x in (a, c, e)]
                          for a, c, e in zip(A, C, E)])

rng = np.random.default_rng(SEED)
N = len(per[LANGS[0]])
idx = [rng.integers(0, N, N) for _ in range(NBOOT)]

def rate(lang, arm_col, sample=None):
    v = per[lang] if sample is None else per[lang][sample]
    return float((v[:, 0] - v[:, arm_col] >= COLLAPSE).mean())

out = []
for lang in LANGS:
    rc, re_ = rate(lang, 1), rate(lang, 2)
    bc = np.array([rate(lang, 1, s) for s in idx])
    be = np.array([rate(lang, 2, s) for s in idx])
    d  = bc - be
    out.append(dict(flores_code=lang, name=M[lang]["name"], tier=M[lang]["tier"],
                    collapse_rate_4bit=round(rc, 4), collapse_rate_mitigated=round(re_, 4),
                    reduction_pp=round((rc - re_) * 100, 2),
                    ci_low=round(np.percentile(d, 2.5) * 100, 2),
                    ci_high=round(np.percentile(d, 97.5) * 100, 2)))

with open("results/collapse.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

allv = np.vstack([per[L] for L in LANGS])
drop = allv[:, 0] - allv[:, 1]
rec  = allv[:, 2] - allv[:, 1]
mask = drop >= COLLAPSE

print(f"Per-sentence chrF lost to four bits, n = {len(drop)}")
for q in (50, 75, 90, 95, 99):
    print(f"  {q}th percentile   {np.percentile(drop, q):6.1f}")
print(f"  mean {drop.mean():.2f}, sd {drop.std():.1f}\n")
print(f"Sentences losing at least {COLLAPSE} points: {mask.sum()} of {len(drop)} = {mask.mean():.1%}")
print(f"  they account for {drop[mask].sum()/drop.sum():.0%} of all degradation\n")
print(f"Where the mitigation acts")
print(f"  collapsed sentences (n={mask.sum():4})  mean recovery {rec[mask].mean():+6.2f} chrF")
print(f"  everything else     (n={(~mask).sum():4})  mean recovery {rec[~mask].mean():+6.2f} chrF")
print(f"  collapses account for {rec[mask].sum()/rec.sum():.0%} of total recovery\n")
print(f"{'language':24}{'tier':6}{'4-bit':>8}{'mitigated':>11}{'reduction':>11}{'95% CI':>18}")
for r in sorted(out, key=lambda r: -r["collapse_rate_4bit"]):
    star = "  *" if r["ci_low"] > 0 else ""
    print(f"{r['name']:24}{r['tier']:6}{r['collapse_rate_4bit']:7.1%}{r['collapse_rate_mitigated']:11.1%}"
          f"{r['reduction_pp']:10.1f}pp  [{r['ci_low']:5.1f},{r['ci_high']:5.1f}]{star}")

for t in ("high", "mid", "low"):
    v = [r["collapse_rate_4bit"] for r in out if r["tier"] == t]
    print(f"\n  {t:5} mean collapse rate {np.mean(v):.1%}", end="")
print("\n\n-> results/collapse.csv")
