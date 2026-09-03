"""Registered L2 analysis: delta chrF++ with a paired bootstrap, and the H2 gap test.

A sensitivity analysis on plain chrF is included and LABELLED POST-HOC. chrF++ adds word bigrams,
and "word" means whitespace-split, which is meaningless for Chinese and Burmese. The direction of
that bias is known in advance from the script, not chosen from the results, but the metric was
registered as chrF++ so the registered number leads and the alternative follows as a check on
whether the conclusion depends on the choice.
"""
import json, csv, re, sys
import numpy as np, sacrebleu

SPECIAL = re.compile(r"<\|[^|]*\|>"); clean = lambda t: SPECIAL.sub("", t).strip()
ARMS = ["A-bf16", "C-q4-g64", "E-q4-emb8"]; BASE = "A-bf16"
NBOOT, SEED = 10_000, 1337
M = {r["flores_code"]: r for r in csv.DictReader(open("data/lang_meta.csv"))}
LANGS = [L for L in M if L != "eng_Latn"]
HI = [L for L in LANGS if M[L]["tier"] == "high"]; LO = [L for L in LANGS if M[L]["tier"] == "low"]

def stats(arm, lang, wo):
    """Per-sentence sufficient statistics, so the bootstrap can re-aggregate without rescoring.

    Validated against corpus_score on every call: if the reconstruction from summed statistics
    does not reproduce the direct score, the aggregation is wrong and everything downstream is
    meaningless. The first version of this function passed references in the wrong shape and
    produced zero-width intervals, which is what made the bug visible.
    """
    d = [json.loads(l) for l in open(f"outputs/{arm}_{lang}.jsonl", encoding="utf-8")]
    hyps = [clean(x["hyp"]) for x in d]
    refs = [[x["ref"] for x in d]]                     # one reference stream, as corpus_score wants
    m = sacrebleu.CHRF(word_order=wo)
    st = np.array(m._extract_corpus_statistics(hyps, refs), float)
    direct = m.corpus_score(hyps, refs).score
    rebuilt = m._compute_score_from_stats(list(st.sum(0))).score
    assert abs(direct - rebuilt) < 1e-6, (
        f"statistics aggregation is wrong for {arm}/{lang}: {direct} vs {rebuilt}")
    return m, st

def run(wo, label):
    S = {(a, L): stats(a, L, wo) for a in ARMS for L in LANGS}
    n = len(S[(BASE, LANGS[0])][1])
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, n, n) for _ in range(NBOOT)]

    def score(a, L, i=None):
        m, st = S[(a, L)]
        agg = st.sum(0) if i is None else st[i].sum(0)
        return m._compute_score_from_stats(list(agg)).score

    print(f"\n{'='*78}\n{label}\n{'='*78}")
    print(f"{'language':24}{'tier':6}{'bf16':>7}" + "".join(f"{a.split('-')[0]:>26}" for a in ARMS[1:]))
    pt, dist = {}, {}
    for L in LANGS:
        b = score(BASE, L); bd = np.array([score(BASE, L, i) for i in idx])
        line = f"{M[L]['name']:24}{M[L]['tier']:6}{b:7.2f}"
        for a in ARMS[1:]:
            s = score(a, L); sd = np.array([score(a, L, i) for i in idx])
            pt[(a, L)] = (s - b) / b * 100
            dist[(a, L)] = (sd - bd) / bd * 100
            lo, hi = np.percentile(dist[(a, L)], [2.5, 97.5])
            star = "" if lo <= 0 <= hi else "*"
            line += f"  {s:6.2f} {pt[(a,L)]:+6.2f}% [{lo:+5.1f},{hi:+5.1f}]{star:1}"
        print(line)
    print("  * = the interval excludes zero")

    print(f"\n{'tier':10}" + "".join(f"{a.split('-')[0]:>12}" for a in ARMS[1:]))
    for t, ls in [("high", HI), ("mid", [L for L in LANGS if M[L]['tier']=='mid']), ("low", LO)]:
        print(f"{t:10}" + "".join(f"{np.mean([pt[(a,L)] for L in ls]):11.2f}%" for a in ARMS[1:]))

    print("\nFairness gap, low resource mean minus high resource mean, percentage points.")
    print("More negative means low resource languages lose more.")
    g, gd = {}, {}
    for a in ARMS[1:]:
        g[a] = np.mean([pt[(a,L)] for L in LO]) - np.mean([pt[(a,L)] for L in HI])
        gd[a] = (np.mean([dist[(a,L)] for L in LO], axis=0) - np.mean([dist[(a,L)] for L in HI], axis=0))
        lo, hi = np.percentile(gd[a], [2.5, 97.5])
        star = "" if lo <= 0 <= hi else "*"
        print(f"  {a:12}{g[a]:+7.2f} pp  [{lo:+6.2f},{hi:+6.2f}]{star}")

    C, E = "C-q4-g64", "E-q4-emb8"
    diff = gd[C] - gd[E]; lo, hi = np.percentile(diff, [2.5, 97.5])
    shrink = (abs(g[C]) - abs(g[E])) / abs(g[C]) * 100
    sd = (np.abs(gd[C]) - np.abs(gd[E])) / np.abs(gd[C]) * 100
    slo, shi = np.percentile(sd, [2.5, 97.5])
    print(f"\nH2 decision rule, as registered")
    print(f"  gap(C) {g[C]:+.2f} pp   gap(E) {g[E]:+.2f} pp")
    print(f"  difference {g[C]-g[E]:+.2f} pp  95% CI [{lo:+.2f},{hi:+.2f}]")
    print(f"  condition 1, CI excludes zero        {'MET' if not (lo <= 0 <= hi) else 'NOT MET'}")
    print(f"  shrinkage {shrink:.0f}%  95% CI [{slo:.0f},{shi:.0f}]")
    print(f"  condition 2, at least 50% shrinkage  {'MET' if shrink >= 50 else 'NOT MET'}")
    print(f"  H2 on this metric: "
          f"{'SUPPORTED' if (not (lo <= 0 <= hi)) and shrink >= 50 else 'NOT SUPPORTED'}")
    return pt

pt2 = run(2, "REGISTERED METRIC: chrF++ (word_order=2)")
pt0 = run(0, "[POST-HOC] SENSITIVITY: plain chrF (word_order=0), no whitespace assumption")

with open("results/chrf_delta.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["arm","flores_code","tier","rel_delta_chrfpp_pct","rel_delta_chrf_pct"])
    for a in ARMS[1:]:
        for L in LANGS:
            w.writerow([a,L,M[L]["tier"],round(pt2[(a,L)],3),round(pt0[(a,L)],3)])
print("\n-> results/chrf_delta.csv")
