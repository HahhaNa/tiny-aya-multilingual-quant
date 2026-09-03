"""Mine the saved generations for cases where four bits visibly breaks and the mitigation recovers.

Selection is by per-sentence chrF computed from the stored outputs, not by reading through them,
so the examples are the ones the metric actually picks out.
"""
import json, csv, collections
from sacrebleu.metrics import CHRF
chrf = CHRF(word_order=2)

M = {r["flores_code"]: r for r in csv.DictReader(open("data/lang_meta.csv"))}
LANGS = [L for L in M if L != "eng_Latn"]

def load(arm, lang):
    return [json.loads(l) for l in open(f"outputs/{arm}_{lang}.jsonl", encoding="utf-8")]

def rep_rate(t, n=3):
    w = t.split()
    if len(w) < n + 4: return 0.0
    g = collections.Counter(tuple(w[i:i+n]) for i in range(len(w)-n+1))
    return max(g.values()) / max(1, sum(g.values()))

rows = []
for lang in LANGS:
    A, C, E = load("A-bf16", lang), load("C-q4-g64", lang), load("E-q4-emb8", lang)
    for a, c, e in zip(A, C, E):
        sa = chrf.sentence_score(a["hyp"], [a["ref"]]).score
        sc = chrf.sentence_score(c["hyp"], [c["ref"]]).score
        se = chrf.sentence_score(e["hyp"], [e["ref"]]).score
        rows.append(dict(lang=lang, idx=a["idx"], A=sa, C=sc, E=se,
                         drop=sa-sc, recover=se-sc, rep_C=rep_rate(c["hyp"]),
                         rep_A=rep_rate(a["hyp"]), src=a["src"], ref=a["ref"],
                         hyp_A=a["hyp"], hyp_C=c["hyp"], hyp_E=e["hyp"]))

# The demo wants cases that are unambiguous: bf16 was fine, four bits collapsed, the mitigation
# brought it back. Require a real baseline, a large drop, and a real recovery.
cand = [r for r in rows if r["A"] >= 40 and r["drop"] >= 20 and r["recover"] >= 15]
cand.sort(key=lambda r: -(r["drop"] + r["recover"]))
print(f"{len(cand)} sentences where bf16 was fine, 4-bit collapsed, and the mitigation recovered\n")
by = collections.Counter(r["lang"] for r in cand)
for L, n in by.most_common():
    print(f"  {M[L]['name']:24}{n}")

print("\nTop candidates")
for r in cand[:6]:
    print(f"\n  {M[r['lang']]['name']}  #{r['idx']}   "
          f"chrF  bf16 {r['A']:.0f}  4-bit {r['C']:.0f}  mitigation {r['E']:.0f}"
          f"   repetition at 4-bit {r['rep_C']:.0%}")
    print(f"    src : {r['src'][:96]}")
    print(f"    ref : {r['ref'][:96]}")
    print(f"    4bit: {r['hyp_C'][:96]}")
    print(f"    mit : {r['hyp_E'][:96]}")

json.dump(cand[:20], open("demo/examples.json","w"), ensure_ascii=False, indent=1)
print(f"\n-> demo/examples.json ({min(20,len(cand))} cases)")
