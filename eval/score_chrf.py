"""chrF++ for every completed (arm, language) output file."""
import json, csv, glob, re
from pathlib import Path
import sacrebleu

ARMS = ["A-bf16", "C-q4-g64", "E-q4-emb8"]
M = {r["flores_code"]: r for r in csv.DictReader(open("data/lang_meta.csv"))}
LANGS = [L for L in M if L != "eng_Latn"]

rows = []
for arm in ARMS:
    for lang in LANGS:
        p = Path(f"outputs/{arm}_{lang}.jsonl")
        if not p.exists(): continue
        d = [json.loads(l) for l in open(p, encoding="utf-8")]
        hyps = [x["hyp"] for x in d]; refs = [[x["ref"] for x in d]]
        # chrF++ is chrF with word bigrams, which is what --chrf-word-order 2 means
        m = sacrebleu.CHRF(word_order=2).corpus_score(hyps, refs)
        rows.append(dict(arm=arm, flores_code=lang, n_sents=len(d), chrf2=round(m.score, 4)))
        print(f"  {arm:12}{lang:10}{m.score:7.2f}")

with open("results/chrf.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["arm","flores_code","n_sents","chrf2"]); w.writeheader(); w.writerows(rows)
print(f"\n-> results/chrf.csv ({len(rows)} rows)")
