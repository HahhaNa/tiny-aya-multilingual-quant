"""L3: failures that chrF++ scores as mild.

Four cheap, deterministic proxies for the collapse that automatic metrics understate:
script drift, language confusion, degenerate repetition, and empty or truncated output.
"""
import json, csv, re, collections
from pathlib import Path
import numpy as np

ARMS = ["A-bf16", "C-q4-g64", "E-q4-emb8"]
M = {r["flores_code"]: r for r in csv.DictReader(open("data/lang_meta.csv"))}
LANGS = [L for L in M if L != "eng_Latn"]

RANGES = {   # the Unicode blocks each target script is allowed to use
 "spa_Latn": [(0x0041,0x024F)],
 "rus_Cyrl": [(0x0400,0x04FF)],
 "cmn_Hant": [(0x3000,0x303F),(0x4E00,0x9FFF),(0xF900,0xFAFF)],
 "hin_Deva": [(0x0900,0x097F)],
 "arb_Arab": [(0x0600,0x06FF),(0x0750,0x077F),(0xFB50,0xFDFF),(0xFE70,0xFEFF)],
 "swh_Latn": [(0x0041,0x024F)],
 "yor_Latn": [(0x0041,0x024F),(0x0300,0x036F),(0x1E00,0x1EFF)],
 "amh_Ethi": [(0x1200,0x137F)],
 "mya_Mymr": [(0x1000,0x109F),(0xAA60,0xAA7F)],
}
def in_script(ch, lang):
    o = ord(ch)
    return any(a <= o <= b for a, b in RANGES[lang])

def script_share(text, lang):
    letters = [c for c in text if c.isalpha()]
    if not letters: return 0.0
    return sum(in_script(c, lang) for c in letters) / len(letters)

def top_3gram_share(text):
    toks = text.split()
    if len(toks) < 6: return 0.0
    g = [" ".join(toks[i:i+3]) for i in range(len(toks)-2)]
    return collections.Counter(g).most_common(1)[0][1] / len(g)

try:
    from lingua import LanguageDetectorBuilder, Language
    LID = {"spa_Latn":Language.SPANISH,"rus_Cyrl":Language.RUSSIAN,"cmn_Hant":Language.CHINESE,
           "hin_Deva":Language.HINDI,"arb_Arab":Language.ARABIC,"swh_Latn":Language.SWAHILI,
           "yor_Latn":Language.YORUBA,"amh_Ethi":Language.AMHARIC}
    det = LanguageDetectorBuilder.from_all_languages().build()
except Exception as e:
    print("lingua unavailable:", e); LID, det = {}, None

rows, cands = [], []
for arm in ARMS:
    for lang in LANGS:
        p = Path(f"outputs/{arm}_{lang}.jsonl")
        if not p.exists(): continue
        d = [json.loads(l) for l in open(p, encoding="utf-8")]
        drift = [script_share(x["hyp"], lang) < 0.70 for x in d]
        rep   = [top_3gram_share(x["hyp"]) > 0.20 for x in d]
        short = [len(x["hyp"]) < 5 or len(x["hyp"]) < 0.20*len(x["ref"]) for x in d]
        if det and lang in LID:
            lid = [det.detect_language_of(x["hyp"]) != LID[lang] if len(x["hyp"]) > 10 else True for x in d]
            lid_rate = float(np.mean(lid))
        else:
            lid_rate = float("nan")
        rows.append(dict(arm=arm, flores_code=lang, n=len(d),
                         script_drift_rate=round(float(np.mean(drift)),4),
                         lid_mismatch_rate=round(lid_rate,4) if lid_rate==lid_rate else "",
                         rep_3gram_rate=round(float(np.mean(rep)),4),
                         empty_or_short_rate=round(float(np.mean(short)),4)))
        print(f"  {arm:12}{lang:10} drift {np.mean(drift):6.1%}  lid {lid_rate:6.1%}  "
              f"rep {np.mean(rep):5.1%}  short {np.mean(short):5.1%}")
        for i,x in enumerate(d):
            if drift[i] or rep[i] or short[i]:
                cands.append(dict(arm=arm, flores_code=lang, idx=x["idx"],
                                  script_share=round(script_share(x["hyp"],lang),3),
                                  rep=round(top_3gram_share(x["hyp"]),3),
                                  hyp=x["hyp"][:200], ref=x["ref"][:200]))

with open("results/fidelity.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
if cands:
    with open("results/qualitative_candidates.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(cands[0].keys())); w.writeheader(); w.writerows(cands)
print(f"\n-> results/fidelity.csv ({len(rows)} rows), {len(cands)} qualitative candidates")
