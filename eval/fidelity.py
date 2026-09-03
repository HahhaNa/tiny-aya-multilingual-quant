"""L3: failures that chrF++ scores as mild.

Three deterministic proxies for collapse that automatic metrics understate, plus language ID where
a detector exists for the language.

Two corrections over the first version, both found by checking the metric against the bf16 baseline:

  Script drift is measured RELATIVE TO THE REFERENCE for the same sentence. An absolute threshold
  measured proper nouns, not failure: Chinese references legitimately contain strings like 802.11n
  and TogiNet, so 44.5% of the untouched bf16 outputs were flagged. A baseline that high means the
  metric is wrong, not the model.

  The stop token <|END_RESPONSE|> is stripped. mlx-lm emitted it as text in 5329 of 5400
  generations. No text followed it, so truncation is lossless, but leaving it in depressed every
  chrF++ score and polluted the character counts.
"""
import json, csv, re, collections, sys
from pathlib import Path
import numpy as np

ARMS = ["A-bf16", "C-q4-g64", "E-q4-emb8"]
BASE = "A-bf16"
NBOOT, SEED = 10_000, 1337
SPECIAL = re.compile(r"<\|[^|]*\|>")

M = {r["flores_code"]: r for r in csv.DictReader(open("data/lang_meta.csv"))}
LANGS = [L for L in M if L != "eng_Latn"]

RANGES = {   # Unicode blocks each target script may legitimately use
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
DRIFT_MARGIN = 0.25   # how far below the reference's own script share counts as drift

def clean(t):
    return SPECIAL.sub("", t).strip()

def script_share(text, lang):
    letters = [c for c in text if c.isalpha()]
    if not letters: return 0.0
    return sum(any(a <= ord(c) <= b for a, b in RANGES[lang]) for c in letters) / len(letters)

def top_3gram_share(text):
    toks = text.split()
    if len(toks) < 6: return 0.0
    g = [" ".join(toks[i:i+3]) for i in range(len(toks)-2)]
    return collections.Counter(g).most_common(1)[0][1] / len(g)

from lingua import LanguageDetectorBuilder, Language
LID = {}
for code, name in [("spa_Latn","SPANISH"),("rus_Cyrl","RUSSIAN"),("cmn_Hant","CHINESE"),
                   ("hin_Deva","HINDI"),("arb_Arab","ARABIC"),("swh_Latn","SWAHILI"),
                   ("yor_Latn","YORUBA"),("amh_Ethi","AMHARIC"),("mya_Mymr","BURMESE")]:
    if hasattr(Language, name): LID[code] = getattr(Language, name)
missing = [c for c in LANGS if c not in LID]
print(f"language detector covers {len(LID)} of {len(LANGS)} languages; "
      f"missing {', '.join(M[c]['name'] for c in missing)}\n")
det = LanguageDetectorBuilder.from_all_languages().build()

flags, rows = {}, []
for arm in ARMS:
    for lang in LANGS:
        p = Path(f"outputs/{arm}_{lang}.jsonl")
        if not p.exists(): continue
        d = [json.loads(l) for l in open(p, encoding="utf-8")]
        hyp = [clean(x["hyp"]) for x in d]; ref = [x["ref"] for x in d]
        hs = [script_share(h, lang) for h in hyp]
        rs = [script_share(r, lang) for r in ref]
        drift = np.array([(r - h) > DRIFT_MARGIN for h, r in zip(hs, rs)])
        rep   = np.array([top_3gram_share(h) > 0.20 for h in hyp])
        short = np.array([len(h) < 5 or len(h) < 0.20*len(r) for h, r in zip(hyp, ref)])
        if lang in LID:
            lid = np.array([det.detect_language_of(h) != LID[lang] if len(h) > 10 else True for h in hyp])
        else:
            lid = None
        flags[(arm, lang)] = dict(drift=drift, rep=rep, short=short, lid=lid)
        rows.append(dict(arm=arm, flores_code=lang, n=len(d),
                         script_drift_rate=round(float(drift.mean()),4),
                         lid_mismatch_rate=round(float(lid.mean()),4) if lid is not None else "",
                         rep_3gram_rate=round(float(rep.mean()),4),
                         empty_or_short_rate=round(float(short.mean()),4)))

# --- validity check: the untouched baseline should be near zero on every proxy ---
print("Validity check. The bf16 baseline is an unquantized model, so a high rate here means the")
print("metric is broken rather than the model.\n")
print(f"  {'language':24}{'drift':>8}{'lid':>8}{'rep':>8}{'short':>8}")
for lang in LANGS:
    f = flags[(BASE, lang)]
    lid = f"{f['lid'].mean():7.1%}" if f["lid"] is not None else "      -"
    print(f"  {M[lang]['name']:24}{f['drift'].mean():7.1%}{lid}{f['rep'].mean():7.1%}{f['short'].mean():7.1%}")

# --- change against baseline, with paired bootstrap ---
rng = np.random.default_rng(SEED)
print("\nChange against bf16, in percentage points, 95% CI from a paired bootstrap over sentences.")
print("An interval containing zero means the arm is indistinguishable from bf16 on that measure.\n")
for metric in ["drift", "rep", "lid"]:
    print(f"  {metric}")
    print(f"    {'language':24}{'tier':6}{'bf16':>7}" + "".join(f"{a.split('-')[0]:>22}" for a in ARMS[1:]))
    for lang in LANGS:
        b = flags[(BASE, lang)][metric]
        if b is None: continue
        n = len(b); idx = [rng.integers(0, n, n) for _ in range(NBOOT)]
        line = f"    {M[lang]['name']:24}{M[lang]['tier']:6}{b.mean():6.1%}"
        for arm in ARMS[1:]:
            a = flags[(arm, lang)][metric]
            pt = (a.mean() - b.mean()) * 100
            dist = np.array([(a[i].mean() - b[i].mean())*100 for i in idx])
            lo, hi = np.percentile(dist, [2.5, 97.5])
            star = "" if lo <= 0 <= hi else "*"
            line += f"  {pt:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{star:1}"
        print(line)
    print()

with open("results/fidelity.csv","w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

cands=[]
for lang in LANGS:
    b=flags[(BASE,lang)]
    for arm in ARMS[1:]:
        a=flags[(arm,lang)]
        d=[json.loads(l) for l in open(f"outputs/{arm}_{lang}.jsonl",encoding="utf-8")]
        db=[json.loads(l) for l in open(f"outputs/{BASE}_{lang}.jsonl",encoding="utf-8")]
        for i in range(len(d)):
            newly = [k for k in ("drift","rep","short") if a[k][i] and not b[k][i]]
            if newly:
                cands.append(dict(arm=arm, flores_code=lang, idx=i, newly_flagged="+".join(newly),
                                  base_hyp=clean(db[i]["hyp"])[:180], arm_hyp=clean(d[i]["hyp"])[:180],
                                  ref=d[i]["ref"][:180]))
if cands:
    with open("results/qualitative_candidates.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(cands[0].keys())); w.writeheader(); w.writerows(cands)
print(f"-> results/fidelity.csv, {len(cands)} sentences newly flagged relative to bf16")
