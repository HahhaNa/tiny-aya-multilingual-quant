"""Build the parallel corpus and measure tokenizer fertility. Touches the tokenizer and text only,
   never the model."""
import json, csv
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

REPO = "openlanguagedata/flores_plus"
MODEL = "CohereLabs/tiny-aya-global"

# Fixed in advance (PREREGISTRATION section 5). Not to be adjusted after seeing results.
LANGS = [
    # flores_code, name, script, script_family, tier, role in the design
    ("eng_Latn", "English",  "Latin",      "Latin",     "high", "control"),
    ("spa_Latn", "Spanish",  "Latin",      "Latin",     "high", "second control"),
    ("rus_Cyrl", "Russian",  "Cyrillic",   "non-Latin", "high", "high resource non-Latin; separates script from resource level"),
    ("cmn_Hant", "Chinese (Traditional)", "Han", "non-Latin", "high", "second high resource non-Latin; the only language spot checked by hand"),
    ("hin_Deva", "Hindi",    "Devanagari", "non-Latin", "mid",  "midpoint"),
    ("arb_Arab", "Arabic",   "Arabic",     "non-Latin", "mid",  "midpoint; RTL makes broken output visible"),
    ("swh_Latn", "Swahili",  "Latin",      "Latin",     "low",  "low resource Latin; the key counterexample"),
    ("yor_Latn", "Yoruba",   "Latin+tone", "Latin",     "low",  "second low resource Latin; diacritics raise fertility"),
    ("amh_Ethi", "Amharic",  "Ge'ez",      "non-Latin", "low",  "low resource non-Latin; in theory the most fragile corner"),
    ("mya_Mymr", "Burmese",  "Myanmar",    "non-Latin", "low",  "second low resource non-Latin; highest expected fertility"),
]

def load_lang(code):
    p = hf_hub_download(REPO, f"devtest/{code}.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    rows.sort(key=lambda r: r["id"])
    return rows

print("=== downloading FLORES+ devtest ===")
data = {c: load_lang(c) for c, *_ in LANGS}
ns = {c: len(v) for c, v in data.items()}
print("  sentence counts:", ns)
assert len(set(ns.values())) == 1, f"unequal sentence counts, corpus is not parallel: {ns}"
ids = [r["id"] for r in data["eng_Latn"]]
for c, v in data.items():
    assert [r["id"] for r in v] == ids, f"{c} ids do not align with English"
N = len(ids)
print(f"  ok: {N} sentences in each of 10 languages, ids fully aligned")

# write the parallel corpus
Path("data").mkdir(exist_ok=True)
with open("data/flores_10lang.jsonl", "w", encoding="utf-8") as f:
    for i in range(N):
        f.write(json.dumps({"id": ids[i], **{c: data[c][i]["text"] for c, *_ in LANGS}},
                           ensure_ascii=False) + "\n")

print("\n=== tokenizer fertility ===")
tok = AutoTokenizer.from_pretrained(MODEL)
meta = []
for code, name, script, fam, tier, role in LANGS:
    texts = [r["text"] for r in data[code]]
    n_tok = sum(len(tok.encode(t, add_special_tokens=False)) for t in texts)
    n_byte = sum(len(t.encode("utf-8")) for t in texts)
    n_char = sum(len(t) for t in texts)
    meta.append(dict(flores_code=code, name=name, script=script, script_family=fam,
                     tier=tier, region_cluster="", n_sents=N,
                     tokens=n_tok, bytes=n_byte, chars=n_char,
                     fertility_tok_per_100b=round(n_tok/n_byte*100, 2),
                     fertility_tok_per_100c=round(n_tok/n_char*100, 2),
                     bytes_per_char=round(n_byte/n_char, 2), notes=role))

with open("data/lang_meta.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(meta[0].keys())); w.writeheader(); w.writerows(meta)

base = next(m for m in meta if m["flores_code"] == "eng_Latn")["fertility_tok_per_100c"]
print(f"\n{'language':24}{'tier':6}{'tok/100 chars':>14}{'vs English':>12}{'tok/100 bytes':>15}")
for m in sorted(meta, key=lambda x: -x["fertility_tok_per_100c"]):
    print(f"{m['name']:24}{m['tier']:6}{m['fertility_tok_per_100c']:14.1f}"
          f"{m['fertility_tok_per_100c']/base:11.2f}x{m['fertility_tok_per_100b']:15.1f}")
print("\n→ data/flores_10lang.jsonl, data/lang_meta.csv")
