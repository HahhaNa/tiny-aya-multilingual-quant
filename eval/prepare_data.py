"""T4 · 資料層與 tokenizer fertility。不碰模型，只碰 tokenizer 與文字。"""
import json, csv
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

REPO = "openlanguagedata/flores_plus"
MODEL = "CohereLabs/tiny-aya-global"

# 事前選定（PREREGISTRATION §5），不得依結果調整
LANGS = [
    # flores_code, 中文名, script, script_family, tier, 在設計中的角色
    ("eng_Latn", "英語",   "Latin",       "Latin",     "high", "對照組"),
    ("spa_Latn", "西班牙語", "Latin",       "Latin",     "high", "對照組第二點"),
    ("rus_Cyrl", "俄語",   "Cyrillic",    "non-Latin", "high", "高資源x非拉丁（分離文字系統與資源量）"),
    ("cmn_Hant", "繁體中文", "Han",         "non-Latin", "high", "高資源x非拉丁第二點；唯一可人工抽檢"),
    ("hin_Deva", "印地語",  "Devanagari",  "non-Latin", "mid",  "中段橋樑"),
    ("arb_Arab", "阿拉伯語", "Arabic",      "non-Latin", "mid",  "中段橋樑；RTL"),
    ("swh_Latn", "斯瓦希里語","Latin",      "Latin",     "low",  "低資源x拉丁（關鍵反例）"),
    ("yor_Latn", "約魯巴語", "Latin+聲調",  "Latin",     "low",  "低資源x拉丁第二點"),
    ("amh_Ethi", "阿姆哈拉語","Ge'ez",      "non-Latin", "low",  "低資源x非拉丁（理論最脆弱）"),
    ("mya_Mymr", "緬甸語",  "Myanmar",     "non-Latin", "low",  "低資源x非拉丁第二點；預期最高 fertility"),
]

def load_lang(code):
    p = hf_hub_download(REPO, f"devtest/{code}.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    rows.sort(key=lambda r: r["id"])
    return rows

print("=== 下載 FLORES+ devtest ===")
data = {c: load_lang(c) for c, *_ in LANGS}
ns = {c: len(v) for c, v in data.items()}
print("  句數:", ns)
assert len(set(ns.values())) == 1, f"句數不一致，非平行語料: {ns}"
ids = [r["id"] for r in data["eng_Latn"]]
for c, v in data.items():
    assert [r["id"] for r in v] == ids, f"{c} 的 id 序列與英語不一致"
N = len(ids)
print(f"  ✓ 10 語言各 {N} 句，id 完全對齊（平行語料）")

# 平行語料落檔
Path("data").mkdir(exist_ok=True)
with open("data/flores_10lang.jsonl", "w", encoding="utf-8") as f:
    for i in range(N):
        f.write(json.dumps({"id": ids[i], **{c: data[c][i]["text"] for c, *_ in LANGS}},
                           ensure_ascii=False) + "\n")

print("\n=== 計算 tokenizer fertility ===")
tok = AutoTokenizer.from_pretrained(MODEL)
meta = []
for code, name, script, fam, tier, role in LANGS:
    texts = [r["text"] for r in data[code]]
    n_tok = sum(len(tok.encode(t, add_special_tokens=False)) for t in texts)
    n_byte = sum(len(t.encode("utf-8")) for t in texts)
    n_char = sum(len(t) for t in texts)
    meta.append(dict(flores_code=code, name_zh=name, script=script, script_family=fam,
                     tier=tier, region_cluster="", n_sents=N,
                     tokens=n_tok, bytes=n_byte, chars=n_char,
                     fertility_tok_per_100b=round(n_tok/n_byte*100, 2),
                     fertility_tok_per_100c=round(n_tok/n_char*100, 2),
                     bytes_per_char=round(n_byte/n_char, 2), notes=role))

with open("data/lang_meta.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(meta[0].keys())); w.writeheader(); w.writerows(meta)

base = next(m for m in meta if m["flores_code"] == "eng_Latn")["fertility_tok_per_100c"]
print(f"\n{'語言':12}{'tier':6}{'tok/100字元':>12}{'vs 英語':>9}{'tok/100位元組':>14}")
for m in sorted(meta, key=lambda x: -x["fertility_tok_per_100c"]):
    print(f"{m['name_zh']:12}{m['tier']:6}{m['fertility_tok_per_100c']:12.1f}"
          f"{m['fertility_tok_per_100c']/base:8.2f}x{m['fertility_tok_per_100b']:14.1f}")
print("\n→ data/flores_10lang.jsonl, data/lang_meta.csv")
