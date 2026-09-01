"""T4 圖表：tokenizer fertility 的三種正規化，以及它們為什麼給出不同答案。
   圖內一律用英文標籤（作品集受眾為國際）。"""
import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EN = {"eng_Latn":"English","spa_Latn":"Spanish","rus_Cyrl":"Russian","cmn_Hant":"Chinese (Trad)",
      "hin_Deva":"Hindi","arb_Arab":"Arabic","swh_Latn":"Swahili","yor_Latn":"Yoruba",
      "amh_Ethi":"Amharic","mya_Mymr":"Burmese"}
COL = {"high":"#22607F","mid":"#7A8B87","low":"#A6650F"}   # 冷色=高資源，暖色=低資源

rows = list(csv.DictReader(open("data/lang_meta.csv")))
f = lambda r,k: float(r[k])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

# --- 左：正確的度量（平行語料，同一句話需要幾個 token）---
d = sorted(rows, key=lambda r: f(r,"tok_ratio_vs_eng"))
ax1.barh([EN[r["flores_code"]] for r in d], [f(r,"tok_ratio_vs_eng") for r in d],
         color=[COL[r["tier"]] for r in d])
ax1.axvline(1.0, color="#55635F", lw=1, ls="--")
for i, r in enumerate(d):
    v = f(r,"tok_ratio_vs_eng")
    ax1.text(v+.06, i, f"{v:.2f}x  ({1/v:.0%} content/token)", va="center", fontsize=8.5, color="#14201D")
ax1.set_xlim(0, 6.3)
ax1.set_xlabel("Tokens needed for the same sentence (English = 1.0)")
ax1.set_title("Token cost on parallel text\nThe only denominator with no confound", fontsize=11, loc="left")

# --- 右：三種正規化互相矛盾 ---
d2 = sorted(rows, key=lambda r: -f(r,"tok_ratio_vs_eng"))
x = range(len(d2))
def norm(k):
    v = [f(r,k) for r in d2]; e = f(next(r for r in d2 if r["flores_code"]=="eng_Latn"), k)
    return [i/e for i in v]
ax2.plot(x, norm("tok_ratio_vs_eng"), "o-", color="#22607F", label="per parallel sentence  (correct)")
ax2.plot(x, norm("fertility_tok_per_100c"), "s--", color="#A6650F", label="per character  (biased by info density)")
ax2.plot(x, norm("fertility_tok_per_100b"), "^--", color="#96382C", label="per UTF-8 byte  (biased by encoding cost)")
ax2.set_xticks(list(x)); ax2.set_xticklabels([EN[r["flores_code"]] for r in d2], rotation=40, ha="right", fontsize=8.5)
ax2.axhline(1.0, color="#55635F", lw=1, ls="--")
ax2.set_ylabel("Fertility relative to English")
ax2.set_title("Three denominators, three different answers\nChinese looks worst per character, near best per sentence",
              fontsize=11, loc="left")
ax2.legend(fontsize=8.5, frameon=False)
for a in (ax1, ax2):
    a.spines[["top","right"]].set_visible(False); a.grid(axis="x" if a is ax1 else "y", alpha=.25)

plt.tight_layout(); plt.savefig("figures/02_fertility.png", dpi=170)
print("→ figures/02_fertility.png")
