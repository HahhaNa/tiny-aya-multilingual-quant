"""T2 驗收：五個 arm 各生成 en / zh-Hant / sw 三語，確認沒有轉壞。
   這不是評測，只是 smoke test——但它會第一次讓你看到 4-bit 在低資源語言上長什麼樣。"""
import json, time, gc
from pathlib import Path
from mlx_lm import load, generate
import mlx.core as mx

ARMS = ["A-bf16", "B-q8-g64", "C-q4-g64", "D-q4-g32", "E-q4-emb8"]
PROMPTS = {
    "en":      "Explain in three sentences why quantization can hurt some languages more than others.",
    "zh-Hant": "用三句話說明為什麼模型量化可能對某些語言的傷害比其他語言更大。",
    "sw":      "Eleza kwa sentensi tatu kwa nini upunguzaji wa modeli unaweza kudhuru lugha fulani zaidi kuliko nyingine.",
}
out, rows = ["# T2 Sanity Check\n"], []
for arm in ARMS:
    p = Path("models") / arm
    if not p.exists():
        out.append(f"\n## {arm} — 缺少模型目錄\n"); continue
    t0 = time.time(); model, tok = load(str(p)); load_s = time.time() - t0
    out.append(f"\n## {arm}  （載入 {load_s:.1f}s）\n")
    for lang, prompt in PROMPTS.items():
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        t0 = time.time()
        resp = generate(model, tok, prompt=text, max_tokens=120, verbose=False)
        dt = time.time() - t0
        n = len(tok.encode(resp))
        out.append(f"### {lang}  ({n} tok / {dt:.1f}s = {n/dt:.1f} tok/s)\n\n> {resp.strip()}\n")
        rows.append({"arm": arm, "lang": lang, "n_tok": n, "sec": round(dt, 2),
                     "tok_per_s": round(n / dt, 1), "chars": len(resp.strip())})
        print(f"{arm:12} {lang:8} {n:4}tok {n/dt:6.1f} tok/s  {resp.strip()[:60]!r}")
    del model, tok; gc.collect(); mx.clear_cache()

Path("outputs/sanity_check.md").write_text("\n".join(out), encoding="utf-8")
Path("outputs/sanity_check.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
print("\n→ outputs/sanity_check.md")
