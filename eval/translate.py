"""L2: English into each target language, then chrF++.

Fixed template, greedy decoding, fixed sentence order, resumable per (arm, language).
max_tokens is scaled by the parallel-sentence token ratio from data/lang_meta.csv, because a
fixed budget would truncate Burmese long before it truncates English.
"""
import json, csv, time, gc, sys
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, generate

ARMS = ["A-bf16", "C-q4-g64", "E-q4-emb8"]
LANGS = ["eng_Latn","spa_Latn","rus_Cyrl","cmn_Hant","hin_Deva",
         "arb_Arab","swh_Latn","yor_Latn","amh_Ethi","mya_Mymr"]
TARGETS = [L for L in LANGS if L != "eng_Latn"]
N = 200

M = {r["flores_code"]: r for r in csv.DictReader(open("data/lang_meta.csv"))}
rows = [json.loads(l) for l in open("data/flores_10lang.jsonl", encoding="utf-8")][:N]

def build(src, lang):
    return [{"role": "user", "content":
             f"Translate the following English sentence into {M[lang]['name']}. "
             f"Reply with the translation only, no explanation.\n\n{src}"}]

def budget(lang):
    # generous headroom so truncation never masquerades as degradation
    return int(120 * float(M[lang]["tok_ratio_vs_eng"]) + 60)

def run(arm, lang, model, tok, limit=None):
    n = limit or N
    out = []
    t0 = time.time()
    for i, r in enumerate(rows[:n]):
        text = tok.apply_chat_template(build(r["eng_Latn"], lang),
                                       add_generation_prompt=True, tokenize=False)
        hyp = generate(model, tok, prompt=text, max_tokens=budget(lang), verbose=False)
        out.append({"idx": i, "src": r["eng_Latn"], "hyp": hyp.strip(), "ref": r[lang],
                    "n_tok": len(tok.encode(hyp))})
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n}  {(time.time()-t0)/(i+1):.1f}s per sentence", flush=True)
    return out

if __name__ == "__main__":
    if sys.argv[1:2] == ["validate"]:
        model, tok = load("models/C-q4-g64")
        for lang in ["spa_Latn", "mya_Mymr"]:
            print(f"=== {M[lang]['name']} ===")
            for r in run("C-q4-g64", lang, model, tok, limit=3):
                trunc = r["n_tok"] >= budget(lang) - 2
                print(f"  ref: {r['ref'][:70]}")
                print(f"  hyp: {r['hyp'][:70]}   [{r['n_tok']} tok{' TRUNCATED' if trunc else ''}]")
        sys.exit(0)

    for arm in ARMS:
        todo = [L for L in TARGETS if not Path(f"outputs/{arm}_{L}.jsonl").exists()]
        if not todo:
            print(f"[skip] {arm} complete"); continue
        print(f"=== {arm} ({len(todo)} languages) ===", flush=True)
        model, tok = load(f"models/{arm}")
        for lang in todo:
            print(f"  {lang}", flush=True)
            out = run(arm, lang, model, tok)
            with open(f"outputs/{arm}_{lang}.jsonl", "w", encoding="utf-8") as f:
                for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        del model, tok; gc.collect(); mx.clear_cache()
    print("done")
