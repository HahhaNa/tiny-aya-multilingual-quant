"""Bits per byte, the primary continuous metric.

BPB = (sum NLL_nats / ln 2) / sum len(utf8_bytes)
  - NLL covers content tokens only; a BOS is prepended so every content token is predicted
  - the byte denominator removes the tokenizer dependence
  - absolute BPB is not compared across languages (see amendment A2); the dependent variable is
    within-language delta BPB

Resumable: one results/parts/bpb_{arm}_{lang}.json per pair, skipped if it already exists.
"""
import json, math, sys, time, gc
from pathlib import Path
import mlx.core as mx
from mlx_lm import load

ARMS = ["A-bf16", "C-q4-g64", "E-q4-emb8", "D-q4-g32", "B-q8-g64"]  # priority order
LANGS = ["eng_Latn","spa_Latn","rus_Cyrl","cmn_Hant","hin_Deva",
         "arb_Arab","swh_Latn","yor_Latn","amh_Ethi","mya_Mymr"]
MAX_BATCH_TOKENS = 1024      # bounds logits memory: 1024 x 262144 x 2 bytes is about 537 MB

rows = [json.loads(l) for l in open("data/flores_10lang.jsonl", encoding="utf-8")]

def seq_nll(model, ids_batch, lens):
    """Total NLL in nats per sequence. ids_batch is right-padded."""
    x = mx.array(ids_batch)
    logits = model(x[:, :-1]).astype(mx.float32)          # (B, T-1, V)
    tgt = x[:, 1:]
    lse = mx.logsumexp(logits, axis=-1)
    gat = mx.take_along_axis(logits, tgt[..., None], axis=-1).squeeze(-1)
    nll = lse - gat                                        # (B, T-1)
    T = nll.shape[1]
    pos = mx.arange(T)[None, :]
    mask = pos < (mx.array(lens)[:, None] - 1)             # each sequence yields len-1 predictions
    out = mx.sum(nll * mask, axis=1)
    mx.eval(out)
    return [float(v) for v in out]

def run(arm, lang, model, tok, bos):
    texts = [r[lang] for r in rows]
    enc = [([bos] if bos is not None else []) + tok.encode(t, add_special_tokens=False) for t in texts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))   # sort by length to limit padding
    per_nll = [0.0]*len(texts); per_byte = [len(t.encode("utf-8")) for t in texts]
    per_tok = [0]*len(texts)
    tot_nll = 0.0; tot_tok = 0; tot_byte = sum(per_byte)
    i = 0; t0 = time.time()
    while i < len(order):
        b = []
        while i < len(order):
            cand = b + [order[i]]
            if b and max(len(enc[j]) for j in cand) * len(cand) > MAX_BATCH_TOKENS: break
            b.append(order[i]); i += 1
        L = max(len(enc[j]) for j in b)
        pad = tok.pad_token_id or 0
        ids = [enc[j] + [pad]*(L-len(enc[j])) for j in b]
        lens = [len(enc[j]) for j in b]
        for idx, n, l in zip(b, seq_nll(model, ids, lens), lens):
            tot_nll += n; tot_tok += l - 1
            per_nll[idx] = n; per_tok[idx] = l - 1
        mx.clear_cache()
    bpb = (tot_nll / math.log(2)) / tot_byte
    # Per-sentence values, needed for the paired bootstrap that gives per-language CIs.
    Path(f"results/parts/sent_{arm}_{lang}.json").write_text(json.dumps(
        {"arm": arm, "flores_code": lang,
         "nll": [round(v, 6) for v in per_nll], "bytes": per_byte, "tokens": per_tok}))
    return dict(arm=arm, flores_code=lang, n_sents=len(texts),
                sum_nll_nats=round(tot_nll, 4), sum_bytes=tot_byte, sum_tokens=tot_tok,
                bpb=round(bpb, 6), ppl_ref=round(math.exp(tot_nll / tot_tok), 4),
                sec=round(time.time()-t0, 1))

if __name__ == "__main__":
    only = sys.argv[1:] or ARMS
    for arm in only:
        todo = [L for L in LANGS if not Path(f"results/parts/sent_{arm}_{L}.json").exists()]
        if not todo:
            print(f"[skip] {arm} already complete"); continue
        print(f"=== {arm} ({len(todo)} languages to run) ===", flush=True)
        model, tok = load(f"models/{arm}")
        bos = tok.bos_token_id
        for L in todo:
            r = run(arm, L, model, tok, bos)
            Path(f"results/parts/bpb_{arm}_{L}.json").write_text(json.dumps(r))
            print(f"  {L:10} BPB {r['bpb']:.4f}  ppl {r['ppl_ref']:8.2f}  "
                  f"{r['sum_tokens']:6}tok  {r['sec']:6.1f}s", flush=True)
        del model, tok; gc.collect(); mx.clear_cache()
