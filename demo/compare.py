"""Side by side: the same prompt through 4-bit and through the 8-bit-embedding mitigation.

The sentence is not chosen for effect. It is FLORES devtest item 181, one of 11 of 200 Yoruba
sentences where four bits costs more than 12 chrF++ and the mitigation recovers more than 10.
The demo prints that denominator, because a single dramatic failure is an anecdote until you say
how often it happens.

Usage:  python demo/compare.py [--arms C,E] [--idx 181] [--lang yor_Latn]
"""
import argparse, json, re, sys, time, gc, shutil
import mlx.core as mx
from mlx_lm import load, stream_generate

SPECIAL = re.compile(r"<\|[^|]*\|>")
NAMES = {"A-bf16":"bf16  6.7 GB", "C-q4-g64":"4-bit  1.9 GB", "E-q4-emb8":"4-bit + 8-bit embedding  2.2 GB"}
CSI = "\033["
def at(r, c): sys.stdout.write(f"{CSI}{r};{c}H")
def clear():  sys.stdout.write(f"{CSI}2J{CSI}H")
def dim(s):   return f"{CSI}90m{s}{CSI}0m"
def bold(s):  return f"{CSI}1m{s}{CSI}0m"
def warm(s):  return f"{CSI}33m{s}{CSI}0m"
def cool(s):  return f"{CSI}36m{s}{CSI}0m"

def wrap(text, w):
    out, line = [], ""
    for word in text.split(" "):
        if len(line) + len(word) + 1 > w:
            out.append(line); line = word
        else:
            line = (line + " " + word).strip()
    out.append(line)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="C-q4-g64,E-q4-emb8")
    ap.add_argument("--idx", type=int, default=181)
    ap.add_argument("--lang", default="yor_Latn")
    a = ap.parse_args()
    arms = a.arms.split(",")

    meta = {r["flores_code"]: r for r in __import__("csv").DictReader(open("data/lang_meta.csv"))}
    row = [json.loads(l) for l in open("data/flores_10lang.jsonl", encoding="utf-8")][a.idx]
    src, ref = row["eng_Latn"], row[a.lang]
    lang_name = meta[a.lang]["name"]

    W = min(shutil.get_terminal_size((120, 40)).columns, 160)
    col = (W - 6) // 2

    clear()
    print(bold("  Tiny Aya 3.35B on an M3 MacBook Air") + dim("   ·   same prompt, two quantizations"))
    print(dim(f"  translating English into {lang_name}   ·   FLORES devtest item {a.idx}"))
    print()
    print(dim("  EN   ") + src)
    for i, l in enumerate(wrap(ref, W - 9)):
        print(dim("  ref  " if i == 0 else "       ") + dim(l))
    print()
    top = len(wrap(ref, W - 9)) + 6
    for i, arm in enumerate(arms):
        at(top, 3 + i * (col + 3)); sys.stdout.write((cool if i == 0 else warm)(bold(NAMES[arm])))
    sys.stdout.flush()

    results = {}
    for i, arm in enumerate(arms):
        c0 = 3 + i * (col + 3)
        at(top + 1, c0); sys.stdout.write(dim("loading…")); sys.stdout.flush()
        model, tok = load(f"models/{arm}")
        at(top + 1, c0); sys.stdout.write(" " * col)
        msgs = [{"role": "user", "content":
                 f"Translate the following English sentence into {lang_name}. "
                 f"Reply with the translation only, no explanation.\n\n{src}"}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        mx.clear_cache()
        # MLX tracks a process-wide high-water mark that clear_cache does not reset, so without
        # this the second arm inherits the first one's peak and both report the same number.
        mx.reset_peak_memory()
        acc, last, t0 = "", None, time.time()
        for resp in stream_generate(model, tok, text, max_tokens=200):
            acc += resp.text; last = resp
            body = SPECIAL.sub("", acc)
            for r, l in enumerate(wrap(body, col)[:9]):
                at(top + 3 + r, c0); sys.stdout.write(l.ljust(col))
            at(top + 1, c0)
            sys.stdout.write(dim(f"{resp.generation_tps:5.1f} tok/s   {resp.peak_memory:4.2f} GB peak").ljust(col + 12))
            sys.stdout.flush()
        results[arm] = dict(text=SPECIAL.sub("", acc).strip(),
                            tps=last.generation_tps, peak=last.peak_memory, secs=time.time() - t0)
        del model, tok; gc.collect(); mx.clear_cache(); mx.reset_peak_memory()

    import sacrebleu
    m = sacrebleu.CHRF(word_order=2)
    at(top + 13, 1)
    print()
    for i, arm in enumerate(arms):
        r = results[arm]
        sc = m.sentence_score(r["text"], [ref]).score
        paint = cool if i == 0 else warm
        print(f"  {paint(NAMES[arm]):<52}  chrF++ {bold(f'{sc:5.1f}')}   "
              f"{r['tps']:5.1f} tok/s   {r['peak']:.2f} GB")
    print()
    print(dim("  Item 181 is one of 11 of 200 Yoruba sentences where four bits costs more than 12 chrF++"))
    print(dim("  and the mitigation recovers more than 10. Across all 200, four bits costs 13.0% of chrF++"))
    print(dim("  and the mitigation leaves 5.4%. Both intervals exclude zero; see REPORT.md."))

if __name__ == "__main__":
    main()
