"""Side by side: the same prompt through bf16, four bits, and the mitigation.

Live, so the throughput and memory numbers are measured on the spot rather than read from a table.
The default sentence is one the metric picked out in demo/find_examples.py, not one chosen by eye.
"""
import json, time, argparse, gc, sys
import mlx.core as mx
from mlx_lm import load, stream_generate

ARMS = [("A-bf16", "bf16", "6.72 GB"),
        ("C-q4-g64", "4-bit", "1.91 GB"),
        ("E-q4-emb8", "4-bit + 8-bit embedding", "2.17 GB")]
BOLD, DIM, RED, GRN, CYN, OFF = "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[36m", "\033[0m"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=0, help="index into demo/examples.json")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    cases = json.load(open("demo/examples.json"))
    if a.list:
        for i, c in enumerate(cases):
            print(f"  {i:2}  {c['lang']:10} chrF {c['A']:.0f} -> {c['C']:.0f} -> {c['E']:.0f}   {c['src'][:60]}")
        return
    c = cases[a.case]
    lang = {"swh_Latn":"Swahili","yor_Latn":"Yoruba","amh_Ethi":"Amharic","mya_Mymr":"Burmese",
            "rus_Cyrl":"Russian","hin_Deva":"Hindi","arb_Arab":"Arabic","cmn_Hant":"Chinese (Traditional)",
            "spa_Latn":"Spanish"}[c["lang"]]

    print(f"\n{BOLD}Translate into {lang}{OFF}")
    print(f"{DIM}{c['src']}{OFF}\n")
    print(f"{DIM}reference: {c['ref'][:110]}{OFF}\n")
    print("=" * 96)

    for arm, label, size in ARMS:
        model, tok = load(f"models/{arm}")
        prompt = tok.apply_chat_template(
            [{"role":"user","content":
              f"Translate the following English sentence into {lang}. "
              f"Reply with the translation only, no explanation.\n\n{c['src']}"}],
            add_generation_prompt=True, tokenize=False)
        mx.clear_cache()
        # peak_memory is a process-wide high-water mark, so it has to be reset per arm or every
        # arm reports whatever the largest one used.
        mx.reset_peak_memory()
        print(f"\n{BOLD}{label}{OFF}  {DIM}{size}{OFF}")
        sys.stdout.write("  ")
        t0 = time.time(); last = None; n = 0
        for r in stream_generate(model, tok, prompt, max_tokens=400):
            sys.stdout.write(r.text.replace("<|END_RESPONSE|>", "")); sys.stdout.flush()
            last = r; n += 1
        dt = time.time() - t0
        col = GRN if arm != "C-q4-g64" else RED
        peak = mx.get_peak_memory() / 1024**3
        print(f"\n  {col}{last.generation_tps:.1f} tok/s{OFF}   {DIM}peak {peak:.2f} GB"
              f"   {n} tokens in {dt:.1f}s{OFF}")
        del model, tok; gc.collect(); mx.clear_cache()

    print("\n" + "=" * 96)
    print(f"{DIM}stored chrF for this sentence: bf16 {c['A']:.0f}, four bits {c['C']:.0f}, "
          f"mitigation {c['E']:.0f}{OFF}")
    print(f"{DIM}This sentence is one of the 2.3% that collapse. Median damage across all 1800 "
          f"is 0.2 chrF.{OFF}\n")

if __name__ == "__main__":
    main()
