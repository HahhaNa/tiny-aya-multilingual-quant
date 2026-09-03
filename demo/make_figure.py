"""Figure 07: the failure the aggregate numbers describe, on one sentence.

Highlights the repeated segment in the four-bit output, because 'degenerate repetition' as a rate
in a table does not convey what the failure actually looks like.
"""
import json, re, csv, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

IDX, LANG = 181, "yor_Latn"
SPECIAL = re.compile(r"<\|[^|]*\|>"); clean = lambda t: SPECIAL.sub("", t).strip()
INK, MUTE, COOL, WARM, RED = "#14201D", "#55635F", "#22607F", "#A6650F", "#96382C"

D = {a: [json.loads(l) for l in open(f"outputs/{a}_{LANG}.jsonl", encoding="utf-8")][IDX]
     for a in ["A-bf16", "C-q4-g64", "E-q4-emb8"]}
src, ref = D["A-bf16"]["src"], D["A-bf16"]["ref"]

def repeated_span(text, n=4):
    """Find the longest n-gram that repeats, and the character range it covers."""
    toks = text.split()
    grams = collections.Counter(" ".join(toks[i:i+n]) for i in range(len(toks)-n+1))
    g, c = grams.most_common(1)[0]
    if c < 2: return None
    return g, c

def wrap(t, w):
    out, line = [], ""
    for word in t.split():
        if len(line) + len(word) + 1 > w: out.append(line); line = word
        else: line = (line + " " + word).strip()
    out.append(line); return out

W = 104
fig, ax = plt.subplots(figsize=(12.4, 7.4))
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
y = 0.965
def line(t, x=0.035, color=INK, size=10.5, weight="normal", family="sans-serif"):
    global y
    ax.text(x, y, t, transform=ax.transAxes, fontsize=size, color=color,
            va="top", weight=weight, family=family)
    y -= 0.045 * (size / 10.5)

line("One sentence, two quantizations", size=15, weight="bold")
line(f"FLORES devtest {IDX}, English into Yoruba. Same prompt, greedy decoding, same seed.",
     color=MUTE, size=10)
y -= 0.015
line("ENGLISH", color=MUTE, size=8.5, weight="bold")
for l in wrap(src, W): line(l, size=10.5)
y -= 0.008
line("REFERENCE", color=MUTE, size=8.5, weight="bold")
for l in wrap(ref, W): line(l, color=MUTE, size=10.5)
y -= 0.02

c_txt, e_txt = clean(D["C-q4-g64"]["hyp"]), clean(D["E-q4-emb8"]["hyp"])
rep = repeated_span(c_txt)

line("4-BIT   ·   1.9 GB   ·   chrF++ 5.4", color=WARM, size=9.5, weight="bold")
for l in wrap(c_txt, W):
    col = RED if (rep and rep[0].split()[0] in l and rep[0].split()[1] in l) else INK
    line(l, color=col, size=10.5, family="monospace")
if rep:
    line(f'the phrase "{rep[0]}" repeats {rep[1]} times', color=RED, size=9)
y -= 0.012
line("4-BIT WITH 8-BIT EMBEDDING   ·   2.2 GB   ·   chrF++ 33.5", color=COOL, size=9.5, weight="bold")
for l in wrap(e_txt, W): line(l, size=10.5, family="monospace")

y -= 0.03
line("This sentence is one of 11 of 200 where four bits costs more than 12 chrF++ and the mitigation "
     "recovers more than 10.", color=MUTE, size=9)
line("Across all 200 Yoruba sentences four bits costs 13.0% of chrF++ and the mitigation leaves 5.4%, "
     "both intervals excluding zero.", color=MUTE, size=9)
line("Across the four low resource languages the mitigation does not close the gap reliably. See REPORT.md.",
     color=MUTE, size=9)

plt.tight_layout(); plt.savefig("figures/07_demo_case.png", dpi=170, bbox_inches="tight")
print("-> figures/07_demo_case.png")
