"""Speed results under the interleaved protocol, and the registered check on whether it worked."""
import csv, numpy as np, statsmodels.api as sm, pandas as pd

GB = {"A-bf16":6.72,"B-q8-g64":3.58,"C-q4-g64":1.91,"D-q4-g32":2.11,"E-q4-emb8":2.17}
BW = 100.0
SEQ = {"A-bf16":79,"B-q8-g64":72,"C-q4-g64":56,"D-q4-g32":51,"E-q4-emb8":49}  # the sequential smoke test

d = pd.DataFrame([{**r, "decode_tps":float(r["decode_tps"]), "prefill_tps":float(r["prefill_tps"]),
                   "ttft_ms":float(r["ttft_ms"]), "peak_gb":float(r["peak_gb"]),
                   "order_idx":int(r["order_idx"])} for r in csv.DictReader(open("results/speed.csv"))])

print("Decode throughput, interleaved, median with IQR over 9 runs per arm\n")
print(f"{'arm':12}{'GB':>6}{'roofline':>10}{'median':>9}{'IQR':>16}{'efficiency':>12}{'sequential':>12}")
eff = {}
for a in GB:
    v = d[d.arm==a].decode_tps.values
    q1,q3 = np.percentile(v,[25,75]); med = np.median(v); ceil = BW/GB[a]
    eff[a] = med/ceil*100
    print(f"{a:12}{GB[a]:6.2f}{ceil:10.1f}{med:9.2f}   [{q1:6.2f},{q3:6.2f}]{eff[a]:11.0f}%{SEQ[a]:11}%")

print("\nThe sequential column is the smoke test that motivated the protocol (METHODOLOGY 1.1).")
print("If interleaving flattens the efficiency trend, the sequential decline was thermal.")
print("If efficiency still falls with bit width, it is genuine dequantization cost.\n")
q = [eff[a] for a in ["A-bf16","B-q8-g64","C-q4-g64","D-q4-g32","E-q4-emb8"]]
s = [SEQ[a] for a in ["A-bf16","B-q8-g64","C-q4-g64","D-q4-g32","E-q4-emb8"]]
print(f"  spread, sequential   {max(s)-min(s):2d} pp   ({max(s)}% down to {min(s)}%)")
print(f"  spread, interleaved  {max(q)-min(q):.0f} pp   ({max(q):.0f}% down to {min(q):.0f}%)")

print("\n" + "="*74)
print("Registered check (METHODOLOGY 1.3): is there a residual order effect after cooldown?")
print("A significant order_idx means 90 seconds was not enough and the run must be repeated.\n")
X = pd.get_dummies(d[["arm"]], drop_first=True).astype(float)
X["order_idx"] = d["order_idx"].astype(float)
X = sm.add_constant(X)
res = sm.OLS(d["decode_tps"].astype(float), X).fit()
c = res.conf_int().loc["order_idx"]
print(f"  order_idx coefficient {res.params['order_idx']:+.4f} tok/s per position")
print(f"  95% CI [{c[0]:+.4f}, {c[1]:+.4f}]   p = {res.pvalues['order_idx']:.3f}")
print(f"  verdict: {'CONTAMINATED, rerun with longer cooldown' if res.pvalues['order_idx']<0.05 else 'no detectable order effect, the cooldown was sufficient'}")

print("\nPrefill, TTFT and peak memory, medians")
print(f"{'arm':12}{'prefill tok/s':>15}{'ttft short':>12}{'ttft long':>11}{'peak GB':>9}")
for a in GB:
    s_ = d[(d.arm==a)&(d.prompt_len=='short')]; l_ = d[(d.arm==a)&(d.prompt_len=='long')]
    print(f"{a:12}{np.median(d[d.arm==a].prefill_tps):15.1f}"
          f"{np.median(s_.ttft_ms):11.0f}ms{np.median(l_.ttft_ms):10.0f}ms"
          f"{np.median(d[d.arm==a].peak_gb):9.2f}")

# content-normalised throughput (METHODOLOGY 1.4)
meta={r["flores_code"]:r for r in csv.DictReader(open("data/lang_meta.csv"))}
print("\nContent-normalised decode rate for arm C, English-equivalent content per second")
print("(METHODOLOGY 1.4: dividing tok/s by the parallel-sentence token ratio)\n")
base=np.median(d[d.arm=="C-q4-g64"].decode_tps)
for L,m in sorted(meta.items(), key=lambda x: float(x[1]["tok_ratio_vs_eng"])):
    r=float(m["tok_ratio_vs_eng"])
    print(f"  {m['name']:24}{m['tier']:6}{base/r:7.1f}  ({100/r:3.0f}% of the English rate)")

with open("results/speed_summary.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["arm","gb","roofline_tps","median_decode_tps","iqr_low","iqr_high",
                                 "efficiency_pct","median_prefill_tps","median_peak_gb"])
    for a in GB:
        v=d[d.arm==a].decode_tps.values; q1,q3=np.percentile(v,[25,75])
        w.writerow([a,GB[a],round(BW/GB[a],1),round(np.median(v),2),round(q1,2),round(q3,2),
                    round(eff[a],1),round(np.median(d[d.arm==a].prefill_tps),1),
                    round(np.median(d[d.arm==a].peak_gb),2)])
print("\n-> results/speed_summary.csv")
