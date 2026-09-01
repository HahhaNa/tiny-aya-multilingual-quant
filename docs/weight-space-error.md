# Quantization error in weight space is uniform

**In one line:** uniform 4-bit quantization does not disadvantage any language or any frequency of
token, measured directly in the weights.

## The result

Arm A (bf16) against arm C (4-bit, g=64), relative error `|ΔE|/|E|` per embedding row:

| Slice | Median relative error |
| --- | --- |
| Whole vocabulary | 0.0929 |
| Tokens seen in the corpus (32,088) | 0.0928 |
| Tokens never seen, the genuine low frequency tail | 0.0929 |
| High / mid / low resource tier | 0.0930 / 0.0918 / 0.0925 |
| Correlation of `log(frequency)` with error | **−0.03** |

Ninety percent of tokens fall between 0.0888 and 0.1006, a band 12.7% as wide as the median. **The
largest gap between tiers, 0.0014, is 12% of that band.**

This is a tight null, not an underpowered one.

## Why

Affine quantization takes its scale from the **min and max of each group of 64 weights**, which makes it
scale invariant. Relative error depends on the shape of the distribution inside a group, not on its
magnitude. Embedding rows have similar distribution shapes, so they receive nearly identical relative
error. Four bits, sixteen levels, applied to a roughly Gaussian group gives about 9% relative RMS error,
and 0.093 is exactly that.

**The intuition that rare tokens get quantized worse is wrong**, because per-group scaling has already
adapted to every row.

## What this does and does not say about the mitigation hypothesis

H2 as registered: degradation comes mainly from low-bit quantization of the embedding and output layer
rather than the transformer blocks.

- **Falsified:** the **weight-space premise** behind H2, that low resource and low frequency tokens
  carry larger embedding quantization error. They do not, and the evidence runs the other way.
- **Still standing:** H2 itself is a claim about **behavioural** degradation and can only be tested by
  arm E against arm C. Arm E reduces embedding error by roughly sixteen times. If low resource languages
  turn out to be more sensitive to embedding perturbation for reasons unrelated to the size of that
  perturbation, arm E can still work. **Arm E is not cut.**
- **Untouched:** H1. Behavioural asymmetry is still an open question.

### The evidence running the other way

Per-language absolute MSE, restricted to tokens that occur in essentially one language: English is
highest at 1.69e-05 with a row norm of 1.89, Amharic lowest at 5.20e-06 with a row norm of 1.13. Since
relative error is fixed, absolute perturbation is set by row norm, and **English tokens have the largest
embedding norms in this vocabulary**. Judged in weight space alone, English is the language perturbed
most.

## An aside: the GQA projections are the real hotspot

| Module | Median, layers 0 to 17 | Median, layers 18 to 35 | Max | At layer |
| --- | --- | --- | --- | --- |
| `v_proj` | 0.0937 | **0.1020** | **0.1245** | 27 |
| `k_proj` | 0.0969 | 0.0982 | 0.1112 | 31 |
| `q_proj` | 0.0946 | 0.0933 | 0.1013 | 7 |
| `o_proj` | 0.0916 | 0.0917 | 0.0923 | 5 |
| MLP, all three | ~0.093 | ~0.092 | ≤0.096 | — |

`k_proj` and `v_proj` are the 512-dimensional matrices GQA compresses down to four KV heads. They are
forced to carry denser information, they contain more outliers, and they take the highest quantization
error in the network. `v_proj` clearly degrades with depth.

This is a candidate mechanism with nothing to do with embeddings, and it suggests an experiment: an arm
keeping `k_proj` and `v_proj` at 8-bit would cost far less memory than arm E, because those matrices are
small. It is recorded as a post-hoc note rather than folded into the registered hypotheses.

## Consequences for the plan

1. **Arm E still runs.** The behavioural test of H2 has not happened yet.
2. **Arm D matters more than it did.** If group size turns out to help, the mechanism points at
   outliers, and the KV projections are where the outliers are.
3. **This figure stays in the writeup**, but its role changes. It is no longer evidence for the
   mechanism. It is an intuition falsified by its own data, and a demonstration that per-group affine
   quantization is more adaptive than it looks.
4. It should not be the lead figure. That slot belongs to the memory against quality plot.
