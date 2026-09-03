# What uniform quantization costs each language

A study of Tiny Aya 3.35B under five quantization configurations across ten languages, run
end to end on a fanless M3 MacBook Air. Hypotheses, arms and statistical tests were registered in
[`PREREGISTRATION.md`](PREREGISTRATION.md) before any evaluation ran.

---

## 1. Summary

Four-bit quantization makes this model 3.3 times faster to decode for a third of the memory. The
quality it costs is not distributed evenly: across these ten languages, low resource languages lose
roughly 2.7 times as much translation quality as high resource ones, 8.2% of chrF++ against 3.0%,
a gap of 5.2 percentage points with a bootstrap interval of [−8.3, −2.2].

Three things about that finding matter as much as the finding.

**The registered test does not support it.** The preregistered analysis works at the language level
and there are ten languages against six parameters. Its `tier` coefficient is +0.95 with a 95%
interval of [−0.45, +2.35]. The gap is real for the languages measured; ten languages cannot
establish that it generalizes to low resource languages as a class. Both statements appear in this
report because only reporting one of them would be a distortion.

**The proposed mitigation does not clearly work.** Keeping the embedding table at 8-bit costs 14%
more memory and closes 51% of the gap on the primary metric, 1.2 points past the registered
threshold with an interval reaching down to 40%, and 24% on the secondary metric with an interval
containing zero. On the evidence here it helps some languages substantially and others not at all.

**Writing system does nothing.** Marchisio et al. found non-Latin scripts took roughly three times
the damage at 8B to 103B parameters. Here Latin scripts lose 1.35% of bits per byte and non-Latin
1.34%. The language set was chosen as a 2x2 specifically so that script and resource level could be
separated, and separating them removes the script effect entirely.

---

## 2. Subject and setup

`CohereLabs/tiny-aya-global`, `model_type=cohere2`, 3.349B parameters, 36 layers, grouped query
attention with 16 query and 4 key-value heads, three sliding-window layers to every full-attention
layer. The vocabulary is 262,144 entries because the model covers seventy languages, which puts
536.9M parameters, 16.0% of the model, in the embedding table. Embeddings are tied, verified two
ways: transformers resolves the config default to true, and no `lm_head` tensor exists among the
290 in the checkpoint.

All measurement ran on one M3 MacBook Air, 16 GB unified memory, 10 GPU cores, no fan.

| Arm | Configuration | Bits per weight | Size |
| --- | --- | --- | --- |
| A | bf16 | 16.00 | 6.72 GB |
| B | 8-bit, group 64 | 8.500 | 3.58 GB |
| C | 4-bit, group 64 | 4.500 | 1.91 GB |
| D | 4-bit, group 32 | 5.000 | 2.11 GB |
| E | 4-bit body, 8-bit embedding | 5.141 | 2.17 GB |

Sizes were derived from the config before any model was downloaded and match the converted
checkpoints within 1.5%. Arm E's 5.141 bits per weight matches the hand derivation to three decimals.

Ten languages, as a 2x2 over resource tier and writing system, fixed in advance:

| | Latin | Non-Latin |
| --- | --- | --- |
| High | English, Spanish | Russian, Chinese (Traditional) |
| Mid | | Hindi, Arabic |
| Low | Swahili, Yoruba | Amharic, Burmese |

Swahili is load bearing. If low resource Latin languages held up while low resource non-Latin ones
collapsed, the driver would be script rather than resource level.

---

## 3. Metric design

### Bits per byte, and what it does not fix

Perplexity is not comparable across languages, because its denominator is tokens and token counts
depend on the tokenizer. On parallel text the same sentence costs Burmese 5.04 times as many tokens
as English, which makes each Burmese token easier to predict and drives its perplexity down for
reasons that have nothing to do with quality.

Bits per byte replaces the denominator with UTF-8 bytes, which removes the tokenizer dependence.
It does not remove the encoding cost of the script: Latin runs at 1.00 bytes per character, Cyrillic
and Arabic near 1.83, Ge'ez 2.59, Burmese 2.85. The same meaning spans nearly three times as many
bytes depending on the script, so absolute BPB is not a cross-language quality scale. The dependent
variable throughout is within-language ΔBPB, where the encoding cost cancels. This was recorded as
amendment A2 before any result existed.

### Fertility has three denominators and they disagree

Because FLORES+ is parallel, fertility can be measured per character, per byte, or per sentence of
the same meaning, and the three orderings contradict each other (figure 02). Per character,
Traditional Chinese looks like one of the worst languages, an artifact of how much a Han character
carries. Per byte, Russian looks better than English, an artifact of Cyrillic's encoding cost. Only
the parallel-sentence denominator is free of both.

The consequence is a result in its own right: **at an equal decode rate a Burmese user receives
about a fifth of the content an English user does.** That inequality exists before quantization
enters, so the question this project asks is whether quantization adds a second layer to an existing
tax, not whether it creates the inequality.

It also invalidates the plan's intended speed normalization. Characters per second carries the same
information-density bias, so cross-language speed is converted using the parallel-sentence ratio
instead. That was amendment A3.

### Every failure proxy was checked against the unquantized baseline

bf16 is not a broken model, so any measure of collapse should score near zero on it. Running the
proxies against arm A first caught four defects that would otherwise have propagated into results.

Script drift flagged **44.5%** of untouched Chinese output. The absolute threshold was counting
proper nouns, since Chinese references legitimately contain strings like 802.11n and TogiNet.
Measured against each reference's own script share for the same sentence, the baseline falls to 6.0%
(figure 04).

mlx-lm emitted its stop token as text in **5329 of 5400** generations. No text followed it, so
stripping is lossless, but it depressed every chrF++ score.

The bootstrap that rebuilds corpus chrF from per-sentence statistics passed references in the wrong
shape, producing **zero-width confidence intervals** and baselines that disagreed with a direct call.
Aggregation is now asserted against `corpus_score` on every extraction.

chrF++ adds whitespace-delimited word bigrams, which is meaningless for languages that do not
separate words with spaces. It costs Chinese 6.7 points and Burmese 8.1 against plain chrF. Both are
reported and the conclusion is identical under either.

The last of these is the same failure mode as perplexity: a metric carrying an unexamined assumption
about how writing works, and penalizing the languages that violate it.

### The numerical noise floor

Batching changes matmul tiling, which changes bf16 accumulation order. Before trusting any BPB
number this was measured rather than assumed. A padding leak was ruled out first: error is not
monotonic in padding amount, and a zero-padding batch of two identical sequences produces the same
magnitude of difference. Aggregated over 1012 sentences the floor is about 0.1%. Expected effects are
1 to 10%, and a reporting rule fixed in advance treats any ΔBPB below 0.3% as noise.

---

## 4. Results

### Speed

Forty-five runs, interleaved under a fixed seed committed before execution, 90 seconds of cooldown
between them, medians with IQR (figure 06).

| Arm | Decode tok/s | IQR | Roofline | Efficiency |
| --- | --- | --- | --- | --- |
| bf16 | 13.1 | 12.9 to 13.2 | 14.9 | 88% |
| 8-bit | 24.3 | 23.7 to 24.6 | 27.9 | 87% |
| 4-bit | 43.5 | 42.7 to 47.4 | 52.4 | 83% |
| 4-bit g32 | 40.3 | 39.0 to 40.8 | 47.4 | 85% |
| Mitigation | 39.0 | 37.3 to 41.8 | 46.1 | 85% |

The first smoke test was run sequentially and showed kernel efficiency falling monotonically from
79% to 49%, exactly in execution order. Two explanations were available, thermal throttling and
dequantization overhead, and sequential data cannot separate them. Interleaved, efficiency is flat
at 83 to 88% and the spread collapses from 30 percentage points to 5. **The decline was thermal.**
Dequantization does not measurably reduce kernel efficiency on this hardware.

The registered check for a residual order effect finds none: `order_idx` coefficient −0.008 tok/s
per run, 95% CI [−0.057, +0.041], p = 0.75. Ninety seconds of cooldown was sufficient.

### Likelihood

Eight bits is lossless on every language, low resource included, all within ±0.15% and inside the
noise floor. At four bits (figure 03):

| Language | Tier | ΔBPB | Language | Tier | ΔBPB |
| --- | --- | --- | --- | --- | --- |
| English | high | +0.39% | Arabic | mid | +1.45% |
| Burmese | low | +0.58% | Hindi | mid | +1.60% |
| Chinese | high | +0.99% | Swahili | low | +1.66% |
| Russian | high | +1.35% | Yoruba | low | +1.94% |
| Spanish | high | +1.43% | Amharic | low | +2.08% |

Tier means are 1.04%, 1.53% and 1.56%. The ordering matches the hypothesis but the low resource
group spans more internally, 0.58% to 2.08%, than it differs from the high resource group. **Burmese,
predicted to be the most fragile corner, is the second least affected language in the set.**

### Translation

| Tier | 4-bit | Mitigation |
| --- | --- | --- |
| High | −3.00% | −1.93% |
| Mid | −2.11% | −1.10% |
| Low | −8.23% | −5.92% |

The gap at four bits is 5.2 percentage points, 95% CI [−8.3, −2.2] from a paired bootstrap over
sentences. Under plain chrF it is 4.6 points, [−7.5, −1.6]. The effect is roughly three times larger
on generation than on likelihood, which is worth noting on its own: **a 1.6% rise in bits per byte
corresponds to an 8% drop in translation quality.** Cheap intrinsic metrics understate what happens
downstream even when they are correctly normalized.

### The two registered hypotheses

**H1, unequal degradation.** Not supported by its registered test. The language-level regression
`Δ ~ tier + script + fertility + baseline` yields a `tier` coefficient of +0.95 with 95% CI
[−0.45, +2.35], four residual degrees of freedom, adjusted R² negative, F-test p = 0.60. The design
carried this flaw from the start: six parameters cannot be estimated from ten observations.

The sentence-level bootstrap, which holds the language set fixed, does resolve the gap. The honest
statement is therefore split in two. Among these ten languages the gap is real and about 5 points of
chrF++. Whether it generalizes is not established by this experiment.

Two supporting results are cleaner than the headline. Script contributes nothing, 1.35% against
1.34%. And the standard objection, that low resource languages start lower so any difference is
regression to the mean, is answered directly: the correlation between relative degradation and
baseline BPB is −0.13. The covariate was preregistered to defuse that objection and the data defuses
it independently.

**H2, embedding precision as mitigation.** Not supported.

| Metric | Gap at 4-bit | Gap with mitigation | Shrinkage |
| --- | --- | --- | --- |
| ΔBPB, primary | +0.53 pp | +0.26 pp | 51%, 95% CI [40, 66] |
| ΔchrF++, secondary | −5.23 pp | −3.99 pp | 24%, 95% CI [−50, 77] |

The registered rule required an interval on the difference excluding zero and at least half the gap
closed. The primary metric meets both, by 1.2 points, with the interval on shrinkage reaching to
40%. The secondary metric fails the first condition, difference −1.24 points with 95% CI
[−3.90, +1.44]. Declaring H2 supported would mean selecting the metric that agrees.

The behaviour underneath is selective rather than absent. Yoruba recovers from −13.0% to −5.4% of
chrF++, both intervals excluding zero. Burmese moves the wrong way, −8.9% to −10.2%. Fourteen
percent more memory buys a real improvement for some languages and nothing for others, and this
experiment cannot predict which.

**The premise behind H2 was already falsified in weight space.** Relative quantization error is flat
at 0.093 across the whole vocabulary: 0.0928 for tokens in the corpus, 0.0929 for tokens never seen,
0.0930 / 0.0918 / 0.0925 by tier, correlation with log frequency −0.03. Affine quantization takes its
scale from each group of 64 weights, so it is scale invariant, and per-group scaling has already
adapted to every row. The intuition that rare tokens quantize worse is simply wrong. Whatever arm E
does for Yoruba, it does not do it by repairing embedding rows that were quantized badly.

The evidence in weight space in fact runs the other way. Since relative error is fixed, absolute
perturbation follows row norm, and English tokens carry the largest embedding norms in this
vocabulary.

### Failures that automatic metrics miss

The layer expected to be most revealing produced the least. After the metric was corrected, script
drift, language confusion and degenerate repetition give three intervals excluding zero across
roughly fifty uncorrected tests, which is what chance produces. The one coherent pattern is Yoruba
repetition: 0.5% at bf16, 3.5% at four bits with the interval excluding zero, 1.5% under the
mitigation. It points the same direction as the chrF++ result. One result among fifty is not a
finding.

At 200 sentences per language this layer cannot resolve rate changes in the low single digits. That
is a statement about the experiment rather than the model.

---

## 5. An aside on where else the inequality shows up

The language detector used for the confusion metric supports 75 languages and covers seven of these
nine. Amharic and Burmese are absent. The tooling available to measure whether a model has failed a
low resource language has the same coverage gap as the models being measured, one layer further down
the stack.

---

## 6. Limitations

Ten languages cannot support a claim about low resource languages as a class, and every interval
here that excludes zero comes from resampling sentences with the language set held fixed.

One model on one machine, so none of this generalizes on its own. The obvious next step is a second
model at similar scale.

No human evaluation, since ten native speakers were not available. Language fidelity stands in as a
cheap proxy and is underpowered at 200 sentences. Traditional Chinese is the only language the author
can spot check.

FLORES is news and encyclopedic text and carries that domain bias. Resource tier is a coarse
discretization, assigned in advance and never adjusted after seeing results.

Speed figures describe sustained performance on a fanless M3 Air. Efficiency against roofline is
uniform across bit widths here and may not be elsewhere.

---

## 7. What would sharpen this

A second model, which is the only way to separate properties of quantization from properties of Tiny
Aya. Human evaluation on three or four languages, since the automatic metrics understate damage by a
factor the literature puts near ten. More languages per tier, since the binding constraint on the
registered analysis is language count and not sentence count.

One experiment suggested by the data rather than the plan: `k_proj` and `v_proj`, the 512-dimensional
matrices grouped query attention compresses to four key-value heads, carry the highest quantization
error in the network, with `v_proj` reaching 0.1245 at layer 27 against roughly 0.092 for the MLP
blocks. Those matrices are small, so an arm keeping them at 8-bit would cost far less memory than the
embedding mitigation. It is recorded as a post-hoc note in the preregistration rather than folded
into the registered hypotheses.
