# Results

Five arms, ten languages, 1012 parallel sentences each. Per-language intervals come from a paired
bootstrap over sentences, 10,000 resamples, `seed=1337`. The numerical noise floor established before
the run is 0.3%.

## 1. 8-bit is lossless, including where it was least certain to be

Every language sits inside ±0.15% under arm B, all within the noise floor, and the fairness gap is
−0.02 pp. The open question was whether 8-bit stays lossless on languages the model knows least well.
It does.

## 2. Under 4-bit, degradation is not evenly distributed

Arm C, relative ΔBPB against bf16, ordered:

| Language | Tier | ΔBPB | 95% CI |
| --- | --- | --- | --- |
| Amharic | low | 2.08% | [1.93, 2.23] |
| Yoruba | low | 1.94% | [1.78, 2.09] |
| Swahili | low | 1.66% | [1.47, 1.87] |
| Hindi | mid | 1.60% | [1.48, 1.72] |
| Arabic | mid | 1.45% | [1.28, 1.62] |
| Spanish | high | 1.43% | [1.25, 1.61] |
| Russian | high | 1.35% | [1.14, 1.55] |
| Chinese (Traditional) | high | 0.99% | [0.82, 1.16] |
| Burmese | low | 0.58% | [0.50, 0.66] |
| English | high | 0.39% | [0.21, 0.57] |

Mean by tier: high 1.04%, mid 1.53%, low 1.56%. **English degrades least of all ten.**

**Burmese breaks the pattern.** It is the lowest resource language in the set by most accounts, it has
by far the highest tokenizer fertility, and it degrades second least. Its baseline BPB is also the
highest, meaning the model is worst at it to begin with, so a floor effect is plausible: a model that
already models a language poorly may have less to lose. This is a hypothesis, not a result.

## 3. Writing system does no work at all

The regression coefficient on non-Latin is +0.13 with p = 0.80. Latin languages average 1.35% and
non-Latin 1.34%.

This is the comparison the 2x2 language design exists to make, and it **disagrees with the closest
prior work**. Marchisio et al. (2024) found non-Latin scripts took markedly more damage on 8B to 103B
models. At 3.35B, on this language set, the effect is absent. Whether that is a scale difference, a
model difference, or a language set difference cannot be settled here.

## 4. Where the bits go matters more than how many

Two arms of nearly identical size, opposite outcomes:

| Arm | Size | Mean ΔBPB across ten languages | Fairness gap | 95% CI |
| --- | --- | --- | --- | --- |
| C, 4-bit g=64 | 1.91 GB | 1.346% | **+0.52 pp** | [+0.40, +0.65] |
| D, 4-bit g=32 | 2.11 GB | **0.612%** | **+0.67 pp** | [+0.56, +0.77] |
| E, 4-bit body, 8-bit embedding | 2.17 GB | 0.746% | **+0.26 pp** | [+0.15, +0.36] |

Arm D spends its extra half bit uniformly, on everything. It buys the largest average improvement of
any arm, and it **widens** the gap: uniform extra precision helps high resource languages more than it
helps low resource ones.

Arm E spends a comparable amount of memory on the embedding alone. It gives up some average quality
and cuts the gap roughly in half.

At 3% apart in size, these two arms represent opposite choices. The average and the gap are different
objectives, and an evaluation that reports only the average would rank D above E.

## 5. The registered hypotheses

### H2, the mitigation: supported by the registered rule

| Quantity | Value | 95% CI |
| --- | --- | --- |
| gap(C) | +0.525 pp | [+0.40, +0.65] |
| gap(E) | +0.256 pp | [+0.15, +0.36] |
| gap(C) − gap(E) | +0.269 pp | [+0.201, +0.338] |
| Shrinkage | 51.2% | [40.1, 65.5] |

Condition one, the interval on the difference excludes zero: **met, and not marginally.**
Condition two, shrinkage of at least half: **met on the point estimate, at 51.2%.**

**The honest reading.** The claim that survives without qualification is that arm E significantly
narrows the gap. The specific threshold of one half was set in advance and the point estimate clears it
by 1.2 points, while the interval runs from 40% to 65%. The defensible sentence is that arm E closes
**roughly half** the gap, between 40 and 65 percent, not that it provably closes at least half.

**The mechanism was wrong even though the recipe works.** The weight-space analysis found relative
quantization error to be uniform across languages and frequencies, which falsified the reason arm E was
supposed to work. It works anyway. Low resource languages are evidently more sensitive to embedding
perturbation than high resource ones, for reasons that are not about the size of that perturbation.
Explaining that is the obvious next piece of work.

### H1, the asymmetry: not supported by the registered test

`Δ ~ tier + script_family + fertility + baseline_score`, at the language level, n = 10.

| Term | Coefficient | 95% CI |
| --- | --- | --- |
| tier_low | +0.950 | **[−0.448, +2.348]** |
| tier_mid | +0.736 | [−0.824, +2.296] |
| non_latin | +0.135 | [−1.255, +1.525] |
| fertility | −0.255 | [−1.014, +0.503] |
| baseline | −0.110 | [−1.553, +1.333] |

Adjusted R² is −0.115, F has p = 0.60, and there are **four residual degrees of freedom**. The
coefficient on `tier_low` points the right way and its interval contains zero.

**This is a power failure, not a null.** Six parameters on ten observations cannot resolve an effect of
this size, and the preregistration required that a null be reported with the width of its interval for
exactly this reason. The interval on `tier_low` spans 2.8 percentage points, against an observed
between-tier difference of about 0.5. The design could not have detected the effect it was looking for.

The flaw is in the design and it was there from the start: a language-level regression has as many
observations as it has languages.

### What can be said, and at what scope

**Among these ten languages**, the gap under 4-bit is +0.52 pp with an interval of [+0.40, +0.65]. The
sentence bootstrap estimates that precisely, and it is the quantity arm E is shown to reduce.

**Beyond these ten languages**, nothing is established. Generalising from four low resource languages
to low resource languages in general requires between-language variation this study does not have. No
amount of sentence resampling substitutes for more languages.

Fixing it means more languages, not more sentences. Twenty-five to thirty per tier would be needed
before the language-level test could resolve an effect of this size.

## 6. Where this leaves the project

- Report both objectives. An evaluation reporting only mean degradation ranks arm D above arm E.
- Arm E is worth its 14%, and the reason it works is still unknown.
- The next experiment is not more sentences. It is more languages, and a mechanism for why embedding
  precision matters more to some languages when the weight error is the same.
