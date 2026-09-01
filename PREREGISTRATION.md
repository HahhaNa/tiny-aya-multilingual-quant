# Preregistration

**Registered 2026-09-01.** The commit timestamp is the record. Written before any evaluation result
existed.

The hypotheses and statistical tests below must not be changed because of what the numbers turn out
to be. Anything thought of after seeing results is appended under `[POST-HOC]` and reported separately
from the registered analysis.

---

## 1. Question

Does uniform 4-bit quantization degrade a 3.35B multilingual edge model unequally across languages,
when that model was deliberately designed for low resource languages? If it does, where is the
mechanism, and can it be mitigated cheaply?

## 2. Subject, as verified on day one

`CohereLabs/tiny-aya-global`, `model_type=cohere2`, 3.349B parameters, 36 layers, GQA with 16 query
heads and 4 KV heads, three sliding-window (4096) layers to every full-attention layer, vocabulary
262,144, embedding 536.9M parameters which is **16.0% of the model**, `tie_word_embeddings=True` with
no separate `lm_head` tensor in the checkpoint.

## 3. Hypotheses

### H1, primary

Going from bf16 (arm A) to 4-bit (arm C), **low resource languages degrade more than high resource
ones**, and the difference survives controlling for the fact that their baseline scores are lower to
begin with.

- **Dependent variables:** ΔBPB (primary), ΔchrF++ (secondary)
- **Test:** OLS regression, `Δ ~ tier + script_family + fertility + baseline_score`
- **Decision rule:** the 95% CI on the `tier` coefficient (low against high) excludes zero, in the
  direction of greater degradation for low resource languages
- **Falsified if:** that CI crosses zero

### H2, mechanism

Degradation comes mainly from low-bit quantization of the embedding and output layer rather than the
transformer blocks. Keeping the embedding at 8-bit (arm E, 14% more memory) should therefore close
most of the gap between languages.

- **Dependent variable:** fairness gap = mean degradation of high resource languages minus mean
  degradation of low resource ones
- **Test:** compare the gap for arm C against arm E, paired bootstrap with 10,000 resamples,
  `seed=1337`, CI on the difference
- **Decision rule:** the 95% CI on `gap(C) − gap(E)` excludes zero and arm E closes at least **half**
  the gap
- **Falsified if:** the CI crosses zero, or the gap closes by less than half
- **Note:** H2 can fail while H1 holds. That would mean the mechanism is not in the embedding, and the
  next comparison is arm C against arm D. If group size is what matters, the mechanism points at
  activation outliers instead.

### H0, the null is also a result

Tiny Aya may show no such asymmetry, precisely because it was built for low resource languages, with a
language-bucketed tokenizer and balanced training data.

- **Why that is still worth reporting:** the literature contradicts itself. Marchisio et al. (2024)
  found non-Latin scripts take more damage; arXiv 2503.03592 found English-calibrated K-quantization
  does not disproportionately hurt multilingual performance. A clean null **with a narrow confidence
  interval**, on a model deliberately designed for these languages, is direct evidence about whether
  model design can protect them.
- **Reporting requirement:** any null must be reported together with the width of the effect size CI.
  A wide interval means insufficient power, not a null.

## 4. Arms

| Arm | Configuration | Predicted size | What it alone answers |
| --- | --- | --- | --- |
| A | bf16 | 6.70 GB | Baseline for every delta |
| B | 8-bit, g=64 | 3.56 GB | Is 8-bit really near lossless, in low resource languages too |
| C | 4-bit, g=64 | 1.88 GB | **Primary treatment**, the configuration that gets deployed |
| D | 4-bit, g=32 | 2.09 GB | Separates bit depth from quantization granularity |
| E | 4-bit body, 8-bit embedding | 2.15 GB | **Mitigation**, at 14% more memory |
| F/G | AWQ, English against ten-language calibration | — | **Optional**, only if the day 7 review leaves room |

## 5. Languages

Chosen as a 2x2 before any measurement, and not to be adjusted afterwards.

| Language | FLORES code | Script | Tier | Role in the design |
| --- | --- | --- | --- | --- |
| English | eng_Latn | Latin | high | Control |
| Spanish | spa_Latn | Latin | high | Second control |
| Russian | rus_Cyrl | Cyrillic | high | **High resource, non-Latin.** Separates script from resource level |
| Chinese (Traditional) | cmn_Hant | Han | high | Second high resource non-Latin; the only language the author can spot check |
| Hindi | hin_Deva | Devanagari | mid | Midpoint |
| Arabic | arb_Arab | Arabic, RTL | mid | Midpoint; RTL layout makes broken output easy to see |
| Swahili | swh_Latn | Latin | low | **Low resource, Latin.** The key counterexample |
| Yoruba | yor_Latn | Latin with tone marks | low | Second low resource Latin; diacritics raise tokenizer fertility |
| Amharic | amh_Ethi | Ge'ez | low | Low resource, non-Latin. In theory the most fragile corner |
| Burmese | mya_Mymr | Myanmar | low | Second low resource non-Latin; complex orthography, highest expected fertility |

`tier` is assigned in advance from FLORES-200 and Joshi et al., never inferred backwards from observed
baseline scores.

## 6. Metrics

| Level | Metric | Role |
| --- | --- | --- |
| L0 | Weight-space quantization error, per-token embedding MSE | Mechanism evidence for H2, needs no inference |
| L1 | **ΔBPB** on FLORES+ devtest | **Primary dependent variable.** `BPB = (Σ NLL / ln 2) / Σ bytes` |
| L2 | ΔchrF++, English into X, 200 sentences | Secondary, arms A, C and E only |
| L3 | Script drift, language confusion, degenerate repetition | Cheap proxy for the failures automatic metrics miss |
| L4 | Δ Global-MMLU | **Optional**, first thing cut on day 7 |
| L5 | Decode and prefill tok/s, TTFT, content per second, peak memory | The other half of the tradeoff |

**Explicitly excluded:** perplexity as a cross-language comparison. Tokenizer fertility makes its
denominator incomparable. It is recorded only as a within-language reference quantity.

## 7. Statistical procedure

1. **Paired bootstrap**, 10,000 resamples, `seed=1337`, over the same sentences, for the 95% CI on
   ΔchrF++
2. **Primary regression** `Δ ~ tier + script_family + fertility + baseline_score`, OLS
3. **Report absolute and relative degradation both**, and state plainly that they tell different
   stories
4. Every per-language result carries a CI or an IQR. No bar chart without error bars
5. Speed is reported as **median and IQR**, never mean, because thermal throttling produces a
   one-sided tail

## 8. Controls against measurement contamination, fixed in advance

- The 45 speed runs (5 arms x 3 prompt lengths x 3 repeats) are **shuffled under `seed=1337`**, never
  executed arm by arm
- 90 seconds of forced cooldown between runs, with `powermetrics` recorded in the background
- Declared in advance: if the same arm varies by more than 20% between runs, that is reported
  explicitly and the median is used

## 9. Stopping rule and scope reduction

The day 7 review cuts in this **preassigned** order, never according to which result looks better:

> L4 Global-MMLU → AWQ ablation → arm D → L2 languages from ten to six → L2 sentences from 200 to 100

**Never cut:** the full L1 BPB matrix, L3 fidelity, arm E, the speed protocol.

## 10. Known limitations, admitted in advance

1. One model, one machine, so the conclusion does not generalize
2. No human evaluation, since ten native speakers were not available. L3 is a proxy, not a gold
   standard, and Traditional Chinese is the only language the author can check
3. FLORES is news and encyclopedic text and carries that domain bias
4. Resource tier is a coarse discretization
5. The M3 Air has no fan, so speed figures describe sustained performance on this machine rather than
   an architectural ceiling

---

## Amendments (identifier level, not hypothesis level)

### A1 · 2026-09-01 · FLORES code for Chinese
**Change:** `zho_Hant` becomes `cmn_Hant`.
**Reason:** FLORES+ uses ISO 639-3 individual language codes, replacing the FLORES-200 era macrolanguage
codes. Same data, same language, same script; the identifier changed. The 221 devtest languages include
`cmn_Hans`, `cmn_Hant` and `yue_Hant`.
**State at the time:** `results/` was empty. No evaluation number existed.
**Not a hypothesis change:** H1, H2, H0, the dependent variables, the tests, the tier assignment and the
stopping rule are all untouched.

### A2 · 2026-09-01 · Clarifying what BPB claims
**Change:** the scope of BPB is stated explicitly. No hypothesis or test is modified.
**Reason:** measurement on day 4 showed UTF-8 bytes per character varies about threefold across the ten
languages, from 1.00 for Latin to 2.85 for Burmese. **Absolute** BPB therefore carries the encoding cost
of the script and is not a cross-language quality scale.
**Clarification:** the dependent variable was always **within-language ΔBPB relative to arm A**, where
the encoding cost cancels. The report will not contain any statement of the form "language A has lower
absolute BPB than language B and is therefore better served".
**State at the time:** `results/` was empty.

### A3 · 2026-09-01 · Cross-language speed normalization
**Change:** characters per second becomes a secondary figure; the primary conversion is
`decode_tps / tok_ratio_vs_eng`.
**Reason:** characters per second is contaminated by information density per character, since a Han
character carries far more than a Latin letter. The parallel corpus supplies a denominator with no such
confound.
**State at the time:** `results/` was empty and the speed measurement had not been run.

---

## [POST-HOC]

> Everything below was thought of **after** seeing data. It is kept separate from the registered
> hypotheses and must be labelled exploratory in any report. It is not a registered test.

### P1 · 2026-09-01 · Sequence length as a candidate mechanism
Weight-space error turned out to be essentially constant across languages and frequencies, at a
relative error of 0.093. So if the behavioural evaluation does find asymmetry, it **cannot** be
explained by low resource tokens carrying larger weight error.

An alternative mechanism falls out of this project's own data. The same sentence costs Burmese 5.04
times as many tokens as English. More tokens means more forward passes, which means more opportunities
for quantization error to compound.

**Prediction:** controlling for tier, `fertility` should still correlate with degradation. The primary
regression already includes a `fertility` term, so this is visible in the registered analysis, but the
**causal reading of it is exploratory**.

### P2 · 2026-09-01 · The GQA key and value projections are the error hotspot
`v_proj` and `k_proj`, the 512-dimensional matrices compressed to four KV heads, carry the highest
quantization error in the network. `v_proj` has a median of 0.1020 in the deep layers and a maximum of
0.1245, against roughly 0.092 for the MLP blocks.

**Possible experiment:** an arm that keeps `k_proj` and `v_proj` at 8-bit. Those matrices are small, so
the memory cost would be far below arm E's.
**Optional**, ranked below the five registered arms, only if the day 7 review leaves room.
