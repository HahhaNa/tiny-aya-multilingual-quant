# Methodology

## 1. Speed measurement

### 1.1 Why a protocol is needed: a real failure, shown

The day 2 smoke test was run **sequentially**, A through E, three generations per arm, no cooldown.
Against the roofline implied by memory bandwidth (about 100 GB/s on this M3):

| Arm | Weights, GB | Roofline tok/s | Measured median | Efficiency | Bandwidth extrapolation anchored on C | Shortfall |
| --- | --- | --- | --- | --- | --- | --- |
| A-bf16 | 6.72 | 14.9 | 11.8 | **79%** | 8.3 | +43% |
| B-q8-g64 | 3.58 | 27.9 | 20.1 | **72%** | 15.5 | +29% |
| C-q4-g64 | 1.91 | 52.4 | 29.1 | **56%** | 29.1 | 0% |
| D-q4-g32 | 2.11 | 47.4 | 24.2 | **51%** | 26.3 | −8% |
| E-q4-emb8 | 2.17 | 46.1 | 22.6 | **49%** | 25.6 | −12% |

Efficiency falls monotonically from 79% to 49%, **in exactly the order the arms were executed**.

Two explanations compete:

1. **Thermal throttling.** The M3 Air has no fan, and arm E ran when the machine was hottest.
2. **Dequantization overhead.** bf16 needs no dequantization and 4-bit does, so low-bit kernels have
   lower efficiency by construction.

**Sequential data cannot separate them.** They are perfectly confounded with execution order. That is
why the speed measurement in this project is interleaved, and this table is the evidence for it.

> This comes from a smoke test. It is **not a study result** and does not live in `results/`. Its only
> job is to justify the protocol.

### 1.2 The protocol

- **Interleaving.** 5 arms x 3 prompt lengths x 3 repeats, 45 runs, shuffled under `seed=1337`. The
  order is written to `bench/run_order.csv` and committed with the results. `order_idx` is carried into
  the results table so the residual order effect can be tested afterwards.
- **Cooldown.** A forced 90 second sleep after every run.
- **Environment.** On mains power, low power mode off, other applications closed,
  `sudo sysctl iogpu.wired_limit_mb=13000`.
- **Thermal record.** `powermetrics --samplers cpu_power,gpu_power,thermal -i 1000` in the background,
  reconciled afterwards. It needs root, so the run reported here was made without it; the order-effect
  test in section 1.4 is what stands in for it.
- **Statistics.** Median and IQR, never the mean, because throttling produces a one-sided tail.
- **Across languages.** tok/s is not comparable between languages and must be converted using fertility.

### 1.3 What the protocol changed

Run under the interleaved protocol, against the same roofline:

| Arm | Roofline tok/s | Median, interleaved | IQR | Efficiency, interleaved | Efficiency, sequential |
| --- | --- | --- | --- | --- | --- |
| A-bf16 | 14.9 | 13.12 | 12.94 to 13.23 | **88%** | 79% |
| B-q8-g64 | 27.9 | 24.30 | 23.69 to 24.57 | **87%** | 72% |
| C-q4-g64 | 52.4 | 43.49 | 42.70 to 47.36 | **83%** | 56% |
| D-q4-g32 | 47.4 | 40.29 | 38.97 to 40.79 | **85%** | 51% |
| E-q4-emb8 | 46.1 | 38.96 | 37.31 to 41.76 | **85%** | 49% |

The efficiency spread collapses from **30 percentage points to 5**. Almost all of what the sequential
run showed was heat, not dequantization cost.

A small gradient survives, 88% for bf16 against 83 to 85% for the 4-bit arms. That residual is the
plausible genuine cost of dequantization, and it is an order of magnitude smaller than the sequential
run implied. Reporting the sequential numbers would have overstated it roughly sixfold.

### 1.4 Testing whether the protocol worked

After interleaving, fit `decode_tps ~ arm + order_idx` on `results/speed.csv`. If the `order_idx`
coefficient is still significant, 90 seconds of cooldown was not enough and the measurement has to be
rerun with a longer one. **This test gets reported either way.**

Result: the coefficient is **−0.0080 tok/s per position, 95% CI [−0.0573, +0.0413], p = 0.746**. No
detectable residual order effect, so 90 seconds was sufficient.

### 1.5 Peak memory has to be measured in a separate process

The `peak_gb` column in `results/speed.csv` is **invalid and superseded** by
`results/peak_memory.csv`. MLX reports peak memory as a process-wide high-water mark, and
`mx.clear_cache()` does not reset it, so once the bf16 arm had been loaded every later arm in the same
process reported the same 7.25 GB. Five arms differing by more than threefold in size all reporting an
identical peak is what gave it away.

The fix is `mx.reset_peak_memory()`, now called in `bench/speed.py`, plus `bench/peak_memory.py`, which
measures each arm in its own subprocess. Peak memory is not thermally sensitive, so it needs process
isolation rather than the interleaved protocol.

| Arm | Weights | Peak with a 2048-token context | KV cache and activations |
| --- | --- | --- | --- |
| A-bf16 | 6.70 GB | 7.25 GB | 0.55 GB |
| B-q8-g64 | 3.56 GB | 4.28 GB | 0.72 GB |
| C-q4-g64 | 1.88 GB | 2.61 GB | 0.73 GB |
| D-q4-g32 | 2.09 GB | 2.82 GB | 0.73 GB |
| E-q4-emb8 | 2.15 GB | 2.88 GB | 0.73 GB |

The weight figures match `analysis/param_budget.py` exactly, which is a useful independent check on
both the derivation and the conversion.

### 1.4 Converting speed across languages

The original plan said to convert tok/s into characters per second. **Characters per second is biased
too.** A Han character carries far more than a Latin letter, so characters per second systematically
understates languages written in dense scripts.

The correct conversion uses the parallel-corpus token ratio from section 4:

```
content_per_s(L) = decode_tps(L) / tok_ratio_vs_eng(L)
```

`tok_ratio_vs_eng` is stored in `data/lang_meta.csv`. Characters per second is still recorded, as a
secondary figure.

---

## 2. Why bits per byte

```
BPB = (Σ NLL_nats / ln 2) / Σ len(utf8_bytes)
```

Perplexity has a token denominator, and token counts depend on the tokenizer. The same sentence is cut
into five times as many tokens in Burmese, which makes each token easier to predict and pushes
perplexity down artificially. With a UTF-8 byte denominator the measure no longer depends on the
tokenizer.

Perplexity is still recorded, but **only as a within-language reference**, never for cross-language
comparison.

### 2.1 What BPB fixes, and what it does not

The original plan said the byte denominator makes the measure "tokenizer independent, so languages can
be compared directly". **The first half is right and the second overclaims.**

Byte normalization does remove the **tokenizer dependence**, which is its purpose. But bytes carry the
**encoding cost of the script**: Latin runs at 1 byte per character, Cyrillic and Arabic at about 1.8,
Devanagari, Ge'ez and Myanmar at 2.6 to 2.9 (measured in section 4). The same meaning spans three times
as many bytes depending on the script, so **absolute BPB is not a cross-language quality scale**.

The dependent variable in this study is **within-language ΔBPB**, from bf16 to a quantized arm, where
the encoding cost cancels in the subtraction. No claim of the form "language A has lower absolute BPB
than language B" appears in the results.

### 2.2 The numerical noise floor of batching, measured before the run

The likelihood evaluation batches dynamically at `MAX_BATCH_TOKENS=1024`. Batching changes matmul
tiling, which changes bf16 accumulation order, so batched and unbatched results are not identical.
**This has to be measured before trusting the run, not assumed negligible.**

**First, rule out a masking bug.** If right-side padding leaked into causal attention, error would grow
monotonically with the amount of padding. Measured:

| Configuration | Relative difference against one sequence with no padding |
| --- | --- |
| One sequence + 10 padding | 1.64e-03 |
| One sequence + 50 padding | 5.03e-03 |
| One sequence + 250 padding | **6.35e-04**, smaller, so not monotonic |
| **Two identical sequences, zero padding** | **1.64e-03**, same magnitude as 10 padding |
| The same computation run twice | **Bit identical** |

A zero-padding batch already produces a difference of the same size, and error is not monotonic in
padding. That is tiling, not a leak. The computation itself is deterministic.

**Then measure the floor after aggregation** (arm C, English, 300 sentences):

| MAX_BATCH_TOKENS | BPB | Relative difference |
| --- | --- | --- |
| 1024 | 1.368635 | reference |
| 512 | 1.368512 | 9.0e-05 |
| 256 | 1.368615 | 1.5e-05 |
| 1, no batching at all | 1.367280 | **9.9e-04** |

The 5e-03 seen per sequence falls to about **1e-03, or 0.1%,** once aggregated over a thousand
sentences. Quantization is expected to move BPB by 1 to 10%, so the signal-to-noise ratio is 10 to 100
and batching is safe.

**An additional protection:** batch composition depends only on the tokenizer and the length sort, both
identical across all five arms, so the systematic component cancels in ΔBPB.

**Reporting rule, fixed in advance:** any ΔBPB smaller than 0.3% is inside the noise floor and must not
be interpreted as an effect.

---

## 3. Bit overhead of the quantization configurations

Affine quantization stores every `group_size` weights as integers plus one scale and one bias, each
16-bit, so the effective bits per weight is `bits + 32/group_size`:

- 4-bit, g=64 → 4.500
- 4-bit, g=32 → 5.000
- 8-bit, g=64 → 8.500
- Arm E, 4-bit body at g=64 with 8-bit embedding → **5.141**

These were derived by hand in `analysis/param_budget.py` and confirmed against the bits-per-weight
mlx-lm reports at conversion time. All five arms land within 1.5% of the predicted size on disk.

---

## 4. Three denominators for tokenizer fertility

FLORES+ is a parallel corpus, so the same sentence exists in all ten languages. That gives fertility
three possible denominators, and **they disagree with each other** (right panel of
`figures/02_fertility.png`):

| Denominator | Source of bias | Symptom |
| --- | --- | --- |
| Per character | Information density of a character | Traditional Chinese ranks second worst, although it is high resource and the tokenizer handles it efficiently |
| Per UTF-8 byte | Encoding cost of the script | Russian ranks last, below English, because Cyrillic spends about 1.8 bytes per character and inflates the denominator |
| **Per parallel sentence** | **None** | How many tokens the same meaning costs, directly comparable |

Measured, English as 1.0, median: Burmese 5.04x, Hindi 3.35x, Yoruba 2.12x, Amharic 2.08x,
Swahili 1.46x, Russian 1.46x, Spanish 1.31x, Arabic 1.31x, Traditional Chinese 1.19x.

Two consequences:

1. **At an equal decode rate, a Burmese user receives about a fifth of the content an English user
   does.** That inequality exists before quantization enters. The argument of this project is therefore
   that quantization adds a second layer to an existing tax, not that it creates the inequality.
2. **Fertility is not resource level.** Traditional Chinese is high resource and looks expensive per
   character; Swahili is low resource and costs the same as Russian. That is exactly what the 2x2
   language design exists to separate, and it is why `fertility` and `tier` both appear in the
   regression rather than standing in for one another.
