# Tiny Aya, ten languages, four bits

**What uniform 4-bit quantization costs each of ten languages on a fanless laptop, and whether a small change to the recipe can pay some of it back.**

> **Status: in progress.** Days 1 to 5 of a 14 day sprint are done. The behavioural evaluation is
> running now, so this README reports method and the weight level results only. Nothing here claims
> a quality finding yet. Hypotheses were registered in [`PREREGISTRATION.md`](PREREGISTRATION.md)
> before any evaluation was run, and the commit history shows the ordering.

---

## The question

[Tiny Aya](https://huggingface.co/CohereLabs/tiny-aya-global) is a 3.35B multilingual model from
Cohere Labs built to fit on a phone, and Cohere ships GGUF quantizations of it. In its real
deployment setting, quantization is not optional. It is the default state.

Almost every quantization evaluation is done in English. This model exists for the seventy languages
that nobody evaluates.

There is a specific mechanism worth suspecting. The vocabulary is 262,144 entries wide because it has
to cover those seventy languages, so the embedding table alone is 537M parameters, sixteen percent of
the whole model. A lot of what the model knows about low resource languages lives in the sparse,
low frequency rows of that table. Does uniform 4-bit quantization treat those rows the same way it
treats the rest? That is directly measurable.

## Design

Five quantization arms, chosen so that each one answers a question the others cannot.

| Arm | Configuration | Size | What it isolates |
| --- | --- | --- | --- |
| A | bf16 | 6.72 GB | Baseline. Every delta is measured against it |
| B | 8-bit, g=64 | 3.58 GB | Is 8-bit really near lossless, in low resource languages too |
| C | 4-bit, g=64 | 1.91 GB | The configuration that actually gets deployed |
| D | 4-bit, g=32 | 2.11 GB | Separates bit depth from quantization granularity |
| E | 4-bit body, 8-bit embedding | 2.17 GB | Mitigation, at 14% more memory than C |

Ten languages, picked as a 2x2 so that resource level can be separated from writing system. Those two
factors are correlated in practice, and a reviewer will ask which one is doing the work.

| | Latin | Non-Latin |
| --- | --- | --- |
| **High resource** | English, Spanish | Russian, Chinese (Traditional) |
| **Mid** | | Hindi, Arabic |
| **Low resource** | Swahili, Yoruba | Amharic, Burmese |

Swahili is the load bearing choice. If low resource Latin languages degrade little while low resource
non-Latin ones degrade a lot, the driver is the script and not the resource level.

## What is done, and one result that came out empty

### Token cost is not distributed evenly, before quantization enters

![fertility](figures/02_fertility.png)

On parallel text, the same sentence costs Burmese about five times as many tokens as English. At an
equal decode rate, a Burmese user receives roughly a fifth of the content per second.

The right panel is the methodological point. Fertility has three plausible denominators and they
disagree. Per character, Chinese looks like one of the worst languages, but that is an artifact of
how much meaning a Han character carries. Per UTF-8 byte, Russian looks better than English, because
Cyrillic spends about 1.8 bytes per character and inflates the denominator. Only the parallel
sentence denominator is free of both.

This also bounds what bits-per-byte can claim, which is written up in
[`METHODOLOGY.md`](METHODOLOGY.md) section 2.1.
Byte normalization removes the tokenizer dependence, which is its job, but bytes still carry the
encoding cost of the script. Absolute BPB is therefore not a cross language quality scale. The
dependent variable here is within language ΔBPB, where the encoding cost cancels.

### Uniform quantization does not disadvantage rare tokens

![weight error](figures/01_weight_error.png)

The starting hypothesis was that rare tokens, which is where low resource language knowledge lives,
would be quantized worse. Measured directly in weight space, they are not.

Relative error `|ΔE|/|E|` is essentially constant at 0.093. In corpus tokens 0.0928, out of corpus
tokens 0.0929. High, mid and low resource tiers 0.0930, 0.0918, 0.0925. The correlation between
log frequency and error is −0.03. Ninety percent of tokens fall in a band 12.7% wide, and the largest
gap between tiers is 12% of that band, so this is a tight null rather than an underpowered one.

The reason is that affine quantization takes its scale from the min and max of each group of 64
weights, which makes it scale invariant. Relative error depends on the shape of the distribution
inside a group, not its magnitude. Per group scaling has already adapted to each row.

The evidence in fact runs the other way. Since relative error is fixed, absolute perturbation is set
by row norm, and English tokens have the largest embedding norms in this vocabulary.

This falsifies the weight space premise behind the mitigation hypothesis. It does not falsify the
mitigation itself, which is a claim about behaviour and is tested by arm E against arm C. Arm E still
runs. Full write up in [`docs/T3_FINDINGS.md`](docs/T3_FINDINGS.md).

One thing did stand out. The GQA key and value projections, the 512 dimensional matrices compressed
to four KV heads, carry the highest quantization error in the network, and `v_proj` gets worse with
depth, reaching 0.1245 at layer 27 against roughly 0.092 for the MLP blocks. That is a candidate
mechanism with nothing to do with embeddings, recorded as a post-hoc note rather than folded into the
registered hypotheses.

## Measuring speed on a machine with no fan

The first smoke test was run sequentially, A through E. Kernel efficiency against the memory
bandwidth roofline fell monotonically, 79%, 72%, 56%, 51%, 49%, in exactly the order the arms were
executed.

That is not enough to call it thermal throttling. Dequantization overhead also lowers efficiency at
lower bit widths, and sequential execution confounds the two beyond separation. The real measurement
protocol therefore shuffles all 45 runs under a fixed seed, forces a 90 second cooldown between them,
records `powermetrics` in the background, keeps `order_idx` in the results table so the residual
order effect can be tested afterwards, and reports medians with IQR rather than means.

Batching for the likelihood evaluation was checked the same way. Batched and unbatched results differ
by about 5e-3 per sequence, which had to be shown to be tiling rather than a padding leak: error is
non monotonic in padding amount, and a zero padding batch of two identical sequences produces the same
magnitude of difference. Aggregated over a thousand sentences the floor is about 1e-3, and any ΔBPB
below 0.3% is treated as noise. Both checks are in [`METHODOLOGY.md`](METHODOLOGY.md).

## Reproducing

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install mlx-lm huggingface_hub sacrebleu datasets pandas matplotlib scipy statsmodels lingua-language-detector
hf auth login          # accept the licences on the two gated pages first

python convert/gate_check.py     # confirms mlx-lm supports the cohere2 architecture
python convert/check_tie.py      # confirms tied embeddings, which arm E depends on
bash   convert/make_arms.sh      # arms A to D
python convert/arm_e.py          # arm E, custom quant predicate
python eval/prepare_data.py      # FLORES+ parallel corpus and fertility
python analysis/weight_error.py  # L0, needs no inference
python eval/bpb.py               # L1, resumable per arm and language
```

`analysis/param_budget.py` derives the arm sizes from the config alone. Its predictions match the
converted models to within 1.5%, and its 5.141 bits per weight for arm E matches what mlx-lm reports.

## Limitations

One model, one machine, so nothing here generalizes on its own. No human evaluation, because ten
native speakers were not available, which is why script fidelity is used as a cheap proxy and why
Traditional Chinese is the only language spot checked by the author. FLORES is news and encyclopedic
text and carries that domain bias. Resource tier is a coarse discretization. Speed numbers describe
sustained performance on a fanless M3 Air, not an architectural ceiling.

## Prior work

The closest predecessor is Cohere's own
[How Does Quantization Affect Multilingual LLMs?](https://arxiv.org/abs/2407.03211)
(Marchisio et al., EMNLP Findings 2024), which covered 8B to 103B models and 23 languages and found
that automatic metrics badly understate the damage. The literature disagrees with itself:
[English K-quantization does not disproportionately diminish multilingual performance](https://arxiv.org/abs/2503.03592)
reaches the opposite conclusion on a different method and scale. This project sits at a smaller scale,
on lower resource languages, and tests a mitigation rather than only measuring.

## Licences

Tiny Aya is CC-BY-NC, research use only. FLORES+ is CC-BY-SA 4.0. No model weights or corpus files are
redistributed in this repository. Code here is MIT.
