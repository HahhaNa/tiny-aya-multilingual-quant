# Does MLX support the Tiny Aya architecture

**Conclusion: yes, natively. No fallback needed.**

This was the first thing to settle, because if MLX could not run the architecture the whole plan would
have had to change on day one.

## How it was established

`CohereLabs/tiny-aya-global` is gated, so fetching `config.json` directly returns 403. The config was
read instead from the **ungated MLX mirror** `mlx-community/tiny-aya-global-8bit-mlx`. The existence of
that repository is itself evidence that MLX can run the architecture.

The gate was accepted shortly afterwards and the table below uses the original repository's values.

## Key fields

| Field | Value | Why it matters |
| --- | --- | --- |
| `model_type` | `cohere2` | **Present in the 118 architectures mlx-lm 0.31.3 supports** |
| `architectures` | `Cohere2ForCausalLM` | Same family as Command R7B |
| `vocab_size` x `hidden_size` | 262144 x 2048 | Embedding is 536.9M parameters, **16.0% of the model** |
| `num_hidden_layers` | 36 | Body is 2.812B, total 3.349B, matching the stated 3.35B |
| GQA | 16 query heads, 4 KV heads, head_dim 128 | KV cache is a quarter of what MHA would need |
| `layer_types` | 3 sliding (4096) then 1 full, repeated 9 times | The `cohere2` pattern |
| `max_position_embeddings` | **8192** | The mirror says 500000. **The original says 8192** |
| `tie_word_embeddings` | Absent from config, **verified True** | See below |

### On tied embeddings

The field is not in `config.json`, so it had to be established rather than assumed, because arm E
depends on it. `convert/check_tie.py` checks it two independent ways: transformers resolves the config
default to `True`, and **none of the 290 tensors in the checkpoint is `lm_head.weight`**. MLX's cohere2
implementation computes the output as `self.model.embed_tokens.as_linear(out)`, so the output layer is
literally the embedding table.

The consequence is that arm E's quantization predicate only has to match `embed_tokens`. One module,
both ends protected.

## Derived weight budget

`analysis/param_budget.py` derives these from the config alone, with no model loaded.

| Arm | Bits per weight | GB | Against C |
| --- | --- | --- | --- |
| A-bf16 | 16.00 | 6.70 | 3.56x |
| B-q8-g64 | 8.50 | 3.56 | 1.89x |
| C-q4-g64 | 4.50 | 1.88 | 1.00x |
| D-q4-g32 | 5.00 | 2.09 | 1.11x |
| E-q4-emb8 | 5.14 | 2.15 | **1.14x** |

Arm E's "only 14% more memory" is a derived number, not a quoted one. After conversion, mlx-lm reported
5.141 bits per weight for arm E and every arm landed within 1.5% of its predicted size on disk.

## A lesson about mirrors

A mirror repository is good enough to answer "does the framework support this architecture". It is not
good enough to read configuration values from: `max_position_embeddings` differs between the mirror and
the original.

## Where this leaves the project

`mlx-community` already publishes 8-bit and 4-bit conversions of Tiny Aya. **Quantizing this model is
not novel.** Whatever this project contributes has to come from the cross-language evaluation, the
mitigation arm, and the measurement protocol, which is a useful thing to know on day one rather than
day fourteen.
