"""Arm E, the mitigation: 4-bit body with an 8-bit embedding.

Two verified premises (see docs/architecture-support.md):
  1. tie_word_embeddings is True and the checkpoint has no lm_head, so matching embed_tokens alone
     protects both ends. MLX's cohere2 computes `out = self.model.embed_tokens.as_linear(out)`.
  2. In mlx-lm 0.31.3 the quant_predicate signature is (path, module), TWO arguments and not three.
     Call site: wrapped_predicate inside mlx_lm.utils.quantize_model.
"""
import sys
from mlx_lm.convert import convert

MATCHED = []

def keep_embeddings_high(path, module):
    """Keep the embedding, which is also the output layer, at 8-bit. Returning a dict overrides
    the quantization parameters for that module; returning True uses the defaults."""
    if "embed_tokens" in path:
        MATCHED.append(path)
        return {"group_size": 64, "bits": 8}
    return True          # defaults: 4-bit, g=64

if __name__ == "__main__":
    convert(
        "CohereLabs/tiny-aya-global",
        mlx_path="models/E-q4-emb8",
        quantize=True,
        q_bits=4,
        q_group_size=64,
        quant_predicate=keep_embeddings_high,
    )
    print(f"\n[check] modules the predicate promoted to 8-bit: {MATCHED}")
    if len(MATCHED) != 1:
        print("[FAIL] expected exactly one embed_tokens match", file=sys.stderr)
        sys.exit(1)
    print("[check] the bits per weight mlx-lm printed above should be about 5.14")
