"""Establish tie_word_embeddings. The config omits it, so check the transformers default and
   whether the checkpoint actually contains an lm_head tensor. Arm E's predicate depends on this."""
import json
from huggingface_hub import hf_hub_download
REPO="CohereLabs/tiny-aya-global"

print("=== 1. what the transformers config class resolves to ===")
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(REPO)
print(f"  resolved tie_word_embeddings = {cfg.tie_word_embeddings}")

print("\n=== 2. is there a separate lm_head tensor in the checkpoint ===")
try:
    idx = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))
    keys = list(idx["weight_map"].keys())
except Exception:
    from safetensors import safe_open
    f = hf_hub_download(REPO, "model.safetensors")
    with safe_open(f, framework="np") as s: keys = list(s.keys())
emb  = [k for k in keys if "embed_tokens" in k]
head = [k for k in keys if "lm_head" in k]
print(f"  tensors in checkpoint: {len(keys)}")
print(f"  embed_tokens : {emb}")
print(f"  lm_head      : {head or 'absent, so it shares weights with the embedding'}")

tied = cfg.tie_word_embeddings and not head
print(f"\n=== verdict ===")
print(f"  tied = {tied}")
print("  arm E predicate:",
      "match embed_tokens only; that protects both input and output" if tied
      else "must match embed_tokens and lm_head separately; recompute the memory cost")
