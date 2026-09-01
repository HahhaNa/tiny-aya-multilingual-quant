"""Does mlx-lm recognise the Tiny Aya architecture? Settle this before anything else."""
import json, pkgutil, sys
from huggingface_hub import hf_hub_download
import mlx_lm.models as M

REPO = "CohereLabs/tiny-aya-global"
KEYS = ["model_type", "architectures", "hidden_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "intermediate_size",
        "vocab_size", "sliding_window", "tie_word_embeddings",
        "max_position_embeddings", "logit_scale", "rope_theta"]

try:
    cfg = json.load(open(hf_hub_download(REPO, "config.json")))
except Exception as e:
    print(f"[FAIL] could not fetch config.json: {type(e).__name__}: {e}")
    print("-> on 401/403, accept the CC-BY-NC licence at huggingface.co/%s first" % REPO)
    sys.exit(1)

print("=== config.json ===")
for k in KEYS:
    print(f"  {k:26} {cfg.get(k)}")

vocab, hid = cfg.get("vocab_size", 0), cfg.get("hidden_size", 0)
emb = vocab * hid
print(f"\n  embedding params    {emb/1e6:.1f}M  ({vocab} x {hid})")
print(f"  tie_word_embeddings {cfg.get('tie_word_embeddings')}"
      f"  -> arm E {'protects embed and lm_head at once' if cfg.get('tie_word_embeddings') else 'must name embed and lm_head separately'}")

supported = sorted(m.name for m in pkgutil.iter_modules(M.__path__))
mt = cfg.get("model_type")
print(f"\n=== architectures supported by mlx-lm ({len(supported)}) ===")
print("  " + ", ".join(supported))
hit = mt in supported
print(f"\n=== verdict ===")
print(f"  model_type = {mt!r} supported? {'yes, proceed as planned' if hit else 'no, fall back'}")
if not hit:
    near = [s for s in supported if mt and (s.startswith(mt[:4]) or mt.startswith(s[:4]))]
    print(f"  nearest candidates to try as an alias: {near or None}")
sys.exit(0 if hit else 2)
