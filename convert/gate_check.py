"""G1 關卡：MLX 認不認得 Tiny Aya 的架構。Day 1 唯一必須回答的問題。"""
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
    print(f"[FAIL] 下載 config.json 失敗：{type(e).__name__}: {e}")
    print("→ 若是 401/403：先到 huggingface.co/%s 按下接受 CC-BY-NC 授權" % REPO)
    sys.exit(1)

print("=== config.json ===")
for k in KEYS:
    print(f"  {k:26} {cfg.get(k)}")

vocab, hid = cfg.get("vocab_size", 0), cfg.get("hidden_size", 0)
emb = vocab * hid
print(f"\n  embedding 參數     {emb/1e6:.1f}M  ({vocab} x {hid})")
print(f"  tie_word_embeddings {cfg.get('tie_word_embeddings')}"
      f"  → arm E {'一次保護 embed+lm_head' if cfg.get('tie_word_embeddings') else '要分別指定 embed 與 lm_head'}")

supported = sorted(m.name for m in pkgutil.iter_modules(M.__path__))
mt = cfg.get("model_type")
print(f"\n=== MLX 支援清單（{len(supported)} 個架構）===")
print("  " + ", ".join(supported))
hit = mt in supported
print(f"\n=== G1 結論 ===")
print(f"  model_type = {mt!r} 在清單裡？ {'是 → 走原計畫' if hit else '否 → 進備援 (a)/(b)/(c)'}")
if not hit:
    near = [s for s in supported if mt and (s.startswith(mt[:4]) or mt.startswith(s[:4]))]
    print(f"  近似候選（試別名用）: {near or '無'}")
sys.exit(0 if hit else 2)
