"""確定 tie_word_embeddings。config 沒寫 => 要看 transformers 的預設 + 權重檔實際有沒有 lm_head。
   這決定 arm E 的 quant_predicate 要匹配幾個模組。"""
import json
from huggingface_hub import hf_hub_download
REPO="CohereLabs/tiny-aya-global"

print("=== 1. transformers 的 config 類別預設 ===")
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(REPO)
print(f"  解析後 tie_word_embeddings = {cfg.tie_word_embeddings}")

print("\n=== 2. 權重檔裡到底有沒有獨立的 lm_head ===")
try:
    idx = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))
    keys = list(idx["weight_map"].keys())
except Exception:
    from safetensors import safe_open
    f = hf_hub_download(REPO, "model.safetensors")
    with safe_open(f, framework="np") as s: keys = list(s.keys())
emb  = [k for k in keys if "embed_tokens" in k]
head = [k for k in keys if "lm_head" in k]
print(f"  總 tensor 數 {len(keys)}")
print(f"  embed_tokens : {emb}")
print(f"  lm_head      : {head or '（不存在 → 與 embedding 共用權重）'}")

tied = cfg.tie_word_embeddings and not head
print(f"\n=== 結論 ===")
print(f"  tied = {tied}")
print("  arm E predicate:",
      "只需匹配 'embed_tokens'，一次同時保護輸入與輸出端" if tied
      else "必須同時匹配 'embed_tokens' 與 'lm_head'，記憶體代價需重算")
