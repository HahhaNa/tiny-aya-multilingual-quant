"""T2 · arm E — 緩解組：body 4-bit，embedding 8-bit。

兩個已驗證的前提（見 docs/GATE_G1.md）：
  1. tie_word_embeddings=True 且權重檔無 lm_head → 匹配 embed_tokens 一處即保護輸入與輸出
     （MLX 的 cohere2 是 `out = self.model.embed_tokens.as_linear(out)`）
  2. mlx-lm 0.31.3 的 quant_predicate 簽名是 (path, module)，**2 個參數不是 3 個**
     呼叫點：mlx_lm.utils.quantize_model 內的 wrapped_predicate
"""
import sys
from mlx_lm.convert import convert

MATCHED = []

def keep_embeddings_high(path, module):
    """embedding（＝輸出層）留 8-bit，其餘 4-bit。回傳 dict 會覆寫該層的量化參數。"""
    if "embed_tokens" in path:
        MATCHED.append(path)
        return {"group_size": 64, "bits": 8}
    return True          # True = 用預設的 4-bit / g64

if __name__ == "__main__":
    convert(
        "CohereLabs/tiny-aya-global",
        mlx_path="models/E-q4-emb8",
        quantize=True,
        q_bits=4,
        q_group_size=64,
        quant_predicate=keep_embeddings_high,
    )
    print(f"\n[驗收] predicate 命中的 8-bit 模組：{MATCHED}")
    if len(MATCHED) != 1:
        print("[FAIL] 預期恰好命中 1 個 embed_tokens", file=sys.stderr)
        sys.exit(1)
    print("[驗收] 上面 mlx-lm 印的 bits per weight 應約為 5.14")
