# G1 關卡結論 — MLX 支不支援 Tiny Aya 架構

**日期**：2026-09-01 · **結論：通過，走原計畫（不需備援 a/b/c）**

## 怎麼確認的
`CohereLabs/tiny-aya-global` 是 gated（`gated=auto`），直接抓 config.json 得到 403。
改從**未受限的 MLX 鏡像** `mlx-community/tiny-aya-global-8bit-mlx` 取得同一份 config —
該 repo 存在本身就是 MLX 可跑此架構的證據。

## config.json 關鍵欄位
| 欄位 | 值 | 意義 |
|---|---|---|
| `model_type` | `cohere2` | **在 mlx-lm 0.31.3 的 118 個支援架構清單內** |
| `architectures` | `Cohere2ForCausalLM` | |
| `vocab_size` × `hidden_size` | 262144 × 2048 | embedding = 536.9M 參數 = **全模型 16.0%** |
| `num_hidden_layers` | 36 | body = 2.812B → 總計 3.349B ≈ 3.35B ✓ |
| GQA | 16 q-head / 4 kv-head, head_dim 128 | KV cache 只有 MHA 的 1/4 |
| `layer_types` | 3 × sliding(4096) + 1 × full，重複 9 次 | 與 Command R7B 同構，正是 `cohere2` |
| `max_position_embeddings` | **8192** | 鏡像 repo 寫 500000，**原 repo 是 8192**——不要信鏡像的 config |
| `tie_word_embeddings` | 未列出 → **實測確認 True** | `convert/check_tie.py` 兩路驗證：transformers 解析為 True，且 290 個 tensor 中**不存在 `lm_head.weight`**。arm E 的 predicate 只需匹配 `embed_tokens`，一次同時保護輸入與輸出端 |

## 推導出的權重預算（`analysis/param_budget.py`，與計畫預估完全吻合）
| arm | bits/param | GB | vs C |
|---|---|---|---|
| A-bf16 | 16.00 | 6.70 | 3.56× |
| B-q8-g64 | 8.50 | 3.56 | 1.89× |
| C-q4-g64 | 4.50 | 1.88 | 1.00× |
| D-q4-g32 | 5.00 | 2.09 | 1.11× |
| E-q4-emb8 | 5.14 | 2.15 | **1.14×** |

arm E 的「只多 14% 記憶體」是算出來的，不是引用的。

## 授權（已解除，2026-09-01）
`gated=auto`，網頁按下接受後立即生效。原 repo 的 config 已可直接取得，上表已改用原 repo 的值。

## 曾經的 blocker（保留紀錄）
`CohereLabs/tiny-aya-global` 需在網頁按下接受 CC-BY-NC。`gated=auto` = 按完立即生效，不需等審核。
未解前無法產生 **arm A（bf16 基線）**，而所有 Δ 都相對它 → T2 之後全部卡住。
（8-bit 鏡像不能當基線：它已經是被量化過的模型。）

**教訓**：鏡像 repo 可以用來回答「MLX 支不支援」，但**不能用來抄 config 數值**——`max_position_embeddings` 就對不上。

## 順手發現：作品集的相對定位
`mlx-community` 已有 `tiny-aya-global-8bit-mlx` 與 `tiny-aya-fire-4bit`。
也就是**「把它量化」本身沒有新意**——新意在多語言不對稱的量測、arm E 的緩解、以及無風扇機器的量測協定。
這反而讓論述更聚焦：你不是在做轉檔，你是在做評測。
