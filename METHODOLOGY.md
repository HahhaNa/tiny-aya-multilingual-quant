# METHODOLOGY

## 1. 速度量測協定

### 1.1 為什麼需要協定：一次真實的失敗示範

Day 2 的 smoke test 是**循序**跑的（A → B → C → D → E，每 arm 三次生成，中間無冷卻）。
把實測中位數對照從記憶體頻寬推出的 roofline 上限（M3 約 100 GB/s）：

| arm | 權重 GB | roofline tok/s | 實測中位 | 效率 | 純頻寬外推（錨定 C）| 落差 |
|---|---|---|---|---|---|---|
| A-bf16 | 6.72 | 14.9 | 11.8 | **79%** | 8.3 | +43% |
| B-q8-g64 | 3.58 | 27.9 | 20.1 | **72%** | 15.5 | +29% |
| C-q4-g64 | 1.91 | 52.4 | 29.1 | **56%** | 29.1 | 0% |
| D-q4-g32 | 2.11 | 47.4 | 24.2 | **51%** | 26.3 | −8% |
| E-q4-emb8 | 2.17 | 46.1 | 22.6 | **49%** | 25.6 | −12% |

效率單調下降 79% → 49%，**且順序與執行順序完全一致**。

有兩個互相競爭的解釋：

1. **熱節流**：M3 Air 無風扇，E 跑在機器最熱的時候
2. **dequant 開銷**：bf16 不需要反量化，4-bit 需要；低位元的 kernel 效率天生較低

**循序資料無法分離這兩者**——它們被執行順序完全混淆了。
這就是本研究速度量測必須交錯的理由，而這張表是它的直接證據。

> 注意：此表來自 smoke test，**不是研究結果**，不進 `results/`。
> 它的用途是說明協定的必要性。

### 1.2 正式協定（T6）

- **交錯**：5 arm × 3 種 prompt 長度 × 3 重複 = 45 次，用 `seed=1337` 洗牌，
  執行順序寫入 `bench/run_order.csv` 並隨結果一起提交。`order_idx` 進入結果表，
  事後可檢定「順序」是否仍有殘餘效應
- **冷卻**：每次量測後強制 `sleep 90`
- **環境**：接電源、關閉低耗電模式、關閉其他 app、`sudo sysctl iogpu.wired_limit_mb=13000`
- **熱記錄**：背景 `powermetrics --samplers cpu_power,gpu_power,thermal -i 1000`，事後對照
- **統計量**：報**中位數與 IQR**，不報平均（熱節流造成單邊長尾）
- **跨語言**：tok/s 不可跨語言比較，須以 fertility 換算 chars/s

### 1.3 事後檢定（協定是否奏效）

交錯之後，對 `results/speed.csv` 跑 `decode_tps ~ arm + order_idx`。
`order_idx` 的係數若仍顯著，代表 90 秒冷卻不足，需延長並重跑。
**這個檢定要報出來，不論結果**。

---

## 2. 為什麼是 bits-per-byte

`BPB = (Σ NLL_nats / ln 2) / Σ len(utf8_bytes)`

perplexity 的分母是 token 數，而 token 數受 tokenizer 影響：Burmese 的同一句話可能被切成
英語三倍的 token，每個 token 因此更好猜，perplexity 假性偏低。改用 UTF-8 位元組作分母後，
分母與 tokenizer 無關，跨語言才可直接比較。

perplexity 仍會記錄，但**僅供同語言內對照**，不用於跨語言比較（見 PREREGISTRATION §6）。

---

## 3. 量化配置的位元開銷

affine 量化把每 `group_size` 個權重存成整數，另配一個 scale 與一個 bias（各 16-bit），
因此每權重的有效位元為 `bits + 32/group_size`：

- 4-bit, g=64 → 4.500
- 4-bit, g=32 → 5.000
- 8-bit, g=64 → 8.500
- arm E（body 4-bit g64 + embedding 8-bit g64）→ **5.141**

上述由 `analysis/param_budget.py` 手推，並經 mlx-lm 轉檔時回報的 bits-per-weight 驗證，
五個 arm 的實際磁碟大小與預估誤差皆 <1.5%。
