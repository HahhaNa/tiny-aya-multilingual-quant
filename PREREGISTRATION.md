# 預先登錄 · Tiny Aya 邊緣量化的多語言不對稱

**登錄日期**：2026-09-01（commit 時間戳為準）
**狀態**：在任何評測結果產生**之前**寫定。
本檔的假設與統計檢定**不得因為看到數字而修改**。事後才想到的分析一律標 `[POST-HOC]` 附在最後，
且在報告中與預先登錄的分析分開呈現。

---

## 1. 研究問題

均勻 4-bit 量化對一個**專為低資源語言設計**的 3.35B 邊緣模型，
是否在不同語言上造成**不等量**的品質退化？若是，機制在哪裡、能不能便宜地補救？

## 2. 受測對象（已於 Day 1 驗證）

`CohereLabs/tiny-aya-global`，`model_type=cohere2`，3.349B 參數，36 層，
GQA 16q/4kv，3×sliding(4096)+1×full 重複 9 次，vocab 262144，
embedding 536.9M = **全模型 16.0%**，`tie_word_embeddings=True`（權重檔無獨立 `lm_head`）。

## 3. 假設

### H1（主假設）
從 bf16（arm A）到 4-bit（arm C），**低資源語言的品質退化幅度顯著大於高資源語言**，
且此差距在控制「基線分數本來就較低」之後仍然存在。

- **應變數**：ΔBPB（主）、ΔchrF++（次）
- **檢定**：OLS 迴歸 `Δ ~ tier + script_family + fertility + baseline_score`
- **判準**：`tier`（low vs high）係數的 95% CI **不跨 0** 且方向為「低資源退化較大」
- **否證條件**：`tier` 係數 95% CI 跨 0 → H1 不成立

### H2（機制假設）
退化主要來自 embedding／輸出層的低位元化，而非 transformer block。
因此把 embedding 保留 8-bit（arm E，記憶體僅 +14%）能補回大部分語言間差距。

- **應變數**：公平性缺口 gap = (高資源平均退化) − (低資源平均退化)
- **檢定**：比較 arm C 與 arm E 的 gap，配對 bootstrap 10,000 次（`seed=1337`）求差值 CI
- **判準**：`gap(C) − gap(E)` 的 95% CI 不跨 0，且 arm E 的 gap 至少縮小 **50%**
- **否證條件**：CI 跨 0，或縮小幅度 <50%
- **重要**：H2 被否證時 **H1 仍可能成立**——那代表機制不在 embedding。
  後續改比較 arm C vs D（group size），若 D 顯著較佳則機制指向 activation outlier。

### H0（空結果假設）
Tiny Aya 因為是為低資源語言專門設計的（語言桶式 tokenizer、平衡訓練資料），
可能**不存在**這種不對稱。

- **這仍然可發表的理由**：現有文獻互相矛盾——Marchisio et al. 2024 發現非拉丁文字受害較大；
  2503.03592 用英語校準 K-quant 卻發現沒有不成比例的傷害。
  一個**窄信賴區間**的乾淨 null，在一個刻意為低資源語言設計的模型上，
  是對「模型設計能否保護低資源語言」的直接證據。
- **報告要求**：報 null 時**必須**同時報效果量的 CI 寬度。寬 CI = 檢定力不足，不是 null。

## 4. 實驗組（arm）

| arm | 設定 | 預估大小 | 單獨回答什麼 |
|---|---|---|---|
| A | bf16 | 6.70 GB | 基線。所有 Δ 相對它 |
| B | 8-bit, g=64 | 3.56 GB | 8-bit 是否真的「幾乎無損」——在低資源語言上也是嗎 |
| C | 4-bit, g=64 | 1.88 GB | **主要處理組**，實際會被部署的配置 |
| D | 4-bit, g=32 | 2.09 GB | 分離「位元深度」與「量化粒度」 |
| E | 4-bit body + 8-bit embed | 2.15 GB | **緩解組**（+14% 記憶體） |
| F/G | AWQ 英語校準 / 10 語混合校準 | — | **加分題**，僅 Day 7 中場關卡決定有時間才做 |

## 5. 語言（2×2 設計，事前選定，不得因結果調整）

| 語言 | FLORES 碼 | 文字系統 | tier | 角色 |
|---|---|---|---|---|
| English | eng_Latn | Latin | high | 對照 |
| Spanish | spa_Latn | Latin | high | 對照 |
| Russian | rus_Cyrl | Cyrillic | high | **高資源 × 非拉丁**（分離文字系統與資源量）|
| Chinese (Trad) | cmn_Hant | Han | high | 高資源 × 非拉丁；唯一可人工抽檢（見修訂 A1）|
| Hindi | hin_Deva | Devanagari | mid | 中段橋樑 |
| Arabic | arb_Arab | Arabic (RTL) | mid | 中段橋樑 |
| Swahili | swh_Latn | Latin | low | **低資源 × 拉丁**（關鍵反例）|
| Yoruba | yor_Latn | Latin + 聲調 | low | 低資源 × 拉丁；高 fertility |
| Amharic | amh_Ethi | Ge'ez | low | 低資源 × 非拉丁 |
| Burmese | mya_Mymr | Myanmar | low | 低資源 × 非拉丁；預期最高 fertility |

`tier` 依 FLORES-200 / Joshi et al. 的資源分級**事前**指定，不得依觀測到的基線分數回推。

## 6. 指標

| 層級 | 指標 | 角色 |
|---|---|---|
| L0 | 權重空間量化誤差（逐 token embedding MSE）| H2 的機制證據，不需推論 |
| L1 | **ΔBPB**，FLORES-200 devtest | **主要應變數**。`BPB = (ΣNLL/ln2)/Σbytes` |
| L2 | ΔchrF++，en→X 200 句 | 次要應變數，僅 arm A/C/E |
| L3 | 文字系統偏離率、語言混淆率、退化重複率 | 自動指標低估傷害的廉價代理 |
| L4 | Δ Global-MMLU | **選作**，Day 7 第一個砍 |
| L5 | decode/prefill tok/s、TTFT、chars/s、峰值記憶體 | 「換到什麼」的那一半 |

**明確排除**：perplexity 不得用於跨語言比較（tokenizer fertility 使其分母不可比）。
僅作為同語言內的對照量記錄。

## 7. 統計程序

1. **配對 bootstrap** 10,000 次（`seed=1337`），同一批句子重抽，求 ΔchrF++ 的 95% CI
2. **主迴歸** `Δ ~ tier + script_family + fertility + baseline_score`（OLS）
3. **絕對與相對退化都報**，並明講兩者故事不同
4. 逐語言結果**一律附 CI 或 IQR**，不接受無誤差棒的長條圖
5. 速度：報**中位數與 IQR**，不報平均（熱節流造成單邊長尾）

## 8. 量測污染的事前控制

- 速度量測 45 次（5 arm × 3 prompt 長度 × 3 重複）**依 `seed=1337` 洗牌交錯**，不得依 arm 循序
- 每次量測間強制冷卻 90 秒；背景記錄 `powermetrics`
- 事前宣告：若同一 arm 前後兩次差異 >20%，於報告中明列並以中位數為準

## 9. 停止規則與範圍縮減

Day 7 中場關卡依**事前排定**的順序砍除，不得依「哪個結果比較好看」決定：

> L4 Global-MMLU → AWQ 消融 → arm D → L2 語言數 10→6 → L2 句數 200→100

**絕不砍**：L1 BPB 全矩陣、L3 語言忠實度、arm E、速度量測協定。

## 10. 已知限制（事前承認）

1. 單一模型、單一硬體 → 結論不可推廣
2. 無人工評估（無 10 語母語者）→ L3 是代理指標，非黃金標準；僅 zho_Hant 有作者抽檢
3. FLORES 是新聞／維基領域，有領域偏差
4. tier 分級本身是粗糙的離散化
5. M3 Air 無風扇 → 速度數字是這台機器的持續效能，非架構上限

---

## 修訂紀錄（識別碼層級，非假設層級）

### A1 · 2026-09-01 — 中文的 FLORES 代碼
**改動**：`zho_Hant` → `cmn_Hant`
**原因**：FLORES+（`openlanguagedata/flores_plus`）採 ISO 639-3 個別語言碼，
把 FLORES-200 時代的巨集語言碼 `zho_*` 改為 `cmn_*`。**同一份資料、同一個語言、同一個文字系統**，
只是識別碼命名更新。221 個 devtest 語言中存在 `cmn_Hans` / `cmn_Hant` / `yue_Hant`。
**當下狀態**：`results/` 仍為空，尚未產生任何評測數字。
**這不構成假設修改**：H1/H2/H0、應變數、統計檢定、tier 分級、停止規則皆未動。

### A2 · 2026-09-01 — BPB 的宣稱範圍澄清
**改動**：明確限定 BPB 的用途，不修改任何假設或檢定。
**原因**：Day 4 實測顯示 UTF-8 位元組／字元在十種語言間相差約三倍（拉丁 1.00 至緬甸文 2.85），
因此**絕對** BPB 帶有文字系統的編碼成本，不是跨語言的品質尺規。
**澄清**：本研究的應變數自始即為 **ΔBPB（同語言內，相對 arm A）**，編碼成本相減時抵消。
報告中不得出現「語言 A 絕對 BPB 低於語言 B 故品質較佳」這類敘述。
**當下狀態**：`results/` 仍為空。H1/H2/H0、應變數、檢定、tier 分級、停止規則皆未動。

### A3 · 2026-09-01 — 跨語言速度正規化改用平行句 token 比值
**改動**：原計畫的 chars/s 改為輔助指標，主要換算改用 `decode_tps / tok_ratio_vs_eng`。
**原因**：chars/s 被字元資訊密度污染（漢字承載量遠高於拉丁字母）。
平行語料提供無混淆的分母。
**當下狀態**：`results/` 仍為空，速度量測（T6）尚未執行。

---

## [POST-HOC] 事後補充
（尚無。任何在看到結果之後才想到的分析，寫在這裡並註明日期。）
