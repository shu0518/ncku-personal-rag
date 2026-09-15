# HW3 AI 評分回饋

> 本份回饋由 **Anthropic Claude Opus 4.7** 與 **OpenAI GPT-5.5** 兩個獨立模型，分別針對 Phase 1 與 Phase 2 程式碼與文件進行評分；最終加權分數為兩模型平均後，進行全班性的 Normalize，以及老師加分。

---

## 本次公布的兩個分數

本次 HW3 作業，每位同學會公布**兩個分數**，計算方式為下列三個候選值中**最高的兩個**：

| 候選 | 來源 | 計算 | 值 |
|---|---|---|---:|
| A | Max(Phase 1 兩模型評分) | max(Opus 85.0, GPT 89.2) | 89.25 |
| B | Max(Phase 2 兩模型評分) | max(Opus 78.1, GPT 79.3) | 79.25 |
| C | 最終加權分數 | 詳見下方加權計算 | 85.06 |

### 你的兩個分數

### **89.25**　／　**85.06**

_（A：Max(Phase 1 兩模型評分) = 89.25；C：最終加權分數 = 85.06）_

## 最終加權分數（候選 C）的計算

| 來源 | Phase 1 (40%) | Phase 2 (60%) | 加權平均 |
|---|---:|---:|---:|
| Anthropic Claude Opus 4.7 | 85.0 | 78.1 | 80.88 |
| OpenAI GPT-5.5 | 89.2 | 79.3 | 83.25 |
| **兩模型平均** | **87.1** | **78.7** | **82.06** |
| 老師加分 (professor_bonus) | | | +3 |
| **最終分數** | | | **85.06** |

## 計算公式

```
Phase 1 平均  = (Opus_P1 + GPT_P1) / 2
Phase 2 平均  = (Opus_P2 + GPT_P2) / 2
加權平均      = 0.4 × Phase 1 平均 + 0.6 × Phase 2 平均
Normalize 後  = 依全班分布做校準
最終分數      = min(100, Normalize 後 + 老師加分)
```

## 專案基本資料

- **github_id**：`shu0518-2`
- **領域**（Haiku 抽取）：`investment_analyst_optical_communications`
- **領域分類**（taxonomy）：`finance_investment`
- **Phase 1 CI**：✓ 通過
- **Phase 2 CI**：✓ 通過

## Phase 1 分項評語（40% 權重）

| 面向 | Opus 分數 | Opus 評語 | GPT 分數 | GPT 評語 |
|---|---:|---|---:|---|
| 資料收集 | 72 | data/raw 收 13 份 PDF（券商與研調報告），規模介於 anchor 低中段，README 以表格列出來源類型、授權依據與數量，合規訊號清楚。但格式僅 .pdf 單一類型，未達多樣性高分標準，且來源僅以「各大券商」概稱，缺乏具體機構/年份 metadata，難以驗證時點覆蓋。 | 78 | data/raw 有 13 份 PDF、processed 有 13 份 txt，數量達標但未超過 20，原始格式單一；README 有說明券商與研調報告來源但未逐檔列名。 |
| RAG 系統完整度 | 88 | data_update.py 555 行，採 BaseExtractor 抽象 + Recursive splitter(300/50)，pgvector HNSW 寫入完整；rag_query.py 584 行，延遲 import 通過 --help CI、cosine threshold 過濾、top-k 可調、LiteLLM 串接 gemma4，並於程式層保底輸出引用來源避免幻覺。Context-only system prompt 與輕量歷史策略設計細緻，超越同領域多數提交。扣分點為僅單階段向量檢索，未實作 reranking 或 hybrid。 | 92 | data_update.py 實作 PDF/TXT/MD 萃取、清理、Recursive chunking、embedding 與 pgvector 寫入；rag_query.py 有 top-k 檢索、引用來源與 LiteLLM。 |
| 冪等性 | 92 | Stage 1 以 SHA-256 file hash 寫入 .file_hashes.json 實作真正的增量更新；Stage 2 以 --rebuild 觸發 TRUNCATE ... RESTART IDENTITY 全量重建，兩階段解耦設計清楚（embedding 出錯只需重跑 Stage 2）。README 與程式碼一致，CI pass，符合 anchor 高標。 | 95 | data_update.py 明確支援 --rebuild，Stage 1 以 SHA-256 記錄 .file_hashes.json 做增量處理，Stage 2 可 TRUNCATE 後重建向量表。 |

## Phase 2 分項評語（60% 權重）

| 面向 | Opus 分數 | Opus 評語 | GPT 分數 | GPT 評語 |
|---|---:|---|---:|---|
| 資料收集深度 | 82 | data/raw 與 processed 各 25 份 PDF，超過門檻；README §5 清楚分類來源（OIF 標準、政府、券商、企業研調、媒體、自媒體）並標註授權依據，多樣性佳。惟全為 PDF 單一格式，缺一手 10-K/法說會原檔與時點 metadata，且授權僅以「公開資訊」籠統帶過，未列具體條款。 | 84 | data/raw 25 份達標，涵蓋 OIF、政府、券商、TrendForce、媒體等多元來源；但皆為 PDF，且 README §5 授權多寫公開資訊，缺精確 URL/條款。 |
| skill.md 品質 | 68 | skill.md 八大必要章節齊全，含 Metadata、具體實體（Ayarlabs、OIF、TrendForce）與引用段落編號。但 Core Concepts 僅 6 條未達 10–15，Key Trends 僅 2 條偏少，Key Entities 中「重要人物」標示無資訊，Example Q&A 全部引用同一份報告同段落，深度不足且未呼應券商分析師應有的估值數字與市占比較。 | 68 | skill.md 必要章節齊全，§Core Concepts 有 6 條；但 §Key Trends 僅 2 項、§Key Entities 實體稀少，且 Metadata 稱 10 份與 25 份資料不一致。 |
| README 設計決策 | 88 | README §3 對 chunking(300/50)、embedding(多語 MiniLM)、pgvector+HNSW、Top-K=5 與 0.3 門檻、Context-only prompt、兩階段冪等設計、skill_builder 10 問框架皆有具體取捨理由（如為何 HNSW 而非 IVFFlat、為何 300 字元）。Mermaid 架構圖清楚，限制與未來改進章節誠實。 | 92 | README §3 完整回答 chunking、embedding、pgvector/HNSW、retrieval、prompt、idempotency 與 skill_builder 問題設計，理由與取捨具體。 |
| skill_builder 品質 | 80 | skill_builder.py 527 行，設計 10 個全域問題涵蓋 Overview/Concepts/Trends/Entities/Methodology/Risks/Gaps/Q&A/Sources，分析師視角清晰且問題分工明確（估值 vs 策略分流）。延遲 import rag_query 重用核心元件，top_k=8 擴大覆蓋。但實際輸出 skill.md 顯示整合深度有限（多處集中單一來源），顯示 LLM 整合 prompt 仍有改進空間。 | 80 | skill_builder.py 設計 10 個全域問題，涵蓋 Overview、Trends、Entities、風險與 Q&A；但產出內容顯示多次查詢整合仍偏淺，來源覆蓋不足。 |

## 整體評語

### 亮點

- 完整的兩階段 ETL 管線設計，Stage 1 與 Stage 2 解耦提升維護彈性
- 多語言 embedding 模型適配中英混合內容，本地離線執行無 API 依賴
- 系統化的 10 問題框架確保 skill.md 涵蓋分析師所需的完整知識維度
- 嚴格的 Context-only prompt 策略與程式層保底引用輸出防止幻覺
- 詳細的設計決策文件說明每個技術選型的取捨邏輯與實測依據

### 改進建議

- Naive RAG 架構無法支援多跳推理，跨報告的時序性問題易遺漏資訊
- pypdf 純文字萃取對複雜表格與圖表結構保留能力有限
- 通用 embedding 模型對光通訊領域高度專業術語（如 coherent DSP）精度不足
- 知識庫資料截止於 2026 年，對歷史演變與未來預測覆蓋不足
- 系統依賴外部 LiteLLM proxy 與教授提供的 API Key，無法完全離線運作

---

_本份評分回饋由自動化評分管線（Anthropic + OpenAI 雙模型獨立評分 → 平均 → rescale → 老師加分）產出，僅作為個人學習參考。若對評分有疑問請於課堂或 office hour 提出。_
