"""
skill_builder.py — Agent Skill 文件自動生成器
================================================
從 RAG 知識庫系統性萃取知識，產出 skill.md。

執行方式：
  python skill_builder.py                        # 使用預設輸出路徑 skill.md
  python skill_builder.py --output skill.md      # 指定輸出路徑
  python skill_builder.py --top-k 8              # 調整每次查詢取回的 chunk 數
  python skill_builder.py --model openai/gemma4  # 指定 LLM 模型
  python skill_builder.py --help

設計說明：
  本腳本直接 import rag_query.py 的核心元件（EmbeddingModel、PgVectorRetriever、
  run_query），以「外資券商分析師」的視角，向知識庫提出 10 個全域問題，分別對應
  skill.md 的各個章節，再透過 LLM 整合為一份可供 Agent 直接使用的技能文件。
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 環境變數
# ---------------------------------------------------------------------------

PGVECTOR_CONN_STR    = os.getenv("PGVECTOR_CONNECTION_STRING",
                                  "postgresql://raguser:ragpassword@localhost:5432/ragdb")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL",
                                  "paraphrase-multilingual-MiniLM-L12-v2")
LITELLM_BASE_URL     = os.getenv("LITELLM_BASE_URL",
                                  "https://litellm.netdb.csie.ncku.edu.tw")
LITELLM_API_KEY      = os.getenv("LITELLM_API_KEY", "")

DEFAULT_LLM_MODEL  = "openai/gemma4"
DEFAULT_OUTPUT     = "skill.md"
DEFAULT_TOP_K      = 8   # skill_builder 每次查詢取較多 chunk，擴大知識覆蓋率

# ---------------------------------------------------------------------------
# 10 個全域問題（分析師框架）
# 每個問題對應 skill.md 的特定章節，以外資券商分析師視角設計，
# 涵蓋：主題範疇、核心概念、市場趨勢、重要實體、投資框架、風險因子、
#        資料品質邊界、代表性問答。
# ---------------------------------------------------------------------------

GLOBAL_QUESTIONS: list[dict[str, str]] = [
    {
        "id":      "q01_domain_scope",
        "section": "Overview",
        "query":   (
            "這個知識庫涵蓋哪些核心主題、產業或領域？"
            "請用一段話總結這份知識庫的知識範疇與主要研究視角。"
        ),
    },
    {
        "id":      "q02_core_concepts",
        "section": "Core Concepts",
        "query":   (
            "這個領域中最重要的 10 至 15 個核心概念、術語或框架是什麼？"
            "請條列說明每個概念的定義與在分析實務中的意義。"
        ),
    },
    {
        "id":      "q03_market_trends",
        "section": "Key Trends",
        "query":   (
            "目前這個產業或市場最重要的發展趨勢、結構性變化或新興主題有哪些？"
            "請從分析師角度條列 5 至 10 個最值得關注的趨勢，並說明其驅動因子。"
        ),
    },
    {
        "id":      "q04_key_players",
        "section": "Key Entities",
        "query":   (
            "這個領域中最重要的公司、機構、研究機構、產業組織或意見領袖有哪些？"
            "請分類條列，並簡述各實體的角色或市場地位。"
        ),
    },
    {
        "id":      "q05_investment_framework",
        "section": "Methodology & Best Practices",
        "query":   (
            "分析師在研究這個產業或市場時，常用的分析框架、評價方法、KPI 或核心指標有哪些？"
            "例如：估值倍數、市場份額計算、供需模型、技術評估方法等，請詳細說明。"
        ),
    },
    {
        "id":      "q06_risk_factors",
        "section": "Methodology & Best Practices",
        "query":   (
            "這個產業或市場目前面臨哪些主要風險因子與挑戰？"
            "請從監管政策、競爭格局、技術替代、總體經濟、地緣政治等面向分別說明。"
        ),
    },
    {
        "id":      "q07_data_coverage",
        "section": "Knowledge Gaps & Limitations",
        "query":   (
            "這份知識庫的資料來源有哪些限制？"
            "哪些子主題、時間區間或地區覆蓋不足？哪些問題是目前知識庫無法回答的？"
        ),
    },
    {
        "id":      "q08_qa_valuation",
        "section": "Example Q&A",
        "query":   (
            "請舉出 2 個關於這個產業估值或財務分析的典型問題，並根據知識庫內容給出簡短答案。"
            "例如：本益比合理區間、營收成長驅動力、毛利率結構等。"
        ),
    },
    {
        "id":      "q09_qa_strategy",
        "section": "Example Q&A",
        "query":   (
            "請舉出 2 個關於這個產業競爭策略或市場格局的典型問題，並根據知識庫內容給出簡短答案。"
            "例如：主要競爭者差異、市場集中度、進入壁壘等。"
        ),
    },
    {
        "id":      "q10_sources",
        "section": "Source References",
        "query":   (
            "這個知識庫包含哪些來源的資料？"
            "請列出所有出現過的文件來源、報告名稱、機構名稱或作者，並說明其類型（研究報告、新聞、官方文件等）。"
        ),
    },
]


# ===========================================================================
# RAG 批量查詢
# ===========================================================================

def run_all_questions(
    top_k: int,
    model_name: str,
) -> list[dict[str, Any]]:
    """
    初始化 RAG 元件後，依序對 GLOBAL_QUESTIONS 中的每個問題執行查詢。
    回傳每題的查詢結果（問題、回答、引用 chunks）。
    """
    # 延遲 import rag_query，確保 --help 不受影響
    try:
        from rag_query import EmbeddingModel, PgVectorRetriever, run_query
    except ImportError as exc:
        logger.error("無法 import rag_query.py：%s", exc)
        logger.error("請確認 skill_builder.py 與 rag_query.py 在同一目錄。")
        sys.exit(1)

    logger.info("載入 Embedding 模型：%s", EMBEDDING_MODEL_NAME)
    try:
        embedding_model = EmbeddingModel(model_name=EMBEDDING_MODEL_NAME)
    except Exception as exc:
        logger.error("Embedding 模型載入失敗：%s", exc)
        sys.exit(1)

    results: list[dict[str, Any]] = []

    try:
        import psycopg2
        with PgVectorRetriever(PGVECTOR_CONN_STR) as retriever:
            for i, q in enumerate(GLOBAL_QUESTIONS, start=1):
                logger.info(
                    "[%d/%d] 查詢中（%s）：%s",
                    i, len(GLOBAL_QUESTIONS), q["id"], q["query"][:60] + "..."
                )
                conversation_history: list[dict[str, str]] = []
                try:
                    answer, chunks = run_query(
                        query=q["query"],
                        retriever=retriever,
                        embedding_model=embedding_model,
                        top_k=top_k,
                        conversation_history=conversation_history,
                        model_name=model_name,
                    )
                    results.append({
                        "id":      q["id"],
                        "section": q["section"],
                        "query":   q["query"],
                        "answer":  answer,
                        "chunks":  chunks,
                    })
                    logger.info("[%d/%d] ✅ 完成（%s）", i, len(GLOBAL_QUESTIONS), q["id"])
                except Exception as exc:
                    logger.warning("[%d/%d] ⚠️  查詢失敗（%s）：%s", i, len(GLOBAL_QUESTIONS), q["id"], exc)
                    results.append({
                        "id":      q["id"],
                        "section": q["section"],
                        "query":   q["query"],
                        "answer":  "（查詢失敗，無法取得資料）",
                        "chunks":  [],
                    })
    except psycopg2.OperationalError:
        logger.error("pgvector 連線失敗，請確認 docker compose up -d 已執行。")
        sys.exit(1)
    except Exception as exc:
        logger.error("執行失敗：%s", exc)
        sys.exit(1)

    return results


# ===========================================================================
# LLM 呼叫（支援本地 Ollama 與教授 proxy server 兩種模式）
# ===========================================================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def call_llm(messages: list[dict[str, str]], model_name: str) -> str:
    try:
        from litellm import completion
    except ImportError:
        logger.error("litellm 套件未安裝，請執行：pip install litellm")
        sys.exit(1)

    # 若模型名稱以 ollama/ 開頭，改走本地 Ollama，不需要 API Key
    if model_name.startswith("ollama/"):
        logger.info("使用本地 Ollama（%s）", OLLAMA_BASE_URL)
        os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE_URL
        try:
            response = completion(
                model=model_name,
                messages=messages,
            )
        except Exception as exc:
            logger.error("Ollama 呼叫失敗（%s）：%s", type(exc).__name__, exc)
            logger.error("請確認 Ollama 已啟動，且模型已下載：ollama pull qwen2.5:14b")
            raise
    else:
        # 走教授 proxy server
        if not LITELLM_API_KEY:
            logger.warning("LITELLM_API_KEY 尚未設定，請在 .env 填入教授提供的金鑰")
        try:
            response = completion(
                model=model_name,
                messages=messages,
                api_key=LITELLM_API_KEY,
                api_base=LITELLM_BASE_URL,
            )
        except Exception as exc:
            logger.error("LLM 呼叫失敗（%s）：%s", type(exc).__name__, exc)
            raise

    return response.choices[0].message.content or ""


# ===========================================================================
# Skill.md 合成
# ===========================================================================

def collect_sources(results: list[dict[str, Any]]) -> str:
    """從所有查詢的 chunks metadata 彙整不重複的來源清單。"""
    seen: set[str] = set()
    lines: list[str] = []
    for r in results:
        for chunk in r.get("chunks", []):
            meta   = chunk.get("metadata", {})
            source = meta.get("source", "unknown")
            title  = meta.get("title", source)
            key    = source
            if key not in seen:
                seen.add(key)
                lines.append(f"- **{title}**（`{source}`）")
    return "\n".join(lines) if lines else "- （來源資訊未能從知識庫中取得）"


def count_sources(results: list[dict[str, Any]]) -> int:
    """直接計算資料夾內的實際檔案數量，而非僅計算檢索到的 chunk 來源"""
    seen: set[str] = set()
    for r in results:
        for chunk in r.get("chunks", []):
            meta = chunk.get("metadata", {})
            seen.add(meta.get("source", "unknown"))
    return len(seen)


def build_skill_md(
    results: list[dict[str, Any]],
    model_name: str,
) -> str:
    """
    將所有全域問題的 RAG 回答，透過 LLM 整合為完整 skill.md。
    採用兩階段策略：
      1. 以分析師角度整合各章節的 RAG 回答原文
      2. 一次 LLM synthesis call 產出完整 skill.md，確保章節間連貫性
    """
    today          = datetime.date.today().strftime("%Y-%m-%d")
    source_count   = count_sources(results)
    source_list    = collect_sources(results)

    # 組裝各章節的 RAG 原始回答，作為 LLM synthesis 的輸入
    rag_summaries: list[str] = []
    for r in results:
        block = (
            f"### [{r['id']}] 章節對應：{r['section']}\n"
            f"**查詢問題**：{r['query']}\n\n"
            f"**RAG 回答**：\n{r['answer']}\n"
        )
        rag_summaries.append(block)

    rag_context = "\n\n---\n\n".join(rag_summaries)

    synthesis_system_prompt = """\
你是一位資深外資券商研究部主管，同時也是 AI Agent 技能文件的架構師。
你的任務是將 RAG 知識庫系統的多次查詢結果，整合為一份高品質的 Agent Skill 文件（skill.md）。

撰寫原則：
1. **分析師視角優先**：用外資券商研究報告的語氣與架構，而非通用 AI 問答的語氣。
2. **知識密度**：每個章節都應有具體的數字、名稱、比率、框架，避免空泛描述。
3. **繁體中文**：全文使用繁體中文，專業術語可中英並列（如：本益比 P/E Ratio）。
4. **嚴格遵守格式**：必須包含所有指定章節標題，CI 會自動檢查章節關鍵字。
5. **僅使用 RAG 提供的資訊**：不得捏造資料或加入知識庫外的內容；若資訊不足，請在 Knowledge Gaps 章節誠實說明。
"""

    synthesis_user_prompt = f"""\
以下是從 RAG 知識庫中，針對 10 個全域問題所取得的查詢結果。
請根據這些資訊，整合產出一份完整的 skill.md，嚴格遵守下列格式規範。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RAG 查詢結果（共 {len(results)} 題）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{rag_context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
已辨識來源（共 {source_count} 份文件）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{source_list}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
請輸出以下格式的 skill.md（直接輸出 Markdown 內容，不要有任何前置說明）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Skill: [根據知識庫內容，自動推斷最適合的分析師職稱與領域名稱]

## Metadata
- **知識領域**：[具體領域名稱，例如：CPO 產業分析 / 半導體供應鏈 / AI PC 市場]
- **資料來源數量**：{source_count} 份文件
- **最後更新時間**：{today}
- **適用 Agent 類型**：外資券商產業分析師 / 投資研究助手 / 市場情報顧問

## Overview
[200 字以內的知識範疇摘要，說明此 Skill 能解答哪類分析問題、覆蓋哪些市場與時間範圍。
用外資報告 Executive Summary 的寫法，第一句話就要說出最核心的洞察。]

## Core Concepts
[條列 10–15 個核心概念，每個概念附 1–2 句說明，需包含業界常用英文縮寫或術語]

## Key Trends
[條列 5–10 個當前最重要的產業趨勢或市場結構變化，每條需說明：
  - 趨勢名稱（加粗）
  - 驅動因子
  - 對市場參與者的影響]

## Key Entities
[分四類條列：
  ### 主要公司 / 市場參與者
  ### 研究機構 / 資料來源
  ### 重要人物 / 意見領袖
  ### 核心工具 / 框架 / 指標]

## Methodology & Best Practices
[分兩部分：
  ### 分析框架與評價方法
  [說明常用的財務模型、估值方法、KPI 指標體系，需有具體公式或指標名稱]

  ### 主要風險因子
  [依監管、競爭、技術、總經、地緣政治等面向分別列出，每條風險需說明影響路徑]]

## Knowledge Gaps & Limitations
[誠實說明：
  - 資料截止時間限制
  - 地區或市場覆蓋不足的部分
  - 無法回答的問題類型
  - 建議補充的資料來源]

## Example Q&A
[列出 4–5 組代表性問答，展示此 Skill 能處理的問題類型。格式：
  **Q：[問題]**
  A：[根據知識庫的簡短答案，50–100 字]]

## Source References
[完整列出所有引用來源，格式：
  | 來源名稱 | 類型 | 語言 |
  |---|---|---|
  | ... | 研究報告 / 新聞文章 / 官方文件 | 中文 / 英文 |

  以下是已辨識的來源清單，請轉換為上述表格格式並補充類型欄位：
{source_list}]
"""

    logger.info("🔧 整合所有 RAG 回答，透過 LLM 生成 skill.md...")
    messages = [
        {"role": "system", "content": synthesis_system_prompt},
        {"role": "user",   "content": synthesis_user_prompt},
    ]
    skill_content = call_llm(messages, model_name=model_name)

    # 若 LLM 在輸出前加了 markdown 圍欄，去除之
    if skill_content.strip().startswith("```"):
        lines = skill_content.strip().splitlines()
        # 移除首尾的 ``` 行
        lines = [l for l in lines if not l.strip().startswith("```")]
        skill_content = "\n".join(lines)

    return skill_content.strip()


# ===========================================================================
# CLI 解析
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "skill_builder.py — 從 RAG 知識庫自動生成 Agent Skill 文件（skill.md）\n"
            "以外資券商分析師視角，系統性萃取知識庫中的核心洞察。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python skill_builder.py                        # 預設輸出 skill.md
  python skill_builder.py --output skill.md      # 指定輸出路徑
  python skill_builder.py --top-k 8              # 每次查詢取 8 個 chunk
  python skill_builder.py --model openai/gemma4  # 指定 LLM 模型
        """,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"輸出的 skill.md 路徑（預設：{DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        dest="top_k",
        help=f"每次查詢從向量資料庫取回的 chunk 數量（預設：{DEFAULT_TOP_K}）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"LLM 模型名稱（預設：{DEFAULT_LLM_MODEL}）",
    )
    return parser.parse_args()


# ===========================================================================
# 入口
# ===========================================================================

def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("🚀 skill_builder.py 啟動")
    logger.info("   輸出路徑：%s", args.output)
    logger.info("   Top-K   ：%d", args.top_k)
    logger.info("   LLM 模型：%s", args.model)
    logger.info("   全域問題：%d 題", len(GLOBAL_QUESTIONS))
    logger.info("=" * 60)

    # Step 1：對所有全域問題執行 RAG 查詢
    logger.info("\n📋 Step 1 / 2：執行全域問題 RAG 查詢...")
    results = run_all_questions(top_k=args.top_k, model_name=args.model)
    logger.info("✅ 全域查詢完成，共取得 %d 筆結果", len(results))

    # Step 2：透過 LLM 整合為 skill.md
    logger.info("\n🧠 Step 2 / 2：LLM 整合生成 skill.md...")
    skill_content = build_skill_md(results=results, model_name=args.model)

    # 寫出檔案
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(skill_content)
        logger.info("✅ skill.md 已寫出至：%s（%d 字元）", args.output, len(skill_content))
    except OSError as exc:
        logger.error("檔案寫出失敗：%s", exc)
        sys.exit(1)

    # 基本品質檢查
    required_sections = [
        "## Overview",
        "## Core Concepts",
        "## Key Trends",
        "## Key Entities",
        "## Methodology",
        "## Knowledge Gaps",
        "## Example Q&A",
        "## Source References",
    ]
    missing = [s for s in required_sections if s not in skill_content]
    if missing:
        logger.warning("⚠️  以下必要章節未出現在輸出中，請手動確認：%s", missing)
    else:
        logger.info("✅ 所有必要章節均已包含於輸出中")

    if len(skill_content) < 500:
        logger.warning("⚠️  skill.md 字元數（%d）低於 CI 要求的 500 字元，請檢查輸出品質", len(skill_content))

    logger.info("\n🎉 skill_builder.py 執行完成！")
    logger.info("   請執行 'cat %s' 或用編輯器開啟檔案確認內容。", args.output)


if __name__ == "__main__":
    main()