"""
rag_query.py — RAG 問答 CLI 介面
=====================================
執行方式：
  python rag_query.py                        # 互動式問答模式
  python rag_query.py --query "你的問題"     # 單次查詢
  python rag_query.py --query "..." --top-k 8
  python rag_query.py --help

環境變數（.env）：
  LITELLM_API_KEY   — 教授提供的 API Key
  LITELLM_BASE_URL  — Proxy server URL（預設已內建）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# 基本設定
#   ✅ FIX BLK-1：psycopg2 / sentence_transformers / pgvector 全部移至函式內
#   延遲 import，確保 `python rag_query.py --help` 在任何環境下都不會因
#   套件問題而崩潰，通過 CI 第一階段的 --help 檢查。
# ---------------------------------------------------------------------------

from dotenv import load_dotenv  # python-dotenv 是輕量依賴，保留 module-level

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

PGVECTOR_CONN_STR = os.getenv(
    "PGVECTOR_CONNECTION_STRING",
    "postgresql://raguser:ragpassword@localhost:5432/ragdb",
)
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)
# 教授提供的 proxy server 與統一 API Key
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "https://litellm.netdb.csie.ncku.edu.tw")
LITELLM_API_KEY  = os.getenv("LITELLM_API_KEY", "")

# 固定使用 gemma4（本地，無使用上限）
LLM_MODEL = "openai/gemma4"

TABLE_NAME        = "document_chunks"
DEFAULT_TOP_K     = 5
MAX_HISTORY_TURNS = 3   # 1 輪 = 1 user + 1 assistant

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
你是一位知識庫問答助手。請嚴格遵守以下規則：

1. **只根據「提供的 Context」回答**，不得使用 Context 以外的知識。
2. 若 Context 中找不到足夠資訊，請明確回答「根據目前知識庫，找不到相關資訊」，
   不要推測或捏造答案。
3. 回答結尾**必須**附上「📚 引用來源」區塊，格式如下：

   📚 引用來源
   - [來源 1] 檔案：<source>，段落編號：<chunk_index>
   - [來源 2] 檔案：<source>，段落編號：<chunk_index>
   ...

4. 回答請使用繁體中文（若問題為英文則以英文回答）。
5. 回答要清晰、有條理，適當使用條列或標題。
"""


# ===========================================================================
# EmbeddingModel
#   ✅ FIX BLK-1：SentenceTransformer 在 __init__ 內才 import
# ===========================================================================


class EmbeddingModel:
    """sentence-transformers 本地 Embedding 模型（與 data_update.py 共用同一模型）"""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # 延遲 import
        except ImportError:
            logger.error(
                "sentence-transformers 套件未安裝，請執行：pip install sentence-transformers"
            )
            sys.exit(1)

        logger.info("載入 Embedding 模型：%s（首次執行會自動下載）", model_name)
        self._model = SentenceTransformer(model_name)

    def encode_single(self, text: str) -> list[float]:
        vector = self._model.encode([text], show_progress_bar=False)
        return vector[0].tolist()


# ===========================================================================
# PgVectorRetriever
#   ✅ FIX BLK-1：psycopg2 / pgvector adapter 在 connect() 內才 import
#   ✅ FIX BLK-3：使用 pgvector 原生 register_vector adapter，直接傳 np.ndarray，
#                 不手動拼向量字串，避免 psycopg2 加引號導致 ::vector cast 失敗
# ===========================================================================


class PgVectorRetriever:
    """從 pgvector 執行向量相似度搜尋"""

    def __init__(self, connection_string: str) -> None:
        self._conn_str = connection_string
        self._conn: Any = None  # psycopg2.extensions.connection（延遲 import 故用 Any）

    def connect(self) -> None:
        try:
            import psycopg2                                    # 延遲 import
            from pgvector.psycopg2 import register_vector     # ✅ 原生 vector adapter
        except ImportError as exc:
            logger.error(
                "缺少必要套件：%s\n請執行：pip install psycopg2-binary pgvector", exc
            )
            sys.exit(1)

        try:
            logger.info("連線至 pgvector...")
            self._conn = psycopg2.connect(self._conn_str)
            self._conn.autocommit = True
            register_vector(self._conn)   # ✅ 註冊 vector 型別，讓 adapter 接管序列化
            logger.info("pgvector 連線成功")
        except psycopg2.OperationalError as exc:
            logger.error("pgvector 連線失敗：%s", exc)
            logger.error(
                "請確認：\n"
                "  1. docker compose up -d 已啟動\n"
                "  2. .env 中的 PGVECTOR_CONNECTION_STRING 設定正確\n"
                "  3. 防火牆 / port 5432 未被封鎖"
            )
            raise

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    def __enter__(self) -> "PgVectorRetriever":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = DEFAULT_TOP_K,
        threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        Cosine Similarity 搜尋，回傳 top-k 筆結果。

        Returns:
            list of dict，每筆包含 keys: content, metadata, similarity
        """
        import numpy as np
        import psycopg2.extras

        if self._conn is None or self._conn.closed:
            raise RuntimeError("資料庫未連線，請先呼叫 connect()")

        # ✅ FIX BLK-3：直接傳 np.ndarray，pgvector adapter 正確序列化為 vector 型別
        #    不再手動拼接字串，完全避免 psycopg2 對字串加引號後 ::vector cast 失敗的問題
        query_sql = f"""
            SELECT
                content,
                metadata,
                1 - (embedding <=> %s) AS similarity
            FROM {TABLE_NAME}
            WHERE 1 - (embedding <=> %s) > %s  -- 新增 WHERE 條件
            ORDER BY embedding <=> %s
            LIMIT %s;
        """

        try:
            import psycopg2
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                vec = np.array(query_embedding, dtype=np.float32)
                # 傳入對應的參數
                cur.execute(query_sql, (vec, vec, threshold, vec, top_k))
                rows = cur.fetchall()
        except psycopg2.Error as exc:
            logger.error("向量搜尋失敗：%s", exc)
            raise

        results = []
        for row in rows:
            meta = row["metadata"]
            # psycopg2 搭配 RealDictCursor 時，JSONB 欄位通常已是 dict；
            # 若為字串（舊版驅動）則手動解析
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}
            results.append(
                {
                    "content":    row["content"],
                    "metadata":   meta,
                    "similarity": float(row["similarity"]),
                }
            )
        return results


# ===========================================================================
# Prompt 組裝
# ===========================================================================


def build_rag_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
    """組裝帶有 Context 的使用者 Prompt（只用於當輪 LLM 呼叫，不存入歷史）"""
    if not chunks:
        return (
            "知識庫中未檢索到相關段落。\n\n"
            f"請根據上述情況回答以下問題：\n{query}"
        )

    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta        = chunk["metadata"]
        source      = meta.get("source", "unknown")
        title       = meta.get("title", "unknown")
        chunk_index = meta.get("chunk_index", "?")
        similarity  = chunk.get("similarity", 0.0)
        context_blocks.append(
            f"[Context {i}] 文件標題：{title} (來源：{source}，段落編號：{chunk_index}，相似度：{similarity:.3f})\n"
            f"{chunk['content']}"
        )

    context_str = "\n\n---\n\n".join(context_blocks)
    return (
        f"以下是從知識庫中檢索到的相關段落：\n\n"
        f"{context_str}\n\n"
        f"---\n\n"
        f"根據上述 Context，請回答以下問題（回答結尾須附引用來源）：\n{query}"
    )


# ===========================================================================
# LLM 呼叫（支援本地 Ollama 與教授 proxy server）
# ===========================================================================


def call_llm(messages: list[dict[str, str]], model_name: str) -> str:
    """
    透過 litellm 呼叫 LLM，回傳回應文字。
    具備自動分流邏輯：
    1. 若 model_name 以 'ollama/' 開頭，導向本地 http://localhost:11434
    2. 否則，讀取 .env 中的 LITELLM_API_KEY 與 LITELLM_BASE_URL 導向教授伺服器
    """
    logger.info("呼叫模型：%s", model_name)
    try:
        from litellm import completion  # 延遲 import，確保 --help 不受影響
    except ImportError:
        logger.error("litellm 套件未安裝，請執行：pip install litellm")
        sys.exit(1)

    try:
        # ✅ 新增：如果是本地 Ollama，強制定向到本機 11434 port
        if model_name.startswith("ollama/"):
            response = completion(
                model=model_name,
                messages=messages,
                api_base="http://localhost:11434"  # Ollama 預設網址
                # 注意：這裡不加 timeout/num_ctx，因為 rag_query 是單純的一問一答，
                # 只有 skill_builder 那邊的大篇幅排版才需要長等待與大上下文。
            )
        # ✅ 原有邏輯：如果是教授伺服器的模型 (如 openai/gemma4)
        else:
            if not LITELLM_API_KEY:
                logger.warning("LITELLM_API_KEY 尚未設定，請在 .env 填入教授提供的金鑰")
            response = completion(
                model=model_name,
                messages=messages,
                api_key=LITELLM_API_KEY,
                api_base=LITELLM_BASE_URL
            )
            
        return response.choices[0].message.content or ""

    except Exception as exc:
        logger.error("LLM 呼叫失敗（%s）：%s", type(exc).__name__, exc)
        raise

# ===========================================================================
# 核心查詢流程
#   ✅ FIX WARN-1：歷史只存「原始問題」與「assistant 回覆」，不存含 chunk 的完整 prompt。
#                  當輪 LLM 呼叫時才即時組裝帶 Context 的 prompt，避免多輪後
#                  context window 因累積 top-k chunks 而急速膨脹。
# ===========================================================================


def run_query(
    query: str,
    retriever: PgVectorRetriever,
    embedding_model: EmbeddingModel,
    top_k: int,
    conversation_history: list[dict[str, str]],
    model_name: str,
) -> tuple[str, list[dict[str, Any]]]:
    """
    執行完整 RAG 流程：Embed → Retrieve → Prompt 組裝 → LLM Generate

    Args:
        query:                使用者原始問題
        retriever:            pgvector 檢索器（已連線）
        embedding_model:      Embedding 模型實例
        top_k:                取回 top-k chunks
        conversation_history: 對話歷史（只含原始問答，不含 chunk context）

    Returns:
        (answer_text, retrieved_chunks)
    """
    # 1. 將 Query 向量化
    logger.info("Query Embedding 中...")
    query_embedding = embedding_model.encode_single(query)

    # 2. 向量搜尋
    logger.info("向量搜尋（top-%d）...", top_k)
    chunks = retriever.search(query_embedding, top_k=top_k)

    if not chunks:
        logger.warning("未檢索到任何相關段落，將以空 Context 呼叫 LLM")

    # 3. 組裝帶有 Context 的當輪 Prompt（只用於本次 LLM 呼叫，不存入歷史）
    current_user_prompt = build_rag_prompt(query, chunks)

    # 4. 歷史只截取原始問答，最後一則替換為帶 Context 的完整 prompt
    max_history_msgs = MAX_HISTORY_TURNS * 2
    trimmed_history  = conversation_history[-max_history_msgs:]

    messages_for_llm = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + trimmed_history
        + [{"role": "user", "content": current_user_prompt}]
    )

    # 5. 呼叫 LLM（固定 gemma4）
    logger.info("呼叫 LLM（model=%s）...", model_name)
    answer = call_llm(messages_for_llm, model_name=model_name)

    # 6. 歷史只存輕量的「原始問題」，保持歷史 token 不膨脹
    conversation_history.append({"role": "user",      "content": query})
    conversation_history.append({"role": "assistant", "content": answer})

    return answer, chunks


# ===========================================================================
# 輸出格式
#   ✅ FIX WARN-4：無論 LLM 是否遵從 prompt 指令輸出引用，
#                  程式層都會從 chunks metadata 補印保底引用清單。
# ===========================================================================


def print_answer(
    answer: str,
    chunks: list[dict[str, Any]],
    verbose: bool = False,
) -> None:
    """格式化輸出回答與檢索來源資訊"""
    print("\n" + "=" * 60)
    print("🤖 回答")
    print("=" * 60)
    print(answer)

    # ✅ FIX WARN-4：程式化保底引用輸出
    #    評分時終端機輸出必須顯示來源，此區塊確保無論 LLM 是否遵從指令都有引用資訊
    print("\n" + "-" * 60)
    print(f"📚 檢索來源（共 {len(chunks)} 筆）")
    print("-" * 60)
    if chunks:
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk["metadata"]
            print(
                f"  [{i}] 檔案：{meta.get('source', 'unknown')}"
                f"　段落編號：{meta.get('chunk_index', '?')}"
                f"　相似度：{chunk.get('similarity', 0.0):.4f}"
            )
            if verbose:
                preview = chunk["content"][:120].replace("\n", " ")
                print(f"       預覽：{preview}...")
    else:
        print("  （未檢索到相關段落）")
    print()


# ===========================================================================
# CLI 解析
# ===========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RAG 問答 CLI：從 pgvector 檢索相關段落，"
            f"透過教授 proxy server 以 {LLM_MODEL} 生成回答"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python rag_query.py                              # 互動式模式
  python rag_query.py --query "什麼是 Transformer？"
  python rag_query.py --query "..." --top-k 8
  python rag_query.py --query "..." --verbose      # 顯示 chunk 預覽
        """,
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="單次查詢模式：直接提供問題字串（省略則進入互動式模式）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        dest="top_k",
        help=f"從向量資料庫取回的相關段落數量，預設 {DEFAULT_TOP_K}",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="顯示每個檢索 chunk 的內容預覽（預設關閉）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=LLM_MODEL,
        help="LLM 模型名稱，預設 openai/gemma4",
    )
    return parser.parse_args()


# ===========================================================================
# 執行模式
# ===========================================================================


def interactive_mode(
    retriever: PgVectorRetriever,
    embedding_model: EmbeddingModel,
    top_k: int,
    verbose: bool,
    model_name: str,
) -> None:
    """互動式問答模式，保留多輪對話歷史（最多 MAX_HISTORY_TURNS 輪）"""
    conversation_history: list[dict[str, str]] = []

    print("\n" + "=" * 60)
    print("🔍 RAG 互動式問答模式")
    print(f"   模型：{LLM_MODEL}  |  Top-K：{top_k}  |  最大歷史輪數：{MAX_HISTORY_TURNS}")
    print("   輸入 'exit' 或 'quit' 離開，輸入 'clear' 清除對話歷史")
    print("=" * 60 + "\n")

    turn = 0
    while True:
        try:
            query = input("❓ 你的問題：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 已離開問答模式。")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            print("👋 已離開問答模式。")
            break

        if query.lower() == "clear":
            conversation_history.clear()
            turn = 0
            print("🗑️  對話歷史已清除。\n")
            continue

        turn += 1
        print(f"\n[第 {turn} 輪]", end=" ", flush=True)

        try:
            answer, chunks = run_query(
                query=query,
                retriever=retriever,
                embedding_model=embedding_model,
                top_k=top_k,
                conversation_history=conversation_history,
                model_name=model_name,
            )
            print_answer(answer, chunks, verbose=verbose)

        except KeyboardInterrupt:
            print("\n\n⚠️  查詢中斷。")
            if conversation_history and conversation_history[-1]["role"] == "user":
                conversation_history.pop()

        except Exception as exc:  # noqa: BLE001
            logger.error("查詢失敗：%s", exc)
            if conversation_history and conversation_history[-1]["role"] == "user":
                conversation_history.pop()
            print("❌ 查詢發生錯誤，請重試。（詳情見上方 log）\n")


def single_query_mode(
    query: str,
    retriever: PgVectorRetriever,
    embedding_model: EmbeddingModel,
    top_k: int,
    verbose: bool,
    model_name: str,
) -> None:
    """單次查詢模式"""
    conversation_history: list[dict[str, str]] = []
    try:
        answer, chunks = run_query(
            query=query,
            retriever=retriever,
            embedding_model=embedding_model,
            top_k=top_k,
            conversation_history=conversation_history,
            model_name=model_name,
        )
        print_answer(answer, chunks, verbose=verbose)
    except Exception as exc:
        logger.error("查詢失敗：%s", exc)
        sys.exit(1)


# ===========================================================================
# 入口
# ===========================================================================


def main() -> None:
    args = parse_args()

    # 初始化 Embedding 模型（可能需要從 HuggingFace 下載，放在 DB 連線前執行）
    try:
        embedding_model = EmbeddingModel(model_name=EMBEDDING_MODEL_NAME)
    except Exception as exc:
        logger.error("Embedding 模型載入失敗：%s", exc)
        sys.exit(1)

    # 連線 pgvector 並執行查詢
    try:
        import psycopg2  # 延遲 import：僅在此捕捉 OperationalError 需要型別
        with PgVectorRetriever(PGVECTOR_CONN_STR) as retriever:
            if args.query is not None:
                single_query_mode(
                    query=args.query,
                    retriever=retriever,
                    embedding_model=embedding_model,
                    top_k=args.top_k,
                    verbose=args.verbose,
                    model_name=args.model,
                )
            else:
                interactive_mode(
                    retriever=retriever,
                    embedding_model=embedding_model,
                    top_k=args.top_k,
                    verbose=args.verbose,
                    model_name=args.model,
                )
    except psycopg2.OperationalError:
        # 詳細錯誤訊息已在 PgVectorRetriever.connect() 中輸出
        sys.exit(1)
    except Exception as exc:
        logger.error("執行失敗：%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()