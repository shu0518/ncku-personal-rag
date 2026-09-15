"""
data_update.py — RAG 資料管線腳本
=====================================
階段一：資料萃取與清理（data/raw/ → data/processed/）
階段二：Chunking、Embedding、寫入 pgvector

執行方式：
  python data_update.py            # 增量更新（跳過已處理的檔案）
  python data_update.py --rebuild  # 全量重建（清空 DB 與 processed/ 再重跑）
  python data_update.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 基本設定
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 路徑
RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))

# pgvector
PGVECTOR_CONN_STR = os.getenv(
    "PGVECTOR_CONNECTION_STRING",
    "postgresql://raguser:ragpassword@localhost:5432/ragdb",
)

# Embedding
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)

TABLE_NAME = "document_chunks"


# ===========================================================================
# 階段一：資料萃取與清理
# ===========================================================================


class BaseExtractor(ABC):
    """文件萃取器基底類別"""

    @abstractmethod
    def extract(self, path: Path) -> str:
        """讀取原始檔案並回傳原始文字"""


class TxtExtractor(BaseExtractor):
    def extract(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")


class MarkdownExtractor(BaseExtractor):
    def extract(self, path: Path) -> str:
        return path.read_text(encoding="utf-8", errors="replace")


class PdfExtractor(BaseExtractor):
    def extract(self, path: Path) -> str:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n".join(pages)


# 副檔名 → 萃取器對應表
EXTRACTOR_MAP: dict[str, BaseExtractor] = {
    ".txt": TxtExtractor(),
    ".md": MarkdownExtractor(),
    ".pdf": PdfExtractor(),
}


def clean_text(raw: str) -> str:
    """
    文字清理流程：
    1. 將各種換行符號統一為 \\n
    2. 移除多餘的空白行（連續超過兩行空白壓縮為一行）
    3. 合併每行內的多餘空白
    4. 去除首尾空白
    """
    # 統一換行
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # 移除頁首頁尾常見的頁碼殘留（如「Page 1 of 10」）
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    # 合併行內多餘空白（保留換行符）
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    # 壓縮連續空白行（最多保留一行空白）
    cleaned_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 1:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def file_hash(path: Path) -> str:
    """計算檔案 SHA-256，用於增量更新判斷"""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def stage1_extract_and_clean(
    raw_dir: Path,
    processed_dir: Path,
    rebuild: bool = False,
) -> list[Path]:
    """
    階段一主函式：萃取 + 清理 + 存入 processed/

    Args:
        raw_dir:       原始資料目錄
        processed_dir: 清理後文字的輸出目錄
        rebuild:       True 時清空 processed/ 並全量重建

    Returns:
        本次實際處理（寫入）的 processed 檔案路徑清單
    """
    if rebuild:
        logger.info("--rebuild 模式：清空 %s", processed_dir)
        if processed_dir.exists():
            shutil.rmtree(processed_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)

    # 用於增量更新：紀錄已處理過的 hash（存在 processed/ 下的隱藏 JSON）
    hash_record_path = processed_dir / ".file_hashes.json"
    hash_record: dict[str, str] = {}
    if not rebuild and hash_record_path.exists():
        try:
            hash_record = json.loads(hash_record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            hash_record = {}

    supported_suffixes = set(EXTRACTOR_MAP.keys())
    raw_files = [
        f
        for f in raw_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in supported_suffixes
    ]

    if not raw_files:
        logger.warning("在 %s 中找不到任何支援的檔案（.pdf / .txt / .md）", raw_dir)

    written: list[Path] = []

    for raw_path in sorted(raw_files):
        current_hash = file_hash(raw_path)
        rel = raw_path.relative_to(raw_dir)

        # 計算 processed 目標路徑（保留子目錄結構，副檔名改 .txt）
        proc_path = processed_dir / rel.with_suffix(".txt")

        # 增量更新：hash 相同且 processed 檔案存在 → 跳過
        hash_key = str(rel)
        if (
            not rebuild
            and hash_record.get(hash_key) == current_hash
            and proc_path.exists()
        ):
            logger.info("  [跳過] %s（未變動）", rel)
            continue

        logger.info("  [處理] %s", rel)
        extractor = EXTRACTOR_MAP[raw_path.suffix.lower()]

        try:
            raw_text = extractor.extract(raw_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("    萃取失敗：%s", exc)
            continue

        cleaned = clean_text(raw_text)

        if not cleaned:
            logger.warning("    清理後文字為空，跳過：%s", rel)
            continue

        proc_path.parent.mkdir(parents=True, exist_ok=True)
        proc_path.write_text(cleaned, encoding="utf-8")
        hash_record[hash_key] = current_hash
        written.append(proc_path)
        logger.info("    ✓ 已寫入 %s", proc_path)

    # 更新 hash 紀錄
    hash_record_path.write_text(
        json.dumps(hash_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("階段一完成：本次處理 %d 個檔案", len(written))
    return written


# ===========================================================================
# 階段二：Chunking、Embedding、Vector DB 寫入
# ===========================================================================


class BaseChunkingStrategy(ABC):
    """Chunking 策略基底類別（Strategy Pattern）"""

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """將文字切分為 chunk 列表"""


class RecursiveCharacterChunker(BaseChunkingStrategy):
    """
    使用 LangChain RecursiveCharacterTextSplitter 的 Chunking 策略。
    預設：chunk_size=300, chunk_overlap=50
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50) -> None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def split(self, text: str) -> list[str]:
        return self._splitter.split_text(text)


class EmbeddingModel:
    """sentence-transformers 本地 Embedding 模型封裝"""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer
        logger.info("載入 Embedding 模型：%s（首次執行會自動下載）", model_name)
        self._model = SentenceTransformer(model_name)
        self.dimension: int = self._model.get_sentence_embedding_dimension()
        logger.info("模型向量維度：%d", self.dimension)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, show_progress_bar=False)
        return vectors.tolist()


class PgVectorStore:
    """pgvector 向量資料庫操作封裝"""

    def __init__(self, connection_string: str, dimension: int) -> None:
        self._conn_str = connection_string
        self.dimension = dimension
        self._conn: Any = None

    # ------------------------------------------------------------------
    # 連線管理
    # ------------------------------------------------------------------

    def connect(self) -> None:
        import psycopg2
        logger.info("連線至 pgvector：%s", self._conn_str)
        self._conn = psycopg2.connect(self._conn_str)
        self._conn.autocommit = False

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    def __enter__(self) -> "PgVectorStore":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Schema 初始化
    # ------------------------------------------------------------------

    def initialize_schema(self) -> None:
        """建立 extension 與資料表（若不存在）"""
        with self._conn.cursor() as cur:  # type: ignore[union-attr]
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                    id        SERIAL PRIMARY KEY,
                    content   TEXT        NOT NULL,
                    embedding vector({self.dimension}) NOT NULL,
                    metadata  JSONB       NOT NULL DEFAULT '{{}}'::jsonb
                );
                """
            )
            # 建立 IVFFlat 近似近鄰搜尋索引（若資料量大時加速查詢）
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {TABLE_NAME}_embedding_idx
                ON {TABLE_NAME}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                """
            )
        self._conn.commit()  # type: ignore[union-attr]
        logger.info("Schema 初始化完成（TABLE: %s）", TABLE_NAME)

    # ------------------------------------------------------------------
    # 資料操作
    # ------------------------------------------------------------------

    def truncate(self) -> None:
        """清空資料表（--rebuild 時呼叫）"""
        with self._conn.cursor() as cur:  # type: ignore[union-attr]
            cur.execute(f"TRUNCATE TABLE {TABLE_NAME} RESTART IDENTITY;")
        self._conn.commit()  # type: ignore[union-attr]
        logger.info("資料表 %s 已清空", TABLE_NAME)

    def delete_by_source(self, source_filename: str) -> None:
        """刪除特定來源檔案的所有 chunk（增量更新時先刪後寫）"""
        with self._conn.cursor() as cur:  # type: ignore[union-attr]
            cur.execute(
                f"DELETE FROM {TABLE_NAME} WHERE metadata->>'source' = %s;",
                (source_filename,),
            )
        self._conn.commit()  # type: ignore[union-attr]

    def source_exists(self, source_filename: str) -> bool:
        """檢查某個來源檔案是否已有資料（增量更新判斷用）"""
        with self._conn.cursor() as cur:  # type: ignore[union-attr]
            cur.execute(
                f"SELECT 1 FROM {TABLE_NAME} WHERE metadata->>'source' = %s LIMIT 1;",
                (source_filename,),
            )
            return cur.fetchone() is not None

    def insert_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        source_filename: str,
        doc_title: str,
    ) -> None:
        """批次插入 chunk + embedding + metadata"""
        import psycopg2.extras
        records = [
            (
                chunk,
                embedding,
                json.dumps({"source": source_filename, "chunk_index": idx, "title": doc_title}, ensure_ascii=False),
            )
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        with self._conn.cursor() as cur:  # type: ignore[union-attr]
            psycopg2.extras.execute_batch(
                cur,
                f"""
                INSERT INTO {TABLE_NAME} (content, embedding, metadata)
                VALUES (%s, %s::vector, %s::jsonb);
                """,
                records,
                page_size=128,
            )
        self._conn.commit()  # type: ignore[union-attr]


def stage2_embed_and_store(
    processed_dir: Path,
    store: PgVectorStore,
    embedding_model: EmbeddingModel,
    chunker: BaseChunkingStrategy,
    rebuild: bool = False,
    newly_written: list[Path] | None = None,
) -> None:
    """
    階段二主函式：讀取 processed/ → Chunk → Embed → 寫入 pgvector

    Args:
        processed_dir:  清理後文字目錄
        store:          PgVectorStore 實例（已連線）
        embedding_model: EmbeddingModel 實例
        chunker:        Chunking 策略實例
        rebuild:        True 時先 Truncate 再全量寫入
        newly_written:  階段一本次實際處理的檔案（增量模式用）
    """
    if rebuild:
        store.truncate()
        # rebuild 時處理所有 processed 檔案
        target_files = sorted(
            f for f in processed_dir.rglob("*.txt") if not f.name.startswith(".")
        )
    else:
        # 增量模式：只處理本次階段一新寫入的檔案
        target_files = newly_written or []

    if not target_files:
        logger.info("階段二：無需更新的檔案，跳過向量化")
        return

    logger.info("階段二：開始向量化，共 %d 個 processed 檔案", len(target_files))

    for proc_path in target_files:
        source_name = proc_path.name  # 作為 metadata["source"]
        text = proc_path.read_text(encoding="utf-8")

        doc_title = "Unknown"
        for line in text.split("\n"):
            if line.strip():
                doc_title = line.strip().lstrip("#").strip()
                break

        if not text.strip():
            logger.warning("  [跳過] %s（內容為空）", source_name)
            continue

        # 增量模式下，若 DB 中已有此來源資料，先刪除（避免重複）
        if not rebuild:
            store.delete_by_source(source_name)

        chunks = chunker.split(text)
        if not chunks:
            logger.warning("  [跳過] %s（Chunking 後無內容）", source_name)
            continue

        logger.info("  [向量化] %s → %d chunks", source_name, len(chunks))
        embeddings = embedding_model.encode(chunks)
        store.insert_chunks(chunks, embeddings, source_name, doc_title)
        logger.info("    ✓ 已寫入 %d 筆 chunk", len(chunks))

    logger.info("階段二完成")


# ===========================================================================
# CLI 入口
# ===========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG 資料管線：從 data/raw/ 萃取、清理、向量化後寫入 pgvector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python data_update.py            # 增量更新
  python data_update.py --rebuild  # 全量重建
        """,
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="清空 data/processed/ 與 Vector DB 後全量重建（冪等性保證）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="Chunk 大小（字元數），預設 300",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk 重疊大小（字元數），預設 50",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EMBEDDING_MODEL_NAME,
        help=f"sentence-transformers 模型名稱，預設 {EMBEDDING_MODEL_NAME}",
    )
    return parser.parse_args()


def main() -> None:

    args = parse_args()

    # 確認 raw 目錄存在
    if not RAW_DIR.exists():
        logger.error("找不到原始資料目錄：%s，請先建立並放入資料", RAW_DIR)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("RAG 資料管線啟動（rebuild=%s）", args.rebuild)
    logger.info("  RAW_DIR:       %s", RAW_DIR)
    logger.info("  PROCESSED_DIR: %s", PROCESSED_DIR)
    logger.info("  chunk_size:    %d", args.chunk_size)
    logger.info("  chunk_overlap: %d", args.chunk_overlap)
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # 階段一：萃取與清理
    # -----------------------------------------------------------------------
    logger.info("▶ 階段一：資料萃取與清理")
    newly_written = stage1_extract_and_clean(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        rebuild=args.rebuild,
    )

    # -----------------------------------------------------------------------
    # 階段二：向量化與寫入
    # -----------------------------------------------------------------------
    logger.info("▶ 階段二：Chunking、Embedding、寫入 pgvector")

    embedding_model = EmbeddingModel(model_name=args.embedding_model)
    chunker = RecursiveCharacterChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    with PgVectorStore(PGVECTOR_CONN_STR, embedding_model.dimension) as store:
        store.initialize_schema()
        stage2_embed_and_store(
            processed_dir=PROCESSED_DIR,
            store=store,
            embedding_model=embedding_model,
            chunker=chunker,
            rebuild=args.rebuild,
            newly_written=newly_written,
        )

    logger.info("=" * 60)
    logger.info("✅ 資料管線執行完畢")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()