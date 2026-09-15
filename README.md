# Industry Analyst RAG Assistant

> Retrieval-augmented question answering over a personal corpus of 25 analyst reports and standards documents on the silicon-photonics and optical-networking industry, served through a local embedding model and a pgvector index.

`Course project` · Netdb Lab, NCKU · AIASE 2026 · Individual
**Stack:** Python >= 3.10 (developed on 3.14.3) · pgvector (PostgreSQL) · sentence-transformers · LangChain · LiteLLM

## Overview

I built a two-stage ETL pipeline that extracts text from 25 PDF/TXT sources (brokerage research, OIF standards, government and news reports), cleans it, and writes it to `data/processed/`. A second stage chunks the cleaned text, embeds it with a local `sentence-transformers` model, and stores the vectors in a pgvector table with an HNSW index. At query time, `rag_query.py` embeds the question, retrieves the closest chunks above a similarity threshold, and asks an LLM (via LiteLLM) to answer using only that retrieved context. A third script, `skill_builder.py`, runs 10 fixed questions against the whole corpus and writes the answers to `skill.md` as a standing summary of what the knowledge base covers.

## Key Design Decisions

| Decision | Rationale |
| --- | --- |
| `chunk_size=300`, `chunk_overlap=50` (`RecursiveCharacterTextSplitter`) | Analyst reports are text-dense; larger chunks (~1000 chars) diluted embedding similarity in testing, smaller chunks (~100 chars) lost surrounding context. 300 chars holds roughly one to two full sentences |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim) | Source text mixes English technical terms with Chinese prose; this model handles that mix better than an English-only model, runs fully offline, and has no API cost or rate limit |
| pgvector with HNSW (`m=16, ef_construction=64`) over IVFFlat | HNSW builds a usable index without needing a large initial dataset and supports incremental inserts, which fits the incremental-update design below |
| Top-K = 5, similarity threshold = 0.3 | 5 chunks of 300 chars stays within the LLM's context window while giving source diversity; chunks below 0.3 cosine similarity are dropped as noise before they reach the prompt |
| Two-stage idempotency: SHA-256 hash per source file (extract stage), `TRUNCATE` + full rebuild on `--rebuild` (embed stage) | Decouples "did this file already get cleaned" from "does the index need to be rebuilt," so re-running extraction is cheap and rebuilding the index is explicit |

## Challenges

**Problem.** Chunk size was a direct trade-off: large chunks (~1000 chars) diluted the embedding and lowered retrieval hit rate on specific questions; small chunks (~100 chars) didn't carry enough surrounding context for the LLM to answer correctly.
**Approach.** Set `chunk_size=300, chunk_overlap=50` in `RecursiveCharacterTextSplitter`, sized to hold about one to two full sentences, with the overlap so a sentence isn't cut across a chunk boundary.
**Result.** Retrieval returns relevant chunks for topic-specific questions in manual testing against the silicon-photonics corpus. There is no quantitative retrieval benchmark (precision/recall) yet — TODO.

## Limitations

- No multi-hop reasoning: a question that requires combining facts from several different reports (e.g., a trend over time across sources) can miss information, since retrieval is single-pass over chunks.
- Table and chart content in PDFs is not reliably extracted — `pypdf` preserves plain text well but loses complex table structure.
- The embedding model is general-purpose multilingual, not domain-tuned, so precision on highly specific terms (e.g. `coherent DSP`) is limited.
- `data/raw/` and `data/processed/` are excluded via `.gitignore` and not included in this repo — the source PDFs are third-party copyrighted analyst reports and news articles, and several contain real contact emails. To reproduce, supply your own documents in `data/raw/` before running `data_update.py`.

## Running It

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# fill in LITELLM_API_KEY, LITELLM_BASE_URL; PGVECTOR_CONNECTION_STRING
# defaults already match docker-compose.yml

docker compose up -d
docker compose ps   # wait for rag_pgvector to be "healthy"

python data_update.py --rebuild                 # extract, clean, chunk, embed
python rag_query.py --query "What are the main trends in this knowledge base?"
python skill_builder.py --output skill.md
```

## Structure

    data_update.py      Two-stage ETL: extract+clean (stage 1), chunk+embed+write to pgvector (stage 2)
    rag_query.py         Embed query -> retrieve top-k chunks -> generate answer via LiteLLM
    skill_builder.py     Runs 10 fixed questions against the corpus, writes skill.md
    data/raw, data/processed   Source documents (gitignored, not included — see Limitations)

---
Original course-assignment README (in Chinese, incl. full data-source citation table): [docs/course-requirements.md](docs/course-requirements.md)
