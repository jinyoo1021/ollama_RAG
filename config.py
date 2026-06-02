import os
from pathlib import Path

# PATH
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "pdfs"
VECTOR_DB_DIR = BASE_DIR / "chroma_db"
INDEX_DIR = BASE_DIR / "indexes"
CHUNK_STORE_PATH = INDEX_DIR / "chunks.json"

# MODEL
OLLAMA_MODEL = "exaone3.5:7.8b"
FALLBACK_OLLAMA_MODEL = "exaone3.5:2.4b"
EMBEDDING_MODEL = "nlpai-lab/KURE-v1"
ALTERNATIVE_EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# DEVICE
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", EMBED_DEVICE)

# CHUNKING
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 120
MIN_CHUNK_CHARS = 30
PAGE_CONTEXT_MAX_CHARS = 3500
MARKDOWN_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

# SEARCH
RETRIEVAL_TOP_K = 10
RERANK_TOP_K = 3
BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"
RERANKER_USE_FP16 = EMBED_DEVICE == "cuda"
RERANKER_BATCH_SIZE = 8
SOURCE_SCORE_MARGIN = 0.05
SOURCE_MAX_PAGES = None

# CONVERSION
MAX_HISTORY_TURNS = 5
CHROMA_COLLECTION_NAME = "pdf_rag"

# Development default: rebuild indexes to avoid silent duplicate chunks.
REBUILD_INDEX = True
