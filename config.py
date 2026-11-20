import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _get_config(name: str, default: str = "") -> str:
    """Read configuration from Streamlit Secrets or local environment."""
    try:
        import streamlit as st

        value = st.secrets.get(name, None)
        if value is not None:
            return str(value)
    except (ImportError, FileNotFoundError):
        pass

    return os.getenv(name, default)


def _get_int(name: str, default: int) -> int:
    raw_value = _get_config(name)

    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        return int(raw_value)
    except ValueError:
        print(f"[config] '{name}={raw_value}' is not a number. Using {default}")
        return default


PROJECT_ROOT = Path(__file__).parent

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
FAISS_DIR = VECTORSTORE_DIR / "faiss"
DOC_STORE_PATH = VECTORSTORE_DIR / "doc_store.json"


# Groq models
GROQ_API_KEY = _get_config("GROQ_API_KEY")
GROQ_CHAT_MODEL = _get_config("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")

# Embedding model
EMBEDDING_MODEL = _get_config("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

LLM_TEMPERATURE = 0.0


# Vector database
VECTOR_DB = _get_config("VECTOR_DB", "faiss").strip().lower()
PINECONE_API_KEY = _get_config("PINECONE_API_KEY")
PINECONE_INDEX_NAME = _get_config("PINECONE_INDEX_NAME", "advanced-rag")
PINECONE_NAMESPACE = _get_config("PINECONE_NAMESPACE", "default")


# Chunking
PARENT_CHUNK_SIZE = _get_int("PARENT_CHUNK_SIZE", 1600)
PARENT_CHUNK_OVERLAP = _get_int("PARENT_CHUNK_OVERLAP", 200)
CHILD_CHUNK_SIZE = _get_int("CHILD_CHUNK_SIZE", 400)
CHILD_CHUNK_OVERLAP = _get_int("CHILD_CHUNK_OVERLAP", 60)


# Retrieval pipeline
NUM_QUERY_VARIANTS = _get_int("NUM_QUERY_VARIANTS", 3)
VECTOR_TOP_K = _get_int("VECTOR_TOP_K", 6)
BM25_TOP_K = _get_int("BM25_TOP_K", 6)
HYBRID_TOP_K = _get_int("HYBRID_TOP_K", 10)
RERANK_TOP_K = _get_int("RERANK_TOP_K", 4)

MIN_RERANK_SCORE = float(_get_config("MIN_RERANK_SCORE", "-4.0"))

MAX_CONTEXT_CHARS = _get_int("MAX_CONTEXT_CHARS", 8000)

RERANKER = _get_config("RERANKER", "cross-encoder").strip().lower()

CROSS_ENCODER_MODEL = _get_config(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

MEMORY_WINDOW = _get_int("MEMORY_WINDOW", 4)


def check_groq_api_key() -> None:
    if not GROQ_API_KEY:
        raise ValueError("GROQ API key is not set")


def check_pinecone_config() -> None:
    if not PINECONE_API_KEY:
        raise ValueError("VECTOR_DB=pinecone but PINECONE_API_KEY is empty.")
    if not PINECONE_INDEX_NAME:
        raise ValueError("VECTOR_DB=pinecone but PINECONE_INDEX_NAME is empty.")


def describe_settings() -> dict:
    return {
        "Chat Model": GROQ_CHAT_MODEL,
        "Embedding Model": EMBEDDING_MODEL,
        "VectorDB": VECTOR_DB,
        "Parent Chunk Size": PARENT_CHUNK_SIZE,
        "Child Chunk Size": CHILD_CHUNK_SIZE,
    }
