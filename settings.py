"""Centralized configuration for the RAG chatbot.

This module exposes a single, frozen dataclass instance `settings` that the rest
of the codebase can import. Defaults are sensible for local development, and every
setting can be overridden by environment variables (optionally loaded from `.env`).

Typical usage:
    from settings import settings as cfg
    vectordb_dir = cfg.CHROMA_DIR
    model = cfg.GEMINI_MODEL

Environment variables (selected):
    - GOOGLE_API_KEY            : Required to call Gemini (generation + embeddings)
    - DATA_DIR                  : Path to raw documents folder (default: "data")
    - CHROMA_DIR                : Path to Chroma persistence (default: "storage")
    - COLLECTION_NAME           : Chroma collection name (default: "knowledge_base")
    - EMBED_MODEL               : Embedding model id (default: "models/text-embedding-004")
    - GEMINI_MODEL              : Chat model id     (default: "gemini-2.5-flash-lite")
    - GEMINI_TRANSPORT          : "rest" (default) or "grpc"; "rest" avoids Windows gRPC loop issues
    - CHUNK_SIZE, CHUNK_OVERLAP : Ingestion chunking knobs (ints)
    - RETRIEVE_K, FETCH_K       : Retrieval breadth for MMR (ints)
    - MMR_LAMBDA                : MMR diversity (float 0..1; lower = more diversity)
    - MIN_CONTEXT_CHARS         : Guard threshold: below this, context considered thin
    - RETRY_K                   : Breadth for broaden attempts
    - WEAK_SUPPORT              : Token-overlap threshold for “is context about the question?”
    - BROADEN_MAX_TRIES         : Max iterations of broaden-then-check loop
    - BROADEN_SCHEDULE          : Semicolon-separated rounds "k,lambda,bm25Top;..."
    - TOPIC_ROUTING_ENABLED     : If True, retrieval may use doc metadata topic filters
    - RAG_DEBUG                 : Reserved flag for verbose diagnostics

See `Settings.from_env()` docstring for details on parsing and defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

try:
    # Optional: if present, load .env to populate environment variables in dev.
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # Silently ignore if python-dotenv is not installed.
    pass


def _getenv_str(key: str, default: str) -> str:
    """Return an environment variable as a non-empty string, else the default.

    Leading/trailing whitespace is stripped. If the env var is set but empty,
    the provided default is returned instead.
    """
    return os.getenv(key, default).strip() or default


def _getenv_int(key: str, default: int) -> int:
    """Return an environment variable parsed as int, else the default."""
    try:
        return int(os.getenv(key, "").strip())
    except Exception:
        return default


def _getenv_float(key: str, default: float) -> float:
    """Return an environment variable parsed as float, else the default."""
    try:
        return float(os.getenv(key, "").strip())
    except Exception:
        return default


def _getenv_bool(key: str, default: bool) -> bool:
    """Return an environment variable parsed as boolean, else the default.

    Truthy values: 1, true, t, yes, y
    Falsy values : 0, false, f, no, n
    Comparison is case-insensitive; any other value yields the default.
    """
    v = os.getenv(key, "").strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    return default


def _getenv_path(key: str, default: str) -> Path:
    """Return an environment variable as a `pathlib.Path`, else the default path."""
    val = _getenv_str(key, default)
    return Path(val)


def _parse_schedule(env_val: str | None, default: List[Tuple[int, float, int]]) -> List[Tuple[int, float, int]]:
    """Parse a broaden schedule string into a list of (k, lambda_mult, bm25_top) tuples.

    Expected format in env (semicolon-separated rounds):
        "10,0.25,10;14,0.18,12;18,0.12,16"

    Returns:
        A list like: [(10, 0.25, 10), (14, 0.18, 12), (18, 0.12, 16)].
        If parsing fails, the provided default is returned unchanged.
    """
    if not env_val:
        return default
    out: List[Tuple[int, float, int]] = []
    try:
        parts = [p.strip() for p in env_val.split(";") if p.strip()]
        for p in parts:
            k_s, lam_s, top_s = [x.strip() for x in p.split(",")]
            out.append((int(k_s), float(lam_s), int(top_s)))
        return out or default
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable configuration container for the RAG system.

    Attributes:
        DOMAIN: Human-friendly scope label used in prompts/UX.
        DATA_DIR: Directory containing raw documents for ingestion.
        CHROMA_DIR: Directory for Chroma’s persisted index (vector store).
        COLLECTION_NAME: Chroma collection name for this project.

        GOOGLE_API_KEY: API key for Gemini (embeddings + chat via LangChain).
        EMBED_MODEL: Embedding model identifier.
        GEMINI_MODEL: Chat/generation model identifier.
        GEMINI_TRANSPORT: Client transport ("rest" recommended on Windows).

        CHUNK_SIZE: Character window for chunking documents during ingest.
        CHUNK_OVERLAP: Overlap between consecutive chunks (characters).

        RETRIEVE_K: Number of documents to keep after MMR.
        FETCH_K: Candidate pool size for MMR.
        MMR_LAMBDA: MMR diversity parameter (0..1; lower = more diversity).

        MIN_CONTEXT_CHARS: If retrieved context is shorter than this, it’s “thin”.
        RETRY_K: Breadth for a broadened retrieval attempt.
        WEAK_SUPPORT: Token-overlap threshold indicating weak question-context support.
        BROADEN_MAX_TRIES: Max rounds of broaden-then-check loop.
        BROADEN_SCHEDULE: Per-round (k, lambda_mult, bm25_top) settings.

        TOPIC_ROUTING_ENABLED: If True, retrieval may apply metadata topic filters.
        RAG_DEBUG: Reserved flag for verbose logging/diagnostics (off by default).
    """

    # --------- Core domain / UX ----------
    DOMAIN: str

    # --------- Data & Vector store -------
    DATA_DIR: Path
    CHROMA_DIR: Path
    COLLECTION_NAME: str

    # --------- Models / Providers --------
    GOOGLE_API_KEY: str
    EMBED_MODEL: str                # embeddings
    GEMINI_MODEL: str               # chat/generation (Gemini via langchain-google-genai)
    GEMINI_TRANSPORT: str           # "rest" avoids gRPC event loop issues on Windows

    # --------- Ingestion / Chunking -----
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int

    # --------- Retrieval / Rerank -------
    RETRIEVE_K: int                 # top docs to keep after MMR
    FETCH_K: int                    # candidate pool for MMR
    MMR_LAMBDA: float               # 0..1 (lower = more diversity)

    # --------- Guardrail / Fallback -----
    MIN_CONTEXT_CHARS: int          # too-thin context triggers broaden/clarify
    RETRY_K: int                    # breadth for broaden attempts
    WEAK_SUPPORT: float             # token-overlap threshold (0.04–0.08 typical)
    BROADEN_MAX_TRIES: int          # max broadening rounds before clarification
    BROADEN_SCHEDULE: List[Tuple[int, float, int]]  # [(k, lambda_mult, bm25_top), ...]

    # --------- Feature flags -------------
    TOPIC_ROUTING_ENABLED: bool     # if you tag docs with "topic" in metadata
    RAG_DEBUG: bool                 # reserved; keep False unless you add logging

    MCP_SERVER_URL: str
    @staticmethod
    def from_env() -> "Settings":
        """Build a `Settings` object from environment variables (with defaults).

        Parsing rules:
            - Strings are stripped; empty strings revert to defaults.
            - Integers/floats/bools use forgiving converters; on parse error, defaults apply.
            - BROADEN_SCHEDULE uses `_parse_schedule` and accepts:
              "k,lambda,bm25Top;...". Example: "10,0.25,10;14,0.18,12".

        Returns:
            A frozen `Settings` instance that can be imported application-wide.
        """
        # Core
        DOMAIN = _getenv_str("DOMAIN", "Cloud Security (AWS/IAM)")

        # Storage
        DATA_DIR = _getenv_path("DATA_DIR", "data")
        CHROMA_DIR = _getenv_path("CHROMA_DIR", "storage")
        COLLECTION_NAME = _getenv_str("COLLECTION_NAME", "knowledge_base")

        # Models
        GOOGLE_API_KEY = _getenv_str("GOOGLE_API_KEY", "")
        EMBED_MODEL = _getenv_str("EMBED_MODEL", "models/text-embedding-004")
        GEMINI_MODEL = _getenv_str("GEMINI_MODEL", "gemini-2.5-flash-lite")
        GEMINI_TRANSPORT = _getenv_str("GEMINI_TRANSPORT", "rest")

        # Ingestion
        CHUNK_SIZE = _getenv_int("CHUNK_SIZE", 600)          # bump to 900 if lists get split
        CHUNK_OVERLAP = _getenv_int("CHUNK_OVERLAP", 120)

        # Retrieval / Rerank
        RETRIEVE_K = _getenv_int("RETRIEVE_K", 8)
        FETCH_K = _getenv_int("FETCH_K", 40)
        MMR_LAMBDA = _getenv_float("MMR_LAMBDA", 0.6)

        # Guardrail / Fallback
        MIN_CONTEXT_CHARS = _getenv_int("MIN_CONTEXT_CHARS", 250)
        RETRY_K = _getenv_int("RETRY_K", 10)
        WEAK_SUPPORT = _getenv_float("WEAK_SUPPORT", 0.05)
        BROADEN_MAX_TRIES = _getenv_int("BROADEN_MAX_TRIES", 3)
        BROADEN_SCHEDULE = _parse_schedule(
            os.getenv("BROADEN_SCHEDULE"),
            default=[(max(RETRY_K, 10), 0.25, 10),
                     (max(RETRY_K, 14), 0.18, 12),
                     (max(RETRY_K, 18), 0.12, 16)]
        )

        # Feature flags
        TOPIC_ROUTING_ENABLED = _getenv_bool("TOPIC_ROUTING_ENABLED", True)
        RAG_DEBUG = _getenv_bool("RAG_DEBUG", False)
       
        MCP_SERVER_URL = _getenv_str("MCP_SERVER_URL", "http://localhost:8999")
       
        return Settings(
            DOMAIN=DOMAIN,
            DATA_DIR=DATA_DIR,
            CHROMA_DIR=CHROMA_DIR,
            COLLECTION_NAME=COLLECTION_NAME,
            GOOGLE_API_KEY=GOOGLE_API_KEY,
            EMBED_MODEL=EMBED_MODEL,
            GEMINI_MODEL=GEMINI_MODEL,
            GEMINI_TRANSPORT=GEMINI_TRANSPORT,
            CHUNK_SIZE=CHUNK_SIZE,
            CHUNK_OVERLAP=CHUNK_OVERLAP,
            RETRIEVE_K=RETRIEVE_K,
            FETCH_K=FETCH_K,
            MMR_LAMBDA=MMR_LAMBDA,
            MIN_CONTEXT_CHARS=MIN_CONTEXT_CHARS,
            RETRY_K=RETRY_K,
            WEAK_SUPPORT=WEAK_SUPPORT,
            BROADEN_MAX_TRIES=BROADEN_MAX_TRIES,
            BROADEN_SCHEDULE=BROADEN_SCHEDULE,
            TOPIC_ROUTING_ENABLED=TOPIC_ROUTING_ENABLED,
            RAG_DEBUG=RAG_DEBUG,
            MCP_SERVER_URL=MCP_SERVER_URL,
        )


# Singleton instance you can import everywhere:
settings = Settings.from_env()
