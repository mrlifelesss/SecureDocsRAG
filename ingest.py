# ingest.py
"""
Ingestion script for processing and embedding documents for a RAG system.

This script performs the following actions:
1.  Scans the `DATA_DIR` for supported document types (.pdf, .html, .md, .txt).
2.  Uses appropriate LangChain loaders to extract text content from each document.
3.  Cleans and normalizes the extracted text.
4.  Adds metadata, including the source filename and a constructed title/header.
5.  Splits the documents into smaller, overlapping chunks using a recursive text splitter.
6.  Filters out any empty or trivially short chunks.
7.  Initializes Google Generative AI embeddings and performs a sanity check.
8.  Stores the document chunks and their embeddings in a Chroma vector database.
9.  (Optional) Creates and saves a BM25 sparse retriever index for keyword search.

Configuration settings are pulled from `settings.py`.
Requires a `GOOGLE_API_KEY` environment variable.
"""
import os
import logging
from settings import settings as cfg
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyMuPDFLoader,           # better PDF text extraction
    PyPDFLoader,             # fallback
    BSHTMLLoader,            # HTML via BeautifulSoup
    UnstructuredMarkdownLoader,
    TextLoader,
)

# Load GOOGLE_API_KEY from .env if present (optional)
load_dotenv()


# ==== LOADED SETTINGS ====
DATA_DIR = cfg.DATA_DIR
CHROMA_DIR = cfg.CHROMA_DIR
COLLECTION_NAME = cfg.COLLECTION_NAME
EMBED_MODEL = cfg.EMBED_MODEL
CHUNK_SIZE = cfg.CHUNK_SIZE
CHUNK_OVERLAP = cfg.CHUNK_OVERLAP
# ============================

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing. Set it as an env var or in .env")

logging.basicConfig(
    level=logging.INFO,  # change to DEBUG for more detail
    format="[%(levelname)s] %(message)s"
)
log = logging.getLogger("ingest")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

def normalize_text(text: str) -> str:
    """
    Cleans and standardizes text content.

    This function performs the following operations:
    - Replaces non-breaking spaces (`\xa0`) and right-to-left marks (`\u200f`) with a standard space.
    - Collapses multiple consecutive spaces or tabs into a single space.
    - Reduces three or more newlines into exactly two.
    - Strips leading/trailing whitespace.

    Args:
        text: The input string to normalize.

    Returns:
        The cleaned and normalized string.
    """
    import re
    text = text.replace("\xa0", " ").replace("\u200f", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_docs() -> List:
    """
    Loads and processes documents from the DATA_DIR directory.

    Iterates through files in the configured data directory, determines the
    file type by its extension, and uses the appropriate LangChain document
    loader. It skips empty or unsupported files. For each loaded document,
    it normalizes the text and prepends a header with the document's title
    and page number (if available) for better context.

    Returns:
        A list of LangChain `Document` objects, each representing a page or
        a whole document.

    Raises:
        RuntimeError: If no documents with valid text content could be extracted.
    """
    docs = []
    for fp in sorted(DATA_DIR.iterdir()):
        if not fp.is_file():
            continue
        size = fp.stat().st_size
        suffix = fp.suffix.lower()
        log.info(f"parsing {fp.name} ({size} bytes)")

        if size == 0:
            log.warning(f"  -> SKIP (empty file)")
            continue

        try:
            if suffix == ".pdf":
                # Try PyMuPDF first (best results), fallback to PyPDF
                try:
                    loader = PyMuPDFLoader(str(fp.resolve()))
                    _docs = loader.load()
                except Exception as e1:
                    log.warning(f"  -> PyMuPDF failed ({e1}), trying PyPDF")
                    loader = PyPDFLoader(str(fp.resolve()))
                    _docs = loader.load()
            elif suffix in {".html", ".htm"}:
                loader = BSHTMLLoader(str(fp.resolve()))
                _docs = loader.load()
            elif suffix == ".md":
                loader = UnstructuredMarkdownLoader(str(fp.resolve()))
                _docs = loader.load()
            elif suffix in {".txt"}:
                loader = TextLoader(str(fp.resolve()), autodetect_encoding=True)
                _docs = loader.load()
            else:
                log.warning(f"  -> SKIP (unsupported extension: {suffix})")
                continue

            # Normalize + attach source and header
            processed_docs = []
            for d in _docs:
                d.page_content = normalize_text(d.page_content or "")
                if not d.page_content.strip():
                    continue # Skip pages with no real content after normalization
                d.metadata.setdefault("source", str(fp.name))

                # Preserve page for citation and add a simple title anchor
                title = Path(fp).stem.replace("-", " ").title()
                page = d.metadata.get("page", None)
                header = f"{title} — Page {page+1}" if page is not None else title
                d.page_content = f"{header}\n\n{d.page_content}"
                processed_docs.append(d)

            docs.extend(processed_docs)
            log.info(f"  -> pages loaded: {len(_docs)}, kept (non-empty): {len(processed_docs)}")

        except Exception as e:
            log.error(f"  -> SKIP (failed to parse): {fp.name} -> {e}")

    if not docs:
        raise RuntimeError("No valid pages with text extracted. Check if source files are empty or unreadable (e.g., scanned PDFs).")
    return docs

def chunk_docs(docs: List) -> List:
    """
    Divides a list of LangChain documents into smaller chunks.

    Uses a `RecursiveCharacterTextSplitter` configured with settings from
    `settings.py` (CHUNK_SIZE, CHUNK_OVERLAP) to split the documents.

    Args:
        docs: A list of LangChain `Document` objects.

    Returns:
        A list of smaller `Document` chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

def non_empty_chunks(chunks: List) -> List:
    """
    Filters a list of chunks, keeping only those with meaningful text content.

    A chunk is kept if its `page_content` is not empty after stripping
    whitespace and has a minimum length of 5 characters.

    Args:
        chunks: A list of `Document` chunks.

    Returns:
        A filtered list of `Document` chunks.
    """
    out = []
    for d in chunks:
        content = d.page_content.strip()
        # enforce a tiny minimum length to avoid empty/garbage rows
        if content and len(content) >= 5:
            d.page_content = content # Use the stripped version
            out.append(d)
    return out

def main():
    """
    Executes the main document ingestion and embedding pipeline.

    Orchestrates the loading, chunking, embedding, and storage of documents.
    It persists the resulting vectors to a Chroma database and optionally
    creates a BM25 retriever index.
    """
    # 1) Load & chunk
    docs = load_docs()
    log.info(f"Loaded {len(docs)} document pages in total.")
    chunks = chunk_docs(docs)
    log.info(f"Split documents into {len(chunks)} raw chunks.")

    # 2) Filter out empties
    chunks = non_empty_chunks(chunks)
    log.info(f"Filtered down to {len(chunks)} non-empty chunks.")
    if not chunks:
        raise RuntimeError("All chunks are empty after filtering. Check your source files in data/.")

    # 3) Quick sanity test: embed a sample
    from itertools import islice
    log.info("Performing embedding sanity check...")
    sample_texts = [c.page_content[:5000] for c in islice(chunks, 3)]
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=API_KEY)
    try:
        test_vecs = embeddings.embed_documents(sample_texts)
        if not test_vecs or any(len(v) == 0 for v in test_vecs):
            raise ValueError("Got empty vectors from embedding service.")
        log.info(f"Embedding sanity check PASSED. Vector dimension: {len(test_vecs[0])}")
    except Exception as e:
        log.error(f"Embedding sanity check FAILED: {e}")
        raise RuntimeError("Could not generate embeddings. Verify GOOGLE_API_KEY and network access.") from e

    # 4) Persist to Chroma
    log.info("Storing document chunks in ChromaDB...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME, # Use variable from settings
    )
    vectorstore.persist()
    log.info(f"OK: Stored {len(chunks)} chunks in Chroma collection '{COLLECTION_NAME}' at {CHROMA_DIR}")

    # 5) Create and save a BM25 sparse retriever index (optional)
    try:
        import pickle
        from langchain_community.retrievers import BM25Retriever
        log.info("Creating BM25 keyword index...")
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_path = CHROMA_DIR / "bm25.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
        log.info(f"Saved BM25 keyword index -> {bm25_path}")
    except Exception as e:
        log.warning(f"BM25 index not created (optional): {e}")

if __name__ == "__main__":
    main()