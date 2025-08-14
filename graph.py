# graph.py
"""
Defines the core RAG (Retrieval-Augmented Generation) graph using LangGraph.

This module sets up a multi-step process to answer questions based on a document
corpus stored in a Chroma vector database. The graph is designed to be robust,
with steps for query rewriting, hybrid retrieval, re-ranking, and a generation
stage that includes guardrails against answering with low-confidence context.

Key Components:
- **GraphState**: A TypedDict defining the data structure that flows through the graph.
- **Core Components**: Initializes the Gemini LLM, GoogleGenAIEmbeddings, and the Chroma vector store retriever.
- **Prompt Templates**: Defines the system and user prompts for rewriting, re-ranking, and final answer generation.
- **Graph Nodes**:
    1. `rewrite_node`: Rewrites the user's question into a better search query.
    2. `retrieve_node`: Fetches documents using a hybrid approach (vector search + optional BM25 keyword search).
    3. `rerank_node`: Uses the LLM to re-rank retrieved documents for relevance.
    4. `generate_node`: Generates an answer based on the re-ranked context, with fallbacks for low-quality context.
- **Orchestration**:
    - `build_app()`: Compiles the nodes into a runnable LangGraph application.
    - `prepare_context()`: A helper to run the retrieval pipeline up to the re-ranking step.
    - `stream_generate()`: A helper to stream the final answer generation.
"""
import os
import re
import json
import pickle
from pathlib import Path
from typing import TypedDict, List, Dict, Any

from dotenv import load_dotenv
from settings import settings as cfg

# --- Imports from LangChain/LangGraph ---
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------------------------------
# 1. SETUP AND CONFIGURATION
# --------------------------------------------------------------------------

# Load GOOGLE_API_KEY from .env if present (optional)
load_dotenv()

# --- Loaded Settings ---
DOMAIN = cfg.DOMAIN
CHROMA_DIR = cfg.CHROMA_DIR
COLLECTION_NAME = cfg.COLLECTION_NAME
EMBED_MODEL = cfg.EMBED_MODEL
GEMINI_MODEL = cfg.GEMINI_MODEL

# Retrieval & Reranking Parameters
RETRIEVE_K = cfg.RETRIEVE_K
FETCH_K = cfg.FETCH_K
MMR_LAMBDA = cfg.MMR_LAMBDA
RETRY_K = cfg.RETRY_K

# Generation Guardrail Parameters
MIN_CONTEXT_CHARS = cfg.MIN_CONTEXT_CHARS
WEAK_SUPPORT = cfg.WEAK_SUPPORT

# --- Static Data ---
TOPIC_KEYWORDS = {
    "caf":    ["caf", "cloud adoption framework", "capabilities"],
    "sra":    ["security reference architecture", "sra", "multi-account"],
    "ir":     ["incident response", "nist", "detect", "analyz", "contain", "eradicate", "recover"],
    "lambda": ["lambda", "function", "execution role", "microvm"],
    "iam":    ["iam", "policy", "role", "least privilege", "access analyzer"],
    "s3":     ["s3", "bucket", "encryption", "sse-kms", "sse-s3"],
    "kms":    ["kms", "key management", "cmk", "customer managed key"],
}

STOPWORDS = {
    "a","an","the","and","or","of","to","in","on","for","with","by","is","are","was","were",
    "be","as","at","from","that","this","these","those","it","its","into","your","you","what",
    "which","when","why","how","do","does","did","can","could","should","would"
}
NUM_WORDS = {
    "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10
}
# --- API Key Validation ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing. Set it as an env var or in .env")


# --------------------------------------------------------------------------
# 2. STATE DEFINITION
# --------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    """
    Defines the state that flows through the RAG graph.

    Attributes:
        question: The original user question.
        search_query: The rewritten query for retrieval.
        chat_history: A list of previous conversation turns.
        raw_docs: All unique documents retrieved from vector and keyword search.
        kept_docs: The subset of documents kept after re-ranking.
        prior_docs: Documents from a previous turn, used for conversational memory.
        context: The formatted string of document content passed to the LLM.
        answer: The final generated answer.
        sources: A list of source metadata for cited documents.
    """
    question: str
    search_query: str
    chat_history: List[Dict[str, str]]
    raw_docs: List[Any]
    kept_docs: List[Any]
    prior_docs: List[Any]
    context: str
    answer: str
    sources: List[Dict[str, Any]]
    text: str

# --------------------------------------------------------------------------
# 3. CORE COMPONENT INITIALIZATION
# --------------------------------------------------------------------------

# --- Embeddings and Vector Store ---
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBED_MODEL,
    google_api_key=API_KEY,
    transport="rest"
)
vectordb = Chroma(
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
    collection_name=COLLECTION_NAME
)
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": RETRIEVE_K, "fetch_k": FETCH_K, "lambda_mult": MMR_LAMBDA},
)

# --- Optional BM25 Keyword Retriever ---
bm25 = None
try:
    bm25_path = CHROMA_DIR / "bm25.pkl"
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    print(f"[graph] Loaded BM25 retriever from {bm25_path}")
except Exception:
    print("[graph] BM25 retriever not found or failed to load (optional).")
    pass

# --- LLM ---
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0,
    google_api_key=API_KEY,
    transport="rest",
    # Add safety_settings for production use if needed
)


# --------------------------------------------------------------------------
# 4. PROMPT TEMPLATES
# --------------------------------------------------------------------------

REWRITE_SYS = """Rewrite the user's question into a concise, standalone search query for retrieval in the domain: {domain}.
- Expand acronyms (e.g., IAM, KMS, CAF) when helpful.
- Add likely synonyms and key entities from the wording.
- Keep it one line; DO NOT answer the question; return only the rewritten query text."""

REWRITE_USER = """Chat history (last turns may help disambiguate):
{history}

Original question:
{question}
"""
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", REWRITE_SYS),
    ("human", REWRITE_USER),
])

RERANK_SYS = """Score each passage from 0–10 for how directly and completely it can answer the user's question.
Prefer passages that:
- Contain explicit lists, enumerations, definitions, or step-by-step guidance relevant to the question.
- Use exact AWS terms mentioned (e.g., IAM, KMS, Access Analyzer, CAF Security capabilities).

Return JSON ONLY:
{{ "top_indices": [ <indices> ] }}
Select the best {top_k} indices from the list."""

RERANK_USER = """Question: {question}
Passages:
{passages}
top_k={top_k}"""

rerank_prompt = ChatPromptTemplate.from_messages([
    ("system", RERANK_SYS),
    ("human", RERANK_USER)
])

SYSTEM = """You are an expert assistant in the domain: {domain}.

You must answer **strictly from the provided Document Context**. Do not use outside knowledge.
**After each sentence or claim you make, you MUST cite the index of the source(s) that support it in square brackets, like [1] or [2][3].**
The citation must come immediately after the statement it supports. Do not add a "Sources" section at the end.
- If the user asks for N items, list **exactly** N items and stop. n_items={n_items}
If the context is insufficient, off-topic, or does not contain the requested facts, do **not** guess. Instead, respond with a short out-of-scope notice.

# Output format:
- Your answer with in-line citations. Example: The sky is blue [1]. Clouds are white [2].
- On a new line, add your confidence level. Example: Confidence: High

# Confidence rubric:
- High   — multiple passages directly and explicitly answer the question.
- Medium — partially supported; the answer is indirect or from a single short mention.
- Low    — context is sparse, generic, or you are providing an out-of-scope notice.
"""

USER = """Question:
{question}

Document Context:
{context}
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", USER),
])


# --------------------------------------------------------------------------
# 5. HELPER FUNCTIONS
# --------------------------------------------------------------------------
def requested_count(question: str) -> int | None:
    m = re.search(r"\b(\d+)\b", question)
    if m:
        return int(m.group(1))
    for w, n in NUM_WORDS.items():
        if re.search(rf"\b{w}\b", question, re.I):
            return n
    return None
    
def detect_topic(q: str) -> str | None:
    """Detects a topic from the question using keywords."""
    ql = q.lower()
    for topic, keys in TOPIC_KEYWORDS.items():
        if any(k in ql for k in keys):
            return topic
    return None

def _doc_key(d: Any) -> tuple:
    """Creates a simple, unique key for a document to aid deduplication."""
    return (d.metadata.get("source", ""), d.metadata.get("page", ""), (d.page_content or "")[:80])

def _format_history(hist: List[Dict[str, str]], max_turns: int = 6) -> str:
    """Formats chat history for inclusion in a prompt."""
    if not hist:
        return "(none)"
    lines = []
    for m in hist[-max_turns:]:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

# --- Low-Confidence Guardrail Helpers ---

def _tokens(s: str) -> list[str]:
    """Tokenizes a string for keyword analysis, removing stopwords."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\- ]+", " ", s)
    return [t for t in s.split() if t and t not in STOPWORDS]

def _support_ratio(context: str, q_tokens: list[str]) -> float:
    """Calculates a simple ratio of question tokens found in the context."""
    if not context or not q_tokens:
        return 0.0
    ctx_lower = " " + context.lower() + " "
    hits = sum(1 for t in set(q_tokens) if f" {t} " in ctx_lower)
    return hits / max(len(set(q_tokens)), 1)

def _generic_expansion(question: str) -> str:
    """Adds neutral, helpful keywords to a query to broaden the search."""
    ql = question.lower()
    extras = {
        "definition","overview","best practices","recommendations","capabilities",
        "list","enumeration","steps","guidance","principles","controls","architecture"
    }
    if any(w in ql for w in ["how ", "steps", "procedure", "configure", "implement"]):
        extras.update({"procedure","step-by-step","implementation"})
    if any(w in ql for w in ["which","what","list","enumerate"]):
        extras.update({"bulleted list"})
    return " ".join(sorted(extras))

def _broaden_once(question: str) -> list:
    """Performs a single, broad retrieval using relaxed MMR and BM25."""
    docs = []
    try:
        docs = vectordb.max_marginal_relevance_search(
            question, k=max(RETRY_K, 10), fetch_k=max(RETRY_K, 10) * 3, lambda_mult=0.2
        )
    except Exception as e:
        print(f"Broadened vector search failed: {e}")
    if bm25:
        try:
            docs.extend(bm25.get_relevant_documents(question)[:max(RETRY_K, 10)])
        except Exception as e:
            print(f"Broadened BM25 search failed: {e}")
    return docs

def _dedupe_merge(base_sources: list[dict], new_docs: list) -> tuple[str, list[dict]]:
    """Merge new docs into context while avoiding duplicates by (source,page)."""
    seen = {(s.get("source"), s.get("page")) for s in (base_sources or [])}
    extra_ctx, extra_srcs = [], []
    start_index = len(base_sources) + 1

    for d in new_docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", d.metadata.get("source_id", "n/a"))
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        extra_ctx.append(d.page_content)
        extra_srcs.append({
            "index": start_index + len(extra_srcs),
            "source": src,
            "page": page,
            "text": d.page_content
        })

    return ("\n\n".join(extra_ctx), extra_srcs)


# --------------------------------------------------------------------------
# 6. GRAPH NODES
# --------------------------------------------------------------------------

def rewrite_node(state: GraphState) -> GraphState:
    """
    Rewrites the user's question into a better search query using the LLM.
    Reads: `question`, `chat_history`.
    Writes: `search_query`.
    """
    msgs = rewrite_prompt.format_messages(
        domain=DOMAIN,
        history=_format_history(state.get("chat_history", [])),
        question=state["question"],
    )
    try:
        resp = llm.invoke(msgs)
        rewritten = (resp.content or "").strip()
        # Fallback to original if the LLM returns something odd/empty
        if not rewritten or len(rewritten) < 3:
            rewritten = state["question"]
    except Exception:
        rewritten = state["question"]
    return {"search_query": rewritten}

def retrieve_node(state: GraphState) -> GraphState:
    """
    Retrieves documents using hybrid search (vector + keyword) and deduplicates.
    Reads: `search_query` (or `question`), `prior_docs`.
    Writes: `raw_docs`.
    """
    q = state.get("search_query") or state["question"]
    topic = detect_topic(q)
    chroma_filter = {"topic": topic} if topic else None

    try:
        v_hits = vectordb.max_marginal_relevance_search(
            q, k=RETRIEVE_K, fetch_k=FETCH_K, lambda_mult=MMR_LAMBDA, filter=chroma_filter
        )
    except Exception:
        # Fallback if filter not supported or old LC version
        v_hits = retriever.invoke(q)

    k_hits = bm25.get_relevant_documents(q) if bm25 else []
    mem_hits = state.get("prior_docs", []) # Carry-over from conversation

    # Merge and deduplicate results
    merged = v_hits + k_hits + mem_hits
    seen, docs = set(), []
    for d in merged:
        key = _doc_key(d)
        if key not in seen:
            seen.add(key)
            docs.append(d)
        if len(docs) >= 16: # Cap total raw docs
            break

    return {"raw_docs": docs}

def rerank_node(state: GraphState) -> GraphState:
    """
    Re-ranks retrieved documents for relevance using the LLM.
    Reads: `question`, `raw_docs`.
    Writes: `context`, `sources`, `kept_docs`.
    """
    docs = state.get("raw_docs", [])
    if not docs:
        return {"context": "", "sources": [], "kept_docs": []}

    top_k = min(4, len(docs))
    passages = "\n\n".join(f"[{i}] {d.page_content[:1200]}" for i, d in enumerate(docs))
    msgs = rerank_prompt.format_messages(question=state["question"], passages=passages, top_k=top_k)
    kept = []
    try:
        resp = llm.invoke(msgs)
        text = resp.content.strip()
        # The LLM may wrap the JSON in markdown, so we extract it
        match = re.search(r"\{.*\}", text, flags=re.S)
        js = json.loads(match.group(0) if match else text)
        keep_indices = set(js.get("top_indices", []))
        kept = [d for i, d in enumerate(docs) if i in keep_indices] or docs[:top_k]
    except Exception as e:
        print(f"Re-ranker failed to parse LLM response, falling back to top_k. Error: {e}")
        kept = docs[:top_k]

    ctx_parts, sources = [], []
    for i, d in enumerate(kept, 1):
        sources.append({"index": i, "source": d.metadata.get("source"), "page": d.metadata.get("page"),"text": d.page_content})
        ctx_parts.append(f"[{i}] {d.page_content}")

    return {"context": "\n\n".join(ctx_parts), "sources": sources, "kept_docs": kept}

def generate_node(state: GraphState) -> GraphState:
    """
    Generates a final answer, applying guardrails for low-confidence context.
    Reads: `question`, `context`, `sources`, `search_query`.
    Writes: `answer`, `sources`.
    """
    question = state["question"]
    ctx = state.get("context") or ""
    sources = state.get("sources", [])
    q_tokens = _tokens(question)
    support = _support_ratio(ctx, q_tokens)

    # Guardrail 1: If context is too short or has low token overlap, try a broader search
    q_tokens = _tokens(question)
    support = _support_ratio(ctx, q_tokens)
    if len(ctx) < MIN_CONTEXT_CHARS or support < WEAK_SUPPORT:
        expanded_q = f"{state.get('search_query') or question} {_generic_expansion(question)}"
        extra_docs = _broaden_once(expanded_q)
        extra_ctx, extra_srcs = _dedupe_merge(sources, extra_docs)

        if extra_ctx:
            ctx = (ctx + "\n\n" if ctx else "") + extra_ctx
            sources.extend(extra_srcs)
            support = _support_ratio(ctx, q_tokens) # Recalculate support

    # Guardrail 2: If context is still weak, do not guess. Ask for clarification.
    if len(ctx) < MIN_CONTEXT_CHARS or support < WEAK_SUPPORT:
        clarify_msg = (
            "Answer:\nI don’t have enough grounded evidence in the documents to answer confidently.\n\n"
            "To search better, could you narrow one of these:\n"
            "- Service / area (e.g., IAM, S3, KMS, networking, incident response)\n"
            "- Scope (organization / account / region / workload)\n"
            "- Artifact type (definition, best practices, steps, capabilities, architecture)\n"
            "- A keyword you expect to see\n\n"
            "Confidence: Low"
        )
        return {"answer": clarify_msg, "sources": sources}

    # If context is sufficient, generate the answer
    n_items = requested_count(question)
    msgs = prompt.format_messages(domain=DOMAIN, question=question, context=ctx,n_items=n_items)
    resp = llm.invoke(msgs)
    answer = resp.content

    return {"answer": answer, "sources": sources}


# --------------------------------------------------------------------------
# 7. PUBLIC API & ORCHESTRATION
# --------------------------------------------------------------------------

def prepare_context(question: str, chat_history: List[Dict[str, str]] | None = None, prior_docs: List[Any] | None = None):
    """
    Runs the retrieval pipeline to prepare context for streaming.

    This function executes the rewrite -> retrieve -> rerank sequence to gather
    the necessary context, sources, and kept documents before the final
    generation step, which can then be streamed.

    Args:
        question: The user's question.
        chat_history: The history of the conversation.
        prior_docs: Documents from a previous turn to maintain context.

    Returns:
        A GraphState dictionary containing `context`, `sources`, and `kept_docs`.
    """
    st: GraphState = {"question": question}
    if chat_history:
        st["chat_history"] = chat_history
    if prior_docs:
        st["prior_docs"] = prior_docs

    st.update(rewrite_node(st))
    st.update(retrieve_node(st))
    st.update(rerank_node(st))
    return st

def stream_generate(question: str, context: str, domain: str | None = None):
    """
    Yields answer text chunks from the LLM based on the provided context.

    Args:
        question: The user's original question.
        context: The prepared, re-ranked context string.
        domain: The domain of expertise for the system prompt.

    Yields:
        String chunks of the generated answer.
    """
    n_items = requested_count(question) or ""
    msgs = prompt.format_messages(
        domain=domain or DOMAIN,
        question=question,
        context=context,
        n_items=n_items
    )
    for chunk in llm.stream(msgs):
        text = getattr(chunk, "content", "")
        if text:
            yield text

def build_app():
    """
    Builds and compiles the LangGraph RAG application.
    """
    g = StateGraph(GraphState)
    g.add_node("rewrite", rewrite_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("generate", generate_node)

    g.set_entry_point("rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", END)

    return g.compile()