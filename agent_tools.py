# agent_tools.py
from langchain_core.tools import tool
from typing import Optional,List, Dict, Any
from graph import (
    prepare_context,
    DOMAIN,
    MIN_CONTEXT_CHARS,
    prompt,            # ChatPromptTemplate (needs domain, question, context, n_items)
    llm,               # your chat model
    requested_count,   # extracts a requested number from the question
)
import requests 
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import RequestException, Timeout, HTTPError
from settings import settings as cfg

from mcp_client_docs import search_docs, read_doc

# ---- shared HTTP session with hygiene ----
_session = requests.Session()
_session.headers.update({"User-Agent": "aws-sec-rag/1.0"})
_session.mount(
    "http://",
    HTTPAdapter(max_retries=Retry(
        total=2, connect=2, read=2, backoff_factor=0.3,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset({"GET"})
    ))
)
_session.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(
        total=2, connect=2, read=2, backoff_factor=0.3,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset({"GET"})
    ))
)
_TIMEOUT = (3.05, 15)  # (connect, read) seconds

def _validate_search(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict) or "results" not in payload or not isinstance(payload["results"], list):
        return []
    # normalize to [{"id": "..."} ...]
    out = []
    for item in payload["results"]:
        if isinstance(item, dict) and "id" in item:
            out.append(item)
    return out

def _validate_read(payload: Dict[str, Any]) -> Dict[str, Any]:
    # expect {"content": "...", "metadata": {"title": "...", "url": "..."}}
    if not isinstance(payload, dict):
        return {}
    if "content" not in payload or not isinstance(payload["content"], str):
        return {}
    meta = payload.get("metadata") or {}
    title = meta.get("title") or "AWS Documentation"
    url = meta.get("url") or ""
    return {"content": payload["content"], "title": title, "url": url}

@tool
def rag_answer(question: str, n_items: Optional[int] = None) -> str:
    """Grounded KB answer. Returns text+Sources+Confidence, or NO_CONTEXT/ERROR. Never raises."""
    # 1) Decide how many items to ask for
    try:
        inferred = requested_count(question)
        top_n = int(n_items) if n_items is not None else (inferred if inferred else 3)
        if top_n <= 0:
            top_n = 3
    except Exception:
        top_n = 3

    # 2) Retrieve context
    try:
        ctx = prepare_context(question)
    except Exception as e:
        return f"ERROR: {e}"

    context = (ctx.get("context") or "").strip()
    sources = ctx.get("sources") or []
    avg     = float(ctx.get("avg_score") or 0.0)

    if len(context) < MIN_CONTEXT_CHARS:
        return ("NO_CONTEXT: not enough grounded evidence. "
                "Please narrow the scope (service/scope/keyword).")

    # 3) PRE-BIND required vars so n_items/domain can’t be missing
    try:
        bound = prompt.partial(
            domain=DOMAIN,
            n_items=str(top_n),   # important: bind it here
        )
        msgs = bound.format_messages(
            question=question,
            context=context,
        )
        ai = llm.invoke(msgs)
        text = getattr(ai, "content", ai) or ""
    except Exception as e:
        return f"ERROR: {e}"

    # 4) Confidence + Sources
    conf = "High" if avg >= 0.75 else "Medium" if avg >= 0.5 else "Low"

    def _fmt(s, i):
        if isinstance(s, dict):
            name = s.get("source") or s.get("path") or s.get("id") or f"doc{i+1}"
            page = s.get("page")
            return f"[{i+1}] {name}" + (f":{page}" if page is not None else "")
        return f"[{i+1}] {str(s)}"

    src = "; ".join(_fmt(s, i) for i, s in enumerate(sources[:top_n])) or "local KB"
    return f"{text}\n\nSources: {src}\nConfidence: {conf}"

# Optional: lightweight utility tools
@tool
def summarize(text: str) -> str:
    """Summarize long text into 3–5 bullets."""
    return "\n".join("- " + line.strip() for line in text.splitlines() if line.strip())[:1500]

@tool
def calculator(expr: str) -> str:
    """Evaluate a simple arithmetic expression, e.g. '2*(3+5)'."""
    try:
        return str(eval(expr, {"__builtins__": {}}))
    except Exception as e:
        return f"ERROR: {e}"
@tool
def aws_docs_search(question: str, n_items: Optional[int] = None) -> str:
    """Search & read AWS docs via the official AWS Documentation MCP Server."""
    limit = max(1, int(n_items)) if n_items else 3
    results = search_docs(question, limit)
    if not results:
        return "NO_CONTEXT: No matching AWS documentation found."

    contexts, sources = [], []
    for i, r in enumerate(results, start=1):
        url = r.get("url") or r.get("id")
        if not url:
            continue
        page = read_doc(url)
        if page.get("content_markdown"):
            contexts.append(page["content_markdown"])
            title = r.get("title") or "AWS Documentation"
            sources.append(f"[{i}] {title} ({url})")

    if not contexts:
        return "NO_CONTEXT: Failed to read content from the AWS docs results."
    return f"Context:\n\n" + "\n\n---\n\n".join(contexts) + "\n\nSources:\n" + "\n".join(sources)