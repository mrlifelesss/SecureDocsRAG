# mcp_client_docs.py
import asyncio
from typing import Any, Dict, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_ENV = {
    "AWS_DOCUMENTATION_PARTITION": "aws",
    "FASTMCP_LOG_LEVEL": "ERROR",
}
_TOOL_TIMEOUT = 20.0  # seconds

def _server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command="uv",
        args=[
            "tool", "run", "--from",
            "awslabs.aws-documentation-mcp-server@latest",
            "awslabs.aws-documentation-mcp-server.exe",
        ],
        env=SERVER_ENV,
        cwd=None,
    )

async def _tool_call(tool: str, args: Dict[str, Any]) -> Any:
    # Open a NEW stdio session per call; close it cleanly to avoid cancel-scope errors
    async with stdio_client(_server_params()) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            res = await asyncio.wait_for(s.call_tool(tool, args), timeout=_TOOL_TIMEOUT)
            return res.structuredContent

def _as_list(payload: Any) -> List[Any]:
    if isinstance(payload, dict):
        for k in ("result", "results", "items"):
            v = payload.get(k)
            if isinstance(v, list):
                return v
            if isinstance(v, dict):
                return [v]
        return []
    return payload if isinstance(payload, list) else []

def _as_markdown(payload: Any) -> str:
    obj = payload
    if isinstance(obj, dict) and "result" in obj:
        obj = obj["result"]
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for k in ("markdown", "text", "content"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v
        parts = [v for v in obj.values() if isinstance(v, str) and v.strip()]
        return "\n".join(parts)
    if isinstance(obj, list):
        return "\n".join(s if isinstance(s, str) else str(s) for s in obj)
    return str(obj or "")

def search_docs(question: str, limit: int = 3) -> List[Dict[str, Any]]:
    async def run():
        data = await _tool_call("search_documentation",
                                {"search_phrase": question, "limit": int(limit)})
        hits_raw = _as_list(data)
        out: List[Dict[str, Any]] = []
        for h in hits_raw:
            if not isinstance(h, dict):
                continue
            url = h.get("url") or h.get("id")
            title = h.get("title") or "AWS Documentation"
            ctx = h.get("context") or ""
            if url:
                out.append({"url": url, "title": title, "context": ctx})
        return out
    return asyncio.run(run())

def read_doc(url: str) -> Dict[str, Any]:
    async def run():
        data = await _tool_call("read_documentation", {"url": url})
        md = _as_markdown(data)
        return {"content_markdown": (md or ""), "url": url}
    return asyncio.run(run())

def recommend(url: str, limit: int = 5) -> List[Dict[str, Any]]:
    async def run():
        data = await _tool_call("recommend", {"url": url})
        items = _as_list(data)
        out: List[Dict[str, Any]] = []
        for it in items[:limit]:
            if isinstance(it, dict):
                u = it.get("url") or it.get("id")
                t = it.get("title") or "AWS Documentation"
                if u:
                    out.append({"url": u, "title": t})
        return out
    return asyncio.run(run())
