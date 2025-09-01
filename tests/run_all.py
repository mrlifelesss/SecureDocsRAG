# tests/run_all.py
import os, sys, shutil, asyncio, traceback

# Ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS = []

def record(name, ok, err=None):
    RESULTS.append((name, ok, err))
    status = "PASS" if ok else ("SKIP" if err == "SKIP" else "FAIL")
    print(f"[{status}] {name}")
    if err not in (None, "SKIP"):
        print(err, end="\n\n")

def need_tools():
    try:
        import mcp  # noqa
    except Exception:
        record("env:mcp", False, "mcp SDK not installed (pip install \"mcp[cli]\")")
        return False
    if shutil.which("uv") is None:
        record("env:uv", False, "uv not on PATH (winget install --id Astral-Sh.uv)")
        return False
    return True

def test_smoke():
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        async def run():
            server = StdioServerParameters(
                command="uv",
                args=["tool","run","--from",
                      "awslabs.aws-documentation-mcp-server@latest",
                      "awslabs.aws-documentation-mcp-server.exe"],
                env={"AWS_DOCUMENTATION_PARTITION":"aws","FASTMCP_LOG_LEVEL":"ERROR"},
                cwd=None,
            )
            async with stdio_client(server) as (r, w):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    tools = await s.list_tools()
                    names = [t.name for t in tools.tools]
                    assert "search_documentation" in names
                    res = await s.call_tool("search_documentation",
                                            {"search_phrase":"KMS multi-Region keys","limit":1})
                    assert isinstance(res.structuredContent, (dict, list))
        asyncio.run(run())
        record("smoke: mcp server reachable", True)
    except Exception:
        record("smoke: mcp server reachable", False, traceback.format_exc())

def test_wrapper():
    try:
        from mcp_client_docs import search_docs, read_doc
        hits = search_docs("S3 default encryption KMS", 2)
        assert hits and isinstance(hits, list)
        url = hits[0].get("url") or hits[0].get("id")
        page = read_doc(url)
        assert page.get("content_markdown")
        record("wrapper: search+read", True)
    except Exception:
        record("wrapper: search+read", False, traceback.format_exc())

def test_agent_tool():
    try:
        from agent_tools import aws_docs_search
    except Exception as e:
        record("agent_tool import", False, f"Cannot import agent_tools: {e}")
        return
    try:
        out = aws_docs_search.invoke({"question":
            "Enable S3 Bucket Keys with a KMS key; give steps.", "n_items": 2})
        assert isinstance(out, str) and "Context:" in out and "\nSources:\n" in out
        record("agent tool: aws_docs_search", True)
    except Exception:
        record("agent tool: aws_docs_search", False, traceback.format_exc())

if __name__ == "__main__":
    print("== MCP end-to-end checks ==\n")
    if need_tools():
        test_smoke()
        test_wrapper()
        test_agent_tool()
    else:
        record("all", False, "SKIP")
    print("\n== Summary ==")
    passed = sum(1 for _, ok, err in RESULTS if ok)
    failed = sum(1 for _, ok, err in RESULTS if not ok and err != "SKIP")
    skipped = sum(1 for _, ok, err in RESULTS if err == "SKIP")
    for name, ok, err in RESULTS:
        status = "PASS" if ok else ("SKIP" if err == "SKIP" else "FAIL")
        print(f"- {status} {name}")
    print(f"\nTotals: PASS {passed}  FAIL {failed}  SKIP {skipped}")
    sys.exit(1 if failed else 0)
