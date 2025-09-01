# tests/mcp_wrapper_test.py
from mcp_client_docs import search_docs, read_doc

def main():
    results = search_docs("S3 default encryption KMS", limit=2)
    print("RESULTS:", results)
    assert results and isinstance(results, list), "No results from MCP search"
    url = results[0].get("url") or results[0].get("id")
    page = read_doc(url)
    print("PAGE KEYS:", list(page.keys()))
    print("CONTENT HEAD:", (page["content_markdown"] or "")[:300])
    assert page.get("content_markdown"), "Empty page content"
    print("OK")
if __name__ == "__main__":
    main()