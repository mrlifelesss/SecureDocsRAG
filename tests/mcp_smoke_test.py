import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_ENV = {"AWS_DOCUMENTATION_PARTITION": "aws", "FASTMCP_LOG_LEVEL": "ERROR"}

async def main():
    server = StdioServerParameters(
        command="uv",
        args=[
            "tool","run","--from",
            "awslabs.aws-documentation-mcp-server@latest",
            "awslabs.aws-documentation-mcp-server.exe",
        ],
        env=SERVER_ENV,
        cwd=None,
    )
    async with stdio_client(server) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("TOOLS:", [t.name for t in tools.tools])
            res = await s.call_tool("search_documentation",
                                    {"search_phrase": "KMS multi-Region keys", "limit": 2})
            print("SEARCH TYPE:", type(res.structuredContent).__name__)
            print("SEARCH PAYLOAD:", res.structuredContent)

if __name__ == "__main__":
    asyncio.run(main())
