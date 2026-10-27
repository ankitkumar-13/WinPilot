import asyncio
import sys

from pathlib import Path
from contextlib import AsyncExitStack


from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent

class WinPilotMCPClient:
    def __init__(self) -> None:
        self.stack = AsyncExitStack()
        self.session: ClientSession | None =  None

    async def connect(self):
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_servers.win_utils_server"],
            cwd=str(ROOT)
        )

        read, write = await self.stack.enter_async_context(
            stdio_client(params)
        )

        self.session = await self.stack.enter_async_context(
            ClientSession(read, write)
        )

        await self.session.initialize()

    async def list_tools(self):
        if self.session is None:
            raise RuntimeError("MCP client is not connected.")
        
        result = await self.session.list_tools()
        return result.tools

    async def call_tool(self, name: str, args: dict):
        if self.session is None:
            raise RuntimeError("MCP client is not connected.")
        
        return await self.session.call_tool(name, args)

    async def close(self):
        await self.stack.aclose()

async def main():
    client = WinPilotMCPClient()

    try:
        await client.connect()

        tools = await client.list_tools()
        print("Connected to WinPilot MCP Server")

        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
