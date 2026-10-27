import asyncio
import sys
import json

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

async def chat_loop(client):
    print("\nWinPilot MCP Client")
    print("Type 'help' to get some basic commands, 'quit' to exit.")

    while True:
        try:
            command = await asyncio.to_thread(
                input, "\nWinPilot> "
            )
        except (EOFError, KeyboardInterrupt):
            break

        command = command.strip()

        if command.lower() == "quit":
            break

        if command.lower() == "tools":
            for tool in await client.list_tools():
                print(f"- {tool.name}: {tool.description.splitlines()[0]}")
            continue

        if command.lower() == "call":
            name = await asyncio.to_thread(input, "Tool name: ")
            name = name.strip()

            raw_args = await asyncio.to_thread(input, "Arguments (JSON): ")
            raw_args = raw_args.strip()

            try:
                args = json.loads(raw_args)
                result = await client.call_tool(name, args)

                for item in result.content:
                    if hasattr(item, "text"):
                        print(item.text)

                if result.is_error:
                    print("Tool reported an error.")
            except json.JSONDecodeError:
                print("Invalid JSON. Example:  {\"application\": \"notepad\"}")
            except Exception as exc:
                print(f"Tool call failed: {exc}")
            continue

        if command.lower() == "help":
            print("Commands:")
            print("  tools - List available MCP tools")
            print("  call  - Call an MCP tool manually")
            print("  help  - Show this help")
            print("  quit  - Exit WinPilot")
            continue

        print("Unknown command. Use 'tools', 'call' or 'quit'.")



async def main():
    client = WinPilotMCPClient()

    try:
        await client.connect()

        tools = await client.list_tools()
        print("Connected to WinPilot MCP Server")
        print(f"{len(tools)} tools available.")

        await chat_loop(client)
    except Exception as exc:
        print(f"MCP client error: ", exc)
    finally:
        try:
            await client.close()
        except Exception as exc:
            print(f"Error closing MCP connection: {exc}")
        print("MCP connection closed.")

if __name__ == "__main__":
    asyncio.run(main())
