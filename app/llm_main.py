
import asyncio

from app.mcp_client import WinPilotMCPClient
from app.orchestrator import WinPilotOrchestrator
from llm.router import LLMRouter


async def main():
    client = WinPilotMCPClient()

    try:
        await client.connect()
        print("Connected to MCP server.")

        router = LLMRouter(provider="gemini")
        app = WinPilotOrchestrator(client, router)

        tools = await client.list_tools()
        print(f"Available MCP tools: {len(tools)}")

        while True:
            try:
                query = await asyncio.to_thread(
                    input, "\nWinPilot> "
                )
            except (EOFError, KeyboardInterrupt):
                break

            query = query.strip()

            if query.lower() in {"quit", "exit"}:
                break

            if not query:
                continue

            try:
                result = await app.process(query)
                print(f"WinPilot: {result}")
            except Exception as exc:
                print(f"Request failed: {exc}")

    except Exception as exc:
        print(f"Startup failed: {exc}")
    finally:
        await client.close()
        print("\nMCP connection closed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nWinPilot stopped.")
