import argparse
import asyncio
import logging
import sys

from app.mcp_client import WinPilotMCPClient
from app.orchestrator import WinPilotOrchestrator
from core.logging_config import setup_logging
from llm.router import LLMRouter

logger = logging.getLogger(__name__)


async def run(provider: str) -> int:
    """Initialise and run the WinPilot application.

    Args:
        provider - LLM provider to use for processing requests.

    Returns:
        Zero if the application exits normally, otherwise one.
    """
    client = WinPilotMCPClient()
    connected = False

    logger.info("Starting WinPilot")

    try:
        await client.connect()
        connected = True
        logger.info("Connected to MCP server")

        router = LLMRouter(provider=provider)
        orchestrator = WinPilotOrchestrator(client, router)

        await orchestrator.initialise()

        print("WinPilot - Windows Desktop Automation")
        print("Connected to MCP server.")
        print(f"Available tools: {orchestrator.tool_count}")
        print("Type 'exit' or 'quit' to close.")

        while True:
            try:
                query = await asyncio.to_thread(
                    input, "\nWinPilot > "
                )
            except (EOFError, KeyboardInterrupt):
                break

            query = query.strip()

            if query.lower() in {"exit", "quit"}:
                break

            if not query:
                continue

            try:
                result = await orchestrator.process(query)
                print(f"WinPilot: {result}")
            except Exception:
                logger.exception("Request processing failed")
                print("Request failed. Check the application logs.")

        return 0

    except Exception:
        logger.exception("WinPilot startup or runtime failure")
        print("WinPilot could not start. Check the application logs.")
        return 1

    finally:
        if connected:
            try:
                await client.close()
                logger.info("MCP connection closed")
            except Exception:
                logger.exception("Failed to close MCP connection")

        logger.info("WinPilot stopped")


def main() -> int:
    """Parse command-line arguments and start WinPilot.

    Returns:
        The application exit code.
    """
    parser = argparse.ArgumentParser(
        description="WinPilot Windows Desktop Automation"
    )
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini"],
        help="LLM provider to use",
    )

    args = parser.parse_args()

    setup_logging()
    logger.info("Selected LLM provider: %s", args.provider)

    try:
        return asyncio.run(run(args.provider))
    except KeyboardInterrupt:
        logger.info("WinPilot interrupted by user")
        print("\nWinPilot stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
