import logging
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


class WinPilotMCPClient:
    """Manage connections to the WinPilot MCP server."""

    def __init__(self) -> None:
        """Initialise the MCP client without opening a connection."""
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None

    async def connect(self) -> None:
        """Connect to the local WinPilot MCP server.

        Raises:
            RuntimeError - If the client is already connected.
            Exception - If the connection or initialisation fails.
        """
        if self.session is not None:
            raise RuntimeError("MCP client is already connected.")

        stack = AsyncExitStack()
        await stack.__aenter__()

        try:
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "mcp_servers.win_utils_server"],
                cwd=str(ROOT),
            )

            read, write = await stack.enter_async_context(
                stdio_client(params)
            )

            session = await stack.enter_async_context(
                ClientSession(read, write)
            )

            await session.initialize()

            self._stack = stack
            self.session = session

            logger.info("Connected to WinPilot MCP server")

        except Exception:
            logger.exception("Failed to connect to MCP server")
            await stack.aclose()
            raise

    async def list_tools(self) -> list[Any]:
        """Retrieve the tools exposed by the MCP server.

        Returns:
            List of available MCP tool definitions.

        Raises:
            RuntimeError - If the client is not connected.
        """
        if self.session is None:
            raise RuntimeError("MCP client is not connected.")

        result = await self.session.list_tools()
        return result.tools

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any],
    ) -> Any:
        """Execute a tool exposed by the MCP server.

        Args:
            name - Name of the MCP tool to execute.
            args - Arguments supplied to the tool.

        Returns:
            MCP tool execution result.

        Raises:
            RuntimeError - If the client is not connected.
            Exception - If the MCP tool call fails.
        """
        if self.session is None:
            raise RuntimeError("MCP client is not connected.")

        return await self.session.call_tool(name, args)

    async def close(self) -> None:
        """Close the MCP connection and release its resources."""
        try:
            await self._stack.aclose()
        finally:
            self.session = None
            self._stack = AsyncExitStack()

        logger.info("MCP client closed")
