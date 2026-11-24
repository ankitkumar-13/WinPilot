import logging
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

MCP_SERVERS: list[dict[str, Any]] = [
    {
        "name": "WinPilot",
        "module": "mcp_servers.win_utils_server",
    },
    {
        "name": "WinPilot-LibreOffice",
        "module": "mcp_servers.libreoffice_server",
    },
]


class WinPilotMCPClient:
    """Manage connections to WinPilot MCP servers."""

    def __init__(self) -> None:
        """Initialise the MCP client without opening connections.

        Returns:
            None.
        """
        self._stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}

    async def connect(self) -> None:
        """Connect to all configured WinPilot MCP servers.

        Raises:
            RuntimeError - If the client is already connected.
            Exception - If any connection or initialisation fails.

        Returns:
            None.
        """
        if self._sessions:
            raise RuntimeError("MCP client is already connected.")

        stack = AsyncExitStack()
        await stack.__aenter__()

        try:
            for server in MCP_SERVERS:
                name = server["name"]
                module = server["module"]

                params = StdioServerParameters(
                    command=sys.executable,
                    args=["-m", module],
                    cwd=str(ROOT),
                )

                read, write = await stack.enter_async_context(
                    stdio_client(params)
                )

                session = await stack.enter_async_context(
                    ClientSession(read, write)
                )

                await session.initialize()
                self._sessions[name] = session

                logger.info("Connected to %s MCP server", name)

            self._stack = stack

        except Exception:
            logger.exception("Failed to connect to MCP servers")
            await stack.aclose()
            self._sessions.clear()
            raise

    @property
    def session(self) -> ClientSession | None:
        """Return the primary MCP session for backward compatibility.

        Returns:
            Primary session, or None when not connected.
        """
        if not self._sessions:
            return None
        return next(iter(self._sessions.values()))

    async def list_tools(self) -> list[Any]:
        """Retrieve the tools exposed by all MCP servers.

        Returns:
            Combined list of available MCP tool definitions.

        Raises:
            RuntimeError - If the client is not connected.
        """
        if not self._sessions:
            raise RuntimeError("MCP client is not connected.")

        all_tools: list[Any] = []

        for name, session in self._sessions.items():
            try:
                result = await session.list_tools()
                all_tools.extend(result.tools)
                logger.debug(
                    "Discovered %d tools from %s",
                    len(result.tools),
                    name,
                )
            except Exception:
                logger.warning(
                    "Failed to list tools from %s",
                    name,
                    exc_info=True,
                )

        return all_tools

    async def call_tool(
        self,
        name: str,
        args: dict[str, Any],
    ) -> Any:
        """Execute a tool by searching all connected MCP servers.

        Args:
            name - Name of the MCP tool to execute.
            args - Arguments supplied to the tool.

        Returns:
            MCP tool execution result.

        Raises:
            RuntimeError - If the client is not connected.
            Exception - If the MCP tool call fails.
        """
        if not self._sessions:
            raise RuntimeError("MCP client is not connected.")

        for server_name, session in self._sessions.items():
            try:
                result = await session.list_tools()
                tool_names = {tool.name for tool in result.tools}

                if name in tool_names:
                    return await session.call_tool(name, args)

            except Exception:
                logger.debug(
                    "Could not check tools on %s",
                    server_name,
                    exc_info=True,
                )

        raise ValueError(f"Tool '{name}' not found on any MCP server.")

    async def close(self) -> None:
        """Close all MCP connections and release their resources.

        Returns:
            None.
        """
        try:
            await self._stack.aclose()
        finally:
            self._sessions.clear()
            self._stack = AsyncExitStack()

        logger.info("MCP client closed")
