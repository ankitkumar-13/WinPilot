import logging
from typing import Any

from app.mcp_client import WinPilotMCPClient
from core.command import Command
from core.command_parser import CommandParser
from llm.router import LLMRouter
from nlp.adapter import LocalNLPInterpreter

logger = logging.getLogger(__name__)


class WinPilotOrchestrator:
    """Coordinate request routing and MCP tool execution."""

    def __init__(
        self,
        client: WinPilotMCPClient,
        router: LLMRouter,
    ) -> None:
        """Initialise the orchestrator and its dependencies.

        Args:
            client - MCP client used to discover and execute tools.
            router - LLM router used to process user requests.

        Returns:
            None.
        """
        self.client = client
        self.router = router
        self.parser = CommandParser()
        self.local_nlp = LocalNLPInterpreter()
        self._tools: list[Any] = []
        self._tool_names: set[str] = set()
        self._initialised = False

    async def initialise(self) -> None:
        """Discover available MCP tools and cache their schemas.

        Raises:
            Exception - If tool discovery fails.

        Returns:
            None.
        """
        try:
            tools = await self.client.list_tools()
            tool_names = {tool.name for tool in tools}

            if len(tool_names) != len(tools):
                raise ValueError("Duplicate MCP tool names detected.")

            self._tools = tools
            self._tool_names = tool_names
            self._initialised = True

            logger.info(
                "Discovered %d MCP tools: %s",
                len(self._tools),
                sorted(self._tool_names),
            )

        except Exception:
            logger.exception("Failed to initialise MCP tools")
            raise

    @property
    def tool_count(self) -> int:
        """Return the number of cached tools.

        Returns:
            Number of available tools.
        """
        return len(self._tools)

    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        """Convert cached tools into the LLM tool format.

        Returns:
            Tool definitions containing names, descriptions and schemas.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.input_schema,
            }
            for tool in self._tools
        ]

    @staticmethod
    def _format_tool_result(name: str, result: Any) -> str:
        """Format an MCP tool result while retaining its domain status.

        Args:
            name - Name of the MCP tool that ran.
            result - Result object returned by the MCP client.

        Returns:
            User-facing text describing the result.
        """
        content = getattr(result, "content", None) or []
        output = "\n".join(
            item.text
            for item in content
            if getattr(item, "type", None) == "text"
            and isinstance(getattr(item, "text", None), str)
        )
        structured = getattr(result, "structured_content", None)
        if not isinstance(structured, dict):
            structured = getattr(result, "structuredContent", None)

        status = structured.get("status") if isinstance(structured, dict) else None
        message = structured.get("message") if isinstance(structured, dict) else None

        if getattr(result, "is_error", False):
            detail = message if isinstance(message, str) else output
            logger.error("MCP tool %s returned an error: %s", name, detail)
            if isinstance(status, str) and status:
                return f"Tool execution failed [{status}]: {detail or 'Unknown error'}"
            return f"Tool execution failed: {detail or 'Unknown error'}"

        logger.info("MCP tool completed: %s", name)
        logger.debug("Tool result: %s", output)
        return output or "The tool completed without a text result."

    async def process(self, query: str) -> str:
        """Process a request and execute at most one MCP tool.

        Args:
            query - Natural-language request from the user.

        Returns:
            User-facing response describing the result or failure.
        """
        if not isinstance(query, str) or not query.strip():
            return "Please enter a valid command."

        query = query.strip()
        logger.info("Processing user request")

        if not self._initialised:
            try:
                await self.initialise()
            except Exception:
                return "Failed to retrieve MCP tools. Check the logs."

        command = self.parser.parse_command(query)
        if command.action == "unknown":
            try:
                command = self.local_nlp.interpret(query) or command
            except Exception:
                logger.exception("Local NLP interpretation failed")

        if command.action != "unknown":
            response = self._command_response(command)
            if response is None:
                logger.warning("Parsed command could not be mapped to an MCP tool")
                response = None
        else:
            response = None

        if response is None:
            try:
                response = await self.router.process_query(
                    query,
                    self._get_tool_schemas(),
                )
            except Exception:
                logger.exception("LLM failed to process request")
                return "Failed to process the request. Check the logs."

        if not isinstance(response, dict):
            logger.error(
                "Unexpected response type: %s",
                type(response).__name__,
            )
            return "The model returned an invalid response."

        response_type = response.get("type")

        if response_type == "text":
            text = response.get("text", "")

            if not isinstance(text, str):
                logger.error("Invalid text response from model")
                return "The model returned an invalid response."

            return text

        if response_type != "tool_call":
            logger.warning(
                "Unsupported response type: %r",
                response_type,
            )
            return "The model returned an unsupported response."

        name = response.get("tool_name")
        args = response.get("arguments")

        if not isinstance(name, str) or name not in self._tool_names:
            logger.error("Model requested an unknown tool: %r", name)
            return "The model requested an unavailable tool."

        if not isinstance(args, dict):
            logger.error("Invalid arguments for tool %s", name)
            return "The model returned invalid tool arguments."

        logger.info("Executing MCP tool: %s", name)
        logger.debug("Tool arguments: %r", args)

        try:
            result = await self.client.call_tool(name, args)
        except Exception:
            logger.exception("MCP call failed for tool %s", name)
            return f"Tool '{name}' failed. Check the logs."

        return self._format_tool_result(name, result)

    def _command_response(self, command: Command) -> dict[str, Any] | None:
        """Map an interpreted command to its discovered MCP tool.

        Args:
            command - Locally interpreted command to map.

        Returns:
            Tool-call response, or None when no matching tool is available.
        """
        action = command.action
        arguments = command.arguments

        if action == "open_application":
            name, args = "open_application", {"application": arguments.get("target")}
        elif action in {"create_file", "create_folder"}:
            name = action
            args = {"name": arguments.get("target")}
        elif action in {"move", "copy"}:
            name = "move_item" if action == "move" else "copy_item"
            args = {
                "source": arguments.get("source"),
                "destination": arguments.get("destination"),
            }
        elif action in {"delete_file", "delete_folder"}:
            name = "delete_item"
            args = {
                "target": arguments.get("target"),
                "kind": action.removeprefix("delete_"),
            }
        elif action in {"rename_file", "rename_folder"}:
            name = "rename_item"
            args = {
                "path": arguments.get("target"),
                "new_name": arguments.get("new_name"),
                "kind": action.removeprefix("rename_"),
            }
        else:
            return None

        if name not in self._tool_names or any(value is None for value in args.values()):
            return None
        logger.info("Command interpreted locally as MCP tool: %s", name)
        return {"type": "tool_call", "tool_name": name, "arguments": args}
