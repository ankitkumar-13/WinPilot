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
        self._pending_clarification: dict[str, Any] | None = None
        self.last_route = "Not started"

    def _set_route(self, route: str) -> None:
        """Record and log the route used for the current request."""
        self.last_route = route
        logger.info("Route: %s", route)

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
        schemas = []
        for tool in self._tools:
            input_schema = dict(tool.input_schema)
            properties = dict(input_schema.get("properties", {}))
            properties.pop("clarified_path", None)
            properties.pop("clarified_field", None)
            input_schema["properties"] = properties
            input_schema["required"] = [
                name
                for name in input_schema.get("required", [])
                if name not in {"clarified_path", "clarified_field"}
            ]
            schemas.append({
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": input_schema,
            })
        return schemas

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
            if status == "clarification_required":
                return detail or "Please clarify the target location and try again."
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
            self._set_route("Input validation")
            return "Please enter a valid command."

        query = query.strip()
        logger.info("Processing user request")
        self._set_route("Classifying")

        if self._pending_clarification is not None:
            self._set_route(
                f"Clarification → MCP: {self._pending_clarification['tool_name']}"
            )
            return await self._resolve_clarification(query)

        if not self._initialised:
            try:
                await self.initialise()
            except Exception:
                self._set_route("MCP tool discovery failed")
                return "Failed to retrieve MCP tools. Check the logs."

        command = self.parser.parse_command(query)
        source = "RegEx"
        if command.action == "unknown":
            source = "Local NLP"
            self._set_route("Local NLP: loading models / interpreting")
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
            source = "Gemini"
            logger.info("Local routes did not resolve the request; falling back to Gemini")
            self._set_route("Gemini")
            try:
                response = await self.router.process_query(
                    query,
                    self._get_tool_schemas(),
                )
            except Exception:
                logger.exception("LLM failed to process request")
                self._set_route("Gemini failed")
                return "Failed to process the request. Check the logs."

        if not isinstance(response, dict):
            self._set_route(f"{source} returned invalid response")
            logger.error(
                "Unexpected response type: %s",
                type(response).__name__,
            )
            return "The model returned an invalid response."

        response_type = response.get("type")

        if response_type == "text":
            self._set_route("Gemini → Text" if source == "Gemini" else f"{source} → Text")
            text = response.get("text", "")

            if not isinstance(text, str):
                logger.error("Invalid text response from model")
                return "The model returned an invalid response."

            return text

        if response_type != "tool_call":
            self._set_route(f"{source} returned unsupported response")
            logger.warning(
                "Unsupported response type: %r",
                response_type,
            )
            return "The model returned an unsupported response."

        name = response.get("tool_name")
        args = response.get("arguments")

        if not isinstance(name, str) or name not in self._tool_names:
            self._set_route(f"{source} requested unavailable tool")
            logger.error("Model requested an unknown tool: %r", name)
            return "The model requested an unavailable tool."

        if not isinstance(args, dict):
            self._set_route(f"{source} returned invalid tool arguments")
            logger.error("Invalid arguments for tool %s", name)
            return "The model returned invalid tool arguments."

        logger.info("Executing MCP tool: %s", name)
        if source == "Gemini":
            self._set_route(f"Gemini → MCP: {name}")
        else:
            self._set_route(f"{source} → MCP: {name}")
        logger.debug("Tool arguments: %r", args)

        return await self._call_tool(name, args)

    async def _call_tool(self, name: str, args: dict[str, Any]) -> str:
        """Call an MCP tool and retain any pending path choice.

        Args:
            name - Name of the MCP tool to call.
            args - Arguments supplied to the tool.

        Returns:
            User-facing result or path selection question.
        """
        try:
            result = await self.client.call_tool(name, args)
        except Exception:
            logger.exception("MCP call failed for tool %s", name)
            self._pending_clarification = None
            return f"Tool '{name}' failed. Check the logs."

        structured = getattr(result, "structured_content", None)
        if not isinstance(structured, dict):
            structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict):
            choices = structured.get("clarification_choices")
            field = structured.get("clarification_field")
            if (
                structured.get("status") == "clarification_required"
                and isinstance(choices, list)
                and choices
                and isinstance(field, str)
                and field in args
            ):
                valid_choices = [choice for choice in choices if isinstance(choice, str)]
                if valid_choices:
                    default_path = structured.get("clarification_default")
                    self._pending_clarification = {
                        "tool_name": name,
                        "arguments": dict(args),
                        "field": field,
                        "choices": valid_choices,
                    }
                    detail = structured.get("message")
                    heading = detail if isinstance(detail, str) else "Which path should I use?"
                    options = "\n".join(
                        f"{index}. {path}"
                        f"{' (topmost match)' if path == default_path else ''}"
                        for index, path in enumerate(valid_choices, start=1)
                    )
                    return (
                        f"{heading}\n{options}\n"
                        "Reply with a number or paste a path. Type 'cancel' to stop."
                    )

        self._pending_clarification = None
        return self._format_tool_result(name, result)

    async def _resolve_clarification(self, answer: str) -> str:
        """Apply a user's selected path to the pending tool request.

        Args:
            answer - Number or absolute path selected by the user.

        Returns:
            User-facing result or a repeated selection prompt.
        """
        pending = self._pending_clarification
        if pending is None:
            return "There is no pending path choice."

        if answer.casefold() in {"cancel", "cancelled", "never mind"}:
            self._pending_clarification = None
            return "Cancelled. No file operation was performed."

        choices = pending["choices"]
        selected_path = None
        if answer.isdigit():
            index = int(answer) - 1
            if 0 <= index < len(choices):
                selected_path = choices[index]
        else:
            selected_path = next(
                (path for path in choices if answer == path),
                None,
            )

        if selected_path is None:
            return (
                "Please choose one of the listed numbers or paste one of the "
                "listed paths. Type 'cancel' to stop."
            )

        name = pending["tool_name"]
        field = pending["field"]
        args = dict(pending["arguments"])
        args[field] = selected_path
        args["clarified_path"] = selected_path
        args["clarified_field"] = field
        self._pending_clarification = None
        return await self._call_tool(name, args)

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
