
from typing import Any

from app.mcp_client import WinPilotMCPClient
from llm.router import LLMRouter


class WinPilotOrchestrator:
    """Coordinate LLM requests and MCP tool execution."""

    def __init__(
        self,
        client: WinPilotMCPClient,
        router: LLMRouter,
    ) -> None:
        self.client = client
        self.router = router

    async def process(self, query: str) -> str:
        # Get available MCP tools.
        mcp_tools = await self.client.list_tools()

        # Convert MCP tool objects into dictionaries.
        tools: list[dict[str, Any]] = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.input_schema,
            }
            for tool in mcp_tools
        ]

        # Ask the LLM to interpret the request.
        response = await self.router.process_query(query, tools)

        if response["type"] == "text":
            return response["text"]

        if response["type"] != "tool_call":
            return "The model returned an unsupported response."

        name = response["tool_name"]
        args = response["arguments"]

        # Verify the requested tool is actually available.
        if name not in {tool.name for tool in mcp_tools}:
            return f"The model requested an unknown tool: {name}"

        # Execute the selected MCP tool.
        result = await self.client.call_tool(name, args)

        
        output_parts: list[str] = []

        for item in result.content:
            if item.type == "text":
                output_parts.append(item.text)

        output = "\n".join(output_parts)

        if result.is_error:
            return f"Tool execution failed: {output}"

        return output or "The tool completed without a text result."
