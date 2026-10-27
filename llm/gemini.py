
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from llm.base import BaseLLM

load_dotenv()


class GeminiLLM(BaseLLM):
    """Gemini provider for WinPilot."""

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        super().__init__(model)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing from .env")

        self.client = genai.Client(api_key=api_key)

    async def process_query(
        self,
        query: str,
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Send a user query and available MCP tools to Gemini."""

        declarations = [
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters_json_schema=tool.get(
                    "inputSchema",
                    {
                        "type": "object",
                        "properties": {},
                    },
                ),
            )
            for tool in tools
        ]

        config = types.GenerateContentConfig(
            system_instruction=(
                "You are WinPilot, a Windows desktop automation assistant. "
                "Use available tools when the user requests an action. "
                "Never claim an action succeeded unless a tool result confirms it. "
                "Ask for clarification when the request is ambiguous."
            ),
            tools=[
                types.Tool(function_declarations=declarations)
            ] if declarations else None,
            automatic_function_calling=(
                types.AutomaticFunctionCallingConfig(disable=True)
            ),
        )

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=query,
            config=config,
        )

        if not response.candidates:
            return {
                "type": "text",
                "text": "Gemini returned no response.",
            }

        candidate = response.candidates[0]
        if candidate.content is None or candidate.content.parts is None:
            return {
                "type": "text",
                "text": "Gemini returned no response content.",
            }

        for part in candidate.content.parts:
            if part.function_call is not None:
                call = part.function_call

                return {
                    "type": "tool_call",
                    "tool_name": call.name,
                    "arguments": dict(call.args or {}),
                }

        return {
            "type": "text",
            "text": response.text or "I couldn't generate a response.",
        }

    async def process_tool_result(
        self,
        tool_call_id: str,
        result: str,
    ) -> dict[str, Any]:
        """Temporary placeholder for tool-result handling."""

        return {
            "type": "text",
            "text": result,
        }
