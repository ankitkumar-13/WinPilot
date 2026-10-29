import logging
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from llm.base import BaseLLM

logger = logging.getLogger(__name__)

load_dotenv()


class GeminiLLM(BaseLLM):
    """Process WinPilot requests using the Gemini API."""

    def __init__(self, model: str = "gemini-3.5-flash-lite") -> None:
        """Initialise the Gemini client.

        Args:
            model - Gemini model identifier.

        Raises:
            ValueError - If the API key is missing.

        Returns:
            None.
        """
        super().__init__(model)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Configure it in your .env file."
            )

        self.client = genai.Client(api_key=api_key)
        logger.info("Initialised Gemini provider with model %s", model)

    def _build_tool_declarations(
        self,
        tools: list[dict[str, Any]],
    ) -> list[types.FunctionDeclaration]:
        """Convert MCP tool definitions into Gemini declarations.

        Args:
            tools - MCP tool definitions with input schemas.

        Returns:
            Gemini-compatible function declarations.
        """
        declarations = []

        for tool in tools:
            name = tool.get("name")
            if not isinstance(name, str) or not name.strip():
                logger.warning("Skipping tool with invalid name")
                continue

            schema = tool.get("inputSchema")
            if not isinstance(schema, dict):
                schema = {
                    "type": "object",
                    "properties": {},
                }

            description = tool.get("description", "")
            if not isinstance(description, str):
                description = ""

            declarations.append(
                types.FunctionDeclaration(
                    name=name,
                    description=description,
                    parameters_json_schema=schema,
                )
            )

        return declarations

    async def process_query(
        self,
        query: str,
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Send a user query and available tools to Gemini.

        Args:
            query - Natural-language request from the user.
            tools - MCP tool definitions available for this request.

        Returns:
            Dictionary containing a text response or tool call.
        """
        query = query.strip()

        if not query:
            return {
                "type": "text",
                "text": "Please enter a command.",
            }

        declarations = self._build_tool_declarations(tools)

        config = types.GenerateContentConfig(
            system_instruction=(
                "You are WinPilot, a Windows desktop automation assistant. "
                "Keep responses concise and conversational. "
                "Use available tools for clear, supported actions. "
                "If a request is ambiguous, ask one brief clarifying question. "
                "Never claim an action succeeded unless a tool result confirms it. "
                "Do not execute destructive actions without explicit confirmation."
            ),
            tools=(
                [types.Tool(function_declarations=declarations)]
                if declarations
                else None
            ),
            automatic_function_calling=(
                types.AutomaticFunctionCallingConfig(disable=True)
            ),
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=query,
                config=config,
            )
        except Exception:
            logger.exception("Gemini request failed")
            return {
                "type": "text",
                "text": "Gemini could not process the request. Check the logs.",
            }

        candidates = response.candidates or []
        if not candidates:
            logger.warning("Gemini returned no candidates")
            return {
                "type": "text",
                "text": "Gemini returned no response.",
            }

        candidate = candidates[0]
        content = candidate.content

        if content is None or not content.parts:
            logger.warning("Gemini returned no response content")
            return {
                "type": "text",
                "text": "Gemini returned no response content.",
            }

        for part in content.parts:
            call = part.function_call

            if call is None:
                continue

            if not isinstance(call.name, str) or not call.name:
                logger.warning("Gemini returned a function call without a name")
                return {
                    "type": "text",
                    "text": "Gemini returned an invalid tool call.",
                }

            arguments = dict(call.args or {})

            logger.info("Gemini selected tool: %s", call.name)

            return {
                "type": "tool_call",
                "tool_name": call.name,
                "arguments": arguments,
            }

        text = response.text

        return {
            "type": "text",
            "text": text.strip() if text else "I couldn't generate a response.",
        }

    async def process_tool_result(
        self,
        tool_call_id: str,
        result: str,
    ) -> dict[str, Any]:
        """Return a completed tool result to the caller.

        Args:
            tool_call_id - Identifier associated with the tool call.
            result - Textual result returned by the executed tool.

        Returns:
            Dictionary containing the tool result as text.
        """
        logger.debug("Received result for tool call %s", tool_call_id)

        return {
            "type": "text",
            "text": result,
        }
