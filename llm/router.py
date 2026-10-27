
from typing import Any

from llm.base import BaseLLM
from llm.gemini import GeminiLLM


class LLMRouter:
    """Select and expose the configured LLM provider."""

    def __init__(self, provider: str = "gemini") -> None:
        provider = provider.lower().strip()

        if provider == "gemini":
            self.llm: BaseLLM = GeminiLLM()
        else:
            raise ValueError(
                f"Unsupported LLM provider: {provider}"
            )

    async def process_query(
        self,
        query: str,
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return await self.llm.process_query(query, tools)

    async def process_tool_result(
        self,
        tool_call_id: str,
        result: str,
    ) -> dict[str, Any]:
        return await self.llm.process_tool_result(
            tool_call_id,
            result,
        )
