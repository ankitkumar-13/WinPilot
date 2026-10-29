from collections.abc import Callable
from typing import Any

from llm.base import BaseLLM
from llm.gemini import GeminiLLM


class LLMRouter:
    """Select and expose the configured LLM provider."""

    def __init__(self, provider: str = "gemini") -> None:
        """Initialise the selected LLM provider.

        Args:
            provider - Name of the provider to initialise.

        Raises:
            ValueError - If the requested provider is unsupported.

        Returns:
            None.
        """
        provider = provider.strip().lower()

        providers: dict[str, Callable[[], BaseLLM]] = {
            "gemini": GeminiLLM,
        }

        provider_factory = providers.get(provider)

        if provider_factory is None:
            supported = ", ".join(sorted(providers))
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers: {supported}"
            )

        self.llm = provider_factory()

    async def process_query(
        self,
        query: str,
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Forward a user query to the configured provider.

        Args:
            query - Natural-language request from the user.
            tools - Available tool definitions and input schemas.

        Returns:
            Provider response containing text or a tool call.
        """
        return await self.llm.process_query(query, tools)

    async def process_tool_result(
        self,
        tool_call_id: str,
        result: str,
    ) -> dict[str, Any]:
        """Forward a tool result to the configured provider.

        Args:
            tool_call_id - Identifier associated with the tool call.
            result - Textual result returned by the tool.

        Returns:
            Provider response after processing the tool result.
        """
        return await self.llm.process_tool_result(tool_call_id, result)
