from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    """Define the interface implemented by each LLM provider."""

    def __init__(self, model: str) -> None:
        """Initialise the provider with its model identifier.

        Args:
            model - Name or identifier of the LLM model.
        """
        self.model = model

    @abstractmethod
    async def process_query(
        self,
        query: str,
        tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Process a user query using the available tools.

        Args:
            query - Natural-language request from the user.
            tools - Available tool definitions and input schemas.

        Returns:
            Dictionary containing a text response or tool call.
        """
        raise NotImplementedError

    @abstractmethod
    async def process_tool_result(
        self,
        tool_call_id: str,
        result: str
    ) -> dict[str, Any]:
        """Process a tool result and produce a response.

        Args:
            tool_call_id - Identifier associated with the tool call.
            result - Textual result returned by the executed tool.

        Returns:
            Dictionary containing the provider's response.
        """
        raise NotImplementedError
