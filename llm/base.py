from abc import ABC, abstractmethod
from typing import Any

class BaseLLM(ABC):
    """Shared interface for WinPilot LLM providers."""

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    async def process_query(
        self,
        query: str,
        tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Process a user query using the available tools."""
        raise NotImplementedError

    @abstractmethod
    async def process_tool_result(
        self,
        tool_call_id: str,
        result:str,
    ) -> dict[str, Any]:
        """Process a tool result and decide the next response or action."""
        raise NotImplementedError
