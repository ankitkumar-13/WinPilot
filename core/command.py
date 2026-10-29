from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Command:
    """One action request with optional app-level approval.

    Attributes:
        action - Executor action name.
        arguments - Arguments consumed by the executor.
        requires_approval - Whether WinPilot asks for app-level approval.
            This does not request Windows administrator privileges.
    """

    action: str
    arguments: dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False

    def to_payload(self) -> dict[str, Any]:
        """Return the action and arguments expected by the executor.

        Returns:
            Dictionary containing the action and its arguments.
        """
        return {"action": self.action, **self.arguments}

    def __getitem__(self, key: str) -> Any:
        """Read a payload value using dictionary-style access.

        Args:
            key - Payload key to read.

        Returns:
            Value associated with the key.
        """
        return self.to_payload()[key]
