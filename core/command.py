from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Command:
    """One action request with optional app-level approval.

    Attributes:
        action - Executor action name.
        arguments - Arguments consumed by the executor.
        requires_elevation - Whether WinPilot asks for app-level approval.
            This does not request Windows administrator privileges.
    """

    action: str
    arguments: dict[str, Any] = field(default_factory=dict)
    requires_elevation: bool = False

    def to_payload(self) -> dict[str, Any]:
        """Return the action and arguments expected by the executor.

        Returns:
            Dictionary containing the action and its arguments.
        """
        return {"action": self.action, **self.arguments}
