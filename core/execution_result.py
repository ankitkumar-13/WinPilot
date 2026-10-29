from dataclasses import dataclass
from enum import Enum


class ExecutionStatus(str, Enum):
    """Classify the outcome of an executed command."""

    SUCCESS = "success"
    TARGET_NOT_FOUND = "target_not_found"
    PERMISSION_DENIED = "permission_denied"
    USER_CANCELLED = "user_cancelled"
    INVALID_ARGUMENT = "invalid_argument"
    OPERATION_FAILED = "operation_failed"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of one operation; check ``success`` for the Boolean result.

    Attributes:
        success - Whether the operation completed successfully.
        status - Machine-readable classification of the outcome.
        message - Human-readable explanation of the outcome.
    """

    success: bool
    status: ExecutionStatus
    message: str

    @classmethod
    def ok(cls, message: str) -> "ExecutionResult":
        """Create a successful result.

        Args:
            message - User-facing description of the completed operation.

        Returns:
            Successful execution result.
        """
        return cls(True, ExecutionStatus.SUCCESS, message)

    @classmethod
    def fail(
        cls,
        status: ExecutionStatus,
        message: str,
    ) -> "ExecutionResult":
        """Create a failed result with its reason.

        Args:
            status - Classification of the failure.
            message - User-facing explanation of the failure.

        Returns:
            Failed execution result.
        """
        return cls(False, status, message)
