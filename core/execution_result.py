
from dataclasses import dataclass
from enum import Enum


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    TARGET_NOT_FOUND = "target_not_found"
    PERMISSION_DENIED = "permission_denied"
    USER_CANCELLED = "user_cancelled"
    INVALID_ARGUMENT = "invalid_argument"
    OPERATION_FAILED = "operation_failed"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    status: ExecutionStatus
    message: str

    @classmethod
    def ok(cls, message: str) -> "ExecutionResult":
        return cls(
            success=True,
            status=ExecutionStatus.SUCCESS,
            message=message,
        )

    @classmethod
    def fail(
        cls,
        status: ExecutionStatus,
        message: str,
    ) -> "ExecutionResult":
        return cls(
            success=False,
            status=status,
            message=message,
        )
