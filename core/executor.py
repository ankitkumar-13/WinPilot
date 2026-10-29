import logging
import os
from pathlib import Path
from typing import Any

from core.confirmation import confirm_action
from core.command import Command
from core.execution_result import ExecutionResult, ExecutionStatus
from plugins.explorer_context import ExplorerContext
from plugins.folder_utils import FolderUtil
from plugins.windows_utils import WindowsUtils

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Dispatch validated commands and return structured results."""

    def __init__(self) -> None:
        """Create the utilities used to perform WinPilot actions.

        Returns:
            None.
        """
        self.windows_utils = WindowsUtils()
        self.folder_utils = FolderUtil()
        self.explorer_context = ExplorerContext()

    def _fail(
        self,
        status: ExecutionStatus,
        message: str,
    ) -> ExecutionResult:
        """Log and return a structured failure.

        Args:
            status - Classification of the failure.
            message - Explanation of the failure.

        Returns:
            Failed execution result.
        """
        logger.warning("%s", message)
        return ExecutionResult.fail(status, message)

    def get_explorer_workspace(self) -> str:
        """Get the active workspace or fall back to the Desktop.

        Returns:
            Path of the active workspace or Desktop fallback.
        """
        workspace = self.explorer_context.get_folder()

        if not workspace:
            workspace = os.path.join(
                os.path.expanduser("~"),
                "Desktop",
            )

        return workspace

    def _resolve_path(self, path: str, workspace: str) -> str:
        """Resolve a path relative to the current workspace.

        Args:
            path - Absolute or workspace-relative path to resolve.
            workspace - Base directory for relative paths.

        Returns:
            Normalised absolute path.
        """
        if os.path.isabs(path):
            return os.path.abspath(os.path.normpath(path))

        return os.path.abspath(
            os.path.normpath(os.path.join(workspace, path))
        )

    def _request_approval(
        self,
        action: str,
        details: str,
        required: bool,
    ) -> ExecutionResult | None:
        """Return a failure when required approval is denied or unavailable.

        Args:
            action - Human-readable name of the requested operation.
            details - Application or paths affected by the operation.
            required - Whether this command requires app-level approval.

        Returns:
            Failure result if approval is denied; otherwise None.
        """
        if not required:
            return None

        try:
            approved = confirm_action(action, details)
        except Exception:
            logger.exception("Command approval failed: %s", action)
            return self._fail(
                ExecutionStatus.INTERNAL_ERROR,
                "Could not request permission for this command.",
            )

        if not approved:
            return self._fail(
                ExecutionStatus.USER_CANCELLED,
                f"Command not approved: {action} ({details})",
            )

        return None

    def _delete_item(
        self,
        kind: str,
        path: str,
        requires_elevation: bool,
    ) -> ExecutionResult:
        """Approve and delete one file or folder.

        Args:
            kind - Expected target type, either file or folder.
            path - Resolved path of the item to delete.
            requires_elevation - Whether WinPilot approval is required.

        Returns:
            Structured result describing the deletion outcome.
        """
        target = Path(path)

        if not target.exists() and not target.is_symlink():
            return self._fail(
                ExecutionStatus.TARGET_NOT_FOUND,
                f"Cannot delete {kind}: target does not exist: {target}",
            )

        if target.is_symlink() or target.is_file():
            actual_kind = "file"
        elif target.is_dir():
            actual_kind = "folder"
        else:
            return self._fail(
                ExecutionStatus.OPERATION_FAILED,
                f"Unsupported deletion target: {target}",
            )

        if actual_kind != kind:
            return self._fail(
                ExecutionStatus.INVALID_ARGUMENT,
                f"Cannot delete {target}: expected a {kind}, "
                f"but the target is a {actual_kind}.",
            )

        denied = self._request_approval(
            f"delete {kind}",
            str(target),
            requires_elevation,
        )
        if denied:
            return denied

        try:
            if kind == "file":
                result = self.folder_utils.delete_file_result(str(target))
            else:
                result = self.folder_utils.delete_folder_result(str(target))
        except Exception:
            logger.exception("Deletion utility failed: %s", target)
            return self._fail(
                ExecutionStatus.INTERNAL_ERROR,
                f"An unexpected error occurred while deleting {target}.",
            )

        if result.success:
            return ExecutionResult.ok(f"Deleted {kind}: {target}")

        status_value = getattr(result.status, "value", str(result.status))
        message = getattr(result, "message", None) or status_value

        logger.warning(
            "Deletion failed for %s: %s (%s)",
            target,
            message,
            status_value,
        )

        if status_value in {"NOT_FOUND", "not_found"}:
            status = ExecutionStatus.TARGET_NOT_FOUND
        elif status_value in {"PERMISSION_DENIED", "permission_denied"}:
            status = ExecutionStatus.PERMISSION_DENIED
        else:
            status = ExecutionStatus.OPERATION_FAILED

        return self._fail(status, f"Cannot delete {target}: {message}")

    def execute(self, command: Command | dict[str, Any]) -> ExecutionResult:
        """Validate and execute a command, returning its outcome.

        Args:
            command - Command object or action dictionary to execute.

        Returns:
            Structured result describing success or failure.
        """
        if isinstance(command, Command):
            requires_elevation = command.requires_elevation
            command = command.to_payload()
        else:
            requires_elevation = False

        if not isinstance(command, dict):
            return self._fail(
                ExecutionStatus.INVALID_ARGUMENT,
                "Invalid command: expected a dictionary.",
            )

        action = command.get("action")
        target = command.get("target")

        if not isinstance(action, str):
            return self._fail(
                ExecutionStatus.INVALID_ARGUMENT,
                "Invalid command: missing action.",
            )

        try:
            if action == "open_application":
                if not isinstance(target, str) or not target.strip():
                    return self._fail(
                        ExecutionStatus.INVALID_ARGUMENT,
                        "Application name is missing.",
                    )

                denied = self._request_approval(
                    "open an application",
                    target.strip(),
                    requires_elevation,
                )
                if denied:
                    return denied

                if not self.windows_utils.open_application(target):
                    return self._fail(
                        ExecutionStatus.OPERATION_FAILED,
                        f"Could not open application: {target}",
                    )

                return ExecutionResult.ok(f"Opened {target}")

            if action in {"create_file", "create_folder"}:
                if not isinstance(target, str) or not target.strip():
                    return self._fail(
                        ExecutionStatus.INVALID_ARGUMENT,
                        f"Target path is missing for {action}.",
                    )

                workspace = self.get_explorer_workspace()
                path = self._resolve_path(target, workspace)
                item = "file" if action == "create_file" else "folder"
                denied = self._request_approval(
                    f"create a {item}",
                    path,
                    requires_elevation,
                )
                if denied:
                    return denied

                if action == "create_file":
                    succeeded = self.folder_utils.create_file(path)
                else:
                    succeeded = self.folder_utils.create_folder(path)

                if not succeeded:
                    return self._fail(
                        ExecutionStatus.OPERATION_FAILED,
                        f"Could not create {item}: {path}",
                    )

                return ExecutionResult.ok(f"Created {item}: {path}")

            if action in {"copy", "move"}:
                source = command.get("source")
                destination = command.get("destination")

                if (
                    not isinstance(source, str)
                    or not source.strip()
                    or not isinstance(destination, str)
                    or not destination.strip()
                ):
                    return self._fail(
                        ExecutionStatus.INVALID_ARGUMENT,
                        f"Source or destination is missing for {action}.",
                    )

                workspace = self.get_explorer_workspace()
                source_path = self._resolve_path(source, workspace)
                destination_path = self._resolve_path(
                    destination,
                    workspace,
                )
                denied = self._request_approval(
                    action,
                    f"From: {source_path}\nTo: {destination_path}",
                    requires_elevation,
                )
                if denied:
                    return denied

                if action == "copy":
                    succeeded = self.folder_utils.copy(
                        source_path,
                        destination_path,
                    )
                else:
                    succeeded = self.folder_utils.move(
                        source_path,
                        destination_path,
                    )

                if not succeeded:
                    return self._fail(
                        ExecutionStatus.OPERATION_FAILED,
                        f"Could not {action} {source_path} "
                        f"to {destination_path}.",
                    )

                return ExecutionResult.ok(
                    f"{'Copied' if action == 'copy' else 'Moved'} "
                    f"{source_path} to {destination_path}"
                )

            if action in {"delete_file", "delete_folder"}:
                if not isinstance(target, str) or not target.strip():
                    return self._fail(
                        ExecutionStatus.INVALID_ARGUMENT,
                        f"Target path is missing for {action}.",
                    )

                workspace = self.get_explorer_workspace()
                path = self._resolve_path(target, workspace)
                kind = "file" if action == "delete_file" else "folder"

                return self._delete_item(kind, path, requires_elevation)

            if action in {"rename_file", "rename_folder"}:
                new_name = command.get("new_name")

                if (
                    not isinstance(target, str)
                    or not target.strip()
                    or not isinstance(new_name, str)
                    or not new_name.strip()
                ):
                    return self._fail(
                        ExecutionStatus.INVALID_ARGUMENT,
                        "Target path or new name is missing for rename.",
                    )

                workspace = self.get_explorer_workspace()
                path = self._resolve_path(target, workspace)
                rename_target = Path(path)

                if rename_target.exists() or rename_target.is_symlink():
                    matches_kind = (
                        rename_target.is_dir() and not rename_target.is_symlink()
                        if action == "rename_folder"
                        else rename_target.is_file() or rename_target.is_symlink()
                    )
                    if not matches_kind:
                        expected = "folder" if action == "rename_folder" else "file"
                        return self._fail(
                            ExecutionStatus.INVALID_ARGUMENT,
                            f"Cannot rename {path}: expected a {expected}.",
                        )

                kind = "folder" if action == "rename_folder" else "file"
                destination_path = (
                    new_name
                    if os.path.dirname(new_name)
                    else os.path.join(os.path.dirname(path), new_name)
                )
                denied = self._request_approval(
                    f"rename a {kind}",
                    f"From: {path}\nTo: {destination_path}",
                    requires_elevation,
                )
                if denied:
                    return denied

                if not self.folder_utils.rename(path, new_name):
                    return self._fail(
                        ExecutionStatus.OPERATION_FAILED,
                        f"Could not rename {path} to {new_name}.",
                    )

                return ExecutionResult.ok(
                    f"Renamed {path} to {new_name}"
                )

            return self._fail(
                ExecutionStatus.INVALID_ARGUMENT,
                f"Unknown action: {action!r}",
            )

        except Exception:
            logger.exception("Unexpected error executing action: %s", action)
            return self._fail(
                ExecutionStatus.INTERNAL_ERROR,
                f"Unexpected internal error while executing {action}.",
            )
