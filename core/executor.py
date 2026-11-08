import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from core.confirmation import confirm_action
from core.command import Command
from core.execution_result import ExecutionResult, ExecutionStatus
from plugins.explorer_context import ExplorerContext
from plugins.folder_utils import FolderUtil
from plugins.windows_utils import WindowsUtils

logger = logging.getLogger(__name__)

APPROVAL_REQUIRED_ACTIONS = frozenset({"delete_file", "delete_folder"})


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
        load_dotenv()
        self.ambiguous_target_policy = os.environ.get(
            "WINPILOT_AMBIGUOUS_TARGET_POLICY",
            "ask",
        ).strip().lower()
        if self.ambiguous_target_policy not in {"ask", "topmost"}:
            logger.warning(
                "Unknown ambiguous-target policy %r; using 'ask'",
                self.ambiguous_target_policy,
            )
            self.ambiguous_target_policy = "ask"

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

    def get_explorer_workspace(self) -> str | None:
        """Get the foremost Explorer folder or the Desktop fallback.

        Returns:
            Path of the foremost Explorer folder or the user's Desktop.
        """
        workspace = self.explorer_context.get_folder()
        if workspace:
            return workspace

        return os.path.join(os.path.expanduser("~"), "Desktop")

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

    def _workspace_context(
        self,
    ) -> tuple[str | None, list[str], ExecutionResult | None]:
        """Get the top Explorer folder and the folders in other open windows.

        Returns:
            Workspace path, open Explorer folders, and a clarification result
            when no usable workspace is available.
        """
        workspace = self.get_explorer_workspace()
        if not workspace or not os.path.isdir(workspace):
            return None, [], ExecutionResult.fail(
                ExecutionStatus.CLARIFICATION_REQUIRED,
                "I can't identify a usable file workspace. When no visible "
                "Explorer window is available, the Desktop is the fallback.",
            )

        workspace = os.path.abspath(os.path.normpath(workspace))
        folders = [workspace]
        try:
            open_folders = self.explorer_context.get_open_folders()
        except Exception:
            logger.exception("Could not inspect all open Explorer windows")
            return None, [], ExecutionResult.fail(
                ExecutionStatus.CLARIFICATION_REQUIRED,
                "I couldn't check all open File Explorer windows for matching "
                "items. I can't safely check for duplicate names right now.",
            )

        for folder in open_folders:
            if not isinstance(folder, str) or not folder:
                continue
            normalized = os.path.abspath(os.path.normpath(folder))
            if os.path.normcase(normalized) not in {
                os.path.normcase(item) for item in folders
            }:
                folders.append(normalized)
        return workspace, folders, None

    @staticmethod
    def _is_within_workspace(path: str, workspace: str) -> bool:
        """Return whether a path is contained by the selected workspace.

        Args:
            path - Candidate absolute path.
            workspace - Selected top Explorer folder.

        Returns:
            True when the candidate is inside the workspace.
        """
        try:
            common_path = os.path.commonpath((path, workspace))
            return os.path.normcase(common_path) == os.path.normcase(workspace)
        except ValueError:
            return False

    @staticmethod
    def _matches_target_kind(path: str, expected_kind: str | None) -> bool:
        """Check whether an existing path matches the requested item type.

        Args:
            path - Candidate filesystem path.
            expected_kind - Required type, or None when either type is valid.

        Returns:
            True when the path exists and matches the requested type.
        """
        if not os.path.lexists(path):
            return False
        if expected_kind == "file":
            return os.path.isfile(path) or os.path.islink(path)
        if expected_kind == "folder":
            return os.path.isdir(path) and not os.path.islink(path)
        return True

    def _resolve_existing_target(
        self,
        target: str,
        workspace: str,
        explorer_folders: list[str],
        expected_kind: str | None = None,
        field_name: str = "target",
        clarified_path: str | None = None,
        clarified_field: str | None = None,
    ) -> tuple[str | None, ExecutionResult | None]:
        """Resolve and validate an existing target against open Explorer folders.

        Args:
            target - Name or path supplied by the command.
            workspace - Selected top Explorer folder.
            explorer_folders - Paths represented by open Explorer windows.
            expected_kind - Required target type, when the action specifies one.
            field_name - MCP argument associated with this target.
            clarified_path - Path explicitly selected by the user.
            clarified_field - MCP argument resolved by the user's selection.

        Returns:
            Resolved target and None when valid, otherwise a clarification result.
        """
        path = self._resolve_path(target, workspace)
        if (
            clarified_field == field_name
            and clarified_path
            and os.path.isabs(target)
            and os.path.normcase(os.path.abspath(target))
            == os.path.normcase(os.path.abspath(clarified_path))
            and os.path.lexists(path)
        ):
            if expected_kind == "file" and not (
                os.path.isfile(path) or os.path.islink(path)
            ):
                return None, ExecutionResult.fail(
                    ExecutionStatus.CLARIFICATION_REQUIRED,
                    f"'{path}' is not a file. Choose a file and try again.",
                )
            if expected_kind == "folder" and not (
                os.path.isdir(path) and not os.path.islink(path)
            ):
                return None, ExecutionResult.fail(
                    ExecutionStatus.CLARIFICATION_REQUIRED,
                    f"'{path}' is not a folder. Choose a folder and try again.",
                )
            return path, None

        if not self._is_within_workspace(path, workspace):
            return None, ExecutionResult.fail(
                ExecutionStatus.CLARIFICATION_REQUIRED,
                f"'{target}' is outside the top File Explorer folder '{workspace}'. "
                "Choose a listed path or provide an absolute path.",
            )

        if not self._matches_target_kind(path, expected_kind):
            other_matches = []
            if not os.path.isabs(target):
                for folder in explorer_folders:
                    if os.path.normcase(folder) == os.path.normcase(workspace):
                        continue
                    candidate = self._resolve_path(target, folder)
                    if self._matches_target_kind(candidate, expected_kind):
                        other_matches.append(candidate)
            if other_matches:
                choices = tuple(sorted(set(other_matches)))
                locations = ", ".join(choices)
                message = (
                    f"I found '{target}' at these paths: {locations}. "
                    "Which one should I use?"
                )
                return None, ExecutionResult.fail(
                    ExecutionStatus.CLARIFICATION_REQUIRED,
                    message,
                    clarification_choices=choices,
                    clarification_field=field_name,
                )
            if os.path.lexists(path):
                expected = expected_kind or "requested"
                return None, ExecutionResult.fail(
                    ExecutionStatus.CLARIFICATION_REQUIRED,
                    f"'{path}' isn't a {expected}. Choose the correct item "
                    "type or provide its full path.",
                )
            else:
                message = (
                    f"I couldn't find '{target}' in the top Explorer folder "
                    f"'{workspace}'. Check the name or bring the folder "
                    "containing it to the front, then try again."
                )
                choices = ()
            return None, ExecutionResult.fail(
                ExecutionStatus.CLARIFICATION_REQUIRED,
                message,
                clarification_choices=choices,
                clarification_field=field_name if choices else None,
            )

        other_matches = []
        if not os.path.isabs(target):
            for folder in explorer_folders:
                if os.path.normcase(folder) == os.path.normcase(workspace):
                    continue
                candidate = self._resolve_path(target, folder)
                if (
                    self._matches_target_kind(candidate, expected_kind)
                    and os.path.normcase(candidate) != os.path.normcase(path)
                ):
                    other_matches.append(candidate)
        if other_matches:
            choices = tuple(dict.fromkeys([path, *sorted(set(other_matches))]))
            if self.ambiguous_target_policy == "topmost":
                logger.info(
                    "Using top Explorer match for %s: %s",
                    target,
                    path,
                )
            else:
                locations = ", ".join(choices)
                message = (
                    f"I found multiple items named '{target}': {locations}. "
                    "Which one should I use?"
                )
                return None, ExecutionResult.fail(
                    ExecutionStatus.CLARIFICATION_REQUIRED,
                    message,
                    clarification_choices=choices,
                    clarification_field=field_name,
                    clarification_default=path,
                )

        return path, None

    def _resolve_destination(
        self,
        destination: str,
        workspace: str,
        explorer_folders: list[str],
        base_folder: str | None = None,
        field_name: str = "destination",
        clarified_path: str | None = None,
        clarified_field: str | None = None,
        check_duplicates: bool = True,
    ) -> tuple[str | None, ExecutionResult | None]:
        """Resolve a destination and require it to remain in the workspace.

        Args:
            destination - Destination name or path supplied by the command.
            workspace - Selected top Explorer folder.
            explorer_folders - Paths represented by open Explorer windows.
            base_folder - Folder used for a simple rename destination.
            field_name - MCP argument associated with this destination.
            clarified_path - Path explicitly selected by the user.
            clarified_field - MCP argument resolved by the user's selection.
            check_duplicates - Whether to compare this path across open windows.

        Returns:
            Resolved destination and None when valid, otherwise a clarification
            result.
        """
        if os.path.isabs(destination):
            path = os.path.abspath(os.path.normpath(destination))
        elif base_folder and not os.path.dirname(destination):
            path = os.path.abspath(os.path.join(base_folder, destination))
        else:
            path = self._resolve_path(destination, workspace)

        selected_path = (
            clarified_field == field_name
            and clarified_path
            and os.path.isabs(destination)
            and os.path.normcase(path)
            == os.path.normcase(os.path.abspath(clarified_path))
        )
        if not selected_path and not self._is_within_workspace(path, workspace):
            return None, ExecutionResult.fail(
                ExecutionStatus.CLARIFICATION_REQUIRED,
                f"The destination '{destination}' is outside the top File "
                f"Explorer folder '{workspace}'. Choose a listed folder or "
                "provide a destination inside the current workspace.",
            )

        compare_other_windows = check_duplicates and not selected_path and not os.path.isabs(destination) and not (
            base_folder is not None and not os.path.dirname(destination)
        )
        if compare_other_windows:
            other_matches = []
            for folder in explorer_folders:
                if os.path.normcase(folder) == os.path.normcase(workspace):
                    continue
                candidate = self._resolve_path(destination, folder)
                if (
                    os.path.isdir(candidate)
                    and os.path.normcase(candidate) != os.path.normcase(path)
                ):
                    other_matches.append(candidate)
            if other_matches:
                if not os.path.isdir(path):
                    locations = ", ".join(sorted(set(other_matches)))
                    message = (
                        f"The destination '{destination}' exists at "
                        f"'{locations}'. Choose the destination to use."
                    )
                    choices = tuple(sorted(set(other_matches)))
                    choices = tuple(sorted(set(other_matches)))
                else:
                    choices = tuple(dict.fromkeys([path, *sorted(set(other_matches))]))
                    if self.ambiguous_target_policy == "topmost":
                        logger.info(
                            "Using top Explorer destination for %s: %s",
                            destination,
                            path,
                        )
                        choices = ()
                        message = ""
                    else:
                        locations = ", ".join(choices)
                        message = (
                            f"The destination '{destination}' exists in multiple "
                            f"open folders: {locations}. Which one should I use?"
                        )
                if choices:
                    return None, ExecutionResult.fail(
                        ExecutionStatus.CLARIFICATION_REQUIRED,
                        message,
                        clarification_choices=choices,
                        clarification_field=field_name,
                        clarification_default=path if os.path.isdir(path) else None,
                    )

        if selected_path and not os.path.lexists(path):
            return None, ExecutionResult.fail(
                ExecutionStatus.CLARIFICATION_REQUIRED,
                f"The selected destination '{path}' no longer exists. Choose "
                "another destination and try again.",
            )

        if os.path.lexists(path) and not os.path.isdir(path):
            return None, ExecutionResult.fail(
                ExecutionStatus.CLARIFICATION_REQUIRED,
                f"The destination '{path}' already exists as a file. Choose "
                "a different destination and try again.",
            )

        parent = path if os.path.isdir(path) else os.path.dirname(path)
        if not os.path.isdir(parent):
            return None, ExecutionResult.fail(
                ExecutionStatus.CLARIFICATION_REQUIRED,
                f"The destination folder '{parent}' isn't available under the "
                f"top Explorer folder '{workspace}'. Open the correct folder "
                "and try again.",
            )
        return path, None

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
        requires_approval: bool,
    ) -> ExecutionResult:
        """Approve and move one file or folder to the Recycle Bin.

        Args:
            kind - Expected target type, either file or folder.
            path - Resolved path of the item to delete.
            requires_approval - Whether WinPilot approval is required.

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
            requires_approval,
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
            return ExecutionResult.ok(
                f"Moved {kind} to the Recycle Bin: {target}"
            )

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

    def execute(
        self,
        command: Command,
        clarified_path: str | None = None,
        clarified_field: str | None = None,
    ) -> ExecutionResult:
        """Validate and execute a command, returning its outcome.

        Args:
            command - Command object to execute.
            clarified_path - Absolute path selected in a clarification response.
            clarified_field - Command field resolved by the selected path.

        Returns:
            Structured result describing success or failure.
        """
        if not isinstance(command, Command):
            return self._fail(
                ExecutionStatus.INVALID_ARGUMENT,
                "Invalid command: expected a Command object.",
            )

        if not isinstance(command.requires_approval, bool):
            return self._fail(
                ExecutionStatus.INVALID_ARGUMENT,
                "Command approval flag must be a Boolean.",
            )

        requires_approval = command.requires_approval
        payload = command.to_payload()

        action = payload.get("action")
        target = payload.get("target")

        if not isinstance(action, str):
            return self._fail(
                ExecutionStatus.INVALID_ARGUMENT,
                "Invalid command: missing action.",
            )

        requires_approval = (
            requires_approval or action in APPROVAL_REQUIRED_ACTIONS
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
                    requires_approval,
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

                workspace, explorer_folders, clarification = self._workspace_context()
                if clarification:
                    return clarification
                assert workspace is not None
                path, clarification = self._resolve_destination(
                    target,
                    workspace,
                    explorer_folders,
                    field_name="name",
                    check_duplicates=False,
                )
                if clarification:
                    return clarification
                assert path is not None
                item = "file" if action == "create_file" else "folder"
                denied = self._request_approval(
                    f"create a {item}",
                    path,
                    requires_approval,
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
                source = payload.get("source")
                destination = payload.get("destination")

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

                workspace, explorer_folders, clarification = self._workspace_context()
                if clarification:
                    return clarification
                assert workspace is not None
                source_path, clarification = self._resolve_existing_target(
                    source,
                    workspace,
                    explorer_folders,
                    field_name="source",
                    clarified_path=clarified_path,
                    clarified_field=clarified_field,
                )
                if clarification:
                    return clarification
                destination_path, clarification = self._resolve_destination(
                    destination,
                    workspace,
                    explorer_folders,
                    field_name="destination",
                    clarified_path=clarified_path,
                    clarified_field=clarified_field,
                )
                if clarification:
                    return clarification
                assert source_path is not None and destination_path is not None
                denied = self._request_approval(
                    action,
                    f"From: {source_path}\nTo: {destination_path}",
                    requires_approval,
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

                kind = "file" if action == "delete_file" else "folder"
                workspace, explorer_folders, clarification = self._workspace_context()
                if clarification:
                    return clarification
                assert workspace is not None
                path, clarification = self._resolve_existing_target(
                    target,
                    workspace,
                    explorer_folders,
                    expected_kind=kind,
                    field_name="target",
                    clarified_path=clarified_path,
                    clarified_field=clarified_field,
                )
                if clarification:
                    return clarification
                assert path is not None

                return self._delete_item(kind, path, requires_approval)

            if action in {"rename_file", "rename_folder"}:
                new_name = payload.get("new_name")

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

                kind = "folder" if action == "rename_folder" else "file"
                workspace, explorer_folders, clarification = self._workspace_context()
                if clarification:
                    return clarification
                assert workspace is not None
                path, clarification = self._resolve_existing_target(
                    target,
                    workspace,
                    explorer_folders,
                    expected_kind=kind,
                    field_name="path",
                    clarified_path=clarified_path,
                    clarified_field=clarified_field,
                )
                if clarification:
                    return clarification
                assert path is not None
                kind = "folder" if action == "rename_folder" else "file"
                destination_path, clarification = self._resolve_destination(
                    new_name,
                    workspace,
                    explorer_folders,
                    base_folder=os.path.dirname(path),
                    field_name="new_name",
                    clarified_path=clarified_path,
                    clarified_field=clarified_field,
                )
                if clarification:
                    return clarification
                assert destination_path is not None
                denied = self._request_approval(
                    f"rename a {kind}",
                    f"From: {path}\nTo: {destination_path}",
                    requires_approval,
                )
                if denied:
                    return denied

                if not self.folder_utils.rename(path, destination_path):
                    return self._fail(
                        ExecutionStatus.OPERATION_FAILED,
                        f"Could not rename {path} to {destination_path}.",
                    )

                return ExecutionResult.ok(
                    f"Renamed {path} to {destination_path}"
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
