import logging
import os
from typing import Any, Literal

from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent

from core.command import Command
from core.executor import ExecutionEngine
from core.execution_result import ExecutionResult, ExecutionStatus
from core.logging_config import setup_logging
from plugins.windows_stats import collect_windows_stats, format_windows_stats

setup_logging("mcp_server")

logger = logging.getLogger(__name__)

mcp = MCPServer("WinPilot")
executor = ExecutionEngine()


def execute_action(
    tool_name: str,
    command: Command,
    clarified_path: str | None = None,
    clarified_field: str | None = None,
) -> ExecutionResult:
    """Execute an action and return its structured result.

    Args:
        tool_name - Name of the MCP tool being executed.
        command - Command object, including its approval policy.
        clarified_path - Absolute path selected by the user.
        clarified_field - Command argument resolved by the selected path.

    Returns:
        Structured execution result.
    """
    logger.info("Executing tool: %s", tool_name)
    logger.debug("Action payload: %r", command)

    try:
        import pythoncom
        pythoncom.CoInitialize()
        try:
            result = executor.execute(command, clarified_path, clarified_field)
        finally:
            pythoncom.CoUninitialize()
    except Exception:
        logger.exception(
            "Unexpected error executing tool: %s",
            tool_name,
        )
        return ExecutionResult.fail(
            status=ExecutionStatus.INTERNAL_ERROR,
            message=f"{tool_name} failed because of an internal error.",
        )

    if not result.success:
        logger.warning(
            "Tool execution failed: %s [%s]: %s",
            tool_name,
            result.status.value,
            result.message,
        )
        return result

    logger.info("Tool completed successfully: %s", tool_name)
    return result


def tool_result(
    message: str,
    is_error: bool = False,
    status: ExecutionStatus | None = None,
    clarification_choices: tuple[str, ...] = (),
    clarification_field: str | None = None,
    clarification_default: str | None = None,
) -> CallToolResult:
    """Create an MCP response with text and domain outcome details.

    Args:
        message - Text presented to the MCP client.
        is_error - Whether the MCP protocol result represents failure.
        status - WinPilot status associated with the operation.
        clarification_choices - Absolute paths available for selection.
        clarification_field - Tool argument to replace with the selection.
        clarification_default - Topmost match, when available.

    Returns:
        MCP result with text, error flag, and structured status.
    """
    if status is None:
        status = (
            ExecutionStatus.OPERATION_FAILED
            if is_error
            else ExecutionStatus.SUCCESS
        )

    structured_content = {
        "success": not is_error,
        "status": status.value,
        "message": message,
    }
    if clarification_choices and clarification_field:
        structured_content["clarification_choices"] = list(clarification_choices)
        structured_content["clarification_field"] = clarification_field
        if clarification_default:
            structured_content["clarification_default"] = clarification_default

    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        is_error=is_error,
        structuredContent=structured_content,
    )


def run_tool(
    tool_name: str,
    action: dict[str, Any],
    success_message: str,
    requires_approval: bool = False,
    clarified_path: str | None = None,
    clarified_field: str | None = None,
) -> CallToolResult:
    """Execute a tool and convert its result into an MCP response.

    Args:
        tool_name - Name of the MCP tool being executed.
        action - Action payload passed to the execution engine.
        success_message - Text returned when execution succeeds.
        requires_approval - Whether WinPilot approval is required.
        clarified_path - Absolute path selected by the user.
        clarified_field - Tool argument resolved by the selected path.

    Returns:
        MCP result describing the operation outcome.
    """
    command = Command(
        action=action["action"],
        arguments={key: value for key, value in action.items() if key != "action"},
        requires_approval=requires_approval,
    )
    result = execute_action(
        tool_name,
        command,
        clarified_path,
        clarified_field,
    )

    if not result.success:
        return tool_result(
            result.message,
            is_error=True,
            status=result.status,
            clarification_choices=result.clarification_choices,
            clarification_field=result.clarification_field,
            clarification_default=result.clarification_default,
        )

    return tool_result(
        success_message,
        status=result.status,
    )


@mcp.tool()
def open_application(application: str) -> CallToolResult:
    """Open an installed Windows application by its name or a popular alias.

    Args:
        application - Name of the application to open.

    Returns:
        MCP result describing the launch outcome.
    """
    return run_tool(
        "open_application",
        {
            "action": "open_application",
            "target": application,
        },
        f"Opened {application}",
    )


@mcp.tool()
def list_available_applications() -> CallToolResult:
    """List applications currently discovered from this Windows system."""
    import pythoncom
    pythoncom.CoInitialize()
    try:
        applications = executor.windows_utils.list_available_applications()
    finally:
        pythoncom.CoUninitialize()
    if not applications:
        return tool_result("No installed applications were discovered.")
    names = "\n".join(f"- {application['name']}" for application in applications)
    return tool_result(f"Applications currently available to open:\n{names}")


@mcp.tool()
def create_file(
    name: str,
    clarified_path: str | None = None,
    clarified_field: str | None = None,
) -> CallToolResult:
    """Create a file in the current WinPilot workspace.

    Args:
        name - File name or path to create.
        clarified_path - Absolute path selected by the user.
        clarified_field - Tool argument resolved by the selected path.

    Returns:
        MCP result describing the creation outcome.
    """
    return run_tool(
        "create_file",
        {
            "action": "create_file",
            "target": name,
        },
        f"Created file: {name}",
        clarified_path=clarified_path,
        clarified_field=clarified_field,
    )


@mcp.tool()
def create_folder(
    name: str,
    clarified_path: str | None = None,
    clarified_field: str | None = None,
) -> CallToolResult:
    """Create a folder in the current WinPilot workspace.

    Args:
        name - Folder name or path to create.
        clarified_path - Absolute path selected by the user.
        clarified_field - Tool argument resolved by the selected path.

    Returns:
        MCP result describing the creation outcome.
    """
    return run_tool(
        "create_folder",
        {
            "action": "create_folder",
            "target": name,
        },
        f"Created folder: {name}",
        clarified_path=clarified_path,
        clarified_field=clarified_field,
    )


@mcp.tool()
def copy_item(
    source: str,
    destination: str,
    clarified_path: str | None = None,
    clarified_field: str | None = None,
) -> CallToolResult:
    """Copy a file or folder to a destination.

    Args:
        source - File or folder to copy.
        destination - Path for the new copy.
        clarified_path - Absolute path selected by the user.
        clarified_field - Tool argument resolved by the selected path.

    Returns:
        MCP result describing the copy outcome.
    """
    return run_tool(
        "copy_item",
        {
            "action": "copy",
            "source": source,
            "destination": destination,
        },
        f"Copied {source} to {destination}",
        clarified_path=clarified_path,
        clarified_field=clarified_field,
    )
    

@mcp.tool()
def get_system_stats() -> CallToolResult:
    """Show current CPU, memory, and fixed-disk usage on Windows."""
    try:
        stats = collect_windows_stats()
    except (OSError, RuntimeError) as error:
        logger.error("Could not retrieve Windows system statistics: %s", error)
        return tool_result(
            str(error),
            is_error=True,
            status=ExecutionStatus.OPERATION_FAILED,
        )

    return tool_result(format_windows_stats(stats))
    

@mcp.tool()
def move_item(
    source: str,
    destination: str,
    clarified_path: str | None = None,
    clarified_field: str | None = None,
) -> CallToolResult:
    """Move a file or folder to a destination.

    Args:
        source - File or folder to move.
        destination - Path to move the item to.
        clarified_path - Absolute path selected by the user.
        clarified_field - Tool argument resolved by the selected path.

    Returns:
        MCP result describing the move outcome.
    """
    return run_tool(
        "move_item",
        {
            "action": "move",
            "source": source,
            "destination": destination,
        },
        f"Moved {source} to {destination}",
        clarified_path=clarified_path,
        clarified_field=clarified_field,
    )


@mcp.tool()
def delete_item(
    target: str,
    kind: Literal["file", "folder"],
    clarified_path: str | None = None,
    clarified_field: str | None = None,
) -> CallToolResult:
    """Move a file or folder to the Recycle Bin after approval.

    Args:
        target - File or folder to delete.
        kind - Expected target type, either file or folder.
        clarified_path - Absolute path selected by the user.
        clarified_field - Tool argument resolved by the selected path.

    Returns:
        MCP result describing the deletion outcome.
    """
    if kind not in {"file", "folder"}:
        return tool_result(
            "Invalid deletion type. Expected 'file' or 'folder'.",
            is_error=True,
            status=ExecutionStatus.INVALID_ARGUMENT,
        )

    return run_tool(
        "delete_item",
        {
            "action": f"delete_{kind}",
            "target": target,
        },
        f"Moved {kind} to the Recycle Bin: {target}",
        requires_approval=True,
        clarified_path=clarified_path,
        clarified_field=clarified_field,
    )


@mcp.tool()
def rename_item(
    path: str,
    new_name: str,
    kind: Literal["file", "folder"],
    clarified_path: str | None = None,
    clarified_field: str | None = None,
) -> CallToolResult:
    """Rename a file or folder.

    Args:
        path - Existing file or folder to rename.
        new_name - New name or destination path.
        kind - Expected target type, either file or folder.
        clarified_path - Absolute path selected by the user.
        clarified_field - Tool argument resolved by the selected path.

    Returns:
        MCP result describing the rename outcome.
    """
    if kind not in {"file", "folder"}:
        return tool_result(
            "Invalid rename type. Expected 'file' or 'folder'.",
            is_error=True,
            status=ExecutionStatus.INVALID_ARGUMENT,
        )

    return run_tool(
        "rename_item",
        {
            "action": f"rename_{kind}",
            "target": path,
            "new_name": new_name,
        },
        f"Renamed {path} to {new_name}",
        clarified_path=clarified_path,
        clarified_field=clarified_field,
    )


@mcp.tool()
def write_to_application(
    text: str,
    application: str | None = None,
) -> CallToolResult:
    """Type or insert text into an application window (such as Notepad) or the active window.

    Args:
        text - Text to write or type. If 'my name', resolves to the user's name.
        application - Name of target application to write to, e.g. 'notepad'.

    Returns:
        MCP result describing the outcome.
    """
    if not isinstance(text, str) or not text.strip():
        return tool_result("Text content is missing.", is_error=True, status=ExecutionStatus.INVALID_ARGUMENT)

    if text.strip().lower() in {"my name", "myname"}:
        text = os.environ.get("USERNAME", "Ashish")

    try:
        import win32gui
        import win32con
        import win32api
        import win32clipboard
        import ctypes
        import time

        user32 = ctypes.windll.user32
        target_hwnd = None
        app_name = application.strip().lower() if application else None

        if app_name:
            def cb(hwnd, _):
                nonlocal target_hwnd
                if win32gui.IsWindowVisible(hwnd) and not win32gui.IsIconic(hwnd):
                    t = win32gui.GetWindowText(hwnd).lower()
                    c = win32gui.GetClassName(hwnd).lower()
                    if app_name in t or app_name in c:
                        target_hwnd = hwnd
                        return False
                return True
            win32gui.EnumWindows(cb, None)

            if not target_hwnd:
                executor.windows_utils.open_application(app_name)
                for _ in range(25):
                    time.sleep(0.3)
                    win32gui.EnumWindows(cb, None)
                    if target_hwnd:
                        break

            if not target_hwnd:
                return tool_result(
                    f"Could not find or launch application: {application}",
                    is_error=True,
                    status=ExecutionStatus.TARGET_NOT_FOUND,
                )

            user32.ShowWindow(target_hwnd, 9)
            user32.SwitchToThisWindow(target_hwnd, True)
            user32.SetForegroundWindow(target_hwnd)
            time.sleep(0.2)
        else:
            target_hwnd = win32gui.GetForegroundWindow()

        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
        finally:
            win32clipboard.CloseClipboard()

        time.sleep(0.05)
        win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
        win32api.keybd_event(ord('V'), 0, 0, 0)
        time.sleep(0.05)
        win32api.keybd_event(ord('V'), 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)

        dest = application if application else "active window"
        return tool_result(f"Wrote to {dest}: '{text}'")

    except Exception:
        logger.exception("Failed to write to application: %s", application)
        return tool_result(
            f"Failed to write to {application or 'window'}.",
            is_error=True,
            status=ExecutionStatus.INTERNAL_ERROR,
        )


if __name__ == "__main__":
    logger.info("Starting WinPilot MCP server")

    try:
        mcp.run(transport="stdio")
    except Exception:
        logger.exception("WinPilot MCP server terminated unexpectedly")
        raise
    finally:
        logger.info("WinPilot MCP server stopped")
