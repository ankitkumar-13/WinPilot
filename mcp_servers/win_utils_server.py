import logging
from typing import Any, Literal

from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent

from core.command import Command
from core.executor import ExecutionEngine
from core.execution_result import ExecutionResult, ExecutionStatus
from core.logging_config import setup_logging

setup_logging("mcp_server")

logger = logging.getLogger(__name__)

mcp = MCPServer("WinPilot")
executor = ExecutionEngine()


def execute_action(
    tool_name: str,
    action: Command | dict[str, Any],
) -> ExecutionResult:
    """Execute an action and return its structured result.

    Args:
        tool_name - Name of the MCP tool being executed.
        action - Command object or action payload for the executor.

    Returns:
        Structured execution result.
    """
    logger.info("Executing tool: %s", tool_name)
    logger.debug("Action payload: %r", action)

    try:
        result = executor.execute(action)
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
) -> CallToolResult:
    """Create an MCP response with text and domain outcome details.

    Args:
        message - Text presented to the MCP client.
        is_error - Whether the MCP protocol result represents failure.
        status - WinPilot status associated with the operation.

    Returns:
        MCP result with text, error flag, and structured status.
    """
    if status is None:
        status = (
            ExecutionStatus.OPERATION_FAILED
            if is_error
            else ExecutionStatus.SUCCESS
        )

    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        is_error=is_error,
        structuredContent={
            "success": not is_error,
            "status": status.value,
            "message": message,
        },
    )


def run_tool(
    tool_name: str,
    action: dict[str, Any],
    success_message: str,
    requires_elevation: bool = False,
) -> CallToolResult:
    """Execute a tool and convert its result into an MCP response.

    Args:
        tool_name - Name of the MCP tool being executed.
        action - Action payload passed to the execution engine.
        success_message - Text returned when execution succeeds.
        requires_elevation - Whether WinPilot approval is required.

    Returns:
        MCP result describing the operation outcome.
    """
    command = Command(
        action=action["action"],
        arguments={key: value for key, value in action.items() if key != "action"},
        requires_elevation=requires_elevation,
    )
    result = execute_action(tool_name, command)

    if not result.success:
        return tool_result(
            result.message,
            is_error=True,
            status=result.status,
        )

    return tool_result(
        success_message,
        status=result.status,
    )


@mcp.tool()
def open_application(application: str) -> CallToolResult:
    """Open a supported Windows application.

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
def create_file(name: str) -> CallToolResult:
    """Create a file in the current WinPilot workspace.

    Args:
        name - File name or path to create.

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
    )


@mcp.tool()
def create_folder(name: str) -> CallToolResult:
    """Create a folder in the current WinPilot workspace.

    Args:
        name - Folder name or path to create.

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
    )


@mcp.tool()
def copy_item(
    source: str,
    destination: str,
) -> CallToolResult:
    """Copy a file or folder to a destination.

    Args:
        source - File or folder to copy.
        destination - Path for the new copy.

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
    )


@mcp.tool()
def move_item(
    source: str,
    destination: str,
) -> CallToolResult:
    """Move a file or folder to a destination.

    Args:
        source - File or folder to move.
        destination - Path to move the item to.

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
    )


@mcp.tool()
def delete_item(
    target: str,
    kind: Literal["file", "folder"],
) -> CallToolResult:
    """Delete a file or folder after WinPilot approval.

    Args:
        target - File or folder to delete.
        kind - Expected target type, either file or folder.

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
        f"Deleted {kind}: {target}",
        requires_elevation=True,
    )


@mcp.tool()
def rename_item(
    path: str,
    new_name: str,
    kind: Literal["file", "folder"],
) -> CallToolResult:
    """Rename a file or folder.

    Args:
        path - Existing file or folder to rename.
        new_name - New name or destination path.
        kind - Expected target type, either file or folder.

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
