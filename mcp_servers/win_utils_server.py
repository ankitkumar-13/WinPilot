from typing import Any, Literal

import httpx2
from mcp.server import MCPServer

from core.executor import ExecutionEngine

# Initialize MCPServer
mcp = MCPServer("WinPilot")



# Actual Tools

executor = ExecutionEngine()

@mcp.tool()
def open_application(application: str) -> str:
    """Open a supported windows application, such as Notepad, Calculator or Paint.
    
    Args:
        application: A valid application name on the device (e.g. notepad)
    """
    succeeded = executor.execute({
        "action": "open_application",
        "target": application,
    })

    if succeeded:
        return f"Opened {application}"

    return f"Could not open {application}. It may not be supported."

@mcp.tool()
def create_file(name: str) -> str:
    """Create a file in the current WinPilot workspace.
    
    Args:
        name: Name of the file you want to create
    """
    succeeded = executor.execute({
        "action": "create_file",
        "target": name,
    })

    if succeeded:
        return f"Created file: {name}"

    return f"Could not create file: {name}"

@mcp.tool()
def create_folder(name: str) -> str:
    """Create a folder in the current WinPilot workspace as per the internal selection.
    
    Args:
        name: Name of the folder you want to create
    """
    succeeded = executor.execute({
        "action": "create_folder",
        "target": name,
    })

    if succeeded:
        return f"Created folder: {name}"

    return f"Could not create folder: {name}"

@mcp.tool()
def copy_item(source: str, destination: str) -> str:
    """Copy a file or folder within the current WinPilot workspace as per the internal selection.
    
    Args:
        source: The name of the file or folder you want to copy
        destination: The name of the folder or the complete path, where you want the item to be copied to
    """
    succeeded = executor.execute({
        "action": "copy",
        "source": source,
        "destination": destination,
    })

    if succeeded:
        return f"Copied {source} to {destination}"

    return f"Could not copy {source} to {destination}"

@mcp.tool()
def move_item(source: str, destination: str) -> str:
    """Move a file or folder within the current WinPilot workspace as per the internal selection.
    
    Args:
            source: The name of the file or folder you want to move
            destination: The name of the folder or the complete path, where you want the item to be moved to
    """
    succeeded = executor.execute({
        "action": "move",
        "source": source,
        "destination": destination,
    })

    if succeeded:
        return f"Moved {source} to {destination}"

    return f"Could not move {source} to {destination}"


@mcp.tool()
def delete_item(target: str, kind: Literal["file", "folder"]) -> str:
    """Delete a specifically named file or folder in the current WinPilot workspace as per the internal selection.
    
    Args:
        target: It is the file or folder you want to delete
        kind: Selection of whether it is a file or a folder
    """
    succeeded = executor.execute({
        "action": f"delete_{kind}",
        "target": target,
    })

    if succeeded:
        return f"Deleted {kind}: {target}"

    return f"Could not delete {kind}: {target}"


@mcp.tool()
def rename_item(path: str, new_name: str, kind: Literal["file", "folder"]) -> str:
    """Rename a file or folder in the current WinPilot workspace as per internal selection.
    
    Args:
        path: The name of the folder/file name or the complete path (with name) of the item you want to rename
        new_name: It is the updated or new name you want to give to a file/folder
        kind: Selection of whether it is a file or a folder
    """
    succeeded = executor.execute({
        "action": f"rename_{kind}",
        "target":path,
        "new_name": new_name,
    })

    if succeeded:
        return f"Renamed {path} to {new_name}"

    return f"Could not rename {path}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
