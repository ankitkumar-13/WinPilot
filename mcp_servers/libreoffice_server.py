"""MCP server exposing LibreOffice document operations via UNO."""

import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Literal

LO_PROGRAM_PATHS = [
    r"C:\Program Files\LibreOffice\program",
    r"C:\Program Files (x86)\LibreOffice\program",
]
for lo_path in LO_PROGRAM_PATHS:
    if os.path.isdir(lo_path):
        if lo_path not in sys.path:
            sys.path.append(lo_path)
        os.environ["PATH"] = lo_path + ";" + os.environ.get("PATH", "")
        break

from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent

from core.execution_result import ExecutionStatus
from core.logging_config import setup_logging
from plugins.libreoffice_context import LibreOfficeContext

setup_logging("libreoffice_mcp_server")

logger = logging.getLogger(__name__)

mcp = MCPServer("WinPilot-LibreOffice")
context = LibreOfficeContext()

COMPONENT_URL_MAP: dict[str, str] = {
    "writer": "private:factory/swriter",
    "calc": "private:factory/scalc",
    "impress": "private:factory/simpress",
    "draw": "private:factory/sdraw",
    "base": "private:factory/sdatabase",
    "math": "private:factory/smath",
}

SOFFICE_PATHS: list[str] = [
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
]


def _tool_result(
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

    structured_content = {
        "success": not is_error,
        "status": status.value,
        "message": message,
    }

    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        is_error=is_error,
        structuredContent=structured_content,
    )


def _find_soffice() -> str | None:
    """Locate the LibreOffice soffice executable.

    Returns:
        Absolute path to soffice.exe, or None when not found.
    """
    for path in SOFFICE_PATHS:
        if os.path.isfile(path):
            return path

    env_path = os.environ.get("LIBREOFFICE_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    return None


def _ensure_libreoffice_running() -> bool:
    """Start LibreOffice in listening mode if not already running.

    Returns:
        True when LibreOffice is running or was started successfully.
    """
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-Process soffice* -ErrorAction SilentlyContinue "
                "| Select-Object -First 1 | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        if result.stdout.strip() and result.stdout.strip() != "null":
            return True

    except Exception:
        logger.debug("Could not check LibreOffice process", exc_info=True)

    soffice = _find_soffice()
    if soffice is None:
        logger.error("LibreOffice executable not found")
        return False

    try:
        subprocess.Popen(
            [
                soffice,
                "--accept=socket,host=localhost,port=2002;urp;",
            ],
        )
        time.sleep(2)
        return True

    except Exception:
        logger.exception("Failed to start LibreOffice")
        return False


def _is_uno_port_open() -> bool:
    """Check if the LibreOffice UNO socket port 2002 is open."""
    import socket
    try:
        with socket.create_connection(("127.0.0.1", 2002), timeout=0.1):
            return True
    except OSError:
        return False


def _find_lo_python() -> str | None:
    """Locate LibreOffice's internal python executable."""
    lo_pythons = [
        r"C:\Program Files\LibreOffice\program\python.exe",
        r"C:\Program Files (x86)\LibreOffice\program\python.exe",
    ]
    for p in lo_pythons:
        if os.path.isfile(p):
            return p
    soffice = _find_soffice()
    if soffice:
        cand = os.path.join(os.path.dirname(soffice), "python.exe")
        if os.path.isfile(cand):
            return cand
    return None


def _run_lo_uno(command_dict: dict) -> dict:
    """Execute a UNO command via LibreOffice's python bridge."""
    if not _is_uno_port_open():
        return {"success": False, "error": "UNO port 2002 is not open"}
    lo_python = _find_lo_python()
    if not lo_python:
        return {"success": False, "error": "LibreOffice python executable not found"}

    runner_script = '''
import uno, sys, json, os
from com.sun.star.beans import PropertyValue

try:
    cmd = json.loads(sys.argv[1])
    action = cmd.get("action")

    local_context = uno.getComponentContext()
    resolver = local_context.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_context
    )
    ctx = resolver.resolve("uno:socket,host=127.0.0.1,port=2002;urp;StarOffice.ComponentContext")
    smgr = ctx.ServiceManager
    desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
    doc = desktop.getCurrentComponent()

    if action == "insert_text":
        text = cmd.get("text", "")
        if not doc:
            doc = desktop.loadComponentFromURL("private:factory/swriter", "_blank", 0, ())
        if hasattr(doc, "getText"):
            t = doc.getText()
            c = t.createTextCursor()
            c.gotoEnd(False)
            t.insertString(c, text + "\\n", False)
            print(json.dumps({"success": True, "title": doc.getTitle()}))
        elif hasattr(doc, "getSheets"):
            sheet = doc.getCurrentController().getActiveSheet()
            cell_obj = sheet.getCellRangeByName("A1")
            cell_obj.setString(text)
            print(json.dumps({"success": True, "title": doc.getTitle()}))
        else:
            print(json.dumps({"success": False, "error": "No suitable active document"}))

    elif action == "save_document":
        if not doc:
            print(json.dumps({"success": False, "error": "No active document to save"}))
        else:
            file_path = cmd.get("file_path")
            desktop_dir = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop")
            if not os.path.isdir(desktop_dir):
                desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")

            ext = ".odt" if hasattr(doc, "getText") else (".ods" if hasattr(doc, "getSheets") else ".odp" if hasattr(doc, "getPresentation") else ".odt")

            if file_path:
                clean_path = str(file_path).strip().strip('"').strip("'")
                if not os.path.isabs(clean_path):
                    abs_path = os.path.abspath(os.path.join(desktop_dir, clean_path))
                else:
                    abs_path = os.path.abspath(clean_path)

                if not os.path.splitext(abs_path)[1]:
                    abs_path += ext

                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                file_url = uno.systemPathToFileUrl(abs_path)
                doc.storeAsURL(file_url, ())
                print(json.dumps({"success": True, "path": abs_path, "title": doc.getTitle()}))
            else:
                if doc.hasLocation() and not doc.isReadonly():
                    doc.store()
                    print(json.dumps({"success": True, "title": doc.getTitle()}))
                else:
                    title = doc.getTitle() if hasattr(doc, "getTitle") else "document"
                    invalid_chars = '<>:"/\\|?*'
                    clean_title = "".join(c for c in title if c not in invalid_chars)
                    if not clean_title.lower().endswith(ext):
                        clean_title += ext
                    abs_path = os.path.abspath(os.path.join(desktop_dir, clean_title))
                    file_url = uno.systemPathToFileUrl(abs_path)
                    doc.storeAsURL(file_url, ())
                    print(json.dumps({"success": True, "path": abs_path, "title": doc.getTitle()}))

    elif action == "export_pdf":
        if not doc:
            print(json.dumps({"success": False, "error": "No active document to export"}))
        else:
            output_path = cmd.get("output_path")
            abs_path = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            pdf_url = uno.systemPathToFileUrl(abs_path)
            prop = PropertyValue()
            prop.Name = "FilterName"
            prop.Value = "writer_pdf_Export"
            doc.storeToURL(pdf_url, (prop,))
            print(json.dumps({"success": True, "path": abs_path, "title": doc.getTitle()}))

    elif action == "new_document":
        comp = cmd.get("component", "writer")
        factories = {
            "writer": "private:factory/swriter",
            "calc": "private:factory/scalc",
            "impress": "private:factory/simpress",
            "draw": "private:factory/sdraw",
        }
        url = factories.get(comp, "private:factory/swriter")
        new_doc = desktop.loadComponentFromURL(url, "_blank", 0, ())
        print(json.dumps({"success": True, "title": new_doc.getTitle() if new_doc else ""}))

    elif action == "set_cell":
        if not doc or not hasattr(doc, "getSheets"):
            print(json.dumps({"success": False, "error": "No active Calc spreadsheet"}))
        else:
            sheet_name = cmd.get("sheet")
            cell_ref = cmd.get("cell", "A1")
            value = cmd.get("value", "")
            sheets = doc.getSheets()
            sheet = sheets.getByName(sheet_name) if sheet_name and sheets.hasByName(sheet_name) else doc.getCurrentController().getActiveSheet()
            cell_obj = sheet.getCellRangeByName(cell_ref)
            try:
                num = float(value)
                cell_obj.setValue(num)
            except ValueError:
                cell_obj.setString(value)
            print(json.dumps({"success": True, "cell": cell_ref, "value": value}))

    elif action == "get_cell":
        if not doc or not hasattr(doc, "getSheets"):
            print(json.dumps({"success": False, "error": "No active Calc spreadsheet"}))
        else:
            sheet_name = cmd.get("sheet")
            cell_ref = cmd.get("cell", "A1")
            sheets = doc.getSheets()
            sheet = sheets.getByName(sheet_name) if sheet_name and sheets.hasByName(sheet_name) else doc.getCurrentController().getActiveSheet()
            cell_obj = sheet.getCellRangeByName(cell_ref)
            val = cell_obj.getString()
            print(json.dumps({"success": True, "cell": cell_ref, "value": val}))

    elif action == "list_sheets":
        if not doc or not hasattr(doc, "getSheets"):
            print(json.dumps({"success": False, "error": "No active Calc spreadsheet"}))
        else:
            names = list(doc.getSheets().getElementNames())
            print(json.dumps({"success": True, "sheets": names}))

    elif action == "get_component_info":
        if not doc:
            print(json.dumps({"success": False, "error": "No document active"}))
        else:
            title = doc.getTitle() if hasattr(doc, "getTitle") else ""
            comp_type = "writer" if hasattr(doc, "getText") else ("calc" if hasattr(doc, "getSheets") else "unknown")
            print(json.dumps({"success": True, "title": title, "component": comp_type}))

    else:
        print(json.dumps({"success": False, "error": f"Unknown action {action}"}))

except Exception as e:
    import traceback
    print(json.dumps({"success": False, "error": str(e), "trace": traceback.format_exc()}))
'''

    try:
        proc = subprocess.run(
            [lo_python, "-c", runner_script, json.dumps(command_dict)],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        out = proc.stdout.strip()
        if not out:
            return {"success": False, "error": proc.stderr or "Empty response from UNO"}
        start_idx = out.find("{")
        end_idx = out.rfind("}")
        if start_idx != -1 and end_idx != -1:
            return json.loads(out[start_idx : end_idx + 1])
        return {"success": False, "error": out}
    except Exception as e:
        logger.debug("UNO bridge call failed: %s", e)
        return {"success": False, "error": str(e)}


def _force_foreground_window(hwnd: int) -> bool:
    """Safely bring a window to the foreground without stealing from unintended targets."""
    import ctypes
    import win32gui
    import win32con
    import win32process

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    else:
        win32gui.ShowWindow(hwnd, win32con.SW_SHOW)

    try:
        user32.LockSetForegroundWindow(2)  # LSFW_UNLOCK
        user32.AllowSetForegroundWindow(0xFFFFFFFF)  # ASFW_ANY
    except Exception:
        pass

    fg_hwnd = user32.GetForegroundWindow()
    if fg_hwnd == hwnd:
        return True

    cur_tid = kernel32.GetCurrentThreadId()
    fg_tid, _ = win32process.GetWindowThreadProcessId(fg_hwnd)
    target_tid, _ = win32process.GetWindowThreadProcessId(hwnd)

    attached_fg = False
    attached_target = False
    try:
        if fg_tid != 0 and fg_tid != cur_tid:
            attached_fg = bool(user32.AttachThreadInput(cur_tid, fg_tid, True))
        if target_tid != 0 and target_tid != cur_tid:
            attached_target = bool(user32.AttachThreadInput(cur_tid, target_tid, True))

        # Alt key trick to bypass Windows foreground lock
        user32.keybd_event(0x12, 0, 0, 0)
        user32.keybd_event(0x12, 0, 2, 0)

        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        user32.SetFocus(hwnd)
    finally:
        if attached_fg:
            user32.AttachThreadInput(cur_tid, fg_tid, False)
        if attached_target:
            user32.AttachThreadInput(cur_tid, target_tid, False)

    time.sleep(0.15)
    return user32.GetForegroundWindow() == hwnd


def _get_desktop() -> Any:
    """Connect to the running LibreOffice instance via UNO."""
    # Retained for backward compatibility
    return None


def _open_via_command(
    soffice: str,
    component: str | None,
    file_path: str | None = None,
) -> bool:
    """Open a document or new file using the soffice command line.

    Args:
        soffice - Path to the soffice executable.
        component - LibreOffice component identifier.
        file_path - Path to a file to open.

    Returns:
        True when the command was launched successfully.
    """
    args = [soffice]

    if file_path:
        args.append(file_path)
    elif component:
        flag_map: dict[str, str] = {
            "writer": "--writer",
            "calc": "--calc",
            "impress": "--impress",
            "draw": "--draw",
            "base": "--base",
            "math": "--math",
        }
        flag = flag_map.get(component)
        if flag:
            args.append(flag)

    try:
        subprocess.Popen(args)
        return True

    except Exception:
        logger.exception("Failed to launch LibreOffice via command line")
        return False


@mcp.tool()
def libreoffice_detect_foreground() -> CallToolResult:
    """Detect the active LibreOffice component in the foreground.

    Returns:
        MCP result describing the foreground LibreOffice component.
    """
    info = context.get_foreground_component()

    if info is None:
        return _tool_result(
            "No LibreOffice window is in the foreground.",
            is_error=True,
            status=ExecutionStatus.TARGET_NOT_FOUND,
        )

    component = info.get("component", "unknown")
    title = info.get("title", "")

    return _tool_result(
        f"LibreOffice {component.title()} is in the foreground: {title}"
    )


@mcp.tool()
def libreoffice_get_commands(
    component: str | None = None,
) -> CallToolResult:
    """List commands available for a LibreOffice component.

    Args:
        component - Component identifier. Auto-detected when omitted.

    Returns:
        MCP result listing available commands.
    """
    commands = context.get_app_commands(component)

    if not commands:
        return _tool_result(
            "No LibreOffice commands available. "
            "Bring a LibreOffice window to the foreground first.",
            is_error=True,
            status=ExecutionStatus.TARGET_NOT_FOUND,
        )

    lines = [
        f"- {cmd['name']}: {cmd['description']}"
        for cmd in commands
    ]

    return _tool_result(
        f"Available commands:\n" + "\n".join(lines)
    )


@mcp.tool()
def libreoffice_new_document(
    component: Literal[
        "writer", "calc", "impress", "draw", "base", "math"
    ],
) -> CallToolResult:
    """Create a new blank document in the specified LibreOffice component.

    Use this when the user explicitly asks for a new document, blank document,
    or new spreadsheet/presentation. Do NOT use this tool if the user is asking
    to type or write content; use libreoffice_insert_text instead.

    Args:
        component - LibreOffice component to create a document in.

    Returns:
        MCP result describing the creation outcome.
    """
    desktop = _get_desktop()

    if desktop is not None:
        try:
            import uno
            from com.sun.star.beans import PropertyValue

            url = COMPONENT_URL_MAP.get(component)
            if url is None:
                return _tool_result(
                    f"Unknown component: {component}",
                    is_error=True,
                    status=ExecutionStatus.INVALID_ARGUMENT,
                )

            desktop.loadComponentFromURL(url, "_blank", 0, ())
            return _tool_result(
                f"Created a new {component.title()} document."
            )

        except Exception:
            logger.debug("UNO new document failed", exc_info=True)

    soffice = _find_soffice()
    if soffice is None:
        return _tool_result(
            "LibreOffice is not installed or could not be found.",
            is_error=True,
            status=ExecutionStatus.OPERATION_FAILED,
        )

    if _open_via_command(soffice, component):
        return _tool_result(
            f"Created a new {component.title()} document."
        )

    return _tool_result(
        f"Failed to create a new {component.title()} document.",
        is_error=True,
        status=ExecutionStatus.OPERATION_FAILED,
    )


@mcp.tool()
def libreoffice_open_document(file_path: str) -> CallToolResult:
    """Open a document in the appropriate LibreOffice component.

    Args:
        file_path - Path to the document to open.

    Returns:
        MCP result describing the open outcome.
    """
    if not isinstance(file_path, str) or not file_path.strip():
        return _tool_result(
            "File path is missing.",
            is_error=True,
            status=ExecutionStatus.INVALID_ARGUMENT,
        )

    file_path = file_path.strip()

    if not os.path.isfile(file_path):
        return _tool_result(
            f"File not found: {file_path}",
            is_error=True,
            status=ExecutionStatus.TARGET_NOT_FOUND,
        )

    desktop = _get_desktop()

    if desktop is not None:
        try:
            import uno

            url = uno.systemPathToFileUrl(os.path.abspath(file_path))
            desktop.loadComponentFromURL(url, "_blank", 0, ())
            return _tool_result(f"Opened document: {file_path}")

        except Exception:
            logger.debug("UNO open document failed", exc_info=True)

    soffice = _find_soffice()
    if soffice is None:
        return _tool_result(
            "LibreOffice is not installed or could not be found.",
            is_error=True,
            status=ExecutionStatus.OPERATION_FAILED,
        )

    if _open_via_command(soffice, None, file_path):
        return _tool_result(f"Opened document: {file_path}")

    return _tool_result(
        f"Failed to open document: {file_path}",
        is_error=True,
        status=ExecutionStatus.OPERATION_FAILED,
    )


@mcp.tool()
def libreoffice_save_document(file_path: str | None = None) -> CallToolResult:
    """Save the currently active LibreOffice document.

    Args:
        file_path - Optional destination file path to save as. If omitted,
            triggers standard Save (Ctrl+S).

    Returns:
        MCP result describing the save outcome.
    """
    # 1. Attempt native UNO save if port is open
    if _is_uno_port_open():
        res = _run_lo_uno({"action": "save_document", "file_path": file_path})
        if res.get("success"):
            saved_to = f" to: {res['path']}" if res.get("path") else f" ({res.get('title', 'document')})"
            return _tool_result(f"Saved document{saved_to}")

    # 2. Fallback: find the LibreOffice window and send Ctrl+S
    try:
        import pythoncom
        pythoncom.CoInitialize()
        try:
            import win32gui
            import win32con
            import win32api
            import win32clipboard
            import time

            lo_hwnd = None
            def enum_cb(hwnd, _):
                nonlocal lo_hwnd
                if win32gui.IsWindowVisible(hwnd) and not win32gui.IsIconic(hwnd):
                    title = win32gui.GetWindowText(hwnd).lower()
                    cls = win32gui.GetClassName(hwnd)
                    if cls == "SALFRAME" and ("libreoffice" in title or "writer" in title or "calc" in title or "impress" in title or "untitled" in title):
                        lo_hwnd = hwnd
                        return False
                return True
            win32gui.EnumWindows(enum_cb, None)

            if not lo_hwnd:
                return _tool_result(
                    "No open LibreOffice document window was found to save.",
                    is_error=True,
                    status=ExecutionStatus.TARGET_NOT_FOUND,
                )

            # Restore and bring window to front safely
            _force_foreground_window(lo_hwnd)
            time.sleep(0.3)

            # Check if LibreOffice is actually foreground
            if win32gui.GetForegroundWindow() != lo_hwnd:
                return _tool_result(
                    "Could not focus LibreOffice window to perform save.",
                    is_error=True,
                    status=ExecutionStatus.OPERATION_FAILED,
                )

            # Send Ctrl+S
            win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
            win32api.keybd_event(ord('S'), 0, 0, 0)
            time.sleep(0.05)
            win32api.keybd_event(ord('S'), 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)

            # If a specific file path was supplied, paste into the Save As dialog
            if file_path and isinstance(file_path, str) and file_path.strip():
                abs_path = os.path.abspath(file_path.strip())
                time.sleep(0.8)
                win32clipboard.OpenClipboard()
                try:
                    win32clipboard.EmptyClipboard()
                    win32clipboard.SetClipboardText(abs_path, win32clipboard.CF_UNICODETEXT)
                finally:
                    win32clipboard.CloseClipboard()

                time.sleep(0.1)
                win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
                win32api.keybd_event(ord('V'), 0, 0, 0)
                time.sleep(0.05)
                win32api.keybd_event(ord('V'), 0, win32con.KEYEVENTF_KEYUP, 0)
                win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)

                time.sleep(0.2)
                win32api.keybd_event(win32con.VK_RETURN, 0, 0, 0)
                time.sleep(0.05)
                win32api.keybd_event(win32con.VK_RETURN, 0, win32con.KEYEVENTF_KEYUP, 0)
                return _tool_result(f"Saved document to: {abs_path}")

            return _tool_result("Saved document in LibreOffice.")
        finally:
            pythoncom.CoUninitialize()

    except Exception:
        logger.exception("Failed to save document")
        return _tool_result(
            "Failed to save the document.",
            is_error=True,
            status=ExecutionStatus.INTERNAL_ERROR,
        )


@mcp.tool()
def libreoffice_export_pdf(output_path: str) -> CallToolResult:
    """Export the active LibreOffice document as a PDF file.

    Args:
        output_path - File path for the exported PDF.

    Returns:
        MCP result describing the export outcome.
    """
    if not isinstance(output_path, str) or not output_path.strip():
        return _tool_result(
            "Output path is missing.",
            is_error=True,
            status=ExecutionStatus.INVALID_ARGUMENT,
        )

    output_path = output_path.strip()

    if _is_uno_port_open():
        res = _run_lo_uno({"action": "export_pdf", "output_path": output_path})
        if res.get("success"):
            return _tool_result(f"Exported PDF to: {res.get('path', output_path)}")

    return _tool_result(
        f"Failed to export PDF to {output_path}. Make sure LibreOffice is running.",
        is_error=True,
        status=ExecutionStatus.OPERATION_FAILED,
    )


@mcp.tool()
def libreoffice_insert_text(text: str) -> CallToolResult:
    """Type, write, or insert text into an active LibreOffice document.

    Works across any active LibreOffice document (Writer, Impress slides, Calc, Draw).
    Use this tool whenever the user wants to write or type words, text,
    or sentences into a document.

    Args:
        text - The exact text string to type or insert.

    Returns:
        MCP result describing the insertion outcome.
    """
    if not isinstance(text, str):
        return _tool_result(
            "Text content is missing.",
            is_error=True,
            status=ExecutionStatus.INVALID_ARGUMENT,
        )

    # 1. Attempt native UNO insertion first (direct document object model manipulation)
    if _is_uno_port_open():
        res = _run_lo_uno({"action": "insert_text", "text": text})
        if res.get("success"):
            target_name = res.get("title") or "document"
            return _tool_result(f"Wrote to LibreOffice ({target_name}): '{text}'")

    # 2. Fallback: paste into active LibreOffice window using Windows clipboard
    try:
        import pythoncom
        pythoncom.CoInitialize()
        try:
            import win32clipboard
            import win32con
            import win32api
            import win32gui
            import time

            # Find visible LibreOffice document window
            doc_hwnd = None
            doc_title = ""
            def enum_cb(hwnd, _):
                nonlocal doc_hwnd, doc_title
                if win32gui.IsWindowVisible(hwnd) and not win32gui.IsIconic(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    cls = win32gui.GetClassName(hwnd)
                    if cls == "SALFRAME" and (
                        "libreoffice" in title.lower()
                        or "writer" in title.lower()
                        or "impress" in title.lower()
                        or "calc" in title.lower()
                        or "draw" in title.lower()
                        or "untitled" in title.lower()
                        or ".odt" in title.lower()
                        or ".odp" in title.lower()
                        or ".ods" in title.lower()
                        or ".docx" in title.lower()
                        or ".pptx" in title.lower()
                    ):
                        doc_hwnd = hwnd
                        doc_title = title
                        return False
                    elif cls == "SALFRAME" and title:
                        doc_hwnd = hwnd
                        doc_title = title
                return True
            win32gui.EnumWindows(enum_cb, None)

            # If no LibreOffice window is open, launch Writer as default
            if not doc_hwnd:
                soffice = _find_soffice()
                if soffice:
                    args = "--writer --norestore --accept=\"socket,host=127.0.0.1,port=2002;urp;\""
                    os.startfile(soffice, arguments=args)
                else:
                    writer_lnk = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\LibreOffice\LibreOffice Writer.lnk"
                    if os.path.isfile(writer_lnk):
                        os.startfile(writer_lnk)

                for _ in range(25):
                    time.sleep(0.4)
                    if _is_uno_port_open():
                        res = _run_lo_uno({"action": "insert_text", "text": text})
                        if res.get("success"):
                            target_name = res.get("title") or "document"
                            return _tool_result(f"Wrote to LibreOffice ({target_name}): '{text}'")
                    win32gui.EnumWindows(enum_cb, None)
                    if doc_hwnd:
                        break

            if not doc_hwnd:
                return _tool_result(
                    "Could not find or launch an active LibreOffice document.",
                    is_error=True,
                    status=ExecutionStatus.TARGET_NOT_FOUND,
                )

            # Bring the document window to the front safely
            _force_foreground_window(doc_hwnd)
            time.sleep(0.2)

            # CRITICAL CHECK: Verify that doc_hwnd is actually foreground!
            fg_hwnd = win32gui.GetForegroundWindow()
            if fg_hwnd != doc_hwnd:
                logger.warning("Foreground window is %s, expected %s (%s). Keystrokes suppressed.", fg_hwnd, doc_hwnd, doc_title)
                return _tool_result(
                    f"LibreOffice document window '{doc_title}' could not be brought to the front. "
                    "Please click on the document window and try again.",
                    is_error=True,
                    status=ExecutionStatus.OPERATION_FAILED,
                )

            # Copy text to clipboard and simulate Ctrl+V
            win32clipboard.OpenClipboard()
            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
            finally:
                win32clipboard.CloseClipboard()

            time.sleep(0.05)

            # Paste via Ctrl+V and Enter
            win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
            win32api.keybd_event(ord('V'), 0, 0, 0)
            time.sleep(0.05)
            win32api.keybd_event(ord('V'), 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
            time.sleep(0.05)
            win32api.keybd_event(win32con.VK_RETURN, 0, 0, 0)
            time.sleep(0.05)
            win32api.keybd_event(win32con.VK_RETURN, 0, win32con.KEYEVENTF_KEYUP, 0)

            target_name = doc_title if doc_title else "document"
            return _tool_result(f"Wrote to LibreOffice ({target_name}): '{text}'")
        finally:
            pythoncom.CoUninitialize()

    except Exception:
        logger.exception("Failed to insert text into document")
        return _tool_result(
            "Failed to insert text. Make sure a LibreOffice document is open.",
            is_error=True,
            status=ExecutionStatus.INTERNAL_ERROR,
        )


@mcp.tool()
def libreoffice_set_cell(
    cell: str,
    value: str,
    sheet: str | None = None,
) -> CallToolResult:
    """Set a cell value in the active Calc spreadsheet.

    Args:
        cell - Cell reference such as A1 or B3.
        value - Value to place in the cell.
        sheet - Sheet name. Uses the active sheet when omitted.

    Returns:
        MCP result describing the outcome.
    """
    if not isinstance(cell, str) or not cell.strip():
        return _tool_result(
            "Cell reference is missing.",
            is_error=True,
            status=ExecutionStatus.INVALID_ARGUMENT,
        )

    if _is_uno_port_open():
        res = _run_lo_uno({"action": "set_cell", "cell": cell.strip(), "value": value, "sheet": sheet})
        if res.get("success"):
            return _tool_result(f"Set cell {cell.strip()} to '{value}'.")
        return _tool_result(f"Failed to set cell {cell}: {res.get('error')}", is_error=True)

    return _tool_result(
        "Cannot connect to LibreOffice Calc via UNO. Make sure Calc is running.",
        is_error=True,
        status=ExecutionStatus.OPERATION_FAILED,
    )


@mcp.tool()
def libreoffice_get_cell(
    cell: str,
    sheet: str | None = None,
) -> CallToolResult:
    """Read a cell value from the active Calc spreadsheet.

    Args:
        cell - Cell reference such as A1 or B3.
        sheet - Sheet name. Uses the active sheet when omitted.

    Returns:
        MCP result containing the cell value.
    """
    if not isinstance(cell, str) or not cell.strip():
        return _tool_result(
            "Cell reference is missing.",
            is_error=True,
            status=ExecutionStatus.INVALID_ARGUMENT,
        )

    if _is_uno_port_open():
        res = _run_lo_uno({"action": "get_cell", "cell": cell.strip(), "sheet": sheet})
        if res.get("success"):
            return _tool_result(f"Cell {cell.strip()} = {res.get('value', '(empty)')}")
        return _tool_result(f"Failed to read cell {cell}: {res.get('error')}", is_error=True)

    return _tool_result(
        "Cannot connect to LibreOffice Calc via UNO. Make sure Calc is running.",
        is_error=True,
        status=ExecutionStatus.OPERATION_FAILED,
    )


@mcp.tool()
def libreoffice_list_sheets() -> CallToolResult:
    """List all sheet names in the active Calc spreadsheet.

    Returns:
        MCP result listing the sheet names.
    """
    if _is_uno_port_open():
        res = _run_lo_uno({"action": "list_sheets"})
        if res.get("success"):
            sheets = res.get("sheets", [])
            return _tool_result(f"Sheets: {', '.join(sheets)}")
        return _tool_result(f"Failed to list sheets: {res.get('error')}", is_error=True)

    return _tool_result(
        "Cannot connect to LibreOffice Calc via UNO. Make sure Calc is running.",
        is_error=True,
        status=ExecutionStatus.OPERATION_FAILED,
    )


@mcp.tool()
def libreoffice_close_document() -> CallToolResult:
    """Close the currently active LibreOffice document.

    Returns:
        MCP result describing the close outcome.
    """
    if _is_uno_port_open():
        res = _run_lo_uno({"action": "close_document"})
        if res.get("success"):
            return _tool_result("Closed the active document.")

    # Fallback via Alt+F4 / Ctrl+W
    try:
        import win32api
        import win32con
        import time
        win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
        win32api.keybd_event(ord('W'), 0, 0, 0)
        time.sleep(0.05)
        win32api.keybd_event(ord('W'), 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
        return _tool_result("Closed the active document.")
    except Exception:
        logger.exception("Failed to close document")
        return _tool_result("Failed to close the document.", is_error=True, status=ExecutionStatus.INTERNAL_ERROR)


if __name__ == "__main__":
    logger.info("Starting WinPilot LibreOffice MCP server")

    try:
        mcp.run(transport="stdio")
    except Exception:
        logger.exception(
            "WinPilot LibreOffice MCP server terminated unexpectedly"
        )
        raise
    finally:
        logger.info("WinPilot LibreOffice MCP server stopped")
