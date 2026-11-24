"""Detect and interact with foreground LibreOffice windows."""

import logging
import subprocess

import win32gui
import win32process

logger = logging.getLogger(__name__)

LIBREOFFICE_PROCESS_NAMES = frozenset({
    "soffice.bin",
    "soffice.exe",
    "swriter.exe",
    "scalc.exe",
    "simpress.exe",
    "sdraw.exe",
    "sbase.exe",
    "smath.exe",
})

COMPONENT_KEYWORDS: dict[str, str] = {
    "Writer": "writer",
    "Calc": "calc",
    "Impress": "impress",
    "Draw": "draw",
    "Base": "base",
    "Math": "math",
    "Start Center": "start_center",
}

COMPONENT_COMMANDS: dict[str, list[dict[str, str]]] = {
    "writer": [
        {"name": "new_document", "description": "Create a new Writer document"},
        {"name": "save_document", "description": "Save the current document"},
        {"name": "export_pdf", "description": "Export the document as PDF"},
        {"name": "insert_text", "description": "Insert text at the cursor"},
        {"name": "close_document", "description": "Close the current document"},
    ],
    "calc": [
        {"name": "new_document", "description": "Create a new Calc spreadsheet"},
        {"name": "save_document", "description": "Save the current spreadsheet"},
        {"name": "export_pdf", "description": "Export the spreadsheet as PDF"},
        {"name": "set_cell", "description": "Set a cell value"},
        {"name": "get_cell", "description": "Get a cell value"},
        {"name": "list_sheets", "description": "List all sheets"},
        {"name": "close_document", "description": "Close the current spreadsheet"},
    ],
    "impress": [
        {"name": "new_document", "description": "Create a new Impress presentation"},
        {"name": "save_document", "description": "Save the current presentation"},
        {"name": "export_pdf", "description": "Export the presentation as PDF"},
        {"name": "insert_text", "description": "Insert text into the current presentation or slide"},
        {"name": "close_document", "description": "Close the current presentation"},
    ],
    "draw": [
        {"name": "new_document", "description": "Create a new Draw document"},
        {"name": "save_document", "description": "Save the current drawing"},
        {"name": "export_pdf", "description": "Export the drawing as PDF"},
        {"name": "insert_text", "description": "Insert text into the current drawing"},
        {"name": "close_document", "description": "Close the current drawing"},
    ],
    "base": [
        {"name": "new_document", "description": "Create a new Base database"},
        {"name": "save_document", "description": "Save the current database"},
        {"name": "close_document", "description": "Close the current database"},
    ],
    "math": [
        {"name": "new_document", "description": "Create a new Math formula"},
        {"name": "save_document", "description": "Save the current formula"},
        {"name": "export_pdf", "description": "Export the formula as PDF"},
        {"name": "close_document", "description": "Close the current formula"},
    ],
    "start_center": [
        {"name": "new_document", "description": "Create a new document"},
        {"name": "open_document", "description": "Open an existing document"},
    ],
}


class LibreOfficeContext:
    """Detect LibreOffice windows and provide app-specific commands."""

    def __init__(self) -> None:
        """Initialise the LibreOffice context helper.

        Returns:
            None.
        """

    @staticmethod
    def _get_process_name(hwnd: int) -> str | None:
        """Return the executable name for a window handle.

        Args:
            hwnd - Handle of the window to inspect.

        Returns:
            Lowercase executable name, or None when unavailable.
        """
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)

            result = subprocess.run(
                [
                    "powershell.exe",
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    f"(Get-Process -Id {pid}).ProcessName",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

            name = result.stdout.strip()
            return f"{name}.exe".lower() if name else None

        except Exception:
            logger.debug(
                "Could not determine process name for hwnd %d",
                hwnd,
                exc_info=True,
            )
            return None

    @staticmethod
    def _identify_component(title: str) -> str | None:
        """Identify the LibreOffice component from a window title.

        Args:
            title - Window title to inspect.

        Returns:
            Component identifier, or None when unrecognised.
        """
        if not isinstance(title, str) or not title.strip():
            return None

        for keyword, component in COMPONENT_KEYWORDS.items():
            if keyword in title:
                return component

        lower = title.lower()

        if "writer" in lower or ".odt" in lower or ".docx" in lower:
            return "writer"
        if "calc" in lower or ".ods" in lower or ".xlsx" in lower:
            return "calc"
        if "impress" in lower or ".odp" in lower or ".pptx" in lower:
            return "impress"
        if "draw" in lower or ".odg" in lower:
            return "draw"
        if "base" in lower or ".odb" in lower:
            return "base"
        if "math" in lower or ".odf" in lower:
            return "math"

        return None

    def get_visible_component(self) -> dict[str, str | None] | None:
        """Find the top visible LibreOffice window in Z-order."""
        visible_info = None

        def callback(hwnd: int, _: object) -> bool:
            nonlocal visible_info
            if win32gui.IsWindowVisible(hwnd) and not win32gui.IsIconic(hwnd):
                title = win32gui.GetWindowText(hwnd)
                cls = win32gui.GetClassName(hwnd)
                if (cls == "SALFRAME" and title) or (
                    "libreoffice" in title.lower()
                    or "writer" in title.lower()
                    or "calc" in title.lower()
                    or "impress" in title.lower()
                ):
                    component = self._identify_component(title) or "start_center"
                    visible_info = {
                        "component": component,
                        "title": title,
                        "hwnd": str(hwnd),
                    }
                    return False
            return True

        win32gui.EnumWindows(callback, None)
        return visible_info

    def get_foreground_component(self) -> dict[str, str | None] | None:
        """Identify the foreground LibreOffice component.

        Returns:
            Dictionary with component, title, and hwnd when LibreOffice
            is in the foreground (or the foremost visible window), otherwise None.
        """
        hwnd = win32gui.GetForegroundWindow()

        if hwnd:
            title = win32gui.GetWindowText(hwnd)
            process_name = self._get_process_name(hwnd)

            if process_name and process_name in LIBREOFFICE_PROCESS_NAMES:
                component = self._identify_component(title) or "start_center"
                logger.info(
                    "LibreOffice foreground: component=%s title=%s",
                    component,
                    title,
                )
                return {
                    "component": component,
                    "title": title,
                    "hwnd": str(hwnd),
                }

        # Fallback to foremost visible LibreOffice window (e.g. when typing in terminal)
        visible = self.get_visible_component()
        if visible:
            logger.info(
                "LibreOffice visible fallback: component=%s title=%s",
                visible.get("component"),
                visible.get("title"),
            )
            return visible

        return None

    def get_app_commands(
        self,
        component: str | None = None,
    ) -> list[dict[str, str]]:
        """Return the commands available for a LibreOffice component.

        Args:
            component - Component identifier. When None, the foreground
                component is detected automatically.

        Returns:
            List of command definitions for the active component.
        """
        if component is None:
            context = self.get_foreground_component()

            if context is None:
                return []

            component = context.get("component")

        if not isinstance(component, str):
            return []

        return list(COMPONENT_COMMANDS.get(component, []))

    def is_foreground(self) -> bool:
        """Check whether a LibreOffice window is in the foreground.

        Returns:
            True when a LibreOffice window is the foreground window.
        """
        return self.get_foreground_component() is not None
