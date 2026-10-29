import logging

import pythoncom
import win32com.client
import win32gui

logger = logging.getLogger(__name__)


class ExplorerContext:
    """Retrieve folder paths from open File Explorer windows."""

    def __init__(self) -> None:
        """Initialise the Explorer context helper.

        Returns:
            None.
        """

    def get_explorer_windows(self) -> dict[int, str]:
        """Retrieve Explorer window handles and their folder paths.

        Returns:
            Dictionary mapping window handles to folder paths.

        Raises:
            Exception - If Explorer windows cannot be enumerated.
        """
        pythoncom.CoInitialize()

        try:
            shell = win32com.client.Dispatch("Shell.Application")
            explorer_windows: dict[int, str] = {}

            for window in shell.Windows():
                try:
                    if window.Name != "File Explorer":
                        continue

                    hwnd = int(window.HWND)
                    folder = window.Document.Folder.Self.Path

                    if isinstance(folder, str) and folder:
                        explorer_windows[hwnd] = folder

                except Exception:
                    logger.debug(
                        "Skipping an inaccessible Explorer window",
                        exc_info=True,
                    )

            return explorer_windows

        except Exception:
            logger.exception("Failed to retrieve Explorer windows")
            raise

        finally:
            pythoncom.CoUninitialize()

    def get_foreground_explorer(self) -> str | None:
        """Get the folder path of the foreground Explorer window.

        Returns:
            Folder path if File Explorer is in the foreground,
            otherwise None.
        """
        foreground_window = win32gui.GetForegroundWindow()
        explorer_windows = self.get_explorer_windows()

        return explorer_windows.get(foreground_window)

    def get_visible_explorer(self) -> str | None:
        """Get a visible, non-minimised Explorer folder path.

        Returns:
            Folder path of the first suitable window, or None.
        """
        explorer_windows = self.get_explorer_windows()

        if not explorer_windows:
            return None

        visible_windows: list[int] = []

        def callback(hwnd: int, _: object) -> bool:
            """Collect visible Explorer windows in Z-order.

            Args:
                hwnd - Handle of the window being inspected.
                _ - Unused callback parameter.

            Returns:
                True to continue enumerating windows.
            """
            if (
                hwnd in explorer_windows
                and win32gui.IsWindowVisible(hwnd)
                and not win32gui.IsIconic(hwnd)
            ):
                visible_windows.append(hwnd)

            return True

        win32gui.EnumWindows(callback, None)

        if not visible_windows:
            return None

        return explorer_windows.get(visible_windows[0])

    def get_folder(self) -> str | None:
        """Get the most relevant available Explorer folder.

        Returns:
            Foreground Explorer folder, another visible Explorer
            folder, or None if no suitable window is available.
        """
        try:
            folder = self.get_foreground_explorer()

            if folder:
                return folder

            return self.get_visible_explorer()

        except Exception:
            logger.exception("Failed to determine the Explorer folder")
            return None
