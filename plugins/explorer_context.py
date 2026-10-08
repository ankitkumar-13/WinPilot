import win32com.client
import win32gui


class ExplorerContext:
    """
    Maintains the context of currently open File Explorer windows.
    """

    def __init__(self):
        pass

    def get_explorer_windows(self):
        """
        Returns all open File Explorer windows mapped by their HWND.
        """

        shell = win32com.client.Dispatch("Shell.Application")

        explorer_windows = {}

        for window in shell.Windows():
            try:
                if window.Name == "File Explorer":
                    explorer_windows[window.HWND] = window
            except Exception:
                continue

        return explorer_windows

    def get_foreground_explorer(self):
        """
        Returns the File Explorer window currently in the foreground.
        Returns None if another application is focused.
        """

        foreground_window = win32gui.GetForegroundWindow()

        explorer_windows = self.get_explorer_windows()

        return explorer_windows.get(foreground_window)

    def get_visible_explorer(self):
        """
        Returns a visible, non-minimised Explorer window.

        The first Explorer found in Z-order is returned.
        """

        explorer_windows = self.get_explorer_windows()

        if not explorer_windows:
            return None

        result = []

        def callback(hwnd, _):
            if hwnd in explorer_windows:
                if win32gui.IsWindowVisible(hwnd):
                    if not win32gui.IsIconic(hwnd):
                        result.append(hwnd)

        win32gui.EnumWindows(callback, None)

        if not result:
            return None

        return explorer_windows[result[0]]

    def get_folder_from_window(self, explorer_window):
        """
        Returns the folder path represented by an Explorer window.
        """

        if not explorer_window:
            return None

        try:
            folder = explorer_window.Document.Folder.Self.Path

            if folder:
                return folder

        except Exception as e:
            print(f"Failed to get folder from Explorer window: {e}")

        return None

    def get_folder(self):
        """
        Returns the most relevant currently available Explorer folder.

        Priority:
            1. Foreground Explorer
            2. Visible, non-minimised Explorer
            3. None

        If all Explorer windows are minimised or closed,
        returns None so the caller can fall back to Desktop.
        """

        # 1. Foreground Explorer
        explorer = self.get_foreground_explorer()

        if explorer:
            folder = self.get_folder_from_window(explorer)

            if folder:
                return folder

        # 2. Another visible Explorer
        explorer = self.get_visible_explorer()

        if explorer:
            folder = self.get_folder_from_window(explorer)

            if folder:
                return folder

        # 3. No usable Explorer
        return None