import win32com.client
import win32gui


class ExplorerContext:
    """
        Description: Helps maintain the context of the most recently active File Explorer folder.
    """
    
    def __init__(self):
        self.last_folder = None
    
    def get_explorer_windows(self):
        """
            Description: Returns all Explorer windows mapped by their HWND.
        """

        shell = win32com.client.Dispatch("Shell.Application")

        explorer_windows = {}

        for window in shell.Windows():

            try:
                if window.Name == "File Explorer":
                    explorer_windows[window.HWND] = window
            except Exception:
                continue  # Skip any windows that don't have a Name attribute

        return explorer_windows

    def get_foreground_explorer(self):
        """
            Description: Returns the Explorer window that is currently in the foreground.
                Returns None if no Explorer window is in the foreground.
        """

        foreground_window = win32gui.GetForegroundWindow()

        explorer_windows = self.get_explorer_windows()

        return explorer_windows.get(foreground_window, None)
    
    def get_top_explorer_window(self):
        """
            Description: Returns the explorer windows(HWND) with the highest Z-index or order in Windows.
        """

        explorer_windows = self.get_explorer_windows()

        if not explorer_windows:
            return None  # No open File Explorer windows

        result = []

        def callback(hwnd, _):
            if hwnd in explorer_windows:
                result.append(hwnd)

        win32gui.EnumWindows(callback, None)

        if not result:
            return None

        return explorer_windows[result[0]]

    def get_folder_from_window(self, explorer_window):
        """
            Description: Returns the folder path represented by an Explorer window.
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
            Description: Returns the most relevant Explorer folder.
                If an Explorer window is in the foreground , use it.
                Otherwise, use the highest-priority (Z-order) one.
                If no explorer windows are open, return the last known folder.
        """

        foreground_window = win32gui.GetForegroundWindow()

        explorer_windows = self.get_explorer_windows()

        if foreground_window in explorer_windows:

            folder = self.get_folder_from_window(explorer_windows[foreground_window])

            if folder:
                self.last_folder = folder
                return folder

        # No Explorer window is in the foreground.
        explorer = self.get_top_explorer_window()

        if explorer:

            folder = self.get_folder_from_window(explorer)
            if folder:
                self.last_folder = folder
                return folder

        return self.last_folder
