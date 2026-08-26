from plugins.windows_utils import WindowsUtils
from plugins.folder_util import FolderUtil
from plugins.explorer_context import ExplorerContext

class ExecutionEngine:
    """
        Description: A class for executing commands.
    """
    def __init__(self):
        self.windows_utils = WindowsUtils()
        self.folder_util = FolderUtil()
        self.explorer_context = ExplorerContext()

    def execute(self, command):

        action = command.get("action")
        target = command.get("target")

        if action == "open_application":
            return self.windows_utils.open_application(target)

        if action == "create_folder":
            return self.folder_util.create_folder(target)

        if action == "move":
            source = command.get("source")
            destination = command.get("destination")
            return self.folder_util.move(source, destination)

        print("Unknown Action.")
        return False
