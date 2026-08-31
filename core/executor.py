import os

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

    def get_explorer_workspace(self):
        workspace = self.explorer_context.get_folder()
        if not workspace:
            workspace = os.path.join(os.path.expanduser("~"), "Desktop")
        return workspace

    def execute(self, command):

        action = command.get("action")
        target = command.get("target")

        if action == "open_application":
            return self.windows_utils.open_application(target)

        if action == "create_folder":
            workspace = self.get_explorer_workspace()
            folder_path = os.path.join(workspace, target)

            return self.folder_util.create_folder(folder_path)

        if action == "create_file":
            workspace = self.get_explorer_workspace()
            file_path = os.path.join(workspace, target)

            return self.folder_util.create_file(file_path)

        if action == "move":
            workspace = self.get_explorer_workspace()
            
            source = command.get("source")
            destination = command.get("destination")

            if not os.path.isabs(source):
                source = os.path.join(workspace, source)
            if not os.path.isabs(destination):
                destination = os.path.join(workspace, destination)

            return self.folder_util.move(source, destination)

        if action == "delete_file":
            workspace = self.get_explorer_workspace()

            if not os.path.isabs(target):
                target = os.path.join(workspace, target)

            return self.folder_util.delete_file(target)

        if action == "delete_folder":
            workspace = self.get_explorer_workspace()

            if not os.path.isabs(target):
                target = os.path.join(workspace, target)

            return self.folder_util.delete_folder(target)

        if action == "rename_file":
            workspace = self.get_explorer_workspace()

            if not os.path.isabs(target):
                target = os.path.join(workspace, target)

            return self.folder_util.rename(target, command.get("new_name"))

        print("Unknown Action.")
        return False
