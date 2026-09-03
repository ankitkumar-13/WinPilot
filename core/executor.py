import os

from plugins.windows_utils import WindowsUtils
from plugins.folder_utils import FolderUtil
from plugins.explorer_context import ExplorerContext

class ExecutionEngine:
    """
        Description: A class for executing commands.
    """
    def __init__(self):
        self.windows_utils = WindowsUtils()
        self.folder_utils = FolderUtil()
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

            if not os.path.isabs(target):
                target = os.path.join(workspace, target)

            return self.folder_utils.create_folder(target)

        if action == "search":
            workspace = self.get_explorer_workspace()

            if not target:
                print("Search term is missing.")
                return False

            return self.folder_utils.search(workspace, target)
        
        if action == "create_file":
            workspace = self.get_explorer_workspace()

            if not os.path.isabs(target):
                target = os.path.join(workspace, target)

            return self.folder_utils.create_file(target)

        if action == "move":
            workspace = self.get_explorer_workspace()

            source = command.get("source")
            destination = command.get("destination")

            if not source or not destination:
                print("Source or destination not provided for move action.")
                return False

            if not os.path.isabs(source):
                source = os.path.join(workspace, source)
            if not os.path.isabs(destination):
                destination = os.path.join(workspace, destination)

            return self.folder_utils.move(source, destination)

        if action == "delete_file":
            workspace = self.get_explorer_workspace()

            if not os.path.isabs(target):
                target = os.path.join(workspace, target)

            return self.folder_utils.delete_file(target)

        if action == "delete_folder":
            workspace = self.get_explorer_workspace()

            if not os.path.isabs(target):
                target = os.path.join(workspace, target)

            return self.folder_utils.delete_folder(target)

        if action in ["rename_file", "rename_folder"]:
            workspace = self.get_explorer_workspace()

            target = command.get("target")
            new_name = command.get("new_name")

            if not target or not new_name:
                print("Target or new name not provided for rename action.")
                return False

            if not os.path.isabs(target):
                target = os.path.join(workspace, target)

            return self.folder_utils.rename(target, new_name)

        print("Unknown Action.")
        return False
