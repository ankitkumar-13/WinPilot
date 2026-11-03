import os

from plugins.windows_utils import WindowsUtils
from plugins.folder_utils import FolderUtil
from plugins.explorer_context import ExplorerContext
from plugins.libre_utils import LibreUtils


class ExecutionEngine:
    """
    Description: A class for executing commands.
    """

    def __init__(self):
        self.windows_utils = WindowsUtils()
        self.folder_utils = FolderUtil()
        self.explorer_context = ExplorerContext()
        self.libre_utils = LibreUtils()

    def get_explorer_workspace(self):
        workspace = self.explorer_context.get_folder()
        if not workspace:
            workspace = os.path.join(os.path.expanduser("~"), "Desktop")
        return workspace

    def execute(self, command):

        action = command.get("action")
        target = command.get("target")

        # ------------------------------------------------------------------ #
        #  Standard Application & File Commands                               #
        # ------------------------------------------------------------------ #

        if action == "open_application":
            return self.windows_utils.open_application(target)

        if action == "create_folder":
            workspace = self.get_explorer_workspace()

            if not os.path.isabs(target):
                target = os.path.join(workspace, target)

            return self.folder_utils.create_folder(target)

        if action == "create_file":
            workspace = self.get_explorer_workspace()

            if not os.path.isabs(target):
                target = os.path.join(workspace, target)

            return self.folder_utils.create_file(target)

        if action == "copy":
            workspace = self.get_explorer_workspace()

            source = command.get("source")
            destination = command.get("destination")

            if not source or not destination:
                print("Source or destination not provided for copy action.")
                return False

            if not os.path.isabs(source):
                source = os.path.join(workspace, source)

            if not os.path.isabs(destination):
                destination = os.path.join(workspace, destination)

            return self.folder_utils.copy(source, destination)

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

        # ------------------------------------------------------------------ #
        #  LibreOffice Commands                                               #
        # ------------------------------------------------------------------ #

        if action == "libre_start_uno":
            return self.libre_utils.start_with_uno()

        if action == "libre_close":
            return self.libre_utils.close_libre()

        if action == "libre_open_app":
            return self.libre_utils.open_app(target)

        if action == "libre_open_file":
            workspace = self.get_explorer_workspace()
            if not os.path.isabs(target):
                target = os.path.join(workspace, target)
            return self.libre_utils.open_file(target)

        if action == "libre_convert":
            workspace = self.get_explorer_workspace()
            source = command.get("source")
            fmt = command.get("format")
            output_dir = command.get("output_dir")

            if not source:
                print("Source file not provided for conversion.")
                return False

            if not os.path.isabs(source):
                source = os.path.join(workspace, source)

            if output_dir and not os.path.isabs(output_dir):
                output_dir = os.path.join(workspace, output_dir)

            return self.libre_utils.convert_file(source, fmt, output_dir)

        if action == "libre_create_writer":
            workspace = self.get_explorer_workspace()
            if target and not os.path.isabs(target):
                target = os.path.join(workspace, target)
            return self.libre_utils.create_writer_document(target)

        if action == "libre_create_calc":
            workspace = self.get_explorer_workspace()
            if target and not os.path.isabs(target):
                target = os.path.join(workspace, target)
            return self.libre_utils.create_calc_spreadsheet(target)

        if action == "libre_add_text":
            text = command.get("text")
            is_heading = command.get("heading", False)
            return self.libre_utils.add_text_to_writer(text, is_heading=is_heading)

        if action == "libre_replace":
            find_text = command.get("find_text")
            replace_text = command.get("replace_text")
            return self.libre_utils.replace_text_in_writer(find_text, replace_text)

        if action == "libre_count_words":
            return self.libre_utils.count_words_in_writer()

        if action == "libre_set_cell":
            cell = command.get("cell")
            value = command.get("value")
            return self.libre_utils.set_cell(cell, value)

        if action == "libre_get_cell":
            cell = command.get("cell")
            return self.libre_utils.get_cell(cell)

        print("Unknown Action.")
        return False
