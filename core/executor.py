from plugins.windows_utils import WindowsUtils

class ExecutionEngine:
    """
        Description: A class for executing commands.
    """
    def __init__(self):
        self.windows_utils = WindowsUtils()

    def execute(self, command):

        action = command.get("action")
        target = command.get("target")

        if action == "open_application":
            return self.windows_utils.open_application(target)

        print("Unknown Action.");
        return False
