import os
import subprocess

class WindowsUtils:
    """
        Description: A utility class for Windows-specific operations, such as opening applications.
    """
    
    APPLICATIONS = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "paint": "mspaint.exe",
    }

    def open_application(self, application):
        application = application.lower().strip()

        executable = self.APPLICATIONS.get(application)

        if executable is None:
            print(f"Application not supported: {application}")
            return False

        try:
            subprocess.Popen(executable)
            return True

        except Exception as error:
            print(f"Failed to open {application}: {error}")
            return False

    def delete_file(self, file_path):
        file_path = file_path.strip()

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False

        if os.path.isdir(file_path):
            print(f"Path is a directory, not a file: {file_path}")
            return False

        try:
            os.remove(file_path)
            return True

        except Exception as error:
            print(f"Failed to delete {file_path}: {error}")
            return False

    def rename(self, path, new_name):
        path = path.strip()
        new_name = new_name.strip() if new_name else ""

        if not new_name:
            print("New name is missing.")
            return False

        if not os.path.exists(path):
            print(f"Path not found: {path}")
            return False

        new_path = new_name

        if os.path.dirname(path) and not os.path.dirname(new_name):
            new_path = os.path.join(
                os.path.dirname(path),
                new_name
            )

        if os.path.exists(new_path):
            print(f"A path already exists with that name: {new_path}")
            return False

        try:
            os.rename(path, new_path)
            print(f"Renamed: {path} -> {new_path}")
            return True

        except Exception as error:
            print(f"Failed to rename {path}: {error}")
            return False
