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
