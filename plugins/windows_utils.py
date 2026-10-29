import logging
import subprocess

logger = logging.getLogger(__name__)


class WindowsUtils:
    """Handle Windows-specific operations."""

    APPLICATIONS = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "paint": "mspaint.exe",
    }

    def open_application(self, application: str) -> bool:
        """Open a supported Windows application.

        Args:
            application - Name of the application to launch.

        Returns:
            True if the application launch is initiated, otherwise False.
        """
        if not isinstance(application, str) or not application.strip():
            logger.error("Cannot open application: invalid name")
            return False

        application = application.strip().lower()
        executable = self.APPLICATIONS.get(application)

        if executable is None:
            logger.warning(
                "Application is not supported: %s",
                application,
            )
            return False

        try:
            subprocess.Popen([executable])
            logger.info("Application launch initiated: %s", application)
            return True

        except OSError:
            logger.exception(
                "Failed to launch application: %s",
                application,
            )
            return False
