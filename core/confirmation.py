import logging

import win32api
import win32con

logger = logging.getLogger(__name__)


def confirm_action(action: str, details: str) -> bool:
    """Ask the user to approve one operation before WinPilot runs it.

    Args:
        action - Short description of the requested operation.
        details - Target application or paths involved in the operation.

    Returns:
        True if the user approves the command, otherwise False.
    """
    message = (
        f"WinPilot wants permission to {action}.\n\n"
        f"{details}\n\n"
        "Allow this command?"
    )
    flags = (
        win32con.MB_YESNO
        | win32con.MB_ICONWARNING
        | win32con.MB_DEFBUTTON2
        | win32con.MB_SETFOREGROUND
    )

    try:
        approved = (
            win32api.MessageBox(
                0,
                message,
                "WinPilot - Command Approval",
                flags,
            )
            == win32con.IDYES
        )
    except Exception:
        logger.exception("Could not display command approval prompt")
        return False

    logger.info(
        "User %s command: %s | %s",
        "approved" if approved else "denied",
        action,
        details,
    )
    return approved
