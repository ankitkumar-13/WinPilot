import ctypes
import logging

logger = logging.getLogger(__name__)

MB_YESNO = 0x00000004
MB_ICONWARNING = 0x00000030
MB_SETFOREGROUND = 0x00010000
IDYES = 6


def confirm_deletion(kind: str, target: str) -> bool:
    """Ask the user to confirm deleting a file or folder.

    Args:
        kind - Type of item being deleted.
        target - Resolved path of the item to delete.

    Returns:
        True if the user clicks Yes, otherwise False.
    """
    message = (
        f"Are you sure you want to delete this {kind}?\n\n"
        f"{target}\n\n"
        "This action may permanently remove data."
    )

    try:
        result = ctypes.windll.user32.MessageBoxW(
            None,
            message,
            "WinPilot - Confirm Deletion",
            MB_YESNO | MB_ICONWARNING | MB_SETFOREGROUND,
        )
    except (AttributeError, OSError):
        logger.exception("Failed to display deletion confirmation")
        return False

    if result == IDYES:
        logger.info("User confirmed deletion: %s", target)
        return True

    logger.info("User cancelled deletion: %s", target)
    return False
