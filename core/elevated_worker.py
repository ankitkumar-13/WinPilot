import argparse
import ctypes
import logging
import os
import shutil
from pathlib import Path
from ctypes import wintypes

logger = logging.getLogger(__name__)

MB_YESNO = 0x00000004
MB_ICONWARNING = 0x00000030
MB_DEFBUTTON2 = 0x00000100
MB_SETFOREGROUND = 0x00010000

IDYES = 6


def confirm_elevated_deletion(kind: str, target: Path) -> bool:
    """Confirm the exact deletion target in the elevated process.

    Args:
        kind - Type of item being deleted.
        target - Absolute path of the item being deleted.

    Returns:
        True if the user explicitly confirms deletion.
    """
    user32 = ctypes.WinDLL("user32.dll", use_last_error=True)
    user32.MessageBoxW.argtypes = [
        wintypes.HWND,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.UINT,
    ]
    user32.MessageBoxW.restype = ctypes.c_int

    message = (
        f"Delete this {kind} using administrator privileges?\n\n"
        f"{target}\n\n"
        "This action may permanently remove data."
    )

    result = user32.MessageBoxW(
        None,
        message,
        "WinPilot - Elevated Deletion",
        MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2 | MB_SETFOREGROUND,
    )

    return result == IDYES


def delete_target(kind: str, target: Path) -> bool:
    """Delete one validated file or folder.

    Args:
        kind - Type of item being deleted.
        target - Absolute path of the item to delete.

    Returns:
        True if deletion succeeds, otherwise False.
    """
    try:
        if kind not in {"file", "folder"}:
            logger.error("Rejected unsupported deletion type: %s", kind)
            return False

        if not target.is_absolute():
            logger.error("Rejected non-absolute path: %s", target)
            return False

        if not target.exists() and not target.is_symlink():
            logger.error("Deletion target does not exist: %s", target)
            return False

        if target.is_symlink():
            target.unlink()
        elif kind == "file":
            if not target.is_file():
                logger.error("Target is not a file: %s", target)
                return False
            target.unlink()
        else:
            if not target.is_dir():
                logger.error("Target is not a folder: %s", target)
                return False

            if target.resolve() == Path(target.anchor).resolve():
                logger.error("Refused to delete filesystem root: %s", target)
                return False

            shutil.rmtree(target)

        logger.info("Deleted %s: %s", kind, target)
        return True

    except (OSError, shutil.Error):
        logger.exception("Deletion failed: %s", target)
        return False


def main() -> int:
    """Validate arguments and run the elevated deletion worker.

    Returns:
        Zero on success, otherwise a non-zero exit code.
    """
    parser = argparse.ArgumentParser(
        description="WinPilot elevated deletion worker"
    )
    parser.add_argument(
        "--kind",
        required=True,
        choices=("file", "folder"),
    )
    parser.add_argument("--target", required=True)

    args = parser.parse_args()

    if os.name != "nt":
        logger.error("Elevated deletion is only supported on Windows")
        return 2

    try:
        target = Path(args.target)

        if not target.is_absolute():
            logger.error("Rejected non-absolute deletion path")
            return 2

        if not target.exists() and not target.is_symlink():
            logger.error("Deletion target does not exist: %s", target)
            return 2

        if not confirm_elevated_deletion(args.kind, target):
            logger.info("Elevated deletion cancelled: %s", target)
            return 3

        return 0 if delete_target(args.kind, target) else 1

    except (OSError, ValueError):
        logger.exception("Elevated deletion worker failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
