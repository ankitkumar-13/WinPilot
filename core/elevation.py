import ctypes
import logging
import os
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path

logger = logging.getLogger(__name__)

SEE_MASK_NOCLOSEPROCESS = 0x00000040
WAIT_OBJECT_0 = 0
WAIT_FAILED = 0xFFFFFFFF
ERROR_CANCELLED = 1223


class ShellExecuteInfo(ctypes.Structure):
    """Represent the Windows ShellExecuteEx configuration."""

    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("fMask", ctypes.c_ulong),
        ("hwnd", wintypes.HWND),
        ("lpVerb", wintypes.LPCWSTR),
        ("lpFile", wintypes.LPCWSTR),
        ("lpParameters", wintypes.LPCWSTR),
        ("lpDirectory", wintypes.LPCWSTR),
        ("nShow", ctypes.c_int),
        ("hInstApp", wintypes.HINSTANCE),
        ("lpIDList", ctypes.c_void_p),
        ("lpClass", wintypes.LPCWSTR),
        ("hkeyClass", wintypes.HKEY),
        ("dwHotKey", wintypes.DWORD),
        ("hIcon", wintypes.HANDLE),
        ("hProcess", wintypes.HANDLE),
    ]


def elevated_delete(kind: str, target: str) -> bool:
    """Request UAC elevation to delete one specific file or folder.

    Args:
        kind - Item type, either file or folder.
        target - Absolute path of the item to delete.

    Returns:
        True if deletion succeeds, otherwise False.
    """
    if os.name != "nt":
        logger.error("Elevated deletion is only supported on Windows")
        return False

    if kind not in {"file", "folder"}:
        logger.error("Rejected invalid elevated deletion type: %r", kind)
        return False

    if not isinstance(target, str) or not target.strip():
        logger.error("Rejected empty elevated deletion target")
        return False

    path = Path(target).resolve()

    if not path.is_absolute() or not path.exists():
        logger.error("Deletion target does not exist: %s", path)
        return False

    worker = Path(__file__).with_name("elevated_worker.py")

    if not worker.is_file():
        logger.error("Elevated deletion worker is missing: %s", worker)
        return False

    parameters = subprocess.list2cmdline(
        [
            str(worker),
            "--kind",
            kind,
            "--target",
            str(path),
        ]
    )

    info = ShellExecuteInfo()
    info.cbSize = ctypes.sizeof(ShellExecuteInfo)
    info.fMask = SEE_MASK_NOCLOSEPROCESS
    info.hwnd = None
    info.lpVerb = "runas"
    info.lpFile = sys.executable
    info.lpParameters = parameters
    info.lpDirectory = str(worker.parent.parent)
    info.nShow = 1

    shell32 = ctypes.WinDLL("shell32.dll", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)

    shell32.ShellExecuteExW.argtypes = [
        ctypes.POINTER(ShellExecuteInfo)
    ]
    shell32.ShellExecuteExW.restype = wintypes.BOOL

    kernel32.WaitForSingleObject.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
    ]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD

    kernel32.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL

    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    try:
        if not shell32.ShellExecuteExW(ctypes.byref(info)):
            error = ctypes.get_last_error()

            if error == ERROR_CANCELLED:
                logger.info("User cancelled UAC elevation")
            else:
                logger.error(
                    "Could not launch elevated deletion worker: %s",
                    ctypes.WinError(error),
                )

            return False

        if not info.hProcess:
            logger.error("Elevated worker returned no process handle")
            return False

        wait_result = kernel32.WaitForSingleObject(
            info.hProcess,
            0xFFFFFFFF,
        )

        if wait_result != WAIT_OBJECT_0:
            logger.error("Waiting for elevated worker failed")
            return False

        exit_code = wintypes.DWORD()

        if not kernel32.GetExitCodeProcess(
            info.hProcess,
            ctypes.byref(exit_code),
        ):
            logger.error("Could not retrieve elevated worker exit code")
            return False

        if exit_code.value != 0:
            logger.warning(
                "Elevated deletion failed with exit code %d",
                exit_code.value,
            )
            return False

        logger.info("Elevated deletion completed: %s", path)
        return True

    except OSError:
        logger.exception("Elevated deletion failed: %s", path)
        return False

    finally:
        if info.hProcess:
            kernel32.CloseHandle(info.hProcess)
