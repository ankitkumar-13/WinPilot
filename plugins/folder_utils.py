import errno
import ctypes
import logging
import os
import shutil
from ctypes import wintypes
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

FO_DELETE = 0x0003
FOF_SILENT = 0x0004
FOF_NOCONFIRMATION = 0x0010
FOF_ALLOWUNDO = 0x0040
FOF_NOERRORUI = 0x0400
FOF_WANTNUKEWARNING = 0x4000


class DeletionStatus(Enum):
    """Classify the outcome of a deletion attempt."""

    SUCCESS = "success"
    PERMISSION_DENIED = "permission_denied"
    LOCKED = "locked"
    NOT_FOUND = "not_found"
    INVALID_TARGET = "invalid_target"
    FAILED = "failed"


@dataclass(frozen=True)
class DeletionResult:
    """Describe the outcome of a file or folder deletion.

    Attributes:
        status - Classification of the deletion outcome.
        target - Path supplied for deletion.
        message - Human-readable result.
        error_code - Operating-system error code, if available.

    Returns:
        Immutable description of a deletion outcome.
    """

    status: DeletionStatus
    target: str
    message: str
    error_code: int | None = None

    @property
    def success(self) -> bool:
        """Return whether deletion succeeded.

        Returns:
            True when the target was deleted.
        """
        return self.status is DeletionStatus.SUCCESS

class FolderUtil:
    """Perform file and folder operations."""

    def __init__(self) -> None:
        """Initialise deletion result tracking.

        Returns:
            None.
        """
        self.last_deletion_result: DeletionResult | None = None

    def create_file(self, file_path: str) -> bool:
        """Create a new empty file.

        Args:
            file_path - Path where the file will be created.

        Returns:
            True if the file is created, otherwise False.
        """
        if not isinstance(file_path, str) or not file_path.strip():
            logger.error("Cannot create file: invalid path")
            return False

        file_path = file_path.strip()

        try:
            with open(file_path, "x", encoding="utf-8"):
                pass

            logger.info("File created: %s", file_path)
            return True

        except FileExistsError:
            logger.warning("File already exists: %s", file_path)
        except OSError:
            logger.exception("Failed to create file: %s", file_path)

        return False

    def create_folder(self, folder_path: str) -> bool:
        """Create a new folder.

        Args:
            folder_path - Path where the folder will be created.

        Returns:
            True if the folder is created, otherwise False.
        """
        if not isinstance(folder_path, str) or not folder_path.strip():
            logger.error("Cannot create folder: invalid path")
            return False

        folder_path = folder_path.strip()

        try:
            os.makedirs(folder_path, exist_ok=False)
            logger.info("Folder created: %s", folder_path)
            return True

        except FileExistsError:
            logger.warning("Path already exists: %s", folder_path)
        except OSError:
            logger.exception("Failed to create folder: %s", folder_path)

        return False

    def _classify_deletion_error(
        self,
        target: str,
        error: OSError,
    ) -> DeletionResult:
        """Classify an operating-system deletion error.

        Args:
            target - Path involved in the failed operation.
            error - Operating-system exception.

        Returns:
            A structured deletion failure result.
        """
        if isinstance(error, FileNotFoundError):
            status = DeletionStatus.NOT_FOUND
        elif isinstance(error, PermissionError):
            status = DeletionStatus.PERMISSION_DENIED
        elif error.errno in {errno.EACCES, errno.EPERM}:
            status = DeletionStatus.PERMISSION_DENIED
        elif error.errno in {errno.EBUSY, errno.ETXTBSY}:
            status = DeletionStatus.LOCKED
        elif isinstance(error, IsADirectoryError):
            status = DeletionStatus.INVALID_TARGET
        elif isinstance(error, NotADirectoryError):
            status = DeletionStatus.INVALID_TARGET
        elif os.name == "nt" and getattr(error, "winerror", None) in {
            32,  # Sharing violation.
            33,  # Lock violation.
        }:
            status = DeletionStatus.LOCKED
        else:
            status = DeletionStatus.FAILED

        result = DeletionResult(
            status=status,
            target=target,
            message=str(error) or status.value,
            error_code=getattr(error, "winerror", None) or error.errno,
        )

        logger.warning(
            "Deletion failed: target=%s status=%s error=%s",
            target,
            result.status.value,
            result.message,
        )

        return result

    @staticmethod
    def _move_to_recycle_bin(path: str) -> None:
        """Send a path to the Windows Recycle Bin.

        Args:
            path - Absolute path of the file or folder to remove.

        Returns:
            None.

        Raises:
            OSError - If Windows cannot move the item to the Recycle Bin.
        """
        if os.name != "nt":
            raise OSError("Moving items to the Recycle Bin requires Windows.")
        if not os.path.isabs(path):
            raise OSError("The Recycle Bin requires a fully qualified path.")

        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [
                ("hwnd", wintypes.HWND),
                ("wFunc", wintypes.UINT),
                ("pFrom", wintypes.LPCWSTR),
                ("pTo", wintypes.LPCWSTR),
                ("fFlags", wintypes.UINT),
                ("fAnyOperationsAborted", wintypes.BOOL),
                ("hNameMappings", wintypes.LPVOID),
                ("lpszProgressTitle", wintypes.LPCWSTR),
            ]

        source = ctypes.create_unicode_buffer(path + "\0")
        operation = SHFILEOPSTRUCTW()
        operation.hwnd = None
        operation.wFunc = FO_DELETE
        operation.pFrom = ctypes.cast(source, wintypes.LPCWSTR)
        operation.pTo = None
        operation.fFlags = (
            FOF_SILENT
            | FOF_NOCONFIRMATION
            | FOF_ALLOWUNDO
            | FOF_NOERRORUI
            | FOF_WANTNUKEWARNING
        )
        operation.fAnyOperationsAborted = False
        operation.hNameMappings = None
        operation.lpszProgressTitle = None

        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
        shell32.SHFileOperationW.argtypes = [ctypes.POINTER(SHFILEOPSTRUCTW)]
        shell32.SHFileOperationW.restype = ctypes.c_int
        result = shell32.SHFileOperationW(ctypes.byref(operation))
        if result != 0:
            raise OSError(result, ctypes.FormatError(result), path)
        if operation.fAnyOperationsAborted:
            raise OSError("The Recycle Bin operation was cancelled.")

    def delete_file_result(self, file_path: str) -> DeletionResult:
        """Move a file or symbolic link to the Recycle Bin and classify it.

        Args:
            file_path - Path to the file to delete.

        Returns:
            Structured result describing the deletion outcome.
        """
        if not isinstance(file_path, str) or not file_path.strip():
            result = DeletionResult(
                DeletionStatus.INVALID_TARGET,
                str(file_path),
                "Invalid file path.",
            )
            self.last_deletion_result = result
            return result

        file_path = file_path.strip()

        if not os.path.lexists(file_path):
            result = DeletionResult(
                DeletionStatus.NOT_FOUND,
                file_path,
                "File does not exist.",
            )
            self.last_deletion_result = result
            return result

        if os.path.isdir(file_path) and not os.path.islink(file_path):
            result = DeletionResult(
                DeletionStatus.INVALID_TARGET,
                file_path,
                "Target is a directory, not a file.",
            )
            self.last_deletion_result = result
            return result

        try:
            self._move_to_recycle_bin(file_path)
            result = DeletionResult(
                DeletionStatus.SUCCESS,
                file_path,
                "File moved to the Recycle Bin.",
            )
            logger.info("File moved to the Recycle Bin: %s", file_path)

        except OSError as error:
            result = self._classify_deletion_error(file_path, error)

        self.last_deletion_result = result
        return result

    def delete_folder_result(self, folder_path: str) -> DeletionResult:
        """Move a folder to the Recycle Bin and classify the result.

        Args:
            folder_path - Path to the folder to delete.

        Returns:
            Structured result describing the deletion outcome.
        """
        if not isinstance(folder_path, str) or not folder_path.strip():
            result = DeletionResult(
                DeletionStatus.INVALID_TARGET,
                str(folder_path),
                "Invalid folder path.",
            )
            self.last_deletion_result = result
            return result

        folder_path = folder_path.strip()

        if os.path.islink(folder_path):
            result = DeletionResult(
                DeletionStatus.INVALID_TARGET,
                folder_path,
                "Refusing to recursively delete a symbolic link.",
            )
            self.last_deletion_result = result
            return result

        if not os.path.lexists(folder_path):
            result = DeletionResult(
                DeletionStatus.NOT_FOUND,
                folder_path,
                "Folder does not exist.",
            )
            self.last_deletion_result = result
            return result

        if not os.path.isdir(folder_path):
            result = DeletionResult(
                DeletionStatus.INVALID_TARGET,
                folder_path,
                "Target is not a folder.",
            )
            self.last_deletion_result = result
            return result

        try:
            self._move_to_recycle_bin(folder_path)
            result = DeletionResult(
                DeletionStatus.SUCCESS,
                folder_path,
                "Folder moved to the Recycle Bin.",
            )
            logger.info("Folder moved to the Recycle Bin: %s", folder_path)

        except OSError as error:
            result = self._classify_deletion_error(folder_path, error)

        self.last_deletion_result = result
        return result

    def delete_file(self, file_path: str) -> bool:
        """Move a file to the Recycle Bin without accepting directories.

        Args:
            file_path - Path to the file to delete.

        Returns:
            True if the file is deleted, otherwise False.
        """
        return self.delete_file_result(file_path).success

    def delete_folder(self, folder_path: str) -> bool:
        """Move a folder and its contents to the Recycle Bin.

        Args:
            folder_path - Path to the folder to delete.

        Returns:
            True if the folder is deleted, otherwise False.
        """
        return self.delete_folder_result(folder_path).success

    def move(self, source: str, destination: str) -> bool:
        """Move a file or folder to a destination.

        Args:
            source - Path to the item to move.
            destination - Destination path for the item.

        Returns:
            True if the item is moved, otherwise False.
        """
        if (
            not isinstance(source, str)
            or not source.strip()
            or not isinstance(destination, str)
            or not destination.strip()
        ):
            logger.error("Cannot move item: invalid source or destination")
            return False

        source = source.strip()
        destination = destination.strip()

        if not os.path.lexists(source):
            logger.warning("Move source not found: %s", source)
            return False

        try:
            shutil.move(source, destination)
            logger.info("Moved item: %s -> %s", source, destination)
            return True

        except (OSError, shutil.Error):
            logger.exception(
                "Failed to move item: %s -> %s",
                source,
                destination,
            )
            return False

    def copy(self, source: str, destination: str) -> bool:
        """Copy a file or folder to a destination.

        Args:
            source - Path to the item to copy.
            destination - Destination path for the copy.

        Returns:
            True if the item is copied, otherwise False.
        """
        if (
            not isinstance(source, str)
            or not source.strip()
            or not isinstance(destination, str)
            or not destination.strip()
        ):
            logger.error("Cannot copy item: invalid source or destination")
            return False

        source = source.strip()
        destination = destination.strip()

        if not os.path.lexists(source):
            logger.warning("Copy source not found: %s", source)
            return False

        try:
            if os.path.isdir(source) and not os.path.islink(source):
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)

            logger.info("Copied item: %s -> %s", source, destination)
            return True

        except (OSError, shutil.Error):
            logger.exception(
                "Failed to copy item: %s -> %s",
                source,
                destination,
            )
            return False

    def rename(self, path: str, new_name: str) -> bool:
        """Rename a file or folder.

        Args:
            path - Path to the existing file or folder.
            new_name - New name or destination path.

        Returns:
            True if the item is renamed, otherwise False.
        """
        if not isinstance(path, str) or not path.strip():
            logger.error("Cannot rename item: invalid source path")
            return False

        if not isinstance(new_name, str) or not new_name.strip():
            logger.error("Cannot rename item: new name is missing")
            return False

        path = path.strip()
        new_name = new_name.strip()

        if not os.path.lexists(path):
            logger.warning("Rename source not found: %s", path)
            return False

        if os.path.basename(new_name.rstrip("\\/")) in {"", ".", ".."}:
            logger.warning("Invalid rename destination: %s", new_name)
            return False

        if not os.path.dirname(new_name):
            new_path = os.path.join(
                os.path.dirname(os.path.abspath(path)),
                new_name,
            )
        else:
            new_path = new_name

        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(
            os.path.abspath(new_path)
        ):
            logger.warning("Source and destination are identical: %s", path)
            return False

        if os.path.lexists(new_path):
            logger.warning("Rename destination already exists: %s", new_path)
            return False

        try:
            os.rename(path, new_path)
            logger.info("Renamed item: %s -> %s", path, new_path)
            return True

        except OSError:
            logger.exception(
                "Failed to rename item: %s -> %s",
                path,
                new_path,
            )
            return False
