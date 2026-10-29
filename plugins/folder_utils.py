import errno
import logging
import os
import shutil
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


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

    def delete_file_result(self, file_path: str) -> DeletionResult:
        """Delete a file or symbolic link and classify the result.

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
            os.remove(file_path)
            result = DeletionResult(
                DeletionStatus.SUCCESS,
                file_path,
                "File deleted successfully.",
            )
            logger.info("File deleted: %s", file_path)

        except OSError as error:
            result = self._classify_deletion_error(file_path, error)

        self.last_deletion_result = result
        return result

    def delete_folder_result(self, folder_path: str) -> DeletionResult:
        """Delete a folder and classify the result.

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
            shutil.rmtree(folder_path)
            result = DeletionResult(
                DeletionStatus.SUCCESS,
                folder_path,
                "Folder deleted successfully.",
            )
            logger.info("Folder deleted: %s", folder_path)

        except OSError as error:
            result = self._classify_deletion_error(folder_path, error)

        self.last_deletion_result = result
        return result

    def delete_file(self, file_path: str) -> bool:
        """Delete a file without deleting a directory.

        Args:
            file_path - Path to the file to delete.

        Returns:
            True if the file is deleted, otherwise False.
        """
        return self.delete_file_result(file_path).success

    def delete_folder(self, folder_path: str) -> bool:
        """Delete a folder and its contents.

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
