import os
import shutil


class FolderUtil:
    """
    Utility class for folder and file operations.
    """

    def create_file(self, file_path):
        try:
            with open(file_path, "x"):
                pass

            print(f"File created: {file_path}")
            return True

        except FileExistsError:
            print(f"File already exists: {file_path}")
            return False

        except Exception as error:
            print(f"Failed to create file: {error}")
            return False

    def create_folder(self, folder_path):
        try:
            os.makedirs(folder_path, exist_ok=False)
            print(f"Folder created: {folder_path}")
            return True

        except FileExistsError:
            print(f"Folder already exists: {folder_path}")
            return False

        except Exception as error:
            print(f"Failed to create folder: {error}")
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

    def delete_folder(self, folder_path):
        try:
            if not os.path.isdir(folder_path):
                print(f"Folder does not exist: {folder_path}")
                return False

            shutil.rmtree(folder_path)

            print(f"Folder deleted: {folder_path}")
            return True

        except Exception as error:
            print(f"Failed to delete folder: {error}")
            return False

    def move(self, source, destination):
        try:
            shutil.move(source, destination)
            print(f"Moved: {source} -> {destination}")
            return True

        except Exception as error:
            print(f"Failed to move: {error}")
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
