import os
import shutil


class FolderUtil:
    """
    Utility class for folder and file operations.
    """

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

    def move(self, source, destination):
        try:
            shutil.move(source, destination)
            print(f"Moved: {source} -> {destination}")
            return True

        except Exception as error:
            print(f"Failed to move: {error}")
            return False

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
