import os
import shutil
from urllib.parse import quote

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

    def search(self, folder_path, search_term):
        search_term = search_term.strip()

        if not search_term:
            print("Search term is missing.")
            return False

        if not os.path.isdir(folder_path):
            print(f"Folder does not exist: {folder_path}")
            return False

        try:
            # Open Windows File Explorer's native search results.
            encoded_term = quote(search_term)
            encoded_folder = quote(folder_path, safe=":/\\")

            search_url = (
                f"search-ms:query={encoded_term}"
                f"&crumb=location:{encoded_folder}"
            )

            os.startfile(search_url)

            print(f"Opened Windows search for: {search_term}")
            print(f"Search location: {folder_path}")

            # Also print matching paths in the terminal.
            matches = []

            for current_path, folder_names, file_names in os.walk(folder_path):
                for name in folder_names + file_names:
                    if search_term.lower() in name.lower():
                        matches.append(os.path.join(current_path, name))

            if not matches:
                print(f"No results found for: {search_term}")
            else:
                print(f"\nMatching paths for: {search_term}")

                for match in matches:
                    print(match)

            return True

        except Exception as error:
            print(f"Failed to open Windows search: {error}")
            return False