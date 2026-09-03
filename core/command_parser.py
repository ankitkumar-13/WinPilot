import re

class CommandParser:

    OPEN_WORDS = ["open", "launch", "start", "run"]
    SEARCH_WORDS = ["search", "find"]
    CREATE_WORDS = ["create", "make"]
    DELETE_WORDS = ["delete", "remove"]
    MOVE_WORDS = ["move", "transfer"]
    RENAME_WORDS = ["rename"]

    def parse_command(self, command):
        """
        Parses the given command and returns a dictionary
        with the action and required information.
        """

        command = command.strip()

        if not command:
            return {
                "action": "unknown",
                "target": None
            }

        words = command.split()
        
        # Search Files and Folders
        if (
            len(words) > 1
            and words[0].lower() in self.SEARCH_WORDS
        ):
            return {
                "action": "search",
                "target": " ".join(words[1:])
            }

        # Open Application
        if words[0].lower() in self.OPEN_WORDS and len(words) > 1:
            return {
                "action": "open_application",
                "target": " ".join(words[1:])
            }

        # Create Folder
        if (
            len(words) > 2
            and words[0].lower() in self.CREATE_WORDS
            and words[1].lower() == "folder"
        ):
            return {
                "action": "create_folder",
                "target": " ".join(words[2:])
            }

        # Create File
        if (
            len(words) > 2
            and words[0].lower() in self.CREATE_WORDS
            and words[1].lower() == "file"
        ):
            return {
                "action": "create_file",
                "target": " ".join(words[2:])
            }
        
        # Move
        if words[0].lower() in self.MOVE_WORDS and len(words) > 1:
            move_command = command[len(words[0]):].strip()

            parts = re.split(r"\s+to\s+", move_command, maxsplit=1, flags=re.IGNORECASE)

            if len(parts) == 2:
                return {
                    "action": "move",
                    "source": parts[0].strip(),
                    "destination": parts[1].strip()
                }
            
        # Delete File
        if (
            len(words) > 2
            and words[0].lower() in self.DELETE_WORDS
            and words[1].lower() == "file"
        ):
            return {
                "action": "delete_file",
                "target": " ".join(words[2:])
            }

        # Delete Folder
        if (
            len(words) > 2
            and words[0].lower() in self.DELETE_WORDS
            and words[1].lower() == "folder"
        ):
            return {
                "action": "delete_folder",
                "target": " ".join(words[2:])
            }

        # Rename File / Folder
        if (
            len(words) > 2
            and words[0].lower() in self.RENAME_WORDS
            and words[1].lower() in ["file", "folder"]
        ):
            lower_words = [word.lower() for word in words]

            if "to" in lower_words[2:]:
                to_index = lower_words.index("to", 2)

                old_name = " ".join(words[2:to_index])
                new_name = " ".join(words[to_index + 1:])

                if old_name and new_name:
                    return {
                        "action": f"rename_{words[1].lower()}",
                        "target": old_name,
                        "new_name": new_name
                    }

        # Unknown Command
        return {
            "action": "unknown",
            "target": None
        }
