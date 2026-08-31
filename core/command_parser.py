import re

class CommandParser:

    OPEN_WORDS = ["open", "launch", "start", "run"]
    DELETE_WORDS = ["delete", "remove"]
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

        # Open Application
        if words[0].lower() in self.OPEN_WORDS and len(words) > 1:
            return {
                "action": "open_application",
                "target": " ".join(words[1:])
            }

        # Create Folder
        if (
            len(words) > 2
            and words[0].lower() == "create"
            and words[1].lower() == "folder"
        ):
            return {
                "action": "create_folder",
                "target": " ".join(words[2:])
            }

        # Move
        if words[0].lower() == "move":
            move_command = command[len(words[0]):].strip()

            parts = re.split(r"\s+to\s+", move_command, maxsplit=1, flags=re.IGNORECASE)

            if len(parts) == 2:
                return {
                    "action": "move",
                    "source": parts[0].strip(),
                    "destination": parts[1].strip()
                }
            
        # Delete File
        if words[0] in self.DELETE_WORDS and len(words) > 1:
            return {
                "action": "delete_file",
                "target": " ".join(words[1:])
            }

        # Rename File
        if words[0] in self.RENAME_WORDS and len(words) > 2:
            if "to" in words[1:]:
                to_index = words.index("to", 1)
                old_name = " ".join(words[1:to_index])
                new_name = " ".join(words[to_index + 1:])

                if old_name and new_name:
                    return {
                        "action": "rename_file",
                        "target": old_name,
                        "new_name": new_name
                    }
            else:
                return {
                    "action": "rename_file",
                    "target": " ".join(words[1:-1]),
                    "new_name": words[-1]
                }

        # Unknown Command
        return {
            "action": "unknown",
            "target": None
        }
