import re

class CommandParser:

    OPEN_WORDS = ["open", "launch", "start", "run"]

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

            parts = re.split("\s+to\s+", move_command, maxsplit=1, flags=re.IGNORECASE)

            if len(parts) == 2:
                return {
                    "action": "move",
                    "source": parts[0].strip(),
                    "destination": parts[1].strip()
                }

        # Unknown Command
        return {
            "action": "unknown",
            "target": None
        }
