class CommandParser:

    OPEN_WORDS = ["open", "launch", "start", "run"]

    def parse_command(self, command):
        """
            Description: Parses the given command and returns a dictionary with the action and target.
        """
        command = command.strip().lower()

        if not command:
            return {
                "action": "unknown",
                "target": None
            }
        words = command.split()

        # Open Application
        if words[0] in self.OPEN_WORDS and len(words) > 1:
            return {
                "action": "open_application",
                "target": " ".join(words[1:])
            }


        # Unknown Command
        return {
            "action": "unknown",
            "target": None
        }
