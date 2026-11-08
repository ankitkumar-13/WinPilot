import logging
import re
from core.command import Command

logger = logging.getLogger(__name__)


class CommandParser:
    """Parse supported commands with deterministic regular expressions."""

    COPY_WORDS = {"copy", "duplicate"}

    PREFIX_PATTERN = re.compile(
        r"^(?:please|kindly)\s+|"
        r"^(?:can|could|would)\s+you\s+|"
        r"^i\s+(?:want|need)\s+(?:you\s+)?to\s+",
        flags=re.IGNORECASE,
    )

    def _unknown_command(self) -> Command:
        """Return a consistent result for unsupported commands.

        Returns:
            Command representing an unknown action.
        """
        return Command("unknown", {"target": None})

    def parse_command(self, command: str) -> Command:
        """Parse a command into an executor command object.

        Args:
            command - User command to parse.

        Returns:
            Command containing the action, arguments, and approval setting.
        """
        if not isinstance(command, str) or not command.strip():
            logger.debug("Received an empty or invalid command")
            return self._unknown_command()

        command = self._normalise_command(command)
        if not command:
            return self._unknown_command()

        open_match = re.fullmatch(
            r"(?:open|launch|start|run)\s+"
            r"(?:(?:the|a|an)\s+)?"
            r"(?:(?:application|app|program)\s+(?:(?:called|named)\s+)?)?"
            r"(.+)",
            command,
            flags=re.IGNORECASE,
        )
        if open_match:
            target = self._clean_target(open_match.group(1))
            if target:
                return Command("open_application", {"target": target})

        create_match = re.fullmatch(
            r"(?:create|make)(?:\s+me)?\s+(?:(?:a|an|the|new)\s+)*"
            r"(file|folder|directory)\s+"
            r"(?:(?:called|named)\s+)?(.+)",
            command,
            flags=re.IGNORECASE,
        )
        if create_match:
            item_type = self._normalise_item_type(create_match.group(1))
            target = self._clean_target(create_match.group(2))
            if target:
                return Command(f"create_{item_type}", {"target": target})

        delete_match = re.fullmatch(
            r"(?:delete|remove|erase)\s+(?:(?:the|a|an)\s+)*"
            r"(file|folder|directory)\s+"
            r"(?:(?:called|named)\s+)?(.+)",
            command,
            flags=re.IGNORECASE,
        )
        if delete_match:
            item_type = self._normalise_item_type(delete_match.group(1))
            target = self._clean_target(delete_match.group(2))
            if target:
                return Command(
                    f"delete_{item_type}",
                    {"target": target},
                    requires_approval=True,
                )

        rename_match = re.fullmatch(
            r"(?:rename\s+(?:(?:the|a|an)\s+)?|"
            r"change\s+(?:(?:the|a|an)\s+)?name\s+of\s+)"
            r"(?:(?:the|a|an)\s+)?(file|folder|directory)\s+"
            r"(?:(?:called|named)\s+)?"
            r"(.+?)\s+(?:to|as)\s+(.+)",
            command,
            flags=re.IGNORECASE,
        )
        if rename_match:
            item_type = self._normalise_item_type(rename_match.group(1))
            old_name = self._clean_target(rename_match.group(2))
            new_name = self._clean_target(rename_match.group(3))
            if old_name and new_name:
                return Command(
                    f"rename_{item_type}",
                    {"target": old_name, "new_name": new_name},
                )

        transfer_match = re.fullmatch(
            r"(?:move|transfer|relocate|put|copy|duplicate)\s+"
            r"(?:(?:the|a|an)\s+)?"
            r"(?:(?:file|folder|item)\s+)?"
            r"(?:(?:called|named)\s+)?"
            r"(.+?)\s+(?:to|into)\s+(.+)",
            command,
            flags=re.IGNORECASE,
        )
        if transfer_match:
            source = self._clean_target(transfer_match.group(1))
            destination = self._clean_target(transfer_match.group(2))
            if source and destination:
                verb = command.split(maxsplit=1)[0].lower()
                action = "copy" if verb in self.COPY_WORDS else "move"
                return Command(
                    action,
                    {"source": source, "destination": destination},
                )

        logger.debug("Could not parse command: %s", command)
        return self._unknown_command()

    @classmethod
    def _normalise_command(cls, command: str) -> str:
        """Remove common polite prefixes and surrounding whitespace.

        Args:
            command - User command to normalise.

        Returns:
            Command text without leading polite phrases.
        """
        command = command.strip()
        while True:
            stripped = cls.PREFIX_PATTERN.sub("", command, count=1)
            if stripped == command:
                return command
            command = stripped.strip()

    @staticmethod
    def _normalise_item_type(item_type: str) -> str:
        """Convert item type aliases to the executor's supported names.

        Args:
            item_type - File, folder, or directory label from a command.

        Returns:
            Either file or folder.
        """
        return "folder" if item_type.lower() in {"folder", "directory"} else "file"

    @staticmethod
    def _clean_target(target: str) -> str:
        """Trim a captured name and remove matching surrounding quotes.

        Args:
            target - Captured path, file name, folder name, or application.

        Returns:
            Clean target text.
        """
        target = target.strip()
        if (
            len(target) >= 2
            and target[0] == target[-1]
            and target[0] in {"'", '"'}
        ):
            target = target[1:-1].strip()
        return target
