import logging
import re
from core.command import Command

logger = logging.getLogger(__name__)


class CommandParser:
    """Parse supported commands using deterministic keyword rules."""

    OPEN_WORDS = {"open", "launch", "start", "run"}
    CREATE_WORDS = {"create", "make"}
    DELETE_WORDS = {"delete", "remove"}
    MOVE_WORDS = {"move", "transfer"}
    RENAME_WORDS = {"rename"}
    COPY_WORDS = {"copy", "duplicate"}

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

        command = command.strip()
        words = command.split()
        verb = words[0].lower()

        if verb in self.OPEN_WORDS and len(words) > 1:
            return Command(
                "open_application",
                {"target": " ".join(words[1:])},
            )

        if verb in self.CREATE_WORDS and len(words) > 2:
            item_type = words[1].lower()

            if item_type in {"file", "folder"}:
                return Command(
                    f"create_{item_type}",
                    {"target": " ".join(words[2:])},
                )

        if verb in self.COPY_WORDS and len(words) > 1:
            return self._parse_transfer(command, verb, "copy")

        if verb in self.MOVE_WORDS and len(words) > 1:
            return self._parse_transfer(command, verb, "move")

        if verb in self.DELETE_WORDS and len(words) > 2:
            item_type = words[1].lower()

            if item_type in {"file", "folder"}:
                return Command(
                    f"delete_{item_type}",
                    {"target": " ".join(words[2:])},
                    requires_approval=True,
                )

        if (
            verb in self.RENAME_WORDS
            and len(words) > 3
            and words[1].lower() in {"file", "folder"}
        ):
            item_type = words[1].lower()
            remaining_words = words[2:]

            for index, word in enumerate(remaining_words):
                if word.lower() != "to":
                    continue

                old_name = " ".join(remaining_words[:index]).strip()
                new_name = " ".join(remaining_words[index + 1:]).strip()

                if old_name and new_name:
                    return Command(
                        f"rename_{item_type}",
                        {"target": old_name, "new_name": new_name},
                    )

                break

        logger.debug("Could not parse command: %s", command)
        return self._unknown_command()

    def _parse_transfer(
        self,
        command: str,
        verb: str,
        action: str,
    ) -> Command:
        """Parse a copy or move command.

        Args:
            command - Complete user command.
            verb - Leading command word.
            action - Action to produce, either copy or move.

        Returns:
            Command containing the transfer action, or an unknown action.
        """
        transfer_command = command[len(verb):].strip()

        parts = re.split(
            r"\s+to\s+",
            transfer_command,
            maxsplit=1,
            flags=re.IGNORECASE,
        )

        if len(parts) != 2:
            return self._unknown_command()

        source, destination = (
            part.strip() for part in parts
        )

        if not source or not destination:
            return self._unknown_command()

        return Command(
            action,
            {"source": source, "destination": destination},
        )
