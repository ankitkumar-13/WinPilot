"""
CommandParser — two-stage command resolution for WinPilot.

Stage 1: Fast keyword/regex matching (original logic, zero latency).
Stage 2: ML intent classifier fallback (handles natural language).

The executor.py receives the same dict structure from both stages.
"""

import re


class CommandParser:

    OPEN_WORDS   = ["open", "launch", "start", "run"]
    CREATE_WORDS = ["create", "make"]
    DELETE_WORDS = ["delete", "remove"]
    MOVE_WORDS   = ["move", "transfer"]
    RENAME_WORDS = ["rename"]
    COPY_WORDS   = ["copy", "duplicate"]

    def __init__(self):
        # Lazy-load the ML classifier so startup stays instant
        self._classifier = None

    def _get_classifier(self):
        if self._classifier is None:
            try:
                from core.intent_classifier import IntentClassifier
                self._classifier = IntentClassifier()
            except Exception as e:
                print(f"[Warning] ML classifier unavailable: {e}")
                self._classifier = False   # mark as failed so we don't retry
        return self._classifier if self._classifier else None

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def parse_command(self, command: str) -> dict:
        """
        Parses a command string and returns a structured action dict.

        Tries fast keyword matching first; falls back to the ML
        intent classifier for natural language input.
        """
        command = command.strip()

        if not command:
            return {"action": "unknown", "target": None}

        # Stage 1: keyword / regex matching
        result = self._keyword_parse(command)

        if result["action"] != "unknown":
            return result

        # Stage 2: ML intent classifier
        clf = self._get_classifier()
        if clf:
            ml_result = clf.predict(command)
            if ml_result["action"] != "unknown":
                return ml_result

        return {"action": "unknown", "target": None}

    # ------------------------------------------------------------------ #
    #  Stage 1: Fast keyword / regex matching                              #
    # ------------------------------------------------------------------ #

    def _keyword_parse(self, command: str) -> dict:
        words = command.split()
        first = words[0].lower()

        # ---- LibreOffice commands (prefix: "libre") ----

        # Aliases that mean "open LibreOffice" without specifying an app
        LIBRE_ALIASES = {"libre", "liber", "libreoffice", "libre office", "liber office"}

        # "open libre", "launch libreoffice", "start liber" etc.
        if first in self.OPEN_WORDS + ["start", "launch", "run"] and len(words) > 1:
            rest = " ".join(words[1:]).lower().strip()
            if rest in LIBRE_ALIASES:
                return {"action": "libre_open_app", "target": "libreoffice"}

        if first == "libre" and len(words) > 1:
            sub = words[1].lower()

            if sub in ["start", "launch"] and len(words) > 2 and words[2].lower() == "uno":
                return {"action": "libre_start_uno"}

            if sub in ["close", "quit", "exit"]:
                return {"action": "libre_close"}

            if sub in ["open", "launch", "start"] and len(words) > 2:
                app_or_file = " ".join(words[2:]).strip()
                libre_apps = {"writer", "calc", "impress", "draw", "math", "base", "libreoffice"}
                if app_or_file.lower() in libre_apps:
                    return {"action": "libre_open_app", "target": app_or_file.lower()}
                return {"action": "libre_open_file", "target": app_or_file}

            if sub in ["convert", "export"]:
                pattern = r"libre\s+(?:convert|export)\s+(.+?)\s+to\s+(\w+)(?:\s+in\s+(.+))?$"
                match = re.match(pattern, command, re.IGNORECASE)
                if match:
                    return {
                        "action":     "libre_convert",
                        "source":     match.group(1).strip().strip("\"'"),
                        "format":     match.group(2).strip().lower(),
                        "output_dir": match.group(3).strip().strip("\"'") if match.group(3) else None
                    }

            if sub == "create" and len(words) > 3 and words[2].lower() == "writer":
                return {
                    "action": "libre_create_writer",
                    "target": " ".join(words[3:]).strip().strip("\"'")
                }

            if sub == "create" and len(words) > 3 and words[2].lower() == "calc":
                return {
                    "action": "libre_create_calc",
                    "target": " ".join(words[3:]).strip().strip("\"'")
                }

            if sub in ["add", "append"] and len(words) > 3:
                is_heading = words[2].lower() == "heading"
                pattern = r"libre\s+(?:add|append)\s+(?:text\s+|heading\s+)?(.+?)\s+to\s+(.+)$"
                match = re.match(pattern, command, re.IGNORECASE)
                if match:
                    return {
                        "action":  "libre_add_text",
                        "text":    match.group(1).strip().strip("\"'"),
                        "target":  match.group(2).strip().strip("\"'"),
                        "heading": is_heading
                    }

            if sub == "replace" and len(words) > 4:
                pattern = r"libre\s+replace\s+[\"']?(.+?)[\"']?\s+with\s+[\"']?(.+?)[\"']?$"
                match = re.match(pattern, command, re.IGNORECASE)
                if match:
                    return {
                        "action":       "libre_replace",
                        "find_text":    match.group(1).strip(),
                        "replace_text": match.group(2).strip()
                    }

            if sub == "count" and len(words) > 2 and words[2].lower() == "words":
                return {"action": "libre_count_words"}

            if sub == "set" and len(words) > 3 and words[2].lower() == "cell":
                if len(words) >= 5:
                    return {
                        "action": "libre_set_cell",
                        "cell":   words[3].upper(),
                        "value":  " ".join(words[4:]).strip().strip("\"'")
                    }

            if sub == "get" and len(words) > 3 and words[2].lower() == "cell":
                return {"action": "libre_get_cell", "cell": words[3].upper()}

        # close libre (alternative order)
        if first in ["close", "quit"] and len(words) > 1 and words[1].lower() == "libre":
            return {"action": "libre_close"}

        # ---- Standard File & Folder commands ----
        # Only match exact known application names to avoid intercepting
        # natural language phrases like "open libreoffice writer"
        KNOWN_APPS = {"notepad", "calculator", "paint", "calc", "mspaint"}
        if first in self.OPEN_WORDS and len(words) > 1:
            second = words[1].lower()
            if second in KNOWN_APPS:
                return {"action": "open_application", "target": " ".join(words[1:])}

        if len(words) > 2 and first in self.CREATE_WORDS and words[1].lower() == "folder":
            return {"action": "create_folder", "target": " ".join(words[2:])}

        if len(words) > 2 and first in self.CREATE_WORDS and words[1].lower() == "file":
            return {"action": "create_file", "target": " ".join(words[2:])}

        if first in self.COPY_WORDS and len(words) > 1:
            parts = re.split(r"\s+to\s+", command[len(words[0]):].strip(), maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                return {"action": "copy", "source": parts[0].strip(), "destination": parts[1].strip()}

        if first in self.MOVE_WORDS and len(words) > 1:
            parts = re.split(r"\s+to\s+", command[len(words[0]):].strip(), maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                return {"action": "move", "source": parts[0].strip(), "destination": parts[1].strip()}

        if len(words) > 2 and first in self.DELETE_WORDS and words[1].lower() == "file":
            return {"action": "delete_file", "target": " ".join(words[2:])}

        if len(words) > 2 and first in self.DELETE_WORDS and words[1].lower() == "folder":
            return {"action": "delete_folder", "target": " ".join(words[2:])}

        if len(words) > 2 and first in self.RENAME_WORDS and words[1].lower() in ["file", "folder"]:
            lower_words = [w.lower() for w in words]
            if "to" in lower_words[2:]:
                to_idx   = lower_words.index("to", 2)
                old_name = " ".join(words[2:to_idx])
                new_name = " ".join(words[to_idx + 1:])
                if old_name and new_name:
                    return {
                        "action":   f"rename_{words[1].lower()}",
                        "target":   old_name,
                        "new_name": new_name
                    }

        return {"action": "unknown", "target": None}
