"""
Intent Classifier for WinPilot.

Uses TF-IDF vectorisation + Logistic Regression to map natural language
sentences to structured action dictionaries — identical in shape to what
CommandParser already returns, so executor.py needs zero changes.

Two-stage pipeline:
    1. Intent classification  — "what does the user want to do?"
    2. Entity extraction      — "which file / format / cell / text?"
"""

import re
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------------ #
#  Training Data                                                       #
#  Format: (example_phrase, intent_label)                             #
# ------------------------------------------------------------------ #

TRAINING_DATA = [

    # --- libre_open_libre (open LibreOffice without specifying an app) ---
    ("open libre",                          "libre_open_libre"),
    ("open liber",                          "libre_open_libre"),
    ("open libreoffice",                    "libre_open_libre"),
    ("launch libre",                        "libre_open_libre"),
    ("launch libreoffice",                  "libre_open_libre"),
    ("start libre",                         "libre_open_libre"),
    ("start libreoffice",                   "libre_open_libre"),
    ("run libre",                           "libre_open_libre"),
    ("run libreoffice",                     "libre_open_libre"),
    ("open libre office",                   "libre_open_libre"),
    ("launch libre office",                 "libre_open_libre"),

    # --- libre_open_writer ---
    ("open writer",                         "libre_open_writer"),
    ("launch writer",                       "libre_open_writer"),
    ("start writer",                        "libre_open_writer"),
    ("open libreoffice writer",             "libre_open_writer"),
    ("launch libreoffice writer",           "libre_open_writer"),
    ("open a text editor",                  "libre_open_writer"),
    ("i want to write something",           "libre_open_writer"),
    ("open word processor",                 "libre_open_writer"),
    ("open document editor",                "libre_open_writer"),
    ("start a document",                    "libre_open_writer"),
    ("open libre writer",                   "libre_open_writer"),
    ("open writing app",                    "libre_open_writer"),
    ("give me a writer",                    "libre_open_writer"),
    ("open text document",                  "libre_open_writer"),

    # --- libre_open_calc ---
    ("open calc",                           "libre_open_calc"),
    ("launch calc",                         "libre_open_calc"),
    ("open spreadsheet",                    "libre_open_calc"),
    ("launch spreadsheet",                  "libre_open_calc"),
    ("open libreoffice calc",               "libre_open_calc"),
    ("i need a spreadsheet",                "libre_open_calc"),
    ("start a spreadsheet",                 "libre_open_calc"),
    ("open excel alternative",              "libre_open_calc"),
    ("open libre calc",                     "libre_open_calc"),
    ("open a sheet",                        "libre_open_calc"),
    ("give me a spreadsheet",               "libre_open_calc"),
    ("start calc",                          "libre_open_calc"),
    ("open data sheet",                     "libre_open_calc"),

    # --- libre_open_impress ---
    ("open impress",                        "libre_open_impress"),
    ("launch impress",                      "libre_open_impress"),
    ("open presentation",                   "libre_open_impress"),
    ("start a presentation",                "libre_open_impress"),
    ("open libreoffice impress",            "libre_open_impress"),
    ("create a slideshow",                  "libre_open_impress"),
    ("open libre impress",                  "libre_open_impress"),
    ("open slide editor",                   "libre_open_impress"),
    ("make a presentation",                 "libre_open_impress"),
    ("start presentation software",        "libre_open_impress"),
    ("i want to make slides",               "libre_open_impress"),

    # --- libre_open_file ---
    ("open the file",                       "libre_open_file"),
    ("open this file",                      "libre_open_file"),
    ("open that document",                  "libre_open_file"),
    ("launch this document",                "libre_open_file"),
    ("open document in libreoffice",        "libre_open_file"),
    ("open file in libreoffice",            "libre_open_file"),
    ("open report in libre",                "libre_open_file"),
    ("load this file",                      "libre_open_file"),

    # --- libre_convert ---
    ("convert to pdf",                      "libre_convert"),
    ("export to pdf",                       "libre_convert"),
    ("save as pdf",                         "libre_convert"),
    ("make a pdf",                          "libre_convert"),
    ("convert the document to pdf",         "libre_convert"),
    ("convert file to pdf",                 "libre_convert"),
    ("export file as pdf",                  "libre_convert"),
    ("convert to docx",                     "libre_convert"),
    ("convert to word",                     "libre_convert"),
    ("convert to excel",                    "libre_convert"),
    ("convert to xlsx",                     "libre_convert"),
    ("convert to csv",                      "libre_convert"),
    ("convert to odt",                      "libre_convert"),
    ("change file format",                  "libre_convert"),
    ("convert document",                    "libre_convert"),
    ("export as word",                      "libre_convert"),
    ("turn it into a pdf",                  "libre_convert"),
    ("make it a pdf",                       "libre_convert"),
    ("save document as pdf",                "libre_convert"),
    ("convert this to pdf",                 "libre_convert"),
    ("convert report to pdf",               "libre_convert"),

    # --- libre_start_uno ---
    ("start uno",                           "libre_start_uno"),
    ("start libreoffice with uno",          "libre_start_uno"),
    ("enable uno",                          "libre_start_uno"),
    ("start live editing",                  "libre_start_uno"),
    ("connect to libreoffice",              "libre_start_uno"),
    ("start uno listener",                  "libre_start_uno"),
    ("enable live editing mode",            "libre_start_uno"),
    ("activate uno",                        "libre_start_uno"),
    ("launch libreoffice for editing",      "libre_start_uno"),

    # --- libre_create_writer ---
    ("create a writer document",            "libre_create_writer"),
    ("make a new document",                 "libre_create_writer"),
    ("new writer file",                     "libre_create_writer"),
    ("create new document",                 "libre_create_writer"),
    ("make a document",                     "libre_create_writer"),
    ("new odt file",                        "libre_create_writer"),
    ("create a text document",              "libre_create_writer"),
    ("start a new writer file",             "libre_create_writer"),
    ("create writer file",                  "libre_create_writer"),
    ("new document in writer",              "libre_create_writer"),

    # --- libre_create_calc ---
    ("create a spreadsheet",                "libre_create_calc"),
    ("make a new spreadsheet",              "libre_create_calc"),
    ("new calc file",                       "libre_create_calc"),
    ("create spreadsheet",                  "libre_create_calc"),
    ("new ods file",                        "libre_create_calc"),
    ("create a new sheet",                  "libre_create_calc"),
    ("make a calc file",                    "libre_create_calc"),
    ("start a new spreadsheet",             "libre_create_calc"),
    ("create spreadsheet file",             "libre_create_calc"),

    # --- libre_add_text ---
    ("add text",                            "libre_add_text"),
    ("append text",                         "libre_add_text"),
    ("write text to document",              "libre_add_text"),
    ("insert text",                         "libre_add_text"),
    ("add paragraph",                       "libre_add_text"),
    ("add some text",                       "libre_add_text"),
    ("type into document",                  "libre_add_text"),
    ("append to document",                  "libre_add_text"),
    ("add content to document",             "libre_add_text"),

    # --- libre_add_heading ---
    ("add heading",                         "libre_add_heading"),
    ("insert heading",                      "libre_add_heading"),
    ("add a title",                         "libre_add_heading"),
    ("insert a heading",                    "libre_add_heading"),
    ("add chapter title",                   "libre_add_heading"),
    ("add section heading",                 "libre_add_heading"),
    ("add a section title",                 "libre_add_heading"),

    # --- libre_replace ---
    ("replace text",                        "libre_replace"),
    ("find and replace",                    "libre_replace"),
    ("replace word",                        "libre_replace"),
    ("change word in document",             "libre_replace"),
    ("find replace",                        "libre_replace"),
    ("substitute text",                     "libre_replace"),
    ("swap text",                           "libre_replace"),
    ("change text in document",             "libre_replace"),
    ("replace a word",                      "libre_replace"),

    # --- libre_count_words ---
    ("count words",                         "libre_count_words"),
    ("how many words",                      "libre_count_words"),
    ("word count",                          "libre_count_words"),
    ("get word count",                      "libre_count_words"),
    ("how long is the document",            "libre_count_words"),
    ("count characters",                    "libre_count_words"),
    ("document statistics",                 "libre_count_words"),
    ("show word count",                     "libre_count_words"),

    # --- libre_set_cell ---
    ("set cell",                            "libre_set_cell"),
    ("write to cell",                       "libre_set_cell"),
    ("put value in cell",                   "libre_set_cell"),
    ("enter value in cell",                 "libre_set_cell"),
    ("fill cell",                           "libre_set_cell"),
    ("write in spreadsheet cell",           "libre_set_cell"),
    ("insert value into cell",              "libre_set_cell"),
    ("set a cell value",                    "libre_set_cell"),

    # --- libre_get_cell ---
    ("get cell",                            "libre_get_cell"),
    ("read cell",                           "libre_get_cell"),
    ("what is in cell",                     "libre_get_cell"),
    ("show cell value",                     "libre_get_cell"),
    ("read value from cell",                "libre_get_cell"),
    ("get cell value",                      "libre_get_cell"),
    ("fetch cell",                          "libre_get_cell"),
    ("what is the value of cell",           "libre_get_cell"),

    # --- libre_close ---
    ("close libreoffice",                   "libre_close"),
    ("close libre",                         "libre_close"),
    ("close it",                            "libre_close"),
    ("shut down libreoffice",               "libre_close"),
    ("exit libreoffice",                    "libre_close"),
    ("quit libreoffice",                    "libre_close"),
    ("kill libreoffice",                    "libre_close"),
    ("stop libreoffice",                    "libre_close"),
    ("close all libreoffice windows",       "libre_close"),
    ("shut down libre",                     "libre_close"),
    ("exit libre",                          "libre_close"),
    ("close the office app",               "libre_close"),

    # --- open_application (non-libre) ---
    ("open notepad",                        "open_application"),
    ("launch notepad",                      "open_application"),
    ("open calculator",                     "open_application"),
    ("open paint",                          "open_application"),
    ("launch calculator",                   "open_application"),
    ("start paint",                         "open_application"),

    # --- create_file ---
    ("create a file",                       "create_file"),
    ("make a file",                         "create_file"),
    ("new file",                            "create_file"),
    ("create new file",                     "create_file"),

    # --- create_folder ---
    ("create a folder",                     "create_folder"),
    ("make a folder",                       "create_folder"),
    ("new folder",                          "create_folder"),
    ("create directory",                    "create_folder"),
    ("make a new folder",                   "create_folder"),

    # --- delete_file ---
    ("delete file",                         "delete_file"),
    ("remove file",                         "delete_file"),
    ("delete a file",                       "delete_file"),

    # --- delete_folder ---
    ("delete folder",                       "delete_folder"),
    ("remove folder",                       "delete_folder"),
    ("delete directory",                    "delete_folder"),

    # --- move ---
    ("move file",                           "move"),
    ("transfer file",                       "move"),
    ("move the file",                       "move"),

    # --- copy ---
    ("copy file",                           "copy"),
    ("duplicate file",                      "copy"),
    ("copy the file",                       "copy"),

    # --- rename_file ---
    ("rename file",                         "rename_file"),
    ("rename a file",                       "rename_file"),

    # --- rename_folder ---
    ("rename folder",                       "rename_folder"),
    ("rename a folder",                     "rename_folder"),
]


# ------------------------------------------------------------------ #
#  Entity Extraction Patterns                                          #
# ------------------------------------------------------------------ #

# Supported conversion formats
FORMATS = {
    "pdf", "docx", "doc", "odt", "xlsx", "xls", "ods",
    "pptx", "ppt", "odp", "csv", "html", "txt", "png", "jpg"
}

# Detect file paths / filenames (with extension)
FILE_PATTERN = re.compile(
    r'["\']?([A-Za-z]:\\[\w\s\\.-]+\.[\w]+|'   # absolute Windows path
    r'[\w\s.-]+\.[\w]{2,5})["\']?',              # relative filename
    re.IGNORECASE
)

# Detect cell references like A1, B23, AA5
CELL_PATTERN = re.compile(r'\b([A-Z]{1,3}\d{1,5})\b', re.IGNORECASE)

# Detect target format
FORMAT_PATTERN = re.compile(
    r'\b(pdf|docx?|odt|xlsx?|ods|pptx?|odp|csv|html?|txt|png|jpg)\b',
    re.IGNORECASE
)

# Detect output directory
OUTPUT_DIR_PATTERN = re.compile(
    r'(?:in|to|into|at|save to|output to)\s+["\']?([A-Za-z]:\\[\w\s\\.-]+|'
    r'[\w/\\][\w\s/\\.-]+)["\']?$',
    re.IGNORECASE
)

# Detect quoted or trailing text content
QUOTED_TEXT_PATTERN = re.compile(r'["\'](.+?)["\']')


def _extract_files(text: str) -> list:
    return [m.strip() for m in FILE_PATTERN.findall(text)]


def _extract_format(text: str) -> Optional[str]:
    match = FORMAT_PATTERN.search(text)
    return match.group(1).lower() if match else None


def _extract_cell(text: str) -> Optional[str]:
    match = CELL_PATTERN.search(text)
    return match.group(1).upper() if match else None


def _extract_output_dir(text: str) -> Optional[str]:
    match = OUTPUT_DIR_PATTERN.search(text)
    return match.group(1).strip() if match else None


def _extract_quoted_or_trailing(text: str, keyword: str) -> Optional[str]:
    """
    Extracts quoted text, or the content after a keyword like 'text', 'heading'.
    """
    quoted = QUOTED_TEXT_PATTERN.findall(text)
    if quoted:
        return quoted[0]

    # Try everything after the keyword
    kw_match = re.search(
        rf'\b{re.escape(keyword)}\b\s+(.+)',
        text,
        re.IGNORECASE
    )
    if kw_match:
        return kw_match.group(1).strip().strip("\"'")
    return None


# ------------------------------------------------------------------ #
#  Intent Classifier                                                   #
# ------------------------------------------------------------------ #

class IntentClassifier:
    """
    Trains a TF-IDF + Logistic Regression pipeline on startup and
    predicts structured command dicts from natural language input.

    Usage:
        clf = IntentClassifier()
        result = clf.predict("turn this spreadsheet into a pdf")
        # → {"action": "libre_convert", "source": ..., "format": "pdf", ...}
    """

    CONFIDENCE_THRESHOLD = 0.30

    def __init__(self):
        self._pipeline = None
        self._labels = None
        self._train()

    def _train(self):
        texts  = [item[0] for item in TRAINING_DATA]
        labels = [item[1] for item in TRAINING_DATA]

        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                sublinear_tf=True,
                min_df=1,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=5.0,
                solver="lbfgs",
            )),
        ])

        self._pipeline.fit(texts, labels)
        self._labels = list(set(labels))

    def predict(self, text: str) -> dict:
        """
        Returns a command dict from a natural language sentence.
        Returns {"action": "unknown", "target": None} if confidence is low.
        """
        text_lower = text.lower().strip()

        proba = self._pipeline.predict_proba([text_lower])[0]
        classes = self._pipeline.classes_
        best_idx = proba.argmax()
        confidence = proba[best_idx]
        intent = classes[best_idx]

        if confidence < self.CONFIDENCE_THRESHOLD:
            return {"action": "unknown", "target": None}

        return self._build_command(intent, text)

    def _build_command(self, intent: str, original_text: str) -> dict:
        """
        Attaches extracted entities to the intent to build a full command dict.
        """
        files      = _extract_files(original_text)
        fmt        = _extract_format(original_text)
        cell       = _extract_cell(original_text)
        output_dir = _extract_output_dir(original_text)

        # ---- LibreOffice ----

        if intent == "libre_open_libre":
            # Open LibreOffice Start Center (no specific app selected)
            return {"action": "libre_open_app", "target": "libreoffice"}

        if intent == "libre_open_writer":
            return {"action": "libre_open_app", "target": "writer"}

        if intent == "libre_open_calc":
            return {"action": "libre_open_app", "target": "calc"}

        if intent == "libre_open_impress":
            return {"action": "libre_open_app", "target": "impress"}

        if intent == "libre_open_file":
            return {
                "action": "libre_open_file",
                "target": files[0] if files else None
            }

        if intent == "libre_convert":
            # Smarter extraction: "convert <source> to <format>"
            conv_match = re.search(
                r'(?:convert|export|save|turn)\s+(.+?)\s+(?:to|as|into)\s+(\w+)',
                original_text, re.IGNORECASE
            )
            source_file = None
            target_fmt = fmt  # fallback to regex-extracted format
            if conv_match:
                raw_source = conv_match.group(1).strip().strip("\"'")
                raw_fmt = conv_match.group(2).strip().lower()
                # Only use conv_match source if it's not a noise word
                noise = {"the", "a", "this", "that", "document", "file", "it"}
                source_file = None if raw_source.lower() in noise else raw_source
                if raw_fmt in FORMATS:
                    target_fmt = raw_fmt
            if source_file is None and files:
                source_file = files[0]
            return {
                "action":     "libre_convert",
                "source":     source_file,
                "format":     target_fmt,
                "output_dir": output_dir
            }

        if intent == "libre_start_uno":
            return {"action": "libre_start_uno"}

        if intent == "libre_create_writer":
            return {
                "action": "libre_create_writer",
                "target": files[0] if files else None
            }

        if intent == "libre_create_calc":
            return {
                "action": "libre_create_calc",
                "target": files[0] if files else None
            }

        if intent == "libre_add_text":
            content = _extract_quoted_or_trailing(original_text, "text")
            return {
                "action": "libre_add_text",
                "text": content,
                "target": files[0] if files else None,
                "heading": False
            }

        if intent == "libre_add_heading":
            content = _extract_quoted_or_trailing(original_text, "heading")
            return {
                "action": "libre_add_text",
                "text": content,
                "target": files[0] if files else None,
                "heading": True
            }

        if intent == "libre_replace":
            # Pattern: replace X with Y  /  change X to Y
            match = re.search(
                r'(?:replace|change|swap|substitute)\s+["\']?(.+?)["\']?\s+'
                r'(?:with|to|by)\s+["\']?(.+?)["\']?$',
                original_text,
                re.IGNORECASE
            )
            if match:
                return {
                    "action": "libre_replace",
                    "find_text": match.group(1).strip(),
                    "replace_text": match.group(2).strip()
                }
            return {"action": "libre_replace", "find_text": None, "replace_text": None}

        if intent == "libre_count_words":
            return {"action": "libre_count_words"}

        if intent == "libre_set_cell":
            # Pattern: set cell A1 Revenue  |  put 500 in B2
            match = re.search(
                r'(?:set|write|fill|enter|put|insert)\s+(?:cell\s+)?'
                r'([A-Z]{1,3}\d{1,5})\s+(.+)',
                original_text,
                re.IGNORECASE
            )
            if match:
                return {
                    "action": "libre_set_cell",
                    "cell": match.group(1).upper(),
                    "value": match.group(2).strip().strip("\"'")
                }
            # Fallback: just cell ref
            return {
                "action": "libre_set_cell",
                "cell": cell,
                "value": None
            }

        if intent == "libre_get_cell":
            return {
                "action": "libre_get_cell",
                "cell": cell
            }

        if intent == "libre_close":
            return {"action": "libre_close"}

        # ---- Standard commands ----

        if intent == "open_application":
            apps = {"notepad", "calculator", "paint"}
            for app in apps:
                if app in original_text.lower():
                    return {"action": "open_application", "target": app}
            return {"action": "open_application", "target": original_text}

        if intent == "create_file":
            return {"action": "create_file", "target": files[0] if files else None}

        if intent == "create_folder":
            # Extract folder name — last word(s) that aren't keywords
            parts = re.sub(
                r'\b(create|make|new|folder|directory|a)\b', '', original_text,
                flags=re.IGNORECASE
            ).strip()
            return {"action": "create_folder", "target": parts or None}

        if intent == "delete_file":
            return {"action": "delete_file", "target": files[0] if files else None}

        if intent == "delete_folder":
            return {"action": "delete_folder", "target": files[0] if files else None}

        if intent == "move":
            return {
                "action": "move",
                "source": files[0] if len(files) > 0 else None,
                "destination": files[1] if len(files) > 1 else None
            }

        if intent == "copy":
            return {
                "action": "copy",
                "source": files[0] if len(files) > 0 else None,
                "destination": files[1] if len(files) > 1 else None
            }

        if intent == "rename_file":
            return {"action": "rename_file", "target": files[0] if files else None, "new_name": None}

        if intent == "rename_folder":
            return {"action": "rename_folder", "target": files[0] if files else None, "new_name": None}

        return {"action": "unknown", "target": None}
