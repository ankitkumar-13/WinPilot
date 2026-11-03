import os
import subprocess
import sys


SOFFICE_PATH = r"C:\Program Files\LibreOffice\program\soffice.exe"

# Supported conversion target formats
SUPPORTED_FORMATS = {
    "pdf", "docx", "doc", "odt", "xlsx", "xls", "ods",
    "pptx", "ppt", "odp", "csv", "html", "txt", "png", "jpg"
}


def _get_soffice() -> str:
    """
    Returns the path to the soffice executable.
    Raises FileNotFoundError if LibreOffice is not found.
    """
    if os.path.exists(SOFFICE_PATH):
        return SOFFICE_PATH

    raise FileNotFoundError(
        "LibreOffice not found. Expected location: "
        f"{SOFFICE_PATH}"
    )


class LibreUtils:
    """
    Utility class for LibreOffice automation.

    Uses two approaches:
        - Headless soffice (subprocess) for conversions and launching apps.
        - UNO Python bridge for live document interaction (Writer, Calc).
    """

    LIBRE_APPS = {
        "writer":  "swriter.exe",
        "calc":        "scalc.exe",
        "impress":     "simpress.exe",
        "draw":        "sdraw.exe",
        "math":        "smath.exe",
        "base":        "sbase.exe",
        "libreoffice": "soffice.exe",   # opens LibreOffice Start Center
    }

    # ------------------------------------------------------------------ #
    #  App Launching                                                        #
    # ------------------------------------------------------------------ #

    def open_app(self, app_name: str) -> bool:
        """
        Opens a LibreOffice application (Writer, Calc, Impress, Draw, etc.).
        """
        app_name = app_name.lower().strip()
        exe_name = self.LIBRE_APPS.get(app_name)

        if not exe_name:
            print(f"LibreOffice application not recognised: {app_name}")
            print(f"Supported apps: {', '.join(self.LIBRE_APPS.keys())}")
            return False

        exe_path = os.path.join(
            os.path.dirname(_get_soffice()), exe_name
        )

        try:
            subprocess.Popen([exe_path])
            print(f"Opened LibreOffice {app_name.capitalize()}.")
            return True

        except Exception as error:
            print(f"Failed to open {app_name}: {error}")
            return False

    def open_file(self, file_path: str) -> bool:
        """
        Opens any file in the appropriate LibreOffice application.
        """
        file_path = os.path.abspath(file_path.strip())

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False

        try:
            soffice = _get_soffice()
            subprocess.Popen([soffice, file_path])
            print(f"Opened in LibreOffice: {file_path}")
            return True

        except Exception as error:
            print(f"Failed to open file: {error}")
            return False

    # ------------------------------------------------------------------ #
    #  File Conversion (Headless)                                          #
    # ------------------------------------------------------------------ #

    def convert_file(
        self,
        source_path: str,
        target_format: str,
        output_dir: str = None
    ) -> bool:
        """
        Converts a document to a different format using LibreOffice headless.

        Supported target formats: pdf, docx, odt, xlsx, ods, pptx, odp,
                                   csv, html, txt, png, jpg

        Examples:
            convert_file("report.odt", "pdf")
            convert_file("data.xlsx", "csv", output_dir="C:/exports")
        """
        source_path = os.path.abspath(source_path.strip())
        target_format = target_format.lower().strip()

        if not os.path.exists(source_path):
            print(f"Source file not found: {source_path}")
            return False

        if target_format not in SUPPORTED_FORMATS:
            print(
                f"Format '{target_format}' is not supported.\n"
                f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )
            return False

        if output_dir is None:
            output_dir = os.path.dirname(source_path)
        else:
            output_dir = os.path.abspath(output_dir.strip())
            os.makedirs(output_dir, exist_ok=True)

        try:
            soffice = _get_soffice()

            result = subprocess.run(
                [
                    soffice,
                    "--headless",
                    "--convert-to", target_format,
                    "--outdir", output_dir,
                    source_path,
                ],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                base_name = os.path.splitext(os.path.basename(source_path))[0]
                output_file = os.path.join(output_dir, f"{base_name}.{target_format}")
                print(f"Converted: {source_path} -> {output_file}")
                return True
            else:
                print(f"Conversion failed: {result.stderr.strip()}")
                return False

        except subprocess.TimeoutExpired:
            print("Conversion timed out after 60 seconds.")
            return False

        except Exception as error:
            print(f"Conversion error: {error}")
            return False

    # ------------------------------------------------------------------ #
    #  Writer: Document Automation via UNO                                 #
    # ------------------------------------------------------------------ #

    def _get_uno_context(self):
        """
        Connects to a running LibreOffice instance via the UNO socket bridge.
        LibreOffice must be running with:
            soffice.exe --accept="socket,host=localhost,port=2002;urp;"
        Returns the component context or None on failure.
        """
        try:
            lo_program = os.path.join(
                os.path.dirname(_get_soffice())
            )
            if lo_program not in sys.path:
                sys.path.insert(0, lo_program)

            import uno
            from com.sun.star.beans import PropertyValue

            local_ctx = uno.getComponentContext()
            resolver = local_ctx.ServiceManager.createInstanceWithContext(
                "com.sun.star.bridge.UnoUrlResolver", local_ctx
            )
            ctx = resolver.resolve(
                "uno:socket,host=localhost,port=2002;"
                "urp;StarOffice.ComponentContext"
            )
            return ctx

        except Exception as error:
            print(
                "Could not connect to LibreOffice via UNO.\n"
                "Make sure LibreOffice is running with:\n"
                '  soffice.exe --accept="socket,host=localhost,port=2002;urp;"\n'
                f"Error: {error}"
            )
            return None

    def _start_libre_with_uno(self) -> bool:
        """
        Launches LibreOffice with the UNO socket listener enabled.
        """
        try:
            soffice = _get_soffice()
            subprocess.Popen([
                soffice,
                '--accept=socket,host=localhost,port=2002;urp;',
                '--norestore',
            ])
            import time
            time.sleep(3)
            print("LibreOffice started with UNO socket listener on port 2002.")
            return True

        except Exception as error:
            print(f"Failed to start LibreOffice with UNO: {error}")
            return False

    def create_writer_document(self, file_path: str = None) -> bool:
        """
        Creates a new blank Writer document via UNO.
        Optionally saves it immediately if file_path is given.
        """
        ctx = self._get_uno_context()
        if not ctx:
            return False

        try:
            import uno
            smgr = ctx.ServiceManager
            desktop = smgr.createInstanceWithContext(
                "com.sun.star.frame.Desktop", ctx
            )
            url = "private:factory/swriter"
            doc = desktop.loadComponentFromURL(url, "_blank", 0, ())

            if file_path:
                file_path = os.path.abspath(file_path.strip())
                file_url = uno.systemPathToFileUrl(file_path)
                doc.store()
                from com.sun.star.beans import PropertyValue
                prop = PropertyValue()
                prop.Name = "FilterName"
                prop.Value = "writer8"
                doc.storeToURL(file_url, (prop,))
                print(f"Writer document created: {file_path}")
            else:
                print("New blank Writer document created.")

            return True

        except Exception as error:
            print(f"Failed to create Writer document: {error}")
            return False

    def add_text_to_writer(
        self,
        text: str,
        file_path: str = None,
        is_heading: bool = False
    ) -> bool:
        """
        Appends text or a heading paragraph to the active Writer document.
        """
        ctx = self._get_uno_context()
        if not ctx:
            return False

        try:
            smgr = ctx.ServiceManager
            desktop = smgr.createInstanceWithContext(
                "com.sun.star.frame.Desktop", ctx
            )

            # Get active document or open specified one
            doc = desktop.getCurrentComponent()
            if not doc:
                print("No active Writer document found.")
                return False

            text_obj = doc.getText()
            cursor = text_obj.createTextCursor()
            cursor.gotoEnd(False)

            if is_heading:
                cursor.ParaStyleName = "Heading 1"
                text_obj.insertString(cursor, text, False)
                cursor.ParaStyleName = "Default Paragraph Style"
                text_obj.insertControlCharacter(
                    cursor,
                    1,  # PARAGRAPH_BREAK
                    False
                )
            else:
                text_obj.insertString(cursor, text, False)
                text_obj.insertControlCharacter(cursor, 1, False)

            doc.store()
            print(f"Text added to Writer document.")
            return True

        except Exception as error:
            print(f"Failed to add text: {error}")
            return False

    def replace_text_in_writer(
        self,
        find_text: str,
        replace_text: str
    ) -> bool:
        """
        Finds and replaces text across the active Writer document.
        """
        ctx = self._get_uno_context()
        if not ctx:
            return False

        try:
            smgr = ctx.ServiceManager
            desktop = smgr.createInstanceWithContext(
                "com.sun.star.frame.Desktop", ctx
            )
            doc = desktop.getCurrentComponent()
            if not doc:
                print("No active Writer document found.")
                return False

            search_desc = doc.createSearchDescriptor()
            search_desc.SearchRegularExpression = False
            search_desc.SearchString = find_text
            search_desc.ReplaceString = replace_text
            count = doc.replaceAll(search_desc)
            doc.store()
            print(f"Replaced {count} occurrence(s) of '{find_text}' with '{replace_text}'.")
            return True

        except Exception as error:
            print(f"Failed to replace text: {error}")
            return False

    def count_words_in_writer(self) -> bool:
        """
        Prints word and character count of the active Writer document.
        """
        ctx = self._get_uno_context()
        if not ctx:
            return False

        try:
            smgr = ctx.ServiceManager
            desktop = smgr.createInstanceWithContext(
                "com.sun.star.frame.Desktop", ctx
            )
            doc = desktop.getCurrentComponent()
            if not doc:
                print("No active Writer document found.")
                return False

            text = doc.getText().getString()
            words = len(text.split())
            chars = len(text)
            print(f"Document statistics: {words} word(s), {chars} character(s).")
            return True

        except Exception as error:
            print(f"Failed to count words: {error}")
            return False

    # ------------------------------------------------------------------ #
    #  Calc: Spreadsheet Automation via UNO                                #
    # ------------------------------------------------------------------ #

    def create_calc_spreadsheet(self, file_path: str = None) -> bool:
        """
        Creates a new blank Calc spreadsheet via UNO.
        """
        ctx = self._get_uno_context()
        if not ctx:
            return False

        try:
            import uno
            smgr = ctx.ServiceManager
            desktop = smgr.createInstanceWithContext(
                "com.sun.star.frame.Desktop", ctx
            )
            url = "private:factory/scalc"
            doc = desktop.loadComponentFromURL(url, "_blank", 0, ())

            if file_path:
                file_path = os.path.abspath(file_path.strip())
                file_url = uno.systemPathToFileUrl(file_path)
                from com.sun.star.beans import PropertyValue
                prop = PropertyValue()
                prop.Name = "FilterName"
                prop.Value = "calc8"
                doc.storeToURL(file_url, (prop,))
                print(f"Calc spreadsheet created: {file_path}")
            else:
                print("New blank Calc spreadsheet created.")

            return True

        except Exception as error:
            print(f"Failed to create Calc spreadsheet: {error}")
            return False

    def set_cell(
        self,
        cell_ref: str,
        value: str,
        sheet_index: int = 0
    ) -> bool:
        """
        Sets the value of a cell in the active Calc spreadsheet.

        Args:
            cell_ref: Cell reference string, e.g. "A1", "B3"
            value:    The string or numeric value to write
            sheet_index: Zero-based sheet index (default: 0 = first sheet)
        """
        ctx = self._get_uno_context()
        if not ctx:
            return False

        try:
            smgr = ctx.ServiceManager
            desktop = smgr.createInstanceWithContext(
                "com.sun.star.frame.Desktop", ctx
            )
            doc = desktop.getCurrentComponent()
            if not doc:
                print("No active Calc spreadsheet found.")
                return False

            sheets = doc.getSheets()
            sheet = sheets.getByIndex(sheet_index)

            # Parse cell reference e.g. "B3" -> col=1, row=2
            cell_ref = cell_ref.strip().upper()
            col_str = "".join(c for c in cell_ref if c.isalpha())
            row_str = "".join(c for c in cell_ref if c.isdigit())

            if not col_str or not row_str:
                print(f"Invalid cell reference: {cell_ref}")
                return False

            col = sum(
                (ord(c) - ord("A") + 1) * (26 ** i)
                for i, c in enumerate(reversed(col_str))
            ) - 1
            row = int(row_str) - 1

            cell = sheet.getCellByPosition(col, row)

            try:
                cell.setValue(float(value))
            except ValueError:
                cell.setString(value)

            doc.store()
            print(f"Cell {cell_ref} set to: {value}")
            return True

        except Exception as error:
            print(f"Failed to set cell: {error}")
            return False

    def get_cell(
        self,
        cell_ref: str,
        sheet_index: int = 0
    ) -> bool:
        """
        Reads and prints the value of a cell from the active Calc spreadsheet.
        """
        ctx = self._get_uno_context()
        if not ctx:
            return False

        try:
            smgr = ctx.ServiceManager
            desktop = smgr.createInstanceWithContext(
                "com.sun.star.frame.Desktop", ctx
            )
            doc = desktop.getCurrentComponent()
            if not doc:
                print("No active Calc spreadsheet found.")
                return False

            sheets = doc.getSheets()
            sheet = sheets.getByIndex(sheet_index)

            cell_ref = cell_ref.strip().upper()
            col_str = "".join(c for c in cell_ref if c.isalpha())
            row_str = "".join(c for c in cell_ref if c.isdigit())

            col = sum(
                (ord(c) - ord("A") + 1) * (26 ** i)
                for i, c in enumerate(reversed(col_str))
            ) - 1
            row = int(row_str) - 1

            cell = sheet.getCellByPosition(col, row)

            # CellType: 0=empty, 1=numeric, 2=string, 3=formula
            cell_type = cell.getType()
            if cell_type == 0:
                value = "(empty)"
            elif cell_type == 1:
                value = cell.getValue()
            else:
                value = cell.getString()

            print(f"Cell {cell_ref}: {value}")
            return True

        except Exception as error:
            print(f"Failed to get cell: {error}")
            return False

    # ------------------------------------------------------------------ #
    #  Close LibreOffice                                                   #
    # ------------------------------------------------------------------ #

    def close_libre(self) -> bool:
        """
        Closes all LibreOffice windows by terminating the process.
        """
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/IM", "soffice.exe", "/IM", "soffice.bin"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("LibreOffice closed.")
                return True
            else:
                print("LibreOffice was not running.")
                return True

        except Exception as error:
            print(f"Failed to close LibreOffice: {error}")
            return False

    def start_with_uno(self) -> bool:
        """
        Starts LibreOffice with the UNO socket listener enabled (port 2002).
        Required before using Writer or Calc UNO commands.
        """
        return self._start_libre_with_uno()
