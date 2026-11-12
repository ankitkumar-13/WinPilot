"""Discover and launch installed Windows applications."""

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALIASES_FILE = PROJECT_ROOT / "data" / "application_aliases.json"


class WindowsUtils:
    """Handle Windows application discovery and launching."""

    @staticmethod
    def _normalise_name(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    def _aliases(self) -> dict[str, str]:
        try:
            data = json.loads(ALIASES_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not load application aliases from %s", ALIASES_FILE)
            return {}

    def _search_index_applications(self) -> list[tuple[str, str]]:
        """Query Windows Search's local index without opening its UI."""
        if sys.platform != "win32":
            return []
        try:
            import win32com.client

            roots = [
                Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
                Path(os.environ.get("PROGRAMDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
            ]
            connection = win32com.client.Dispatch("ADODB.Connection")
            connection.Open(
                "Provider=Search.CollatorDSO;Extended Properties='Application=Windows';"
            )
            results: list[tuple[str, str]] = []
            try:
                for root in roots:
                    if not root.is_dir():
                        continue
                    scope = root.resolve().as_posix().replace("'", "''")
                    sql = (
                        "SELECT System.ItemNameDisplay, System.ItemPathDisplay "
                        "FROM SystemIndex "
                        f"WHERE SCOPE='file:{scope}' "
                        "AND (System.FileExtension='.lnk' OR System.FileExtension='.url')"
                    )
                    recordset = connection.Execute(sql)[0]
                    try:
                        while not recordset.EOF:
                            name = str(recordset.Fields.Item(0).Value or "").strip()
                            target = str(recordset.Fields.Item(1).Value or "").strip()
                            if name and target and Path(target).is_file():
                                results.append((Path(name).stem, target))
                            recordset.MoveNext()
                    finally:
                        recordset.Close()
            finally:
                connection.Close()
            return results
        except Exception:
            # Windows Search may be disabled, unavailable, or not have indexed
            # Start Menu entries yet. Direct shortcut scanning remains active.
            logger.debug("Windows Search index lookup unavailable", exc_info=True)
            return []

    def _start_apps_applications(self) -> list[tuple[str, str]]:
        """Read Windows' registered Start Apps catalog without showing UI."""
        if sys.platform != "win32":
            return []
        try:
            result = subprocess.run(
                [
                    "powershell.exe",
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress",
                ],
                capture_output=True,
                text=True,
                timeout=12,
                check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            items = json.loads(result.stdout or "[]")
            if isinstance(items, dict):
                items = [items]
            return [
                (str(item["Name"]).strip(), str(item["AppID"]).strip())
                for item in items
                if isinstance(item, dict) and item.get("Name") and item.get("AppID")
            ]
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError, TypeError):
            logger.debug("Windows Start Apps catalog lookup unavailable", exc_info=True)
            return []

    def list_available_applications(self) -> list[dict[str, str]]:
        """Return applications found on this Windows installation right now.

        Start Menu shortcuts are the primary source, with registered app
        executables as a fallback/additional source.
        """
        if sys.platform != "win32":
            return []

        found: dict[str, tuple[str, str]] = {}
        priorities: dict[str, int] = {}

        def add(name: str, target: str, *, priority: int = 1) -> None:
            key = self._normalise_name(name)
            if key not in found or priority >= priorities[key]:
                found[key] = (name, target)
                priorities[key] = priority

        # AppsFolder is the same registered-app catalog Windows Search uses
        # for results labeled "App". It includes classic and packaged apps.
        try:
            import win32com.client

            namespace = win32com.client.Dispatch("Shell.Application").NameSpace(
                "shell:AppsFolder"
            )
            if namespace is not None:
                for item in namespace.Items():
                    name = str(item.Name or "").strip()
                    target = str(item.Path or "").strip()
                    if name and target:
                        if not Path(target).is_file() and not target.lower().endswith((".lnk", ".url")):
                            target = f"apps-folder:{target}"
                        add(name, target, priority=3)
        except Exception:
            logger.debug("Windows AppsFolder lookup unavailable", exc_info=True)

        # This is the registered app catalog exposed to Start search. PowerShell
        # runs non-interactively with CREATE_NO_WINDOW, so no search UI appears.
        for name, app_id in self._start_apps_applications():
            target = app_id
            if not Path(app_id).is_file() and not app_id.lower().endswith((".lnk", ".url")):
                target = f"apps-folder:{app_id}"
            add(name, target, priority=4)

        start_menu = [
            Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
            Path(os.environ.get("PROGRAMDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
        ]
        for root in start_menu:
            if not root.is_dir():
                continue
            try:
                for shortcut in root.rglob("*.lnk"):
                    add(shortcut.stem, str(shortcut), priority=2)
                for shortcut in root.rglob("*.url"):
                    add(shortcut.stem, str(shortcut), priority=2)
            except OSError:
                logger.debug("Unable to scan Start Menu folder %s", root, exc_info=True)

        # Use the background Windows Search index to pick up indexed shortcuts
        # that may not be found by direct traversal (for example, virtualized
        # Start Menu entries). This does not display or invoke the search UI.
        for name, target in self._search_index_applications():
            add(name, target, priority=2)

        try:
            import winreg

            uninstall_paths = (
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
                r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
            )
            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                for subkey in uninstall_paths:
                    try:
                        with winreg.OpenKey(hive, subkey) as key:
                            for index in range(winreg.QueryInfoKey(key)[0]):
                                try:
                                    child_name = winreg.EnumKey(key, index)
                                    with winreg.OpenKey(key, child_name) as child:
                                        name = winreg.QueryValueEx(child, "DisplayName")[0]
                                        command = winreg.QueryValueEx(child, "DisplayIcon")[0]
                                    executable = command.strip().strip('"').split(",", 1)[0]
                                    if (
                                        name
                                        and Path(executable).suffix.casefold() == ".exe"
                                        and Path(executable).is_file()
                                    ):
                                        add(str(name), executable, priority=1)
                                except (OSError, ValueError, TypeError):
                                    continue
                    except OSError:
                        continue
        except ImportError:
            pass

        return [
            {"name": name, "target": target}
            for name, target in sorted(found.values(), key=lambda item: item[0].casefold())
        ]

    def _resolve_application(self, requested: str) -> tuple[str, str] | None:
        normalized = self._normalise_name(requested)
        aliases = self._aliases()
        alias_targets = {
            self._normalise_name(alias): canonical
            for canonical, names in aliases.items()
            for alias in [canonical, *(names if isinstance(names, list) else [])]
        }
        canonical = alias_targets.get(normalized, requested)
        applications = self.list_available_applications()
        requested_aliases = {
            self._normalise_name(alias)
            for alias in aliases.get(canonical, [])
            if isinstance(alias, str)
        }
        requested_aliases.add(self._normalise_name(canonical))
        for application in applications:
            if self._normalise_name(application["name"]) in requested_aliases:
                return application["name"], application["target"]
        # Permit direct executable names and paths when the user specifies one.
        if requested.lower().endswith((".exe", ".lnk", ".url")):
            return requested, requested
        return None

    def open_application(self, application: str) -> bool:
        """Launch an installed application by its discovered name or alias."""
        if not isinstance(application, str) or not application.strip():
            logger.error("Cannot open application: invalid name")
            return False
        resolved = self._resolve_application(application.strip())
        if resolved is None:
            logger.warning("Application not found on this system: %s", application)
            return False
        name, target = resolved
        try:
            if target.startswith("apps-folder:"):
                import win32com.client

                app_id = target.removeprefix("apps-folder:")
                namespace = win32com.client.Dispatch("Shell.Application").NameSpace(
                    "shell:AppsFolder"
                )
                item = namespace.ParseName(app_id)
                if item is None:
                    logger.warning("AppsFolder no longer contains %s", name)
                    return False
                item.InvokeVerb("open")
                logger.info("Application launch initiated: %s", name)
                return True
            # Let Windows ShellExecute handle shortcuts, URL entries, and app
            # executables consistently (including file associations and shell
            # launch metadata stored in .lnk files).
            os.startfile(target)  # type: ignore[attr-defined]
            logger.info("Application launch initiated: %s", name)
            return True
        except (OSError, AttributeError):
            logger.exception("Failed to launch application %s from %s", name, target)
            return False
