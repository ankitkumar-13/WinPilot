import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | "
    "%(name)s | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    name: str = "winpilot",
    level: int = logging.INFO,
) -> None:
    """Configure rotating file logging for a WinPilot component.

    Args:
        name - Component name used to select the log file.
        level - Minimum logging level to record.

    Returns:
        None.
    """
    if not isinstance(name, str) or not name.strip():
        name = "winpilot"

    if not isinstance(level, int):
        level = logging.INFO

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = name.strip().lower()
    log_file = LOG_DIR / f"{safe_name}.log"

    formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=level,
        handlers=[file_handler],
        force=True,
    )

    logging.captureWarnings(True)
