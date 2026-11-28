"""Read CPU, memory, and disk usage statistics from Windows."""

import json
import logging
import math
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)

POWERSHELL_SCRIPT = """
$os = Get-CimInstance -ClassName Win32_OperatingSystem
$cpu = Get-CimInstance -ClassName Win32_Processor |
    Measure-Object -Property LoadPercentage -Average
$disks = @(Get-CimInstance -ClassName Win32_LogicalDisk -Filter "DriveType=3" |
    ForEach-Object {
        [PSCustomObject]@{
            Drive = $_.DeviceID
            TotalBytes = [long]$_.Size
            FreeBytes = [long]$_.FreeSpace
        }
    })
[PSCustomObject]@{
    CpuPercent = $cpu.Average
    MemoryTotalBytes = [long]$os.TotalVisibleMemorySize * 1024
    MemoryAvailableBytes = [long]$os.FreePhysicalMemory * 1024
    Disks = $disks
} | ConvertTo-Json -Depth 4 -Compress
"""


def collect_windows_stats() -> dict[str, Any]:
    """Collect and validate current CPU, memory, and fixed-disk usage.

    Returns:
        Normalized system statistics with byte counts and percentages.

    Raises:
        OSError: If called on a non-Windows platform.
        RuntimeError: If PowerShell fails or returns invalid statistics.
    """
    if sys.platform != "win32":
        raise OSError("Windows system statistics are only available on Windows.")

    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                POWERSHELL_SCRIPT,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as error:
        logger.exception("Could not query Windows system statistics")
        raise RuntimeError("Unable to query Windows system statistics.") from error

    try:
        data = json.loads(result.stdout)
        if not isinstance(data, dict):
            raise TypeError("Expected a JSON object.")

        cpu_percent = float(data["CpuPercent"])
        memory_total = int(data["MemoryTotalBytes"])
        memory_available = int(data["MemoryAvailableBytes"])
        disks = data["Disks"]

        if (
            not math.isfinite(cpu_percent)
            or not 0 <= cpu_percent <= 100
            or memory_total <= 0
            or not 0 <= memory_available <= memory_total
            or not isinstance(disks, (list, dict))
        ):
            raise ValueError("Statistics are outside their valid ranges.")

        if isinstance(disks, dict):
            disks = [disks]

        normalized_disks = []
        for disk in disks:
            if not isinstance(disk, dict):
                raise TypeError("Expected each disk statistic to be an object.")
            drive = str(disk["Drive"])
            total_bytes = int(disk["TotalBytes"])
            free_bytes = int(disk["FreeBytes"])
            if not drive or total_bytes <= 0 or not 0 <= free_bytes <= total_bytes:
                raise ValueError("Disk statistics are outside their valid ranges.")
            normalized_disks.append(
                {
                    "drive": drive,
                    "total_bytes": total_bytes,
                    "free_bytes": free_bytes,
                }
            )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        logger.exception("PowerShell returned invalid Windows system statistics")
        raise RuntimeError("Windows system statistics were invalid.") from error

    return {
        "cpu_percent": cpu_percent,
        "memory_total_bytes": memory_total,
        "memory_available_bytes": memory_available,
        "disks": normalized_disks,
    }


def format_windows_stats(stats: dict[str, Any]) -> str:
    """Format normalized system statistics for display in WinPilot.

    Args:
        stats: Normalized statistics returned by collect_windows_stats.

    Returns:
        Human-readable CPU, memory, and disk usage.
    """
    gib = 1024**3
    memory_total = stats["memory_total_bytes"]
    memory_used = memory_total - stats["memory_available_bytes"]
    memory_used_percent = memory_used / memory_total * 100
    lines = [
        f"CPU usage: {stats['cpu_percent']:.1f}%",
        (
            f"Memory usage: {memory_used / gib:.1f} / {memory_total / gib:.1f} GiB "
            f"({memory_used_percent:.1f}% used)"
        ),
    ]

    if stats["disks"]:
        for disk in stats["disks"]:
            total_bytes = disk["total_bytes"]
            used_bytes = total_bytes - disk["free_bytes"]
            used_percent = used_bytes / total_bytes * 100
            lines.append(
                f"Disk {disk['drive']} usage: {used_bytes / gib:.1f} / "
                f"{total_bytes / gib:.1f} GiB ({used_percent:.1f}% used)"
            )
    else:
        lines.append("Disk usage: no fixed drives found")

    return "\n".join(lines)
