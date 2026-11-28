"""Tests for Windows system statistics collection and formatting."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from plugins import windows_stats


class CollectWindowsStatsTests(unittest.TestCase):
    """Verify Windows statistics collection and validation."""

    def test_collects_and_normalizes_powershell_output(self) -> None:
        """Parse system metrics and normalize a single disk result."""
        powershell_output = {
            "CpuPercent": 18.5,
            "MemoryTotalBytes": 16 * 1024**3,
            "MemoryAvailableBytes": 4 * 1024**3,
            "Disks": {
                "Drive": "C:",
                "TotalBytes": 500 * 1024**3,
                "FreeBytes": 100 * 1024**3,
            },
        }
        completed = SimpleNamespace(stdout=json.dumps(powershell_output))

        with (
            patch.object(windows_stats.sys, "platform", "win32"),
            patch.object(windows_stats.subprocess, "run", return_value=completed) as run,
        ):
            stats = windows_stats.collect_windows_stats()

        self.assertEqual(stats["cpu_percent"], 18.5)
        self.assertEqual(stats["memory_available_bytes"], 4 * 1024**3)
        self.assertEqual(
            stats["disks"],
            [
                {
                    "drive": "C:",
                    "total_bytes": 500 * 1024**3,
                    "free_bytes": 100 * 1024**3,
                }
            ],
        )
        self.assertEqual(run.call_args.kwargs["timeout"], 15)
        self.assertTrue(run.call_args.kwargs["check"])

    def test_rejects_non_windows_platform(self) -> None:
        """Fail clearly instead of running PowerShell on another platform."""
        with (
            patch.object(windows_stats.sys, "platform", "linux"),
            patch.object(windows_stats.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(OSError, "only available on Windows"):
                windows_stats.collect_windows_stats()

        run.assert_not_called()

    def test_rejects_invalid_powershell_output(self) -> None:
        """Report malformed PowerShell JSON instead of returning empty metrics."""
        completed = SimpleNamespace(stdout="not json")

        with (
            patch.object(windows_stats.sys, "platform", "win32"),
            patch.object(windows_stats.subprocess, "run", return_value=completed),
        ):
            with self.assertLogs(windows_stats.logger, level="ERROR"):
                with self.assertRaisesRegex(RuntimeError, "statistics were invalid"):
                    windows_stats.collect_windows_stats()


class FormatWindowsStatsTests(unittest.TestCase):
    """Verify human-readable system statistics."""

    def test_formats_cpu_memory_and_disk_usage(self) -> None:
        """Show usage and percentages for each collected resource."""
        stats = {
            "cpu_percent": 18.5,
            "memory_total_bytes": 16 * 1024**3,
            "memory_available_bytes": 4 * 1024**3,
            "disks": [
                {
                    "drive": "C:",
                    "total_bytes": 500 * 1024**3,
                    "free_bytes": 100 * 1024**3,
                }
            ],
        }

        result = windows_stats.format_windows_stats(stats)

        self.assertEqual(
            result,
            "CPU usage: 18.5%\n"
            "Memory usage: 12.0 / 16.0 GiB (75.0% used)\n"
            "Disk C: usage: 400.0 / 500.0 GiB (80.0% used)",
        )

    def test_reports_when_no_fixed_disks_are_found(self) -> None:
        """Make the empty fixed-disk result explicit."""
        stats = {
            "cpu_percent": 0.0,
            "memory_total_bytes": 1024**3,
            "memory_available_bytes": 1024**3,
            "disks": [],
        }

        self.assertIn(
            "Disk usage: no fixed drives found",
            windows_stats.format_windows_stats(stats),
        )


if __name__ == "__main__":
    unittest.main()
