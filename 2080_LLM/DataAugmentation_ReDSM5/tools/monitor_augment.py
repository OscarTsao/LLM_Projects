#!/usr/bin/env python3
"""
monitor_augment.py
Simple progress monitor for parallel augmentation generation.

Parses log files to show real-time progress of all shards.

Usage:
    python tools/monitor_augment.py [OPTIONS]

Options:
    --log-dir PATH     Directory containing shard logs (default: logs/augment)
    --num-shards N     Number of shards to monitor (default: 7)
    --interval SECS    Update interval in seconds (default: 10)
    --follow           Keep monitoring until all shards complete
"""

import argparse
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple


class ShardStatus:
    """Track status of a single shard."""

    def __init__(self, shard_id: int):
        self.shard_id = shard_id
        self.status = "pending"  # pending, running, completed, failed
        self.current_combo: Optional[str] = None
        self.combos_processed = 0
        self.total_combos: Optional[int] = None
        self.rows_generated = 0
        self.last_update: Optional[datetime] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message: Optional[str] = None

    @property
    def progress_pct(self) -> Optional[float]:
        """Calculate progress percentage."""
        if self.total_combos and self.total_combos > 0:
            return (self.combos_processed / self.total_combos) * 100
        return None

    @property
    def eta(self) -> Optional[str]:
        """Estimate time to completion."""
        if not self.start_time or not self.progress_pct or self.progress_pct <= 0:
            return None

        elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.progress_pct >= 100:
            return "Complete"

        total_time = elapsed / (self.progress_pct / 100)
        remaining = total_time - elapsed
        return str(timedelta(seconds=int(remaining)))


class LogParser:
    """Parse augmentation log files."""

    # Log patterns to match
    PATTERNS = {
        "prepared_combos": re.compile(r"Prepared (\d+) combos for shard (\d+)/(\d+)"),
        "processing_combo": re.compile(r"Processing combo ([\w-]+) \| methods=([^\|]+)"),
        "combo_generated": re.compile(r"Combo ([\w-]+) generated (\d+) rows"),
        "skipping_combo": re.compile(r"Skipping combo ([^\s]+)"),
        "completed": re.compile(
            r"Completed in .* \| combos attempted=(\d+) \| produced=(\d+) .* rows=(\d+)"
        ),
        "error": re.compile(r"\[ERROR\](.*)"),
        "exception": re.compile(r"Traceback|Exception|Error:"),
    }

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.last_position = 0

    def parse_new_lines(self, status: ShardStatus) -> None:
        """Parse new lines since last check."""
        if not self.log_path.exists():
            return

        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                # Seek to last position
                f.seek(self.last_position)
                new_lines = f.readlines()
                self.last_position = f.tell()

                for line in new_lines:
                    self._process_line(line, status)

        except Exception as e:
            status.error_message = f"Log parse error: {e}"

    def _process_line(self, line: str, status: ShardStatus) -> None:
        """Process a single log line."""
        # Check for prepared combos (startup)
        match = self.PATTERNS["prepared_combos"].search(line)
        if match:
            status.total_combos = int(match.group(1))
            status.status = "running"
            if not status.start_time:
                status.start_time = datetime.now()
            return

        # Check for processing combo
        match = self.PATTERNS["processing_combo"].search(line)
        if match:
            status.current_combo = match.group(1)
            status.status = "running"
            status.last_update = datetime.now()
            return

        # Check for combo completion
        match = self.PATTERNS["combo_generated"].search(line)
        if match:
            status.combos_processed += 1
            status.rows_generated += int(match.group(2))
            status.last_update = datetime.now()
            return

        # Check for skipped combo
        match = self.PATTERNS["skipping_combo"].search(line)
        if match:
            status.combos_processed += 1
            status.last_update = datetime.now()
            return

        # Check for completion
        match = self.PATTERNS["completed"].search(line)
        if match:
            status.status = "completed"
            status.combos_processed = int(match.group(1))
            status.rows_generated = int(match.group(3))
            status.end_time = datetime.now()
            return

        # Check for errors
        if self.PATTERNS["error"].search(line) or self.PATTERNS["exception"].search(line):
            status.status = "failed"
            status.error_message = line.strip()[:100]
            return


def format_status_line(status: ShardStatus) -> str:
    """Format a single shard status line."""
    # Status indicator
    if status.status == "pending":
        indicator = "⏳"
    elif status.status == "running":
        indicator = "▶ "
    elif status.status == "completed":
        indicator = "✓"
    elif status.status == "failed":
        indicator = "✗"
    else:
        indicator = "?"

    # Progress info
    progress_str = ""
    if status.progress_pct is not None:
        progress_str = f"{status.progress_pct:5.1f}%"
    else:
        progress_str = "  N/A"

    # Combo info
    combo_str = f"{status.combos_processed:3d}"
    if status.total_combos:
        combo_str += f"/{status.total_combos}"
    else:
        combo_str += "/?  "

    # Current combo
    current = status.current_combo or "-"
    if len(current) > 20:
        current = current[:17] + "..."

    # ETA
    eta = status.eta or "N/A"

    # Rows
    rows_str = f"{status.rows_generated:7,d}"

    # Build line
    line = f"{indicator} Shard {status.shard_id:2d} | "
    line += f"{progress_str} | "
    line += f"Combos: {combo_str} | "
    line += f"Current: {current:20s} | "
    line += f"ETA: {eta:12s} | "
    line += f"Rows: {rows_str}"

    return line


def monitor_shards(
    log_dir: Path, num_shards: int, interval: int, follow: bool
) -> Tuple[int, int]:
    """
    Monitor shard progress.

    Returns:
        Tuple of (completed_count, failed_count)
    """
    # Initialize status trackers
    statuses = {i: ShardStatus(i) for i in range(num_shards)}
    parsers = {
        i: LogParser(log_dir / f"shard_{i}_of_{num_shards}.log")
        for i in range(num_shards)
    }

    start_time = datetime.now()

    while True:
        # Parse new log lines
        for shard_id in range(num_shards):
            parsers[shard_id].parse_new_lines(statuses[shard_id])

        # Clear screen (simple approach)
        print("\033[2J\033[H", end="")

        # Header
        print("=" * 120)
        print("Augmentation Progress Monitor")
        print("=" * 120)
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed: {datetime.now() - start_time}")
        print("")

        # Shard statuses
        for shard_id in range(num_shards):
            print(format_status_line(statuses[shard_id]))

        # Overall summary
        print("")
        print("-" * 120)

        completed = sum(1 for s in statuses.values() if s.status == "completed")
        failed = sum(1 for s in statuses.values() if s.status == "failed")
        running = sum(1 for s in statuses.values() if s.status == "running")
        pending = sum(1 for s in statuses.values() if s.status == "pending")

        total_rows = sum(s.rows_generated for s in statuses.values())
        total_combos_processed = sum(s.combos_processed for s in statuses.values())

        print(f"Overall: {completed} completed | {running} running | {failed} failed | {pending} pending")
        print(f"Total: {total_combos_processed} combos processed | {total_rows:,} rows generated")

        # Show errors if any
        errors = [s for s in statuses.values() if s.error_message]
        if errors:
            print("")
            print("Errors:")
            for s in errors:
                print(f"  Shard {s.shard_id}: {s.error_message}")

        print("=" * 120)
        print(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if not follow:
            print("")
            print("Run with --follow to continuously monitor")
            break

        # Check if all complete
        if completed + failed == num_shards:
            print("")
            print("All shards finished!")
            return completed, failed

        # Wait for next update
        time.sleep(interval)

    return completed, failed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/augment"),
        help="Directory containing shard logs",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=7,
        help="Number of shards to monitor",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Update interval in seconds",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Keep monitoring until all shards complete",
    )

    args = parser.parse_args()

    # Validate log directory
    if not args.log_dir.exists():
        print(f"Error: Log directory not found: {args.log_dir}", file=sys.stderr)
        print("Have you started the parallel augmentation?", file=sys.stderr)
        sys.exit(1)

    try:
        completed, failed = monitor_shards(
            args.log_dir, args.num_shards, args.interval, args.follow
        )

        if args.follow:
            if failed > 0:
                sys.exit(1)
            else:
                sys.exit(0)

    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
