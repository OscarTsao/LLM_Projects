"""Lightweight MLflow training monitor.

Polls an MLflow SQLite DB for a RUNNING run, prints the latest epoch metrics
and estimates ETA to a target number of epochs. Optionally follows updates.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class RunInfo:
    run_uuid: str
    status: str
    start_time_ms: Optional[int]
    end_time_ms: Optional[int]


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def find_running_run(conn: sqlite3.Connection) -> Optional[RunInfo]:
    cur = conn.execute(
        """
        SELECT run_uuid, status, start_time, end_time
        FROM runs
        WHERE status = 'RUNNING'
        ORDER BY start_time DESC
        LIMIT 1
        """
    )
    row = cur.fetchone()
    if not row:
        return None
    return RunInfo(
        run_uuid=row["run_uuid"],
        status=row["status"],
        start_time_ms=row["start_time"],
        end_time_ms=row["end_time"],
    )


def get_run_info(conn: sqlite3.Connection, run_uuid: str) -> Optional[RunInfo]:
    cur = conn.execute(
        """
        SELECT run_uuid, status, start_time, end_time
        FROM runs
        WHERE run_uuid = ?
        LIMIT 1
        """,
        (run_uuid,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return RunInfo(
        run_uuid=row["run_uuid"],
        status=row["status"],
        start_time_ms=row["start_time"],
        end_time_ms=row["end_time"],
    )


def latest_step(conn: sqlite3.Connection, run_uuid: str, key_preference: Sequence[str]) -> Optional[int]:
    for key in key_preference:
        cur = conn.execute(
            """
            SELECT MAX(step) AS s
            FROM metrics
            WHERE run_uuid = ? AND key = ?
            """,
            (run_uuid, key),
        )
        row = cur.fetchone()
        if row and row["s"] is not None:
            return int(row["s"])
    # Fallback to any metric
    cur = conn.execute(
        """
        SELECT MAX(step) AS s
        FROM metrics
        WHERE run_uuid = ?
        """,
        (run_uuid,),
    )
    row = cur.fetchone()
    return int(row["s"]) if row and row["s"] is not None else None


def metrics_for_step(conn: sqlite3.Connection, run_uuid: str, step: int) -> Dict[str, float]:
    cur = conn.execute(
        """
        SELECT key, value
        FROM metrics
        WHERE run_uuid = ? AND step = ?
        ORDER BY key
        """,
        (run_uuid, step),
    )
    return {str(r["key"]): float(r["value"]) for r in cur.fetchall()}


def step_timestamps_ms(conn: sqlite3.Connection, run_uuid: str, metric_key: str) -> List[Tuple[int, int]]:
    cur = conn.execute(
        """
        SELECT step, timestamp
        FROM metrics
        WHERE run_uuid = ? AND key = ?
        ORDER BY step
        """,
        (run_uuid, metric_key),
    )
    return [(int(r["step"]), int(r["timestamp"])) for r in cur.fetchall()]


def human_td(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    td = timedelta(seconds=int(seconds))
    # Format as H:MM:SS
    total_seconds = int(td.total_seconds())
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def estimate_epoch_time_seconds(ts: List[Tuple[int, int]]) -> Optional[float]:
    if len(ts) < 2:
        return None
    diffs = [
        (ts[i][1] - ts[i - 1][1]) / 1000.0
        for i in range(1, len(ts))
        if ts[i][1] is not None and ts[i - 1][1] is not None
    ]
    return sum(diffs) / len(diffs) if diffs else None


def now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


def print_snapshot(
    conn: sqlite3.Connection,
    run: RunInfo,
    num_epochs: Optional[int],
    key_preference: Sequence[str],
) -> None:
    step = latest_step(conn, run.run_uuid, key_preference)
    if step is None:
        print(f"[{now_str()}] No metrics logged yet for run {run.run_uuid} (status={run.status}).")
        return

    m = metrics_for_step(conn, run.run_uuid, step)
    # Prefer to estimate from validation metric timestamps
    ts = step_timestamps_ms(conn, run.run_uuid, key_preference[0])
    avg_epoch_sec = estimate_epoch_time_seconds(ts)

    # Compute ETA
    eta_str = "?"
    epoch_time_str = human_td(avg_epoch_sec) if avg_epoch_sec else "?"
    if num_epochs and avg_epoch_sec:
        remaining = max(0, num_epochs - step)
        eta_seconds = remaining * avg_epoch_sec
        eta_str = human_td(eta_seconds)

    parts = [
        f"[{now_str()}] step {step}{f'/{num_epochs}' if num_epochs else ''}",
    ]
    # Show a concise set of metrics if present
    for k in (
        "train/loss",
        "val/accuracy",
        "val/precision",
        "val/recall",
        "val/f1",
        "train/lr",
    ):
        if k in m:
            val = m[k]
            if k.endswith("lr"):
                parts.append(f"{k}={val:.2e}")
            else:
                parts.append(f"{k}={val:.6f}" if val < 0.1 else f"{k}={val:.4f}")

    parts.append(f"epoch_time~{epoch_time_str}")
    parts.append(f"ETA~{eta_str}")
    print(" | ".join(parts))


def run_monitor(
    db_path: str,
    run_uuid: Optional[str],
    poll_interval: float,
    num_epochs: Optional[int],
    follow: bool,
) -> int:
    key_preference = ("val/f1", "train/loss")
    conn = _connect(db_path)

    run = None
    if run_uuid:
        run = get_run_info(conn, run_uuid)
        if not run:
            print(f"No run found with id {run_uuid} in {db_path}.")
            return 2
    else:
        run = find_running_run(conn)
        if not run:
            print("No RUNNING MLflow run found.")
            return 1

    # initial snapshot
    print_snapshot(conn, run, num_epochs, key_preference)
    if not follow:
        return 0

    # Follow updates
    last_seen_step = latest_step(conn, run.run_uuid, key_preference) or -1
    while True:
        time.sleep(poll_interval)
        # Refresh run status
        run = get_run_info(conn, run.run_uuid)
        if not run:
            print("Run disappeared from DB; stopping monitor.")
            return 3
        if run.status in {"FINISHED", "FAILED", "KILLED"}:
            print(f"[{now_str()}] Run status is {run.status}. Stopping monitor.")
            print_snapshot(conn, run, num_epochs, key_preference)
            return 0

        step = latest_step(conn, run.run_uuid, key_preference)
        if step is None:
            continue
        if step > last_seen_step:
            print_snapshot(conn, run, num_epochs, key_preference)
            last_seen_step = step


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor MLflow training metrics from SQLite DB")
    p.add_argument("--mlflow-db", default="mlflow.db", help="Path to MLflow SQLite DB (default: mlflow.db)")
    p.add_argument("--run-id", help="Specific run_uuid to monitor (defaults to the latest RUNNING run)")
    p.add_argument("--num-epochs", type=int, default=None, help="Total epochs to estimate ETA (optional)")
    p.add_argument("--poll-interval", type=float, default=30.0, help="Seconds between polls when following (default: 30)")
    p.add_argument("--follow", action="store_true", help="Keep running and print updates when a new epoch logs")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    return run_monitor(
        db_path=args.mlflow_db,
        run_uuid=args.run_id,
        poll_interval=args.poll_interval,
        num_epochs=args.num_epochs,
        follow=args.follow,
    )


if __name__ == "__main__":
    raise SystemExit(main())

