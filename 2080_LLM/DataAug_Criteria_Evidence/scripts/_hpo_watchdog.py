import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

HPO_PID_FILE = Path('.hpo_all.pid')
GPU_MONITOR_PID_FILE = Path('.gpu_monitor.pid')
ALERT_LOG = Path('hpo_alerts.log')
MONITOR_LOG = Path('hpo_monitor.log')
STUDY_INFO_FILE = Path('.hpo_watchdog_state.json')

# Track the study-specific runner PIDs (criteria/evidence/share/joint)
EXPECTED_PROCS = {
    'criteria': 'scripts/tune_max.py --agent criteria',
    'evidence': 'scripts/tune_max.py --agent evidence',
    'share': 'scripts/tune_max.py --agent share',
    'joint': 'scripts/tune_max.py --agent joint',
}

CHECK_INTERVAL = 60


def log_alert(message: str) -> None:
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ALERT_LOG.open('a', encoding='utf-8') as fp:
        fp.write(f"[{timestamp}] {message}\n")


def read_pids() -> dict[str, int]:
    state = {}
    if STUDY_INFO_FILE.exists():
        try:
            state = json.loads(STUDY_INFO_FILE.read_text())
        except json.JSONDecodeError:
            state = {}
    return state


def write_pids(state: dict[str, int]) -> None:
    STUDY_INFO_FILE.write_text(json.dumps(state, indent=2))


def list_matching_pids(pattern: str) -> list[int]:
    pids = []
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info.get('cmdline') or [])
            if pattern in cmdline:
                pids.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids


def check_gpu_monitor() -> None:
    if GPU_MONITOR_PID_FILE.exists():
        try:
            pid = int(GPU_MONITOR_PID_FILE.read_text().strip())
        except ValueError:
            log_alert('GPU monitor PID file corrupted')
            return
        if not psutil.pid_exists(pid):
            log_alert('GPU monitor process not running')
    else:
        log_alert('GPU monitor PID file missing')


def check_hpo_make() -> None:
    if HPO_PID_FILE.exists():
        try:
            pid = int(HPO_PID_FILE.read_text().strip())
        except ValueError:
            log_alert('HPO PID file corrupted')
            return
        if not psutil.pid_exists(pid):
            log_alert('make maximal-hpo-all exited unexpectedly')
    else:
        log_alert('HPO PID file missing')


def check_trial_runners() -> None:
    state = read_pids()
    for agent, pattern in EXPECTED_PROCS.items():
        pids = list_matching_pids(pattern)
        if not pids:
            log_alert(f'HPO process missing for agent={agent}')
        else:
            alive = [pid for pid in pids if psutil.pid_exists(pid)]
            if not alive:
                log_alert(f'HPO process disappeared for agent={agent}')
            state[agent] = alive[0]
    write_pids(state)


def check_gpu_stalled(threshold_seconds: int = 180) -> None:
    if not MONITOR_LOG.exists():
        return
    with MONITOR_LOG.open('r', encoding='utf-8') as fp:
        lines = [line.strip() for line in fp.readlines() if line.strip()]
    if not lines:
        return
    last_line = lines[-1]
    try:
        timestamp_str = last_line.split(']')[0].strip('[')
        timestamp_struct = time.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        last_ts = time.mktime(timestamp_struct)
    except Exception:
        return
    now = time.time()
    if now - last_ts > threshold_seconds:
        log_alert('GPU monitor stale: no updates for >{}s'.format(threshold_seconds))


def main() -> None:
    # Run forever until interrupted
    try:
        while True:
            check_hpo_make()
            check_trial_runners()
            check_gpu_monitor()
            check_gpu_stalled()
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
