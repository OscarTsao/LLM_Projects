import subprocess
import time
import psutil
import datetime
from pathlib import Path

log_path = Path('hpo_monitor.log')

try:
    while True:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            gpu_util_raw = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                text=True,
            ).strip().split('\n')
            gpu_mem_raw = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                text=True,
            ).strip().split('\n')
            gpu_mem_pct = []
            for entry in gpu_mem_raw:
                used, total = entry.split(', ')
                gpu_mem_pct.append(f"{(float(used)/float(total))*100:.1f}")
            gpu_util = ','.join(gpu_util_raw)
            gpu_mem = ','.join(gpu_mem_pct)
        except Exception:
            gpu_util = 'NA'
            gpu_mem = 'NA'
        ram_pct = psutil.virtual_memory().percent
        cpu_pct = psutil.cpu_percent(interval=None)
        line = f"[{timestamp}] GPU: {gpu_util}% | GPU_MEM: {gpu_mem}% | RAM: {ram_pct:.1f}% | CPU: {cpu_pct:.1f}%\n"
        with log_path.open('a', encoding='utf-8') as fp:
            fp.write(line)
        time.sleep(15)
except KeyboardInterrupt:
    pass
