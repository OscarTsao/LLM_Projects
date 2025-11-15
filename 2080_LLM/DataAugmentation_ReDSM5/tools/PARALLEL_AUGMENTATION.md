# Parallel Augmentation Infrastructure

This directory contains scripts for parallel execution of augmentation generation across multiple shards.

## Overview

The parallel execution infrastructure splits augmentation combo generation across multiple independent processes (shards), enabling:

- **Faster processing**: Distribute work across 7 parallel processes
- **Better resource utilization**: Keep all CPU cores busy
- **Fault tolerance**: Individual shard failures don't affect others
- **Progress monitoring**: Real-time visibility into generation status
- **Simple merge**: Combine results into a single manifest

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│         run_parallel_augment.sh (Launcher)              │
│  Spawns 7 independent generate_augsets.py processes     │
└───┬─────────┬─────────┬─────────┬─────────┬─────────┬──┘
    │         │         │         │         │         │
    ▼         ▼         ▼         ▼         ▼         ▼
 Shard 0   Shard 1   Shard 2   Shard 3   Shard 4   Shard 5   Shard 6
    │         │         │         │         │         │         │
    ▼         ▼         ▼         ▼         ▼         ▼         ▼
 combo_1_*  combo_1_*  combo_1_*  combo_1_*  combo_1_*  combo_1_*  combo_1_*
 dataset    dataset    dataset    dataset    dataset    dataset    dataset
    │         │         │         │         │         │         │
    ▼         ▼         ▼         ▼         ▼         ▼         ▼
manifest  manifest  manifest  manifest  manifest  manifest  manifest
shard0.csv shard1.csv shard2.csv shard3.csv shard4.csv shard5.csv shard6.csv
    │         │         │         │         │         │         │
    └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
                              ▼
              merge_shard_manifests.sh (Merger)
                              ▼
                     manifest_final.csv
```

## Scripts

### 1. `run_parallel_augment.sh` - Parallel Launcher

Launches multiple instances of `generate_augsets.py` in parallel, each processing a different subset of combos.

**Usage:**
```bash
# Launch 7 shards (default)
./tools/run_parallel_augment.sh

# Test with dry-run mode
./tools/run_parallel_augment.sh --dry-run

# Use different number of shards
./tools/run_parallel_augment.sh --num-shards 4
```

**Features:**
- Launches N shards in background with proper PID tracking
- Separate log file per shard: `logs/augment/shard_{N}_of_7.log`
- Separate disk cache per shard: `data/cache/augment_shard{N}.db`
- Graceful cleanup on SIGINT (Ctrl+C)
- Exit code tracking and summary report
- Waits for all shards to complete

**Configuration:**
All settings are embedded in the script but can be customized by editing:
- `NUM_SHARDS`: Number of parallel processes (default: 7)
- `VARIANTS_PER_SAMPLE`: Augmented variants per sample (default: 2)
- `QUALITY_MIN_SIM`: Minimum similarity threshold (default: 0.55)
- `QUALITY_MAX_SIM`: Maximum similarity threshold (default: 0.95)
- `SEED`: Random seed for reproducibility (default: 42)

### 2. `monitor_augment.py` - Progress Monitor

Real-time monitoring of shard progress by parsing log files.

**Usage:**
```bash
# Single snapshot of current progress
python tools/monitor_augment.py

# Continuous monitoring until completion
python tools/monitor_augment.py --follow

# Custom log directory and update interval
python tools/monitor_augment.py --log-dir /path/to/logs --interval 5 --follow
```

**Display:**
```
================================================================
Augmentation Progress Monitor
================================================================
Started: 2025-10-24 17:30:00
Elapsed: 0:15:32

▶  Shard  0 |  45.2% | Combos:  12/27 | Current: nlp_wordnet_syn     | ETA: 0:18:45     | Rows:   1,234
✓ Shard  1 | 100.0% | Combos:  27/27 | Current: -                   | ETA: Complete    | Rows:   3,456
▶  Shard  2 |  23.1% | Combos:   6/27 | Current: ta_wordswap_qwerty  | ETA: 0:45:12     | Rows:     567
...

Overall: 1 completed | 5 running | 0 failed | 1 pending
Total: 78 combos processed | 12,345 rows generated
================================================================
```

**Legend:**
- `⏳` Pending (not started)
- `▶ ` Running (active)
- `✓` Completed (success)
- `✗` Failed (error)

### 3. `merge_shard_manifests.sh` - Manifest Merger

Combines individual shard manifests into a single final manifest.

**Usage:**
```bash
# Merge all shard manifests
./tools/merge_shard_manifests.sh

# Keep individual shard manifests
./tools/merge_shard_manifests.sh --keep-shards

# Custom output location
./tools/merge_shard_manifests.sh --output-root /path/to/augsets --output-name manifest.csv
```

**Output:**
- Creates `manifest_final.csv` with all combos
- Removes individual shard manifests (unless `--keep-shards`)
- Prints summary statistics:
  - Total combos
  - Total augmented rows
  - Combos by size (k=1, k=2, etc.)
  - Min/max/avg rows per combo

## Complete Workflow

### Step 1: Launch Parallel Augmentation

```bash
# Start all 7 shards
./tools/run_parallel_augment.sh
```

This will:
1. Validate input files and directories
2. Launch 7 parallel processes
3. Create log files in `logs/augment/`
4. Wait for all shards to complete
5. Report success/failure summary

### Step 2: Monitor Progress (Optional)

In a separate terminal:

```bash
# Watch progress in real-time
python tools/monitor_augment.py --follow
```

Or check logs directly:

```bash
# View specific shard log
tail -f logs/augment/shard_0_of_7.log

# Check all shard logs
tail -f logs/augment/shard_*.log
```

### Step 3: Merge Results

After all shards complete:

```bash
# Merge shard manifests
./tools/merge_shard_manifests.sh
```

This creates `data/processed/augsets/manifest_final.csv` with all combo metadata.

### Step 4: Verify Output

```bash
# Check final manifest
head data/processed/augsets/manifest_final.csv

# Count datasets
ls -la data/processed/augsets/combo_1_*/dataset.parquet | wc -l

# Check specific combo
ls -R data/processed/augsets/combo_1_nlp_wordnet_syn/
```

## Directory Structure

After execution:

```
data/
├── cache/
│   ├── augment_shard0.db
│   ├── augment_shard1.db
│   └── ... (7 total)
└── processed/
    └── augsets/
        ├── manifest_final.csv
        ├── combo_1_nlp_wordnet_syn/
        │   ├── dataset.parquet
        │   └── meta.json
        ├── combo_1_nlp_spelling/
        │   ├── dataset.parquet
        │   └── meta.json
        └── ... (one directory per combo)

logs/
└── augment/
    ├── shard_0_of_7.log
    ├── shard_1_of_7.log
    └── ... (7 total)
```

## Performance Considerations

### Shard Count

The default of 7 shards is optimized for:
- Balanced load distribution across methods
- Good parallelism without excessive overhead
- Manageable number of log files

Adjust based on:
- Available CPU cores: `--num-shards $(nproc)`
- Memory constraints: Fewer shards = less memory
- I/O throughput: Too many shards can saturate disk

### Resource Usage

Each shard:
- **CPU**: 1-2 cores (depends on `--num-proc` setting)
- **Memory**: ~2-4 GB (varies by method, especially GPU methods)
- **Disk**:
  - Cache: ~100-500 MB per shard
  - Output: Depends on dataset size
  - Logs: ~10-50 MB per shard

### GPU Methods

Note: GPU methods (like `nlp_cwe_sub_roberta`, `ta_mlm_sub_bert`) will:
- Run sequentially within each shard (`--num-proc 1`)
- May compete for GPU memory across shards
- Consider reducing shard count if GPU memory is limited

## Troubleshooting

### Shard Fails to Start

Check log file for errors:
```bash
cat logs/augment/shard_N_of_7.log
```

Common issues:
- Missing dependencies: Install with `pip install -r requirements.txt`
- Invalid input CSV: Verify file exists and has correct columns
- GPU unavailable: GPU methods will fail if CUDA not available

### Shard Hangs or Stalls

1. Check if process is alive:
```bash
ps aux | grep generate_augsets.py
```

2. Monitor system resources:
```bash
htop
nvidia-smi  # for GPU monitoring
```

3. Check log for last activity:
```bash
tail -50 logs/augment/shard_N_of_7.log
```

### Out of Memory

Reduce parallel processes:
```bash
# Edit run_parallel_augment.sh
NUM_PROC=1  # Already default for safety
```

Or reduce shard count:
```bash
./tools/run_parallel_augment.sh --num-shards 4
```

### Disk Space Issues

Monitor disk usage:
```bash
du -sh data/cache data/processed/augsets logs/augment
```

Clean up caches:
```bash
rm -rf data/cache/augment_shard*.db
```

## Customization

### Modify Augmentation Parameters

Edit `run_parallel_augment.sh`:

```bash
# Change quality thresholds
QUALITY_MIN_SIM=0.60  # More conservative
QUALITY_MAX_SIM=0.90  # More diverse

# Change variants per sample
VARIANTS_PER_SAMPLE=5  # More variants

# Change combo mode
# Edit the command template:
--combo-mode bounded_k \
--max-combo-size 2 \
```

### Add New Methods

1. Update `conf/augment_methods.yaml` with new method
2. Rerun: `./tools/run_parallel_augment.sh`
3. New method will be included automatically

### Custom Input Data

Edit `run_parallel_augment.sh`:

```bash
INPUT_CSV="${PROJECT_ROOT}/path/to/custom.csv"

# Update column names if needed:
--text-col my_text_column \
--evidence-col my_evidence_column \
```

## Integration with Training

After generating augmented datasets:

```bash
# List all available combos
python -c "
import pandas as pd
manifest = pd.read_csv('data/processed/augsets/manifest_final.csv')
print(manifest[['combo_id', 'methods', 'rows']])
"

# Use specific combo in training
python tools/train_model.py \
  --augmented-dataset data/processed/augsets/combo_1_nlp_wordnet_syn/dataset.parquet

# Or use verification suite
python tools/verify/verify_augmentation.py
```

## Best Practices

1. **Always use dry-run first**: Test configuration with `--dry-run`
2. **Monitor early**: Start monitoring before expecting first shard to complete
3. **Keep logs**: Don't delete logs until you've verified output
4. **Backup manifests**: Keep `--keep-shards` for debugging
5. **Clean caches periodically**: Disk caches can grow large
6. **Use consistent seeds**: Ensure reproducibility with fixed `--seed`

## Notes

- **Deterministic**: Each shard produces identical results given the same seed
- **Independent**: Shards don't communicate or share state
- **Resumable**: Re-run with `--force` removed to skip existing combos
- **Scalable**: Add more shards for larger method sets

## Support

For issues or questions:
1. Check logs in `logs/augment/`
2. Verify input files exist and are readable
3. Ensure all dependencies are installed
4. Review error messages in script output
