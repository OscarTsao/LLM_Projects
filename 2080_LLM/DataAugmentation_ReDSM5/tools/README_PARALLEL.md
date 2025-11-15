# Parallel Augmentation Quick Start

## Overview

Three simple scripts for parallel augmentation generation:

1. **`run_parallel_augment.sh`** - Launch 7 parallel shards
2. **`monitor_augment.py`** - Monitor progress in real-time
3. **`merge_shard_manifests.sh`** - Combine results

## Quick Start (3 Commands)

```bash
# 1. Launch parallel augmentation (7 shards)
./tools/run_parallel_augment.sh

# 2. Monitor progress (in another terminal)
python tools/monitor_augment.py --follow

# 3. Merge results when complete
./tools/merge_shard_manifests.sh
```

## Detailed Usage

### Option 1: Run with Monitoring

```bash
# Terminal 1: Launch
./tools/run_parallel_augment.sh

# Terminal 2: Monitor
python tools/monitor_augment.py --follow
```

### Option 2: Run and Wait

```bash
# Just run (check logs manually)
./tools/run_parallel_augment.sh

# View logs
tail -f logs/augment/shard_0_of_7.log

# After completion
./tools/merge_shard_manifests.sh
```

### Option 3: Test First

```bash
# Test with dry-run (no execution)
./tools/run_parallel_augment.sh --dry-run

# Test with 2 shards (faster)
./tools/test_parallel_infrastructure.sh
```

## What Gets Created

```
data/
├── cache/
│   ├── augment_shard0.db        # Disk cache per shard
│   ├── augment_shard1.db
│   └── ... (7 total)
└── processed/
    └── augsets/
        ├── manifest_final.csv    # Final merged manifest
        ├── combo_1_*/            # One directory per combo
        │   ├── dataset.parquet   # Augmented dataset
        │   └── meta.json         # Metadata
        └── ...

logs/
└── augment/
    ├── shard_0_of_7.log         # Execution logs
    ├── shard_1_of_7.log
    └── ... (7 total)
```

## Common Options

### Launcher (`run_parallel_augment.sh`)

```bash
# Dry run (test without executing)
./tools/run_parallel_augment.sh --dry-run

# Use different shard count
./tools/run_parallel_augment.sh --num-shards 4

# Show help
./tools/run_parallel_augment.sh --help
```

### Monitor (`monitor_augment.py`)

```bash
# Single snapshot
python tools/monitor_augment.py

# Continuous monitoring
python tools/monitor_augment.py --follow

# Custom update interval (default: 10s)
python tools/monitor_augment.py --follow --interval 5

# Custom log directory
python tools/monitor_augment.py --log-dir /path/to/logs --num-shards 4
```

### Merger (`merge_shard_manifests.sh`)

```bash
# Standard merge
./tools/merge_shard_manifests.sh

# Keep shard manifests
./tools/merge_shard_manifests.sh --keep-shards

# Custom output
./tools/merge_shard_manifests.sh --output-name my_manifest.csv
```

## Expected Timeline

For the full ReDSM5 dataset with all methods:

- **Startup**: 1-2 minutes (loading models)
- **Per combo**: 2-5 minutes (depends on method)
- **Total**: ~2-4 hours for singletons mode with 7 shards

GPU methods (RoBERTa, BERT, back-translation) take longer.

## Monitoring Example

```
================================================================
Augmentation Progress Monitor
================================================================
Started: 2025-10-24 17:30:00
Elapsed: 0:15:32

▶  Shard  0 |  45.2% | Combos:  12/27 | Current: nlp_wordnet_syn     | ETA: 0:18:45     | Rows:   1,234
✓ Shard  1 | 100.0% | Combos:  27/27 | Current: -                   | ETA: Complete    | Rows:   3,456
▶  Shard  2 |  23.1% | Combos:   6/27 | Current: ta_wordswap_qwerty  | ETA: 0:45:12     | Rows:     567

Overall: 1 completed | 5 running | 0 failed | 1 pending
Total: 78 combos processed | 12,345 rows generated
================================================================
```

## Troubleshooting

### Check Shard Status

```bash
# Are processes running?
ps aux | grep generate_augsets.py

# View specific shard log
cat logs/augment/shard_0_of_7.log

# Check for errors
grep -i error logs/augment/*.log
```

### Restart Failed Shard

```bash
# Kill all shards
pkill -f generate_augsets.py

# Rerun (will skip completed combos without --force)
./tools/run_parallel_augment.sh
```

### Clean Start

```bash
# Remove all output
rm -rf data/processed/augsets data/cache logs/augment

# Rerun from scratch
./tools/run_parallel_augment.sh
```

## Integration

After generating augmented datasets:

```bash
# View final manifest
cat data/processed/augsets/manifest_final.csv

# List all combos
ls -d data/processed/augsets/combo_1_*/

# Use in verification
python tools/verify/verify_augmentation.py

# Use in training
python tools/train_model.py --augmented-dataset data/processed/augsets/combo_1_METHOD/dataset.parquet
```

## Key Features

1. **Parallel Execution**: 7 independent processes
2. **Fault Tolerant**: Shards run independently
3. **Resumable**: Skip completed combos (without `--force`)
4. **Deterministic**: Fixed seed ensures reproducibility
5. **Progress Tracking**: Real-time monitoring
6. **Clean Logs**: Separate log per shard
7. **Graceful Shutdown**: Ctrl+C cleans up all processes

## Configuration

Default settings in `run_parallel_augment.sh`:

```bash
NUM_SHARDS=7                    # Parallel processes
VARIANTS_PER_SAMPLE=2           # Augmented samples per original
QUALITY_MIN_SIM=0.55           # Minimum similarity
QUALITY_MAX_SIM=0.95           # Maximum similarity
SEED=42                         # Random seed
```

Edit the script to customize.

## Notes

- Each shard processes a subset of method combos
- Shards don't communicate (fully independent)
- Output is deterministic given same seed
- GPU methods run sequentially within each shard
- Disk caches persist across runs for speed

## Full Documentation

See `tools/PARALLEL_AUGMENTATION.md` for complete details.
