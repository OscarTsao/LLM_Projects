# Parallel Infrastructure Verification Checklist

## Pre-Execution Checks

- [ ] All scripts are executable
  ```bash
  ls -l tools/run_parallel_augment.sh tools/monitor_augment.py tools/merge_shard_manifests.sh
  ```

- [ ] Required directories exist
  ```bash
  ls -d Data/ReDSM5 conf logs data/cache data/processed
  ```

- [ ] Input files exist
  ```bash
  ls -l Data/ReDSM5/redsm5_annotations.csv conf/augment_methods.yaml
  ```

- [ ] Dependencies installed
  ```bash
  python -c "import pandas, numpy, tqdm, nlpaug" 2>/dev/null && echo "OK" || echo "MISSING"
  ```

## Dry-Run Test

- [ ] Launcher dry-run works
  ```bash
  ./tools/run_parallel_augment.sh --dry-run --num-shards 2
  ```

- [ ] Commands look correct
  - Check input paths
  - Check output paths
  - Check parameters

## Quick Test (Optional)

- [ ] Run infrastructure test
  ```bash
  ./tools/test_parallel_infrastructure.sh
  ```

- [ ] Test produces output
  - Check: data/processed/augsets_test/
  - Check: logs/augment_test/
  - Check: manifest_final.csv exists

## Full Execution

- [ ] Launch parallel augmentation
  ```bash
  ./tools/run_parallel_augment.sh
  ```

- [ ] All 7 shards start
  ```bash
  ps aux | grep generate_augsets.py | wc -l  # Should show 7+
  ```

- [ ] Log files created
  ```bash
  ls logs/augment/shard_*_of_7.log | wc -l  # Should show 7
  ```

- [ ] Monitor works
  ```bash
  python tools/monitor_augment.py
  ```

## Post-Execution Checks

- [ ] All shards completed successfully
  ```bash
  grep "Completed in" logs/augment/shard_*_of_7.log | wc -l  # Should show 7
  ```

- [ ] No errors in logs
  ```bash
  grep -i error logs/augment/shard_*_of_7.log
  ```

- [ ] Shard manifests created
  ```bash
  ls data/processed/augsets/manifest_shard*_of_7.csv | wc -l  # Should show 7
  ```

- [ ] Datasets created
  ```bash
  find data/processed/augsets -name "dataset.parquet" | wc -l
  ```

## Merge Verification

- [ ] Run manifest merger
  ```bash
  ./tools/merge_shard_manifests.sh
  ```

- [ ] Final manifest created
  ```bash
  ls -lh data/processed/augsets/manifest_final.csv
  ```

- [ ] Manifest has correct structure
  ```bash
  head -1 data/processed/augsets/manifest_final.csv
  # Should show: combo_id,methods,k,rows,dataset_path,status
  ```

- [ ] Row count correct
  ```bash
  wc -l data/processed/augsets/manifest_final.csv
  ```

## Output Validation

- [ ] Check manifest content
  ```bash
  cat data/processed/augsets/manifest_final.csv | column -t -s,
  ```

- [ ] Verify combo directories
  ```bash
  ls -d data/processed/augsets/combo_1_* | head -5
  ```

- [ ] Check dataset files
  ```bash
  python -c "import pandas as pd; df=pd.read_parquet('data/processed/augsets/combo_1_nlp_wordnet_syn/dataset.parquet'); print(f'Rows: {len(df)}, Cols: {list(df.columns)}')"
  ```

- [ ] Check metadata files
  ```bash
  cat data/processed/augsets/combo_1_nlp_wordnet_syn/meta.json | python -m json.tool
  ```

## Performance Verification

- [ ] Check execution time
  ```bash
  grep "Completed in" logs/augment/shard_0_of_7.log
  ```

- [ ] Check resource usage
  - CPU usage was reasonable
  - Memory didn't exceed limits
  - Disk space sufficient

- [ ] Check cache sizes
  ```bash
  du -sh data/cache/*.db
  ```

## Integration Test

- [ ] Manifest readable by pandas
  ```bash
  python -c "import pandas as pd; df=pd.read_csv('data/processed/augsets/manifest_final.csv'); print(f'Total combos: {len(df)}')"
  ```

- [ ] All datasets loadable
  ```bash
  python -c "
  import pandas as pd
  manifest = pd.read_csv('data/processed/augsets/manifest_final.csv')
  for path in manifest['dataset_path'].head(3):
      df = pd.read_parquet(path)
      print(f'{path}: {len(df)} rows')
  "
  ```

## Cleanup Test

- [ ] Cleanup works
  ```bash
  rm -rf data/processed/augsets_test data/cache_test logs/augment_test
  ```

- [ ] No orphan processes
  ```bash
  ps aux | grep generate_augsets.py  # Should show nothing
  ```

## Final Checklist

- [ ] All scripts work as documented
- [ ] Output structure matches documentation
- [ ] No critical errors in logs
- [ ] Manifests contain expected combos
- [ ] Datasets contain augmented samples
- [ ] Ready for integration with training pipeline

## Success Criteria

✅ **PASS** if:
- All 7 shards complete successfully
- Final manifest created with N combos
- All datasets loadable as parquet
- No critical errors in logs
- Output structure matches docs

❌ **FAIL** if:
- Any shard fails with error
- Manifest incomplete or corrupted
- Datasets missing or unreadable
- Critical errors in logs

## Notes

Record any issues or observations:

```
Date: ___________
Shards: 7
Duration: _________
Total Combos: _________
Total Rows: _________
Issues: None / [describe]
```

