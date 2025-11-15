#!/usr/bin/env python3
import argparse, json, math, sys, os


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_first_present(d, keys):
    for k in keys:
        if k in d and isinstance(d[k], (int, float)):
            return float(d[k]), k
    return None, None


def main(argv=None):
    p = argparse.ArgumentParser(description="Check acceptance metric gates and exit non-zero on breach.")
    p.add_argument('--metrics', help='Path to a single metrics JSON to evaluate')
    p.add_argument('--val', help='Optional path to val metrics JSON')
    p.add_argument('--test', help='Optional path to test metrics JSON')

    # Thresholds (defaults from constitution/spec)
    p.add_argument('--evidence-f1-min', type=float, default=None,
                   help='Minimum Evidence macro-F1 (present). If omitted, this gate is skipped.')
    p.add_argument('--neg-precision-min', type=float, default=0.90)
    p.add_argument('--criteria-auroc-min', type=float, default=0.80)
    p.add_argument('--ece-max', type=float, default=0.05)

    # Metric key aliases (to avoid coupling to exact names across repos)
    p.add_argument('--evidence-f1-keys', nargs='*', default=[
        'evidence_macro_f1_present', 'macro_F1_present', 'evidence_f1_present', 'evidence_f1'
    ])
    p.add_argument('--neg-precision-keys', nargs='*', default=[
        'negation_precision', 'neg_precision'
    ])
    p.add_argument('--criteria-auroc-keys', nargs='*', default=[
        'criteria_auroc', 'auroc_criteria', 'criteria_roc_auc', 'criteria_auroc_macro'
    ])
    p.add_argument('--ece-keys', nargs='*', default=[
        'ece', 'expected_calibration_error', 'criteria_ece'
    ])

    args = p.parse_args(argv)

    metrics_paths = []
    if args.metrics:
        metrics_paths.append(args.metrics)
    if args.val:
        metrics_paths.append(args.val)
    if args.test:
        metrics_paths.append(args.test)
    metrics_paths = [p for p in metrics_paths if p]

    if not metrics_paths:
        print('ERROR: Provide --metrics or --val/--test paths', file=sys.stderr)
        return 2

    # Load and merge metrics: last wins for overlapping keys
    merged = {}
    loaded_any = False
    for path in metrics_paths:
        if not os.path.isfile(path):
            print(f'WARNING: metrics file not found: {path}', file=sys.stderr)
            continue
        try:
            data = load_json(path)
            if isinstance(data, dict):
                merged.update(data)
                loaded_any = True
            else:
                print(f'WARNING: metrics at {path} is not a JSON object', file=sys.stderr)
        except Exception as e:
            print(f'WARNING: failed to load metrics {path}: {e}', file=sys.stderr)

    if not loaded_any:
        print('ERROR: No metrics loaded; cannot evaluate gates', file=sys.stderr)
        return 2

    failures = []
    findings = []

    # Evidence F1 gate (optional unless provided)
    if args.evidence_f1_min is not None:
        v, key = get_first_present(merged, args.evidence_f1_keys)
        if v is None:
            failures.append('Missing evidence F1 metric')
            findings.append({'gate': 'evidence_f1', 'status': 'missing', 'key': None, 'need': args.evidence_f1_min})
        else:
            ok = v >= args.evidence_f1_min
            if not ok:
                failures.append(f'F1 {v:.4f} < min {args.evidence_f1_min:.4f}')
            findings.append({'gate': 'evidence_f1', 'status': 'pass' if ok else 'fail', 'key': key, 'got': v, 'need': args.evidence_f1_min})

    # Negation precision gate
    v, key = get_first_present(merged, args.neg_precision_keys)
    if v is None:
        failures.append('Missing negation precision metric')
        findings.append({'gate': 'neg_precision', 'status': 'missing', 'key': None, 'need': args.neg_precision_min})
    else:
        ok = v >= args.neg_precision_min
        if not ok:
            failures.append(f'Neg precision {v:.4f} < min {args.neg_precision_min:.4f}')
        findings.append({'gate': 'neg_precision', 'status': 'pass' if ok else 'fail', 'key': key, 'got': v, 'need': args.neg_precision_min})

    # Criteria AUROC gate
    v, key = get_first_present(merged, args.criteria_auroc_keys)
    if v is None:
        failures.append('Missing criteria AUROC metric')
        findings.append({'gate': 'criteria_auroc', 'status': 'missing', 'key': None, 'need': args.criteria_auroc_min})
    else:
        ok = v >= args.criteria_auroc_min
        if not ok:
            failures.append(f'AUROC {v:.4f} < min {args.criteria_auroc_min:.4f}')
        findings.append({'gate': 'criteria_auroc', 'status': 'pass' if ok else 'fail', 'key': key, 'got': v, 'need': args.criteria_auroc_min})

    # ECE gate (<= max)
    v, key = get_first_present(merged, args.ece_keys)
    if v is None:
        failures.append('Missing ECE metric')
        findings.append({'gate': 'ece', 'status': 'missing', 'key': None, 'need': args.ece_max})
    else:
        ok = v <= args.ece_max
        if not ok:
            failures.append(f'ECE {v:.4f} > max {args.ece_max:.4f}')
        findings.append({'gate': 'ece', 'status': 'pass' if ok else 'fail', 'key': key, 'got': v, 'need': args.ece_max})

    # Compact output
    print('Gate Check Summary:')
    for f in findings:
        label = f['gate']
        status = f['status']
        if status == 'missing':
            print(f"- {label}: MISSING (need {f['need']})")
        else:
            print(f"- {label}: {status.upper()} (got {f['got']:.4f} via '{f['key']}', need {f['need']})")

    if failures:
        print('\nGATES FAILED:', file=sys.stderr)
        for msg in failures:
            print(f'- {msg}', file=sys.stderr)
        return 1

    print('\nAll gates passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

