#!/usr/bin/env python3
"""Generate ground truth datasets with STRICT validation.

This script:
1. Loads posts, annotations, and DSM criteria
2. Validates required columns per field_map.yaml
3. Generates criteria_groundtruth.csv (uses ONLY status field)
4. Generates evidence_groundtruth.csv (uses ONLY cases field)
5. Generates splits.json with train/val/test post_ids
6. Prints validation report

STRICT RULES:
- Criteria labels come ONLY from 'status' field
- Evidence comes ONLY from 'cases' field
- Assertions FAIL if rules are violated
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from psy_agents_noaug.data.groundtruth import (
    create_criteria_groundtruth,
    create_evidence_groundtruth,
    load_field_map,
    validate_strict_separation,
    GroundTruthValidator,
)
from psy_agents_noaug.data.loaders import (
    ReDSM5DataLoader,
    group_split_by_post_id,
    save_splits_json,
)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print('=' * 80)


def print_dataframe_info(df: pd.DataFrame, name: str):
    """Print DataFrame info."""
    print(f"\n{name}:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    if 'post_id' in df.columns:
        print(f"  Unique post_ids: {df['post_id'].nunique()}")
    if 'criterion_id' in df.columns:
        print(f"  Unique criterion_ids: {df['criterion_id'].nunique()}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate ground truth datasets with STRICT validation'
    )
    parser.add_argument(
        '--config-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'configs' / 'data',
        help='Directory containing field_map.yaml'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'data' / 'raw' / 'redsm5',
        help='Directory containing posts.csv and annotations.csv'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'data' / 'processed',
        help='Output directory for ground truth files'
    )
    parser.add_argument(
        '--dsm-criteria',
        type=Path,
        help='Path to DSM criteria JSON (default: data-dir/dsm_criteria.json)'
    )
    parser.add_argument(
        '--data-source',
        choices=['local', 'huggingface'],
        default='local',
        help='Data source type'
    )
    parser.add_argument(
        '--hf-dataset',
        type=str,
        default='irlab-udc/redsm5',
        help='HuggingFace dataset name'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for splits'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation checks (not recommended)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    field_map_path = args.config_dir / 'field_map.yaml'
    dsm_criteria_path = args.dsm_criteria or (args.data_dir / 'dsm_criteria.json')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print_section("Configuration")
    print(f"Field map: {field_map_path}")
    print(f"Data source: {args.data_source}")
    print(f"Data directory: {args.data_dir}")
    print(f"DSM criteria: {dsm_criteria_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print(f"Random seed: {args.random_seed}")
    
    # Load field mapping
    print_section("Loading Field Mapping")
    field_map = load_field_map(field_map_path)
    print(f"Loaded field mapping configuration")
    print(f"  Status field (criteria): {field_map['annotations']['status']}")
    print(f"  Cases field (evidence): {field_map['annotations']['cases']}")
    print(f"  Strict mode: {field_map.get('validation', {}).get('strict_mode', True)}")
    
    # Initialize loader
    print_section("Loading Data")
    loader = ReDSM5DataLoader(
        field_map=field_map,
        data_source=args.data_source,
        data_dir=args.data_dir if args.data_source == 'local' else None,
        hf_dataset_name=args.hf_dataset if args.data_source == 'huggingface' else None
    )
    
    # Load posts and annotations
    posts = loader.load_posts()
    print_dataframe_info(posts, "Posts")
    
    annotations = loader.load_annotations()
    print_dataframe_info(annotations, "Annotations")
    
    # Load DSM criteria
    dsm_criteria = loader.load_dsm_criteria(dsm_criteria_path)
    valid_criterion_ids = {c['id'] for c in dsm_criteria}
    print(f"\nDSM Criteria:")
    print(f"  Count: {len(dsm_criteria)}")
    print(f"  IDs: {sorted(valid_criterion_ids)}")
    
    # Create splits
    print_section("Creating Splits")
    train_post_ids, val_post_ids, test_post_ids = group_split_by_post_id(
        df=annotations,
        post_id_col=field_map['annotations']['post_id'],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
    
    print(f"Split sizes:")
    print(f"  Train: {len(train_post_ids)} posts")
    print(f"  Val: {len(val_post_ids)} posts")
    print(f"  Test: {len(test_post_ids)} posts")
    
    # Save splits
    splits_path = args.output_dir / 'splits.json'
    save_splits_json(
        train_post_ids=train_post_ids,
        val_post_ids=val_post_ids,
        test_post_ids=test_post_ids,
        output_path=splits_path,
        metadata={
            'random_seed': args.random_seed,
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio,
            'test_ratio': args.test_ratio,
            'data_source': args.data_source
        }
    )
    
    # Generate criteria groundtruth
    print_section("Generating Criteria Groundtruth")
    print("USING ONLY 'status' FIELD (STRICT RULE)")
    
    criteria_gt = create_criteria_groundtruth(
        annotations=annotations,
        posts=posts,
        field_map=field_map,
        valid_criterion_ids=valid_criterion_ids
    )
    
    print_dataframe_info(criteria_gt, "Criteria Groundtruth")
    print(f"  Label distribution:")
    print(criteria_gt['label'].value_counts().to_string())
    
    # Save criteria groundtruth
    criteria_path = args.output_dir / 'criteria_groundtruth.csv'
    criteria_gt.to_csv(criteria_path, index=False)
    print(f"\nSaved to: {criteria_path}")
    
    # Generate evidence groundtruth
    print_section("Generating Evidence Groundtruth")
    print("USING ONLY 'cases' FIELD (STRICT RULE)")
    
    evidence_gt = create_evidence_groundtruth(
        annotations=annotations,
        posts=posts,
        field_map=field_map,
        valid_criterion_ids=valid_criterion_ids
    )
    
    print_dataframe_info(evidence_gt, "Evidence Groundtruth")
    if len(evidence_gt) > 0:
        print(f"  Evidence spans per criterion:")
        print(evidence_gt.groupby('criterion_id').size().to_string())
    
    # Save evidence groundtruth
    evidence_path = args.output_dir / 'evidence_groundtruth.csv'
    evidence_gt.to_csv(evidence_path, index=False)
    print(f"\nSaved to: {evidence_path}")
    
    # Validate strict separation
    if not args.skip_validation:
        print_section("Validating Strict Separation")
        try:
            validate_strict_separation(criteria_gt, evidence_gt, field_map)
            print("\n✓ VALIDATION PASSED: Strict field separation maintained")
        except AssertionError as e:
            print(f"\n✗ VALIDATION FAILED: {e}")
            sys.exit(1)
        
        # Additional validation
        validator = GroundTruthValidator(field_map, valid_criterion_ids)
        
        print("\nValidating criteria groundtruth...")
        criteria_validation = validator.validate_criteria_groundtruth(criteria_gt)
        if criteria_validation['errors']:
            print("ERRORS:")
            for error in criteria_validation['errors']:
                print(f"  ✗ {error}")
            sys.exit(1)
        else:
            print("  ✓ No errors")
        
        if criteria_validation['warnings']:
            print("WARNINGS:")
            for warning in criteria_validation['warnings']:
                print(f"  ⚠ {warning}")
        
        print("\nValidating evidence groundtruth...")
        evidence_validation = validator.validate_evidence_groundtruth(evidence_gt)
        if evidence_validation['errors']:
            print("ERRORS:")
            for error in evidence_validation['errors']:
                print(f"  ✗ {error}")
            sys.exit(1)
        else:
            print("  ✓ No errors")
        
        if evidence_validation['warnings']:
            print("WARNINGS:")
            for warning in evidence_validation['warnings']:
                print(f"  ⚠ {warning}")
    
    # Final summary
    print_section("Summary")
    print("Generated files:")
    print(f"  1. {criteria_path}")
    print(f"     - {len(criteria_gt)} rows")
    print(f"     - Columns: {list(criteria_gt.columns)}")
    print(f"  2. {evidence_path}")
    print(f"     - {len(evidence_gt)} rows")
    print(f"     - Columns: {list(evidence_gt.columns)}")
    print(f"  3. {splits_path}")
    print(f"     - Train/Val/Test post IDs")
    print("\n✓ Ground truth generation complete!")
    print("\nSTRICT VALIDATION RULES ENFORCED:")
    print("  ✓ Criteria labels from 'status' field ONLY")
    print("  ✓ Evidence from 'cases' field ONLY")
    print("  ✓ No cross-contamination")


if __name__ == '__main__':
    main()
