"""
Standalone evaluation script for trained Gemma QA models.

Loads a checkpoint and evaluates on test set with detailed metrics.
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import argparse
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.gemma_qa import GemmaQA
from data.evidence_dataset import load_redsm5_evidence, get_symptom_labels, SYMPTOM_LABELS
from training.qa_metrics import extract_answer_from_logits, compute_exact_match, compute_f1
from utils.logger import setup_logger

logger = setup_logger('evaluate')


def evaluate_model(
    model: GemmaQA,
    tokenizer,
    dataloader: DataLoader,
    device: str = 'cuda',
) -> dict:
    """
    Evaluate model with detailed metrics.

    Returns overall and per-symptom metrics with predictions.
    """
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions']
            end_positions = batch['end_positions']
            symptom_indices = batch['symptom_idx']
            post_ids = batch['post_id']

            # Forward pass
            start_logits, end_logits = model(input_ids, attention_mask)

            start_logits_np = start_logits.float().cpu().numpy()
            end_logits_np = end_logits.float().cpu().numpy()

            # Extract predictions for each example in batch
            for i in range(len(input_ids)):
                input_ids_list = input_ids[i].cpu().tolist()

                # Predicted answer
                pred_text, pred_start, pred_end, score = extract_answer_from_logits(
                    start_logits_np[i],
                    end_logits_np[i],
                    input_ids_list,
                    tokenizer,
                )

                # Ground truth answer
                start_idx = start_positions[i].item()
                end_idx = end_positions[i].item()

                if start_idx == 0 and end_idx == 0:
                    gt_text = ""
                else:
                    answer_tokens = input_ids[i][start_idx:end_idx + 1]
                    gt_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)

                # Compute metrics
                em = compute_exact_match(pred_text, gt_text)
                f1 = compute_f1(pred_text, gt_text)

                all_predictions.append({
                    'post_id': post_ids[i],
                    'symptom': SYMPTOM_LABELS[symptom_indices[i].item()],
                    'symptom_idx': symptom_indices[i].item(),
                    'prediction': pred_text,
                    'ground_truth': gt_text,
                    'exact_match': em,
                    'f1': f1,
                    'confidence_score': score,
                })

    # Compute overall metrics
    predictions_df = pd.DataFrame(all_predictions)
    overall_metrics = {
        'exact_match': predictions_df['exact_match'].mean(),
        'f1': predictions_df['f1'].mean(),
        'num_examples': len(predictions_df),
    }

    # Per-symptom metrics
    per_symptom_metrics = {}
    for symptom in SYMPTOM_LABELS:
        symptom_df = predictions_df[predictions_df['symptom'] == symptom]
        if len(symptom_df) > 0:
            per_symptom_metrics[symptom] = {
                'exact_match': symptom_df['exact_match'].mean(),
                'f1': symptom_df['f1'].mean(),
                'count': len(symptom_df),
            }

    return {
        'overall': overall_metrics,
        'per_symptom': per_symptom_metrics,
        'predictions': all_predictions,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Gemma QA model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/redsm5',
                        help='Path to data directory')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Gemma model name (defaults to checkpoint metadata or google/gemma-2-2b)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to evaluate')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Load data
    logger.info(f"Loading data from {args.data_dir}")
    train_dataset, val_dataset, test_dataset = load_redsm5_evidence(
        args.data_dir,
        tokenizer,
        max_length=args.max_length,
    )

    # Select split
    if args.split == 'train':
        dataset = train_dataset
    elif args.split == 'val':
        dataset = val_dataset
    else:
        dataset = test_dataset

    logger.info(f"Evaluating on {args.split} set ({len(dataset)} examples)")

    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    logger.info(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    stored_kwargs = checkpoint.get('model_kwargs') or {}
    model_kwargs = dict(stored_kwargs)
    model_name = args.model_name or model_kwargs.get('model_name') or 'google/gemma-2-2b'
    model_kwargs['model_name'] = model_name
    model = GemmaQA(device=device, **model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    logger.info("Evaluating...")
    results = evaluate_model(model, tokenizer, dataloader, device)

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Exact Match: {results['overall']['exact_match']:.4f}")
    logger.info(f"  F1 Score:    {results['overall']['f1']:.4f}")
    logger.info(f"  Examples:    {results['overall']['num_examples']}")

    logger.info(f"\nPer-Symptom Metrics:")
    for symptom, metrics in sorted(results['per_symptom'].items()):
        logger.info(
            f"  {symptom:20s}: EM={metrics['exact_match']:.4f}, "
            f"F1={metrics['f1']:.4f}, N={metrics['count']}"
        )

    # Save results
    logger.info(f"\nSaving results to {output_dir}")

    # Save metrics
    metrics_path = output_dir / f'{args.split}_metrics.json'
    with open(metrics_path, 'w') as f:
        metrics_to_save = {
            'overall': results['overall'],
            'per_symptom': results['per_symptom'],
        }
        json.dump(metrics_to_save, f, indent=2)

    # Save predictions
    predictions_path = output_dir / f'{args.split}_predictions.csv'
    predictions_df = pd.DataFrame(results['predictions'])
    predictions_df.to_csv(predictions_path, index=False)

    logger.info(f"Metrics saved to {metrics_path}")
    logger.info(f"Predictions saved to {predictions_path}")


if __name__ == '__main__':
    main()
