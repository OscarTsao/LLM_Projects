#!/usr/bin/env python3
"""Inference script for evidence sentence classification."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EvidencePredictor:
    """Predictor for evidence sentence classification."""
    
    def __init__(self, model_path: str, tokenizer_name: str = None, device: str = None):
        self.model_path = model_path
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Loading model from {model_path}")
        logger.info(f"Using device: {self.device}")
        
        tokenizer_path = tokenizer_name if tokenizer_name else model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Loaded tokenizer from {tokenizer_path}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded and ready for inference")
    
    def predict_single(self, criterion: str, sentence: str, max_length: int = 512) -> Tuple[int, float]:
        """Predict evidence label for a single criterion-sentence pair."""
        inputs = self.tokenizer(
            criterion, sentence, max_length=max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            predicted_label = torch.argmax(probs, dim=-1).item()
            predicted_prob = probs[0, predicted_label].item()
        
        return predicted_label, predicted_prob
    
    def predict_batch(self, pairs: List[Tuple[str, str]], max_length: int = 512, batch_size: int = 32) -> List[Tuple[int, float]]:
        """Predict evidence labels for multiple criterion-sentence pairs."""
        results = []
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            criteria, sentences = zip(*batch_pairs)
            
            inputs = self.tokenizer(
                list(criteria), list(sentences), max_length=max_length,
                padding=True, truncation=True, return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                predicted_labels = torch.argmax(probs, dim=-1).cpu().numpy()
                predicted_probs = probs.cpu().numpy()
            
            for j, label in enumerate(predicted_labels):
                prob = predicted_probs[j, label]
                results.append((int(label), float(prob)))
        
        return results


def predict_single_cli(args):
    """Handle single prediction CLI command."""
    predictor = EvidencePredictor(
        model_path=args.model_path, tokenizer_name=args.tokenizer_name, device=args.device
    )
    
    label, prob = predictor.predict_single(
        criterion=args.criterion, sentence=args.sentence, max_length=args.max_length
    )
    
    result = {
        'criterion': args.criterion, 'sentence': args.sentence,
        'predicted_label': label, 'predicted_probability': prob, 'is_evidence': bool(label)
    }
    
    print(json.dumps(result, indent=2))
    logger.info(f"Prediction: {'Evidence' if label else 'Not Evidence'} (confidence: {prob:.4f})")


def predict_batch_cli(args):
    """Handle batch prediction CLI command."""
    predictor = EvidencePredictor(
        model_path=args.model_path, tokenizer_name=args.tokenizer_name, device=args.device
    )
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    if input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
    elif input_path.suffix == '.json':
        df = pd.read_json(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    if 'criterion' not in df.columns or 'sentence' not in df.columns:
        raise ValueError("Input file must contain 'criterion' and 'sentence' columns")
    
    pairs = list(zip(df['criterion'], df['sentence']))
    logger.info(f"Processing {len(pairs)} samples...")
    
    results = predictor.predict_batch(pairs, max_length=args.max_length, batch_size=args.batch_size)
    
    df['predicted_label'] = [label for label, _ in results]
    df['predicted_probability'] = [prob for _, prob in results]
    df['is_evidence'] = df['predicted_label'].astype(bool)
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.csv':
        df.to_csv(output_path, index=False)
    elif output_path.suffix == '.json':
        df.to_json(output_path, orient='records', indent=2)
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")
    
    logger.info(f"Saved predictions to {output_path}")
    
    evidence_count = df['is_evidence'].sum()
    logger.info(f"Results: {evidence_count}/{len(df)} classified as evidence ({100 * evidence_count / len(df):.2f}%)")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Evidence Sentence Classification Inference")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    single_parser = subparsers.add_parser('single', help='Predict single pair')
    single_parser.add_argument('--model-path', type=str, required=True, help='Path to trained model directory')
    single_parser.add_argument('--criterion', type=str, required=True, help='DSM-5 criterion text')
    single_parser.add_argument('--sentence', type=str, required=True, help='Sentence text to classify')
    single_parser.add_argument('--tokenizer-name', type=str, default=None, help='Tokenizer name')
    single_parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Device to run on')
    single_parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')
    
    batch_parser = subparsers.add_parser('batch', help='Predict batch from file')
    batch_parser.add_argument('--model-path', type=str, required=True, help='Path to trained model directory')
    batch_parser.add_argument('--input-file', type=str, required=True, help='Input CSV or JSON file')
    batch_parser.add_argument('--output-file', type=str, required=True, help='Output file path for predictions')
    batch_parser.add_argument('--tokenizer-name', type=str, default=None, help='Tokenizer name')
    batch_parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Device to run on')
    batch_parser.add_argument('--max-length', type=int, default=512, help='Maximum sequence length')
    batch_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        predict_single_cli(args)
    elif args.command == 'batch':
        predict_batch_cli(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
