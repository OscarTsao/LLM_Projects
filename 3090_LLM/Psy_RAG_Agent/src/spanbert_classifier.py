#!/usr/bin/env python3
"""
SpanBERT Classification Model for DSM-5 Criteria Matching
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, TrainingArguments, Trainer,
    get_linear_schedule_with_warmup, set_seed
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
set_seed(42)
torch.manual_seed(42)
np.random.seed(42)

class DSMCriteriaDataset(Dataset):
    """Dataset for DSM-5 criteria classification"""

    def __init__(self, posts, criteria_texts, labels, tokenizer, max_length=512):
        self.posts = posts
        self.criteria_texts = criteria_texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, idx):
        post = str(self.posts[idx])
        criteria = str(self.criteria_texts[idx])
        label = self.labels[idx]

        # Combine post and criteria with [SEP] token
        combined_text = f"{post} [SEP] {criteria}"

        # Tokenize
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class SpanBERTClassifier(nn.Module):
    """SpanBERT-based binary classifier for DSM-5 criteria matching"""

    def __init__(self, model_name='SpanBERT/spanbert-base-cased', dropout_rate=0.3, num_labels=2):
        super(SpanBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class DSMClassificationTrainer:
    """Training pipeline for DSM-5 criteria classification"""

    def __init__(self, model_name='SpanBERT/spanbert-base-cased', max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None

        # Add special tokens if needed
        if self.tokenizer.sep_token is None:
            self.tokenizer.add_special_tokens({'sep_token': '[SEP]'})

    def load_and_prepare_data(self, translated_posts_path, criteria_path, groundtruth_path):
        """Load and prepare all data for training"""
        print("Loading data...")

        # Load DSM-5 criteria
        with open(criteria_path, 'r', encoding='utf-8') as f:
            criteria_data = json.load(f)

        # Create criteria lookup
        criteria_lookup = {}
        for disorder in criteria_data:
            disorder_name = disorder['diagnosis']
            for criterion in disorder['criteria']:
                key = f"{disorder_name} - {criterion['id']}"
                criteria_lookup[key] = criterion['text']

        # Load ground truth (contains both posts and labels)
        gt_df = pd.read_csv(groundtruth_path)
        print(f"Loaded {len(gt_df)} posts from ground truth file")

        # Prepare training data
        training_examples = []

        print("Preparing training examples...")
        for idx, row in gt_df.iterrows():
            post_text = row['post_id']  # This contains the actual post text

            # Get all criteria columns (skip post_id column)
            criteria_columns = [col for col in gt_df.columns if col != 'post_id']

            for criterion_col in criteria_columns:
                if criterion_col in criteria_lookup:
                    criterion_text = criteria_lookup[criterion_col]
                    label = int(row[criterion_col])  # 0 or 1

                    training_examples.append({
                        'post': post_text,
                        'criterion': criterion_text,
                        'label': label,
                        'criterion_name': criterion_col
                    })

        df = pd.DataFrame(training_examples)
        print(f"Created {len(df)} training examples")
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")

        # Print some statistics
        print(f"Positive examples: {sum(df['label'])}")
        print(f"Negative examples: {len(df) - sum(df['label'])}")

        return df

    def create_datasets(self, df, test_size=0.2, val_size=0.1):
        """Split data and create datasets"""
        # Split data
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df['label']
        )
        train_df, val_df = train_test_split(
            train_df, test_size=val_size/(1-test_size), random_state=42, stratify=train_df['label']
        )

        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Create datasets
        train_dataset = DSMCriteriaDataset(
            train_df['post'].values,
            train_df['criterion'].values,
            train_df['label'].values,
            self.tokenizer,
            self.max_length
        )

        val_dataset = DSMCriteriaDataset(
            val_df['post'].values,
            val_df['criterion'].values,
            val_df['label'].values,
            self.tokenizer,
            self.max_length
        )

        test_dataset = DSMCriteriaDataset(
            test_df['post'].values,
            test_df['criterion'].values,
            test_df['label'].values,
            self.tokenizer,
            self.max_length
        )

        return train_dataset, val_dataset, test_dataset

    def train(self, train_dataset, val_dataset, output_dir='./spanbert_dsm_model',
              num_epochs=3, batch_size=16, learning_rate=2e-5):
        """Train the model"""
        print("Initializing model...")
        self.model = SpanBERTClassifier(self.model_name)

        # Resize token embeddings if we added special tokens
        if len(self.tokenizer) > self.model.bert.config.vocab_size:
            self.model.bert.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)

        # Training arguments optimized for RTX 3090
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,  # Enable mixed precision for RTX 3090
            dataloader_num_workers=4,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            report_to=None  # Disable wandb
        )

        # Custom trainer for binary classification
        class BinaryClassificationTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("label")
                outputs = model(**inputs)
                logits = outputs
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return (loss, logits) if return_outputs else loss

        trainer = BinaryClassificationTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        print("Starting training...")
        trainer.train()

        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        return trainer

    def evaluate(self, test_dataset, model_path='./spanbert_dsm_model'):
        """Evaluate the trained model"""
        print("Loading trained model for evaluation...")

        if self.model is None:
            self.model = SpanBERTClassifier(self.model_name)
            self.model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))

        self.model.to(self.device)
        self.model.eval()

        # Create data loader
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        all_predictions = []
        all_labels = []

        print("Evaluating model...")
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        print(f"\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=['No Match', 'Match']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_predictions))

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': all_predictions,
            'labels': all_labels
        }

    def predict(self, post_text, criterion_text, model_path='./spanbert_dsm_model'):
        """Make prediction for a single post-criterion pair"""
        if self.model is None:
            self.model = SpanBERTClassifier(self.model_name)
            self.model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))
            self.model.to(self.device)

        self.model.eval()

        # Prepare input
        combined_text = f"{post_text} [SEP] {criterion_text}"
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1)

        return {
            'prediction': prediction.cpu().item(),
            'probability': probabilities.cpu().numpy()[0]
        }

def main():
    """Main training and evaluation pipeline"""
    print("Starting DSM-5 Criteria Classification with SpanBERT")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # Initialize trainer
    trainer = DSMClassificationTrainer()

    # Load and prepare data
    df = trainer.load_and_prepare_data(
        'Data/translated_posts.csv',
        'Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json',
        'Data/Groundtruth/criteria_evaluation.csv'
    )

    # Create datasets
    train_dataset, val_dataset, test_dataset = trainer.create_datasets(df)

    # Train model
    trained_model = trainer.train(
        train_dataset,
        val_dataset,
        num_epochs=3,
        batch_size=16 if torch.cuda.is_available() else 8,
        learning_rate=2e-5
    )

    # Evaluate model
    results = trainer.evaluate(test_dataset)

    print("\nTraining completed successfully!")
    return trainer, results

if __name__ == "__main__":
    trainer, results = main()