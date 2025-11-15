#!/usr/bin/env python3
"""
Basic neural network classifier for DSM-5 criteria matching using only PyTorch and basic libraries
"""

# Standard library imports
import json
import os
import random
import re
import warnings
from collections import Counter

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class BasicTokenizer:
    """Simple tokenizer that creates word counts"""

    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.word_to_idx = {}
        self.vocab_size = 0

    def build_vocab(self, texts):
        """Build vocabulary from training texts"""
        word_counts = Counter()

        for text in texts:
            # Simple tokenization
            words = self.tokenize(text)
            word_counts.update(words)

        # Get most common words
        most_common = word_counts.most_common(self.max_features - 1)  # -1 for unknown token

        # Build word to index mapping
        self.word_to_idx = {'<UNK>': 0}
        for i, (word, count) in enumerate(most_common):
            self.word_to_idx[word] = i + 1

        self.vocab_size = len(self.word_to_idx)
        print(f"Built vocabulary with {self.vocab_size} words")

    def tokenize(self, text):
        """Simple tokenization"""
        # Convert to lowercase and remove punctuation
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        words = text.split()
        return [word for word in words if len(word) > 2]  # Filter short words

    def text_to_vector(self, text):
        """Convert text to word count vector"""
        vector = np.zeros(self.vocab_size)
        words = self.tokenize(text)

        for word in words:
            idx = self.word_to_idx.get(word, 0)  # 0 is <UNK>
            vector[idx] += 1

        return vector

class DSMCriteriaDataset(Dataset):
    """Dataset using simple word count features"""

    def __init__(self, posts, criteria_texts, labels, tokenizer=None, is_training=True):
        self.posts = posts
        self.criteria_texts = criteria_texts
        self.labels = labels
        self.is_training = is_training

        if is_training:
            # Create and fit tokenizer on training data
            self.tokenizer = BasicTokenizer()
            combined_texts = [f"{post} {criteria}" for post, criteria in zip(posts, criteria_texts)]
            self.tokenizer.build_vocab(combined_texts)
        else:
            # Use existing tokenizer
            self.tokenizer = tokenizer

        # Convert texts to feature vectors
        self.features = []
        for post, criteria in zip(posts, criteria_texts):
            combined = f"{post} {criteria}"
            vector = self.tokenizer.text_to_vector(combined)
            self.features.append(vector)

        self.features = torch.FloatTensor(self.features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': self.labels[idx]
        }

class BasicClassifier(nn.Module):
    """Simple neural network classifier"""

    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.3, num_labels=2):
        super(BasicClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_labels)
        )

    def forward(self, features):
        return self.network(features)

def simple_train_test_split(df, test_size=0.2, val_size=0.1, random_state=42):
    """Simple train/validation/test split"""
    np.random.seed(random_state)

    # Separate by labels to ensure stratification
    df_0 = df[df['label'] == 0].copy()
    df_1 = df[df['label'] == 1].copy()

    # Shuffle
    df_0 = df_0.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_1 = df_1.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Calculate split sizes
    test_size_0 = int(len(df_0) * test_size)
    test_size_1 = int(len(df_1) * test_size)

    val_size_0 = int(len(df_0) * val_size)
    val_size_1 = int(len(df_1) * val_size)

    # Split class 0
    test_0 = df_0[:test_size_0]
    val_0 = df_0[test_size_0:test_size_0 + val_size_0]
    train_0 = df_0[test_size_0 + val_size_0:]

    # Split class 1
    test_1 = df_1[:test_size_1]
    val_1 = df_1[test_size_1:test_size_1 + val_size_1]
    train_1 = df_1[test_size_1 + val_size_1:]

    # Combine
    train_df = pd.concat([train_0, train_1]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df = pd.concat([val_0, val_1]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat([test_0, test_1]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    return train_df, val_df, test_df

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics including confusion matrix"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Accuracy
    accuracy = (y_true == y_pred).mean()

    # Confusion Matrix (2x2 for binary classification)
    confusion_matrix = np.zeros((2, 2), dtype=int)
    for true_label in [0, 1]:
        for pred_label in [0, 1]:
            confusion_matrix[true_label, pred_label] = ((y_true == true_label) & (y_pred == pred_label)).sum()

    # Precision, Recall, F1 for each class
    metrics = {}
    for class_id in [0, 1]:
        tp = ((y_pred == class_id) & (y_true == class_id)).sum()
        fp = ((y_pred == class_id) & (y_true != class_id)).sum()
        fn = ((y_pred != class_id) & (y_true == class_id)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[f'class_{class_id}'] = {'precision': precision, 'recall': recall, 'f1': f1}

    # Overall metrics
    overall_precision = sum(metrics[f'class_{i}']['precision'] for i in [0, 1]) / 2
    overall_recall = sum(metrics[f'class_{i}']['recall'] for i in [0, 1]) / 2

    # Weighted metrics
    class_counts = [(y_true == i).sum() for i in [0, 1]]
    total_count = sum(class_counts)
    weighted_precision = sum(metrics[f'class_{i}']['precision'] * class_counts[i] for i in [0, 1]) / total_count
    weighted_recall = sum(metrics[f'class_{i}']['recall'] * class_counts[i] for i in [0, 1]) / total_count
    weighted_f1 = sum(metrics[f'class_{i}']['f1'] * class_counts[i] for i in [0, 1]) / total_count

    return {
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix,
        'precision': overall_precision,
        'recall': overall_recall,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class_metrics': metrics
    }

class BasicTrainer:
    """Training pipeline using basic neural network"""

    def __init__(self, max_features=3000):
        self.max_features = max_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None

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

        # Load ground truth
        gt_df = pd.read_csv(groundtruth_path)
        print(f"Loaded {len(gt_df)} posts from ground truth file")

        # Prepare training data
        training_examples = []

        print("Preparing training examples...")
        for idx, row in gt_df.iterrows():
            post_text = row['post_id']
            criteria_columns = [col for col in gt_df.columns if col != 'post_id']

            for criterion_col in criteria_columns:
                if criterion_col in criteria_lookup:
                    criterion_text = criteria_lookup[criterion_col]
                    label = int(row[criterion_col])

                    training_examples.append({
                        'post': post_text,
                        'criterion': criterion_text,
                        'label': label,
                        'criterion_name': criterion_col
                    })

        df = pd.DataFrame(training_examples)
        print(f"Created {len(df)} training examples")
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")

        return df

    def create_datasets(self, df, test_size=0.2, val_size=0.1):
        """Split data and create datasets"""
        # Split data
        train_df, val_df, test_df = simple_train_test_split(df, test_size, val_size)
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Create training dataset (this builds the tokenizer)
        train_dataset = DSMCriteriaDataset(
            train_df['post'].values,
            train_df['criterion'].values,
            train_df['label'].values,
            is_training=True
        )

        # Create validation and test datasets using the same tokenizer
        val_dataset = DSMCriteriaDataset(
            val_df['post'].values,
            val_df['criterion'].values,
            val_df['label'].values,
            tokenizer=train_dataset.tokenizer,
            is_training=False
        )

        test_dataset = DSMCriteriaDataset(
            test_df['post'].values,
            test_df['criterion'].values,
            test_df['label'].values,
            tokenizer=train_dataset.tokenizer,
            is_training=False
        )

        # Store tokenizer
        self.tokenizer = train_dataset.tokenizer

        return train_dataset, val_dataset, test_dataset

    def train(self, train_dataset, val_dataset, output_dir='./basic_model',
              num_epochs=15, batch_size=64, learning_rate=1e-3):
        """Train the model"""
        print("Initializing model...")

        # Get input dimension from training data
        input_dim = train_dataset.features.shape[1]
        self.model = BasicClassifier(input_dim)
        self.model.to(self.device)

        # Optimizers and loss
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print("Starting training...")
        best_val_acc = 0.0

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_accuracy = 100 * train_correct / train_total

            # Validation phase
            self.model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    labels = batch['label'].to(self.device)

                    outputs = self.model(features)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_accuracy = 100 * val_correct / val_total

            print(f'Epoch {epoch + 1}/{num_epochs}: Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')

            # Save best model
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                os.makedirs(output_dir, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(output_dir, 'model.pth'))
                torch.save(self.tokenizer, os.path.join(output_dir, 'tokenizer.pkl'))

        print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")

    def evaluate(self, test_dataset, model_path='./basic_model'):
        """Evaluate the trained model"""
        print("Evaluating model...")

        if self.model is None:
            input_dim = test_dataset.features.shape[1]
            self.model = BasicClassifier(input_dim)
            self.model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))

        self.model.to(self.device)
        self.model.eval()

        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(features)
                predictions = torch.argmax(outputs, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        metrics_result = calculate_metrics(all_labels, all_predictions)

        print(f"\nEvaluation Results:")
        print(f"Accuracy: {metrics_result['accuracy']:.4f}")
        print(f"Precision: {metrics_result['precision']:.4f}")
        print(f"Recall: {metrics_result['recall']:.4f}")
        print(f"Weighted Precision: {metrics_result['weighted_precision']:.4f}")
        print(f"Weighted Recall: {metrics_result['weighted_recall']:.4f}")
        print(f"Weighted F1 Score: {metrics_result['weighted_f1']:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              0    1")
        print(f"Actual 0   {metrics_result['confusion_matrix'][0, 0]:4d} {metrics_result['confusion_matrix'][0, 1]:4d}")
        print(f"Actual 1   {metrics_result['confusion_matrix'][1, 0]:4d} {metrics_result['confusion_matrix'][1, 1]:4d}")

        print("\nPer-class metrics:")
        for class_name, metrics in metrics_result['per_class_metrics'].items():
            print(f"{class_name}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")

        return {
            'accuracy': metrics_result['accuracy'],
            'precision': metrics_result['precision'],
            'recall': metrics_result['recall'],
            'weighted_precision': metrics_result['weighted_precision'],
            'weighted_recall': metrics_result['weighted_recall'],
            'f1_score': metrics_result['weighted_f1'],
            'confusion_matrix': metrics_result['confusion_matrix'],
            'predictions': all_predictions,
            'labels': all_labels,
            'per_class_metrics': metrics_result['per_class_metrics']
        }

def main():
    """Main training and evaluation pipeline"""
    print("Starting DSM-5 Criteria Classification with Basic Classifier")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # Initialize trainer
    trainer = BasicTrainer(max_features=3000)

    # Load and prepare data
    df = trainer.load_and_prepare_data(
        'Data/translated_posts.csv',
        'Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json',
        'Data/Groundtruth/criteria_evaluation.csv'
    )

    # Create datasets
    train_dataset, val_dataset, test_dataset = trainer.create_datasets(df)

    # Train model
    trainer.train(
        train_dataset,
        val_dataset,
        num_epochs=15,
        batch_size=64,
        learning_rate=1e-3
    )

    # Evaluate model
    results = trainer.evaluate(test_dataset)

    print("\n🎉 Training completed successfully!")
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"Final Test F1 Score: {results['f1_score']:.4f}")

    return trainer, results

if __name__ == "__main__":
    trainer, results = main()