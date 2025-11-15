#!/usr/bin/env python3
"""
Minimal BERT-like classifier for DSM-5 criteria matching using only PyTorch
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

class DSMCriteriaDataset(Dataset):
    """Simple dataset using TF-IDF features"""

    def __init__(self, posts, criteria_texts, labels, max_features=5000):
        self.posts = posts
        self.criteria_texts = criteria_texts
        self.labels = labels

        # Combine posts and criteria
        combined_texts = []
        for post, criteria in zip(posts, criteria_texts):
            combined_texts.append(f"{post} {criteria}")

        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )

        # Fit and transform the texts
        self.features = self.vectorizer.fit_transform(combined_texts).toarray()
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': self.labels[idx]
        }

class SimpleClassifier(nn.Module):
    """Simple neural network classifier"""

    def __init__(self, input_dim, hidden_dim=512, dropout_rate=0.3, num_labels=2):
        super(SimpleClassifier, self).__init__()
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

class MinimalTrainer:
    """Training pipeline using simple neural network"""

    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vectorizer = None

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
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df['label']
        )
        train_df, val_df = train_test_split(
            train_df, test_size=val_size/(1-test_size), random_state=42, stratify=train_df['label']
        )

        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Create training dataset (this fits the vectorizer)
        train_dataset = DSMCriteriaDataset(
            train_df['post'].values,
            train_df['criterion'].values,
            train_df['label'].values,
            self.max_features
        )

        # For validation and test, we need to use the same vectorizer
        val_combined = [f"{post} {criteria}" for post, criteria in zip(val_df['post'], val_df['criterion'])]
        test_combined = [f"{post} {criteria}" for post, criteria in zip(test_df['post'], test_df['criterion'])]

        val_features = train_dataset.vectorizer.transform(val_combined).toarray()
        test_features = train_dataset.vectorizer.transform(test_combined).toarray()

        # Create simple datasets for val and test
        val_dataset = SimpleDataset(val_features, val_df['label'].values)
        test_dataset = SimpleDataset(test_features, test_df['label'].values)

        # Store vectorizer
        self.vectorizer = train_dataset.vectorizer

        return train_dataset, val_dataset, test_dataset

    def train(self, train_dataset, val_dataset, output_dir='./minimal_model',
              num_epochs=10, batch_size=64, learning_rate=1e-3):
        """Train the model"""
        print("Initializing model...")

        # Get input dimension from training data
        input_dim = train_dataset.features.shape[1]
        self.model = SimpleClassifier(input_dim)
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

        print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")

    def evaluate(self, test_dataset, model_path='./minimal_model'):
        """Evaluate the trained model"""
        print("Evaluating model...")

        if self.model is None:
            input_dim = test_dataset.features.shape[1]
            self.model = SimpleClassifier(input_dim)
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
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        print(f"\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=['No Match', 'Match']))

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': all_predictions,
            'labels': all_labels
        }

class SimpleDataset(Dataset):
    """Simple dataset for validation and test"""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': self.labels[idx]
        }

def main():
    """Main training and evaluation pipeline"""
    print("Starting DSM-5 Criteria Classification with Minimal Classifier")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # Initialize trainer
    trainer = MinimalTrainer()

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
        num_epochs=10,
        batch_size=64,
        learning_rate=1e-3
    )

    # Evaluate model
    results = trainer.evaluate(test_dataset)

    print("\nTraining completed successfully!")
    return trainer, results

if __name__ == "__main__":
    trainer, results = main()