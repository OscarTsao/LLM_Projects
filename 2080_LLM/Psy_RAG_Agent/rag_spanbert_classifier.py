#!/usr/bin/env python3
"""
RAG-Enhanced SpanBERT Classification Model for DSM-5 Criteria Matching
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
from rag_retrieval import DSMCriteriaRetriever
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
set_seed(42)
torch.manual_seed(42)
np.random.seed(42)


class RAGDSMCriteriaDataset(Dataset):
    """Dataset for RAG-enhanced DSM-5 criteria classification"""

    def __init__(self, posts, labels, tokenizer, retriever, max_length=512, top_k=10, threshold=0.3):
        """
        Initialize dataset with RAG retrieval

        Args:
            posts: List of post texts
            labels: Dict mapping criterion keys to labels for each post
            tokenizer: Transformer tokenizer
            retriever: DSMCriteriaRetriever instance
            max_length: Max sequence length
            top_k: Number of criteria to retrieve per post
            threshold: Similarity threshold for retrieval
        """
        self.posts = posts
        self.labels = labels
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.max_length = max_length
        self.top_k = top_k
        self.threshold = threshold

        # Pre-retrieve criteria for all posts to avoid repeated computation
        print("Pre-retrieving criteria for all posts...")
        self.retrieved_criteria = {}
        for i, post in enumerate(posts):
            if i % 100 == 0:
                print(f"Processed {i}/{len(posts)} posts")
            self.retrieved_criteria[i] = self.retriever.get_retrieved_criteria_for_classification(
                post, top_k=top_k, threshold=threshold
            )

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, idx):
        post = str(self.posts[idx])
        retrieved_criteria = self.retrieved_criteria[idx]

        # Get all possible criterion keys from labels
        all_criterion_keys = list(self.labels[idx].keys()) if isinstance(self.labels[idx], dict) else []

        # Create examples for each retrieved criterion
        examples = []
        for criterion_key, criterion_text in retrieved_criteria.items():
            # Get label for this criterion (1 if matches, 0 if not retrieved)
            label = self.labels[idx].get(criterion_key, 0)

            # Combine post and criteria with [SEP] token
            combined_text = f"{post} [SEP] {criterion_text}"

            # Tokenize
            encoding = self.tokenizer(
                combined_text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            examples.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long),
                'criterion_key': criterion_key
            })

        # For criteria not retrieved, they are implicitly labeled as 0 (no match)
        # We don't need to create explicit examples for them as per the requirement

        return examples


class RAGSpanBERTClassifier(nn.Module):
    """RAG-enhanced SpanBERT-based binary classifier for DSM-5 criteria matching"""

    def __init__(self, model_name='SpanBERT/spanbert-base-cased', dropout_rate=0.3, num_labels=2):
        super(RAGSpanBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


class RAGDSMClassificationTrainer:
    """Training pipeline for RAG-enhanced DSM-5 criteria classification"""

    def __init__(self, model_name='SpanBERT/spanbert-base-cased', max_length=512,
                 retrieval_top_k=10, retrieval_threshold=0.3):
        self.model_name = model_name
        self.max_length = max_length
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_threshold = retrieval_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.retriever = None

        # Add special tokens if needed
        if self.tokenizer.sep_token is None:
            self.tokenizer.add_special_tokens({'sep_token': '[SEP]'})

    def setup_retriever(self, criteria_path, index_path='./dsm_criteria_index'):
        """Setup the RAG retriever"""
        print("Setting up RAG retriever...")
        self.retriever = DSMCriteriaRetriever(index_path=index_path)

        # Try to load existing index, otherwise build new one
        try:
            self.retriever.load_index()
            print("Loaded existing retrieval index")
        except FileNotFoundError:
            print("Building new retrieval index...")
            self.retriever.load_criteria(criteria_path)
            self.retriever.build_index()

    def load_and_prepare_data(self, translated_posts_path, criteria_path, groundtruth_path):
        """Load and prepare data with RAG retrieval"""
        print("Loading data...")

        # Setup retriever
        self.setup_retriever(criteria_path)

        # Load ground truth
        gt_df = pd.read_csv(groundtruth_path)
        print(f"Loaded {len(gt_df)} posts from ground truth file")

        # Prepare data for RAG training
        posts = []
        labels_list = []

        print("Preparing RAG training data...")
        for idx, row in gt_df.iterrows():
            post_text = row['post_id']  # This contains the actual post text
            posts.append(post_text)

            # Get all criteria columns (skip post_id column)
            criteria_columns = [col for col in gt_df.columns if col != 'post_id']
            post_labels = {}
            for criterion_col in criteria_columns:
                post_labels[criterion_col] = int(row[criterion_col])

            labels_list.append(post_labels)

        print(f"Prepared {len(posts)} posts for RAG training")
        return posts, labels_list

    def create_flat_dataset(self, posts, labels_list, test_size=0.2, val_size=0.1):
        """Create flattened dataset for training"""
        print("Creating flat training dataset...")

        # Split posts and labels
        train_posts, test_posts, train_labels, test_labels = train_test_split(
            posts, labels_list, test_size=test_size, random_state=42
        )
        train_posts, val_posts, train_labels, val_labels = train_test_split(
            train_posts, train_labels, test_size=val_size/(1-test_size), random_state=42
        )

        print(f"Split: Train {len(train_posts)}, Val {len(val_posts)}, Test {len(test_posts)}")

        # Create flat examples for each split
        def create_flat_examples(posts_split, labels_split):
            flat_posts = []
            flat_criteria = []
            flat_labels = []

            for post, post_labels in zip(posts_split, labels_split):
                # Get retrieved criteria for this post
                retrieved_criteria = self.retriever.get_retrieved_criteria_for_classification(
                    post, top_k=self.retrieval_top_k, threshold=self.retrieval_threshold
                )

                # Create examples only for retrieved criteria
                for criterion_key, criterion_text in retrieved_criteria.items():
                    label = post_labels.get(criterion_key, 0)
                    flat_posts.append(post)
                    flat_criteria.append(criterion_text)
                    flat_labels.append(label)

            return flat_posts, flat_criteria, flat_labels

        # Create flat datasets
        train_posts_flat, train_criteria_flat, train_labels_flat = create_flat_examples(train_posts, train_labels)
        val_posts_flat, val_criteria_flat, val_labels_flat = create_flat_examples(val_posts, val_labels)
        test_posts_flat, test_criteria_flat, test_labels_flat = create_flat_examples(test_posts, test_labels)

        print(f"Flat examples: Train {len(train_posts_flat)}, Val {len(val_posts_flat)}, Test {len(test_posts_flat)}")

        # Create standard datasets using the flat examples
        train_dataset = FlatDSMCriteriaDataset(
            train_posts_flat, train_criteria_flat, train_labels_flat, self.tokenizer, self.max_length
        )
        val_dataset = FlatDSMCriteriaDataset(
            val_posts_flat, val_criteria_flat, val_labels_flat, self.tokenizer, self.max_length
        )
        test_dataset = FlatDSMCriteriaDataset(
            test_posts_flat, test_criteria_flat, test_labels_flat, self.tokenizer, self.max_length
        )

        return train_dataset, val_dataset, test_dataset

    def train(self, train_dataset, val_dataset, output_dir='./rag_spanbert_dsm_model',
              num_epochs=3, batch_size=16, learning_rate=2e-5):
        """Train the RAG-enhanced model"""
        print("Initializing RAG-enhanced model...")
        self.model = RAGSpanBERTClassifier(self.model_name)

        # Resize token embeddings if we added special tokens
        if len(self.tokenizer) > self.model.bert.config.vocab_size:
            self.model.bert.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            dataloader_num_workers=4,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            report_to=[]
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        print("Starting RAG-enhanced training...")
        trainer.train()

        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        return trainer

    def predict_for_post(self, post_text, model_path='./rag_spanbert_dsm_model'):
        """Make predictions for a single post using RAG retrieval"""
        if self.model is None:
            self.model = RAGSpanBERTClassifier(self.model_name)
            self.model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))
            self.model.to(self.device)

        self.model.eval()

        # Retrieve relevant criteria
        retrieved_criteria = self.retriever.get_retrieved_criteria_for_classification(
            post_text, top_k=self.retrieval_top_k, threshold=self.retrieval_threshold
        )

        predictions = {}

        with torch.no_grad():
            for criterion_key, criterion_text in retrieved_criteria.items():
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

                outputs = self.model(input_ids, attention_mask)
                logits = outputs["logits"]
                probabilities = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(logits, dim=-1)

                predictions[criterion_key] = {
                    'prediction': prediction.cpu().item(),
                    'probability': probabilities.cpu().numpy()[0],
                    'criterion_text': criterion_text
                }

        return predictions


class FlatDSMCriteriaDataset(Dataset):
    """Standard flat dataset for post-criterion pairs"""

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
            'labels': torch.tensor(label, dtype=torch.long)
        }


def main():
    """Main training and evaluation pipeline for RAG-enhanced model"""
    print("Starting RAG-Enhanced DSM-5 Criteria Classification with SpanBERT")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # Initialize trainer
    trainer = RAGDSMClassificationTrainer(
        retrieval_top_k=10,
        retrieval_threshold=0.3
    )

    # Load and prepare data
    posts, labels_list = trainer.load_and_prepare_data(
        'Data/translated_posts.csv',
        'Data/DSM-5/DSM_Criteria_Array_Fixed_Simplify.json',
        'Data/Groundtruth/criteria_evaluation.csv'
    )

    # Create datasets
    train_dataset, val_dataset, test_dataset = trainer.create_flat_dataset(posts, labels_list)

    # Train model
    trained_model = trainer.train(
        train_dataset,
        val_dataset,
        num_epochs=3,
        batch_size=16 if torch.cuda.is_available() else 8,
        learning_rate=2e-5
    )

    print("\nRAG-enhanced training completed successfully!")

    # Test prediction
    test_post = "I feel very sad and hopeless every day. I can't sleep and have lost my appetite."
    predictions = trainer.predict_for_post(test_post)

    print(f"\nExample predictions for: '{test_post}'")
    for criterion_key, pred_info in predictions.items():
        print(f"{criterion_key}: {pred_info['prediction']} (prob: {pred_info['probability'][1]:.3f})")

    return trainer


if __name__ == "__main__":
    trainer = main()