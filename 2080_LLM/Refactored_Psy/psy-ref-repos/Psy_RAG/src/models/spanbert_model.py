"""
SpanBERT model implementation for criteria filtering and token extraction
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict, Tuple, Optional
import logging
import numpy as np
from dataclasses import dataclass
import re
from ..utils.performance_optimizer import PerformanceOptimizer

logger = logging.getLogger(__name__)


@dataclass
class SpanResult:
    """Result from SpanBERT span extraction"""
    text: str
    start: int
    end: int
    confidence: float
    label: str


class SpanBERTModel:
    """SpanBERT model for span extraction and filtering"""
    
    def __init__(self, model_name: str = "SpanBERT/spanbert-base-cased", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.optimizer = PerformanceOptimizer()
        self._load_model()
    
    def _load_model(self):
        """Load SpanBERT model and tokenizer"""
        try:
            logger.info(f"Loading SpanBERT model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                num_labels=3  # B, I, O tags for span detection
            )
            
            # Apply performance optimizations for RTX 3090
            if self.device == "cuda":
                self.model = self.optimizer.optimize_model_for_inference(self.model)
            else:
                self.model.to(self.device)
                self.model.eval()
            logger.info("SpanBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading SpanBERT model: {e}")
            raise
    
    def extract_spans(
        self, 
        text: str, 
        confidence_threshold: float = 0.5,
        max_spans: int = 5
    ) -> List[SpanResult]:
        """
        Extract relevant spans from text using SpanBERT
        
        Args:
            text: Input text
            confidence_threshold: Minimum confidence for span extraction
            max_spans: Maximum number of spans to return
            
        Returns:
            List of SpanResult objects
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_labels = torch.argmax(predictions, dim=-1)
                confidences = torch.max(predictions, dim=-1)[0]
            
            # Convert to CPU for processing
            predicted_labels = predicted_labels.cpu().numpy()[0]
            confidences = confidences.cpu().numpy()[0]
            input_ids = inputs['input_ids'].cpu().numpy()[0]
            
            # Extract spans
            spans = self._extract_spans_from_predictions(
                text, input_ids, predicted_labels, confidences, 
                confidence_threshold, max_spans
            )
            
            return spans
            
        except Exception as e:
            logger.error(f"Error extracting spans: {e}")
            return []
    
    def _extract_spans_from_predictions(
        self, 
        text: str, 
        input_ids: np.ndarray, 
        predicted_labels: np.ndarray, 
        confidences: np.ndarray,
        confidence_threshold: float,
        max_spans: int
    ) -> List[SpanResult]:
        """Extract spans from model predictions"""
        spans = []
        current_span = None
        
        # Convert input_ids back to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        for i, (token, label, conf) in enumerate(zip(tokens, predicted_labels, confidences)):
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            # Convert token back to text (remove ## prefix)
            clean_token = token.replace('##', '')
            
            if label == 1 and conf > confidence_threshold:  # B tag (beginning of span)
                # Start new span
                if current_span:
                    spans.append(current_span)
                current_span = {
                    'tokens': [clean_token],
                    'start_idx': i,
                    'confidence': conf
                }
            elif label == 2 and conf > confidence_threshold and current_span:  # I tag (inside span)
                # Continue current span
                current_span['tokens'].append(clean_token)
                current_span['confidence'] = max(current_span['confidence'], conf)
            else:
                # End current span
                if current_span:
                    spans.append(current_span)
                    current_span = None
        
        # Add final span if exists
        if current_span:
            spans.append(current_span)
        
        # Convert to SpanResult objects and find text positions
        span_results = []
        for span in spans[:max_spans]:
            span_text = ''.join(span['tokens'])
            # Find position in original text
            start_pos = text.find(span_text)
            if start_pos != -1:
                span_results.append(SpanResult(
                    text=span_text,
                    start=start_pos,
                    end=start_pos + len(span_text),
                    confidence=span['confidence'],
                    label="relevant_span"
                ))
        
        return span_results
    
    def filter_criteria_matches(
        self, 
        post_text: str, 
        criteria_texts: List[str],
        confidence_threshold: float = 0.5
    ) -> List[Tuple[str, float, List[SpanResult]]]:
        """
        Filter criteria that match the post using SpanBERT
        
        Args:
            post_text: The social media post text
            criteria_texts: List of criteria texts to check
            confidence_threshold: Minimum confidence for matching
            
        Returns:
            List of tuples (criteria_text, match_score, supporting_spans)
        """
        try:
            matches = []
            
            for criteria_text in criteria_texts:
                # Combine post and criteria for span extraction
                combined_text = f"{post_text} [SEP] {criteria_text}"
                
                # Extract spans
                spans = self.extract_spans(
                    combined_text, 
                    confidence_threshold=confidence_threshold
                )
                
                # Calculate match score based on span confidence and relevance
                if spans:
                    avg_confidence = np.mean([span.confidence for span in spans])
                    # Filter spans that are likely from the post (not criteria)
                    post_spans = [span for span in spans if span.start < len(post_text)]
                    
                    if post_spans:
                        match_score = avg_confidence * len(post_spans) / len(spans)
                        matches.append((criteria_text, match_score, post_spans))
            
            # Sort by match score
            matches.sort(key=lambda x: x[1], reverse=True)
            
            return matches
            
        except Exception as e:
            logger.error(f"Error filtering criteria matches: {e}")
            return []
    
    def extract_supporting_tokens(
        self, 
        post_text: str, 
        criteria_text: str,
        confidence_threshold: float = 0.5
    ) -> List[SpanResult]:
        """
        Extract tokens that support the criteria description
        
        Args:
            post_text: The social media post text
            criteria_text: The criteria text
            confidence_threshold: Minimum confidence for token extraction
            
        Returns:
            List of supporting token spans
        """
        try:
            # Focus on the post text for supporting evidence
            spans = self.extract_spans(
                post_text, 
                confidence_threshold=confidence_threshold
            )
            
            # Filter spans that might be relevant to mental health criteria
            mental_health_keywords = [
                'depressed', 'anxiety', 'panic', 'mood', 'feeling', 'emotion',
                'sad', 'hopeless', 'worthless', 'guilt', 'shame', 'fear',
                'worry', 'stress', 'overwhelmed', 'empty', 'numb', 'angry',
                'irritable', 'agitated', 'restless', 'tired', 'fatigue',
                'sleep', 'appetite', 'weight', 'concentration', 'focus',
                'suicide', 'self-harm', 'hurt', 'pain', 'suffering'
            ]
            
            relevant_spans = []
            for span in spans:
                span_lower = span.text.lower()
                if any(keyword in span_lower for keyword in mental_health_keywords):
                    relevant_spans.append(span)
            
            return relevant_spans
            
        except Exception as e:
            logger.error(f"Error extracting supporting tokens: {e}")
            return []
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
