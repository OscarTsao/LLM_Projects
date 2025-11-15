"""Multi-agent pipeline for psychiatric diagnosis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .base import AgentOutput, BaseAgent, JointTrainingConfig
from .criteria_matching import CriteriaMatchingAgent, CriteriaMatchingConfig
from .evidence_binding import EvidenceBindingAgent, EvidenceBindingConfig


@dataclass
class PipelineOutput:
    """Output from the multi-agent pipeline."""
    
    # Criteria matching results
    criteria_match: torch.Tensor
    criteria_confidence: torch.Tensor
    criteria_probabilities: torch.Tensor
    
    # Evidence binding results (only when criteria match)
    evidence_spans: Optional[List[List[Tuple[int, int]]]] = None
    evidence_confidence: Optional[torch.Tensor] = None
    evidence_text: Optional[List[List[str]]] = None
    
    # Combined results
    overall_confidence: torch.Tensor = None
    
    # Raw agent outputs
    criteria_output: Optional[AgentOutput] = None
    evidence_output: Optional[AgentOutput] = None


class MultiAgentPipeline(nn.Module):
    """Pipeline that combines criteria matching and evidence binding agents."""
    
    def __init__(
        self,
        criteria_agent: CriteriaMatchingAgent,
        evidence_agent: EvidenceBindingAgent,
        evidence_threshold: float = 0.5
    ):
        super().__init__()
        self.criteria_agent = criteria_agent
        self.evidence_agent = evidence_agent
        self.evidence_threshold = evidence_threshold
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        run_evidence: bool = True,
        **kwargs
    ) -> PipelineOutput:
        """Run the full pipeline: criteria matching → evidence binding."""
        
        # Step 1: Criteria matching
        criteria_output = self.criteria_agent(input_ids=input_ids, attention_mask=attention_mask)
        
        # Step 2: Evidence binding (only for positive criteria matches)
        evidence_output = None
        evidence_spans = None
        evidence_confidence = None
        evidence_text = None
        
        if run_evidence:
            # Run evidence binding for all samples (can filter later)
            evidence_output = self.evidence_agent(input_ids=input_ids, attention_mask=attention_mask)
            evidence_spans = evidence_output.predictions
            evidence_confidence = evidence_output.confidence
            
            # Decode spans to text
            evidence_text = self.evidence_agent.decode_spans(evidence_spans, input_ids)
            
            # Filter evidence for only positive criteria matches
            positive_mask = criteria_output.predictions.bool()
            if positive_mask.any():
                # Keep evidence only for positive matches
                filtered_spans = []
                filtered_text = []
                filtered_confidence = []
                
                for i, is_positive in enumerate(positive_mask):
                    if is_positive:
                        filtered_spans.append(evidence_spans[i])
                        filtered_text.append(evidence_text[i])
                        filtered_confidence.append(evidence_confidence[i])
                    else:
                        filtered_spans.append([])
                        filtered_text.append([])
                        filtered_confidence.append(0.0)
                
                evidence_spans = filtered_spans
                evidence_text = filtered_text
                evidence_confidence = torch.tensor(filtered_confidence, device=criteria_output.confidence.device)
        
        # Calculate overall confidence
        if evidence_confidence is not None:
            overall_confidence = (criteria_output.confidence + evidence_confidence) / 2
        else:
            overall_confidence = criteria_output.confidence
        
        return PipelineOutput(
            criteria_match=criteria_output.predictions,
            criteria_confidence=criteria_output.confidence,
            criteria_probabilities=criteria_output.probabilities,
            evidence_spans=evidence_spans,
            evidence_confidence=evidence_confidence,
            evidence_text=evidence_text,
            overall_confidence=overall_confidence,
            criteria_output=criteria_output,
            evidence_output=evidence_output
        )
    
    def predict(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        run_evidence: bool = True,
        **kwargs
    ) -> PipelineOutput:
        """Make predictions without computing gradients."""
        self.eval()
        with torch.no_grad():
            return self.forward(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                run_evidence=run_evidence,
                **kwargs
            )
    
    def predict_batch(
        self, 
        posts: List[str], 
        criteria: List[str],
        run_evidence: bool = True
    ) -> PipelineOutput:
        """Predict on a batch of post-criteria pairs."""
        inputs = self.criteria_agent.tokenize_inputs(posts, criteria)
        return self.predict(run_evidence=run_evidence, **inputs)
    
    def to(self, device):
        """Move both agents to device."""
        self.criteria_agent = self.criteria_agent.to(device)
        self.evidence_agent = self.evidence_agent.to(device)
        return super().to(device)


class JointTrainingModel(BaseAgent):
    """Joint model for training both agents simultaneously."""
    
    def __init__(self, config: JointTrainingConfig):
        super().__init__(config)
        self.config = config
        
        if config.shared_encoder:
            # Shared BERT encoder
            from transformers import AutoConfig, AutoModel
            self.bert_config = AutoConfig.from_pretrained(config.model_name)
            self.shared_encoder = AutoModel.from_pretrained(config.model_name, config=self.bert_config)
            
            # Task-specific heads
            hidden_size = self.bert_config.hidden_size
            
            # Criteria matching head
            criteria_layers = []
            in_features = hidden_size
            for hidden in config.criteria_config.classifier_hidden_sizes:
                criteria_layers.extend([
                    nn.Linear(in_features, hidden),
                    nn.GELU(),
                    nn.Dropout(config.dropout)
                ])
                in_features = hidden
            criteria_layers.append(nn.Linear(in_features, 1))
            self.criteria_head = nn.Sequential(*criteria_layers)
            
            # Evidence binding heads
            self.evidence_start_head = nn.Linear(hidden_size, 1)
            self.evidence_end_head = nn.Linear(hidden_size, 1)
            
        else:
            # Separate encoders
            self.criteria_agent = CriteriaMatchingAgent(config.criteria_config)
            self.evidence_agent = EvidenceBindingAgent(config.evidence_config)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Loss functions
        from .criteria_matching import AdaptiveFocalLoss
        self.criteria_loss_fn = AdaptiveFocalLoss(
            alpha=config.criteria_config.alpha,
            gamma=config.criteria_config.gamma,
            delta=config.criteria_config.delta
        )
        self.evidence_loss_fn = nn.BCEWithLogitsLoss()
        
        # Multi-task loss
        from .base import MultiTaskLoss
        self.multi_task_loss = MultiTaskLoss(
            criteria_weight=config.criteria_loss_weight,
            evidence_weight=config.evidence_loss_weight
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> AgentOutput:
        """Forward pass through the joint model."""
        if self.config.shared_encoder:
            # Shared encoder forward pass
            encoder_outputs = self.shared_encoder(input_ids=input_ids, attention_mask=attention_mask)
            
            # Criteria matching
            if hasattr(encoder_outputs, 'pooler_output') and encoder_outputs.pooler_output is not None:
                pooled = encoder_outputs.pooler_output
            else:
                pooled = encoder_outputs.last_hidden_state[:, 0]
            pooled = self.dropout(pooled)
            criteria_logits = self.criteria_head(pooled).squeeze(-1)
            
            # Evidence binding
            sequence_output = encoder_outputs.last_hidden_state
            sequence_output = self.dropout(sequence_output)
            start_logits = self.evidence_start_head(sequence_output).squeeze(-1)
            end_logits = self.evidence_end_head(sequence_output).squeeze(-1)
            
            # Apply attention mask to evidence logits
            start_logits = start_logits.masked_fill(~attention_mask.bool(), -1e9)
            end_logits = end_logits.masked_fill(~attention_mask.bool(), -1e9)
            
        else:
            # Separate encoders
            criteria_output = self.criteria_agent(input_ids=input_ids, attention_mask=attention_mask)
            evidence_output = self.evidence_agent(input_ids=input_ids, attention_mask=attention_mask)
            
            criteria_logits = criteria_output.logits
            start_logits = evidence_output.logits["start"]
            end_logits = evidence_output.logits["end"]
        
        # Combine outputs
        criteria_probs = torch.sigmoid(criteria_logits)
        start_probs = torch.sigmoid(start_logits)
        end_probs = torch.sigmoid(end_logits)
        
        return AgentOutput(
            predictions={
                "criteria": (criteria_probs > 0.5).float(),
                "start": start_probs,
                "end": end_probs
            },
            confidence=criteria_probs,  # Use criteria confidence as overall
            logits={
                "criteria": criteria_logits,
                "start": start_logits,
                "end": end_logits
            },
            probabilities={
                "criteria": criteria_probs,
                "start": start_probs,
                "end": end_probs
            },
            metadata={"agent_type": "joint_training"}
        )
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> AgentOutput:
        """Make predictions without computing gradients."""
        self.eval()
        with torch.no_grad():
            return self.forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    def get_loss(self, outputs: AgentOutput, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute multi-task loss for training."""
        # Criteria loss
        criteria_loss = self.criteria_loss_fn(outputs.logits["criteria"], targets["criteria"].float())
        
        # Evidence loss
        start_loss = self.evidence_loss_fn(outputs.logits["start"], targets["start_positions"].float())
        end_loss = self.evidence_loss_fn(outputs.logits["end"], targets["end_positions"].float())
        evidence_loss = (start_loss + end_loss) / 2
        
        # Combined loss
        return self.multi_task_loss(criteria_loss, evidence_loss)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if self.config.shared_encoder:
            if hasattr(self.shared_encoder, 'gradient_checkpointing_enable'):
                self.shared_encoder.gradient_checkpointing_enable()
        else:
            self.criteria_agent.enable_gradient_checkpointing()
            self.evidence_agent.enable_gradient_checkpointing()


def create_multi_agent_pipeline(
    criteria_config: Optional[CriteriaMatchingConfig] = None,
    evidence_config: Optional[EvidenceBindingConfig] = None,
    evidence_threshold: float = 0.5
) -> MultiAgentPipeline:
    """Factory function to create a multi-agent pipeline."""
    if criteria_config is None:
        criteria_config = CriteriaMatchingConfig()
    if evidence_config is None:
        evidence_config = EvidenceBindingConfig()
    
    criteria_agent = CriteriaMatchingAgent(criteria_config)
    evidence_agent = EvidenceBindingAgent(evidence_config)
    
    return MultiAgentPipeline(criteria_agent, evidence_agent, evidence_threshold)
