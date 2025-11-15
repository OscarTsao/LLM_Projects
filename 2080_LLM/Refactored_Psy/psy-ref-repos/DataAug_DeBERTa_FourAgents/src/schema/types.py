from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EvidenceUnit:
    eu_id: str
    post_id: str
    sentence_id: str
    sentence: str
    symptom: str
    assertion: str  # 'present' | 'absent'
    score: float


@dataclass
class CriteriaResult:
    post_id: str
    p_dx: float
    decision: str  # e.g., 'likely' | 'unlikely' | 'uncertain'
    supporting: Dict[str, List[str]] = field(default_factory=dict)
    conflicts: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)


@dataclass
class Suggestion:
    post_id: str
    ranked: List[Dict] = field(default_factory=list)  # [{symptom, delta_p, reason}]

