"""Dataclasses for interactions, user profiles, failure events and lessons."""

import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import re


@dataclass
class UserInteraction:
    """Represents a single user-item interaction"""
    item_id: str
    item_name: str
    item_category: str
    action_type: str  # 'purchase' for implicit feedback
    rating: Optional[float] = None
    timestamp: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class BehaviorMemory:
    """
    Represents a generalized thought about user behavior patterns
    """
    thought_id: int
    interaction_sequence: List[UserInteraction]
    behavior_explanation: str
    pattern_description: str
    # extracted_preferences: List[str]
    keywords: List[str]
    embedding: np.ndarray
    applicable_when: List[str] = field(default_factory=list)
    not_applicable_when: List[str] = field(default_factory=list)
    wrong_item_type: str = ""
    correct_item_type: str = ""
    evidence_terms_required: List[str] = field(default_factory=list)
    specificity_score: float = 0.0
    overgeneralization_risk: float = 0.0
    links: List[int] = field(default_factory=list)
    timestamp: str = None
    evolution_count: int = 0  # Số lần đã evolve
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)  # Lịch sử evolution
    max_evolutions: Optional[int] = None  # Giới hạn số lần evolve (None = unlimited)
    last_evolved_timestamp: Optional[str] = None  # Lần evolve cuối

    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    def can_evolve(self) -> bool:
        """Kiểm tra xem memory này còn được phép evolve không"""
        if self.max_evolutions is None:
            return True
        return self.evolution_count < self.max_evolutions
    def record_evolution(self, 
                        update_type: str,
                        old_values: Dict[str, Any],
                        new_values: Dict[str, Any],
                        reasoning: str) -> None:
        """Ghi lại một lần evolution"""
        self.evolution_count += 1
        self.last_evolved_timestamp = datetime.now().isoformat()
        
        self.evolution_history.append({
            'evolution_number': self.evolution_count,
            'timestamp': self.last_evolved_timestamp,
            'update_type': update_type,
            'old_values': old_values,
            'new_values': new_values,
            'reasoning': reasoning
        })
    def to_dict(self):
        data = asdict(self)
        data['embedding'] = self.embedding.tolist()
        data['interaction_sequence'] = [asdict(i) for i in self.interaction_sequence]
        return data
    
    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data['embedding'] = np.array(data['embedding'])
        data['interaction_sequence'] = [UserInteraction(**i) for i in data['interaction_sequence']]
        # Backward compatibility with memories created before structured fields.
        data.setdefault('applicable_when', [])
        data.setdefault('not_applicable_when', [])
        data.setdefault('wrong_item_type', "")
        data.setdefault('correct_item_type', "")
        data.setdefault('evidence_terms_required', [])
        data.setdefault('specificity_score', 0.0)
        data.setdefault('overgeneralization_risk', 0.0)
        return cls(**data)


@dataclass
class PairwiseUserState:
    """pairwise user state used to bootstrap fail-interaction memory generation."""
    user_id: str
    short_term_memory: str = "I enjoy discovering new items."
    long_term_memory: List[str] = field(default_factory=list)
    interaction_history: List[str] = field(default_factory=list)

    def update_memory(self, new_memory: str):
        self.long_term_memory.append(self.short_term_memory)
        self.short_term_memory = new_memory

    def add_interaction(self, item_id: str):
        self.interaction_history.append(item_id)


@dataclass
class PairwiseItemState:
    """pairwise item state with mutable textual memory."""
    item_id: str
    title: str
    category: str
    memory: str


class TraceRecorder:
    """Small JSONL trace writer for reproducible MEMCF research runs."""

    def __init__(self, trace_dir: str, enabled: bool = True):
        self.trace_dir = trace_dir
        self.enabled = enabled
        self.counts: Dict[str, int] = defaultdict(int)
        if self.enabled:
            os.makedirs(self.trace_dir, exist_ok=True)

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.counts[event_type] += 1
        row = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **make_jsonable(payload),
        }
        path = os.path.join(self.trace_dir, f"{event_type}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        events_path = os.path.join(self.trace_dir, "events.jsonl")
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_manifest(self, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = os.path.join(self.trace_dir, "manifest.json")
        data = {
            "trace_dir": self.trace_dir,
            "created_at": datetime.now().isoformat(),
            "event_counts": dict(self.counts),
            **make_jsonable(payload),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


@dataclass
class UserMemoryProfile:
    """Stable user profile initialized from observed train history."""
    user_id: str
    profile: str
    facets: List[str] = field(default_factory=list)
    evidence_item_ids: List[str] = field(default_factory=list)
    source: str = "history_init"
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "facets": self.facets[:8],
            "evidence_item_ids": self.evidence_item_ids[:10],
        }


@dataclass
class FailureEvent:
    """Full trace object for one failed pairwise interaction."""
    event_id: str
    source_user_id: str
    recent_history: List[Dict[str, Any]]
    user_memory_before: str
    user_memory_after: str
    wrong_item: Dict[str, Any]
    correct_item: Dict[str, Any]
    model_wrong_reasoning: str
    failure_type: str
    prefix_item_ids: List[str] = field(default_factory=list)
    observed_next_item_id: str = ""
    base_selected_item_id: str = ""
    creation_mode: str = "legacy_reflection"
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class FailureLesson:
    """Compact graph-retrievable memory derived from a FailureEvent."""
    memory_id: str
    source_user_id: str
    source_event_id: str
    lesson: str
    prefer: str
    avoid: str
    applies_if: List[str] = field(default_factory=list)
    do_not_apply_if: List[str] = field(default_factory=list)
    evidence_terms: List[str] = field(default_factory=list)
    wrong_item_id: str = ""
    correct_item_id: str = ""
    wrong_item_title: str = ""
    correct_item_title: str = ""
    wrong_item_category: str = ""
    correct_item_category: str = ""
    source_user_preference: str = ""
    # Saved memories created before failure taxonomy was introduced do not
    # contain this field. The default keeps those artifacts loadable.
    failure_type: str = "wrong_choice_between_positive_and_negative"
    history_item_ids: List[str] = field(default_factory=list)
    confidence: float = 0.5
    overgeneralization_risk: float = 0.5
    memory_type: str = "legacy_failure_lesson"
    observed_next_item_id: str = ""
    base_selected_item_id: str = ""
    creation_mode: str = "legacy_llm_distilled"
    factual_statement: str = ""
    # C-MEMCF keeps only aggregate provenance for a cluster-level correction.
    # Individual source users and exact item names remain outside the ranker prompt.
    cluster_id: int = -1
    cluster_support_users: int = 0
    cluster_support_events: int = 0
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def short_facet(self) -> str:
        if self.lesson:
            return self.lesson.strip()
        prefer = self.prefer.strip() or "items matching concrete history signals"
        avoid = self.avoid.strip() or "items matching only superficial signals"
        return f"User likes {prefer}, often prefers it over {avoid}, and should not be matched by generic category alone."

    def safe_fact(self) -> str:
        """Factual memory sentence for ranking prompts, avoiding extra LLM analysis."""
        if self.memory_type in {"temporal_failure_contrast", "cluster_failure_consensus"}:
            if self.factual_statement:
                return re.sub(r"\s+", " ", self.factual_statement).strip()
            correct = self.correct_item_title or self.observed_next_item_id or self.correct_item_id
            wrong = self.wrong_item_title or self.base_selected_item_id or self.wrong_item_id
            return (
                "Given the source user's preceding observed interactions, "
                f"the recorded next item was '{correct}', while the base ranker selected '{wrong}'."
            )
        pref = re.sub(r"\s+", " ", str(self.source_user_preference or self.prefer or "similar observed history")).strip()
        correct = self.correct_item_title or self.prefer or self.correct_item_id
        wrong = self.wrong_item_title or self.avoid or self.wrong_item_id
        if len(pref) > 180:
            pref = pref[:177].rstrip() + "..."
        return (
            f"A user with preference/history '{pref}' preferred/bought "
            f"'{correct}' instead of '{wrong}'."
        )


@dataclass
class GraphRetrievedLesson:
    lesson: FailureLesson
    score: float
    sources: List[str]
    paths: List[str]
    matched_evidence_terms: List[str] = field(default_factory=list)
    exposure_type: str = ""
    candidate_role: str = ""
    shared_history_items: List[str] = field(default_factory=list)
    control_mode: str = ""


# Imported last: these names are only used inside function bodies, and importing them
# at the top would create a circular import between modules.
from fmrec.common import make_jsonable
