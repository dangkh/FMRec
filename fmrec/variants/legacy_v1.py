"""Legacy v1 pipeline (main/evaluate_user/train_memory_from_fail_interactions). Not used by main_v2."""

import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import asdict
from collections import defaultdict
from tqdm import tqdm
import random
import re
import argparse

from fmrec.common import (
    GENERIC_MEMORY_PHRASES,
    GENERIC_MEMORY_TERMS,
    _safe_float,
    deterministic_shuffle,
    item_category,
    item_title,
    normalize_evidence_terms,
    normalize_terms,
    term_matches_context,
)
from fmrec.data_metrics import (
    calculate_ndcg_at_k,
    calculate_recall_at_k,
    load_data,
    save_all_users_ranking_results,
)
from fmrec.llm_client import RecommendationMemorySystem
from fmrec.records import (
    BehaviorMemory,
    PairwiseItemState,
    PairwiseUserState,
    TraceRecorder,
    UserInteraction,
)
from fmrec.variants.legacy_memory_system import behavior_memory_to_trace


def clean_ranked_item_ids(ranked_ids: List[Any], candidate_items: List[Dict[str, Any]]) -> List[str]:
    """Return a valid candidate permutation: no duplicates, no hallucinated IDs, all candidates included."""
    all_candidate_ids = [str(c["item_id"]) for c in candidate_items]
    candidate_set = set(all_candidate_ids)
    cleaned: List[str] = []
    seen: Set[str] = set()

    if not isinstance(ranked_ids, list):
        ranked_ids = []

    for item_id in ranked_ids:
        item_id = str(item_id)
        if item_id in candidate_set and item_id not in seen:
            cleaned.append(item_id)
            seen.add(item_id)

    cleaned.extend(cid for cid in all_candidate_ids if cid not in seen)
    return cleaned


def memory_text_is_too_generic(text: str, min_terms: int = 4) -> bool:
    return len(set(normalize_terms(text))) < min_terms


def ranking_validation(raw_ranked_ids: List[Any], candidate_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidate_ids = [str(c["item_id"]) for c in candidate_items]
    candidate_set = set(candidate_ids)
    raw_ids = [str(x) for x in raw_ranked_ids] if isinstance(raw_ranked_ids, list) else []
    seen: Set[str] = set()
    duplicate_ids: List[str] = []
    for item_id in raw_ids:
        if item_id in seen and item_id not in duplicate_ids:
            duplicate_ids.append(item_id)
        seen.add(item_id)
    hallucinated_ids = [item_id for item_id in raw_ids if item_id not in candidate_set]
    missing_ids = [item_id for item_id in candidate_ids if item_id not in set(raw_ids)]
    return {
        "is_valid": (
            len(raw_ids) == len(candidate_ids)
            and len(duplicate_ids) == 0
            and len(hallucinated_ids) == 0
            and len(missing_ids) == 0
        ),
        "raw_length": len(raw_ids),
        "expected_length": len(candidate_ids),
        "duplicate_ids": duplicate_ids,
        "hallucinated_ids": hallucinated_ids,
        "missing_ids": missing_ids,
    }


def memory_text(memory: "BehaviorMemory") -> str:
    return " ".join(
        [
            str(memory.behavior_explanation),
            str(memory.pattern_description),
            " ".join(str(k) for k in memory.keywords),
            " ".join(str(k) for k in getattr(memory, "applicable_when", [])),
            " ".join(str(k) for k in getattr(memory, "not_applicable_when", [])),
            str(getattr(memory, "wrong_item_type", "")),
            str(getattr(memory, "correct_item_type", "")),
            " ".join(str(k) for k in getattr(memory, "evidence_terms_required", [])),
        ]
    )


def build_retrieval_query(
    user_profile_text: str,
    candidate_items_info: List[Dict[str, Any]],
    mode: str,
) -> str:
    if mode == "user_only":
        return user_profile_text
    if mode != "candidate_aware":
        raise ValueError(f"Unsupported --memory_retrieval_mode={mode}")
    candidate_text = " ".join(
        f"{item.get('title', '')} {item.get('category', '')}" for item in candidate_items_info
    )
    return f"User recent history: {user_profile_text}\nCandidate items: {candidate_text}"


def gate_memory_records(
    memory_records: List[Dict[str, Any]],
    user_profile_text: str,
    candidate_items_info: List[Dict[str, Any]],
    gate_mode: str,
    similarity_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if gate_mode == "none":
        decisions = []
        for record in memory_records:
            decisions.append({
                "memory_id": record["memory"].thought_id,
                "decision": "keep",
                "reason": "memory_gate=none",
                "similarity": float(record["similarity"]),
                "matched_terms": [],
            })
        return memory_records, decisions
    if gate_mode not in {"rule", "strict_rule", "applicability"}:
        raise ValueError(f"Unsupported --memory_gate={gate_mode}")

    candidate_text = " ".join(
        f"{item.get('title', '')} {item.get('category', '')}" for item in candidate_items_info
    ).lower()
    history_text = str(user_profile_text).lower()
    combined_context = f"{history_text} {candidate_text}"
    candidate_categories = {
        str(item.get("category", "Unknown")).strip().lower()
        for item in candidate_items_info
        if item.get("category")
    }

    kept: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    for record in memory_records:
        memory = record["memory"]
        similarity = float(record["similarity"])
        text = memory_text(memory).lower()
        structured_terms = normalize_evidence_terms(
            list(getattr(memory, "evidence_terms_required", []) or [])
            + list(getattr(memory, "applicable_when", []) or [])
            + list(getattr(memory, "keywords", []) or [])
            + [getattr(memory, "wrong_item_type", ""), getattr(memory, "correct_item_type", "")]
        )
        text_terms = normalize_terms(text)
        candidate_terms = [
            term for term in (structured_terms + text_terms)
            if term not in GENERIC_MEMORY_TERMS
            and term not in GENERIC_MEMORY_PHRASES
            and len(term) >= 3
        ]
        seen_terms: Set[str] = set()
        candidate_terms = [t for t in candidate_terms if not (t in seen_terms or seen_terms.add(t))]
        matched_terms = [term for term in candidate_terms[:60] if term_matches_context(term, combined_context)]
        strong_matched_terms = [
            term for term in matched_terms
            if term not in GENERIC_MEMORY_TERMS
            and term not in GENERIC_MEMORY_PHRASES
            and len(term) >= 4
        ]
        not_applicable_terms = normalize_evidence_terms(getattr(memory, "not_applicable_when", []))
        matched_not_applicable_terms = [
            term for term in not_applicable_terms
            if term_matches_context(term, combined_context)
        ]

        category_mismatch_memory = (
            "category" in text
            and any(signal in text for signal in ["mismatch", "unrelated", "non-", "instead of"])
        )
        all_candidates_same_category = len(candidate_categories) <= 1

        decision = "keep"
        reason = "passed rule gate"
        min_strong_terms = int(os.getenv("MEMCF_STRICT_GATE_MIN_STRONG_TERMS", "2"))
        min_app_terms = int(os.getenv("MEMCF_APP_GATE_MIN_TERMS", "1"))
        generic_memory = (
            len(strong_matched_terms) == 0
            and sum(1 for term in ["mismatch", "category", "intent", "preference"] if term in text) >= 2
        )
        applicability_score = (
            len(strong_matched_terms)
            + 0.5 * len([t for t in matched_terms if t not in strong_matched_terms])
            + max(0.0, similarity - similarity_threshold)
        )
        if similarity < similarity_threshold:
            decision = "skip"
            reason = f"similarity {similarity:.4f} below threshold {similarity_threshold:.4f}"
        elif matched_not_applicable_terms:
            decision = "skip"
            reason = f"not_applicable_when matched current context: {matched_not_applicable_terms[:5]}"
        elif category_mismatch_memory and all_candidates_same_category and not matched_terms:
            decision = "skip"
            reason = "category-mismatch memory is not discriminative because candidate categories are identical"
        elif not matched_terms and similarity < similarity_threshold + 0.10:
            decision = "skip"
            reason = "no specific memory terms matched current user/candidates"
        elif gate_mode in {"strict_rule", "applicability"} and len(strong_matched_terms) < min_strong_terms:
            decision = "skip"
            reason = (
                f"{gate_mode} requires at least {min_strong_terms} strong matched terms; "
                f"found {len(strong_matched_terms)}"
            )
        elif gate_mode in {"strict_rule", "applicability"} and generic_memory:
            decision = "skip"
            reason = f"{gate_mode} rejected generic category/intent memory with no strong current evidence"
        elif gate_mode in {"strict_rule", "applicability"} and category_mismatch_memory and all_candidates_same_category:
            decision = "skip"
            reason = f"{gate_mode} rejected category-mismatch memory when all candidates have same category"
        elif gate_mode == "applicability" and applicability_score < min_app_terms:
            decision = "skip"
            reason = (
                f"applicability score {applicability_score:.2f} below required {min_app_terms}; "
                "memory lacks concrete evidence in current context"
            )

        item = {
            "memory_id": memory.thought_id,
            "decision": decision,
            "reason": reason,
            "similarity": similarity,
            "applicability_score": applicability_score,
            "matched_terms": matched_terms,
            "strong_matched_terms": strong_matched_terms,
            "matched_not_applicable_terms": matched_not_applicable_terms,
            "candidate_terms_checked": candidate_terms[:40],
            "gate_mode": gate_mode,
            "memory": behavior_memory_to_trace(memory),
        }
        decisions.append(item)
        if decision == "keep":
            kept.append(record)

    return kept, decisions


def get_or_create_user_state(user_states: Dict[str, PairwiseUserState], user_id: str) -> PairwiseUserState:
    if user_id not in user_states:
        user_states[user_id] = PairwiseUserState(user_id=user_id)
    return user_states[user_id]


def _extract_json_from_llm_output(raw_output: str) -> Dict[str, Any]:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).rstrip("```").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM output")
    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Local OpenAI-compatible models sometimes emit invalid backslash escapes.
        json_str = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", json_str)
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
        return json.loads(json_str)


def corrective_pairwise_reflection(
    memory_system: RecommendationMemorySystem,
    user_state: PairwiseUserState,
    pos_item: PairwiseItemState,
    neg_item: PairwiseItemState,
    chosen_item_id: str,
    explanation: str,
) -> None:
    """pairwise reflection update for user memory and item memories."""
    if chosen_item_id == pos_item.item_id:
        return

    user_prompt = f"""You are an enthusiast with these preferences: "{user_state.short_term_memory}"

Recently, you chose between two items:
1. Title: {neg_item.title}, Description: {neg_item.memory}
2. Title: {pos_item.title}, Description: {pos_item.memory}

You selected item 1, but you discovered you actually prefer item 2 instead.
Your previous explanation was: "{explanation}"

This indicates an incorrect choice, and your previous judgment about your preferences was mistaken. Your task now is to update your self-introduction with your new preferences and dislikes. \n Follow these steps: \n 1. Analyze misconceptions in your previous judgment and correct them.\n 2. Identify new preferences from '{pos_item.title}' and dislikes from '{neg_item.title}'. \n 3. Summarize your past preferences, merging them with new insights and removing conflicting parts.\n 4. Update your self-introduction, starting with new preferences, then summarizing past ones, followed by dislikes. \n\n Important notes: 1. Keep it under 150 words.  \n 2. Be concise and clear. \n 3. Describe only the features of items you prefer or dislike, without mentioning your thought process. \n 4. Your self-introduction should be specific and personalized; avoid generic preferences."

Output format:
My updated self-introduction: [Your updated preferences in under 150 words]

Important: Focus on what features you like and dislike, be specific and personalized."""

    new_user_memory = memory_system.qwen_generate(prompt=user_prompt, call_type="reflection_user")
    memory_system._trace("reflection_user_memory_llm", {
        "user_id": user_state.user_id,
        "prompt": user_prompt,
        "role_prompt": "You are a helpful AI assistant.",
        "answer": new_user_memory,
        "old_user_memory": user_state.short_term_memory,
        "positive_item": asdict(pos_item),
        "negative_item": asdict(neg_item),
        "chosen_item_id": chosen_item_id,
    })
    if "My updated self-introduction:" in new_user_memory:
        new_user_memory = new_user_memory.split("My updated self-introduction:")[1].strip()
    old_user_memory = user_state.short_term_memory
    user_state.update_memory(new_user_memory)
    memory_system._trace("reflection_user_memory_updated", {
        "user_id": user_state.user_id,
        "old_user_memory": old_user_memory,
        "new_user_memory": user_state.short_term_memory,
        "positive_item_id": pos_item.item_id,
        "negative_item_id": neg_item.item_id,
    })

    item_prompt = f"""A user with these preferences browsed items: "{user_state.short_term_memory}"

The user considered two items:
1. Title: {pos_item.title}, Description: {pos_item.memory}
2. Title: {neg_item.title}, Description: {neg_item.memory}

The user initially chose item 2 but actually prefers item 1, indicating the descriptions may be misleading.

Your task is to update the descriptions of these items based on these insights. \n Follow these steps:\n 1. Analyze the user's preferences and dislikes from the self-description. \n 2. Explore the chosen item's features that align with preferences and oppose dislikes, and examine the rejected item's features that align with dislikes and oppose preferences. Highlight the differences thoroughly. \n 3. Incorporate new features into the previous descriptions, preserving key information while being concise.\n\n Important notes: \n 1. Your output should be in the following format: 'The updated description of the first item is: [updated description]. \\n The updated description of the second item is: [updated description].'. \n 2. Each updated description cannot exceed 50 words; be concise and clear! \n 3. In your updated descriptions, refer to preferences collectively, avoiding individual references. For example, say 'the user with ... preferences/dislikes'.\n 4. New features should reflect user preferences, and the updated descriptions must not contradict the inherent characteristics of the items, e.g., do not describe a thriller as having a predictably happy ending.

Update the description of item 1 to better reflect why users with these preferences would like it.

Output format (STRICT JSON, no extra text):
{{
  "item_1": "<updated description, single paragraph>",
  "item_2": "<updated description, single paragraph>"
}}

Important: Make it specific and aligned with user preferences."""

    new_item_memory = memory_system.qwen_generate(prompt=item_prompt, call_type="reflection_item")
    memory_system._trace("reflection_item_memory_llm", {
        "user_id": user_state.user_id,
        "prompt": item_prompt,
        "role_prompt": "You are a helpful AI assistant.",
        "answer": new_item_memory,
        "old_positive_item_memory": pos_item.memory,
        "old_negative_item_memory": neg_item.memory,
        "positive_item": asdict(pos_item),
        "negative_item": asdict(neg_item),
    })

    # Item reflection should improve local item descriptions, but it should not
    # block fail-memory creation when the local LLM emits malformed JSON.
    try:
        data = _extract_json_from_llm_output(new_item_memory)
        item1_desc = data["item_1"].strip()
        item2_desc = data["item_2"].strip()
    except Exception as e:
        print(f"  ⚠ Item reflection JSON parse failed; keeping previous item memories: {e}")
        memory_system._trace("reflection_error", {
            "user_id": user_state.user_id,
            "error": str(e),
            "positive_item_id": pos_item.item_id,
            "negative_item_id": neg_item.item_id,
            "raw_answer": new_item_memory,
        })
        item1_desc = pos_item.memory
        item2_desc = neg_item.memory

    # 5. Update memories
    old_positive_item_memory = pos_item.memory
    old_negative_item_memory = neg_item.memory
    pos_item.memory = item1_desc
    neg_item.memory = item2_desc
    memory_system._trace("reflection_item_memory_updated", {
        "user_id": user_state.user_id,
        "positive_item_id": pos_item.item_id,
        "negative_item_id": neg_item.item_id,
        "old_positive_item_memory": old_positive_item_memory,
        "old_negative_item_memory": old_negative_item_memory,
        "new_positive_item_memory": pos_item.memory,
        "new_negative_item_memory": neg_item.memory,
    })


def train_memory_from_fail_interactions(
    user_id: str,
    user_data: Dict,
    memory_system: RecommendationMemorySystem,
    user_states: Dict[str, PairwiseUserState],
    item_states: Dict[str, PairwiseItemState],
    items_meta: Dict[str, Dict[str, Any]],
    negative_data: Optional[Dict[str, Any]] = None,
    max_iterations: int = 1,
    max_positive_interactions: Optional[int] = None,
    candidate_negative_mode: str = "random",
    min_lesson_confidence: float = 0.25,
    max_lesson_risk: float = 0.85,
    max_failure_lessons_per_user: int = 3,
    training_negative_source: str = "legacy_runtime",
) -> List[BehaviorMemory]:
    """
    Hybrid training:
    - pairwise initialization and interaction loop.
    - Create behavior memories ONLY from failed interactions.
    """
    train_items = user_data["train"]
    if max_positive_interactions and max_positive_interactions > 0:
        train_items = train_items[-max_positive_interactions:]
    else:
        train_items = train_items[-30:]
    if len(train_items) == 0:
        return []

    user_state = get_or_create_user_state(user_states, user_id)
    all_item_ids = list(item_states.keys())
    if not all_item_ids:
        return []

    # temp memory system must be local for this user
    memory_system.user_interaction_history = []
    memory_system.behavior_memories = []
    memory_system.next_thought_id = 0

    new_memories: List[BehaviorMemory] = []
    for pos_item_id in train_items:
        if pos_item_id not in item_states:
            continue

        neg_item_id = choose_training_negative_item_id(
            user_id=str(user_id),
            pos_item_id=str(pos_item_id),
            user_data=user_data,
            negative_data=negative_data,
            items_meta=items_meta,
            all_item_ids=all_item_ids,
            mode=candidate_negative_mode,
            max_positive_interactions=max_positive_interactions,
            training_negative_source=training_negative_source,
        )
        if not neg_item_id:
            continue

        pos_item = item_states[pos_item_id]
        neg_item = item_states[neg_item_id]

        for _ in range(max_iterations):
            chosen_item_id, explanation = autonomous_pairwise_interaction(
                memory_system=memory_system,
                user_state=user_state,
                pos_item=pos_item,
                neg_item=neg_item,
            )

            if chosen_item_id == pos_item_id:
                user_state.add_interaction(pos_item_id)
                break

            try:
                corrective_pairwise_reflection(
                    memory_system=memory_system,
                    user_state=user_state,
                    pos_item=pos_item,
                    neg_item=neg_item,
                    chosen_item_id=chosen_item_id,
                    explanation=explanation,
                )
            except Exception as e:
                print(f"  ⚠ Reflection failed for user {user_id}, item {pos_item_id}: {e}")
                continue

            # Memory unit is one failed interaction pair instead of sliding windows.
            fail_window = [
                UserInteraction(
                    item_id=neg_item.item_id,
                    item_name=neg_item.title,
                    item_category=neg_item.category,
                    action_type="wrong_choice",
                    metadata={"user_id": user_id, "role": "chosen_wrong"},
                ),
                UserInteraction(
                    item_id=pos_item.item_id,
                    item_name=pos_item.title,
                    item_category=pos_item.category,
                    action_type="preferred_item",
                    metadata={"user_id": user_id, "role": "ground_truth"},
                ),
            ]
            try:
                fail_memory = memory_system.create_behavior_thought(fail_window)
                passed_gate, gate_reason = behavior_memory_passes_quality_gate(
                    fail_memory,
                    fail_window,
                    min_confidence=min_lesson_confidence,
                    max_risk=max_lesson_risk,
                )
                memory_system._trace("memory_quality_gate", {
                    "user_id": user_id,
                    "positive_item_id": pos_item.item_id,
                    "negative_item_id": neg_item.item_id,
                    "passed": passed_gate,
                    "reason": gate_reason,
                    "min_lesson_confidence": min_lesson_confidence,
                    "max_lesson_risk": max_lesson_risk,
                    "memory": behavior_memory_to_trace(fail_memory),
                })
                if not passed_gate:
                    continue
                memory_system._trace("fail_memory_from_wrong_choice", {
                    "user_id": user_id,
                    "positive_item_id": pos_item.item_id,
                    "negative_item_id": neg_item.item_id,
                    "chosen_wrong_item_id": neg_item.item_id,
                    "preferred_item_id": pos_item.item_id,
                    "memory": behavior_memory_to_trace(fail_memory),
                    "choice_explanation": explanation,
                })
                new_memories.append(fail_memory)
                if max_failure_lessons_per_user > 0 and len(new_memories) >= max_failure_lessons_per_user:
                    memory_system._trace("memory_generation_limit_reached", {
                        "user_id": user_id,
                        "max_failure_lessons_per_user": max_failure_lessons_per_user,
                        "current_count": len(new_memories),
                    })
                    return new_memories
            except Exception as e:
                print(f"  ⚠ Fail-memory creation error for user {user_id}: {e}")
                memory_system._trace("fail_memory_error", {
                    "user_id": user_id,
                    "positive_item_id": pos_item.item_id,
                    "negative_item_id": neg_item.item_id,
                    "error": str(e),
                })

    return new_memories


def evaluate_user(user_data: Dict,
                 negative_data: Dict,
                 items_meta: Dict,
                 memory_system: RecommendationMemorySystem,
                 eval_type: str = 'test', use_memory = True, k_memories: int = 5, sample_user_list: List = None, negative_data_sample_list: List = None,
                 max_positive_interactions: Optional[int] = None, max_negative_candidates: Optional[int] = None,
                 user_id: Optional[str] = None,
                 memory_retrieval_mode: str = "user_only",
                 memory_gate: str = "none",
                 memory_similarity_threshold: float = 0.35,
                 no_harm_arbitration: bool = False,
                 no_harm_min_applicability: float = 1.0,
                 ranking_prompt_style: str = "memcf") -> Dict[str, float]:
    """Evaluate for a single user with LLM-based ranking"""
    
    # Get ground truth and candidates
    if eval_type == 'val':
        ground_truth = user_data['val']
        negatives = negative_data.get('val_neg', [])
    else:  # test
        ground_truth = user_data['test']
        negatives = negative_data.get('test_neg', [])
    if max_negative_candidates and max_negative_candidates > 0:
        negatives = negatives[:max_negative_candidates]

    if sample_user_list is not None:
        ground_truth_sample_fewshot = []
        negatives_sample_fewshot = []
        for i in range(len(sample_user_list)):
            sample_user_data = sample_user_list[i]
            negative_data_sample = negative_data_sample_list[i]

            ground_truth_sample = sample_user_data.get('val', [])
            ground_truth_sample_fewshot.append(ground_truth_sample)

            negatives_sample = negative_data_sample.get('val_neg', [])
            if max_negative_candidates and max_negative_candidates > 0:
                negatives_sample = negatives_sample[:max_negative_candidates]
            negatives_sample_fewshot.append(negatives_sample)

    # Prepare train items for user profile
    train_items_info = []
    user_profile_texts = []
    train_history_for_profile = user_data['train']
    if max_positive_interactions and max_positive_interactions > 0:
        train_history_for_profile = train_history_for_profile[-max_positive_interactions:]
    else:
        train_history_for_profile = train_history_for_profile[-10:]
    for item_id in train_history_for_profile:
        if item_id in items_meta:
            item_info = items_meta[item_id]
            title = item_title(item_info, str(item_id))
            category = item_category(item_info)
            
            train_items_info.append({
                'item_id': item_id,
                'title': title,
                'category': category
            })
            user_profile_texts.append(f"{title} {category}")
    
    # Create user profile text for retrieval
    user_profile_text = " ".join(user_profile_texts)

    # create sample for fewshot ranking
    prompt_sample = ''
    if sample_user_list is not None:
        prompt_sample = 'Learn from the following examples:\n'
        for i in range(len(sample_user_list)):
            sample_user_data = sample_user_list[i]
            sample_train_items_info = []
            sample_history = sample_user_data['train']
            if max_positive_interactions and max_positive_interactions > 0:
                sample_history = sample_history[-max_positive_interactions:]
            else:
                sample_history = sample_history[-10:]
            for item_id in sample_history:
                if item_id in items_meta:
                    item_info = items_meta[item_id]
                    title = item_title(item_info, str(item_id))
                    category = item_category(item_info)
                    
                    sample_train_items_info.append({
                        'item_id': item_id,
                        'title': title,
                        'category': category
                    })
            sample_user_profile_texts = []
            for item in sample_train_items_info:
                sample_user_profile_texts.append(f"{item['title']} {item['category']}")
            sample_user_profile_text = " ".join(sample_user_profile_texts)

            candidates_sample = deterministic_shuffle(
                ground_truth_sample_fewshot[i] + negatives_sample_fewshot[i],
                salt=f"fewshot_{i}",
            )
            candidate_items_info_sample = []
            for item_id in candidates_sample:
                if item_id in items_meta:
                    item_info = items_meta[item_id]
                    candidate_items_info_sample.append({
                        'item_id': item_id,
                        'title': item_title(item_info, str(item_id)),
                        'category': item_category(item_info)
                    })
                else:
                    candidate_items_info_sample.append({
                        'item_id': item_id,
                        'title': f'Item {item_id}',
                        'category': 'Unknown'
                    })
            prompt_sample += f"""
            Example {i+1}:
            Other user Recent History: {sample_user_profile_text}
            Candidate Items: {json.dumps(candidate_items_info_sample, indent=2)}
            You should set the true items "{json.dumps(ground_truth_sample_fewshot[i], indent=2)}" at the top of the ranking.\n
            """
        # user_profile_text += " " + sample_user_profile_text
    
    # Combine ground truth and negatives as candidates
    candidates = deterministic_shuffle(ground_truth + negatives, salt=f"{eval_type}_candidates")
    candidate_items_info = []
    for item_id in candidates:
        if item_id in items_meta:
            item_info = items_meta[item_id]
            candidate_items_info.append({
                'item_id': item_id,
                'title': item_title(item_info, str(item_id)),
                'category': item_category(item_info)
            })
        else:
            candidate_items_info.append({
                'item_id': item_id,
                'title': f'Item {item_id}',
                'category': 'Unknown'
            })
    
    retrieval_query_text = None
    retrieved_memory_records: List[Dict[str, Any]] = []
    gate_decisions: List[Dict[str, Any]] = []
    if use_memory:
        retrieval_query_text = build_retrieval_query(
            user_profile_text=user_profile_text,
            candidate_items_info=candidate_items_info,
            mode=memory_retrieval_mode,
        )
        retrieved_memory_records = memory_system.retrieve_relevant_memory_records(retrieval_query_text, k=k_memories)
        kept_memory_records, gate_decisions = gate_memory_records(
            memory_records=retrieved_memory_records,
            user_profile_text=user_profile_text,
            candidate_items_info=candidate_items_info,
            gate_mode=memory_gate,
            similarity_threshold=memory_similarity_threshold,
        )
        for decision in gate_decisions:
            memory_system._trace("memory_gate_decision", {
                "user_id": user_id,
                "eval_type": eval_type,
                "retrieval_mode": memory_retrieval_mode,
                "memory_gate": memory_gate,
                **decision,
            })
        memory_system.record_memory_diagnostics(
            retrieved=len(retrieved_memory_records),
            kept=len(kept_memory_records),
            skipped=len(retrieved_memory_records) - len(kept_memory_records),
        )
        retrieved_memories = [record["memory"] for record in kept_memory_records]
        if len(retrieved_memories) == 0:
            retrieved_memories = None
    else:
        retrieved_memories = None
    no_memory_predictions = None
    memory_predictions = None
    selected_ranking_source = "memory" if retrieved_memories else "no_memory"
    arbitration_decision: Dict[str, Any] = {}

    if use_memory and no_harm_arbitration:
        memory_system.memory_diagnostics["no_harm_users"] += 1
        no_memory_predictions = memory_system.llm_ranking(
            train_items_info,
            candidate_items_info,
            None,
            prompt_sample,
            ranking_prompt_style=ranking_prompt_style,
            trace_context={
                "user_id": user_id,
                "eval_type": eval_type,
                "use_memory": False,
                "ranking_path": "no_harm_no_memory_candidate",
                "memory_retrieval_mode": memory_retrieval_mode,
                "memory_gate": memory_gate,
                "memory_similarity_threshold": memory_similarity_threshold,
                "retrieval_query_text": retrieval_query_text,
                "ground_truth": ground_truth,
                "fixed_candidates": candidates,
            },
        )
        if retrieved_memories:
            memory_predictions = memory_system.llm_ranking(
                train_items_info,
                candidate_items_info,
                retrieved_memories,
                prompt_sample,
                ranking_prompt_style=ranking_prompt_style,
                trace_context={
                    "user_id": user_id,
                    "eval_type": eval_type,
                    "use_memory": True,
                    "ranking_path": "no_harm_memory_candidate",
                    "memory_retrieval_mode": memory_retrieval_mode,
                    "memory_gate": memory_gate,
                    "memory_similarity_threshold": memory_similarity_threshold,
                    "retrieval_query_text": retrieval_query_text,
                    "ground_truth": ground_truth,
                    "fixed_candidates": candidates,
                },
            )
            kept_decisions = [d for d in gate_decisions if d.get("decision") == "keep"]
            max_applicability = max(
                [float(d.get("applicability_score", 0.0)) for d in kept_decisions] or [0.0]
            )
            max_strong_terms = max(
                [len(d.get("strong_matched_terms", [])) for d in kept_decisions] or [0]
            )
            use_memory_ranking = (
                max_applicability >= no_harm_min_applicability
                and max_strong_terms > 0
            )
            if use_memory_ranking:
                predictions = memory_predictions
                selected_ranking_source = "memory"
                memory_system.memory_diagnostics["no_harm_used_memory"] += 1
            else:
                predictions = no_memory_predictions
                selected_ranking_source = "no_memory"
                memory_system.memory_diagnostics["no_harm_fallback_no_memory"] += 1
            arbitration_decision = {
                "enabled": True,
                "selected_ranking_source": selected_ranking_source,
                "max_applicability": max_applicability,
                "max_strong_terms": max_strong_terms,
                "no_harm_min_applicability": no_harm_min_applicability,
                "reason": (
                    "memory passed no-harm evidence threshold"
                    if use_memory_ranking
                    else "fallback to no-memory: insufficient memory applicability evidence"
                ),
            }
        else:
            predictions = no_memory_predictions
            selected_ranking_source = "no_memory"
            memory_system.memory_diagnostics["no_harm_fallback_no_memory"] += 1
            arbitration_decision = {
                "enabled": True,
                "selected_ranking_source": selected_ranking_source,
                "reason": "fallback to no-memory: no kept retrieved memories",
            }
        memory_system._trace("no_harm_arbitration", {
            "user_id": user_id,
            "eval_type": eval_type,
            "decision": arbitration_decision,
            "gate_decisions": gate_decisions,
            "no_memory_predictions": no_memory_predictions,
            "memory_predictions": memory_predictions,
            "selected_predictions": predictions,
        })
    else:
        # Use LLM to rank candidates
        predictions = memory_system.llm_ranking(
            train_items_info,
            candidate_items_info,
            retrieved_memories,
            prompt_sample,
            ranking_prompt_style=ranking_prompt_style,
            trace_context={
                "user_id": user_id,
                "eval_type": eval_type,
                "use_memory": use_memory,
                "ranking_path": "single_path",
                "memory_retrieval_mode": memory_retrieval_mode,
                "memory_gate": memory_gate,
                "memory_similarity_threshold": memory_similarity_threshold,
                "retrieval_query_text": retrieval_query_text,
                "ground_truth": ground_truth,
                "fixed_candidates": candidates,
            },
        )
    baseline_metric = {
        'recall@5': calculate_recall_at_k(candidates, ground_truth, 5),
        'recall@10': calculate_recall_at_k(candidates, ground_truth, 10),
        'recall@20': calculate_recall_at_k(candidates, ground_truth, 20),
        'ndcg@5': calculate_ndcg_at_k(candidates, ground_truth, 5),
        'ndcg@10': calculate_ndcg_at_k(candidates, ground_truth, 10),
        'ndcg@20': calculate_ndcg_at_k(candidates, ground_truth, 20),
    }
    # Calculate metrics
    metrics = {
        'recall@5': calculate_recall_at_k(predictions, ground_truth, 5),
        'recall@10': calculate_recall_at_k(predictions, ground_truth, 10),
        'recall@20': calculate_recall_at_k(predictions, ground_truth, 20),
        'ndcg@5': calculate_ndcg_at_k(predictions, ground_truth, 5),
        'ndcg@10': calculate_ndcg_at_k(predictions, ground_truth, 10),
        'ndcg@20': calculate_ndcg_at_k(predictions, ground_truth, 20),
    }
    memory_system._trace("ranking_result", {
        "user_id": user_id,
        "eval_type": eval_type,
        "use_memory": use_memory,
        "ground_truth": ground_truth,
        "candidate_item_ids": candidates,
        "ranked_item_ids": predictions,
        "metrics": metrics,
        "baseline_metrics": baseline_metric,
        "memory_retrieval_mode": memory_retrieval_mode,
        "memory_gate": memory_gate,
        "memory_similarity_threshold": memory_similarity_threshold,
        "ranking_prompt_style": ranking_prompt_style,
        "no_harm_arbitration": arbitration_decision,
        "selected_ranking_source": selected_ranking_source,
        "retrieval_query_text": retrieval_query_text,
        "gate_decisions": gate_decisions,
        "retrieved_memories": [
            behavior_memory_to_trace(mem) for mem in (retrieved_memories or [])
        ],
    })
    
    return baseline_metric,metrics, candidates, predictions, ground_truth


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment configuration")

    # Basic config
    parser.add_argument("--data_name", type=str, default="Video_Game")

    parser.add_argument("--use_memory", action="store_true", default=True)
    parser.add_argument("--no_use_memory", action="store_false", dest="use_memory")

    parser.add_argument("--LOAD_SAVED_MEMORY", action="store_true", default=False)

    # Hyperparameters for training
    parser.add_argument("--wo_evolving", action="store_true", default=True)
    parser.add_argument("--with_evolving", action="store_false", dest="wo_evolving")
    parser.add_argument("--wo_link", action="store_true", default=False)

    parser.add_argument("--max_evolutions_per_memory", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--link_size", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=1)

    # Hyperparameter for ranking
    parser.add_argument("--k_memories", type=int, default=1)
    parser.add_argument("--memory_retrieval_mode", type=str, default="user_only",
                        choices=["user_only", "candidate_aware"],
                        help="Memory retrieval query: user history only or user history plus candidate set.")
    parser.add_argument("--memory_gate", type=str, default="none",
                        choices=["none", "rule", "strict_rule", "applicability"],
                        help="Whether to filter retrieved memories before ranking.")
    parser.add_argument("--memory_similarity_threshold", type=float, default=0.35,
                        help="Minimum retrieval similarity for rule-based memory gate.")
    parser.add_argument("--no_harm_arbitration", action="store_true", default=False,
                        help="Run no-memory and memory ranking, then use memory only if applicability evidence is strong.")
    parser.add_argument("--no_harm_min_applicability", type=float, default=1.0,
                        help="Minimum applicability score required to use memory ranking under --no_harm_arbitration.")

    # Hyperparameter for few-shot ranking (LLM ranking)
    parser.add_argument("--fewshot_ranking", action="store_true", default=False)
    parser.add_argument("--k_shot", type=int, default=3)

    # Other
    parser.add_argument("--number_of_users", type=int, default=100)
    parser.add_argument("--max_positive_interactions", type=int, default=0,
                        help="If >0, use only the latest N positive train interactions per user.")
    parser.add_argument("--max_negative_candidates", type=int, default=0,
                        help="If >0, use only the first N negative candidates per user during evaluation.")
    parser.add_argument("--candidate_negative_mode", type=str, default="candidate_hard",
                        choices=["random", "candidate_hard"],
                        help="Training negative sampling: random global item or user-runtime hard negative.")
    parser.add_argument("--training_negative_source", type=str, default="legacy_runtime",
                        choices=["legacy_runtime", "train_catalog"])
    parser.add_argument("--min_lesson_confidence", type=float, default=0.25,
                        help="Minimum confidence/specificity required to keep a new fail lesson.")
    parser.add_argument("--max_lesson_risk", type=float, default=0.85,
                        help="Maximum overgeneralization risk allowed for a new fail lesson.")
    parser.add_argument("--max_failure_lessons_per_user", type=int, default=3,
                        help="Maximum kept failure lessons per user during training. <=0 means no cap.")
    parser.add_argument("--ranking_prompt_style", type=str, default="compact_score",
                        choices=[
                            "memcf", "compact_score",
                            "memrec_vanilla", "weak_memory_score",
                            "weak_memory_evidence_score", "weak_memory_router_score",
                            "weak_anchor_score", "weak_anchor_router_score",
                            "compact_stage_r", "compact_stage_r_reasoning",
                            "compact_curated_score",
                        ],
                        help="Prompt style for ranking.")
    parser.add_argument("--trace_dir", type=str, default=None,
                        help="Directory for JSONL traces. Default: evaluation_results/<dataset>/traces/<run_name>.")
    parser.add_argument("--disable_trace", action="store_false", dest="trace_enabled",
                        help="Disable JSONL traces for this run.")
    parser.set_defaults(trace_enabled=True)
    return parser.parse_args()


def main():
    # Paths to data files
    args = parse_args()

    data_name = args.data_name
    use_memory = args.use_memory
    LOAD_SAVED_MEMORY = args.LOAD_SAVED_MEMORY

    wo_evolving = args.wo_evolving
    wo_link = args.wo_link
    max_evolutions_per_memory = args.max_evolutions_per_memory
    window_size = args.window_size
    link_size = args.link_size
    max_iterations = args.max_iterations

    k_memories = args.k_memories
    memory_retrieval_mode = args.memory_retrieval_mode
    memory_gate = args.memory_gate
    memory_similarity_threshold = args.memory_similarity_threshold
    no_harm_arbitration = args.no_harm_arbitration
    no_harm_min_applicability = args.no_harm_min_applicability
    fewshot_ranking = args.fewshot_ranking
    k_shot = args.k_shot

    number_of_users = args.number_of_users
    max_positive_interactions = args.max_positive_interactions
    max_negative_candidates = args.max_negative_candidates
    candidate_negative_mode = args.candidate_negative_mode
    training_negative_source = args.training_negative_source
    min_lesson_confidence = args.min_lesson_confidence
    max_lesson_risk = args.max_lesson_risk
    max_failure_lessons_per_user = args.max_failure_lessons_per_user
    ranking_prompt_style = args.ranking_prompt_style
    trace_enabled = args.trace_enabled

    base_dir = (
        os.getenv("MEMCF_ROOT")
        or os.getenv("AGENTICREC_CFMEMORY_ROOT")
        or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    data_root = (
        os.getenv("MEMCF_DATA_ROOT")
        or os.getenv("AGENTICREC_DATA_ROOT")
        or os.path.join(base_dir, "data")
    )
    eval_root = (
        os.getenv("MEMCF_EVAL_ROOT")
        or os.getenv("AGENTICREC_EVAL_ROOT")
        or os.path.join(base_dir, "evaluation_results")
    )
    memory_root = (
        os.getenv("MEMCF_MEMORY_ROOT")
        or os.getenv("AGENTICREC_MEMORY_ROOT")
        or os.path.join(base_dir, "agent_memory")
    )

    items_path = os.path.join(data_root, data_name, "items.json")
    sequences_path = os.path.join(data_root, data_name, "user_sequences_10.json")
    negatives_path = os.path.join(data_root, data_name, "user_negatives_10.json")
    
    if use_memory:
        if wo_evolving:
            output_file = os.path.join(eval_root, data_name, f"nuser{number_of_users}_fail_interactions_no_evolving_k{k_memories}_iter{max_iterations}_memory.json")
            memory_file_path = os.path.join(memory_root, data_name, f"nuser{number_of_users}_fail_interactions_no_evolving_iter{max_iterations}.json")
        elif wo_link:
            output_file = os.path.join(eval_root, data_name, f"nuser{number_of_users}_fail_interactions_no_link_k{k_memories}_iter{max_iterations}_memory_maxevolution{str(max_evolutions_per_memory)}.json")
            memory_file_path = os.path.join(memory_root, data_name, f"nuser{number_of_users}_fail_interactions_no_link_iter{max_iterations}_maxevolution{str(max_evolutions_per_memory)}.json")
        else:
            output_file = os.path.join(eval_root, data_name, f"nuser{number_of_users}_global_fail_interactions_{k_memories}_iter{max_iterations}_link{link_size}_memory_maxevolution{str(max_evolutions_per_memory)}.json")
            memory_file_path = os.path.join(memory_root, data_name, f"nuser{number_of_users}_global_fail_interactions_iter{max_iterations}_link{link_size}_maxevolution{str(max_evolutions_per_memory)}.json")
    else:
        if fewshot_ranking:
            output_file = os.path.join(eval_root, data_name, f"nuser{number_of_users}_fewshot_{k_shot}_users_ranking_no_memory.json")
        else:
            output_file = os.path.join(eval_root, data_name, f"nuser{number_of_users}_zeroshot_users_ranking_no_memory.json")

    if use_memory and (memory_retrieval_mode != "user_only" or memory_gate != "none"):
        phase2_suffix = (
            f"_retrieval{memory_retrieval_mode}_gate{memory_gate}"
            f"_thr{str(memory_similarity_threshold).replace('.', 'p')}"
        )
        output_file = output_file.replace(".json", f"{phase2_suffix}.json")
    if use_memory and no_harm_arbitration:
        output_file = output_file.replace(
            ".json",
            f"_noharm_minapp{str(no_harm_min_applicability).replace('.', 'p')}.json",
        )

    run_name = os.path.splitext(os.path.basename(output_file))[0]
    trace_dir = args.trace_dir or os.path.join(
        eval_root,
        data_name,
        "traces",
        f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    trace_recorder = TraceRecorder(trace_dir=trace_dir, enabled=trace_enabled)
    trace_recorder.write_manifest({
        "run_name": run_name,
        "output_file": output_file,
        "data_name": data_name,
        "number_of_users": number_of_users,
        "use_memory": use_memory,
        "load_saved_memory": LOAD_SAVED_MEMORY,
        "wo_evolving": wo_evolving,
        "wo_link": wo_link,
        "max_iterations": max_iterations,
        "k_memories": k_memories,
        "memory_retrieval_mode": memory_retrieval_mode,
        "memory_gate": memory_gate,
        "memory_similarity_threshold": memory_similarity_threshold,
        "no_harm_arbitration": no_harm_arbitration,
        "no_harm_min_applicability": no_harm_min_applicability,
        "max_positive_interactions": max_positive_interactions,
        "max_negative_candidates": max_negative_candidates,
        "candidate_negative_mode": candidate_negative_mode,
        "min_lesson_confidence": min_lesson_confidence,
        "max_lesson_risk": max_lesson_risk,
        "max_failure_lessons_per_user": max_failure_lessons_per_user,
        "ranking_prompt_style": ranking_prompt_style,
        "phase": "phase1_correctness",
    })
    if trace_enabled:
        print(f"✓ Trace enabled: {trace_dir}")
    # Load data
    items_meta, user_sequences, user_negatives = load_data(
        items_path, sequences_path, negatives_path
    )

    print(f"Total users loaded: {len(user_sequences)}")
    
    # Get first 100 users
    user_ids = list(user_sequences.keys())[: number_of_users]
    if not use_memory and fewshot_ranking:
        sample_user_ids = list(user_sequences.keys())[number_of_users:]
    
    global_memory = RecommendationMemorySystem(use_gemini_embeddings=True)
    global_memory.trace_recorder = trace_recorder
    user_states: Dict[str, PairwiseUserState] = {}
    item_states = init_pairwise_item_states(items_meta)

    if use_memory:
        if LOAD_SAVED_MEMORY and os.path.exists(memory_file_path):
            print("\n" + "="*80)
            print("LOADING SAVED MEMORY SYSTEM")
            print("="*80)
            global_memory.load_memory(memory_file_path)
        else:
            print("\n" + "="*80)
            print("PHASE 1: TRAINING WITH CROSS-USER EVOLVING ONLY")
            print("="*80)
            
            global_memory = RecommendationMemorySystem(use_gemini_embeddings=False)
            global_memory.trace_recorder = trace_recorder
            for user_id in user_ids:
                profile = initialize_user_memory_from_history_v2(
                    memory_system=global_memory,
                    user_id=str(user_id),
                    user_data=user_sequences[user_id],
                    items_meta=items_meta,
                    max_positive_interactions=max_positive_interactions,
                )
                user_states[str(user_id)] = PairwiseUserState(
                    user_id=str(user_id),
                    short_term_memory=profile.profile,
                )
                global_memory._trace("user_memory_initialized", {
                    "user_id": user_id,
                    "profile": asdict(profile),
                })
            
            # shuffled_user_ids = user_ids.copy()
            # random.shuffle(shuffled_user_ids)
            
            for user_id in tqdm(user_ids, desc="Cross-user Training"):
                user_data = user_sequences[user_id]
                effective_train_len = (
                    min(len(user_data['train']), max_positive_interactions)
                    if max_positive_interactions and max_positive_interactions > 0
                    else min(len(user_data['train']), 30)
                )
                print(f"\nProcessing user {user_id} ({effective_train_len}/{len(user_data['train'])} train interactions)")
                
                # Tạo temp system chỉ để generate new memories
                # temp_system = RecommendationMemorySystem(use_gemini_embeddings=False)
                temp_system = RecommendationMemorySystem.__new__(RecommendationMemorySystem)
                temp_system.llm_name = global_memory.llm_name
                temp_system.embedding_model_name = global_memory.embedding_model_name
                temp_system.chat_model_name = global_memory.chat_model_name
                temp_system.chat_api_base = global_memory.chat_api_base
                temp_system.embedding_api_base = global_memory.embedding_api_base
                temp_system.api_key = global_memory.api_key
                temp_system.use_api_chat = global_memory.use_api_chat
                temp_system.use_api_embedding = global_memory.use_api_embedding
                temp_system.model = getattr(global_memory, "model", None)
                temp_system.tokenizer = getattr(global_memory, "tokenizer", None)
                temp_system.embedding_model = getattr(global_memory, "embedding_model", None)
                temp_system.trace_recorder = trace_recorder

                # reset memory data only
                temp_system.behavior_memories = []
                temp_system.user_interaction_history = []
                temp_system.next_thought_id = 0
                temp_system.memory_diagnostics = defaultdict(float)
                
                # Chỉ tạo new memories từ user này (không evolve nội bộ)
                try:
                    new_memories = train_memory_from_fail_interactions(
                        user_id=user_id,
                        user_data=user_data,
                        memory_system=temp_system,
                        user_states=user_states,
                        item_states=item_states,
                        items_meta=items_meta,
                        negative_data=user_negatives.get(user_id, {}),
                        max_iterations=max_iterations,
                        max_positive_interactions=max_positive_interactions,
                        candidate_negative_mode=candidate_negative_mode,
                        min_lesson_confidence=min_lesson_confidence,
                        max_lesson_risk=max_lesson_risk,
                        max_failure_lessons_per_user=max_failure_lessons_per_user,
                        training_negative_source=training_negative_source,
                    )
                except Exception as e:
                    print(f"\nError evaluating user {user_id}: {e}")
                    continue
                if not new_memories:
                    print("  → No new memories generated, skipping...")
                    continue
                
                print(f"  → Generated {len(new_memories)} fail-interaction memories")
               
                max_global_id = global_memory.next_thought_id - 1 if global_memory.behavior_memories else -1

                for new_mem in new_memories:
                    # Offset thought_id của memory mới
                    local_thought_id = new_mem.thought_id
                    new_mem.thought_id += (max_global_id + 1)
                    trace_recorder.log("global_memory_candidate", {
                        "user_id": user_id,
                        "local_thought_id": local_thought_id,
                        "global_thought_id": new_mem.thought_id,
                        "wo_evolving": wo_evolving,
                        "wo_link": wo_link,
                        "memory": behavior_memory_to_trace(new_mem),
                    })
                    
                    # Link với global (linked_ids là id cũ trong global)
                    if not wo_evolving:
                        try:
                            linked_ids = global_memory.link_behavior_memories(new_mem, k=link_size, wo_link=wo_link)
                            new_mem.links = linked_ids  # vẫn là id cũ, đúng
                            
                            # Evolve global dựa trên new_mem
                            global_memory.evolve_behavior_memories(new_mem, linked_ids, max_evolutions_per_memory=max_evolutions_per_memory)
                        except Exception as e:
                            print(f"\nError evolving memory for user {user_id}: {e}")
                    
                    # Add vào global
                    global_memory.behavior_memories.append(new_mem)
                    
                    # Update next_id
                    global_memory.next_thought_id = new_mem.thought_id + 1
                    trace_recorder.log("global_memory_added", {
                        "user_id": user_id,
                        "global_thought_id": new_mem.thought_id,
                        "links": new_mem.links,
                        "memory_pool_size": len(global_memory.behavior_memories),
                    })
                
                print(f"  → Global memory pool now has {len(global_memory.behavior_memories)} memories")
                global_memory.save_memory(memory_file_path, format='json')
                print(f"  → Overwritten common global memory file: {memory_file_path}")
        
        print("\n" + "="*80)
        print("SAVING GLOBAL CROSS-USER EVOLVING MEMORY")
        print("="*80)
        global_memory.save_memory(memory_file_path, format='json')
        global_memory.print_evolution_report()
        stats = global_memory.get_evolution_statistics()
        stats_file = memory_file_path.replace('.json', '_evolution_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Evolution statistics saved to {stats_file}")
            

    
    # PHASE 2: EVALUATE ON VALIDATION SET
    print("\n" + "="*80)
    print("PHASE 2: VALIDATION SET EVALUATION")
    print("="*80)
    
    val_metrics = {
        'recall@5': [], 'recall@10': [], 'recall@20': [],
        'ndcg@5': [], 'ndcg@10': [], 'ndcg@20': []
    }
    baseline_metrics = {
        'recall@5': [], 'recall@10': [], 'recall@20': [],
        'ndcg@5': [], 'ndcg@10': [], 'ndcg@20': []
    }
    
    all_user_results = []
    
    if not use_memory:
        global_memory = RecommendationMemorySystem(use_gemini_embeddings=True)
        global_memory.trace_recorder = trace_recorder
    for user_id in tqdm(user_ids, desc="Validation"):
        try:
            user_data = user_sequences[user_id]
            negative_data = user_negatives.get(user_id, {})

            # sample_user_id = random.choice(sample_user_ids) if not use_memory and fewshot_ranking else None
            sample_user_id_list = random.sample(sample_user_ids, k_shot) if not use_memory and fewshot_ranking else None
            if sample_user_id_list is not None:
                sample_user_list = []
                negative_data_sample_list = []
                for id in sample_user_id_list:
                    sample_user_data = user_sequences[id] if id else None
                    negative_data_sample = user_negatives.get(id, {}) if id else None   
                    sample_user_list.append(sample_user_data)
                    negative_data_sample_list.append(negative_data_sample)
            else:
                sample_user_list = None
                negative_data_sample_list = None
            # print(sample_user_data)

            baseline_metric, metrics, candidates, predictions, ground_truth = evaluate_user(
                user_data, negative_data, 
                items_meta, global_memory, eval_type='test', use_memory=use_memory, k_memories=k_memories, sample_user_list=sample_user_list, negative_data_sample_list=negative_data_sample_list,
                max_positive_interactions=max_positive_interactions, max_negative_candidates=max_negative_candidates,
                user_id=user_id,
                memory_retrieval_mode=memory_retrieval_mode,
                memory_gate=memory_gate,
                memory_similarity_threshold=memory_similarity_threshold,
                no_harm_arbitration=no_harm_arbitration,
                no_harm_min_applicability=no_harm_min_applicability,
                ranking_prompt_style=ranking_prompt_style,
            )
            # Lưu tạm thông tin user này
            all_user_results.append({
                "user_id": user_id,
                "ground_truth": ground_truth,
                "candidates": candidates,
                "predictions": predictions,
                "metrics": metrics,
                "baseline_metrics": baseline_metric
            })
            for key in val_metrics:
                val_metrics[key].append(metrics[key])
                baseline_metrics[key].append(baseline_metric[key])
                
        except Exception as e:
            print(f"\nError evaluating user {user_id}: {e}")
            continue
    
    save_all_users_ranking_results(
        all_results=all_user_results,
        items_meta=items_meta,
        output_file=output_file
    )
    # Print validation results
    print("\nValidation Results:")
    print("-" * 80)
    for metric in ['recall@5', 'recall@10', 'recall@20','ndcg@5', 'ndcg@10', 'ndcg@20']:
        if len(baseline_metrics[metric]) > 0:
            mean_val = np.mean(baseline_metrics[metric])
            print(f"Baseline {metric:10s}: {mean_val:.4f}")
        else:
            print(f"Baseline {metric:10s}: N/A")
    print("-" * 80)
    for metric in ['recall@5', 'recall@10', 'recall@20','ndcg@5', 'ndcg@10', 'ndcg@20']:
        if len(val_metrics[metric]) > 0:
            mean_val = np.mean(val_metrics[metric])
            print(f"{metric:12s}: {mean_val:.4f}")
        else:
            print(f"{metric:12s}: N/A")

    diag = getattr(global_memory, "memory_diagnostics", defaultdict(float))
    diag_users = float(diag.get("eval_users", 0.0))
    retrieved_total = float(diag.get("retrieved_total", 0.0))
    kept_total = float(diag.get("kept_total", 0.0))
    skipped_total = float(diag.get("skipped_total", 0.0))
    memory_diagnostics = {
        "retrieval_mode": memory_retrieval_mode,
        "memory_gate": memory_gate,
        "memory_similarity_threshold": memory_similarity_threshold,
        "eval_users_with_memory_retrieval": int(diag_users),
        "retrieved_total": int(retrieved_total),
        "kept_total": int(kept_total),
        "skipped_total": int(skipped_total),
        "avg_retrieved_memories": retrieved_total / diag_users if diag_users else 0.0,
        "avg_kept_memories": kept_total / diag_users if diag_users else 0.0,
        "avg_skipped_memories": skipped_total / diag_users if diag_users else 0.0,
        "gate_keep_rate": kept_total / retrieved_total if retrieved_total else 0.0,
        "gate_skip_rate": skipped_total / retrieved_total if retrieved_total else 0.0,
        "users_with_kept_memory": int(diag.get("users_with_kept_memory", 0.0)),
        "rank_score_calls": int(diag.get("rank_score_calls", 0.0)),
        "rank_valid_score_outputs": int(diag.get("rank_valid_score_outputs", 0.0)),
        "rank_invalid_score_outputs": int(diag.get("rank_invalid_score_outputs", 0.0)),
        "rank_valid_score_rate": (
            float(diag.get("rank_valid_score_outputs", 0.0)) / float(diag.get("rank_score_calls", 0.0))
            if float(diag.get("rank_score_calls", 0.0)) else 0.0
        ),
        "rank_missing_score_rows": int(diag.get("rank_missing_score_rows", 0.0)),
        "rank_invalid_score_rows": int(diag.get("rank_invalid_score_rows", 0.0)),
        "rank_attempt_errors": int(diag.get("rank_attempt_errors", 0.0)),
        "rank_fallbacks": int(diag.get("rank_fallbacks", 0.0)),
        "no_harm_users": int(diag.get("no_harm_users", 0.0)),
        "no_harm_used_memory": int(diag.get("no_harm_used_memory", 0.0)),
        "no_harm_fallback_no_memory": int(diag.get("no_harm_fallback_no_memory", 0.0)),
        "no_harm_memory_use_rate": (
            float(diag.get("no_harm_used_memory", 0.0)) / float(diag.get("no_harm_users", 0.0))
            if float(diag.get("no_harm_users", 0.0)) else 0.0
        ),
    }

    summary = {
        "model": "MEMCF",
        "dataset": data_name,
        "number_of_users_requested": number_of_users,
        "number_of_users_evaluated": len(all_user_results),
        "use_memory": use_memory,
        "load_saved_memory": LOAD_SAVED_MEMORY,
        "wo_evolving": wo_evolving,
        "wo_link": wo_link,
        "max_iterations": max_iterations,
        "k_memories": k_memories,
        "memory_retrieval_mode": memory_retrieval_mode,
        "memory_gate": memory_gate,
        "memory_similarity_threshold": memory_similarity_threshold,
        "no_harm_arbitration": no_harm_arbitration,
        "no_harm_min_applicability": no_harm_min_applicability,
        "trace_enabled": trace_enabled,
        "trace_dir": trace_dir if trace_enabled else None,
        "max_positive_interactions": max_positive_interactions,
        "max_negative_candidates": max_negative_candidates,
        "candidate_negative_mode": candidate_negative_mode,
        "training_negative_source": training_negative_source,
        "min_lesson_confidence": min_lesson_confidence,
        "max_lesson_risk": max_lesson_risk,
        "max_failure_lessons_per_user": max_failure_lessons_per_user,
        "ranking_prompt_style": ranking_prompt_style,
        "phase1_correctness": {
            "clean_ranked_item_ids": True,
            "drop_hallucinated_item_ids": True,
            "deduplicate_ranked_item_ids": True,
            "append_missing_candidates": True,
            "deterministic_candidate_order": True,
            "api_temperature": float(os.getenv("MEMCF_TEMPERATURE", "0.0")),
            "api_seed": (
                int(os.getenv("MEMCF_LLM_SEED"))
                if os.getenv("MEMCF_LLM_SEED", "").strip()
                else None
            ),
            "rank_max_tokens": int(os.getenv("MEMCF_RANK_MAX_TOKENS", "1500")),
            "rank_retries": int(os.getenv("MEMCF_RANK_RETRIES", "1")),
            "strict_gate_min_strong_terms": int(os.getenv("MEMCF_STRICT_GATE_MIN_STRONG_TERMS", "2")),
            "normalize_categories": True,
            "strict_output_validation": True,
            "retry_invalid_rankings": True,
            "score_based_ranking": True,
            "candidate_aliases": True,
            "structured_memory_fields": True,
            "applicability_gate_available": True,
            "no_harm_arbitration_available": True,
        },
        "baseline_metrics": {
            metric: (float(np.mean(baseline_metrics[metric])) if len(baseline_metrics[metric]) > 0 else None)
            for metric in ['recall@5', 'recall@10', 'recall@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']
        },
        "metrics": {
            metric: (float(np.mean(val_metrics[metric])) if len(val_metrics[metric]) > 0 else None)
            for metric in ['recall@5', 'recall@10', 'recall@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']
        },
        "memory_diagnostics": memory_diagnostics,
    }
    summary_file = output_file.replace(".json", ".summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved MEMCF summary to {summary_file}")
    trace_recorder.write_manifest({
        "run_name": run_name,
        "output_file": output_file,
        "summary_file": summary_file,
        "completed_at": datetime.now().isoformat(),
        "summary": summary,
    })


def behavior_memory_passes_quality_gate(
    memory: "BehaviorMemory",
    interaction_window: List["UserInteraction"],
    min_confidence: float,
    max_risk: float,
) -> Tuple[bool, str]:
    specificity = _safe_float(getattr(memory, "specificity_score", 0.0), 0.0)
    risk = _safe_float(getattr(memory, "overgeneralization_risk", 1.0), 1.0)
    wrong_ids = [
        str(x.item_id) for x in interaction_window
        if str((x.metadata or {}).get("role", "")) in {"chosen_wrong", "wrong_choice"}
    ]
    correct_ids = [
        str(x.item_id) for x in interaction_window
        if str((x.metadata or {}).get("role", "")) in {"ground_truth", "preferred_item"}
    ]
    concrete_terms = normalize_evidence_terms(
        list(getattr(memory, "evidence_terms_required", []) or [])
        + list(getattr(memory, "keywords", []) or [])
        + [getattr(memory, "wrong_item_type", ""), getattr(memory, "correct_item_type", "")]
    )
    combined_text = " ".join([
        str(getattr(memory, "behavior_explanation", "")),
        str(getattr(memory, "pattern_description", "")),
        str(getattr(memory, "wrong_item_type", "")),
        str(getattr(memory, "correct_item_type", "")),
        " ".join(concrete_terms[:8]),
    ])
    if specificity < min_confidence:
        return False, f"low_specificity:{specificity:.2f}"
    if risk > max_risk:
        return False, f"high_risk:{risk:.2f}"
    if not wrong_ids or not correct_ids:
        return False, "missing_wrong_or_correct_item_ids"
    if len(concrete_terms) < 2:
        return False, "too_few_concrete_terms"
    if memory_text_is_too_generic(combined_text):
        return False, "memory_text_too_generic"
    return True, "accepted"


# Imported last: these names are only used inside function bodies, and importing them
# at the top would create a circular import between modules.
from fmrec.memory_build import (
    autonomous_pairwise_interaction,
    choose_training_negative_item_id,
    init_pairwise_item_states,
    initialize_user_memory_from_history_v2,
)
