"""Failure-memory construction (temporal_factual protocol) and memory (de)serialisation."""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import asdict
import random
import re
import hashlib

from fmrec.common import (
    deterministic_shuffle,
    extract_json_object,
    has_metadata_noise,
    item_category,
    item_title,
    make_jsonable,
    normalize_evidence_terms,
    normalize_terms,
)
from fmrec.graph_index import MemoryGraphIndex
from fmrec.llm_client import RecommendationMemorySystem
from fmrec.prompts import _history_item_infos, _item_info_for_prompt
from fmrec.records import FailureLesson, PairwiseItemState, PairwiseUserState, UserMemoryProfile


def init_pairwise_item_states(items_meta: Dict[str, Dict[str, Any]]) -> Dict[str, PairwiseItemState]:
    """Initialize item states for pairwise failure training."""
    item_states: Dict[str, PairwiseItemState] = {}
    for item_id, item_info in items_meta.items():
        title = item_title(item_info, str(item_id))
        category = item_category(item_info)
        memory = f"The item is called '{title}'. The category is: '{category}'."
        item_states[item_id] = PairwiseItemState(
            item_id=item_id,
            title=title,
            category=category,
            memory=memory,
        )
    return item_states


def autonomous_pairwise_interaction(
    memory_system: RecommendationMemorySystem,
    user_state: PairwiseUserState,
    pos_item: PairwiseItemState,
    neg_item: PairwiseItemState,
) -> Tuple[str, str]:
    """pairwise autonomous interaction: choose between positive/negative item."""
    prompt = f"""You are an enthusiast. Here is your self-introduction: "{user_state.short_term_memory}"

Now, you are considering to select an item from two candidates:
1. Title: {neg_item.title}, Description: {neg_item.memory}
2. Title: {pos_item.title}, Description: {pos_item.memory}
\n\n Please select the item that aligns best with your preferences and explain your choice while rejecting the other. \n Follow these steps:\n 1. Extract your preferences and dislikes from your self-introduction. \n 2. Evaluate the two items based on your preferences and how they relate to the item features.\n 3. Explain your choice, detailing the relationship between your preferences/dislikes and the item features

\n\n Important notes:
\n 1. Do not fabricate your preferences! If your self-introduction lacks relevant details, use common knowledge to guide your decision, such as item popularity. \n 2. Select one candidate, not both. \n 3. Your explanation should be specific; general preferences like genre are insufficient. Focus on the item's finer attributes and be concise! \n 4. Base your explanation on facts. If your self-introduction doesn't specify preferences, you cannot claim your decision was influenced by them."

Output format:
Chosen Item: [1 or 2]
Explanation: [Your detailed reasoning]

Important: You must choose one of these two candidates."""

    response = memory_system.qwen_generate(prompt=prompt, call_type="pairwise_choice")
    chosen_item_id = pos_item.item_id
    if "Chosen Item: 1" in response or "chosen item: 1" in response.lower():
        chosen_item_id = neg_item.item_id
    memory_system._trace("autonomous_choice_llm", {
        "user_id": user_state.user_id,
        "prompt": prompt,
        "role_prompt": "You are a helpful AI assistant.",
        "answer": response,
        "positive_item": asdict(pos_item),
        "negative_item": asdict(neg_item),
        "chosen_item_id": chosen_item_id,
        "is_failure": chosen_item_id != pos_item.item_id,
        "user_state": asdict(user_state),
    })
    return chosen_item_id, response


def _item_tokens_for_hard_negative(item_id: str, items_meta: Dict[str, Dict[str, Any]]) -> Set[str]:
    info = _item_info_for_prompt(str(item_id), items_meta)
    text = f"{info.get('title', '')} {info.get('category', '')}"
    return set(normalize_terms(text))


def choose_training_negative_item_id(
    user_id: str,
    pos_item_id: str,
    user_data: Dict[str, Any],
    negative_data: Optional[Dict[str, Any]],
    items_meta: Dict[str, Dict[str, Any]],
    all_item_ids: List[str],
    mode: str = "random",
    max_positive_interactions: Optional[int] = None,
    training_negative_source: str = "legacy_runtime",
    train_catalog_pool_size: int = 512,
    history_override: Optional[List[str]] = None,
) -> Optional[str]:
    valid_item_ids = set(str(x) for x in all_item_ids)
    pos_item_id = str(pos_item_id)
    if training_negative_source == "train_catalog":
        # Clean protocol: sample only from the catalog and train interactions.
        # In particular, do not inspect val_neg/test_neg while creating memory.
        # This means excluding this user's own held-out val/test items and their
        # already-frozen val_neg/test_neg candidates from the sampling pool, not
        # just their train items -- otherwise a training-phase contrastive
        # "negative" example could coincide with this user's real held-out
        # answer or with an eval candidate, which the comment above already
        # says must not happen.
        train_ids = {str(x) for x in user_data.get("train", [])}
        held_out_ids = {str(x) for x in user_data.get("val", [])} | {str(x) for x in user_data.get("test", [])}
        frozen_negative_ids: Set[str] = set()
        if negative_data:
            frozen_negative_ids |= {str(x) for x in negative_data.get("val_neg", [])}
            frozen_negative_ids |= {str(x) for x in negative_data.get("test_neg", [])}
        exclude_ids = train_ids | held_out_ids | frozen_negative_ids | {pos_item_id}
        seed_material = f"train_catalog::{user_id}::{pos_item_id}"
        seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed)
        sample_size = min(
            len(all_item_ids),
            max(int(train_catalog_pool_size), int(train_catalog_pool_size) + len(exclude_ids)),
        )
        sampled_ids = rng.sample(all_item_ids, sample_size) if sample_size else []
        runtime_pool = [
            str(item_id) for item_id in sampled_ids
            if str(item_id) in valid_item_ids and str(item_id) not in exclude_ids
        ][:max(1, int(train_catalog_pool_size))]
        if not runtime_pool:
            runtime_pool = [
                str(item_id) for item_id in all_item_ids
                if str(item_id) not in exclude_ids
            ][:max(1, int(train_catalog_pool_size))]
    else:
        runtime_pool = collect_runtime_negative_pool(
            negative_data=negative_data,
            valid_item_ids=valid_item_ids,
            exclude_ids={pos_item_id},
        )
    if mode != "candidate_hard":
        base_pool = runtime_pool or [str(iid) for iid in all_item_ids if str(iid) != pos_item_id]
        if not base_pool:
            return None
        return deterministic_shuffle(base_pool, salt=f"randneg::{user_id}::{pos_item_id}")[0]

    if not runtime_pool:
        fallback_pool = [str(iid) for iid in all_item_ids if str(iid) != pos_item_id]
        if not fallback_pool:
            return None
        return deterministic_shuffle(fallback_pool, salt=f"hardneg_fallback::{user_id}::{pos_item_id}")[0]

    history_ids = [
        str(x) for x in (
            history_override if history_override is not None else user_data.get("train", [])
        )
    ]
    if max_positive_interactions and max_positive_interactions > 0:
        history_ids = history_ids[-max_positive_interactions:]
    else:
        history_ids = history_ids[-10:]
    anchor_tokens: Set[str] = set()
    for hid in history_ids:
        anchor_tokens.update(_item_tokens_for_hard_negative(hid, items_meta))
    anchor_tokens.update(_item_tokens_for_hard_negative(pos_item_id, items_meta))

    pos_category = _item_info_for_prompt(pos_item_id, items_meta).get("category", "Unknown")
    shuffled_pool = deterministic_shuffle(runtime_pool, salt=f"hardneg_pool::{user_id}::{pos_item_id}")
    best_item_id = shuffled_pool[0]
    best_score = -1.0
    for neg_item_id in shuffled_pool:
        neg_tokens = _item_tokens_for_hard_negative(neg_item_id, items_meta)
        neg_category = _item_info_for_prompt(neg_item_id, items_meta).get("category", "Unknown")
        overlap = len(anchor_tokens & neg_tokens)
        category_bonus = 0.5 if pos_category != "Unknown" and pos_category == neg_category else 0.0
        score = float(overlap) + category_bonus
        if score > best_score:
            best_score = score
            best_item_id = neg_item_id
    return best_item_id


def initialize_user_memory_from_history_v2(
    memory_system: RecommendationMemorySystem,
    user_id: str,
    user_data: Dict[str, Any],
    items_meta: Dict[str, Dict[str, Any]],
    max_positive_interactions: Optional[int] = None,
) -> UserMemoryProfile:
    history_ids = user_data.get("train", [])
    if max_positive_interactions and max_positive_interactions > 0:
        history_ids = history_ids[-max_positive_interactions:]
    else:
        history_ids = history_ids[-10:]
    history_items = _history_item_infos(history_ids, items_meta)
    if not history_items:
        return UserMemoryProfile(
            user_id=str(user_id),
            profile="The user has no usable recent history.",
            facets=[],
            evidence_item_ids=[],
            source="empty_history",
        )

    prompt = f"""Create a compact user memory from observed recommendation history.

Observed positive history:
{json.dumps(history_items, ensure_ascii=False, indent=2)}

Return ONLY valid JSON:
{{
  "profile": "under 80 words describing concrete preferences grounded in history",
  "facets": [
    "User likes ... and often prefers ...",
    "User does not show evidence for ..."
  ],
  "evidence_item_ids": ["item id from history"]
}}

Rules:
- Use only evidence from the observed history.
- Do not infer broad demographic traits.
- Prefer concrete titles, subgenres, product attributes, artist/style/format signals, or use-case signals.
- Avoid generic phrases like 'likes products in this category'."""

    try:
        raw = memory_system.qwen_generate(
            prompt=prompt,
            role_prompt="You summarize user preference evidence for a recommender system.",
            max_new_tokens=700,
            json_mode=True,
            call_type="user_memory_init",
        )
        parsed = extract_json_object(raw)
        profile = str(parsed.get("profile") or "").strip()
        facets = [str(x).strip() for x in parsed.get("facets", []) if str(x).strip()]
        evidence = [str(x).strip() for x in parsed.get("evidence_item_ids", []) if str(x).strip()]
        if not profile:
            raise ValueError("empty profile")
        result = UserMemoryProfile(
            user_id=str(user_id),
            profile=profile,
            facets=facets[:8],
            evidence_item_ids=evidence[:10],
            source="llm_history_init",
        )
        memory_system._trace("user_memory_init_llm", {
            "user_id": user_id,
            "prompt": prompt,
            "answer": raw,
            "parsed": parsed,
            "profile": asdict(result),
        })
        return result
    except Exception as e:
        titles = [x["title"] for x in history_items[:5]]
        categories = sorted({x["category"] for x in history_items if x.get("category")})
        profile = (
            "The user has positive history with "
            + ", ".join(titles[:4])
            + (f". Observed categories: {', '.join(categories[:3])}." if categories else ".")
        )
        result = UserMemoryProfile(
            user_id=str(user_id),
            profile=profile[:500],
            facets=[f"User has positive history with {title}." for title in titles[:3]],
            evidence_item_ids=[str(x["item_id"]) for x in history_items[:10]],
            source="fallback_history_init",
        )
        memory_system._trace("user_memory_init_error", {
            "user_id": user_id,
            "error": str(e),
            "prompt": prompt,
            "fallback_profile": asdict(result),
        })
        return result


def build_temporal_prefix_profile(
    prefix_item_ids: List[str],
    items_meta: Dict[str, Dict[str, Any]],
    max_items: int = 5,
) -> str:
    """Describe only interactions strictly preceding a training target."""
    rows = _history_item_infos([str(x) for x in prefix_item_ids][-max(1, int(max_items)):], items_meta)
    if not rows:
        return "No usable preceding interactions were observed."
    parts = [
        f"'{str(row.get('title', '')).strip() or 'Unknown'}' ({str(row.get('category', 'Unknown')).strip() or 'Unknown'})"
        for row in rows
    ]
    return "Observed preceding interactions, oldest to newest: " + "; ".join(parts) + "."


def temporal_failure_write_gate(
    lesson: FailureLesson,
    valid_item_ids: Optional[Set[str]] = None,
) -> Tuple[bool, str]:
    """Fail-closed schema/provenance gate for temporal factual memories."""
    observed_id = str(lesson.observed_next_item_id or lesson.correct_item_id or "")
    selected_id = str(lesson.base_selected_item_id or lesson.wrong_item_id or "")
    prefix_ids = [str(x) for x in lesson.history_item_ids]
    if lesson.memory_type != "temporal_failure_contrast":
        return False, "not_temporal_failure_contrast"
    if not prefix_ids:
        return False, "empty_temporal_prefix"
    if not observed_id or not selected_id:
        return False, "missing_observed_or_selected_item"
    if observed_id == selected_id:
        return False, "not_a_ranking_failure"
    if observed_id in set(prefix_ids):
        return False, "temporal_target_present_in_prefix"
    if valid_item_ids is not None:
        required_ids = set(prefix_ids) | {observed_id, selected_id}
        if not required_ids.issubset({str(x) for x in valid_item_ids}):
            return False, "unknown_item_id"
    titles = [lesson.correct_item_title, lesson.wrong_item_title]
    if any(not str(title or "").strip() for title in titles):
        return False, "missing_endpoint_title"
    if any(has_metadata_noise(title) for title in titles):
        return False, "metadata_noise"
    placeholder = re.compile(r"^(unknown|item\s+[a-z0-9_-]+|amazon .* logo)$", re.I)
    if any(placeholder.match(str(title).strip()) for title in titles):
        return False, "placeholder_endpoint_title"
    forbidden = re.compile(r"\b(disliked|rejected|actually prefer|preferred over)\b", re.I)
    if forbidden.search(str(lesson.factual_statement or "")):
        return False, "synthetic_preference_wording"
    return True, "accepted"


def make_temporal_factual_lesson(
    user_id: str,
    prefix_item_ids: List[str],
    observed_item: PairwiseItemState,
    selected_item: PairwiseItemState,
    items_meta: Dict[str, Dict[str, Any]],
) -> FailureLesson:
    prefix_ids = [str(x) for x in prefix_item_ids]
    prefix_rows = _history_item_infos(prefix_ids[-3:], items_meta)
    prefix_titles = [
        str(row.get("title", "")).strip()
        for row in prefix_rows
        if str(row.get("title", "")).strip()
    ]
    prefix_text = "; ".join(f"'{title}'" for title in prefix_titles) or "the recorded prefix"
    factual_statement = (
        f"After preceding observed items {prefix_text}, the recorded next item was "
        f"'{observed_item.title}', while the base ranker selected '{selected_item.title}'."
    )
    seed = "|".join([
        str(user_id),
        ",".join(prefix_ids),
        str(observed_item.item_id),
        str(selected_item.item_id),
        "temporal_factual_v1",
    ])
    event_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    memory_id = hashlib.sha256(f"memory|{seed}".encode("utf-8")).hexdigest()[:16]
    evidence_terms = normalize_evidence_terms(
        prefix_titles
        + [
            observed_item.title,
            observed_item.category,
            selected_item.title,
            selected_item.category,
        ]
    )
    return FailureLesson(
        memory_id=memory_id,
        source_user_id=str(user_id),
        source_event_id=event_id,
        lesson=factual_statement[:500],
        prefer=observed_item.title[:250],
        avoid=selected_item.title[:250],
        applies_if=[],
        do_not_apply_if=[],
        evidence_terms=evidence_terms[:16],
        wrong_item_id=str(selected_item.item_id),
        correct_item_id=str(observed_item.item_id),
        wrong_item_title=selected_item.title[:300],
        correct_item_title=observed_item.title[:300],
        wrong_item_category=selected_item.category[:120],
        correct_item_category=observed_item.category[:120],
        source_user_preference=build_temporal_prefix_profile(prefix_ids, items_meta, max_items=3)[:500],
        failure_type="temporal_next_item_ranking_failure",
        history_item_ids=prefix_ids[-3:],
        confidence=1.0,
        overgeneralization_risk=0.0,
        memory_type="temporal_failure_contrast",
        observed_next_item_id=str(observed_item.item_id),
        base_selected_item_id=str(selected_item.item_id),
        creation_mode="fixed_factual_template",
        factual_statement=factual_statement[:500],
    )


def train_temporal_factual_memory_v2(
    user_id: str,
    user_data: Dict[str, Any],
    negative_data: Optional[Dict[str, Any]],
    items_meta: Dict[str, Dict[str, Any]],
    memory_system: RecommendationMemorySystem,
    item_states: Dict[str, PairwiseItemState],
    graph: MemoryGraphIndex,
    max_iterations: int = 1,
    max_positive_interactions: Optional[int] = None,
    candidate_negative_mode: str = "candidate_hard",
    max_failure_lessons_per_user: int = 3,
) -> List[FailureLesson]:
    """Collect prefix-causal failures and store fixed factual memories.

    This protocol never exposes a target through a full-history profile, never
    reads validation/test negatives, and never asks an LLM to invent a diagnosis
    after seeing the observed next item.
    """
    full_train = [
        str(item_id) for item_id in user_data.get("train", [])
        if str(item_id) in item_states
    ]
    if len(full_train) < 2:
        return []
    target_positions = list(range(1, len(full_train)))
    if max_positive_interactions and max_positive_interactions > 0:
        target_positions = target_positions[-int(max_positive_interactions):]

    all_item_ids = list(item_states.keys())
    valid_item_ids = set(all_item_ids)
    new_lessons: List[FailureLesson] = []
    for position in target_positions:
        observed_id = full_train[position]
        prefix_ids = full_train[:position]
        prefix_for_state = prefix_ids[-max(1, int(max_positive_interactions or 5)):]
        selected_id = choose_training_negative_item_id(
            user_id=str(user_id),
            pos_item_id=observed_id,
            user_data=user_data,
            negative_data=None,
            items_meta=items_meta,
            all_item_ids=all_item_ids,
            mode=candidate_negative_mode,
            max_positive_interactions=max_positive_interactions,
            training_negative_source="train_catalog",
            history_override=prefix_for_state,
        )
        if not selected_id or selected_id not in item_states:
            continue
        observed_item = item_states[observed_id]
        selected_item = item_states[selected_id]
        prefix_state = PairwiseUserState(
            user_id=str(user_id),
            short_term_memory=build_temporal_prefix_profile(
                prefix_for_state, items_meta, max_items=max_positive_interactions or 5
            ),
            interaction_history=list(prefix_for_state),
        )

        attempts = max(1, int(max_iterations))
        for attempt in range(attempts):
            chosen_item_id, explanation = autonomous_pairwise_interaction(
                memory_system=memory_system,
                user_state=prefix_state,
                pos_item=observed_item,
                neg_item=selected_item,
            )
            memory_system._trace("temporal_choice_result", {
                "user_id": str(user_id),
                "target_position": position,
                "prefix_item_ids": prefix_for_state,
                "observed_next_item_id": observed_id,
                "negative_item_id": selected_id,
                "chosen_item_id": chosen_item_id,
                "is_failure": chosen_item_id != observed_id,
                "attempt": attempt,
                "explanation_retained_for_trace_only": explanation,
            })
            if chosen_item_id == observed_id:
                break

            lesson = make_temporal_factual_lesson(
                user_id=str(user_id),
                prefix_item_ids=prefix_for_state,
                observed_item=observed_item,
                selected_item=selected_item,
                items_meta=items_meta,
            )
            passed, reason = temporal_failure_write_gate(lesson, valid_item_ids=valid_item_ids)
            memory_system._trace("temporal_memory_write_gate", {
                "user_id": str(user_id),
                "target_position": position,
                "passed": passed,
                "reason": reason,
                "lesson": asdict(lesson),
            })
            if not passed:
                break
            graph.add_lesson(lesson)
            new_lessons.append(lesson)
            memory_system._trace("temporal_memory_added", {
                "lesson": asdict(lesson),
                "provenance": {
                    "prefix_item_ids": lesson.history_item_ids,
                    "observed_next_item_id": lesson.observed_next_item_id,
                    "base_selected_item_id": lesson.base_selected_item_id,
                    "split": "train",
                },
            })
            break

        if max_failure_lessons_per_user > 0 and len(new_lessons) >= max_failure_lessons_per_user:
            break
    return new_lessons


def save_v2_memory(
    path: str,
    graph: MemoryGraphIndex,
    user_profiles: Dict[str, UserMemoryProfile],
    training_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": "MEMCF",
        "saved_at": datetime.now().isoformat(),
        "graph": graph.to_dict(),
        "user_profiles": {uid: asdict(profile) for uid, profile in user_profiles.items()},
        "training_metadata": dict(training_metadata or {}),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_jsonable(payload), f, ensure_ascii=False, indent=2)
    print(f"✓ Saved MEMCF graph memory to {path}")


def memory_artifact_stats(path: Optional[str], graph: MemoryGraphIndex, user_profiles: Dict[str, UserMemoryProfile]) -> Dict[str, Any]:
    file_size_bytes = os.path.getsize(path) if path and os.path.exists(path) else 0
    return {
        "memory_file": path,
        "file_size_bytes": int(file_size_bytes),
        "file_size_mb": file_size_bytes / (1024 * 1024) if file_size_bytes else 0.0,
        "num_user_profiles": len(user_profiles),
        **graph.stats(),
    }


def load_v2_memory(
    path: str,
    user_sequences: Dict[str, Dict[str, Any]],
    build_clusters: bool = True,
) -> Tuple[MemoryGraphIndex, Dict[str, UserMemoryProfile]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    graph = MemoryGraphIndex.from_dict(
        payload.get("graph", {}),
        user_sequences,
        build_clusters=build_clusters,
    )
    graph.artifact_metadata = dict(payload.get("training_metadata", {}))
    user_profiles = {
        str(uid): UserMemoryProfile(**profile)
        for uid, profile in payload.get("user_profiles", {}).items()
    }
    return graph, user_profiles


# Imported last: these names are only used inside function bodies, and importing them
# at the top would create a circular import between modules.
from fmrec.variants.legacy_lessons import collect_runtime_negative_pool
