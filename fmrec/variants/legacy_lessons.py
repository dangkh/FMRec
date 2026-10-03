"""Legacy failure-memory protocol (failure events -> LLM-written lessons)."""

import json
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import asdict
import hashlib

from fmrec.common import _safe_float, extract_json_object, normalize_evidence_terms, normalize_terms
from fmrec.graph_index import MemoryGraphIndex
from fmrec.llm_client import RecommendationMemorySystem
from fmrec.prompts import _history_item_infos
from fmrec.records import FailureEvent, FailureLesson, PairwiseItemState, PairwiseUserState


def dedupe_preserve_order(values: List[Any]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def collect_runtime_negative_pool(
    negative_data: Optional[Dict[str, Any]],
    valid_item_ids: Optional[Set[str]] = None,
    exclude_ids: Optional[Set[str]] = None,
) -> List[str]:
    negative_data = negative_data or {}
    values: List[str] = []
    for key in ("val_neg", "test_neg", "train_neg", "negatives"):
        raw = negative_data.get(key, [])
        if isinstance(raw, list):
            values.extend(str(x) for x in raw)
    values = dedupe_preserve_order(values)
    exclude_ids = set(str(x) for x in (exclude_ids or set()))
    if valid_item_ids is not None:
        values = [x for x in values if x in valid_item_ids and x not in exclude_ids]
    else:
        values = [x for x in values if x not in exclude_ids]
    return values


def failure_lesson_passes_quality_gate_v2(
    lesson: "FailureLesson",
    min_confidence: float,
    max_risk: float,
) -> Tuple[bool, str]:
    confidence = _safe_float(getattr(lesson, "confidence", 0.0), 0.0)
    risk = _safe_float(getattr(lesson, "overgeneralization_risk", 1.0), 1.0)
    concrete_terms = normalize_evidence_terms(
        list(getattr(lesson, "evidence_terms", []) or [])
        + list(getattr(lesson, "applies_if", []) or [])
        + [getattr(lesson, "prefer", ""), getattr(lesson, "avoid", "")]
    )
    combined_text = " ".join([
        str(getattr(lesson, "lesson", "")),
        str(getattr(lesson, "prefer", "")),
        str(getattr(lesson, "avoid", "")),
        " ".join(concrete_terms[:8]),
    ])
    if confidence < min_confidence:
        return False, f"low_confidence:{confidence:.2f}"
    if risk > max_risk:
        return False, f"high_risk:{risk:.2f}"
    if not getattr(lesson, "wrong_item_id", "") or not getattr(lesson, "correct_item_id", ""):
        return False, "missing_wrong_or_correct_item_ids"
    if len(concrete_terms) < 2:
        return False, "too_few_concrete_terms"
    if memory_text_is_too_generic(combined_text):
        return False, "lesson_text_too_generic"
    return True, "accepted"


def make_failure_event_v2(
    user_id: str,
    user_data: Dict[str, Any],
    items_meta: Dict[str, Dict[str, Any]],
    pos_item: PairwiseItemState,
    neg_item: PairwiseItemState,
    explanation: str,
    user_memory_before: str,
    user_memory_after: str,
    max_positive_interactions: Optional[int] = None,
) -> FailureEvent:
    history_ids = user_data.get("train", [])
    if max_positive_interactions and max_positive_interactions > 0:
        history_ids = history_ids[-max_positive_interactions:]
    else:
        history_ids = history_ids[-10:]
    recent_history = _history_item_infos(history_ids, items_meta)
    seed = f"{user_id}|{pos_item.item_id}|{neg_item.item_id}|{hashlib.md5(str(explanation).encode()).hexdigest()[:8]}"
    event_id = hashlib.md5(seed.encode("utf-8")).hexdigest()[:16]
    return FailureEvent(
        event_id=event_id,
        source_user_id=str(user_id),
        recent_history=recent_history,
        user_memory_before=user_memory_before,
        user_memory_after=user_memory_after,
        wrong_item=asdict(neg_item),
        correct_item=asdict(pos_item),
        model_wrong_reasoning=explanation,
        failure_type="wrong_choice_between_positive_and_negative",
    )


def create_failure_lesson_v2(
    memory_system: RecommendationMemorySystem,
    event: FailureEvent,
) -> Optional[FailureLesson]:
    prompt = f"""Convert this failed recommendation event into one compact reusable memory.

Failed event JSON:
{json.dumps(asdict(event), ensure_ascii=False, indent=2)}

Return ONLY valid JSON:
{{
  "lesson": "one sentence: User likes ..., often prefers ..., and not ...",
  "prefer": "short concrete positive preference inferred from correct item/history",
  "avoid": "short concrete negative/superficial signal from wrong item",
  "applies_if": ["specific evidence terms required before using this memory"],
  "do_not_apply_if": ["conditions where this memory should be ignored"],
  "evidence_terms": ["concrete words/phrases from history/candidates that must match"],
  "confidence": 0.0,
  "overgeneralization_risk": 0.0
}}

Rules:
- Preserve the identity of the user, wrong item, and correct item in the reasoning internally, but make the lesson short.
- The lesson must be grounded in the event, not a generic rule.
- If evidence is weak, set confidence low and overgeneralization_risk high.
- Do not say the future user likes something unless it appears in history or the correct item."""

    try:
        raw = memory_system.qwen_generate(
            prompt=prompt,
            role_prompt="You extract reusable failure-correction memories for recommendation.",
            max_new_tokens=900,
            json_mode=True,
            call_type="failure_lesson",
        )
        parsed = extract_json_object(raw)
        lesson_text = str(parsed.get("lesson") or "").strip()
        prefer = str(parsed.get("prefer") or "").strip()
        avoid = str(parsed.get("avoid") or "").strip()
        applies_if = normalize_evidence_terms(parsed.get("applies_if", []))
        do_not_apply_if = normalize_evidence_terms(parsed.get("do_not_apply_if", []))
        evidence_terms = normalize_evidence_terms(parsed.get("evidence_terms", []) + applies_if)
        confidence = max(0.0, min(1.0, _safe_float(parsed.get("confidence"), 0.5)))
        risk = max(0.0, min(1.0, _safe_float(parsed.get("overgeneralization_risk"), 0.5)))
        if not lesson_text:
            lesson_text = f"User likes {prefer}, often prefers it over {avoid}, and not generic category matches."
        if len(normalize_terms(lesson_text)) < 3:
            raise ValueError("lesson too generic")
        memory_id = hashlib.md5(f"{event.event_id}|{lesson_text}".encode("utf-8")).hexdigest()[:16]
        history_item_ids = [str(x.get("item_id")) for x in event.recent_history if x.get("item_id")]
        lesson = FailureLesson(
            memory_id=memory_id,
            source_user_id=event.source_user_id,
            source_event_id=event.event_id,
            lesson=lesson_text[:500],
            prefer=prefer[:250],
            avoid=avoid[:250],
            applies_if=applies_if[:12],
            do_not_apply_if=do_not_apply_if[:12],
            evidence_terms=evidence_terms[:16],
            wrong_item_id=str(event.wrong_item.get("item_id", "")),
            correct_item_id=str(event.correct_item.get("item_id", "")),
            wrong_item_title=str(event.wrong_item.get("title", ""))[:300],
            correct_item_title=str(event.correct_item.get("title", ""))[:300],
            wrong_item_category=str(event.wrong_item.get("category", ""))[:120],
            correct_item_category=str(event.correct_item.get("category", ""))[:120],
            source_user_preference=str(event.user_memory_before or event.user_memory_after or "")[:500],
            history_item_ids=history_item_ids[:20],
            confidence=confidence,
            overgeneralization_risk=risk,
        )
        memory_system._trace("failure_lesson_llm", {
            "event_id": event.event_id,
            "prompt": prompt,
            "answer": raw,
            "parsed": parsed,
            "lesson": asdict(lesson),
        })
        memory_system._trace("failure_lesson_created", {
            "event": asdict(event),
            "lesson": asdict(lesson),
        })
        return lesson
    except Exception as e:
        wrong_title = str(event.wrong_item.get("title", "wrong item"))
        correct_title = str(event.correct_item.get("title", "correct item"))
        history_terms = normalize_evidence_terms([
            x.get("title", "") for x in event.recent_history[:5]
        ] + [correct_title, wrong_title])
        lesson_text = (
            f"User likes signals similar to {correct_title}, often prefers them over "
            f"{wrong_title}, and not superficial category matches."
        )
        memory_id = hashlib.md5(f"{event.event_id}|fallback".encode("utf-8")).hexdigest()[:16]
        lesson = FailureLesson(
            memory_id=memory_id,
            source_user_id=event.source_user_id,
            source_event_id=event.event_id,
            lesson=lesson_text[:500],
            prefer=correct_title[:250],
            avoid=wrong_title[:250],
            applies_if=history_terms[:8],
            do_not_apply_if=[],
            evidence_terms=history_terms[:12],
            wrong_item_id=str(event.wrong_item.get("item_id", "")),
            correct_item_id=str(event.correct_item.get("item_id", "")),
            wrong_item_title=str(event.wrong_item.get("title", ""))[:300],
            correct_item_title=str(event.correct_item.get("title", ""))[:300],
            wrong_item_category=str(event.wrong_item.get("category", ""))[:120],
            correct_item_category=str(event.correct_item.get("category", ""))[:120],
            source_user_preference=str(event.user_memory_before or event.user_memory_after or "")[:500],
            history_item_ids=[str(x.get("item_id")) for x in event.recent_history if x.get("item_id")][:20],
            confidence=0.35,
            overgeneralization_risk=0.75,
        )
        memory_system._trace("failure_lesson_error", {
            "event": asdict(event),
            "error": str(e),
            "fallback_lesson": asdict(lesson),
        })
        return lesson


def train_memory_graph_from_fail_interactions_v2(
    user_id: str,
    user_data: Dict[str, Any],
    negative_data: Optional[Dict[str, Any]],
    items_meta: Dict[str, Dict[str, Any]],
    memory_system: RecommendationMemorySystem,
    user_states: Dict[str, PairwiseUserState],
    item_states: Dict[str, PairwiseItemState],
    graph: MemoryGraphIndex,
    max_iterations: int = 1,
    max_positive_interactions: Optional[int] = None,
    candidate_negative_mode: str = "random",
    min_lesson_confidence: float = 0.25,
    max_lesson_risk: float = 0.85,
    max_failure_lessons_per_user: int = 3,
    training_negative_source: str = "legacy_runtime",
) -> List[FailureLesson]:
    train_items = user_data.get("train", [])
    if max_positive_interactions and max_positive_interactions > 0:
        train_items = train_items[-max_positive_interactions:]
    else:
        train_items = train_items[-30:]
    if not train_items:
        return []

    user_state = get_or_create_user_state(user_states, str(user_id))
    all_item_ids = list(item_states.keys())
    new_lessons: List[FailureLesson] = []
    for pos_item_id in train_items:
        pos_item_id = str(pos_item_id)
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
            user_memory_before = user_state.short_term_memory
            chosen_item_id, explanation = autonomous_pairwise_interaction(
                memory_system=memory_system,
                user_state=user_state,
                pos_item=pos_item,
                neg_item=neg_item,
            )
            memory_system._trace("autonomous_choice_result", {
                "user_id": user_id,
                "positive_item_id": pos_item_id,
                "negative_item_id": neg_item_id,
                "chosen_item_id": chosen_item_id,
                "is_failure": chosen_item_id != pos_item_id,
            })
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
                memory_system._trace("reflection_error", {
                    "user_id": user_id,
                    "positive_item_id": pos_item_id,
                    "negative_item_id": neg_item_id,
                    "error": str(e),
                })
                continue

            event = make_failure_event_v2(
                user_id=str(user_id),
                user_data=user_data,
                items_meta=items_meta,
                pos_item=pos_item,
                neg_item=neg_item,
                explanation=explanation,
                user_memory_before=user_memory_before,
                user_memory_after=user_state.short_term_memory,
                max_positive_interactions=max_positive_interactions,
            )
            memory_system._trace("failure_event_created", asdict(event))
            lesson = create_failure_lesson_v2(memory_system, event)
            if lesson is None:
                continue
            passed_gate, gate_reason = failure_lesson_passes_quality_gate_v2(
                lesson,
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
                "lesson": asdict(lesson),
            })
            if not passed_gate:
                continue
            graph.add_lesson(lesson)
            new_lessons.append(lesson)
            memory_system._trace("global_memory_added", {
                "memory_type": "FailureLesson",
                "lesson": asdict(lesson),
                "graph_edges": {
                    "source_user": lesson.source_user_id,
                    "wrong_item_id": lesson.wrong_item_id,
                    "correct_item_id": lesson.correct_item_id,
                    "history_item_ids": lesson.history_item_ids,
                },
            })
            if max_failure_lessons_per_user > 0 and len(new_lessons) >= max_failure_lessons_per_user:
                memory_system._trace("memory_generation_limit_reached", {
                    "user_id": user_id,
                    "max_failure_lessons_per_user": max_failure_lessons_per_user,
                    "current_count": len(new_lessons),
                })
                return new_lessons
    return new_lessons


# Imported last: these names are only used inside function bodies, and importing them
# at the top would create a circular import between modules.
from fmrec.memory_build import autonomous_pairwise_interaction, choose_training_negative_item_id
from fmrec.variants.legacy_v1 import (
    corrective_pairwise_reflection,
    get_or_create_user_state,
    memory_text_is_too_generic,
)
