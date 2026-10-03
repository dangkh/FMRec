"""Lexicographic temporal-memory reading (temporal_* scopes)."""

from typing import List, Dict, Optional, Any, Tuple

from fmrec.llm_client import RecommendationMemorySystem
from fmrec.prompts import pack_memory_facts
from fmrec.records import GraphRetrievedLesson, UserMemoryProfile
from fmrec.selector import select_memory_facts_with_llm_v2
from fmrec.variants.alt_selectors import select_memory_facts_heuristic


def read_temporal_lexicographic_memory_v2(
    memory_system: RecommendationMemorySystem,
    user_profile: Optional[UserMemoryProfile],
    train_items: List[Dict[str, Any]],
    candidate_items: List[Dict[str, Any]],
    retrieved_lessons: List[GraphRetrievedLesson],
    trace_context: Optional[Dict[str, Any]] = None,
    max_memory_facts: int = 3,
    max_memory_fact_words: int = 55,
    memory_token_budget: int = 420,
    retrieval_audit: Optional[Dict[str, Any]] = None,
    memory_selector: str = "none",
    memory_selector_top_k: int = 3,
    memory_selector_min_relevance: float = 0.70,
    memory_selector_top_m: int = 12,
    memory_selector_neutral_cross: bool = False,
) -> Dict[str, Any]:
    """Render pre-routed temporal evidence without scoring or free-form synthesis."""
    retrieval_audit = dict(retrieval_audit or {})
    candidate_by_id = {
        str(item.get("item_id", "")): item
        for item in candidate_items
        if str(item.get("item_id", ""))
    }
    fact_rows: List[Tuple[str, Dict[str, Any]]] = []
    rejected: List[Dict[str, Any]] = []
    for retrieved in retrieved_lessons:
        lesson = retrieved.lesson
        row = {
            "memory_id": lesson.memory_id,
            "source_user_id": lesson.source_user_id,
            "sources": list(retrieved.sources),
            "paths": list(retrieved.paths),
            "exposure_type": retrieved.exposure_type,
            "candidate_role": retrieved.candidate_role,
            "shared_history_items": list(retrieved.shared_history_items),
            "control_mode": retrieved.control_mode,
            "correct_item_id": lesson.correct_item_id,
            "wrong_item_id": lesson.wrong_item_id,
            "memory_type": lesson.memory_type,
            "creation_mode": lesson.creation_mode,
        }
        if lesson.memory_type not in {
            "temporal_failure_contrast", "cluster_failure_consensus",
        }:
            row["reject_reason"] = "non-temporal memory"
            rejected.append(row)
            continue

        fact = lesson.safe_fact()
        if lesson.memory_type == "cluster_failure_consensus":
            if retrieved.candidate_role not in {"cluster_support", "cluster_avoid"}:
                row["reject_reason"] = "cluster lesson has no current candidate-category match"
                rejected.append(row)
                continue
            rendered = f"[CLUSTER CORRECTION] {fact}"
            row.update({
                "cluster_id": lesson.cluster_id,
                "cluster_support_users": lesson.cluster_support_users,
                "cluster_support_events": lesson.cluster_support_events,
                "direction": "support" if retrieved.candidate_role == "cluster_support" else "avoid",
            })
        elif "same_user" in retrieved.sources:
            rendered = f"[SAME-USER CORRECTION] {fact}"
        elif retrieved.candidate_role in {"observed_next", "base_selected"}:
            candidate_id = (
                str(lesson.observed_next_item_id or lesson.correct_item_id)
                if retrieved.candidate_role == "observed_next"
                else str(lesson.base_selected_item_id or lesson.wrong_item_id)
            )
            candidate = candidate_by_id.get(candidate_id)
            if candidate is None:
                row["reject_reason"] = "cross-user endpoint is not a current candidate"
                rejected.append(row)
                continue
            candidate_title = str(candidate.get("title", candidate_id)).strip() or candidate_id
            direction = "SUPPORT" if retrieved.candidate_role == "observed_next" else "AVOID"
            rendered = f"[COLLABORATIVE {direction}] Candidate '{candidate_title}': {fact}"
            row["candidate_item_id"] = candidate_id
            row["candidate_title"] = candidate_title
            row["direction"] = direction.lower()
        else:
            row["reject_reason"] = "cross-user evidence has no candidate endpoint"
            rejected.append(row)
            continue
        row["safe_fact"] = rendered
        fact_rows.append((rendered, row))

    selector_audit = {"enabled": False}
    pool_size_in = len(fact_rows)
    pool_own_in = sum(1 for _, r in fact_rows if "same_user" in (r.get("sources") or []))
    if memory_selector == "llm" and fact_rows:
        fact_rows, selector_audit = select_memory_facts_with_llm_v2(
            memory_system=memory_system,
            user_profile=user_profile,
            train_items=train_items,
            candidate_items=candidate_items,
            candidate_fact_rows=fact_rows,
            trace_context=trace_context,
            selector_top_k=memory_selector_top_k,
            min_relevance=memory_selector_min_relevance,
            neutral_cross=memory_selector_neutral_cross,
        )
    elif memory_selector == "heuristic" and fact_rows:
        fact_rows, selector_audit = select_memory_facts_heuristic(
            memory_system=memory_system,
            candidate_fact_rows=fact_rows,
            selector_top_k=memory_selector_top_k,
        )
    if isinstance(selector_audit, dict):
        # Real pool size reaching the selector, for the paper: the nominal
        # top_m is an upper bound, not what the user actually had.
        selector_audit.setdefault("pool_size_in", pool_size_in)
        selector_audit.setdefault("pool_own_in", pool_own_in)
        selector_audit.setdefault("pool_cross_in", pool_size_in - pool_own_in)
        selector_audit.setdefault("neutral_cross", bool(memory_selector_neutral_cross))

    used_facts, packing_audit = pack_memory_facts(
        fact_rows,
        max_facts=max_memory_facts,
        max_words=max_memory_fact_words,
        token_budget=memory_token_budget,
    )
    kept_ids = [
        str(row.get("memory_id", ""))
        for row in packing_audit
        if row.get("pack_decision") == "keep"
    ]
    kept_id_set = set(kept_ids)
    selected_rows = [row for _, row in fact_rows if str(row.get("memory_id", "")) in kept_id_set]
    for row in packing_audit:
        if row.get("pack_decision") != "keep":
            rejected_row = dict(row)
            rejected_row["reject_reason"] = "memory pack budget"
            rejected.append(rejected_row)

    if retrieval_audit.get("protocol") == "memcf_j_matched_cf_v1":
        selected_cross_rows = [
            row for row in selected_rows
            if "same_user" not in row.get("sources", [])
        ]
        expected_cross_count = int(retrieval_audit.get("expected_cross_count", 0))
        equal_post_gate_budget = len(selected_cross_rows) == expected_cross_count
        retrieval_audit.update({
            "selected_fact_count_post_pack": len(selected_rows),
            "selected_cross_count_post_pack": len(selected_cross_rows),
            "packed_token_count": sum(
                int(row.get("packed_tokens", 0))
                for row in packing_audit
                if row.get("pack_decision") == "keep"
            ),
            "equal_post_gate_budget": equal_post_gate_budget,
        })
        memory_system.memory_diagnostics["temporal_j_queries"] += 1
        memory_system.memory_diagnostics["temporal_j_matched_eligible"] += int(
            bool(retrieval_audit.get("matched_cf_eligible"))
        )
        memory_system.memory_diagnostics["temporal_j_equal_post_gate_budget"] += int(
            equal_post_gate_budget
        )
        memory_system.memory_diagnostics["temporal_j_selected_cross_facts"] += len(
            selected_cross_rows
        )
        memory_system.memory_diagnostics["temporal_j_exact_replay_filtered"] += int(
            retrieval_audit.get("exact_replay_filtered_before_budget", 0)
        )
        memory_system.memory_diagnostics["temporal_j_retrieval_latency_ms"] += float(
            retrieval_audit.get("retrieval_latency_ms", 0.0)
        )

    result = {
        "use_memory": bool(used_facts),
        "facets": used_facts,
        "memory_facts": used_facts,
        "used_memory_ids": kept_ids,
        "rejected_memory_ids": [str(row.get("memory_id", "")) for row in rejected],
        "selected_rows": selected_rows,
        "rejected_rows": rejected,
        "retrieval_policy": "temporal_lexicographic",
        "retrieval_audit": retrieval_audit,
        "memory_selector": memory_selector,
        "memory_selector_audit": selector_audit,
        "reason": "fixed factual temporal evidence" if used_facts else "no temporal evidence passed routing",
    }
    memory_system.record_memory_diagnostics(
        retrieved=len(retrieved_lessons),
        kept=len(selected_rows),
        skipped=len(rejected),
    )
    memory_system.memory_diagnostics["temporal_memory_users"] += 1
    if any("same_user" not in row.get("sources", []) for row in selected_rows):
        memory_system.memory_diagnostics["temporal_memory_users_with_cross"] += 1
    for row in selected_rows:
        memory_system.memory_diagnostics[f"temporal_exposure_{row.get('exposure_type', 'unknown')}"] += 1
        memory_system.memory_diagnostics[f"temporal_control_{row.get('control_mode', 'true')}"] += 1
    memory_system._trace("temporal_memory_selected", {
        **(trace_context or {}),
        "result": result,
        "packing_audit": packing_audit,
    })
    if retrieval_audit.get("protocol") == "memcf_j_matched_cf_v1":
        memory_system._trace("temporal_treatment_outcome", {
            **(trace_context or {}),
            "audit": retrieval_audit,
            "selected_memory_ids": kept_ids,
        })
    return result
