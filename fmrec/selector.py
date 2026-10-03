"""LLM memory selector and lesson-to-prompt-fact reading."""

import json
from typing import List, Dict, Optional, Any, Set, Tuple

from fmrec.common import (
    _safe_float,
    extract_json_object,
    has_metadata_noise,
    normalize_evidence_terms,
    term_matches_context,
)
from fmrec.llm_client import RecommendationMemorySystem
from fmrec.prompts import _context_text_from_items, pack_memory_facts
from fmrec.records import GraphRetrievedLesson, UserMemoryProfile
from fmrec.variants.alt_selectors import (
    select_memory_facts_heuristic,
    select_safe_residual_memory_rows,
    verify_safe_residual_cross_row,
)


def select_memory_facts_with_llm_v2(
    memory_system: RecommendationMemorySystem,
    user_profile: Optional[UserMemoryProfile],
    train_items: List[Dict[str, Any]],
    candidate_items: List[Dict[str, Any]],
    candidate_fact_rows: List[Tuple[str, Dict[str, Any]]],
    trace_context: Optional[Dict[str, Any]] = None,
    selector_top_k: int = 3,
    min_relevance: float = 0.60,
    neutral_cross: bool = False,
) -> Tuple[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]]:
    """Select applicable memory facts with an LLM without ranking items.

    `neutral_cross=True` removes rule 5 (cross-user memories need stronger
    evidence): with it on, the selector would pre-reject the very cross-user
    lessons a pooling experiment is trying to evaluate.
    """
    if not candidate_fact_rows:
        return [], {"enabled": True, "reason": "no candidate facts"}

    selector_top_k = max(0, int(selector_top_k or 0))
    if selector_top_k == 0:
        return [], {"enabled": True, "reason": "selector_top_k=0"}

    memory_by_id: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    memory_rows = []
    for safe_fact, row in candidate_fact_rows:
        memory_id = str(row.get("memory_id", "")).strip()
        if not memory_id or memory_id in memory_by_id:
            continue
        memory_by_id[memory_id] = (safe_fact, row)
        memory_rows.append({
            "memory_id": memory_id,
            "sources": row.get("sources", []),
            "memory_fact": safe_fact,
            "history_profile_matches": row.get("history_profile_matches", [])[:8],
            "candidate_matches": row.get("candidate_matches", [])[:8],
            "candidate_support": bool(row.get("candidate_support")),
            "direct_correct_candidate": bool(row.get("direct_correct_candidate")),
            "direct_wrong_candidate": bool(row.get("direct_wrong_candidate")),
            "correct_item_title": row.get("correct_item_title", ""),
            "wrong_item_title": row.get("wrong_item_title", ""),
            "confidence": row.get("confidence"),
            "overgeneralization_risk": row.get("overgeneralization_risk"),
        })

    history_payload = [
        {
            "title": str(item.get("title", ""))[:160],
            "category": str(item.get("category", "Unknown"))[:100],
        }
        for item in train_items[-10:]
    ]
    candidate_payload = [
        {
            "candidate_id": str(item.get("item_id", "")),
            "title": str(item.get("title", ""))[:180],
            "category": str(item.get("category", "Unknown"))[:100],
        }
        for item in candidate_items
    ]
    profile_payload = None
    if user_profile is not None:
        profile_payload = {
            "profile": str(user_profile.profile or "")[:600],
            "facets": list(user_profile.facets or [])[:8],
        }
    cross_rule = (
        "Judge same-user and cross-user memories by the same evidence standard; the source of a memory is not a reason to prefer or reject it."
        if neutral_cross else
        "Cross-user memories require stronger evidence than same-user memories."
    )

    prompt = f"""You are a memory applicability selector for a recommender system.

Your task is NOT to rank candidate items.
Your task is ONLY to decide which past failure-memory facts are applicable to the current user history and candidate set.

Select a memory only if all conditions hold:
1. It matches concrete signals in the user's history/profile.
2. It helps distinguish at least one current candidate from another.
3. It is specific, not a generic category match.
4. It is not only about a past wrong item unless it also points to a better current alternative.
5. {cross_rule}
6. Do not select a memory solely because it mentions broad words like music, game, beauty, product, item, unknown, CD, vinyl, album, or category.

Return JSON only with this schema:
{{
  "selected": [
    {{
      "memory_id": "string",
      "relevance": 0.0,
      "reason": "short reason",
      "supported_candidate_ids": ["candidate_id"]
    }}
  ],
  "rejected": [
    {{
      "memory_id": "string",
      "reason": "short reason"
    }}
  ]
}}

Rules:
- Select at most {selector_top_k} memories.
- Use relevance from 0.0 to 1.0.
- Only select memories with relevance >= {min_relevance:.2f}.
- If no memory is clearly applicable, return an empty selected list.

User profile:
{json.dumps(profile_payload, ensure_ascii=False)}

User recent history:
{json.dumps(history_payload, ensure_ascii=False)}

Candidate items:
{json.dumps(candidate_payload, ensure_ascii=False)}

Memory candidates:
{json.dumps(memory_rows, ensure_ascii=False)}
"""

    audit: Dict[str, Any] = {
        "enabled": True,
        "selector_top_k": selector_top_k,
        "min_relevance": min_relevance,
        "input_memory_count": len(memory_rows),
        "fallback_used": False,
    }
    try:
        raw = memory_system.qwen_generate(
            prompt=prompt,
            role_prompt="You select applicable recommendation memories. Return JSON only.",
            max_new_tokens=900,
            json_mode=True,
            call_type="memory_selector",
        )
        parsed = extract_json_object(raw)
        selected = parsed.get("selected", [])
        if not isinstance(selected, list):
            selected = []
        selected_rows: List[Tuple[str, Dict[str, Any]]] = []
        seen: Set[str] = set()
        selector_selected_payload = []
        selector_rejected_payload = parsed.get("rejected", [])
        if not isinstance(selector_rejected_payload, list):
            selector_rejected_payload = []

        for entry in selected:
            if not isinstance(entry, dict):
                continue
            memory_id = str(entry.get("memory_id", "")).strip()
            if not memory_id or memory_id in seen or memory_id not in memory_by_id:
                continue
            relevance = max(0.0, min(1.0, _safe_float(entry.get("relevance"), 0.0)))
            if relevance < min_relevance:
                continue
            safe_fact, row = memory_by_id[memory_id]
            row["selector_relevance"] = relevance
            row["selector_reason"] = str(entry.get("reason", ""))[:300]
            row["selector_supported_candidate_ids"] = [
                str(x) for x in (entry.get("supported_candidate_ids", []) or [])[:8]
            ]
            selected_rows.append((safe_fact, row))
            selector_selected_payload.append(entry)
            seen.add(memory_id)
            if len(selected_rows) >= selector_top_k:
                break

        audit.update({
            "raw_answer": raw,
            "parsed": parsed,
            "selected_memory_ids": [row[1].get("memory_id") for row in selected_rows],
            "selected_count": len(selected_rows),
            "rejected_count": max(0, len(memory_rows) - len(selected_rows)),
            "selector_selected": selector_selected_payload,
            "selector_rejected": selector_rejected_payload,
        })
        memory_system.memory_diagnostics["memory_selector_calls"] += 1
        memory_system.memory_diagnostics["memory_selector_selected"] += len(selected_rows)
        memory_system.memory_diagnostics["memory_selector_rejected"] += max(0, len(memory_rows) - len(selected_rows))
        for _, row in selected_rows:
            for source in row.get("sources", []) or []:
                memory_system.memory_diagnostics[f"memory_selector_selected_source_{source}"] += 1
        memory_system._trace("memory_selector_llm", {
            **(trace_context or {}),
            "prompt": prompt,
            "answer": raw,
            "audit": audit,
        })
        memory_system._trace("memory_selector_decision", {
            **(trace_context or {}),
            "audit": audit,
            "input_rows": memory_rows,
        })
        return selected_rows, audit
    except Exception as e:
        # Fail open to deterministic memory selection so long runs do not crash.
        fallback_rows = candidate_fact_rows[:selector_top_k]
        audit.update({
            "fallback_used": True,
            "error": str(e),
            "selected_memory_ids": [row.get("memory_id") for _, row in fallback_rows],
            "selected_count": len(fallback_rows),
        })
        memory_system.memory_diagnostics["memory_selector_calls"] += 1
        memory_system.memory_diagnostics["memory_selector_errors"] += 1
        memory_system.memory_diagnostics["memory_selector_fallbacks"] += 1
        memory_system._trace("memory_selector_error", {
            **(trace_context or {}),
            "prompt": prompt,
            "error": str(e),
            "audit": audit,
        })
        return fallback_rows, audit


def read_graph_lessons_as_facets_v2(
    memory_system: RecommendationMemorySystem,
    user_profile: Optional[UserMemoryProfile],
    train_items: List[Dict[str, Any]],
    candidate_items: List[Dict[str, Any]],
    retrieved_lessons: List[GraphRetrievedLesson],
    trace_context: Optional[Dict[str, Any]] = None,
    max_memory_facts: int = 3,
    max_memory_fact_words: int = 55,
    memory_token_budget: int = 420,
    strict_candidate_applicability: bool = False,
    min_candidate_matches: int = 1,
    allow_same_user_without_candidate_match: bool = True,
    allow_random_memory_injection: bool = False,
    reject_wrong_only_memory: bool = False,
    memory_selector: str = "none",
    memory_selector_top_k: int = 3,
    memory_selector_min_relevance: float = 0.60,
    memory_selector_top_m: int = 12,
    memory_selector_neutral_cross: bool = False,
    safe_residual_mode: bool = False,
    safe_residual_min_cross_users: int = 2,
    safe_residual_max_cross_facts: int = 1,
    safe_residual_max_same_facts: int = 2,
    safe_residual_semantic_consensus: bool = False,
    safe_residual_semantic_min_terms: int = 2,
    safe_residual_min_vote_margin: int = 0,
    safe_residual_verify_cross: bool = False,
    safe_residual_verify_min_confidence: float = 0.65,
) -> Dict[str, Any]:
    """Select safe factual memory snippets without another LLM call.

    The evaluation prompt receives facts only, not a regenerated analytical
    memory. This is intentionally conservative for Qwen-sized models:
    - same_user memories can be used directly;
    - candidate_item memories need evidence in the current user's history/profile;
    - neighbor_user/history_item memories also need history/profile evidence.
    """
    if not retrieved_lessons:
        return {
            "use_memory": False,
            "facets": [],
            "memory_facts": [],
            "used_memory_ids": [],
            "rejected_memory_ids": [],
            "reason": "no graph lessons retrieved",
        }

    history_profile_text = (
        _context_text_from_items(train_items[-10:])
        + " "
        + (user_profile.profile.lower() if user_profile else "")
        + " "
        + " ".join(user_profile.facets if user_profile else []).lower()
    )
    candidate_text = _context_text_from_items(candidate_items)
    candidate_id_set = {
        str(item.get("item_id", ""))
        for item in candidate_items
        if item.get("item_id") is not None
    }

    candidate_fact_rows: List[Tuple[str, Dict[str, Any]]] = []
    used_ids: List[str] = []
    rejected: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    weak_gate_terms = {
        "like", "likes", "liked", "specific", "history", "brands", "brand",
        "shows", "show", "interest", "interests", "preference", "preferences",
        "mentions", "includes", "recent", "particular", "frequently", "engages",
        "beauty", "product", "products", "category", "item", "items",
    }
    if strict_candidate_applicability:
        weak_gate_terms.update({
            "music", "album", "albums", "song", "songs", "track", "tracks",
            "game", "games", "cd", "vinyl", "unknown", "various",
        })

    for r in retrieved_lessons:
        lesson = r.lesson
        evidence_terms = normalize_evidence_terms(
            list(lesson.evidence_terms or [])
            + list(lesson.applies_if or [])
            + [lesson.prefer, lesson.correct_item_title, lesson.correct_item_category]
        )
        concrete_terms = [
            term for term in evidence_terms
            if term not in weak_gate_terms and len(str(term).strip()) >= 4
        ]
        history_matches = [
            term for term in concrete_terms
            if term_matches_context(term, history_profile_text)
        ]
        candidate_matches = [
            term for term in concrete_terms
            if term_matches_context(term, candidate_text)
        ]
        strong_history_matches = [
            term for term in history_matches
            if (" " in term or len(term) >= 6)
        ]
        same_user = "same_user" in r.sources
        consensus_verified_path = "consensus_verified" in r.sources
        candidate_item_path = "candidate_item" in r.sources
        neighbor_user_path = "neighbor_user" in r.sources
        history_item_path = "history_item" in r.sources
        cluster_user_path = "cluster_user" in r.sources
        random_memory_path = "random_memory" in r.sources or "random_memory_clean" in r.sources or "random_cluster" in r.sources
        shuffled_memory_path = "shuffled_memory" in r.sources or "shuffled_memory_clean" in r.sources or "shuffled_cluster" in r.sources
        strong_graph_path = same_user or history_item_path or cluster_user_path
        direct_correct_candidate = str(lesson.correct_item_id or "") in candidate_id_set
        direct_wrong_candidate = str(lesson.wrong_item_id or "") in candidate_id_set
        direct_candidate_match = direct_correct_candidate or direct_wrong_candidate or candidate_item_path
        candidate_support = direct_candidate_match or len(candidate_matches) >= max(1, min_candidate_matches)
        safe_fact_text = lesson.safe_fact()
        noisy_fact = has_metadata_noise(safe_fact_text)

        row = {
            "memory_id": lesson.memory_id,
            "source_user_id": lesson.source_user_id,
            "safe_fact": safe_fact_text,
            "sources": r.sources,
            "paths": r.paths[:5],
            "retrieval_score": r.score,
            "history_profile_matches": history_matches,
            "strong_history_matches": strong_history_matches,
            "candidate_matches": candidate_matches,
            "concrete_term_count": len(concrete_terms),
            "candidate_support": candidate_support,
            "direct_candidate_match": direct_candidate_match,
            "direct_correct_candidate": direct_correct_candidate,
            "direct_wrong_candidate": direct_wrong_candidate,
            "confidence": lesson.confidence,
            "overgeneralization_risk": lesson.overgeneralization_risk,
            "correct_item_id": lesson.correct_item_id,
            "wrong_item_id": lesson.wrong_item_id,
            "correct_item_title": lesson.correct_item_title,
            "wrong_item_title": lesson.wrong_item_title,
            "correct_item_category": lesson.correct_item_category,
            "wrong_item_category": lesson.wrong_item_category,
            "failure_type": lesson.failure_type,
            "prefer": lesson.prefer,
            "avoid": lesson.avoid,
            "metadata_noise": noisy_fact,
        }
        all_rows.append(row)

        # Safe-residual selection needs the complete candidate pool to compute
        # distinct-source consensus. Defer all acceptance and packing until the
        # loop has collected every retrieved row.
        if safe_residual_mode:
            continue

        # Main safety rule: avoid candidate-only activation. If strict mode is
        # enabled, memory facts must be applicable to the current candidate set,
        # not merely similar to the user's historical context. This is designed
        # to reduce random/shuffled-memory gains and broad history-item noise.
        accept = False
        reason = ""
        if strict_candidate_applicability:
            if random_memory_path and allow_random_memory_injection:
                accept = True
                reason = "random memory injected control"
            elif shuffled_memory_path and allow_random_memory_injection:
                accept = True
                reason = "shuffled memory injected control"
            elif consensus_verified_path:
                accept = True
                reason = "multi-user consensus-verified cross-user match"
            elif same_user and (allow_same_user_without_candidate_match or candidate_support):
                accept = True
                reason = "same_user memory under strict candidate gate"
            elif candidate_item_path and strong_history_matches and candidate_support:
                accept = True
                reason = "candidate_item path plus user-history and candidate evidence"
            elif neighbor_user_path and strong_history_matches and candidate_support:
                accept = True
                reason = "neighbor_user path plus user-history and candidate evidence"
            elif history_item_path and strong_history_matches and candidate_support and len(candidate_matches) >= max(1, min_candidate_matches):
                accept = True
                reason = "history_item path plus strong candidate evidence"
            elif cluster_user_path and strong_history_matches and candidate_support:
                accept = True
                reason = "cluster_user path plus user-history and candidate evidence"
            else:
                reason = "rejected: strict gate requires user-history evidence and current-candidate support"
        else:
            if same_user:
                accept = True
                reason = "same_user memory"
            elif consensus_verified_path:
                accept = True
                reason = "multi-user consensus-verified cross-user match"
            elif candidate_item_path and strong_history_matches:
                accept = True
                reason = "candidate_item path plus strong user-history evidence"
            elif strong_graph_path and strong_history_matches:
                accept = True
                reason = "history/neighbor/cluster graph path plus strong user-history evidence"
            else:
                reason = "rejected: no strong current user-history evidence"

        if lesson.overgeneralization_risk >= 0.85 and not same_user:
            accept = False
            reason = "rejected: high overgeneralization risk"
        if strict_candidate_applicability and noisy_fact and not same_user:
            accept = False
            reason = "rejected: metadata noise under strict candidate gate"
        if reject_wrong_only_memory and direct_wrong_candidate and not direct_correct_candidate:
            accept = False
            reason = "rejected: memory only matches a past wrong item in current candidates"

        if accept:
            used_ids.append(lesson.memory_id)
            safe_fact = safe_fact_text
            candidate_fact_rows.append((safe_fact, row))
            row["accept_reason"] = reason
            selected_rows.append(row)
            memory_system.memory_diagnostics["selected_memory_facts_total"] += 1
            if direct_wrong_candidate and not direct_correct_candidate:
                memory_system.memory_diagnostics["selected_wrong_only_memory_facts"] += 1
            if has_metadata_noise(safe_fact):
                memory_system.memory_diagnostics["selected_memory_facts_noisy"] += 1
            for source in r.sources:
                memory_system.memory_diagnostics[f"selected_source_{source}"] += 1
        else:
            row["reject_reason"] = reason
            rejected.append(row)
            memory_system.memory_diagnostics["rejected_memory_facts_total"] += 1
            if direct_wrong_candidate and not direct_correct_candidate:
                memory_system.memory_diagnostics["rejected_wrong_only_memory_facts"] += 1
            for source in r.sources:
                memory_system.memory_diagnostics[f"rejected_source_{source}"] += 1
        overflow_limit = max_memory_facts * 2
        if memory_selector in {"llm", "heuristic"}:
            # The selector must see the whole pool. This used to be
            # memory_selector_top_k * 2 (= 6), which silently truncated any
            # pool larger than six rows before selection ever ran.
            overflow_limit = max(overflow_limit, int(memory_selector_top_m or 0) * 2)
        if not safe_residual_mode and overflow_limit > 0 and len(candidate_fact_rows) >= overflow_limit:
            # Keep a small overflow buffer for token-budget packing.
            break

    if safe_residual_mode:
        candidate_fact_rows, selected_rows, rejected = select_safe_residual_memory_rows(
            rows=all_rows,
            candidate_items=candidate_items,
            max_memory_facts=max_memory_facts,
            max_same_facts_with_cross=safe_residual_max_same_facts,
            max_cross_facts=safe_residual_max_cross_facts,
            min_cross_users=safe_residual_min_cross_users,
            semantic_consensus=safe_residual_semantic_consensus,
            semantic_min_shared_terms=safe_residual_semantic_min_terms,
            min_vote_margin=safe_residual_min_vote_margin,
        )
        verifier_audits: List[Dict[str, Any]] = []
        provisional_cross_rows = [
            row for _, row in candidate_fact_rows
            if str(row.get("residual_role", "")).startswith("cross_user_")
        ]
        if safe_residual_verify_cross and provisional_cross_rows:
            all_cross_accepted = True
            for row in provisional_cross_rows:
                accepted, audit = verify_safe_residual_cross_row(
                    memory_system=memory_system,
                    row=row,
                    user_profile=user_profile,
                    train_items=train_items,
                    candidate_items=candidate_items,
                    min_confidence=safe_residual_verify_min_confidence,
                    trace_context=trace_context,
                )
                row["cross_verifier"] = audit
                verifier_audits.append(audit)
                all_cross_accepted = all_cross_accepted and accepted
            if not all_cross_accepted:
                verifier_by_memory_id = {
                    str(row.get("memory_id", "")): row.get("cross_verifier", {})
                    for row in provisional_cross_rows
                }
                # Restore the full same-user fallback instead of leaving a
                # reserved cross slot empty after verifier rejection.
                candidate_fact_rows, selected_rows, rejected = select_safe_residual_memory_rows(
                    rows=all_rows,
                    candidate_items=candidate_items,
                    max_memory_facts=max_memory_facts,
                    max_same_facts_with_cross=safe_residual_max_same_facts,
                    max_cross_facts=0,
                    min_cross_users=safe_residual_min_cross_users,
                    semantic_consensus=safe_residual_semantic_consensus,
                    semantic_min_shared_terms=safe_residual_semantic_min_terms,
                    min_vote_margin=safe_residual_min_vote_margin,
                )
                for row in rejected:
                    memory_id = str(row.get("memory_id", ""))
                    if memory_id in verifier_by_memory_id:
                        row["reject_reason"] = "rejected: cross-user applicability verifier"
                        row["cross_verifier"] = verifier_by_memory_id[memory_id]
        used_ids = []
        memory_system.memory_diagnostics["safe_residual_users"] += 1
        cross_rows = [
            row for _, row in candidate_fact_rows
            if str(row.get("residual_role", "")).startswith("cross_user_")
        ]
        same_rows = [
            row for _, row in candidate_fact_rows
            if row.get("residual_role") == "same_user_base"
        ]
        if cross_rows:
            memory_system.memory_diagnostics["safe_residual_users_with_cross"] += 1
        memory_system.memory_diagnostics["safe_residual_same_facts"] += len(same_rows)
        memory_system.memory_diagnostics["safe_residual_cross_facts"] += len(cross_rows)
        memory_system.memory_diagnostics["safe_residual_cross_source_users"] += sum(
            int(row.get("consensus_source_count", 0)) for row in cross_rows
        )
        for _, row in candidate_fact_rows:
            memory_system.memory_diagnostics["selected_memory_facts_total"] += 1
            if row.get("direct_wrong_candidate") and not row.get("direct_correct_candidate"):
                memory_system.memory_diagnostics["selected_wrong_only_memory_facts"] += 1
            for source in row.get("sources", []):
                memory_system.memory_diagnostics[f"selected_source_{source}"] += 1
        for row in rejected:
            memory_system.memory_diagnostics["rejected_memory_facts_total"] += 1
            if row.get("direct_wrong_candidate") and not row.get("direct_correct_candidate"):
                memory_system.memory_diagnostics["rejected_wrong_only_memory_facts"] += 1
            for source in row.get("sources", []):
                memory_system.memory_diagnostics[f"rejected_source_{source}"] += 1

    selector_audit = {"enabled": False}
    deterministic_selected_rows = list(selected_rows)
    pool_size_in = len(candidate_fact_rows)
    pool_own_in = sum(1 for _, r in candidate_fact_rows if "same_user" in (r.get("sources") or []))
    if memory_selector in {"llm", "heuristic"} and candidate_fact_rows:
        if memory_selector == "llm":
            selected_by_selector, selector_audit = select_memory_facts_with_llm_v2(
                memory_system=memory_system,
                user_profile=user_profile,
                train_items=train_items,
                candidate_items=candidate_items,
                candidate_fact_rows=candidate_fact_rows,
                trace_context=trace_context,
                selector_top_k=memory_selector_top_k,
                min_relevance=memory_selector_min_relevance,
                neutral_cross=memory_selector_neutral_cross,
            )
        else:
            selected_by_selector, selector_audit = select_memory_facts_heuristic(
                memory_system=memory_system,
                candidate_fact_rows=candidate_fact_rows,
                selector_top_k=memory_selector_top_k,
            )
        selected_ids = {str(row.get("memory_id", "")) for _, row in selected_by_selector}
        selector_rejected_rows = []
        for _, row in candidate_fact_rows:
            if str(row.get("memory_id", "")) in selected_ids:
                continue
            rejected_row = dict(row)
            rejected_row["reject_reason"] = f"rejected by {memory_selector} memory selector"
            selector_rejected_rows.append(rejected_row)
        rejected.extend(selector_rejected_rows)
        candidate_fact_rows = selected_by_selector
        selected_rows = [row for _, row in candidate_fact_rows]
    if isinstance(selector_audit, dict):
        # Real pool size that reached the selector -- the nominal top_m is an
        # upper bound, not what this user actually had.
        selector_audit.setdefault("pool_size_in", pool_size_in)
        selector_audit.setdefault("pool_own_in", pool_own_in)
        selector_audit.setdefault("pool_cross_in", pool_size_in - pool_own_in)
        selector_audit.setdefault("neutral_cross", bool(memory_selector_neutral_cross))

    used_facts, packing_audit = pack_memory_facts(
        candidate_fact_rows,
        max_facts=max_memory_facts,
        max_words=max_memory_fact_words,
        token_budget=memory_token_budget,
    )
    used_ids = [row.get("memory_id", "") for row in packing_audit if row.get("pack_decision") == "keep"]

    result = {
        "use_memory": bool(used_facts),
        # Keep key name `facets` for backward compatibility with existing code,
        # but the content is now factual memory snippets, not model-generated facets.
        "facets": used_facts,
        "memory_facts": used_facts,
        "used_memory_ids": used_ids,
        "rejected_memory_ids": [x["memory_id"] for x in rejected],
        "selected_rows": selected_rows,
        "rejected_rows": rejected,
        "strict_candidate_applicability": strict_candidate_applicability,
        "min_candidate_matches": min_candidate_matches,
        "allow_same_user_without_candidate_match": allow_same_user_without_candidate_match,
        "allow_random_memory_injection": allow_random_memory_injection,
        "reject_wrong_only_memory": reject_wrong_only_memory,
        "memory_selector": memory_selector,
        "safe_residual_mode": safe_residual_mode,
        "safe_residual_min_cross_users": safe_residual_min_cross_users,
        "safe_residual_max_cross_facts": safe_residual_max_cross_facts,
        "safe_residual_max_same_facts": safe_residual_max_same_facts,
        "safe_residual_semantic_consensus": safe_residual_semantic_consensus,
        "safe_residual_semantic_min_terms": safe_residual_semantic_min_terms,
        "safe_residual_min_vote_margin": safe_residual_min_vote_margin,
        "safe_residual_verify_cross": safe_residual_verify_cross,
        "safe_residual_verify_min_confidence": safe_residual_verify_min_confidence,
        "safe_residual_verifier_audits": verifier_audits if safe_residual_mode else [],
        "memory_selector_audit": selector_audit,
        "deterministic_selected_rows": deterministic_selected_rows,
        "reason": "deterministic safe factual memory selection" if used_facts else "no safe factual memory passed history evidence gate",
    }
    memory_system._trace("memory_facts_selected", {
        **(trace_context or {}),
        "result": result,
        "selected_rows": selected_rows,
        "rejected_rows": rejected,
        "retrieved_graph_lessons": all_rows,
        "history_profile_text": history_profile_text[:2000],
        "packing_audit": packing_audit,
        "max_memory_facts": max_memory_facts,
        "max_memory_fact_words": max_memory_fact_words,
        "memory_token_budget": memory_token_budget,
    })
    return result
