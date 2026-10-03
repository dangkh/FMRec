"""Memory selectors other than the LLM selector: heuristic and safe-residual."""

import json
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import defaultdict
import re

from fmrec.common import _safe_float, extract_json_object, item_category, normalize_terms
from fmrec.llm_client import RecommendationMemorySystem
from fmrec.prompts import _format_compact_history_lines
from fmrec.records import UserMemoryProfile


def select_memory_facts_heuristic(
    memory_system: RecommendationMemorySystem,
    candidate_fact_rows: List[Tuple[str, Dict[str, Any]]],
    selector_top_k: int = 3,
) -> Tuple[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]]:
    """Rule-based counterpart of select_memory_facts_with_llm_v2: no LLM call,
    and no hand-set weights.

    Each pooled memory is scored by TERM COVERAGE: the fraction of the
    lesson's own concrete terms that appear in the current context (the
    user's history/profile text or the candidate-set text), using the term
    matches the gate loop already computed for every row:

        coverage = |history_matches U candidate_matches| / |concrete_terms|

    It is a single normalised ratio in [0, 1] -- no per-feature weights, no
    special cases -- so a lesson whose evidence is mostly present in the
    current decision ranks above one that only brushes it. Rows with zero
    coverage are dropped: that is the "irrelevant" a filter is meant to
    remove. Ties break on lesson confidence, then memory_id. Same-user and
    cross-user rows are scored identically, matching
    --memory_selector_neutral_cross on the LLM side.
    """
    if not candidate_fact_rows:
        return [], {"enabled": True, "mode": "heuristic", "reason": "no candidate facts"}
    selector_top_k = max(0, int(selector_top_k or 0))
    if selector_top_k == 0:
        return [], {"enabled": True, "mode": "heuristic", "reason": "selector_top_k=0"}

    scored = []
    for safe_fact, row in candidate_fact_rows:
        matched = set(row.get("history_profile_matches") or []) | set(row.get("candidate_matches") or [])
        n_terms = int(row.get("concrete_term_count") or 0)
        coverage = (len(matched) / n_terms) if n_terms > 0 else 0.0
        conf = float(row.get("confidence") or 0.0)
        scored.append((coverage, conf, safe_fact, row))
    scored.sort(key=lambda t: (-t[0], -t[1], str(t[3].get("memory_id", ""))))

    kept: List[Tuple[str, Dict[str, Any]]] = []
    rejected_ids: List[str] = []
    for coverage, conf, safe_fact, row in scored:
        if coverage > 0.0 and len(kept) < selector_top_k:
            row = dict(row)
            row["heuristic_coverage"] = round(coverage, 4)
            kept.append((safe_fact, row))
        else:
            rejected_ids.append(str(row.get("memory_id", "")))

    pool_own = sum(1 for _, r in candidate_fact_rows if "same_user" in (r.get("sources") or []))
    memory_system.memory_diagnostics["memory_selector_calls"] += 1
    memory_system.memory_diagnostics["memory_selector_selected"] += len(kept)
    memory_system.memory_diagnostics["memory_selector_rejected"] += len(rejected_ids)
    for _, r in kept:
        for source in r.get("sources", []):
            memory_system.memory_diagnostics[f"memory_selector_selected_source_{source}"] += 1
    audit = {
        "enabled": True,
        "mode": "heuristic",
        "scoring": "term_coverage",
        "selector_top_k": selector_top_k,
        "input_memory_count": len(candidate_fact_rows),
        "pool_own": pool_own,
        "pool_cross": len(candidate_fact_rows) - pool_own,
        "selected_ids": [str(r.get("memory_id", "")) for _, r in kept],
        "selected_coverage": [r.get("heuristic_coverage") for _, r in kept],
        "rejected_ids": rejected_ids,
        "fallback_used": False,
    }
    return kept, audit


SEMANTIC_CONSENSUS_GENERIC_TERMS = {
    "old", "version", "edition", "standard", "complete", "collection",
    "bundle", "pack", "compatible", "software", "windows", "download",
    "product", "item", "series", "system", "digital", "unknown",
}


def _semantic_consensus_terms(*values: Any) -> Set[str]:
    """Return conservative item-identity terms for cross-user voting."""
    terms: Set[str] = set()
    for value in values:
        for term in normalize_terms(str(value or "")):
            if term in SEMANTIC_CONSENSUS_GENERIC_TERMS or term.isdigit():
                continue
            terms.add(term)
    return terms


def _semantic_candidate_action(
    candidate_by_id: Dict[str, Dict[str, Any]],
    side_item_id: str,
    side_title: str,
    side_category: str,
    min_shared_terms: int,
) -> Optional[Dict[str, Any]]:
    """Ground one past failure side to at most one current candidate.

    Exact IDs remain the strongest path. Semantic transfer requires either
    multiple shared item terms or one distinctive long term, and ambiguous
    ties are rejected rather than broken arbitrarily.
    """
    side_item_id = str(side_item_id or "")
    if side_item_id and side_item_id in candidate_by_id:
        return {
            "candidate_id": side_item_id,
            "match_mode": "exact_item",
            "shared_terms": [],
            "match_strength": 1000,
        }

    side_terms = _semantic_consensus_terms(side_title, side_category)
    if not side_terms:
        return None

    matches: List[Dict[str, Any]] = []
    required = max(1, int(min_shared_terms))
    for candidate_id, item in candidate_by_id.items():
        candidate_terms = _semantic_consensus_terms(
            item.get("title", ""), item_category(item, fallback="")
        )
        shared = sorted(side_terms & candidate_terms)
        distinctive = [term for term in shared if len(term) >= 8]
        if len(shared) < required and not distinctive:
            continue
        matches.append({
            "candidate_id": candidate_id,
            "match_mode": "semantic_item",
            "shared_terms": shared,
            "match_strength": 10 * len(shared) + max((len(x) for x in shared), default=0),
        })
    if not matches:
        return None
    matches.sort(key=lambda row: (-int(row["match_strength"]), str(row["candidate_id"])))
    if len(matches) > 1 and matches[0]["match_strength"] == matches[1]["match_strength"]:
        return None
    return matches[0]


def select_safe_residual_memory_rows(
    rows: List[Dict[str, Any]],
    candidate_items: List[Dict[str, Any]],
    max_memory_facts: int,
    max_same_facts_with_cross: int,
    max_cross_facts: int,
    min_cross_users: int,
    semantic_consensus: bool = False,
    semantic_min_shared_terms: int = 2,
    min_vote_margin: int = 0,
) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build a same-user base with optional consensus-backed cross-user facts.

    Cross-user evidence is admitted only when it has two independent graph
    anchors: an exact candidate failure edge and either a shared-history user
    path or a shared history-item path. Distinct source users must also agree on
    the same candidate and direction. If no cross-user fact qualifies, this
    selector reduces to the same-user path rather than forcing graph noise into
    the ranking prompt.
    """
    max_memory_facts = max(0, int(max_memory_facts))
    max_same_facts_with_cross = max(0, int(max_same_facts_with_cross))
    max_cross_facts = max(0, int(max_cross_facts))
    min_cross_users = max(1, int(min_cross_users))
    semantic_min_shared_terms = max(1, int(semantic_min_shared_terms))
    min_vote_margin = max(0, int(min_vote_margin))
    candidate_by_id = {
        str(item.get("item_id", "")): item
        for item in candidate_items
        if str(item.get("item_id", ""))
    }

    same_rows: List[Dict[str, Any]] = []
    cross_votes: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    rejection_reason: Dict[str, str] = {}

    for row in rows:
        memory_id = str(row.get("memory_id", ""))
        sources = set(str(x) for x in row.get("sources", []))
        if "same_user" in sources:
            if row.get("metadata_noise"):
                rejection_reason[memory_id] = "rejected: noisy same-user memory"
                continue
            same_rows.append(row)
            continue

        has_shared_context_path = bool({"neighbor_user", "history_item"} & sources)
        has_candidate_path = "candidate_item" in sources
        if not has_shared_context_path:
            rejection_reason[memory_id] = "rejected: cross-user residual needs a shared-history path"
            continue
        if not semantic_consensus and not has_candidate_path:
            rejection_reason[memory_id] = "rejected: cross-user residual needs candidate and shared-history paths"
            continue
        if row.get("metadata_noise"):
            rejection_reason[memory_id] = "rejected: noisy cross-user memory"
            continue
        if _safe_float(row.get("overgeneralization_risk"), 1.0) >= 0.85:
            rejection_reason[memory_id] = "rejected: high cross-user overgeneralization risk"
            continue
        if not row.get("strong_history_matches"):
            rejection_reason[memory_id] = "rejected: no concrete current-history match"
            continue

        actions: List[Tuple[str, str, Dict[str, Any]]] = []
        correct_id = str(row.get("correct_item_id", ""))
        wrong_id = str(row.get("wrong_item_id", ""))
        if row.get("direct_correct_candidate") and correct_id in candidate_by_id:
            actions.append((correct_id, "support", {
                "candidate_id": correct_id,
                "match_mode": "exact_item",
                "shared_terms": [],
                "match_strength": 1000,
            }))
        if row.get("direct_wrong_candidate") and wrong_id in candidate_by_id:
            actions.append((wrong_id, "avoid", {
                "candidate_id": wrong_id,
                "match_mode": "exact_item",
                "shared_terms": [],
                "match_strength": 1000,
            }))
        if semantic_consensus:
            if not row.get("direct_correct_candidate"):
                match = _semantic_candidate_action(
                    candidate_by_id,
                    correct_id,
                    str(row.get("correct_item_title", "")),
                    str(row.get("correct_item_category", "")),
                    semantic_min_shared_terms,
                )
                if match:
                    actions.append((str(match["candidate_id"]), "support", match))
            if not row.get("direct_wrong_candidate"):
                match = _semantic_candidate_action(
                    candidate_by_id,
                    wrong_id,
                    str(row.get("wrong_item_title", "")),
                    str(row.get("wrong_item_category", "")),
                    semantic_min_shared_terms,
                )
                if match:
                    actions.append((str(match["candidate_id"]), "avoid", match))
            # A single failure must not vote both directions for one candidate.
            conflicted = {
                candidate_id
                for candidate_id, direction, _ in actions
                if any(c == candidate_id and other != direction for c, other, _ in actions)
            }
            actions = [action for action in actions if action[0] not in conflicted]
        if not actions:
            rejection_reason[memory_id] = (
                "rejected: no unambiguous semantic candidate-side match"
                if semantic_consensus else "rejected: no exact candidate-side failure edge"
            )
            continue
        for candidate_id, direction, match in actions:
            action_row = dict(row)
            action_row["consensus_match_mode"] = match["match_mode"]
            action_row["consensus_match_terms"] = list(match["shared_terms"])
            action_row["consensus_match_strength"] = int(match["match_strength"])
            cross_votes[(candidate_id, direction)].append(action_row)

    consensus_groups: List[Dict[str, Any]] = []
    for (candidate_id, direction), raw_group in cross_votes.items():
        # Count each source user once so repeated lessons cannot manufacture
        # consensus. Keep that source's strongest candidate match.
        best_by_source: Dict[str, Dict[str, Any]] = {}
        for row in raw_group:
            source_user = str(row.get("source_user_id", ""))
            if not source_user:
                continue
            previous = best_by_source.get(source_user)
            if previous is None or (
                int(row.get("consensus_match_strength", 0)),
                _safe_float(row.get("retrieval_score"), 0.0),
            ) > (
                int(previous.get("consensus_match_strength", 0)),
                _safe_float(previous.get("retrieval_score"), 0.0),
            ):
                best_by_source[source_user] = row
        group = list(best_by_source.values())
        source_users = sorted({
            str(row.get("source_user_id", ""))
            for row in group
            if str(row.get("source_user_id", ""))
        })
        opposite_users = {
            str(row.get("source_user_id", ""))
            for row in cross_votes.get(
                (candidate_id, "avoid" if direction == "support" else "support"), []
            )
            if str(row.get("source_user_id", ""))
        }
        if len(source_users) < min_cross_users:
            for row in group:
                rejection_reason[str(row.get("memory_id", ""))] = (
                    f"rejected: cross-user consensus {len(source_users)}/{min_cross_users}"
                )
            continue
        if len(source_users) - len(opposite_users) < min_vote_margin:
            for row in group:
                rejection_reason[str(row.get("memory_id", ""))] = (
                    f"rejected: cross-user direction margin "
                    f"{len(source_users)}-{len(opposite_users)}<{min_vote_margin}"
                )
            continue
        representative = max(
            group,
            key=lambda row: (
                _safe_float(row.get("retrieval_score"), 0.0),
                _safe_float(row.get("confidence"), 0.0),
                str(row.get("memory_id", "")),
            ),
        )
        consensus_groups.append({
            "candidate_id": candidate_id,
            "direction": direction,
            "source_users": source_users,
            "rows": group,
            "representative": representative,
            "strength": len(source_users),
            "opposite_strength": len(opposite_users),
        })

    consensus_groups.sort(key=lambda group: (
        -int(group["strength"]),
        0 if group["direction"] == "support" else 1,
        -_safe_float(group["representative"].get("retrieval_score"), 0.0),
        str(group["candidate_id"]),
    ))
    chosen_groups = consensus_groups[:max_cross_facts]

    selected_pairs: List[Tuple[str, Dict[str, Any]]] = []
    selected_rows: List[Dict[str, Any]] = []
    selected_memory_ids: Set[str] = set()

    # Preserve the complete same-user fallback when no collaborative fact is
    # available; reserve room for cross-user evidence only when it passed all
    # checks above.
    same_budget = max_memory_facts
    if chosen_groups:
        same_budget = min(max_same_facts_with_cross, max(0, max_memory_facts - len(chosen_groups)))
    for original in same_rows[:same_budget]:
        row = dict(original)
        row["accept_reason"] = "safe residual same-user base"
        row["residual_role"] = "same_user_base"
        fact = f"[SAME-USER CORRECTION] {row.get('safe_fact', '')}"
        row["prompt_fact"] = fact
        selected_pairs.append((fact, row))
        selected_rows.append(row)
        selected_memory_ids.add(str(row.get("memory_id", "")))

    for group in chosen_groups:
        if len(selected_pairs) >= max_memory_facts:
            break
        original = group["representative"]
        row = dict(original)
        candidate_id = str(group["candidate_id"])
        item = candidate_by_id.get(candidate_id, {})
        title = re.sub(r"\s+", " ", str(item.get("title", candidate_id))).strip()
        source_count = len(group["source_users"])
        supporting_ids = sorted({str(x.get("memory_id", "")) for x in group["rows"]})
        evidence_label = "semantically related prior failures" if semantic_consensus else "prior failure choices"
        if group["direction"] == "support":
            fact = (
                f"[COLLABORATIVE SUPPORT] {source_count} shared-history users supported "
                f"'{title}' in {evidence_label}. Use only when current history agrees."
            )
        else:
            fact = (
                f"[COLLABORATIVE AVOID] {source_count} shared-history users rejected "
                f"'{title}' in {evidence_label}. Downweight only when current history agrees."
            )
        row.update({
            "accept_reason": "safe residual cross-user consensus",
            "residual_role": f"cross_user_{group['direction']}",
            "consensus_candidate_id": candidate_id,
            "consensus_direction": group["direction"],
            "consensus_source_users": group["source_users"],
            "consensus_source_count": source_count,
            "consensus_opposite_source_count": int(group.get("opposite_strength", 0)),
            "consensus_memory_ids": supporting_ids,
            "semantic_consensus": bool(semantic_consensus),
            "prompt_fact": fact,
        })
        selected_pairs.append((fact, row))
        selected_rows.append(row)
        selected_memory_ids.update(supporting_ids)

    rejected_rows: List[Dict[str, Any]] = []
    for original in rows:
        memory_id = str(original.get("memory_id", ""))
        if memory_id in selected_memory_ids:
            continue
        row = dict(original)
        row["reject_reason"] = rejection_reason.get(
            memory_id,
            "rejected: outside safe residual fact budget",
        )
        rejected_rows.append(row)
    return selected_pairs, selected_rows, rejected_rows


def verify_safe_residual_cross_row(
    memory_system: RecommendationMemorySystem,
    row: Dict[str, Any],
    user_profile: Optional[UserMemoryProfile],
    train_items: List[Dict[str, Any]],
    candidate_items: List[Dict[str, Any]],
    min_confidence: float,
    trace_context: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Use one fail-closed applicability call for an eligible cross residual."""
    candidate_id = str(row.get("consensus_candidate_id", ""))
    candidate = next(
        (item for item in candidate_items if str(item.get("item_id", "")) == candidate_id),
        {},
    )
    payload = {
        "direction": row.get("consensus_direction"),
        "candidate_id": candidate_id,
        "candidate_title": candidate.get("title", ""),
        "candidate_category": item_category(candidate, fallback="Unknown") if candidate else "Unknown",
        "source_user_count": row.get("consensus_source_count", 0),
        "opposite_source_count": row.get("consensus_opposite_source_count", 0),
        "semantic_match_terms": row.get("consensus_match_terms", []),
        "current_history_matches": row.get("strong_history_matches", []),
    }
    profile_payload = {
        "profile": user_profile.profile if user_profile else "",
        "facets": list(user_profile.facets if user_profile else []),
    }
    prompt = f"""
You are a conservative applicability verifier for collaborative recommendation evidence.

Target user profile:
{json.dumps(profile_payload, ensure_ascii=False, separators=(",", ":"))}

Recent positive history:
{_format_compact_history_lines(train_items[-10:])}

Proposed collaborative residual:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}

Decision policy:
- APPLY only if the named candidate and direction are concretely relevant to this target user's history.
- Source-user agreement alone is insufficient.
- REJECT generic category, popularity, edition/version, or weak lexical matches.
- REJECT if evidence conflicts with the target user's own history.
- Do not infer new preferences and do not predict the hidden correct item.

Return only JSON:
{{"apply": false, "confidence": 0.0, "reason": "short grounded reason"}}
"""
    audit: Dict[str, Any] = {
        "enabled": True,
        "candidate_id": candidate_id,
        "min_confidence": float(min_confidence),
        "payload": payload,
    }
    memory_system.memory_diagnostics["safe_residual_verifier_calls"] += 1
    try:
        raw = memory_system.qwen_generate(
            prompt=prompt,
            role_prompt="Verify cross-user recommendation evidence conservatively. Return JSON only.",
            max_new_tokens=240,
            json_mode=True,
            call_type="safe_residual_verifier",
        )
        parsed = extract_json_object(raw)
        apply_value = parsed.get("apply", False)
        if isinstance(apply_value, str):
            apply_value = apply_value.strip().lower() in {"true", "yes", "apply", "1"}
        confidence = max(0.0, min(1.0, _safe_float(parsed.get("confidence"), 0.0)))
        accepted = bool(apply_value) and confidence >= float(min_confidence)
        audit.update({
            "raw_answer": raw,
            "parsed": parsed,
            "accepted": accepted,
            "confidence": confidence,
            "reason": str(parsed.get("reason", ""))[:300],
        })
        key = "safe_residual_verifier_accepted" if accepted else "safe_residual_verifier_rejected"
        memory_system.memory_diagnostics[key] += 1
    except Exception as exc:
        accepted = False
        audit.update({"accepted": False, "error": str(exc), "fail_closed": True})
        memory_system.memory_diagnostics["safe_residual_verifier_errors"] += 1
        memory_system.memory_diagnostics["safe_residual_verifier_rejected"] += 1
    memory_system._trace("safe_residual_verifier", {
        **(trace_context or {}),
        "prompt": prompt,
        "audit": audit,
    })
    return accepted, audit
