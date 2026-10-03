"""Typed failure-constraint re-scoring (--failure_constraint_mode)."""

from typing import List, Dict, Optional, Any, Set, Tuple
from collections import defaultdict

from fmrec.common import _safe_float


FAILURE_CONSTRAINT_MODES = {
    "none",
    "same_exact",
    "cross_exact",
    "full_partitioned",
    "full_consensus",
    "polarity_swapped",
    "shuffled_provenance",
    "popularity",
    "cf_shared_cross",
    "cf_same_plus_shared",
    "cf_shuffled_neighbors",
    "cf_random_neighbors",
    "cf_polarity_swapped",
    "g_true_neighbor",
    "g_shuffled_graph",
    "g_random_neighbor",
    "g_matched_random",
}


def aggregate_typed_failure_constraints(
    evidence_rows: List[Dict[str, Any]],
    candidate_item_ids: List[str],
    mode: str,
    min_cross_support: int = 2,
    candidate_popularity: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate exact failure edges by candidate and distinct source user."""
    mode = str(mode or "none").strip().lower()
    if mode not in FAILURE_CONSTRAINT_MODES:
        raise ValueError(f"Unsupported failure_constraint_mode={mode}")

    candidate_set = set(str(x) for x in candidate_item_ids)
    buckets: Dict[str, Dict[str, Set[str]]] = {
        item_id: {
            "same_positive": set(),
            "same_negative": set(),
            "cross_positive": set(),
            "cross_negative": set(),
        }
        for item_id in candidate_item_ids
    }
    evidence_by_candidate: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    swap_polarity = mode in {"polarity_swapped", "cf_polarity_swapped"}
    for row in evidence_rows:
        candidate_id = str(row.get("candidate_item_id", ""))
        if candidate_id not in candidate_set:
            continue
        role = str(row.get("edge_role", ""))
        if role not in {"preferred", "wrong"}:
            continue
        if swap_polarity:
            role = "wrong" if role == "preferred" else "preferred"
        owner = "same" if bool(row.get("same_user")) else "cross"
        direction = "positive" if role == "preferred" else "negative"
        source_user_id = str(row.get("source_user_id", "")) or str(row.get("memory_id", ""))
        buckets[candidate_id][f"{owner}_{direction}"].add(source_user_id)
        evidence_by_candidate[candidate_id].append({**row, "effective_edge_role": role})

    result: Dict[str, Dict[str, Any]] = {}
    for candidate_id in candidate_item_ids:
        bucket = buckets[candidate_id]
        same_net = len(bucket["same_positive"]) - len(bucket["same_negative"])
        cross_positive = len(bucket["cross_positive"])
        cross_negative = len(bucket["cross_negative"])
        cross_net = cross_positive - cross_negative
        cross_support = max(cross_positive, cross_negative)
        if mode in {
            "full_consensus", "cf_shared_cross", "cf_same_plus_shared",
            "cf_shuffled_neighbors", "cf_random_neighbors", "cf_polarity_swapped",
            "g_true_neighbor", "g_shuffled_graph", "g_random_neighbor",
            "g_matched_random",
        }:
            # Strict source-user consensus: repeated lessons from one prolific
            # user count once; any polarity conflict disables cross-user action.
            threshold = max(1, int(min_cross_support))
            if cross_positive >= threshold and cross_negative == 0:
                cross_net = cross_positive
            elif cross_negative >= threshold and cross_positive == 0:
                cross_net = -cross_negative
            else:
                cross_net = 0
        if mode == "same_exact":
            cross_net = 0
        if mode in {
            "cross_exact", "cf_shared_cross", "cf_shuffled_neighbors",
            "cf_random_neighbors", "cf_polarity_swapped",
        }:
            same_net = 0
        if mode == "cf_same_plus_shared" and same_net:
            # Personal evidence has priority; collaborative evidence only fills
            # gaps and can never override the target user's own failure event.
            cross_net = 0
        if mode == "popularity":
            same_net = 0
            cross_net = 0

        result[candidate_id] = {
            "candidate_item_id": candidate_id,
            "same_positive_users": len(bucket["same_positive"]),
            "same_negative_users": len(bucket["same_negative"]),
            "cross_positive_users": cross_positive,
            "cross_negative_users": cross_negative,
            "same_net": same_net,
            "cross_net": cross_net,
            "cross_support": cross_support,
            "popularity": int((candidate_popularity or {}).get(candidate_id, 0)),
            "evidence": evidence_by_candidate.get(candidate_id, []),
        }
    return result


def apply_typed_failure_constraints(
    parsed_scores: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    mode: str,
    tie_epsilon: float = 0.0,
    min_cross_support: int = 2,
    candidate_popularity: Optional[Dict[str, int]] = None,
    max_cross_corrections: int = 3,
) -> Tuple[List[str], Dict[str, Any]]:
    """Stably reorder tied/near-tied candidates using typed failure evidence.

    The base LLM score remains the primary signal. Failure evidence only orders
    candidates inside score groups, avoiding an arbitrary additive coefficient.
    """
    mode = str(mode or "none").strip().lower()
    base_rows = [dict(row) for row in parsed_scores]
    base_rows.sort(key=lambda row: (-_safe_float(row.get("score"), -1.0), int(row.get("original_index", 0))))
    base_ranking = [str(row.get("item_id", "")) for row in base_rows]
    if mode == "none" or not base_rows:
        return base_ranking, {"enabled": False, "mode": mode}

    candidate_ids = [str(row.get("item_id", "")) for row in base_rows]
    signals = aggregate_typed_failure_constraints(
        evidence_rows=evidence_rows,
        candidate_item_ids=candidate_ids,
        mode=mode,
        min_cross_support=min_cross_support,
        candidate_popularity=candidate_popularity,
    )

    if (mode.startswith("cf_") or mode.startswith("g_")) and max_cross_corrections > 0:
        cross_candidates = sorted(
            (
                (item_id, signal) for item_id, signal in signals.items()
                if int(signal["cross_net"]) != 0 and int(signal["same_net"]) == 0
            ),
            key=lambda pair: (
                -abs(int(pair[1]["cross_net"])),
                -int(pair[1]["cross_support"]),
                pair[0],
            ),
        )
        allowed_cross = {
            item_id for item_id, _ in cross_candidates[:max(1, int(max_cross_corrections))]
        }
        for item_id, signal in signals.items():
            if int(signal["same_net"]) == 0 and item_id not in allowed_cross:
                signal["cross_net"] = 0

    epsilon = max(0.0, float(tie_epsilon))
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    group_score: Optional[float] = None
    for row in base_rows:
        score = _safe_float(row.get("score"), -1.0)
        if current and group_score is not None and abs(group_score - score) > epsilon:
            groups.append(current)
            current = []
            group_score = None
        if not current:
            group_score = score
        current.append(row)
    if current:
        groups.append(current)

    def direction(value: int) -> int:
        return 1 if value > 0 else -1 if value < 0 else 0

    constrained_rows: List[Dict[str, Any]] = []
    group_audit: List[Dict[str, Any]] = []
    for group in groups:
        before = [str(row.get("item_id", "")) for row in group]

        def constraint_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
            item_id = str(row.get("item_id", ""))
            signal = signals[item_id]
            if mode == "popularity":
                return (-signal["popularity"], int(row.get("original_index", 0)))
            same_direction = direction(int(signal["same_net"]))
            cross_direction = direction(int(signal["cross_net"]))
            return (
                -same_direction,
                -cross_direction,
                -abs(int(signal["same_net"])),
                -abs(int(signal["cross_net"])),
                int(row.get("original_index", 0)),
            )

        ordered = sorted(group, key=constraint_key)
        after = [str(row.get("item_id", "")) for row in ordered]
        constrained_rows.extend(ordered)
        group_audit.append({
            "score": _safe_float(group[0].get("score"), -1.0),
            "before": before,
            "after": after,
            "changed": before != after,
        })

    constrained_ranking = [str(row.get("item_id", "")) for row in constrained_rows]
    moved = [
        item_id for item_id in candidate_ids
        if base_ranking.index(item_id) != constrained_ranking.index(item_id)
    ]
    active_signals = {
        item_id: signal for item_id, signal in signals.items()
        if signal["same_net"] or signal["cross_net"] or (mode == "popularity" and signal["popularity"])
    }
    audit = {
        "enabled": True,
        "mode": mode,
        "tie_epsilon": epsilon,
        "min_cross_support": int(min_cross_support),
        "max_cross_corrections": int(max_cross_corrections),
        "num_evidence_rows": len(evidence_rows),
        "num_active_candidates": len(active_signals),
        "num_changed_groups": sum(bool(row["changed"]) for row in group_audit),
        "num_moved_candidates": len(moved),
        "moved_candidate_ids": moved,
        "base_ranking": base_ranking,
        "constrained_ranking": constrained_ranking,
        "candidate_signals": active_signals,
        "tie_groups": group_audit,
    }
    return constrained_ranking, audit
