"""Pairwise collaborative-filtering score corrections (--pairwise_cf_rerank)."""

from typing import List, Dict, Optional, Any, Set, Tuple
from collections import defaultdict

from fmrec.common import _safe_float, shorten_words
from fmrec.variants.prompt_styles import _token_set_for_prompt


PAIRWISE_CF_GENERIC_TERMS = {
    "unknown", "amazon", "product", "products", "item", "items", "edition",
    "collection", "series", "pack", "bundle", "digital", "video", "music",
    "album", "albums", "game", "games", "software", "beauty", "industrial",
    "scientific", "pantry", "prime", "vinyl", "with", "from", "that",
    "this", "user", "candidate", "preferred", "instead", "wrong", "correct",
}


def _pairwise_cf_terms(*texts: Any) -> Set[str]:
    terms: Set[str] = set()
    for text in texts:
        terms.update(_token_set_for_prompt(text))
    return {
        term for term in terms
        if term not in PAIRWISE_CF_GENERIC_TERMS and len(term) >= 4
    }


def _pairwise_cf_candidate_tokens(candidate: Dict[str, Any]) -> Set[str]:
    return _pairwise_cf_terms(
        candidate.get("title", ""),
        candidate.get("category", ""),
        shorten_words(candidate.get("description", ""), 35),
    )


def _pairwise_cf_match_aliases(
    item_id: str,
    title: str,
    category: str,
    terms: Set[str],
    aliased_candidates: List[Dict[str, Any]],
    item_id_to_alias: Dict[str, str],
    min_overlap: int = 2,
) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    item_id = str(item_id or "")
    if item_id and item_id in item_id_to_alias:
        matches.append({
            "candidate_id": item_id_to_alias[item_id],
            "match_type": "direct_item_id",
            "overlap_terms": [],
            "strength": 1.0,
        })
        return matches

    title_low = str(title or "").strip().lower()
    category_terms = _pairwise_cf_terms(category)
    for cand in aliased_candidates:
        alias = str(cand.get("candidate_id", ""))
        cand_title = str(cand.get("title", "")).strip().lower()
        cand_tokens = _pairwise_cf_candidate_tokens(cand)
        overlap = sorted(cand_tokens & terms)
        category_overlap = sorted(cand_tokens & category_terms)
        title_match = (
            bool(title_low)
            and len(title_low) >= 8
            and bool(cand_title)
            and (title_low in cand_title or cand_title in title_low)
        )
        if title_match or len(overlap) >= min_overlap:
            strength = 0.55 + min(0.35, 0.08 * len(overlap))
            if title_match:
                strength = 0.9
            if category_overlap:
                strength += 0.05
            matches.append({
                "candidate_id": alias,
                "match_type": "title_or_terms",
                "overlap_terms": overlap[:8],
                "category_overlap": category_overlap[:4],
                "strength": min(1.0, strength),
            })
    matches.sort(key=lambda row: (-row["strength"], row["candidate_id"]))
    return matches[:3]


def _pairwise_cf_source_weight(sources: List[str]) -> float:
    source_set = set(sources or [])
    if "same_user" in source_set:
        return 1.0
    if "candidate_item" in source_set:
        return 0.85
    if "history_item" in source_set:
        return 0.70
    if "cluster_user" in source_set:
        return 0.65
    if "neighbor_user" in source_set:
        return 0.60
    return 0.45


def build_pairwise_cf_corrections(
    memory_rows: List[Dict[str, Any]],
    aliased_candidates: List[Dict[str, Any]],
    alias_to_item_id: Optional[Dict[str, str]] = None,
    max_corrections: int = 8,
) -> List[Dict[str, Any]]:
    """Map selected failure memories to candidate-level boost/demote actions.

    This keeps the graph signal collaborative but avoids asking the ranker to
    interpret raw cross-user failure text. A correction is only formed when a
    selected memory can be grounded to current candidate IDs by item ID or
    concrete title/category terms.
    """
    if not memory_rows:
        return []

    # Prompt-facing candidate rows intentionally omit real item IDs. Use the
    # parser's private alias map for exact grounding, then fall back to semantic
    # title matching only when no exact item is present.
    item_id_to_alias = {
        str(item_id): str(alias)
        for alias, item_id in (alias_to_item_id or {}).items()
    }
    if not item_id_to_alias:
        item_id_to_alias = {
            str(cand.get("item_id")): str(cand.get("candidate_id"))
            for cand in aliased_candidates
            if cand.get("item_id") is not None and cand.get("candidate_id") is not None
        }
    corrections: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()

    for row in memory_rows:
        sources = [str(x) for x in row.get("sources", [])]
        if any(src.startswith("random") or src.startswith("shuffled") for src in sources):
            continue
        risk = _safe_float(row.get("overgeneralization_risk"), 0.5)
        confidence = _safe_float(row.get("confidence"), 0.5)
        if risk >= 0.90 and "same_user" not in sources:
            continue
        correct_terms = _pairwise_cf_terms(row.get("correct_item_title", ""), row.get("correct_item_category", ""))
        wrong_terms = _pairwise_cf_terms(row.get("wrong_item_title", ""), row.get("wrong_item_category", ""))
        correct_matches = _pairwise_cf_match_aliases(
            item_id=str(row.get("correct_item_id", "")),
            title=str(row.get("correct_item_title", "")),
            category=str(row.get("correct_item_category", "")),
            terms=correct_terms,
            aliased_candidates=aliased_candidates,
            item_id_to_alias=item_id_to_alias,
            min_overlap=2,
        )
        wrong_matches = _pairwise_cf_match_aliases(
            item_id=str(row.get("wrong_item_id", "")),
            title=str(row.get("wrong_item_title", "")),
            category=str(row.get("wrong_item_category", "")),
            terms=wrong_terms,
            aliased_candidates=aliased_candidates,
            item_id_to_alias=item_id_to_alias,
            min_overlap=2,
        )
        if not correct_matches and not wrong_matches:
            continue

        source_weight = _pairwise_cf_source_weight(sources)
        weight = max(0.10, min(1.0, confidence)) * source_weight * max(0.25, 1.0 - 0.35 * risk)
        memory_id = str(row.get("memory_id", ""))

        if correct_matches and wrong_matches:
            for cm in correct_matches:
                for wm in wrong_matches:
                    if cm["candidate_id"] == wm["candidate_id"]:
                        continue
                    key = (memory_id, cm["candidate_id"], wm["candidate_id"])
                    if key in seen:
                        continue
                    seen.add(key)
                    corrections.append({
                        "memory_id": memory_id,
                        "sources": sources,
                        "action": "boost_and_demote",
                        "boost_candidate_id": cm["candidate_id"],
                        "demote_candidate_id": wm["candidate_id"],
                        "weight": weight * min(cm.get("strength", 0.5), wm.get("strength", 0.5)),
                        "correct_match": cm,
                        "wrong_match": wm,
                        "correct_item_id": row.get("correct_item_id"),
                        "wrong_item_id": row.get("wrong_item_id"),
                        "correct_item_title": row.get("correct_item_title"),
                        "wrong_item_title": row.get("wrong_item_title"),
                    })
        elif correct_matches:
            for cm in correct_matches:
                key = (memory_id, cm["candidate_id"], "")
                if key in seen:
                    continue
                seen.add(key)
                corrections.append({
                    "memory_id": memory_id,
                    "sources": sources,
                    "action": "boost_only",
                    "boost_candidate_id": cm["candidate_id"],
                    "demote_candidate_id": "",
                    "weight": weight * cm.get("strength", 0.5) * 0.75,
                    "correct_match": cm,
                    "wrong_match": {},
                    "correct_item_id": row.get("correct_item_id"),
                    "wrong_item_id": row.get("wrong_item_id"),
                    "correct_item_title": row.get("correct_item_title"),
                    "wrong_item_title": row.get("wrong_item_title"),
                })
        else:
            for wm in wrong_matches:
                key = (memory_id, "", wm["candidate_id"])
                if key in seen:
                    continue
                seen.add(key)
                corrections.append({
                    "memory_id": memory_id,
                    "sources": sources,
                    "action": "demote_only",
                    "boost_candidate_id": "",
                    "demote_candidate_id": wm["candidate_id"],
                    "weight": weight * wm.get("strength", 0.5) * 0.65,
                    "correct_match": {},
                    "wrong_match": wm,
                    "correct_item_id": row.get("correct_item_id"),
                    "wrong_item_id": row.get("wrong_item_id"),
                    "correct_item_title": row.get("correct_item_title"),
                    "wrong_item_title": row.get("wrong_item_title"),
                })

    corrections.sort(key=lambda row: (-row.get("weight", 0.0), row.get("memory_id", "")))
    return corrections[:max(0, int(max_corrections or 0))]


def apply_pairwise_cf_score_adjustments(
    parsed_scores: List[Dict[str, Any]],
    corrections: List[Dict[str, Any]],
    alpha: float = 0.04,
    beta: float = 0.04,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scores_by_alias: Dict[str, Dict[str, Any]] = {}
    deltas: Dict[str, float] = defaultdict(float)
    for row in parsed_scores:
        alias = str(row.get("candidate_id", "")).upper().strip()
        if not alias:
            continue
        scores_by_alias[alias] = dict(row)

    applied: List[Dict[str, Any]] = []
    for correction in corrections:
        weight = max(0.0, min(1.0, _safe_float(correction.get("weight"), 0.0)))
        boost_alias = str(correction.get("boost_candidate_id", "")).upper().strip()
        demote_alias = str(correction.get("demote_candidate_id", "")).upper().strip()
        boost_delta = float(alpha) * weight if boost_alias in scores_by_alias else 0.0
        demote_delta = -float(beta) * weight if demote_alias in scores_by_alias else 0.0
        if boost_delta:
            deltas[boost_alias] += boost_delta
        if demote_delta:
            deltas[demote_alias] += demote_delta
        if boost_delta or demote_delta:
            applied.append({
                **correction,
                "boost_delta": boost_delta,
                "demote_delta": demote_delta,
            })

    adjusted_scores: List[Dict[str, Any]] = []
    for alias, row in scores_by_alias.items():
        base_score = _safe_float(row.get("score"), 0.0)
        delta = deltas.get(alias, 0.0)
        adjusted = max(0.0, min(1.0, base_score + delta))
        adjusted_scores.append({
            "candidate_id": alias,
            "score": adjusted,
            "rationale": shorten_words(
                f"{row.get('rationale', '')} pairwise_cf_delta={delta:.3f}",
                12,
            ),
            "base_score": base_score,
            "pairwise_cf_delta": delta,
        })

    audit = {
        "enabled": True,
        "num_input_corrections": len(corrections),
        "num_applied_corrections": len(applied),
        "alpha": alpha,
        "beta": beta,
        "deltas_by_candidate": dict(sorted(deltas.items())),
        "applied_corrections": applied,
    }
    return adjusted_scores, audit
