"""Compact score prompt construction and score-output parsing."""

import os
import json
from typing import List, Dict, Optional, Any, Set, Tuple
import re

from fmrec.common import (
    estimate_simple_tokens,
    item_category,
    item_description,
    item_title,
    shorten_words,
)


def pack_memory_facts(
    rows: List[Tuple[str, Dict[str, Any]]],
    max_facts: int,
    max_words: int,
    token_budget: int,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Pack short corrective facts by score order under a small token budget.

    This is deliberately simpler than MemRec's neighbor packer: MEMCF packs only
    failure-contrastive facts that already passed graph/applicability gates.
    """
    facts: List[str] = []
    audit: List[Dict[str, Any]] = []
    used_tokens = 0
    for fact, row in rows:
        if max_facts > 0 and len(facts) >= max_facts:
            break
        short = shorten_words(fact, max_words)
        needed = estimate_simple_tokens(short)
        if token_budget > 0 and facts and used_tokens + needed > token_budget:
            row = dict(row)
            row["pack_decision"] = "skip_budget"
            row["packed_tokens"] = needed
            row["used_tokens_before"] = used_tokens
            audit.append(row)
            continue
        facts.append(short)
        used_tokens += needed
        row = dict(row)
        row["pack_decision"] = "keep"
        row["packed_fact"] = short
        row["packed_tokens"] = needed
        row["used_tokens_after"] = used_tokens
        audit.append(row)
    return facts, audit


def add_candidate_aliases(candidate_items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Use C01/C02 aliases in prompts to reduce item-id hallucination."""
    aliased = []
    alias_to_item_id: Dict[str, str] = {}
    for idx, item in enumerate(candidate_items, 1):
        alias = f"C{idx:02d}"
        item_id = str(item["item_id"])
        alias_to_item_id[alias] = item_id
        row = dict(item)
        row["candidate_id"] = alias
        # Keep real item id out of the main output contract. The title/category
        # are enough for ranking; code maps Cxx back to item_id.
        aliased.append({
            "candidate_id": alias,
            "title": row.get("title", ""),
            "category": row.get("category", "Unknown"),
            "description": row.get("description", ""),
        })
    return aliased, alias_to_item_id


def parse_score_entries_from_text(raw_output: str, alias_to_item_id: Dict[str, str]) -> List[Dict[str, Any]]:
    """Recover candidate scores from malformed JSON text."""
    text = str(raw_output or "")
    valid_aliases = set(alias_to_item_id)
    entries: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    pattern = re.compile(
        r'"candidate_id"\s*:\s*"?(C\d{2})"?\s*,\s*"score"\s*:\s*"?([0-9]*\.?[0-9]+)"?',
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        alias = match.group(1).upper()
        if alias not in valid_aliases or alias in seen:
            continue
        seen.add(alias)
        score = max(0.0, min(1.0, float(match.group(2))))
        entries.append({
            "candidate_id": alias,
            "item_id": alias_to_item_id[alias],
            "score": score,
            "rationale": "Recovered from malformed score JSON",
        })
    return entries


def score_entries_to_ranking(
    raw_scores: List[Dict[str, Any]],
    alias_to_item_id: Dict[str, str],
) -> Tuple[List[str], Dict[str, Any]]:
    """Validate score rows and return a complete ranking."""
    alias_order = list(alias_to_item_id.keys())
    alias_index = {alias: idx for idx, alias in enumerate(alias_order)}
    seen: Set[str] = set()
    parsed: List[Dict[str, Any]] = []
    invalid_rows: List[Any] = []

    for row in raw_scores if isinstance(raw_scores, list) else []:
        if not isinstance(row, dict):
            invalid_rows.append(row)
            continue
        alias = str(row.get("candidate_id", "")).upper().strip()
        if not alias and row.get("item_id") is not None:
            item_id = str(row.get("item_id"))
            alias = next((a for a, iid in alias_to_item_id.items() if iid == item_id), "")
        if alias not in alias_to_item_id or alias in seen:
            invalid_rows.append(row)
            continue
        seen.add(alias)
        try:
            score = float(row.get("score", 0.0))
        except Exception:
            score = 0.0
        parsed.append({
            "candidate_id": alias,
            "item_id": alias_to_item_id[alias],
            "score": max(0.0, min(1.0, score)),
            "rationale": str(row.get("rationale", ""))[:180],
            "original_index": alias_index[alias],
        })

    missing_aliases = [alias for alias in alias_order if alias not in seen]
    for alias in missing_aliases:
        parsed.append({
            "candidate_id": alias,
            "item_id": alias_to_item_id[alias],
            "score": -1.0,
            "rationale": "Missing from LLM score output",
            "original_index": alias_index[alias],
        })

    parsed.sort(key=lambda x: (-x["score"], x["original_index"]))
    ranked = [row["item_id"] for row in parsed]
    validation = {
        "is_valid": len(missing_aliases) == 0 and len(invalid_rows) == 0,
        "raw_score_rows": len(raw_scores) if isinstance(raw_scores, list) else 0,
        "expected_rows": len(alias_order),
        "missing_candidate_ids": missing_aliases,
        "invalid_or_duplicate_rows": invalid_rows,
        "parsed_scores": parsed,
    }
    return ranked, validation


def _item_info_for_prompt(item_id: str, items_meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    item_id = str(item_id)
    info = items_meta.get(item_id, {})
    return {
        "item_id": item_id,
        "title": item_title(info, item_id) if isinstance(info, dict) else f"Item {item_id}",
        "category": item_category(info) if isinstance(info, dict) else "Unknown",
        "description": item_description(info) if isinstance(info, dict) else "",
    }


def _history_item_infos(item_ids: List[str], items_meta: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_item_info_for_prompt(str(item_id), items_meta) for item_id in item_ids if str(item_id) in items_meta]


def _context_text_from_items(items: List[Dict[str, Any]]) -> str:
    return " ".join(
        f"{x.get('title', '')} {x.get('category', '')} {x.get('description', '')}"
        for x in items
    ).lower()


def _format_compact_history_lines(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "- No user history."
    include_desc = os.getenv("MEMCF_INCLUDE_DESCRIPTIONS_IN_PROMPT", "0").strip().lower() in {"1", "true", "yes"}
    lines = []
    for item in items[-10:]:
        desc = str(item.get("description", "")).strip() if include_desc else ""
        suffix = f"; description: {desc}" if desc else ""
        lines.append(
            f"- title: {str(item.get('title', '')).strip() or 'Unknown'}; "
            f"category: {str(item.get('category', 'Unknown')).strip() or 'Unknown'}"
            f"{suffix}"
        )
    return "\n".join(lines)


def _format_compact_candidate_lines(items: List[Dict[str, Any]]) -> str:
    include_desc = os.getenv("MEMCF_INCLUDE_DESCRIPTIONS_IN_PROMPT", "0").strip().lower() in {"1", "true", "yes"}
    lines = []
    for item in items:
        desc = str(item.get("description", "")).strip() if include_desc else ""
        suffix = f", description: {desc}" if desc else ""
        lines.append(
            f"- candidate_id: {item.get('candidate_id')}, "
            f"title: {str(item.get('title', '')).strip() or 'Unknown'}, "
            f"category: {str(item.get('category', 'Unknown')).strip() or 'Unknown'}"
            f"{suffix}"
        )
    return "\n".join(lines)


def build_compact_score_prompt(
    history_items: List[Dict[str, Any]],
    aliased_candidates: List[Dict[str, Any]],
    prompt_sample: str = "",
    memory_payload: Optional[List[Any]] = None,
    user_profile_payload: Optional[Dict[str, Any]] = None,
) -> str:
    using_memory = bool(memory_payload)
    intro = (
        "You are scoring candidate items for a recommender system based on user history, candidate facts, and optional memory facts.\n"
        if using_memory else
        "You are scoring candidate items for a recommender system based only on user history and candidate facts.\n"
    )
    parts = [intro]
    if prompt_sample:
        parts.append(str(prompt_sample).strip() + "\n")
    parts.append("Inputs:\n")
    if user_profile_payload:
        parts.append("User Memory Profile:\n")
        parts.append(json.dumps(user_profile_payload, ensure_ascii=False, indent=2) + "\n")
    parts.append("User recent history:\n")
    parts.append(_format_compact_history_lines(history_items) + "\n")
    if memory_payload:
        parts.append("Memory Facts:\n")
        for row in memory_payload[:5]:
            if isinstance(row, dict):
                text = (
                    row.get("lesson")
                    or row.get("behavior_explanation")
                    or row.get("pattern")
                    or row.get("pattern_description")
                    or json.dumps(row, ensure_ascii=False)
                )
            else:
                text = str(row)
            cleaned_text = re.sub(r"\s+", " ", str(text)).strip()[:220]
            parts.append(f"- {cleaned_text}\n")
    parts.append("Candidate Items:\n")
    parts.append(_format_compact_candidate_lines(aliased_candidates) + "\n")
    parts.append(
        "\nOutput requirements:\n"
        "- Return ONLY valid compact JSON. No markdown.\n"
        "- Output one score row for every candidate_id exactly once.\n"
        "- Score is a number from 0.0 to 1.0.\n"
        "- Rationale must be <= 8 words.\n"
        "- Base scoring on recent history and candidate facts.\n"
        "- If history is weak, prefer broader category relevance.\n"
        "\nJSON format:\n"
        "{\n"
        '  "scores": [\n'
        '    {"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}\n'
        "  ],\n"
        '  "reasoning": "one short sentence"\n'
        "}\n"
    )
    return "".join(parts)
