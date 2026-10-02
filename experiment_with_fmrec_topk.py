import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
# import google.generativeai as genai
from collections import defaultdict, Counter
from tqdm import tqdm
import random
import pickle
import time
import re
import html
import hashlib
import urllib.request
import urllib.error
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


def slugify(value: Any) -> str:
    """Make a short filesystem-safe value for run names."""
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text[:80] or "none"


def float_tag(value: float) -> str:
    """Stable compact float tag for filenames, e.g. 0.25 -> 0p25."""
    return f"{float(value):.3g}".replace(".", "p").replace("-", "m")


def stable_shard_filter(values: List[str], shard_id: int, num_shards: int) -> List[str]:
    """Select a deterministic user shard without changing the global user order."""
    if num_shards <= 1:
        return list(values)
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard_id={shard_id} for num_shards={num_shards}")
    return [v for idx, v in enumerate(values) if idx % num_shards == shard_id]


def shorten_words(text: Any, max_words: int) -> str:
    """Sentence-safe word cap used before memory facts enter ranking prompts."""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if max_words <= 0:
        return cleaned
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned
    return " ".join(words[:max_words]).rstrip(" .,;:") + "..."


def estimate_simple_tokens(text: Any) -> int:
    return max(1, int(len(str(text or "")) / 4)) if str(text or "") else 0


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


def has_metadata_noise(text: Any) -> bool:
    """Detect HTML/entity artifacts that often indicate noisy item metadata."""
    raw = str(text or "")
    return bool(re.search(r"<[^>]+>|&[A-Za-z]+;|a-size-|a-color-|span class|h1 class", raw, re.I))


def extract_json_object(raw_output: str) -> Dict[str, Any]:
    """Extract the first JSON object from an LLM response."""
    result_text = str(raw_output).strip()
    if "```json" in result_text:
        result_text = result_text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in result_text:
        result_text = result_text.split("```", 1)[1].split("```", 1)[0].strip()

    match = re.search(r"\{.*\}", result_text, re.DOTALL)
    json_str = match.group(0) if match else result_text
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Local OpenAI-compatible models sometimes emit invalid backslash escapes
        # or trailing commas even when asked for strict JSON.
        json_str = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", json_str)
        json_str = escape_control_chars_in_json_strings(json_str)
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return json.loads(balance_json_object(json_str))


def escape_control_chars_in_json_strings(text: str) -> str:
    """Escape literal newlines/tabs inside JSON strings."""
    out = []
    in_string = False
    escaped = False
    for ch in str(text):
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ch in {"\n", "\r", "\t"}:
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch])
        else:
            out.append(ch)
    return "".join(out)


def balance_json_object(text: str) -> str:
    """Best-effort close truncated JSON for local Qwen responses."""
    out = []
    stack = []
    in_string = False
    escaped = False
    for ch in str(text):
        out.append(ch)
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    if in_string:
        out.append('"')
    while stack:
        out.append("}" if stack.pop() == "{" else "]")
    return re.sub(r",\s*([}\]])", r"\1", "".join(out))


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


def deterministic_shuffle(values: List[Any], salt: str = "") -> List[Any]:
    """Shuffle reproducibly without depending on global random state consumed during training."""
    values = list(values)
    seed_material = salt + "||" + "||".join(str(v) for v in values)
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    rng.shuffle(values)
    return values


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


def make_jsonable(obj: Any) -> Any:
    """Convert dataclasses/numpy values into JSON-safe objects for traces."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (UserInteraction, BehaviorMemory, PairwiseUserState, PairwiseItemState)):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_jsonable(v) for v in obj]
    return obj


def interaction_to_trace(interaction: "UserInteraction") -> Dict[str, Any]:
    return make_jsonable(asdict(interaction))


def behavior_memory_to_trace(memory: "BehaviorMemory") -> Dict[str, Any]:
    data = make_jsonable(asdict(memory))
    data["embedding"] = None
    data["embedding_dim"] = int(len(memory.embedding)) if memory.embedding is not None else 0
    data["interaction_sequence"] = [interaction_to_trace(i) for i in memory.interaction_sequence]
    return data


GENERIC_MEMORY_TERMS = {
    "all", "beauty", "unknown", "category", "item", "items", "product", "products",
    "preference", "preferences", "user", "users", "recommendation", "recommendations",
    "wrong", "correct", "choice", "chosen", "preferred", "pattern", "future",
    "the", "and", "for", "with", "without", "this", "that", "these", "those",
    "system", "pack", "set", "edition", "standard", "new", "one", "two", "three",
    "likely", "intent", "needs", "need", "prioritize", "relevant", "similar",
    "based", "match", "matches", "matching", "current", "past", "history",
    "video", "game", "games", "gaming", "digital", "music", "album", "albums",
    "cds", "vinyl", "record", "records", "logo", "image", "audio",
    "beauty", "skin", "care", "shopping", "purchase", "purchases",
}

GENERIC_MEMORY_PHRASES = {
    "video game", "video games", "digital music", "cds and vinyl", "all beauty",
    "unknown category", "category mismatch", "user preference", "preferred item",
    "wrong choice", "future ranking", "current candidates",
}

# Ablation flag: MEMCF_DISABLE_GENERIC_TERM_FILTER=1 turns normalize_terms /
# normalize_evidence_terms into pure length-gated extractors, skipping the
# GENERIC_MEMORY_TERMS / GENERIC_MEMORY_PHRASES membership checks below. Off
# by default -- unset (or any value other than "1") reproduces the original
# behavior exactly. Added to empirically test whether the hand-curated
# generic-term gate carries its own weight, rather than assuming it from code
# inspection alone.
_DISABLE_GENERIC_TERM_FILTER = os.getenv("MEMCF_DISABLE_GENERIC_TERM_FILTER", "0") == "1"


def normalize_terms(text: str) -> List[str]:
    if _DISABLE_GENERIC_TERM_FILTER:
        return [t for t in re.findall(r"[a-zA-Z0-9]+", str(text).lower()) if len(t) >= 3]
    return [
        t for t in re.findall(r"[a-zA-Z0-9]+", str(text).lower())
        if len(t) >= 3 and t not in GENERIC_MEMORY_TERMS
    ]


def normalize_evidence_terms(values: Any) -> List[str]:
    """Normalize structured memory evidence terms and remove generic terms
    (unless MEMCF_DISABLE_GENERIC_TERM_FILTER=1; see flag above)."""
    if values is None:
        return []
    if isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    else:
        raw_values = [str(values)]

    terms: List[str] = []
    seen: Set[str] = set()
    for value in raw_values:
        phrase = re.sub(r"\s+", " ", str(value).lower()).strip(" .,:;|")
        if not phrase:
            continue
        if not _DISABLE_GENERIC_TERM_FILTER and phrase in GENERIC_MEMORY_PHRASES:
            continue
        is_generic = (not _DISABLE_GENERIC_TERM_FILTER) and phrase in GENERIC_MEMORY_TERMS
        if 3 <= len(phrase) <= 40 and not is_generic and phrase not in seen:
            terms.append(phrase)
            seen.add(phrase)
        for token in normalize_terms(phrase):
            if token not in seen:
                terms.append(token)
                seen.add(token)
    return terms


def normalize_category(category: Any, fallback: str = "Unknown") -> str:
    """Normalize noisy Amazon metadata categories for prompt/retrieval use."""
    if isinstance(category, list):
        parts: List[str] = []
        for value in category:
            if isinstance(value, list):
                parts.extend(str(x) for x in value)
            else:
                parts.append(str(value))
        raw = " > ".join(p for p in parts if p)
    else:
        raw = str(category or "").strip()

    if not raw or raw.lower() in {"none", "nan", "[]", "unknown"}:
        return fallback

    alt_match = re.search(r'alt=["\']([^"\']+)["\']', raw, flags=re.IGNORECASE)
    if alt_match:
        raw = alt_match.group(1)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip(" /|>")
    return raw or fallback


def memory_text_is_too_generic(text: str, min_terms: int = 4) -> bool:
    return len(set(normalize_terms(text))) < min_terms


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


def item_category(item_info: Dict[str, Any], fallback: str = "Unknown") -> str:
    return normalize_category(
        item_info.get("main_cat")
        or item_info.get("category")
        or item_info.get("categories")
        or fallback,
        fallback=fallback,
    )


def item_title(item_info: Dict[str, Any], item_id: str) -> str:
    title = str(item_info.get("title") or "").strip()
    title = html.unescape(re.sub(r"<[^>]+>", " ", title))
    title = re.sub(r"\s+", " ", title).strip()
    return title if title else f"Item {item_id}"


def item_description(item_info: Dict[str, Any], max_chars: int = 220) -> str:
    raw = (
        item_info.get("description")
        or item_info.get("description_short")
        or item_info.get("feature")
        or ""
    )
    if isinstance(raw, list):
        raw = " ".join(str(x) for x in raw if str(x).strip())
    elif isinstance(raw, dict):
        raw = " ".join(str(x) for x in raw.values() if str(x).strip())
    text = html.unescape(str(raw))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\b(Product Description|Amazon\\.com|Product description)\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" []'\",")
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip(" .,;:") + "..."
    return text


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


def term_matches_context(term: str, context: str) -> bool:
    term = re.sub(r"\s+", " ", str(term).lower()).strip()
    if not term:
        return False
    if " " in term:
        return term in context
    return re.search(rf"\b{re.escape(term)}\b", context) is not None


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
# torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)
# np.random.seed(42)
# random.seed(42)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
set_seed(42)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: transformers/torch not available: {e}")
    print("Will use OpenAI-compatible API endpoints if provided via env vars.")
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

SentenceTransformer = None
SENTENCE_TRANSFORMERS_AVAILABLE = False

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


class RecommendationMemorySystem:
    """A-Mem adapted for Amazon product recommendation with Memory Evolution"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_gemini_embeddings: bool = None,
                 chat_api_base: Optional[str] = None,
                 embedding_api_base: Optional[str] = None,
                 api_key: Optional[str] = None,
                 chat_model_name: Optional[str] = None,
                 embedding_model_name: Optional[str] = None):
        _ = use_gemini_embeddings  # kept for backward compatibility

        self.llm_name = model_name
        self.embedding_model_name = embedding_model_name or os.getenv("embedding_model_name") or embedding_model
        self.chat_model_name = chat_model_name or os.getenv("chat_model_name") or model_name
        self.chat_api_base = (chat_api_base or os.getenv("chat_api_base") or os.getenv("api_base") or "").rstrip("/")
        self.embedding_api_base = (embedding_api_base or os.getenv("embedding_api_base") or os.getenv("api_base") or "").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"

        self.use_api_chat = bool(self.chat_api_base)
        # MEMCF graph retrieval does not require neural embeddings. Legacy
        # similarity/evolution code paths use deterministic hash vectors instead
        # of loading SentenceTransformer, so MEMCF runs never block on local
        # embedding model initialization.
        self.use_api_embedding = False

        self.tokenizer = None
        self.model = None
        self.embedding_model = None

        if not self.use_api_chat:
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError(
                    "Local chat model requires transformers+torch, or set chat_api_base/api_base env vars."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                dtype=torch.float16,
                device_map="auto"
            )


        self.behavior_memories: List[BehaviorMemory] = []
        self.user_interaction_history: List[UserInteraction] = []
        self.next_thought_id = 0
        self.trace_recorder: Optional[TraceRecorder] = None
        self.memory_diagnostics = defaultdict(float)
        self.llm_usage = defaultdict(float)
        self.llm_usage_by_type: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def _trace(self, event_type: str, payload: Dict[str, Any]) -> None:
        if getattr(self, "trace_recorder", None) is not None:
            self.trace_recorder.log(event_type, payload)

    def _estimate_token_count(self, text: str) -> int:
        text = str(text or "")
        if not text:
            return 0
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None:
            try:
                return int(len(tokenizer.encode(text, add_special_tokens=False)))
            except Exception:
                pass
        # Conservative fallback used when running through an API without tokenizer.
        return max(1, int(len(text) / 4))

    def _record_llm_usage(
        self,
        call_type: str,
        prompt: str,
        role_prompt: str,
        output: str,
        duration_seconds: float,
        usage: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        usage = usage or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if prompt_tokens is None:
            prompt_tokens = self._estimate_token_count(str(role_prompt) + "\n" + str(prompt))
        if completion_tokens is None:
            completion_tokens = self._estimate_token_count(output)
        if total_tokens is None:
            total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)

        call_type = str(call_type or "generic")
        metrics = {
            "calls": 1,
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int(total_tokens or 0),
            "seconds": float(duration_seconds),
            "errors": 0 if success else 1,
        }
        for key, value in metrics.items():
            self.llm_usage[key] += value
            self.llm_usage_by_type[call_type][key] += value

        self._trace("llm_call", {
            "call_type": call_type,
            "success": success,
            "error": error,
            "model": self.chat_model_name if self.use_api_chat else self.llm_name,
            "prompt_chars": len(str(prompt or "")),
            "completion_chars": len(str(output or "")),
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int(total_tokens or 0),
            "seconds": float(duration_seconds),
        })

    def get_llm_usage_summary(self) -> Dict[str, Any]:
        total_calls = int(self.llm_usage.get("calls", 0))
        total_seconds = float(self.llm_usage.get("seconds", 0.0))
        total_tokens = int(self.llm_usage.get("total_tokens", 0))
        by_type = {
            key: {
                "calls": int(vals.get("calls", 0)),
                "prompt_tokens": int(vals.get("prompt_tokens", 0)),
                "completion_tokens": int(vals.get("completion_tokens", 0)),
                "total_tokens": int(vals.get("total_tokens", 0)),
                "seconds": float(vals.get("seconds", 0.0)),
                "errors": int(vals.get("errors", 0)),
            }
            for key, vals in sorted(self.llm_usage_by_type.items())
        }
        return {
            "calls": total_calls,
            "prompt_tokens": int(self.llm_usage.get("prompt_tokens", 0)),
            "completion_tokens": int(self.llm_usage.get("completion_tokens", 0)),
            "total_tokens": total_tokens,
            "seconds": total_seconds,
            "errors": int(self.llm_usage.get("errors", 0)),
            "avg_seconds_per_call": total_seconds / total_calls if total_calls else 0.0,
            "avg_tokens_per_call": total_tokens / total_calls if total_calls else 0.0,
            "by_call_type": by_type,
        }

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        last_error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")
                last_error = RuntimeError(f"API HTTP {e.code} at {url}: {body}")
            except Exception as e:
                last_error = e
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"API request failed after 3 attempts at {url}: {last_error}") from last_error

    def qwen_generate(
        self,
        prompt: str,
        role_prompt="You are a helpful AI assistant.",
        max_new_tokens=8000,
        json_schema: Optional[Dict[str, Any]] = None,
        json_mode: bool = False,
        call_type: str = "generic",
    ) -> str:
        start_time = time.time()
        if self.use_api_chat:
            endpoint = f"{self.chat_api_base}/chat/completions"
            temperature = float(os.getenv("MEMCF_TEMPERATURE", "0.0"))
            payload = {
                "model": self.chat_model_name,
                "messages": [
                    {"role": "system", "content": role_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": 1.0,
            }
            llm_seed = os.getenv("MEMCF_LLM_SEED", "").strip()
            if llm_seed:
                try:
                    payload["seed"] = int(llm_seed)
                except ValueError as exc:
                    raise ValueError("MEMCF_LLM_SEED must be an integer") from exc
            repetition_penalty = os.getenv("MEMCF_REPETITION_PENALTY", "1.05").strip()
            if repetition_penalty:
                try:
                    payload["repetition_penalty"] = float(repetition_penalty)
                except ValueError:
                    pass
            if json_schema is not None and os.getenv("MEMCF_USE_GUIDED_JSON", "0") == "1":
                # vLLM's OpenAI-compatible server accepts guided_json as an
                # extra request field. Keep this opt-in because older servers
                # may reject unknown structured-output parameters.
                payload["guided_json"] = json_schema
            elif json_mode and os.getenv("MEMCF_USE_RESPONSE_FORMAT_JSON", "0") == "1":
                # Some OpenAI-compatible Qwen endpoints support JSON mode.
                # The prompt/role must include the word JSON for those servers.
                payload["response_format"] = {"type": "json_object"}
            try:
                result = self._post_json(endpoint, payload)
                content = result["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    output = "".join(
                        chunk.get("text", "") for chunk in content if isinstance(chunk, dict)
                    )
                else:
                    output = str(content)
                self._record_llm_usage(
                    call_type=call_type,
                    prompt=prompt,
                    role_prompt=role_prompt,
                    output=output,
                    duration_seconds=time.time() - start_time,
                    usage=result.get("usage"),
                    success=True,
                )
                return output
            except Exception as e:
                self._record_llm_usage(
                    call_type=call_type,
                    prompt=prompt,
                    role_prompt=role_prompt,
                    output="",
                    duration_seconds=time.time() - start_time,
                    usage=None,
                    success=False,
                    error=str(e),
                )
                raise

        messages = [
            {"role": "system", "content": role_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

            prompt_tokens = int(inputs["input_ids"].shape[-1])
            gen_ids = outputs[0][prompt_tokens:]
            output = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            completion_tokens = int(len(gen_ids))
            self._record_llm_usage(
                call_type=call_type,
                prompt=prompt,
                role_prompt=role_prompt,
                output=output,
                duration_seconds=time.time() - start_time,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                success=True,
            )
            return output
        except Exception as e:
            self._record_llm_usage(
                call_type=call_type,
                prompt=prompt,
                role_prompt=role_prompt,
                output="",
                duration_seconds=time.time() - start_time,
                usage=None,
                success=False,
                error=str(e),
            )
            raise


    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    def _create_embedding(self, text: str) -> np.ndarray:
        # No neural embedding dependency in MEMCF. This only supports legacy
        # memory-link/evolution code paths; main graph retrieval is symbolic.
        return self._simple_hash_embedding(str(text), dim=384).astype(np.float32)

    def _simple_hash_embedding(self, text: str, dim: int = 384) -> np.ndarray:
        digest = hashlib.sha256(str(text).encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False) % (2**32)
        rng = np.random.default_rng(seed)
        embedding = rng.standard_normal(dim).astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)
    
    def add_interaction(self, 
                       item_id: str,
                       item_name: str,
                       item_category: str,
                       action_type: str = "purchase",
                       rating: Optional[float] = None,
                       metadata: Optional[Dict] = None) -> UserInteraction:
        interaction = UserInteraction(
            item_id=item_id,
            item_name=item_name,
            item_category=item_category,
            action_type=action_type,
            rating=rating,
            metadata=metadata or {}
        )
        
        self.user_interaction_history.append(interaction)
        
        return interaction
    
    def create_behavior_thought(self, 
                               interaction_window: List[UserInteraction],
                               k_neighbors: int = 10) -> BehaviorMemory:
        interaction_summary = []
        for interaction in interaction_window:
            summary = {
                "item": interaction.item_name,
                "category": interaction.item_category,
                "action": interaction.action_type
            }
            interaction_summary.append(summary)
        
        prompt = f"""Analyze this failed recommendation interaction.
        Input: {json.dumps(interaction_summary, indent=2)}

        Context:
        - The interaction contains a wrong choice and the preferred correct item.
        - Your job is to capture why the wrong choice happened and what correction rule should be applied next time.
        - This memory will be retrieved for future ranking. It must be specific enough to avoid being applied to unrelated users.

        Return ONLY a JSON object in this format:
        {{
        "behavior_explanation": "2-3 concise sentences explaining why the wrong choice was made versus the correct item",
        "pattern_description": "2-3 concise sentences describing a correction rule, including when it applies and when it should NOT apply",
        "applicable_when": ["specific title/type/category/attribute evidence required before using this memory"],
        "not_applicable_when": ["conditions where this memory should be ignored"],
        "wrong_item_type": "short concrete type/attribute of the wrong choice",
        "correct_item_type": "short concrete type/attribute of the preferred item",
        "evidence_terms_required": ["concrete evidence terms that must appear in future history/candidates before applying this memory"],
        "specificity_score": 0.0 to 1.0,
        "overgeneralization_risk": 0.0 to 1.0,
        "keywords": ["kw1", "kw2", ...] (5-8 concrete fail-interaction signals, no generic words)
        }}

        Requirements:
        - Ground every statement in the input interaction.
        - Emphasize contrast between wrong and correct choice.
        - Avoid generic shopping summaries like 'user preference', 'category mismatch', or 'prioritize relevant items' unless tied to concrete terms.
        - Do not claim the rule applies unless future candidates/history contain the applicable evidence."""

        try:
            # response = self.model.generate_content(prompt)
            response = self.qwen_generate(
                prompt=prompt,
                role_prompt='You are a behavioral memory modeling system.',
                call_type="memory_create",
            )
            # time.sleep(5)
            result = extract_json_object(response)
            self._trace("memory_create_llm", {
                "prompt": prompt,
                "role_prompt": "You are a behavioral memory modeling system.",
                "answer": response,
                "parsed": result,
                "interaction_window": [interaction_to_trace(i) for i in interaction_window],
            })
            
            behavior_explanation = result.get("behavior_explanation", "")
            pattern_description = result.get("pattern_description", "")
            applicable_when = result.get("applicable_when", [])
            not_applicable_when = result.get("not_applicable_when", [])
            wrong_item_type = result.get("wrong_item_type", "")
            correct_item_type = result.get("correct_item_type", "")
            evidence_terms_required = result.get("evidence_terms_required", [])
            try:
                specificity_score = float(result.get("specificity_score", 0.0))
            except Exception:
                specificity_score = 0.0
            try:
                overgeneralization_risk = float(result.get("overgeneralization_risk", 0.0))
            except Exception:
                overgeneralization_risk = 0.0
            if applicable_when:
                pattern_description += f" Applicable when: {json.dumps(applicable_when, ensure_ascii=False)}."
            if not_applicable_when:
                pattern_description += f" Do not apply when: {json.dumps(not_applicable_when, ensure_ascii=False)}."
            if evidence_terms_required:
                pattern_description += f" Evidence required: {json.dumps(evidence_terms_required, ensure_ascii=False)}."
            if wrong_item_type or correct_item_type:
                pattern_description += (
                    f" Wrong item type: {wrong_item_type}. "
                    f"Correct item type: {correct_item_type}."
                )
            keywords = result.get("keywords", [])
            for extra_kw in [wrong_item_type, correct_item_type] + list(evidence_terms_required or []):
                if extra_kw and extra_kw not in keywords:
                    keywords.append(extra_kw)
            # extracted_preferences = result.get("extracted_preferences", [])
            
        except Exception as e:
            print(f"Error in behavior analysis: {e}")
            self._trace("memory_create_error", {
                "error": str(e),
                "prompt": prompt,
                "interaction_window": [interaction_to_trace(i) for i in interaction_window],
            })
            behavior_explanation = f"A failed interaction occurred with {len(interaction_window)} compared items."
            pattern_description = "Correction rule is unclear; prefer signals from the preferred item over the wrong choice."
            # extracted_preferences = []
            keywords = [i.item_category for i in interaction_window[:3]]
            applicable_when = []
            not_applicable_when = []
            wrong_item_type = ""
            correct_item_type = ""
            evidence_terms_required = []
            specificity_score = 0.0
            overgeneralization_risk = 1.0
        
        combined_text = f"{behavior_explanation} {pattern_description} {' '.join(keywords)}"
        embedding = self._create_embedding(combined_text)
        
        behavior_memory = BehaviorMemory(
            thought_id=self.next_thought_id,
            interaction_sequence=interaction_window.copy(),
            behavior_explanation=behavior_explanation,
            pattern_description=pattern_description,
            # extracted_preferences=extracted_preferences,
            keywords=keywords,
            embedding=embedding,
            applicable_when=applicable_when if isinstance(applicable_when, list) else [str(applicable_when)],
            not_applicable_when=not_applicable_when if isinstance(not_applicable_when, list) else [str(not_applicable_when)],
            wrong_item_type=str(wrong_item_type or ""),
            correct_item_type=str(correct_item_type or ""),
            evidence_terms_required=evidence_terms_required if isinstance(evidence_terms_required, list) else [str(evidence_terms_required)],
            specificity_score=max(0.0, min(1.0, specificity_score)),
            overgeneralization_risk=max(0.0, min(1.0, overgeneralization_risk)),
        )
        
        self.next_thought_id += 1
        self._trace("memory_created", {
            "memory": behavior_memory_to_trace(behavior_memory),
        })
        return behavior_memory
    
    def link_behavior_memories(self, 
                               new_memory: BehaviorMemory,
                               k: int = 5, wo_link=False) -> List[int]:
        """Link new behavior memory with similar past patterns"""
        if len(self.behavior_memories) == 0:
            return []
        
        similarities = []
        for memory in self.behavior_memories:
            sim = self._cosine_similarity(new_memory.embedding, memory.embedding)
            similarities.append((memory.thought_id, sim, memory))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        nearest_k = similarities[:min(k, len(similarities))]
        
        if len(nearest_k) == 0:
            return []
        if wo_link:
            linked = [thought_id for thought_id, _, _ in nearest_k]
            self._trace("memory_link_decision", {
                "new_memory": behavior_memory_to_trace(new_memory),
                "wo_link": True,
                "linked_thought_ids": linked,
                "reasoning": "wo_link enabled; using nearest memories without LLM link filtering.",
            })
            return linked
        
        nearest_info = []
        for thought_id, sim, memory in nearest_k:
            nearest_info.append({
                "thought_id": thought_id,
                "behavior_explanation": memory.behavior_explanation,
                "pattern": memory.pattern_description,
                # "preferences": memory.extracted_preferences,
                "similarity": float(sim)
            })
        
        prompt = f"""Determine if the new fail-interaction memory should be linked to past fail memories.
        New Pattern:
        - Behavior: {new_memory.behavior_explanation}
        - Pattern: {new_memory.pattern_description}

        Similar Past Patterns:
        {json.dumps(nearest_info, indent=2)}

        Link ONLY if:
        - They share a similar error/correction pattern (same mismatch type or same correction signal).
        - They imply a consistent fix strategy across users or interactions.
        - Their wrong-vs-correct contrast is semantically aligned.
        Do NOT link if they describe unrelated failure reasons.

        Return JSON:
        {{
        "should_link": true/false,
        "linked_thought_ids": [list of IDs],
        "reasoning": "1-2 sentences explaining shared fail/correction evidence"
        }}
        Keep reasoning concise and specific."""

        try:
            # response = self.model.generate_content(prompt)
            # time.sleep(3)
            response = self.qwen_generate(
                prompt=prompt,
                role_prompt='You are a behavioral memory modeling system.',
                call_type="memory_link",
            )

            result = extract_json_object(response)
            self._trace("memory_link_llm", {
                "prompt": prompt,
                "role_prompt": "You are a behavioral memory modeling system.",
                "answer": response,
                "parsed": result,
                "new_memory": behavior_memory_to_trace(new_memory),
                "nearest_info": nearest_info,
            })
            
            if result.get("should_link", False):
                linked = result.get("linked_thought_ids", [])
            else:
                linked = []
            self._trace("memory_link_decision", {
                "new_memory": behavior_memory_to_trace(new_memory),
                "linked_thought_ids": linked,
                "reasoning": result.get("reasoning", ""),
            })
            return linked
                
        except Exception as e:
            print(f"Error in linking: {e}")
            linked = [thought_id for thought_id, sim, _ in nearest_k if sim > 0.65]
            self._trace("memory_link_error", {
                "error": str(e),
                "new_memory": behavior_memory_to_trace(new_memory),
                "nearest_info": nearest_info,
                "fallback_linked_thought_ids": linked,
            })
            return linked
    
    def evolve_behavior_memories(self,
                                new_memory: BehaviorMemory,
                                linked_ids: List[int],
                                max_evolutions_per_memory: Optional[int] = None) -> None:
        """Evolve existing behavior memories based on new patterns (Section 3.3)"""
        if len(linked_ids) == 0:
            return

        linked_memories = [m for m in self.behavior_memories if m.thought_id in linked_ids]
        if len(linked_memories) == 0:
            return
        # ============ LỌC MEMORIES CÒN CÓ THỂ EVOLVE ============
        evolvable_memories = []
        for mem in linked_memories:
            if max_evolutions_per_memory is not None:
                mem.max_evolutions = max_evolutions_per_memory
            
            if mem.can_evolve():
                evolvable_memories.append(mem)
            else:
                print(f"  ⚠ Memory {mem.thought_id} reached max evolutions ({mem.evolution_count}), skipping...")
        
        if len(evolvable_memories) == 0:
            print("  → No memories available for evolution (all reached max)")
            return
        
        
        mem_info = []
        for mem in evolvable_memories:
            mem_info.append({
                "thought_id": mem.thought_id,
                "behavior_explanation": mem.behavior_explanation,
                "pattern": mem.pattern_description,
                "evolution_count": mem.evolution_count 
            })
        

# Return ONLY JSON."""
        prompt = f"""Determine if past fail memories should be updated using a new fail case.
        New Pattern:
        - Behavior: {new_memory.behavior_explanation}
        - Pattern: {new_memory.pattern_description}

        Linked Past Patterns (with evolution history):
        {json.dumps(mem_info, indent=2)}

        Update Guidelines:
        - Update when the new fail case provides clearer correction evidence for an existing fail pattern.
        - Refine wording toward a stronger wrong-vs-correct contrast.
        - Prefer updates that improve future error avoidance rules.
        - Skip updates when the new fail case is unrelated.

        Return JSON:
        {{
        "should_evolve": true/false,
        "updates": [
            {{
            "thought_id": ID,
            "behavior_explanation": "updated text or null",
            "new_pattern": "updated text or null",
            "reasoning": "1 sentence explaining how the fail-correction rule is refined"
            }}
        ]
        }}
        Ensure updates are grounded in input data and reasoning is concise."""

        try:
            # response = self.model.generate_content(prompt)
            # time.sleep(3)
            response = self.qwen_generate(
                prompt=prompt,
                role_prompt='You are a behavioral memory modeling system.',
                call_type="memory_evolve",
            )
            result = extract_json_object(response)
            self._trace("memory_evolve_llm", {
                "prompt": prompt,
                "role_prompt": "You are a behavioral memory modeling system.",
                "answer": response,
                "parsed": result,
                "new_memory": behavior_memory_to_trace(new_memory),
                "linked_ids": linked_ids,
                "linked_memories": mem_info,
            })
            
            if result.get("should_evolve", False):
                updates = result.get("updates", [])
                
                for update in updates:
                    thought_id = update.get("thought_id")
                    memory = next((m for m in self.behavior_memories if m.thought_id == thought_id), None)
                    
                    if memory:
                        # ============ GHI LẠI GIÁ TRỊ CŨ ============
                        old_values = {
                            'behavior_explanation': memory.behavior_explanation,
                            'pattern_description': memory.pattern_description,
                            # 'extracted_preferences': memory.extracted_preferences.copy()
                        }
                    
                        updated = False
                        update_type = []
                        if update.get("behavior_explanation"):
                            memory.behavior_explanation = update["behavior_explanation"]
                            updated = True
                            update_type.append("behavior_explanation")

                        if update.get("new_pattern"):
                            memory.pattern_description = update["new_pattern"]
                            updated = True
                            update_type.append("pattern")
                        
                        # if update.get("additional_preferences"):
                        #     memory.extracted_preferences.extend(update["additional_preferences"])
                        #     memory.extracted_preferences = list(set(memory.extracted_preferences))
                        #     updated = True
                        #     update_type.append("preferences")
                        
                        # Regenerate embedding if updated
                        if updated:
                            # combined_text = f"{memory.behavior_explanation} {memory.pattern_description} {' '.join(memory.keywords)} {' '.join(memory.extracted_preferences)}"
                            combined_text = f"{memory.behavior_explanation} {memory.pattern_description} {' '.join(memory.keywords)}"
                            memory.embedding = self._create_embedding(combined_text)
                            new_values = {
                                'behavior_explanation': memory.behavior_explanation,
                                'pattern_description': memory.pattern_description,
                                # 'extracted_preferences': memory.extracted_preferences.copy()
                            }
                            
                            memory.record_evolution(
                                update_type=", ".join(update_type),
                                old_values=old_values,
                                new_values=new_values,
                                reasoning=update.get("reasoning", "")
                            )
                            self._trace("memory_evolved", {
                                "new_memory": behavior_memory_to_trace(new_memory),
                                "evolved_thought_id": thought_id,
                                "update_type": ", ".join(update_type),
                                "old_values": old_values,
                                "new_values": new_values,
                                "reasoning": update.get("reasoning", ""),
                                "evolution_count": memory.evolution_count,
                            })
                        
        except Exception as e:
            print(f"Error in memory evolution: {e}")
            self._trace("memory_evolve_error", {
                "error": str(e),
                "new_memory": behavior_memory_to_trace(new_memory),
                "linked_ids": linked_ids,
            })
    
    def add_behavior_memory(self,
                           interaction_window: List[UserInteraction],
                           k_neighbors: int = 5) -> BehaviorMemory:
        """Complete A-Mem pipeline: Create, Link, and Evolve"""
        # Step 1: Create behavior thought
        behavior_memory = self.create_behavior_thought(interaction_window, k_neighbors)
        
        # Step 2: Link with similar patterns
        linked_ids = self.link_behavior_memories(behavior_memory, k_neighbors)
        behavior_memory.links = linked_ids
        
        # Update bidirectional links
        for thought_id in linked_ids:
            memory = next((m for m in self.behavior_memories if m.thought_id == thought_id), None)
            if memory and behavior_memory.thought_id not in memory.links:
                memory.links.append(behavior_memory.thought_id)
        
        # Step 3: Evolve existing memories based on new pattern
        self.evolve_behavior_memories(behavior_memory, linked_ids)
        
        # Add to collection
        self.behavior_memories.append(behavior_memory)
        return behavior_memory
    
    def retrieve_relevant_memory_records(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k memory records with similarity scores."""
        if len(self.behavior_memories) == 0:
            return []
        
        profile_embedding = self._create_embedding(query_text)
        
        similarities = []
        for memory in self.behavior_memories:
            sim = self._cosine_similarity(profile_embedding, memory.embedding)
            similarities.append((memory, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_records = [
            {"memory": mem, "similarity": float(sim)}
            for mem, sim in similarities[:k]
        ]
        self._trace("memory_retrieval", {
            "query_text": query_text,
            "k": k,
            "retrieved": [
                {
                    "similarity": float(sim),
                    "memory": behavior_memory_to_trace(mem),
                }
                for mem, sim in similarities[:k]
            ],
        })
        return top_records

    def retrieve_relevant_memories(self, user_profile_text: str, k: int = 5) -> List[BehaviorMemory]:
        """Backward-compatible retrieval API returning only memories."""
        return [record["memory"] for record in self.retrieve_relevant_memory_records(user_profile_text, k=k)]

    def record_memory_diagnostics(self, retrieved: int, kept: int, skipped: int) -> None:
        self.memory_diagnostics["eval_users"] += 1
        self.memory_diagnostics["retrieved_total"] += retrieved
        self.memory_diagnostics["kept_total"] += kept
        self.memory_diagnostics["skipped_total"] += skipped
        if kept > 0:
            self.memory_diagnostics["users_with_kept_memory"] += 1
    
    def llm_ranking(self,
                   train_items: List[Dict],
                   candidate_items: List[Dict],
                   retrieved_memories: Optional[List[BehaviorMemory]],
                   prompt_sample: str,
                   ranking_prompt_style: str = "memcf",
                   trace_context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Score candidates with the LLM, then sort locally.

        Previous versions asked Qwen to output a full permutation of raw item IDs.
        Traces showed frequent duplicates/missing IDs. This score-based path asks
        for C01..C20 candidate scores and maps them back to item IDs in code.
        """
        user_profile = [
            {"title": item["title"], "category": item["category"]}
            for item in train_items
        ]
        candidate_info = [
            {"item_id": item["item_id"], "title": item["title"], "category": item["category"]}
            for item in candidate_items
        ]
        aliased_candidates, alias_to_item_id = add_candidate_aliases(candidate_info)
        valid_candidate_aliases = list(alias_to_item_id.keys())

        memory_thoughts = []
        if retrieved_memories:
            for mem in retrieved_memories:
                memory_thoughts.append({
                    "memory_id": mem.thought_id,
                    "behavior_explanation": mem.behavior_explanation,
                    "pattern": mem.pattern_description,
                    "applicable_when": getattr(mem, "applicable_when", []),
                    "not_applicable_when": getattr(mem, "not_applicable_when", []),
                    "wrong_item_type": getattr(mem, "wrong_item_type", ""),
                    "correct_item_type": getattr(mem, "correct_item_type", ""),
                    "evidence_terms_required": getattr(mem, "evidence_terms_required", []),
                    "specificity_score": getattr(mem, "specificity_score", 0.0),
                    "overgeneralization_risk": getattr(mem, "overgeneralization_risk", 0.0),
                    "keywords": getattr(mem, "keywords", [])[:10],
                })

        if ranking_prompt_style == "compact_score":
            prompt = build_compact_score_prompt(
                history_items=user_profile[-10:],
                aliased_candidates=aliased_candidates,
                prompt_sample=prompt_sample,
                memory_payload=memory_thoughts if retrieved_memories else None,
                user_profile_payload=None,
            )
        elif retrieved_memories:
            prompt = f"""
You are scoring candidate items for a recommender system.

Important memory policy:
- Retrieved memories may be irrelevant.
- Use a memory only if its applicable_when or evidence_terms_required directly appears in the current user history or candidate items.
- If a memory conflicts with recent user history or candidate facts, ignore the memory.
- If no memory is clearly applicable, score exactly as you would from user history and candidate facts only.
- Memories are weak evidence, not hard rules.

Inputs:
User Recent History (last interactions; prioritize most recent):
{json.dumps(user_profile[-10:], ensure_ascii=False, indent=2)}

Retrieved Fail-Correction Memories:
{json.dumps(memory_thoughts, ensure_ascii=False, indent=2)}

Candidate Items (use candidate_id only in output):
{json.dumps(aliased_candidates, ensure_ascii=False, indent=2)}

Output requirements:
- Return ONLY valid JSON. No markdown.
- Output one score row for every candidate_id exactly once.
- Score is a number from 0.0 to 1.0.
- Rationale must be <= 8 words.

JSON format:
{{
  "scores": [
    {{"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}}
  ],
  "reasoning": "one short sentence"
}}
"""
        else:
            prompt = f"""
You are scoring candidate items for a recommender system based only on user history and candidate facts.
{prompt_sample}

Inputs:
User Recent History (last interactions; prioritize most recent):
{json.dumps(user_profile[-10:], ensure_ascii=False, indent=2)}

Candidate Items (use candidate_id only in output):
{json.dumps(aliased_candidates, ensure_ascii=False, indent=2)}

Output requirements:
- Return ONLY valid JSON. No markdown.
- Output one score row for every candidate_id exactly once.
- Score is a number from 0.0 to 1.0.
- Rationale must be <= 8 words.

JSON format:
{{
  "scores": [
    {{"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}}
  ],
  "reasoning": "one short sentence"
}}
"""

        score_json_schema = {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string", "enum": valid_candidate_aliases},
                            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "rationale": {"type": "string"},
                        },
                        "required": ["candidate_id", "score", "rationale"],
                        "additionalProperties": False,
                    },
                    "minItems": len(valid_candidate_aliases),
                    "maxItems": len(valid_candidate_aliases),
                },
            },
            "required": ["scores"],
            "additionalProperties": False,
        }
        if ranking_prompt_style != "compact_score":
            score_json_schema["properties"]["reasoning"] = {"type": "string"}
            score_json_schema["required"] = ["scores", "reasoning"]

        max_retries = int(os.getenv("MEMCF_RANK_RETRIES", "1"))
        current_prompt = prompt
        last_error: Optional[str] = None
        final_ranked: Optional[List[str]] = None
        final_validation: Optional[Dict[str, Any]] = None
        final_result: Dict[str, Any] = {}
        raw_response = ""

        for attempt in range(max_retries + 1):
            try:
                raw_response = self.qwen_generate(
                    prompt=current_prompt,
                    role_prompt=(
                        "You are a deterministic recommender scorer. "
                        "Return JSON only and follow the provided JSON schema exactly."
                    ),
                    max_new_tokens=int(os.getenv("MEMCF_RANK_MAX_TOKENS", "1400")),
                    json_schema=score_json_schema,
                    json_mode=True,
                    call_type="ranking",
                )
                try:
                    result = extract_json_object(raw_response)
                    raw_scores = result.get("scores", [])
                except Exception as parse_error:
                    last_error = str(parse_error)
                    result = {
                        "scores": parse_score_entries_from_text(raw_response, alias_to_item_id),
                        "reasoning": "Recovered score rows from malformed JSON",
                    }
                    raw_scores = result.get("scores", [])

                ranked_ids, validation = score_entries_to_ranking(raw_scores, alias_to_item_id)
                final_ranked = ranked_ids
                final_validation = validation
                final_result = result

                self.memory_diagnostics["rank_score_calls"] += 1
                self.memory_diagnostics["rank_missing_score_rows"] += len(validation["missing_candidate_ids"])
                self.memory_diagnostics["rank_invalid_score_rows"] += len(validation["invalid_or_duplicate_rows"])
                if validation["is_valid"]:
                    self.memory_diagnostics["rank_valid_score_outputs"] += 1
                else:
                    self.memory_diagnostics["rank_invalid_score_outputs"] += 1

                self._trace("ranking_llm", {
                    **(trace_context or {}),
                    "attempt": attempt,
                    "ranking_mode": "score_based_candidate_alias",
                    "prompt": current_prompt,
                    "role_prompt": (
                        "You are a deterministic recommender scorer. "
                        "Return JSON only and follow the provided JSON schema exactly."
                    ),
                    "answer": raw_response,
                    "parsed": result,
                    "score_validation": validation,
                    "raw_output_valid": validation["is_valid"],
                    "cleaned_ranked_item_ids": ranked_ids,
                    "candidate_items": candidate_info,
                    "aliased_candidate_items": aliased_candidates,
                    "alias_to_item_id": alias_to_item_id,
                    "train_items": user_profile[-10:],
                    "retrieved_memories": [
                        behavior_memory_to_trace(mem) for mem in (retrieved_memories or [])
                    ],
                    "use_retrieved_memories": retrieved_memories is not None,
                })

                if validation["is_valid"] or attempt >= max_retries:
                    if not validation["is_valid"]:
                        self._trace("ranking_retry_exhausted", {
                            **(trace_context or {}),
                            "attempts": attempt + 1,
                            "final_validation": validation,
                            "cleaned_ranked_item_ids": ranked_ids,
                        })
                    return ranked_ids

                current_prompt = f"""{prompt}

The previous answer was invalid:
{json.dumps(validation, ensure_ascii=False, indent=2)}

Retry now. Return ONLY valid JSON with exactly one score row for every candidate_id.
"""
            except Exception as e:
                last_error = str(e)
                self.memory_diagnostics["rank_attempt_errors"] += 1
                self._trace("ranking_attempt_error", {
                    **(trace_context or {}),
                    "attempt": attempt,
                    "error": last_error,
                    "prompt": current_prompt,
                    "candidate_items": candidate_info,
                    "aliased_candidate_items": aliased_candidates,
                    "retrieved_memories": [
                        behavior_memory_to_trace(mem) for mem in (retrieved_memories or [])
                    ],
                })
                if attempt >= max_retries:
                    break
                current_prompt = f"""{prompt}

The previous answer could not be parsed because:
{last_error}

Retry now. Return ONLY valid JSON with exactly one score row for every candidate_id.
"""

        print(f"Error in LLM scoring/ranking: {last_error}")
        fallback_ids = [str(item["item_id"]) for item in candidate_items]
        self.memory_diagnostics["rank_fallbacks"] += 1
        self._trace("ranking_error", {
            **(trace_context or {}),
            "error": last_error,
            "prompt": prompt,
            "answer": raw_response,
            "parsed": final_result,
            "score_validation": final_validation,
            "candidate_items": candidate_info,
            "aliased_candidate_items": aliased_candidates,
            "fallback_ranked_item_ids": fallback_ids,
            "retrieved_memories": [
                behavior_memory_to_trace(mem) for mem in (retrieved_memories or [])
            ],
        })
        return fallback_ids

    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Phân tích thống kê về evolution của các memories"""
        if not self.behavior_memories:
            return {}
        
        evolution_counts = [m.evolution_count for m in self.behavior_memories]
        
        stats = {
            'total_memories': len(self.behavior_memories),
            'total_evolutions': sum(evolution_counts),
            'avg_evolutions_per_memory': np.mean(evolution_counts),
            'max_evolutions': max(evolution_counts),
            'min_evolutions': min(evolution_counts),
            'std_evolutions': np.std(evolution_counts),
            'memories_never_evolved': sum(1 for c in evolution_counts if c == 0),
            'memories_evolved_once': sum(1 for c in evolution_counts if c == 1),
            'memories_evolved_multiple': sum(1 for c in evolution_counts if c > 1),
            'evolution_distribution': {
                f'{i}_times': sum(1 for c in evolution_counts if c == i)
                for i in range(max(evolution_counts) + 1)
            }
        }
        
        # Top memories theo evolution count
        top_evolved = sorted(
            [(m.thought_id, m.evolution_count, m.behavior_explanation) 
            for m in self.behavior_memories],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        stats['top_10_most_evolved'] = [
            {
                'thought_id': tid,
                'evolution_count': count,
                'behavior': behavior[:100]  # Truncate
            }
            for tid, count, behavior in top_evolved
        ]
        
        return stats

    def print_evolution_report(self):
        """In báo cáo evolution"""
        stats = self.get_evolution_statistics()
        
        print("\n" + "="*80)
        print("MEMORY EVOLUTION REPORT")
        print("="*80)
        print(f"Total Memories: {stats['total_memories']}")
        print(f"Total Evolutions: {stats['total_evolutions']}")
        print(f"Average Evolutions per Memory: {stats['avg_evolutions_per_memory']:.2f}")
        print(f"Max Evolutions: {stats['max_evolutions']}")
        print(f"Min Evolutions: {stats['min_evolutions']}")
        print(f"Std Deviation: {stats['std_evolutions']:.2f}")
        print("-"*80)
        print(f"Never Evolved: {stats['memories_never_evolved']}")
        print(f"Evolved Once: {stats['memories_evolved_once']}")
        print(f"Evolved Multiple Times: {stats['memories_evolved_multiple']}")
        print("-"*80)
        print("Evolution Distribution:")
        for times, count in stats['evolution_distribution'].items():
            if count > 0:
                print(f"  {times}: {count} memories")
        print("-"*80)
        print("Top 10 Most Evolved Memories:")
        for item in stats['top_10_most_evolved']:
            print(f"  ID {item['thought_id']}: {item['evolution_count']} evolutions")
            print(f"    → {item['behavior']}")

    def save_memory(self, filepath: str, format: str = 'json') -> None:
        """
        Lưu memory system ra file (chứa memories của TẤT CẢ users)
        
        Args:
            filepath: Đường dẫn file để lưu
            format: Định dạng file ('json' hoặc 'pickle')
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        if format == 'json':
            memories_dict = [mem.to_dict() for mem in self.behavior_memories]
            interactions_dict = [asdict(interaction) for interaction in self.user_interaction_history]
            
            data = {
                'behavior_memories': memories_dict,
                'user_interaction_history': interactions_dict,
                'next_thought_id': self.next_thought_id,
                'metadata': {
                    'num_memories': len(self.behavior_memories),
                    'num_interactions': len(self.user_interaction_history),
                    'save_timestamp': datetime.now().isoformat()
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            file_size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"✓ Memory saved to {filepath}")
            print(f"  - Format: JSON")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
            print(f"  - File size: {file_size_mb:.2f} MB")
            
        elif format == 'pickle':
            data = {
                'behavior_memories': self.behavior_memories,
                'user_interaction_history': self.user_interaction_history,
                'next_thought_id': self.next_thought_id,
                'metadata': {
                    'num_memories': len(self.behavior_memories),
                    'num_interactions': len(self.user_interaction_history),
                    'save_timestamp': datetime.now().isoformat()
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            file_size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"✓ Memory saved to {filepath}")
            print(f"  - Format: Pickle")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
            print(f"  - File size: {file_size_mb:.2f} MB")
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")

    def load_memory(self, filepath: str, format: str = None) -> None:
        """
        Tải memory system từ file
        
        Args:
            filepath: Đường dẫn file để đọc
            format: Định dạng file ('json' hoặc 'pickle'). Nếu None, tự động detect từ extension
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if format is None:
            if filepath.endswith('.json'):
                format = 'json'
            elif filepath.endswith('.pkl') or filepath.endswith('.pickle'):
                format = 'pickle'
            else:
                try:
                    with open(filepath, 'r') as f:
                        json.load(f)
                    format = 'json'
                except:
                    format = 'pickle'
        
        if format == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.behavior_memories = [
                BehaviorMemory.from_dict(mem_dict) 
                for mem_dict in data['behavior_memories']
            ]
            
            self.user_interaction_history = [
                UserInteraction(**interaction_dict)
                for interaction_dict in data['user_interaction_history']
            ]
            
            self.next_thought_id = data['next_thought_id']
            
            print(f"✓ Memory loaded from {filepath}")
            print(f"  - Format: JSON")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
            
        elif format == 'pickle':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.behavior_memories = data['behavior_memories']
            self.user_interaction_history = data['user_interaction_history']
            self.next_thought_id = data['next_thought_id']
            
            print(f"✓ Memory loaded from {filepath}")
            print(f"  - Format: Pickle")
            print(f"  - Memories: {len(self.behavior_memories)}")
            print(f"  - Interactions: {len(self.user_interaction_history)}")
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'pickle'")

import os
from typing import Dict, List

def save_all_users_ranking_results(all_results: List[Dict],
                                  items_meta: Dict,
                                  output_file: str = "all_users_ranking_results.json"):
    """
    Lưu toàn bộ kết quả ranking của tất cả users vào 1 file JSON duy nhất.
    
    Args:
        all_results: List các dict chứa thông tin của từng user
        items_meta: Metadata items để lấy title, category,...
        output_file: Tên file output (sẽ tự tạo thư mục nếu cần)
    """
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    def get_item_info(item_id: str) -> Dict:
        if item_id in items_meta:
            info = items_meta[item_id]
            return {
                "item_id": item_id,
                "title": item_title(info, item_id),
                "category": item_category(info),
                # "brand": info.get("brand", ""),
                # "price": info.get("price", None)
            }
        else:
            return {
                "item_id": item_id,
                "title": f"Unknown Item {item_id}",
                "category": "Unknown",
                # "brand": "",
                # "price": None
            }
    
    # Chuyển đổi chi tiết items cho tất cả users
    final_results = []
    for res in all_results:
        user_result = {
            "user_id": res["user_id"],
            "num_candidates": len(res["candidates"]),
            "ground_truth_item_ids": res["ground_truth"],
            "candidate_item_ids": res["candidates"],
            "reranked_item_ids": res["predictions"],
            # "ground_truth_items": [get_item_info(iid) for iid in res["ground_truth"]],
            "candidate_items": [get_item_info(iid) for iid in res["candidates"]],
            "reranked_items": [get_item_info(iid) for iid in res["predictions"]],
            "metrics": res["metrics"],  # thêm metrics của user này
            "baseline_metrics": res["baseline_metrics"]
        }
        final_results.append(user_result)
    
    # Lưu vào 1 file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved ranking results of {len(final_results)} users to {output_file}")
    print(f"   File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")


def load_data(items_path: str, sequences_path: str, negatives_path: str):
    """Load Amazon dataset"""
    print("Loading data...")
    
    with open(items_path, 'r', encoding="utf-8") as f:
        items_meta = json.load(f)
    for item_id, item_info in list(items_meta.items()):
        if not isinstance(item_info, dict):
            item_info = {"title": str(item_info)}
            items_meta[item_id] = item_info
        item_info["title"] = item_title(item_info, str(item_id))
        item_info["main_cat"] = item_category(item_info)
        item_info["category_normalized"] = True
    
    with open(sequences_path, 'r', encoding="utf-8") as f:
        user_sequences = json.load(f)
    
    with open(negatives_path, 'r', encoding="utf-8") as f:
        user_negatives = json.load(f)
    
    print(f"Loaded {len(items_meta)} items")
    print(f"Loaded {len(user_sequences)} users")
    
    return items_meta, user_sequences, user_negatives

def calculate_recall_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
    """Calculate Recall@K"""
    top_k = predictions[:k]
    hits = len(set(top_k) & set(ground_truth))
    return hits / len(ground_truth) if ground_truth else 0.0

def calculate_ndcg_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
    """Calculate NDCG@K"""
    top_k = predictions[:k]
    
    # DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    
    # IDCG
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k))])
    
    return dcg / idcg if idcg > 0 else 0.0


PAPER_METRIC_KS = (1, 3, 5, 10, 20)
PAPER_METRIC_NAMES = tuple(
    metric_name
    for k in PAPER_METRIC_KS
    for metric_name in (f"hit@{k}", f"recall@{k}", f"ndcg@{k}")
)


def calculate_paper_ranking_metrics(
    predictions: List[str],
    ground_truth: List[str],
) -> Dict[str, float]:
    """Return paper metrics without requiring another ranking run."""
    metrics: Dict[str, float] = {}
    for k in PAPER_METRIC_KS:
        recall = calculate_recall_at_k(predictions, ground_truth, k)
        metrics[f"hit@{k}"] = 1.0 if recall > 0.0 else 0.0
        metrics[f"recall@{k}"] = recall
        metrics[f"ndcg@{k}"] = calculate_ndcg_at_k(predictions, ground_truth, k)
    return metrics


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
        or os.path.dirname(os.path.abspath(__file__))
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


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


class MemoryGraphIndex:
    """Graph-scoped for retrieving fail lessons.

    Nodes:
    - users
    - items
    - failure lessons

    Edges:
    - user -> train/history item
    - lesson -> source user
    - lesson -> wrong/correct/history item evidence

    Retrieval is graph-scoped first, text-gated second. It does not search the
    global memory pool by embedding similarity.
    """

    def __init__(self, user_sequences: Dict[str, Dict[str, Any]], build_clusters: bool = True):
        self.items_by_user: Dict[str, Set[str]] = defaultdict(set)
        self.users_by_item: Dict[str, Set[str]] = defaultdict(set)
        self.memories_by_user: Dict[str, Set[str]] = defaultdict(set)
        self.memories_by_item: Dict[str, Set[str]] = defaultdict(set)
        # D-family retrieval preserves the polarity of lesson-item edges.  The
        # legacy union index remains unchanged for A/B/C compatibility.
        self.memories_by_correct_item: Dict[str, Set[str]] = defaultdict(set)
        self.memories_by_wrong_item: Dict[str, Set[str]] = defaultdict(set)
        self.memories_by_context_item: Dict[str, Set[str]] = defaultdict(set)
        self.lessons: Dict[str, FailureLesson] = {}
        self.item_category_by_id: Dict[str, str] = {}
        self._shuffled_items_cache: Dict[str, Dict[str, Set[str]]] = {}
        self.last_cf_control_audit: Dict[str, Any] = {}
        self.last_temporal_retrieval_audit: Dict[str, Any] = {}
        self.cluster_corrective_lessons: Dict[str, FailureLesson] = {}
        self.cluster_corrective_by_cluster: Dict[int, List[str]] = defaultdict(list)
        self.cluster_corrective_global: List[str] = []
        self.cluster_corrective_metadata: Dict[str, Any] = {}
        self.cluster_memory_loaded: bool = False
        self.cluster_by_user: Dict[str, int] = {}
        self.users_by_cluster: Dict[int, Set[str]] = defaultdict(set)
        self.lgcn_embeddings: Dict[str, List[float]] = {}
        self.lgcn_item_embeddings: Dict[str, List[float]] = {}
        self.lgcn_cluster_by_user: Dict[str, int] = {}
        self.users_by_lgcn_cluster: Dict[int, Set[str]] = defaultdict(set)
        for user_id, user_data in user_sequences.items():
            for item_id in user_data.get("train", []):
                sid = str(item_id)
                self.items_by_user[str(user_id)].add(sid)
                self.users_by_item[sid].add(str(user_id))
        if build_clusters:
            self._build_user_clusters()

    def configure_item_metadata(self, items_meta: Dict[str, Dict[str, Any]]) -> None:
        """Attach coarse item categories used only by matched CF controls."""
        self.item_category_by_id = {
            str(item_id): item_category(meta).lower()
            for item_id, meta in (items_meta or {}).items()
        }

    def load_cluster_corrective_memory(self, path: str) -> None:
        """Load an offline C-MEMCF artifact built from train-only CF embeddings.

        The artifact contains consolidated, category-level corrections. It never
        overwrites raw failure lessons, so A--J retrieval remains unchanged.
        """
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("format") != "memcf_cluster_corrective_memory_v1":
            raise ValueError(f"Unsupported cluster corrective memory artifact: {path}")

        self.cluster_corrective_lessons = {}
        self.cluster_corrective_by_cluster = defaultdict(list)
        self.cluster_corrective_global = []
        self.cluster_corrective_metadata = dict(payload.get("metadata", {}))
        self.cluster_memory_loaded = True

        raw_cluster_by_user = payload.get("cluster_by_user", {})
        for user_id, cluster_id in raw_cluster_by_user.items():
            self.cluster_by_user[str(user_id)] = int(cluster_id)

        for row in payload.get("cluster_lessons", []):
            lesson = FailureLesson(
                memory_id=str(row["memory_id"]),
                source_user_id=f"cluster:{int(row['cluster_id'])}",
                source_event_id="cluster_consensus",
                lesson=str(row.get("factual_statement", "")),
                prefer=str(row.get("support_category", "")),
                avoid=str(row.get("avoid_category", "")),
                applies_if=[str(x) for x in row.get("history_categories", [])],
                evidence_terms=[str(x) for x in row.get("history_categories", [])],
                failure_type="cluster_category_contrast",
                memory_type="cluster_failure_consensus",
                factual_statement=str(row.get("factual_statement", "")),
                cluster_id=int(row["cluster_id"]),
                cluster_support_users=int(row.get("support_users", 0)),
                cluster_support_events=int(row.get("support_events", 0)),
            )
            self.cluster_corrective_lessons[lesson.memory_id] = lesson
            self.cluster_corrective_by_cluster[lesson.cluster_id].append(lesson.memory_id)

        for row in payload.get("global_lessons", []):
            lesson = FailureLesson(
                memory_id=str(row["memory_id"]),
                source_user_id="cluster:global",
                source_event_id="cluster_consensus_global",
                lesson=str(row.get("factual_statement", "")),
                prefer=str(row.get("support_category", "")),
                avoid=str(row.get("avoid_category", "")),
                applies_if=[str(x) for x in row.get("history_categories", [])],
                evidence_terms=[str(x) for x in row.get("history_categories", [])],
                failure_type="cluster_category_contrast",
                memory_type="cluster_failure_consensus",
                factual_statement=str(row.get("factual_statement", "")),
                cluster_id=-1,
                cluster_support_users=int(row.get("support_users", 0)),
                cluster_support_events=int(row.get("support_events", 0)),
            )
            self.cluster_corrective_lessons[lesson.memory_id] = lesson
            self.cluster_corrective_global.append(lesson.memory_id)

        for cluster_id in self.cluster_corrective_by_cluster:
            self.cluster_corrective_by_cluster[cluster_id].sort()
        self.cluster_corrective_global.sort()

    def _cluster_categories(self, item_ids: List[str]) -> Set[str]:
        return {
            self.item_category_by_id.get(str(item_id), "unknown")
            for item_id in item_ids
            if self.item_category_by_id.get(str(item_id), "unknown") not in {"", "unknown"}
        }

    def _cluster_control_pool(self, user_id: str, control_mode: str, salt: str) -> List[int]:
        """Deterministic search order of alternate (non-true) clusters for a control arm.

        Earlier versions picked exactly one alternate cluster per control mode and
        used it only if that single cluster happened to have an eligible lesson --
        a coin flip that made the true/random/shuffled matched triplet rarely align
        even when several *other* clusters would have matched just as well. This
        returns every non-true cluster ordered by a control-mode-specific hash
        permutation; the caller tries them in order and stops at the first
        eligible match, which still guarantees "not the user's real cluster"
        while no longer gambling on a single draw.
        """
        cluster_ids = sorted(self.cluster_corrective_by_cluster)
        target_cluster = self.cluster_by_user.get(str(user_id))
        candidates = [cluster_id for cluster_id in cluster_ids if cluster_id != target_cluster]
        if not candidates:
            return []

        def rank_key(cluster_id: int) -> str:
            return hashlib.sha256(
                f"cluster-pool::{control_mode}::{user_id}::{salt}::{cluster_id}".encode("utf-8")
            ).hexdigest()

        return sorted(candidates, key=rank_key)

    def _cluster_best_eligible(
        self,
        cluster_id: Optional[int],
        history_categories: Set[str],
        candidate_categories: Set[str],
        is_global: bool = False,
    ) -> Optional[Tuple[FailureLesson, str, int]]:
        """Best cluster lesson (if any) from one cluster's pool for this query.

        A lesson is eligible only when its history-category union overlaps the
        query's recent history AND its support/avoid category is one of today's
        candidate categories. Returns (lesson, role, eligible_count) or None.
        """
        lesson_ids = (
            list(self.cluster_corrective_global)
            if is_global
            else list(self.cluster_corrective_by_cluster.get(cluster_id, []))
        )
        eligible: List[Tuple[Tuple[int, int, int, str], FailureLesson, str]] = []
        for memory_id in lesson_ids:
            lesson = self.cluster_corrective_lessons[memory_id]
            lesson_history = {x for x in lesson.applies_if if x and x != "unknown"}
            support = str(lesson.prefer or "").lower()
            avoid = str(lesson.avoid or "").lower()
            if lesson_history and not (lesson_history & history_categories):
                continue
            support_match = support in candidate_categories
            avoid_match = avoid in candidate_categories
            if not support_match and not avoid_match:
                continue
            role = "cluster_support" if support_match else "cluster_avoid"
            key = (
                int(support_match),
                len(lesson_history & history_categories),
                int(lesson.cluster_support_users),
                lesson.memory_id,
            )
            eligible.append((key, lesson, role))
        if not eligible:
            return None
        eligible.sort(key=lambda row: (-row[0][0], -row[0][1], -row[0][2], row[0][3]))
        _, lesson, role = eligible[0]
        return lesson, role, len(eligible)

    def retrieve_temporal_cluster_consensus(
        self,
        user_id: str,
        recent_history_ids: List[str],
        candidate_ids: List[str],
        scope: str,
        same_k: int,
        cross_k: int,
        control_seed: int,
        shuffle_salt: str,
    ) -> List[GraphRetrievedLesson]:
        """Retrieve one consolidated cluster correction, optionally after personal facts.

        C-MEMCF does not expose peer raw lessons. A cluster correction is eligible
        only when its history category and its support/avoid category both occur in
        the current query. This makes it a bounded residual rather than a broad
        neighbor-memory injection.

        The true/random/shuffled arms form a matched triplet, mirroring J/K/L: a
        fact is only injected for true/random/shuffled scopes when ALL THREE would
        independently find an eligible lesson for this exact query. Each of these
        scopes normally runs as a separate process (one variant per run), so this
        gate must be a pure function of (query, cluster artifact, control_seed) --
        never of which scope is actually running -- for the three processes to
        agree on the same matched subset without sharing state. The global scope
        is a separate, non-matched exploratory arm and is not gated this way.
        """
        started = time.perf_counter()
        if not self.cluster_memory_loaded:
            raise ValueError("C-MEMCF scope requires --cluster_memory_file")

        personal: List[GraphRetrievedLesson] = []
        if scope == "temporal_cluster_residual":
            personal = self.retrieve_temporal_lexicographic(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                scope="temporal_same",
                same_k=same_k,
                cross_k=0,
                control_seed=control_seed,
                shuffle_salt=shuffle_salt,
            )
            # Personal evidence is sufficient: do not add transfer noise.
            if len(personal) >= max(0, int(same_k)):
                self.last_temporal_retrieval_audit = {
                    "protocol": "cmemcf_v1",
                    "scope": scope,
                    "cluster_used": False,
                    "cluster_skip_reason": "personal_memory_sufficient",
                    "personal_fact_count": len(personal),
                    "retrieval_latency_ms": (time.perf_counter() - started) * 1000.0,
                }
                return personal

        control_mode = {
            "temporal_cluster_only": "true",
            "temporal_cluster_residual": "true",
            "temporal_cluster_global": "global",
            "temporal_cluster_random": "random_cluster",
            "temporal_cluster_shuffled": "shuffled_cluster",
        }[scope]
        salt = f"{control_seed}::{shuffle_salt}"
        target_cluster = self.cluster_by_user.get(str(user_id))
        history_categories = self._cluster_categories(recent_history_ids)
        candidate_categories = self._cluster_categories(candidate_ids)

        def pool_hit(control_mode_key: str) -> Tuple[Optional[Tuple[FailureLesson, str, int]], Optional[int]]:
            for pool_cluster_id in self._cluster_control_pool(
                user_id=str(user_id), control_mode=control_mode_key, salt=salt,
            ):
                result = self._cluster_best_eligible(pool_cluster_id, history_categories, candidate_categories)
                if result is not None:
                    return result, pool_cluster_id
            return None, None

        if control_mode == "global":
            cluster_id: Optional[int] = None
            hit = self._cluster_best_eligible(None, history_categories, candidate_categories, is_global=True)
            matched_triplet = True  # global is an independent exploratory arm, not matched-gated
        else:
            true_hit = (
                self._cluster_best_eligible(target_cluster, history_categories, candidate_categories)
                if target_cluster is not None else None
            )
            random_hit, random_cluster_used = pool_hit("random_cluster")
            shuffled_hit, shuffled_cluster_used = pool_hit("shuffled_cluster")
            matched_triplet = bool(true_hit and random_hit and shuffled_hit)
            cluster_id = {
                "true": target_cluster,
                "random_cluster": random_cluster_used,
                "shuffled_cluster": shuffled_cluster_used,
            }[control_mode]
            hit = {
                "true": true_hit,
                "random_cluster": random_hit,
                "shuffled_cluster": shuffled_hit,
            }[control_mode] if matched_triplet else None

        eligible_count = hit[2] if hit else 0
        selected = list(personal)
        if hit is not None and cross_k > 0:
            lesson, role, _ = hit
            selected.append(GraphRetrievedLesson(
                lesson=lesson,
                score=float(lesson.cluster_support_users),
                sources=["cluster_consensus"],
                paths=[
                    f"user:{user_id}->cf_cluster:{cluster_id if cluster_id is not None else 'global'}"
                    f"->consensus:{lesson.memory_id}"
                ],
                matched_evidence_terms=sorted(set(lesson.applies_if) & history_categories),
                exposure_type="cluster_consensus",
                candidate_role=role,
                shared_history_items=sorted(set(lesson.applies_if) & history_categories),
                control_mode=control_mode,
            ))
        self.last_temporal_retrieval_audit = {
            "protocol": "cmemcf_v1",
            "scope": scope,
            "control_mode": control_mode,
            "target_cluster_id": target_cluster,
            "retrieval_cluster_id": cluster_id if control_mode != "global" else "global",
            "personal_fact_count": len(personal),
            "matched_triplet_eligible": matched_triplet,
            "eligible_cluster_lessons": eligible_count,
            "cluster_used": len(selected) > len(personal),
            "cluster_support_users": (
                selected[-1].lesson.cluster_support_users if len(selected) > len(personal) else 0
            ),
            "retrieval_latency_ms": (time.perf_counter() - started) * 1000.0,
        }
        return selected

    def _user_category_profile(self, user_id: str) -> Counter:
        return Counter(
            self.item_category_by_id.get(item_id, "unknown")
            for item_id in self.items_by_user.get(str(user_id), set())
        )

    @staticmethod
    def _counter_l1(left: Counter, right: Counter) -> float:
        # Sort keys before summing so floating-point addition order (and
        # therefore the exact result) is identical across processes. Python
        # randomizes string hash seeds per process by default, so an
        # unordered `set` here made this L1 distance non-deterministic at
        # the ULP level across separate evaluation runs. That was invisible
        # under MEMCF-J's narrow "exact" endpoint matching (few candidate
        # pairs, ties rare) but caused real matched-plan-signature audit
        # violations under MEMCF-K's "candidate_pool" matching, where much
        # larger candidate pools make near-exact source_distance ties common
        # enough for min() to flip between processes on ULP noise. See
        # reports/MEMCF_K_pilot_20260802_results_and_fix.md.
        keys = sorted(set(left) | set(right))
        left_total = max(1, sum(left.values()))
        right_total = max(1, sum(right.values()))
        return sum(
            abs(left.get(key, 0) / left_total - right.get(key, 0) / right_total)
            for key in keys
        )

    def _degree_preserving_shuffled_items(self, seed: int) -> Dict[str, Set[str]]:
        """Shuffle the memory-user interaction graph while preserving degrees.

        Double-edge swaps preserve every included user degree and item degree.
        The graph is cached once per process/seed and never uses held-out labels.
        """
        cache_key = str(int(seed))
        if cache_key in self._shuffled_items_cache:
            return self._shuffled_items_cache[cache_key]

        active_users = sorted(
            user_id for user_id, memory_ids in self.memories_by_user.items()
            if memory_ids and self.items_by_user.get(user_id)
        )
        adjacency = {
            user_id: set(self.items_by_user.get(user_id, set()))
            for user_id in active_users
        }
        edges = [
            (user_id, item_id)
            for user_id in active_users
            for item_id in sorted(adjacency[user_id])
        ]
        rng = random.Random(int(seed))
        attempts = min(500000, max(1000, len(edges) * 5))
        swaps = 0
        for _ in range(attempts):
            if len(edges) < 2:
                break
            left = rng.randrange(len(edges))
            right = rng.randrange(len(edges))
            if left == right:
                continue
            user_a, item_a = edges[left]
            user_b, item_b = edges[right]
            if user_a == user_b or item_a == item_b:
                continue
            if item_b in adjacency[user_a] or item_a in adjacency[user_b]:
                continue
            adjacency[user_a].remove(item_a)
            adjacency[user_b].remove(item_b)
            adjacency[user_a].add(item_b)
            adjacency[user_b].add(item_a)
            edges[left] = (user_a, item_b)
            edges[right] = (user_b, item_a)
            swaps += 1

        self._shuffled_items_cache[cache_key] = adjacency
        return adjacency

    def _matched_random_sources(
        self,
        reference_sources: List[str],
        candidate_pool: List[str],
        budget: int,
    ) -> Tuple[List[str], Dict[str, str]]:
        """Match non-neighbors to true neighbors by activity and memory profile."""
        available = set(str(x) for x in candidate_pool)
        selected: List[str] = []
        matched_to: Dict[str, str] = {}
        for reference in reference_sources[:max(0, int(budget))]:
            if not available:
                break
            ref_items = len(self.items_by_user.get(reference, set()))
            ref_memories = len(self.memories_by_user.get(reference, set()))
            ref_categories = self._user_category_profile(reference)

            def distance(source: str) -> Tuple[float, str]:
                source_items = len(self.items_by_user.get(source, set()))
                source_memories = len(self.memories_by_user.get(source, set()))
                activity = abs(source_items - ref_items) / max(1, source_items, ref_items)
                memory_count = abs(source_memories - ref_memories) / max(1, source_memories, ref_memories)
                category = self._counter_l1(ref_categories, self._user_category_profile(source))
                return activity + memory_count + category, source

            chosen = min(available, key=distance)
            available.remove(chosen)
            selected.append(chosen)
            matched_to[chosen] = reference
        return selected, matched_to

    def _target_cluster_count(self, users: Optional[int] = None) -> int:
        users = max(1, int(users if users is not None else len(self.items_by_user)))
        default_k = max(2, min(50, int(np.sqrt(users)) or 2))
        raw = os.getenv("MEMCF_USER_CLUSTER_COUNT", str(default_k)).strip()
        try:
            return max(1, min(users, int(raw)))
        except Exception:
            return default_k

    def _user_jaccard(self, user_a: str, user_b: str) -> float:
        a = self.items_by_user.get(str(user_a), set())
        b = self.items_by_user.get(str(user_b), set())
        if not a or not b:
            return 0.0
        inter = len(a & b)
        if inter <= 0:
            return 0.0
        return inter / max(1, len(a | b))

    def _choose_cluster_anchors(self, k: int, candidate_users: Optional[Set[str]] = None) -> List[str]:
        if candidate_users:
            users = [u for u in self.items_by_user.keys() if str(u) in candidate_users]
        else:
            users = list(self.items_by_user.keys())
        users = sorted(users, key=lambda u: (-len(self.items_by_user[u]), u))
        if not users:
            return []
        anchors = [users[0]]
        remaining = users[1:]
        while remaining and len(anchors) < k:
            best_user = None
            best_key = None
            for user_id in remaining:
                max_sim = max(self._user_jaccard(user_id, anchor) for anchor in anchors)
                # Farthest-first medoids over item histories, deterministic tie-break.
                key = (1.0 - max_sim, len(self.items_by_user[user_id]), user_id)
                if best_key is None or key > best_key:
                    best_key = key
                    best_user = user_id
            anchors.append(str(best_user))
            remaining = [u for u in remaining if u != best_user]
        return anchors

    def _build_user_clusters(self, candidate_users: Optional[Set[str]] = None) -> None:
        self.cluster_by_user = {}
        self.users_by_cluster = defaultdict(set)
        anchor_pool_size = len(candidate_users) if candidate_users else len(self.items_by_user)
        anchors = self._choose_cluster_anchors(self._target_cluster_count(anchor_pool_size), candidate_users)
        if not anchors:
            return
        for user_id in sorted(self.items_by_user.keys()):
            best_cluster = 0
            best_score = -1.0
            for cluster_id, anchor in enumerate(anchors):
                score = self._user_jaccard(user_id, anchor)
                if score > best_score:
                    best_score = score
                    best_cluster = cluster_id
            self.cluster_by_user[user_id] = best_cluster
            self.users_by_cluster[best_cluster].add(user_id)

    def rebuild_clusters_from_memory_users(self, min_lessons: int = 1) -> None:
        """Build cluster anchors from users that actually own failure lessons.

        Evaluation users are still assigned to the nearest memory-source cluster,
        but cluster retrieval only transfers lessons from memory-bearing peers.
        This avoids diluting 100 memory users inside tens of thousands of
        metadata/runtime users.
        """
        memory_users = {
            str(user_id)
            for user_id, memory_ids in self.memories_by_user.items()
            if len(memory_ids) >= max(1, int(min_lessons))
        }
        if memory_users:
            self._build_user_clusters(candidate_users=memory_users)
        else:
            self._build_user_clusters()

    def cluster_users(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        user_id = str(user_id)
        cluster_id = self.cluster_by_user.get(user_id)
        if cluster_id is None:
            return []
        peers = [
            (other_user, self._user_jaccard(user_id, other_user))
            for other_user in self.users_by_cluster.get(cluster_id, set())
            if other_user != user_id and self.memories_by_user.get(other_user)
        ]
        peers.sort(key=lambda x: (-x[1], x[0]))
        return peers[:top_k]

    def load_lightgcn_embeddings(self, path: str) -> None:
        """Load raw per-user (+ per-item, v2 dump) LightGCN propagated embeddings
        (dump_lightgcn_embeddings.py).

        Purely additive: populates self.lgcn_embeddings for cosine-similarity
        cross-user retrieval (full_lgcn scope), self.lgcn_cluster_by_user via
        KMeans for cluster-membership cross-user retrieval (full_lgcn_cluster
        scope), and self.lgcn_item_embeddings for pooled-vector dense retrieval
        (dense_lgcn / dense_lgcn_agree scopes). Never overwrites A--L retrieval
        or the existing Jaccard-based cluster_by_user/users_by_cluster used by
        cluster_* scopes.
        """
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.lgcn_embeddings = {
            str(uid): [float(x) for x in vec]
            for uid, vec in payload.get("users", {}).items()
        }
        self.lgcn_item_embeddings = {
            str(iid): [float(x) for x in vec]
            for iid, vec in payload.get("items", {}).items()
        }
        raw_k = os.getenv("MEMCF_LGCN_CLUSTER_COUNT", "20").strip()
        try:
            n_clusters = max(2, int(raw_k))
        except Exception:
            n_clusters = 20
        self._build_lgcn_clusters(n_clusters)

    def _build_lgcn_clusters(self, n_clusters: int) -> None:
        self.lgcn_cluster_by_user = {}
        self.users_by_lgcn_cluster = defaultdict(set)
        if not self.lgcn_embeddings:
            return
        try:
            from sklearn.cluster import KMeans
        except Exception:
            return
        user_ids = sorted(self.lgcn_embeddings.keys())
        matrix = np.array([self.lgcn_embeddings[u] for u in user_ids], dtype=np.float32)
        if matrix.shape[0] < 2:
            return
        k = max(2, min(n_clusters, matrix.shape[0]))
        labels = KMeans(n_clusters=k, n_init=10, random_state=2027).fit_predict(matrix)
        for user_id, label in zip(user_ids, labels):
            self.lgcn_cluster_by_user[user_id] = int(label)
            self.users_by_lgcn_cluster[int(label)].add(user_id)

    def _user_lgcn_sim(self, user_a: str, user_b: str) -> float:
        va = self.lgcn_embeddings.get(str(user_a))
        vb = self.lgcn_embeddings.get(str(user_b))
        if not va or not vb:
            return 0.0
        # Embeddings are L2-normalized at dump time, so dot product == cosine similarity.
        return float(sum(a * b for a, b in zip(va, vb)))

    def similar_users_lgcn(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """LightGCN-embedding-cosine analogue of similar_users() (raw co-interaction count)."""
        user_id = str(user_id)
        if user_id not in self.lgcn_embeddings:
            return []
        sims = [
            (other_user, self._user_lgcn_sim(user_id, other_user))
            for other_user in self.lgcn_embeddings.keys()
            if other_user != user_id and self.memories_by_user.get(other_user)
        ]
        sims.sort(key=lambda x: (-x[1], x[0]))
        return sims[:top_k]

    def random_users_lgcn(self, user_id: str, top_k: int = 10, shuffle_salt: str = "") -> List[Tuple[str, float]]:
        """Deterministically-seeded RANDOM sibling of similar_users_lgcn():
        the SAME eligible pool (users with memories, excluding self), the
        SAME real cosine similarity reported per pick (for identical scoring
        arithmetic to full_lgcn), but selection is randomized instead of
        ranked by similarity. Isolates whether picking neighbors BY embedding
        similarity actually matters, vs. any same-count cross-user injection
        -- the same true-vs-random causal logic K/L's matched-triplet design
        already applies to categorical retrieval, applied here to the
        embedding-based mechanism (full_lgcn_random scope).
        """
        user_id = str(user_id)
        eligible = [
            other_user for other_user in self.lgcn_embeddings.keys()
            if other_user != user_id and self.memories_by_user.get(other_user)
        ]
        shuffled = deterministic_shuffle(eligible, salt=f"full_lgcn_random::{user_id}::{shuffle_salt}")
        picked = shuffled[:top_k]
        return [(other_user, self._user_lgcn_sim(user_id, other_user)) for other_user in picked]

    def cluster_users_lgcn(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """LightGCN-KMeans-cluster analogue of cluster_users() (Jaccard medoid clusters)."""
        user_id = str(user_id)
        cluster_id = self.lgcn_cluster_by_user.get(user_id)
        if cluster_id is None:
            return []
        peers = [
            (other_user, self._user_lgcn_sim(user_id, other_user))
            for other_user in self.users_by_lgcn_cluster.get(cluster_id, set())
            if other_user != user_id and self.memories_by_user.get(other_user)
        ]
        peers.sort(key=lambda x: (-x[1], x[0]))
        return peers[:top_k]

    def _lgcn_pool(self, vectors: List[List[float]]) -> Optional[np.ndarray]:
        if not vectors:
            return None
        return np.mean(np.array(vectors, dtype=np.float32), axis=0)

    def _lgcn_lesson_vector(self, lesson: "FailureLesson") -> Optional[np.ndarray]:
        """Mean-pool the LightGCN embeddings of everything a lesson is 'about' --
        its source user, its correct/wrong items, its history items -- into one
        vector in the shared LightGCN space. No hand-picked per-field weights:
        every available vector contributes equally to the mean (parameter-free
        pooling), unlike the additive same_user/candidate_item/history_item/
        neighbor_user weights used by the full/full_lgcn scopes.
        """
        parts: List[List[float]] = []
        uvec = self.lgcn_embeddings.get(str(lesson.source_user_id))
        if uvec is not None:
            parts.append(uvec)
        for item_id in (lesson.correct_item_id, lesson.wrong_item_id):
            if item_id:
                ivec = self.lgcn_item_embeddings.get(str(item_id))
                if ivec is not None:
                    parts.append(ivec)
        for item_id in (lesson.history_item_ids or []):
            ivec = self.lgcn_item_embeddings.get(str(item_id))
            if ivec is not None:
                parts.append(ivec)
        return self._lgcn_pool(parts)

    def _lgcn_query_vector(
        self, user_id: str, candidate_ids: List[str], history_ids: List[str]
    ) -> Optional[np.ndarray]:
        parts: List[List[float]] = []
        uvec = self.lgcn_embeddings.get(str(user_id))
        if uvec is not None:
            parts.append(uvec)
        for item_id in candidate_ids:
            ivec = self.lgcn_item_embeddings.get(str(item_id))
            if ivec is not None:
                parts.append(ivec)
        for item_id in history_ids:
            ivec = self.lgcn_item_embeddings.get(str(item_id))
            if ivec is not None:
                parts.append(ivec)
        return self._lgcn_pool(parts)

    def _lgcn_agreement(self, user_id: str, lesson: "FailureLesson") -> Optional[bool]:
        """Independent validity check for dense_lgcn_agree: does LightGCN's own
        dot-product preference score, evaluated for the QUERYING user (not the
        lesson's source user), agree that correct_item_id should outrank
        wrong_item_id? This targets a different failure mode than similarity:
        a lesson can be topically/collaboratively close (high pooled cosine)
        while still recommending the wrong direction for THIS user -- the same
        gap this project's on-topic/misdirection trace analysis already found
        between K/L's 100% on-topic rate and its ~66% misdirection rate.
        Returns None (treated as non-disqualifying) when either item embedding
        is unavailable, so lessons the model can't evaluate aren't silently
        dropped -- this is a filter on disagreement, not a re-weighting.
        """
        uvec = self.lgcn_embeddings.get(str(user_id))
        cvec = self.lgcn_item_embeddings.get(str(lesson.correct_item_id or ""))
        wvec = self.lgcn_item_embeddings.get(str(lesson.wrong_item_id or ""))
        if uvec is None or cvec is None or wvec is None:
            return None
        u = np.array(uvec, dtype=np.float32)
        agree_score = float(np.dot(u, np.array(cvec, dtype=np.float32)) - np.dot(u, np.array(wvec, dtype=np.float32)))
        return agree_score > 0.0

    def retrieve_dense_lgcn(
        self,
        user_id: str,
        recent_history_ids: List[str],
        candidate_ids: List[str],
        top_k: int = 3,
        require_lgcn_agreement: bool = False,
        randomize: bool = False,
        shuffle_salt: str = "",
    ) -> List["GraphRetrievedLesson"]:
        """Pure dense retrieval: rank every lesson by ONE cosine-similarity score
        against a pooled LightGCN query vector. No hand-set additive weights, no
        same-user/cross-user distinction at the scoring stage -- both compete on
        equal footing in the same learned embedding space (dense_lgcn scope).

        If require_lgcn_agreement is set (dense_lgcn_agree scope), lessons whose
        claimed correct>wrong direction LightGCN itself disagrees with for this
        query user are filtered out before ranking -- a validity gate, not an
        extra weighted term, so it does not reopen the fixed-weight-combination
        question the pure-similarity design was meant to avoid.

        If randomize is set (dense_lgcn_random scope, the ablation control for
        the whole dense_lgcn family), the same eligible pool is used and each
        lesson's REAL cosine similarity is still reported for scoring/bookkeeping,
        but final top_k SELECTION is a deterministically-seeded shuffle instead
        of a similarity-ranked sort -- isolating whether the specific similarity
        ranking earns its keep vs. any same-count set of memory facts.
        """
        user_id = str(user_id)
        qvec = self._lgcn_query_vector(user_id, candidate_ids, recent_history_ids)
        if qvec is None:
            return []
        qnorm_val = float(np.linalg.norm(qvec))
        if qnorm_val < 1e-12:
            return []
        qvec = qvec / qnorm_val

        scored: List[Tuple[float, str]] = []
        filtered_disagree = 0
        for mid, lesson in self.lessons.items():
            if require_lgcn_agreement:
                agree = self._lgcn_agreement(user_id, lesson)
                if agree is False:
                    filtered_disagree += 1
                    continue
            lvec = self._lgcn_lesson_vector(lesson)
            if lvec is None:
                continue
            lnorm_val = float(np.linalg.norm(lvec))
            if lnorm_val < 1e-12:
                continue
            sim = float(np.dot(qvec, lvec / lnorm_val))
            scored.append((sim, mid))

        if randomize:
            sim_by_mid = {mid: sim for sim, mid in scored}
            shuffled_mids = deterministic_shuffle(
                list(sim_by_mid.keys()), salt=f"dense_lgcn_random::{user_id}::{shuffle_salt}"
            )
            scored = [(sim_by_mid[mid], mid) for mid in shuffled_mids]
        else:
            scored.sort(key=lambda x: (-x[0], x[1]))

        retrieved: List[GraphRetrievedLesson] = []
        for sim, mid in scored[:top_k]:
            lesson = self.lessons[mid]
            same_user = str(lesson.source_user_id) == user_id
            tag = "dense_lgcn_random" if randomize else ("dense_lgcn_agree" if require_lgcn_agreement else "dense_lgcn")
            retrieved.append(GraphRetrievedLesson(
                lesson=lesson,
                score=sim,
                sources=[f"{tag}_same_user" if same_user else f"{tag}_cross_user"],
                paths=[f"user:{user_id}->{tag}_cosine:{sim:.3f}->memory:{mid}"],
                matched_evidence_terms=[],
            ))
        return retrieved

    def retrieve_dense_lgcn_consensus(
        self,
        user_id: str,
        recent_history_ids: List[str],
        candidate_ids: List[str],
        top_k: int = 3,
        pool_size: int = 20,
        consensus_sim_threshold: float = 0.5,
    ) -> List["GraphRetrievedLesson"]:
        """Third design, distinct from both dense_lgcn (trust the closest match)
        and dense_lgcn_agree (trust an external CF model's corroboration): trust
        what MULTIPLE independently-retrieved lessons agree on.

        This is C-MEMCF's original hypothesis (collaborative consensus filters
        out single-source noise/misdirection) but fixes the bug that made it
        collapse: C-MEMCF required an EXACT (cluster_id, support_category,
        avoid_category) string match across different users, computed offline
        -- 0/2518 lessons survived that filter on Prime_Pantry, 11-102/2046 on
        Software. Here, "agreement" is embedding PROXIMITY between two lessons'
        correct_item_id vectors (continuous, no exact-match cliff), computed
        live over the small per-query top-pool_size candidate pool that
        dense_lgcn already ranks -- there is no offline consolidation stage, so
        this cannot suffer the same coverage collapse; worst case, support
        counts go to ~0 for everyone and ranking degrades to pure similarity.
        """
        user_id = str(user_id)
        qvec = self._lgcn_query_vector(user_id, candidate_ids, recent_history_ids)
        if qvec is None:
            return []
        qnorm_val = float(np.linalg.norm(qvec))
        if qnorm_val < 1e-12:
            return []
        qvec = qvec / qnorm_val

        pool: List[Tuple[float, str]] = []
        for mid, lesson in self.lessons.items():
            lvec = self._lgcn_lesson_vector(lesson)
            if lvec is None:
                continue
            lnorm_val = float(np.linalg.norm(lvec))
            if lnorm_val < 1e-12:
                continue
            sim = float(np.dot(qvec, lvec / lnorm_val))
            pool.append((sim, mid))
        pool.sort(key=lambda x: (-x[0], x[1]))
        pool = pool[:max(top_k, pool_size)]

        correct_vecs: Dict[str, np.ndarray] = {}
        for _, mid in pool:
            lesson = self.lessons[mid]
            cvec = self.lgcn_item_embeddings.get(str(lesson.correct_item_id or ""))
            if cvec is None:
                continue
            arr = np.array(cvec, dtype=np.float32)
            n = float(np.linalg.norm(arr))
            if n > 1e-12:
                correct_vecs[mid] = arr / n

        scored: List[Tuple[int, float, str]] = []
        for sim, mid in pool:
            support = 0
            if mid in correct_vecs:
                for other_mid, other_vec in correct_vecs.items():
                    if other_mid == mid:
                        continue
                    if float(np.dot(correct_vecs[mid], other_vec)) >= consensus_sim_threshold:
                        support += 1
            scored.append((support, sim, mid))
        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))

        retrieved: List[GraphRetrievedLesson] = []
        for support, sim, mid in scored[:top_k]:
            lesson = self.lessons[mid]
            same_user = str(lesson.source_user_id) == user_id
            tag = f"dense_lgcn_consensus_{'same' if same_user else 'cross'}_user"
            retrieved.append(GraphRetrievedLesson(
                lesson=lesson,
                score=sim,
                sources=[tag, f"consensus_support_{support}"],
                paths=[
                    f"user:{user_id}->dense_lgcn_consensus_cosine:{sim:.3f}"
                    f"_support:{support}->memory:{mid}"
                ],
                matched_evidence_terms=[],
            ))
        return retrieved

    def retrieve_dense_lgcn_consensus_anchored(
        self,
        user_id: str,
        recent_history_ids: List[str],
        candidate_ids: List[str],
        top_k: int = 3,
        pool_size: int = 20,
        consensus_sim_threshold: float = 0.5,
        randomize: bool = False,
        shuffle_salt: str = "",
    ) -> List["GraphRetrievedLesson"]:
        """dense_lgcn_consensus, plus a hard topical-relevance pre-filter.

        Diagnostic finding this session: dense_lgcn/consensus retrieve lessons
        that are only 0.7-10.3% "on-topic" (correct/wrong item actually in the
        current candidate_set), vs A5's 53.6% cross-user on-topic rate --
        pure embedding similarity finds semantically-close lessons that are
        not about anything the user is choosing between right now. This scope
        restricts the eligible pool to lessons "anchored" to the current query
        (correct_item_id or wrong_item_id appears in candidate_ids or
        recent_history_ids) BEFORE ranking/consensus, mirroring A4's hard
        candidate_item/history_item bonus terms but as a filter, not a weight.

        Falls back to the full (unanchored) pool for any user whose anchored
        subset is empty, so this cannot reproduce C-MEMCF's coverage collapse.

        If randomize is set (the paired ablation control, matching the rest of
        the dense_lgcn family's true/random philosophy), the SAME anchored
        pool is used but final top_k selection is a deterministic shuffle
        instead of consensus-ranked -- isolating whether the consensus-ranking
        logic earns its keep once the pool is already topically anchored.
        """
        user_id = str(user_id)
        qvec = self._lgcn_query_vector(user_id, candidate_ids, recent_history_ids)
        if qvec is None:
            return []
        qnorm_val = float(np.linalg.norm(qvec))
        if qnorm_val < 1e-12:
            return []
        qvec = qvec / qnorm_val

        anchor_ids = set(str(x) for x in candidate_ids if x) | set(str(x) for x in recent_history_ids if x)

        def _is_anchored(lesson: "FailureLesson") -> bool:
            return (
                (bool(lesson.correct_item_id) and str(lesson.correct_item_id) in anchor_ids)
                or (bool(lesson.wrong_item_id) and str(lesson.wrong_item_id) in anchor_ids)
            )

        anchored_mids = {mid for mid, lesson in self.lessons.items() if _is_anchored(lesson)}
        eligible_mids = anchored_mids if anchored_mids else set(self.lessons.keys())
        used_fallback = not anchored_mids

        pool: List[Tuple[float, str]] = []
        for mid in eligible_mids:
            lesson = self.lessons[mid]
            lvec = self._lgcn_lesson_vector(lesson)
            if lvec is None:
                continue
            lnorm_val = float(np.linalg.norm(lvec))
            if lnorm_val < 1e-12:
                continue
            sim = float(np.dot(qvec, lvec / lnorm_val))
            pool.append((sim, mid))
        pool.sort(key=lambda x: (-x[0], x[1]))
        pool = pool[:max(top_k, pool_size)]

        correct_vecs: Dict[str, np.ndarray] = {}
        for _, mid in pool:
            lesson = self.lessons[mid]
            cvec = self.lgcn_item_embeddings.get(str(lesson.correct_item_id or ""))
            if cvec is None:
                continue
            arr = np.array(cvec, dtype=np.float32)
            n = float(np.linalg.norm(arr))
            if n > 1e-12:
                correct_vecs[mid] = arr / n

        scored: List[Tuple[int, float, str]] = []
        for sim, mid in pool:
            support = 0
            if mid in correct_vecs:
                for other_mid, other_vec in correct_vecs.items():
                    if other_mid == mid:
                        continue
                    if float(np.dot(correct_vecs[mid], other_vec)) >= consensus_sim_threshold:
                        support += 1
            scored.append((support, sim, mid))

        if randomize:
            mids_in_pool = [mid for _, _, mid in scored]
            shuffled_mids = deterministic_shuffle(
                mids_in_pool, salt=f"dense_lgcn_consensus_anchored_random::{user_id}::{shuffle_salt}"
            )
            by_mid = {mid: (support, sim) for support, sim, mid in scored}
            scored = [(by_mid[mid][0], by_mid[mid][1], mid) for mid in shuffled_mids]
        else:
            scored.sort(key=lambda x: (-x[0], -x[1], x[2]))

        retrieved: List[GraphRetrievedLesson] = []
        for support, sim, mid in scored[:top_k]:
            lesson = self.lessons[mid]
            same_user = str(lesson.source_user_id) == user_id
            base_tag = "dense_lgcn_consensus_anchored_random" if randomize else "dense_lgcn_consensus_anchored"
            tag = f"{base_tag}_{'same' if same_user else 'cross'}_user"
            retrieved.append(GraphRetrievedLesson(
                lesson=lesson,
                score=sim,
                sources=[tag, f"consensus_support_{support}", f"anchor_fallback_{used_fallback}"],
                paths=[
                    f"user:{user_id}->{base_tag}_cosine:{sim:.3f}"
                    f"_support:{support}_fallback:{used_fallback}->memory:{mid}"
                ],
                matched_evidence_terms=[],
            ))
        return retrieved

    def _lgcn_components(self, vectors: List[List[float]]) -> Optional[np.ndarray]:
        """L2-normalize each vector in `vectors` and stack them into an
        (n, dim) matrix WITHOUT pooling into a single mean vector. Used by
        the max-sim family so component-level signals (e.g. a lesson's
        source-user vector) are not diluted by averaging with ~20+ unrelated
        candidate/history vectors before comparison.
        """
        normed_list = []
        for v in vectors:
            arr = np.array(v, dtype=np.float32)
            n = float(np.linalg.norm(arr))
            if n > 1e-12:
                normed_list.append(arr / n)
        if not normed_list:
            return None
        return np.stack(normed_list, axis=0)

    def _lgcn_lesson_components(self, lesson: "FailureLesson") -> Optional[np.ndarray]:
        """Same source vectors as _lgcn_lesson_vector (source user,
        correct/wrong items, history items) but returned as separate
        normalized rows instead of one mean-pooled vector.
        """
        parts: List[List[float]] = []
        uvec = self.lgcn_embeddings.get(str(lesson.source_user_id))
        if uvec is not None:
            parts.append(uvec)
        for item_id in (lesson.correct_item_id, lesson.wrong_item_id):
            if item_id:
                ivec = self.lgcn_item_embeddings.get(str(item_id))
                if ivec is not None:
                    parts.append(ivec)
        for item_id in (lesson.history_item_ids or []):
            ivec = self.lgcn_item_embeddings.get(str(item_id))
            if ivec is not None:
                parts.append(ivec)
        return self._lgcn_components(parts)

    def _lgcn_query_components(
        self, user_id: str, candidate_ids: List[str], history_ids: List[str]
    ) -> Optional[np.ndarray]:
        """Same source vectors as _lgcn_query_vector (user, candidates,
        recent history) but returned as separate normalized rows instead of
        one mean-pooled vector.
        """
        parts: List[List[float]] = []
        uvec = self.lgcn_embeddings.get(str(user_id))
        if uvec is not None:
            parts.append(uvec)
        for item_id in candidate_ids:
            ivec = self.lgcn_item_embeddings.get(str(item_id))
            if ivec is not None:
                parts.append(ivec)
        for item_id in history_ids:
            ivec = self.lgcn_item_embeddings.get(str(item_id))
            if ivec is not None:
                parts.append(ivec)
        return self._lgcn_components(parts)

    def retrieve_dense_lgcn_consensus_maxsim(
        self,
        user_id: str,
        recent_history_ids: List[str],
        candidate_ids: List[str],
        top_k: int = 3,
        pool_size: int = 20,
        consensus_sim_threshold: float = 0.5,
        randomize: bool = False,
        shuffle_salt: str = "",
    ) -> List["GraphRetrievedLesson"]:
        """Max-similarity (parameter-free late-interaction) variant of
        retrieve_dense_lgcn_consensus: keeps every component vector separate
        (no mean-pooling) and scores a lesson by the MAX cosine similarity
        across every (lesson component, query component) pair. A genuine
        same-user lesson's source-user vector equals the query's own user
        vector, so that pair's cosine similarity is exactly 1.0 and surfaces
        on its own merit instead of being diluted by averaging with ~20+
        unrelated candidate/history vectors. No hand-picked bonus weights
        are introduced; only the aggregation function changes from
        mean-pool-then-compare to compare-then-max.
        """
        user_id = str(user_id)
        qcomp = self._lgcn_query_components(user_id, candidate_ids, recent_history_ids)
        if qcomp is None:
            return []

        pool: List[Tuple[float, str]] = []
        for mid, lesson in self.lessons.items():
            lcomp = self._lgcn_lesson_components(lesson)
            if lcomp is None:
                continue
            sim_matrix = lcomp @ qcomp.T
            max_sim = float(sim_matrix.max())
            pool.append((max_sim, mid))
        pool.sort(key=lambda x: (-x[0], x[1]))
        pool = pool[:max(top_k, pool_size)]

        correct_vecs: Dict[str, np.ndarray] = {}
        for _, mid in pool:
            lesson = self.lessons[mid]
            cvec = self.lgcn_item_embeddings.get(str(lesson.correct_item_id or ""))
            if cvec is None:
                continue
            arr = np.array(cvec, dtype=np.float32)
            n = float(np.linalg.norm(arr))
            if n > 1e-12:
                correct_vecs[mid] = arr / n

        scored: List[Tuple[int, float, str]] = []
        for sim, mid in pool:
            support = 0
            if mid in correct_vecs:
                for other_mid, other_vec in correct_vecs.items():
                    if other_mid == mid:
                        continue
                    if float(np.dot(correct_vecs[mid], other_vec)) >= consensus_sim_threshold:
                        support += 1
            scored.append((support, sim, mid))

        if randomize:
            mids_in_pool = [mid for _, _, mid in scored]
            shuffled_mids = deterministic_shuffle(
                mids_in_pool, salt=f"dense_lgcn_consensus_maxsim_random::{user_id}::{shuffle_salt}"
            )
            by_mid = {mid: (support, sim) for support, sim, mid in scored}
            scored = [(by_mid[mid][0], by_mid[mid][1], mid) for mid in shuffled_mids]
        else:
            scored.sort(key=lambda x: (-x[0], -x[1], x[2]))

        retrieved: List[GraphRetrievedLesson] = []
        for support, sim, mid in scored[:top_k]:
            lesson = self.lessons[mid]
            same_user = str(lesson.source_user_id) == user_id
            base_tag = "dense_lgcn_consensus_maxsim_random" if randomize else "dense_lgcn_consensus_maxsim"
            tag = f"{base_tag}_{'same' if same_user else 'cross'}_user"
            retrieved.append(GraphRetrievedLesson(
                lesson=lesson,
                score=sim,
                sources=[tag, f"consensus_support_{support}"],
                paths=[
                    f"user:{user_id}->{base_tag}_maxsim:{sim:.3f}"
                    f"_support:{support}->memory:{mid}"
                ],
                matched_evidence_terms=[],
            ))
        return retrieved

    def _lgcn_user_vec_only(self, user_id: str) -> Optional[np.ndarray]:
        """Normalized single-user LightGCN embedding, kept separate from item
        vectors. Used by the decomposed retrieval variant so the user-identity
        signal is never blended with (and therefore never crowded out by)
        item/topical similarity noise.
        """
        uvec = self.lgcn_embeddings.get(str(user_id))
        if uvec is None:
            return None
        arr = np.array(uvec, dtype=np.float32)
        n = float(np.linalg.norm(arr))
        return arr / n if n > 1e-12 else None

    def _lgcn_item_components_only(self, item_ids: List[str]) -> Optional[np.ndarray]:
        """Normalized item-only component matrix (no user vector mixed in)."""
        parts: List[np.ndarray] = []
        for item_id in item_ids:
            if not item_id:
                continue
            ivec = self.lgcn_item_embeddings.get(str(item_id))
            if ivec is not None:
                arr = np.array(ivec, dtype=np.float32)
                n = float(np.linalg.norm(arr))
                if n > 1e-12:
                    parts.append(arr / n)
        if not parts:
            return None
        return np.stack(parts, axis=0)

    def retrieve_dense_lgcn_decomposed(
        self,
        user_id: str,
        recent_history_ids: List[str],
        candidate_ids: List[str],
        top_k: int = 3,
        pool_per_signal: int = 15,
        randomize: bool = False,
        shuffle_salt: str = "",
    ) -> List["GraphRetrievedLesson"]:
        """Decomposed variant of dense_lgcn_consensus_maxsim: keeps the
        user-identity similarity signal (source-user embedding vs query-user
        embedding) and the item/topical similarity signal (lesson item
        embeddings vs query candidate+history item embeddings) SEPARATE
        instead of blending every component pair into one max (which let
        item-level noise crowd out a literal same-user identity match --
        empirically measured at only 10% same-user coverage across 100 users
        vs A5's ~93%, despite a genuine same-user match always scoring a
        mathematically perfect 1.0 on the user signal alone).

        The candidate pool is the union of the top-N lessons by EACH signal
        separately, guaranteeing a genuine same-user match always enters
        consideration regardless of how its item-level similarity fares.
        Final ranking uses a categorical tiebreak -- same_user status first,
        similarity score second -- mirroring A5's own graph retrieval
        priority (retrieve() sorts `0 if "same_user" in sources else 1`
        before score). No magnitude weights are introduced: only the two
        similarity signals are kept separate instead of pre-blended, and
        ties are broken by the same type-then-score priority A5 already
        uses. Offline (no-LLM) simulation of this exact mechanism measured
        97.0% on-topic / 95.0% same-user coverage on 100 Software users
        (vs A5's 85.3%/93% and dense_lgcn_consensus_maxsim's 62%/10%).
        """
        user_id = str(user_id)
        query_uv = self._lgcn_user_vec_only(user_id)
        q_item_mat = self._lgcn_item_components_only(list(candidate_ids) + list(recent_history_ids))

        user_scores: Dict[str, float] = {}
        item_scores: Dict[str, float] = {}
        for mid, lesson in self.lessons.items():
            if query_uv is not None:
                lu = self._lgcn_user_vec_only(str(lesson.source_user_id))
                user_scores[mid] = float(np.dot(lu, query_uv)) if lu is not None else -1.0
            else:
                user_scores[mid] = -1.0
            if q_item_mat is not None:
                item_ids = [lesson.correct_item_id, lesson.wrong_item_id] + list(lesson.history_item_ids or [])
                li = self._lgcn_item_components_only(item_ids)
                item_scores[mid] = float((li @ q_item_mat.T).max()) if li is not None else -1.0
            else:
                item_scores[mid] = -1.0

        top_by_user = sorted(user_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:pool_per_signal]
        top_by_item = sorted(item_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:pool_per_signal]
        pool_mids = set(mid for mid, _ in top_by_user) | set(mid for mid, _ in top_by_item)

        scored: List[Tuple[int, float, str]] = []
        for mid in pool_mids:
            lesson = self.lessons[mid]
            is_same = str(lesson.source_user_id) == user_id
            best = max(user_scores.get(mid, -1.0), item_scores.get(mid, -1.0))
            scored.append((0 if is_same else 1, -best, mid))

        if randomize:
            mids_in_pool = [mid for _, _, mid in scored]
            shuffled_mids = deterministic_shuffle(
                mids_in_pool, salt=f"dense_lgcn_decomposed_random::{user_id}::{shuffle_salt}"
            )
            by_mid = {mid: (tier, negbest) for tier, negbest, mid in scored}
            scored = [(by_mid[mid][0], by_mid[mid][1], mid) for mid in shuffled_mids]
        else:
            scored.sort(key=lambda x: (x[0], x[1], x[2]))

        retrieved: List[GraphRetrievedLesson] = []
        for tier, negbest, mid in scored[:top_k]:
            lesson = self.lessons[mid]
            same_user = str(lesson.source_user_id) == user_id
            base_tag = "dense_lgcn_decomposed_random" if randomize else "dense_lgcn_decomposed"
            tag = f"{base_tag}_{'same' if same_user else 'cross'}_user"
            # Standard-vocabulary tag ("same_user"/"candidate_item") is required
            # for the downstream curation gate's exact-string source check
            # (`"same_user" in r.sources`, etc.) to recognize this fact at all --
            # without it every fact from this scope is silently rejected
            # regardless of retrieval quality. The descriptive tag is kept too
            # for trace/debugging purposes.
            std_tag = "same_user" if same_user else "candidate_item"
            retrieved.append(GraphRetrievedLesson(
                lesson=lesson,
                score=-negbest,
                sources=[std_tag, tag],
                paths=[f"user:{user_id}->{base_tag}:{-negbest:.3f}->memory:{mid}"],
                matched_evidence_terms=[],
            ))
        return retrieved

    def retrieve_dense_lgcn_decomposed_cross_only(
        self,
        user_id: str,
        recent_history_ids: List[str],
        candidate_ids: List[str],
        top_k: int = 3,
        pool_per_signal: int = 15,
        randomize: bool = False,
        shuffle_salt: str = "",
    ) -> List["GraphRetrievedLesson"]:
        """RQ2 diagnostic #2: identical mechanism to retrieve_dense_lgcn_decomposed,
        but lessons whose source_user_id == the querying user are excluded from
        the pool entirely. In the original decomposed mechanism, same-user
        lessons always win the tier tiebreak, so cross-user lessons only ever
        fill leftover slots -- they never get to demonstrate value on their own.
        This variant forces cross-user retrieval to stand alone: compare against
        A0 (no-memory) to test whether cross-user signal has ANY value in
        isolation, and against its own _random control to test whether real
        cross-user identity (vs. shuffled) matters even with same-user removed.
        """
        user_id = str(user_id)
        query_uv = self._lgcn_user_vec_only(user_id)
        q_item_mat = self._lgcn_item_components_only(list(candidate_ids) + list(recent_history_ids))

        user_scores: Dict[str, float] = {}
        item_scores: Dict[str, float] = {}
        for mid, lesson in self.lessons.items():
            if str(lesson.source_user_id) == user_id:
                continue  # exclude same-user entirely -- the whole point of this variant
            if query_uv is not None:
                lu = self._lgcn_user_vec_only(str(lesson.source_user_id))
                user_scores[mid] = float(np.dot(lu, query_uv)) if lu is not None else -1.0
            else:
                user_scores[mid] = -1.0
            if q_item_mat is not None:
                item_ids = [lesson.correct_item_id, lesson.wrong_item_id] + list(lesson.history_item_ids or [])
                li = self._lgcn_item_components_only(item_ids)
                item_scores[mid] = float((li @ q_item_mat.T).max()) if li is not None else -1.0
            else:
                item_scores[mid] = -1.0

        top_by_user = sorted(user_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:pool_per_signal]
        top_by_item = sorted(item_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:pool_per_signal]
        pool_mids = set(mid for mid, _ in top_by_user) | set(mid for mid, _ in top_by_item)

        scored: List[Tuple[float, str]] = []
        for mid in pool_mids:
            best = max(user_scores.get(mid, -1.0), item_scores.get(mid, -1.0))
            scored.append((-best, mid))

        if randomize:
            mids_in_pool = [mid for _, mid in scored]
            shuffled_mids = deterministic_shuffle(
                mids_in_pool, salt=f"dense_lgcn_decomposed_cross_only_random::{user_id}::{shuffle_salt}"
            )
            by_mid = {mid: negbest for negbest, mid in scored}
            scored = [(by_mid[mid], mid) for mid in shuffled_mids]
        else:
            scored.sort(key=lambda x: (x[0], x[1]))

        base_tag = "dense_lgcn_decomposed_cross_only_random" if randomize else "dense_lgcn_decomposed_cross_only"
        retrieved: List[GraphRetrievedLesson] = []
        for negbest, mid in scored[:top_k]:
            lesson = self.lessons[mid]
            # Standard-vocabulary tag required for the downstream curation gate's
            # exact-string check (see identical note in retrieve_dense_lgcn_decomposed).
            # Every fact here is cross-user by construction, so always "candidate_item".
            retrieved.append(GraphRetrievedLesson(
                lesson=lesson,
                score=-negbest,
                sources=["candidate_item", base_tag],
                paths=[f"user:{user_id}->{base_tag}:{-negbest:.3f}->memory:{mid}"],
                matched_evidence_terms=[],
            ))
        return retrieved

    def retrieve_dense_lgcn_decomposed_scored(
        self,
        user_id: str,
        recent_history_ids: List[str],
        candidate_ids: List[str],
        top_k: int = 3,
        pool_per_signal: int = 15,
        randomize: bool = False,
        shuffle_salt: str = "",
    ) -> List["GraphRetrievedLesson"]:
        """RQ2 diagnostic #3: identical mechanism to retrieve_dense_lgcn_decomposed,
        except the final selection is PURE score-based -- no categorical
        same-user-first tiebreak. In the original, a same-user lesson always
        outranks a cross-user lesson regardless of actual similarity score,
        because same-user is tier 0 and cross-user is tier 1. Here there is
        only one tier: whichever lesson has the higher max(user_score,
        item_score) wins. A same-user lesson still tends to win in practice
        (cosine(v,v)=1.0 is usually the highest attainable score), but a
        strong cross-user match can now legitimately beat a weak same-user
        match, instead of automatically losing by category. Compare against
        its own _random control for the same causal test as the original
        decomposed mechanism, under this less rigid selection rule.
        """
        user_id = str(user_id)
        query_uv = self._lgcn_user_vec_only(user_id)
        q_item_mat = self._lgcn_item_components_only(list(candidate_ids) + list(recent_history_ids))

        user_scores: Dict[str, float] = {}
        item_scores: Dict[str, float] = {}
        for mid, lesson in self.lessons.items():
            if query_uv is not None:
                lu = self._lgcn_user_vec_only(str(lesson.source_user_id))
                user_scores[mid] = float(np.dot(lu, query_uv)) if lu is not None else -1.0
            else:
                user_scores[mid] = -1.0
            if q_item_mat is not None:
                item_ids = [lesson.correct_item_id, lesson.wrong_item_id] + list(lesson.history_item_ids or [])
                li = self._lgcn_item_components_only(item_ids)
                item_scores[mid] = float((li @ q_item_mat.T).max()) if li is not None else -1.0
            else:
                item_scores[mid] = -1.0

        top_by_user = sorted(user_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:pool_per_signal]
        top_by_item = sorted(item_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:pool_per_signal]
        pool_mids = set(mid for mid, _ in top_by_user) | set(mid for mid, _ in top_by_item)

        scored: List[Tuple[float, str]] = []
        for mid in pool_mids:
            best = max(user_scores.get(mid, -1.0), item_scores.get(mid, -1.0))
            scored.append((-best, mid))

        if randomize:
            mids_in_pool = [mid for _, mid in scored]
            shuffled_mids = deterministic_shuffle(
                mids_in_pool, salt=f"dense_lgcn_decomposed_scored_random::{user_id}::{shuffle_salt}"
            )
            by_mid = {mid: negbest for negbest, mid in scored}
            scored = [(by_mid[mid], mid) for mid in shuffled_mids]
        else:
            scored.sort(key=lambda x: (x[0], x[1]))

        base_tag = "dense_lgcn_decomposed_scored_random" if randomize else "dense_lgcn_decomposed_scored"
        retrieved: List[GraphRetrievedLesson] = []
        for negbest, mid in scored[:top_k]:
            lesson = self.lessons[mid]
            same_user = str(lesson.source_user_id) == user_id
            tag = f"{base_tag}_{'same' if same_user else 'cross'}_user"
            std_tag = "same_user" if same_user else "candidate_item"
            retrieved.append(GraphRetrievedLesson(
                lesson=lesson,
                score=-negbest,
                sources=[std_tag, tag],
                paths=[f"user:{user_id}->{base_tag}:{-negbest:.3f}->memory:{mid}"],
                matched_evidence_terms=[],
            ))
        return retrieved

    def _lgcn_top_similar_users(
        self,
        user_id: str,
        query_uv: Optional[np.ndarray],
        top_n: int,
        randomize: bool,
        shuffle_salt: str,
        scope_tag: str,
    ) -> Dict[str, float]:
        """Rank OTHER users purely by LGCN user-embedding cosine similarity --
        the one signal confirmed (by audit of real trace data) not to
        trivially saturate, unlike the item-overlap channel used by
        retrieve_dense_lgcn_decomposed (see that method's user-score vs
        item-score comment, and retrieve_dense_lgcn_userscore_consensus_cross_only's
        docstring below for the audit numbers). Returns
        {source_user_id: similarity} for the top_n most similar users, or --
        if randomize -- a same-size, identity-blind random sample instead.
        """
        user_id = str(user_id)
        source_users = {str(lesson.source_user_id) for lesson in self.lessons.values()} - {user_id}
        if query_uv is None or not source_users:
            return {}
        sims: Dict[str, float] = {}
        for src in source_users:
            lu = self._lgcn_user_vec_only(src)
            if lu is not None:
                sims[src] = float(np.dot(lu, query_uv))
        if randomize:
            all_users = sorted(sims.keys())
            shuffled = deterministic_shuffle(all_users, salt=f"{scope_tag}::{user_id}::{shuffle_salt}")
            picked = shuffled[:top_n]
            return {u: sims[u] for u in picked}
        ranked = sorted(sims.items(), key=lambda kv: (-kv[1], kv[0]))
        return dict(ranked[:top_n])

    def _lgcn_consensus_groups(
        self,
        candidate_ids: List[str],
        similar_users: Dict[str, float],
        min_consensus_users: int,
    ) -> List[Tuple[str, str, int, float, str]]:
        """Group lessons sourced from a fixed pool of similar users by which
        CURRENT candidate item they support/avoid; keep only items with
        agreement from >= min_consensus_users DISTINCT source users. This is
        the causal-signal fix: acceptance now requires several independently
        similar users to agree on the same item, not a single lesson whose
        item ID happens to coincide with today's candidate pool.
        Returns (item_id, direction, n_voters, best_similarity, best_mid),
        sorted by (n_voters desc, best_similarity desc, mid).
        """
        candidate_id_set = {str(c) for c in candidate_ids}
        votes: Dict[Tuple[str, str], List[Tuple[str, str, float]]] = defaultdict(list)
        for mid, lesson in self.lessons.items():
            src = str(lesson.source_user_id)
            sim = similar_users.get(src)
            if sim is None:
                continue
            correct_id = str(lesson.correct_item_id or "")
            wrong_id = str(lesson.wrong_item_id or "")
            if correct_id and correct_id in candidate_id_set:
                votes[(correct_id, "prefer")].append((mid, src, sim))
            if wrong_id and wrong_id in candidate_id_set:
                votes[(wrong_id, "avoid")].append((mid, src, sim))

        groups: List[Tuple[str, str, int, float, str]] = []
        for (item_id, direction), entries in votes.items():
            distinct_users = {src for _, src, _ in entries}
            if len(distinct_users) < min_consensus_users:
                continue
            best_mid, _best_src, best_sim = max(entries, key=lambda e: (e[2], e[0]))
            groups.append((item_id, direction, len(distinct_users), best_sim, best_mid))
        groups.sort(key=lambda g: (-g[2], -g[3], g[4]))
        return groups

    def retrieve_dense_lgcn_userscore_consensus_cross_only(
        self,
        user_id: str,
        recent_history_ids: List[str],
        candidate_ids: List[str],
        top_k: int = 3,
        top_n_similar_users: int = 15,
        min_consensus_users: int = 2,
        randomize: bool = False,
        shuffle_salt: str = "",
    ) -> List["GraphRetrievedLesson"]:
        """RQ2 redesign, cross-user-only diagnostic.

        Root cause this fixes: retrieve_dense_lgcn_decomposed's item-
        similarity channel is not a graduated signal -- it is effectively a
        binary flag, because q_item_mat is built from the CURRENT
        candidate/history items and li from the LESSON's own correct/wrong/
        history items; whenever they share a literal item ID, cosine == 1.0
        exactly (a normalized vector dotted with itself), regardless of any
        real behavioral similarity between the two users. Audited on real
        trace data (1000 users, Software, dense_lgcn_decomposed_cross_only):
        100% of the 3000 cross-user facts that made it into the top-3 had
        score >= 0.999 -- every single one was already saturated, so by the
        time facts reached the prompt, the "ranking" carried essentially no
        differentiating similarity information. That mechanistically explains
        why that mechanism's real-vs-random causal test showed no reliable
        difference (0/4 datasets significant, one significantly *worse*).

        Fix, in two parts:
        1. Rank exclusively by the user-identity embedding signal (see
           _lgcn_top_similar_users) -- confirmed non-degenerate, since two
           distinct real users essentially never score exactly 1.0.
        2. Require CONSENSUS (see _lgcn_consensus_groups): a candidate item is
           only surfaced if >= min_consensus_users independently-similar
           source users agree on the same prefer/avoid direction for it. A
           single similar user's one-off lesson is not enough on its own;
           several agreeing is a much stronger collaborative-filtering
           signal, and one a random-user control should reproduce by chance
           far less often than a real one.

        Facts from this mechanism are tagged "consensus_verified", which the
        downstream gate (read_graph_lessons_as_facets_v2) auto-accepts --
        deliberately bypassing the usual own-profile text-overlap
        requirement, since that requirement is orthogonal to (and was
        previously masking) the actual causal question this mechanism is
        built to answer.
        """
        user_id = str(user_id)
        query_uv = self._lgcn_user_vec_only(user_id)
        base_tag = (
            "dense_lgcn_userscore_consensus_cross_only_random" if randomize
            else "dense_lgcn_userscore_consensus_cross_only"
        )
        similar_users = self._lgcn_top_similar_users(
            user_id, query_uv, top_n_similar_users, randomize, shuffle_salt, base_tag
        )
        groups = self._lgcn_consensus_groups(candidate_ids, similar_users, min_consensus_users)

        retrieved: List[GraphRetrievedLesson] = []
        for item_id, direction, n_voters, best_sim, mid in groups[:top_k]:
            lesson = self.lessons[mid]
            retrieved.append(GraphRetrievedLesson(
                lesson=lesson,
                score=best_sim,
                # "candidate_item" is the standard-vocabulary tag the gate's
                # exact-string source check recognizes generically;
                # "consensus_verified" triggers this mechanism's own
                # auto-accept branch specifically.
                sources=["candidate_item", "consensus_verified", base_tag],
                paths=[f"user:{user_id}->{base_tag}:voters={n_voters}:sim={best_sim:.3f}->memory:{mid}"],
                matched_evidence_terms=[],
            ))
        return retrieved

    def retrieve_dense_lgcn_userscore_consensus(
        self,
        user_id: str,
        recent_history_ids: List[str],
        candidate_ids: List[str],
        top_k: int = 3,
        top_n_similar_users: int = 15,
        min_consensus_users: int = 2,
        randomize: bool = False,
        shuffle_salt: str = "",
    ) -> List["GraphRetrievedLesson"]:
        """Full variant: same-user lessons first (unchanged, already-proven
        tier), then the consensus-verified cross-user mechanism above fills
        any remaining top_k slots. Tests whether a causally-fixed cross-user
        signal can add value ON TOP OF same-user-only -- unlike
        dense_lgcn_decomposed, whose "full" mode did not (0/4 datasets in
        prior testing).
        """
        user_id = str(user_id)
        same_user_mids = [
            mid for mid, lesson in self.lessons.items()
            if str(lesson.source_user_id) == user_id
        ]
        base_tag = "dense_lgcn_userscore_consensus_random" if randomize else "dense_lgcn_userscore_consensus"
        same_user_mids = (
            deterministic_shuffle(same_user_mids, salt=f"{base_tag}_same::{user_id}::{shuffle_salt}")
            if randomize else sorted(same_user_mids)
        )

        retrieved: List[GraphRetrievedLesson] = []
        for mid in same_user_mids[:top_k]:
            lesson = self.lessons[mid]
            retrieved.append(GraphRetrievedLesson(
                lesson=lesson,
                score=1.0,
                sources=["same_user", base_tag],
                paths=[f"user:{user_id}->{base_tag}_same_user->memory:{mid}"],
                matched_evidence_terms=[],
            ))

        remaining = top_k - len(retrieved)
        if remaining > 0:
            cross_base_tag = (
                "dense_lgcn_userscore_consensus_cross_only_random" if randomize
                else "dense_lgcn_userscore_consensus_cross_only"
            )
            cross = self.retrieve_dense_lgcn_userscore_consensus_cross_only(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                top_k=remaining,
                top_n_similar_users=top_n_similar_users,
                min_consensus_users=min_consensus_users,
                randomize=randomize,
                shuffle_salt=shuffle_salt,
            )
            for r in cross:
                r.sources = ["candidate_item", "consensus_verified", base_tag]
                r.paths = [p.replace(cross_base_tag, base_tag) for p in r.paths]
            retrieved.extend(cross)
        return retrieved

    def retrieve_dense_lgcn_fmrec_topk(
        self,
        user_id: str,
        recent_history_ids: List[str],
        candidate_ids: List[str],
        top_k: int = 3,
        top_k_neighbors: int = 3,
        include_self: bool = True,
        randomize: bool = False,
        shuffle_salt: str = "",
        neighbor_mode: str = "lgcn",
        pool_all: bool = False,
    ) -> List["GraphRetrievedLesson"]:
        """Faithful port of a collaborator's independent implementation
        (github.com/dangkh/FMRec, scripts/retrieve_fmrec_lessons.py), added
        for a controlled comparison against this project's own mechanisms on
        the same Amazon datasets/LLM/lesson pool.

        `pool_all=True` is the memory-pool variant the collaborator proposed
        on 2026-09-17: instead of ONE best lesson per donor, gather EVERY
        lesson of the target user and of each of the top_k_neighbors donors
        into a single pool ordered by confidence, and leave the choice of
        which lessons to keep to a downstream selector
        (`--memory_selector llm|heuristic`) or, with no selector, to
        `pack_memory_facts` taking the top few by confidence. Neighbour rows
        keep the consensus_verified tag so the applicability gate passes the
        whole pool through: the selector, not the gate, is what is under test.
        The pool is NOT capped at top_k - 1 like the one-per-donor path; the
        caller widens top_k via memory_selector_top_m.

        Deliberately the OPPOSITE design philosophy from
        retrieve_dense_lgcn_userscore_consensus: no consensus requirement,
        no item-overlap gate, no similarity threshold. For each of the
        top-K OTHER users most similar by LGCN user-embedding cosine, take
        their single highest-confidence lesson and surface it
        UNCONDITIONALLY -- regardless of whether it has anything to do with
        the current candidate set -- exactly mirroring
        `retrieve_neighbor_users` + `best_lesson_for_user` in the original.
        Also includes one personal (same-user) lesson first when
        `include_self=True`, matching the original's default.

        One deliberate scoping difference: the original allows up to
        1 personal + top_k_neighbors facts (4 by default); here the total is
        capped at `top_k` to stay consistent with this project's uniform
        max_memory_facts=3 budget used across every other mechanism in this
        comparison, so no version gets an extra prompt-budget advantage.
        """
        user_id = str(user_id)
        base_tag = "dense_lgcn_fmrec_topk_random" if randomize else "dense_lgcn_fmrec_topk"
        retrieved: List[GraphRetrievedLesson] = []

        lessons_by_user: Dict[str, List[FailureLesson]] = defaultdict(list)
        for lesson in self.lessons.values():
            lessons_by_user[str(lesson.source_user_id)].append(lesson)

        def best_lesson_for(uid: str) -> Optional[FailureLesson]:
            rows = lessons_by_user.get(uid)
            if not rows:
                return None
            return min(rows, key=lambda l: (-l.confidence, l.memory_id))

        def all_lessons_for(uid: str) -> List[FailureLesson]:
            return sorted(lessons_by_user.get(uid, []), key=lambda l: (-l.confidence, l.memory_id))

        if pool_all:
            # ---- memory-pool variant: every lesson of self + every donor ----
            if include_self:
                for les in all_lessons_for(user_id):
                    retrieved.append(GraphRetrievedLesson(
                        lesson=les,
                        score=1.0,
                        sources=["same_user", base_tag, "memory_pool"],
                        paths=[f"user:{user_id}->{base_tag}_pool_personal->memory:{les.memory_id}"],
                        matched_evidence_terms=[],
                    ))
            query_uv = self._lgcn_user_vec_only(user_id)
            similar_users = self._lgcn_top_similar_users(
                user_id, query_uv, top_k_neighbors, randomize, shuffle_salt, base_tag
            )
            ranked_neighbors = sorted(similar_users.items(), key=lambda kv: (-kv[1], kv[0]))
            for neighbor_uid, sim in ranked_neighbors[:top_k_neighbors]:
                for les in all_lessons_for(neighbor_uid):
                    retrieved.append(GraphRetrievedLesson(
                        lesson=les,
                        score=sim,
                        sources=["candidate_item", "consensus_verified", base_tag, "memory_pool"],
                        paths=[f"user:{user_id}->{base_tag}_pool:sim={sim:.3f}->memory:{les.memory_id}"],
                        matched_evidence_terms=[],
                    ))
            # Order the pool by lesson confidence so that with NO selector the
            # packer's top-k is "most confident lessons in the pool", mixing
            # own and cross-user, rather than "own lessons first".
            retrieved.sort(key=lambda r: (-r.lesson.confidence, r.lesson.memory_id))
            return retrieved[:top_k] if top_k > 0 else retrieved

        if include_self:
            own = best_lesson_for(user_id)
            if own is not None:
                retrieved.append(GraphRetrievedLesson(
                    lesson=own,
                    score=1.0,
                    sources=["same_user", base_tag],
                    paths=[f"user:{user_id}->{base_tag}_personal->memory:{own.memory_id}"],
                    matched_evidence_terms=[],
                ))

        remaining = top_k - len(retrieved)
        if remaining > 0:
            if neighbor_mode == "shared":
                # co-interaction count instead of a learned embedding: tests
                # whether the LightGCN space adds anything over naive overlap
                ranked_neighbors = [
                    (u, float(c)) for u, c in self.similar_users(user_id, top_k=top_k_neighbors)
                    if self.memories_by_user.get(u)
                ][:top_k_neighbors]
            elif neighbor_mode == "popular":
                # no query-specific signal at all: just the most prolific users
                pool = [(u, float(len(m))) for u, m in self.memories_by_user.items()
                        if u != str(user_id) and m]
                ranked_neighbors = sorted(pool, key=lambda kv: (-kv[1], kv[0]))[:top_k_neighbors]
            else:
                query_uv = self._lgcn_user_vec_only(user_id)
                similar_users = self._lgcn_top_similar_users(
                    user_id, query_uv, top_k_neighbors, randomize, shuffle_salt, base_tag
                )
                ranked_neighbors = sorted(similar_users.items(), key=lambda kv: (-kv[1], kv[0]))
            for neighbor_uid, sim in ranked_neighbors[:remaining]:
                best = best_lesson_for(neighbor_uid)
                if best is None:
                    continue
                retrieved.append(GraphRetrievedLesson(
                    lesson=best,
                    score=sim,
                    # "candidate_item" + "consensus_verified" reuse the same
                    # gate auto-accept path added for the consensus
                    # mechanism: both want the fact accepted without
                    # requiring it to textually overlap with the querying
                    # user's own profile -- this one because that is the
                    # source design's own choice (no gate at all), not
                    # because independent agreement was verified.
                    sources=["candidate_item", "consensus_verified", base_tag],
                    paths=[f"user:{user_id}->{base_tag}:sim={sim:.3f}->memory:{best.memory_id}"],
                    matched_evidence_terms=[],
                ))

        return retrieved

    def retrieve_dense_lgcn_rrf(
        self,
        user_id: str,
        recent_history_ids: List[str],
        candidate_ids: List[str],
        top_k: int = 3,
        rrf_k: int = 60,
        randomize: bool = False,
        shuffle_salt: str = "",
    ) -> List["GraphRetrievedLesson"]:
        """Reciprocal Rank Fusion (RRF) variant of the decomposed retrieval:
        keeps the user-identity similarity signal and the item/topical
        similarity signal as two SEPARATE RANKED lists (not raw magnitudes
        and not a categorical same_user check), then fuses them via
        RRF score = 1/(rrf_k + rank_user) + 1/(rrf_k + rank_item). This is
        a purely embedding-derived mechanism: no lesson's source_user_id is
        ever compared to the query user_id directly. A genuine same-user
        lesson still naturally lands at rank #1 in the user-similarity list
        (cosine(v, v) = 1.0 is the mathematical maximum, unbeatable), so RRF
        surfaces it without any identity check or magnitude-tuned weight.
        rrf_k=60 is the standard constant from Cormack et al. 2009 (not
        tuned for this task). Offline (no-LLM) simulation of this exact
        mechanism measured 97.0% on-topic / 90.0% same-user coverage on 100
        Software users (vs A5's 85.3%/93%, the categorical-tiebreak
        decomposed variant's 97.0%/95.0%, and the blended
        dense_lgcn_consensus_maxsim's 62%/10%).
        """
        user_id = str(user_id)
        query_uv = self._lgcn_user_vec_only(user_id)
        q_item_mat = self._lgcn_item_components_only(list(candidate_ids) + list(recent_history_ids))

        user_scores: Dict[str, float] = {}
        item_scores: Dict[str, float] = {}
        for mid, lesson in self.lessons.items():
            if query_uv is not None:
                lu = self._lgcn_user_vec_only(str(lesson.source_user_id))
                user_scores[mid] = float(np.dot(lu, query_uv)) if lu is not None else -1.0
            else:
                user_scores[mid] = -1.0
            if q_item_mat is not None:
                item_ids = [lesson.correct_item_id, lesson.wrong_item_id] + list(lesson.history_item_ids or [])
                li = self._lgcn_item_components_only(item_ids)
                item_scores[mid] = float((li @ q_item_mat.T).max()) if li is not None else -1.0
            else:
                item_scores[mid] = -1.0

        user_order = sorted(user_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        item_order = sorted(item_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        user_rank = {mid: i + 1 for i, (mid, _) in enumerate(user_order)}
        item_rank = {mid: i + 1 for i, (mid, _) in enumerate(item_order)}

        rrf_scored: List[Tuple[float, str]] = []
        for mid in self.lessons:
            rrf = 1.0 / (rrf_k + user_rank[mid]) + 1.0 / (rrf_k + item_rank[mid])
            rrf_scored.append((rrf, mid))

        if randomize:
            mids_all = [mid for _, mid in rrf_scored]
            shuffled_mids = deterministic_shuffle(
                mids_all, salt=f"dense_lgcn_rrf_random::{user_id}::{shuffle_salt}"
            )
            by_mid = {mid: rrf for rrf, mid in rrf_scored}
            rrf_scored = [(by_mid[mid], mid) for mid in shuffled_mids]
        else:
            rrf_scored.sort(key=lambda x: (-x[0], x[1]))

        retrieved: List[GraphRetrievedLesson] = []
        for rrf, mid in rrf_scored[:top_k]:
            lesson = self.lessons[mid]
            same_user = str(lesson.source_user_id) == user_id
            base_tag = "dense_lgcn_rrf_random" if randomize else "dense_lgcn_rrf"
            tag = f"{base_tag}_{'same' if same_user else 'cross'}_user"
            # See identical note in retrieve_dense_lgcn_decomposed: the downstream
            # curation gate needs the exact "same_user"/"candidate_item" tag to
            # accept a fact at all, or it always rejects it as "no strong
            # current user-history evidence" regardless of retrieval quality.
            std_tag = "same_user" if same_user else "candidate_item"
            retrieved.append(GraphRetrievedLesson(
                lesson=lesson,
                score=rrf,
                sources=[std_tag, tag],
                paths=[f"user:{user_id}->{base_tag}:{rrf:.5f}->memory:{mid}"],
                matched_evidence_terms=[],
            ))
        return retrieved

    def add_lesson(self, lesson: FailureLesson) -> None:
        self.lessons[lesson.memory_id] = lesson
        self.memories_by_user[lesson.source_user_id].add(lesson.memory_id)
        context_item_ids = set(str(x) for x in lesson.history_item_ids)
        item_ids = set(context_item_ids)
        if lesson.wrong_item_id:
            wrong_item_id = str(lesson.wrong_item_id)
            item_ids.add(wrong_item_id)
            self.memories_by_wrong_item[wrong_item_id].add(lesson.memory_id)
        if lesson.correct_item_id:
            correct_item_id = str(lesson.correct_item_id)
            item_ids.add(correct_item_id)
            self.memories_by_correct_item[correct_item_id].add(lesson.memory_id)
        for item_id in context_item_ids:
            self.memories_by_context_item[item_id].add(lesson.memory_id)
        for item_id in item_ids:
            self.memories_by_item[item_id].add(lesson.memory_id)

    def typed_failure_evidence(
        self,
        user_id: str,
        candidate_ids: List[str],
        user_context_text: str,
        mode: str,
        min_context_terms: int = 1,
        max_same_evidence: int = 32,
        max_cross_evidence: int = 128,
        shuffle_salt: str = "",
        min_shared_items: int = 1,
        cf_source_budget: int = 2,
        cf_control_seed: int = 2027,
    ) -> List[Dict[str, Any]]:
        """Return exact, role-preserving evidence for D-family constraints.

        Candidate membership comes only from typed correct/wrong edges. History
        edges can support applicability but can never directly create a ranking
        action. No held-out label is used here.
        """
        user_id = str(user_id)
        mode = str(mode or "none").strip().lower()
        if mode == "none" or mode == "popularity":
            return []

        include_same = mode in {
            "same_exact", "full_partitioned", "full_consensus",
            "polarity_swapped", "shuffled_provenance",
            "cf_same_plus_shared",
        }
        include_cross = mode in {
            "cross_exact", "full_partitioned", "full_consensus",
            "polarity_swapped", "shuffled_provenance",
            "cf_shared_cross", "cf_same_plus_shared", "cf_shuffled_neighbors",
            "cf_random_neighbors", "cf_polarity_swapped",
            "g_true_neighbor", "g_shuffled_graph", "g_random_neighbor",
            "g_matched_random",
        }
        if not include_same and not include_cross:
            raise ValueError(f"Unsupported failure_constraint_mode={mode}")

        # F-family routing uses the complete training graph rather than a top-k
        # lexical/heuristic neighborhood. This makes the CF path explicit:
        # target user -> shared training item -> source user -> failure edge.
        target_items = self.items_by_user.get(user_id, set())
        memory_users = sorted(
            source_user for source_user, memory_ids in self.memories_by_user.items()
            if source_user != user_id and memory_ids
        )
        shared_by_source = {
            source_user: len(target_items & self.items_by_user.get(source_user, set()))
            for source_user in memory_users
        }
        min_shared_items = max(1, int(min_shared_items))
        true_cf_sources = {
            source_user for source_user, shared in shared_by_source.items()
            if shared >= min_shared_items
        }
        eligible_cf_sources = set(true_cf_sources)
        source_rank: Dict[str, int] = {}
        matched_to: Dict[str, str] = {}
        shuffled_shared_by_source: Dict[str, int] = {}

        candidate_memory_sources: Set[str] = set()
        for candidate_id in candidate_ids:
            for memory_id in (
                set(self.memories_by_correct_item.get(str(candidate_id), set()))
                | set(self.memories_by_wrong_item.get(str(candidate_id), set()))
            ):
                lesson = self.lessons.get(memory_id)
                if lesson and str(lesson.source_user_id) != user_id:
                    candidate_memory_sources.add(str(lesson.source_user_id))

        g_modes = {
            "g_true_neighbor", "g_shuffled_graph", "g_random_neighbor",
            "g_matched_random",
        }
        if mode in g_modes:
            budget = max(1, int(cf_source_budget))
            true_ranked = sorted(
                (source for source in candidate_memory_sources if source in true_cf_sources),
                key=lambda source: (
                    -shared_by_source.get(source, 0),
                    -self._user_jaccard(user_id, source),
                    source,
                ),
            )
            desired = min(budget, len(true_ranked))
            if mode == "g_true_neighbor":
                chosen_sources = true_ranked[:desired]
            elif mode == "g_shuffled_graph":
                shuffled_items = self._degree_preserving_shuffled_items(cf_control_seed)
                target_shuffled = shuffled_items.get(user_id, target_items)
                shuffled_shared_by_source = {
                    source: len(target_shuffled & shuffled_items.get(source, set()))
                    for source in candidate_memory_sources
                }
                # Rank all candidate-linked sources under the shuffled graph,
                # including zero-overlap sources. Keeping exactly `desired`
                # sources makes this a topology control rather than a lower-
                # exposure treatment when edge swaps remove all local overlap.
                chosen_sources = sorted(
                    candidate_memory_sources,
                    key=lambda source: (
                        -shuffled_shared_by_source.get(source, 0),
                        source,
                    ),
                )[:desired]
            else:
                nonneighbors = sorted(candidate_memory_sources - true_cf_sources)
                if mode == "g_random_neighbor":
                    chosen_sources = deterministic_shuffle(
                        nonneighbors,
                        salt=f"g_random::{cf_control_seed}::{user_id}::{shuffle_salt}",
                    )[:desired]
                else:
                    chosen_sources, matched_to = self._matched_random_sources(
                        reference_sources=true_ranked,
                        candidate_pool=nonneighbors,
                        budget=desired,
                    )
            eligible_cf_sources = set(chosen_sources)
            source_rank = {source: rank for rank, source in enumerate(chosen_sources, 1)}
            self.last_cf_control_audit = {
                "mode": mode,
                "user_id": user_id,
                "source_budget": budget,
                "desired_sources": desired,
                "candidate_memory_sources": len(candidate_memory_sources),
                "true_neighbor_sources": len(true_ranked),
                "selected_sources": chosen_sources,
                "selected_source_count": len(chosen_sources),
                "selected_real_shared_items": {
                    source: shared_by_source.get(source, 0) for source in chosen_sources
                },
                "selected_shuffled_shared_items": {
                    source: shuffled_shared_by_source.get(source, 0) for source in chosen_sources
                },
                "selected_jaccard": {
                    source: self._user_jaccard(user_id, source) for source in chosen_sources
                },
                "matched_to_true_source": matched_to,
                "control_seed": int(cf_control_seed),
                "degree_preserving_shuffle": mode == "g_shuffled_graph",
            }
        else:
            self.last_cf_control_audit = {
                "mode": mode,
                "user_id": user_id,
                "true_neighbor_sources": len(true_cf_sources),
            }
        if mode in {"cf_shuffled_neighbors", "cf_random_neighbors"}:
            non_neighbors = [u for u in memory_users if u not in true_cf_sources]
            pool = non_neighbors if non_neighbors else memory_users
            salt_kind = "shuffled" if mode == "cf_shuffled_neighbors" else "random"
            eligible_cf_sources = set(deterministic_shuffle(
                pool,
                salt=f"cf_{salt_kind}::{user_id}::{shuffle_salt}",
            )[:len(true_cf_sources)])

        f_modes = {
            "cf_shared_cross", "cf_same_plus_shared", "cf_shuffled_neighbors",
            "cf_random_neighbors", "cf_polarity_swapped",
            "g_true_neighbor", "g_shuffled_graph", "g_random_neighbor",
            "g_matched_random",
        }
        neighbor_shared = dict(self.similar_users(user_id, top_k=max(10, max_cross_evidence)))
        rows: List[Dict[str, Any]] = []
        candidate_ids = [str(x) for x in candidate_ids]
        for candidate_id in candidate_ids:
            role_groups = (
                ("preferred", self.memories_by_correct_item.get(candidate_id, set())),
                ("wrong", self.memories_by_wrong_item.get(candidate_id, set())),
            )
            for role, memory_ids in role_groups:
                for memory_id in sorted(memory_ids):
                    lesson = self.lessons.get(memory_id)
                    if lesson is None:
                        continue
                    source_user_id = str(lesson.source_user_id)
                    is_same_user = source_user_id == user_id
                    if is_same_user and not include_same:
                        continue
                    if not is_same_user and not include_cross:
                        continue
                    if mode in f_modes and not is_same_user and source_user_id not in eligible_cf_sources:
                        continue

                    evidence_terms = normalize_evidence_terms(
                        list(lesson.evidence_terms or [])
                        + list(lesson.applies_if or [])
                        + [lesson.prefer, lesson.avoid]
                    )
                    matched_terms = [
                        term for term in evidence_terms
                        if term_matches_context(term, user_context_text)
                    ]
                    shared_items = int(shared_by_source.get(
                        source_user_id,
                        neighbor_shared.get(source_user_id, 0),
                    ))
                    context_supported = (
                        is_same_user
                        or (mode in f_modes and source_user_id in eligible_cf_sources)
                        or shared_items > 0
                        or len(matched_terms) >= max(0, int(min_context_terms))
                    )
                    if not context_supported:
                        continue
                    rows.append({
                        "memory_id": lesson.memory_id,
                        "source_user_id": source_user_id,
                        "candidate_item_id": candidate_id,
                        "edge_role": role,
                        "same_user": is_same_user,
                        "shared_history_items": shared_items,
                        "cf_source_is_true_neighbor": source_user_id in true_cf_sources,
                        "cf_source_is_eligible": source_user_id in eligible_cf_sources,
                        "cf_min_shared_items": min_shared_items,
                        "cf_control_mode": mode if mode in g_modes else "",
                        "cf_source_rank": source_rank.get(source_user_id, 0),
                        "cf_source_jaccard": self._user_jaccard(user_id, source_user_id),
                        "cf_shuffled_shared_items": shuffled_shared_by_source.get(source_user_id, 0),
                        "cf_matched_to_true_source": matched_to.get(source_user_id, ""),
                        "matched_user_terms": matched_terms[:12],
                        "confidence": _safe_float(lesson.confidence, 0.5),
                        "overgeneralization_risk": _safe_float(lesson.overgeneralization_risk, 0.5),
                        "correct_item_id": str(lesson.correct_item_id or ""),
                        "wrong_item_id": str(lesson.wrong_item_id or ""),
                        "correct_item_title": lesson.correct_item_title,
                        "wrong_item_title": lesson.wrong_item_title,
                        "retrieval_path": (
                            f"candidate:{candidate_id}->typed_{role}:"
                            f"{lesson.memory_id}->user:{source_user_id}"
                        ),
                    })

        same_rows = [row for row in rows if row["same_user"]]
        cross_rows = [row for row in rows if not row["same_user"]]
        row_key = lambda row: (
            -int(row["shared_history_items"]),
            -len(row["matched_user_terms"]),
            -float(row["confidence"]),
            float(row["overgeneralization_risk"]),
            str(row["candidate_item_id"]),
            str(row["edge_role"]),
            str(row["memory_id"]),
        )
        same_rows.sort(key=row_key)
        cross_rows.sort(key=row_key)
        if mode in g_modes:
            # One candidate-linked failure edge per source makes cross-memory
            # count directly comparable across true and control treatments.
            one_per_source: List[Dict[str, Any]] = []
            seen_sources: Set[str] = set()
            for row in cross_rows:
                source = str(row["source_user_id"])
                if source in seen_sources:
                    continue
                seen_sources.add(source)
                one_per_source.append(row)
            cross_rows = one_per_source
        if max_same_evidence > 0:
            same_rows = same_rows[:max_same_evidence]
        if max_cross_evidence > 0:
            cross_rows = cross_rows[:max_cross_evidence]
        selected = same_rows + cross_rows
        if mode in g_modes:
            self.last_cf_control_audit["selected_evidence_rows"] = len(cross_rows)
            self.last_cf_control_audit["equal_budget_satisfied"] = (
                len(cross_rows) == int(self.last_cf_control_audit.get("desired_sources", 0))
            )

        if mode == "shuffled_provenance" and selected and len(candidate_ids) > 1:
            # Rotate evidence targets by a deterministic non-zero offset. This
            # preserves evidence count/polarity but destroys candidate provenance.
            digest = hashlib.sha256(
                f"{user_id}|{shuffle_salt}|{'|'.join(candidate_ids)}".encode("utf-8")
            ).hexdigest()
            offset = 1 + int(digest[:8], 16) % (len(candidate_ids) - 1)
            remap = {
                candidate_id: candidate_ids[(idx + offset) % len(candidate_ids)]
                for idx, candidate_id in enumerate(candidate_ids)
            }
            selected = [
                {
                    **row,
                    "original_candidate_item_id": row["candidate_item_id"],
                    "candidate_item_id": remap[row["candidate_item_id"]],
                    "retrieval_path": f"shuffled_provenance:{row['retrieval_path']}",
                }
                for row in selected
            ]
        return selected

    def similar_users(self, user_id: str, top_k: int = 10) -> List[Tuple[str, int]]:
        user_id = str(user_id)
        counts: Dict[str, int] = defaultdict(int)
        for item_id in self.items_by_user.get(user_id, set()):
            for other_user in self.users_by_item.get(item_id, set()):
                if other_user != user_id:
                    counts[other_user] += 1
        return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top_k]

    @staticmethod
    def _temporal_exposure_type(lesson: FailureLesson, candidate_set: Set[str]) -> Tuple[str, str]:
        observed_id = str(lesson.observed_next_item_id or lesson.correct_item_id or "")
        selected_id = str(lesson.base_selected_item_id or lesson.wrong_item_id or "")
        observed_present = bool(observed_id and observed_id in candidate_set)
        selected_present = bool(selected_id and selected_id in candidate_set)
        if observed_present and selected_present:
            return "exact_replay", "both"
        if observed_present:
            return "candidate_linked_transfer", "observed_next"
        if selected_present:
            return "candidate_linked_transfer", "base_selected"
        return "abstract_transfer", "none"

    def retrieve_temporal_lexicographic(
        self,
        user_id: str,
        recent_history_ids: List[str],
        candidate_ids: List[str],
        scope: str,
        same_k: int = 2,
        cross_k: int = 1,
        control_seed: int = 2027,
        shuffle_salt: str = "",
    ) -> List[GraphRetrievedLesson]:
        """Route temporal failure evidence without a weighted retrieval score.

        Ordering is categorical and deterministic: same-user candidate-linked,
        same-user abstract transfer, then candidate-linked cross-user evidence.
        Cross-user controls preserve the true treatment's source budget.
        """
        user_id = str(user_id)
        scope = str(scope).strip().lower()
        candidate_set = {str(x) for x in candidate_ids}
        recent_history = {str(x) for x in recent_history_ids}
        target_history = set(self.items_by_user.get(user_id, set()))
        same_k = max(0, int(same_k))
        cross_k = max(0, int(cross_k))

        def is_temporal(lesson: Optional[FailureLesson]) -> bool:
            return bool(lesson and lesson.memory_type == "temporal_failure_contrast")

        def make_row(
            lesson: FailureLesson,
            *,
            sources: List[str],
            exposure_type: str,
            candidate_role: str,
            shared_items: Optional[Set[str]] = None,
            control_mode: str = "true",
        ) -> GraphRetrievedLesson:
            candidate_id = (
                str(lesson.observed_next_item_id or lesson.correct_item_id or "")
                if candidate_role == "observed_next"
                else str(lesson.base_selected_item_id or lesson.wrong_item_id or "")
                if candidate_role == "base_selected"
                else ""
            )
            path = (
                f"temporal:{scope}:user:{user_id}->source:{lesson.source_user_id}"
                f"->memory:{lesson.memory_id}"
            )
            if candidate_id:
                path += f"->candidate:{candidate_id}:{candidate_role}"
            return GraphRetrievedLesson(
                lesson=lesson,
                score=0.0,
                sources=sources,
                paths=[path],
                matched_evidence_terms=[],
                exposure_type=exposure_type,
                candidate_role=candidate_role,
                shared_history_items=sorted(shared_items or set()),
                control_mode=control_mode,
            )

        same_rows: List[GraphRetrievedLesson] = []
        for memory_id in sorted(self.memories_by_user.get(user_id, set())):
            lesson = self.lessons.get(memory_id)
            if not is_temporal(lesson):
                continue
            exposure_type, candidate_role = self._temporal_exposure_type(lesson, candidate_set)
            prefix_overlap = recent_history & {str(x) for x in lesson.history_item_ids}
            if exposure_type == "abstract_transfer" and not prefix_overlap:
                continue
            same_rows.append(make_row(
                lesson,
                sources=["same_user", "temporal_lexicographic"],
                exposure_type=exposure_type,
                candidate_role=candidate_role,
                shared_items=prefix_overlap,
            ))
        same_rows.sort(key=lambda row: (
            0 if row.exposure_type == "candidate_linked_transfer" else
            1 if row.exposure_type == "abstract_transfer" else 2,
            0 if row.candidate_role == "observed_next" else 1,
            row.lesson.memory_id,
        ))

        candidate_linked_by_source: Dict[str, List[GraphRetrievedLesson]] = defaultdict(list)
        candidate_memory_ids: Set[str] = set()
        for candidate_id in candidate_set:
            candidate_memory_ids.update(self.memories_by_correct_item.get(candidate_id, set()))
            candidate_memory_ids.update(self.memories_by_wrong_item.get(candidate_id, set()))
        for memory_id in sorted(candidate_memory_ids):
            lesson = self.lessons.get(memory_id)
            if not is_temporal(lesson) or str(lesson.source_user_id) == user_id:
                continue
            exposure_type, candidate_role = self._temporal_exposure_type(lesson, candidate_set)
            if exposure_type not in {"candidate_linked_transfer", "exact_replay"}:
                continue
            source_user = str(lesson.source_user_id)
            shared_items = target_history & self.items_by_user.get(source_user, set())
            sources = ["candidate_item", "temporal_lexicographic"]
            if shared_items:
                sources.append("neighbor_user")
            sources.append("candidate_correct" if candidate_role == "observed_next" else "candidate_wrong")
            candidate_linked_by_source[source_user].append(make_row(
                lesson,
                sources=sources,
                exposure_type=exposure_type,
                candidate_role=candidate_role,
                shared_items=shared_items,
            ))

        true_sources = sorted(
            (source for source, rows in candidate_linked_by_source.items()
             if rows and target_history & self.items_by_user.get(source, set())),
            key=lambda source: (
                -len(target_history & self.items_by_user.get(source, set())),
                source,
            ),
        )
        desired_sources = min(cross_k, len(true_sources))
        control_mode = "true"
        selected_sources = true_sources[:desired_sources]
        if scope == "temporal_shuffled":
            control_mode = "degree_preserving_shuffled"
            shuffled_items = self._degree_preserving_shuffled_items(control_seed)
            shuffled_target = shuffled_items.get(user_id, target_history)
            selected_sources = sorted(
                candidate_linked_by_source,
                key=lambda source: (
                    -len(shuffled_target & shuffled_items.get(source, set())),
                    source,
                ),
            )[:desired_sources]
        elif scope == "temporal_matched_random":
            control_mode = "matched_random"
            nonneighbors = sorted(set(candidate_linked_by_source) - set(true_sources))
            selected_sources, _ = self._matched_random_sources(
                reference_sources=true_sources,
                candidate_pool=nonneighbors,
                budget=desired_sources,
            )

        cross_rows: List[GraphRetrievedLesson] = []
        for source in selected_sources:
            rows = sorted(candidate_linked_by_source.get(source, []), key=lambda row: (
                0 if row.exposure_type == "candidate_linked_transfer" else 1,
                0 if row.candidate_role == "observed_next" else 1,
                row.lesson.memory_id,
            ))
            if rows:
                row = rows[0]
                row.control_mode = control_mode
                if control_mode != "true":
                    row.sources = [x for x in row.sources if x != "neighbor_user"] + [control_mode]
                cross_rows.append(row)

        if scope == "temporal_exact":
            exact_same = [row for row in same_rows if row.exposure_type == "exact_replay"]
            exact_cross = [
                row for rows in candidate_linked_by_source.values() for row in rows
                if row.exposure_type == "exact_replay"
            ]
            selected = (exact_same + exact_cross)[:max(1, same_k + cross_k)]
        elif scope == "temporal_abstract":
            selected = [row for row in same_rows if row.exposure_type == "abstract_transfer"][:same_k]
        elif scope == "temporal_cross_only":
            selected = [row for row in cross_rows if row.exposure_type != "exact_replay"][:cross_k]
        else:
            safe_same = [row for row in same_rows if row.exposure_type != "exact_replay"][:same_k]
            if scope == "temporal_same":
                selected = safe_same
            else:
                safe_cross = [row for row in cross_rows if row.exposure_type != "exact_replay"][:cross_k]
                selected = safe_same + safe_cross

        self.last_temporal_retrieval_audit = {
            "scope": scope,
            "user_id": user_id,
            "same_budget": same_k,
            "cross_budget": cross_k,
            "desired_cross_sources": desired_sources,
            "selected_cross_sources": selected_sources,
            "true_cross_sources": true_sources[:cross_k],
            "equal_cross_budget": len(selected_sources) == desired_sources,
            "control_mode": control_mode,
            "selected_memory_ids": [row.lesson.memory_id for row in selected],
            "selected_exposure_types": [
                row.exposure_type for row in selected
            ],
            "exposure_types": [row.exposure_type for row in selected],
        }
        return selected

    def retrieve_temporal_matched_j(
        self,
        user_id: str,
        recent_history_ids: List[str],
        candidate_ids: List[str],
        scope: str,
        same_k: int = 2,
        cross_k: int = 1,
        control_seed: int = 2027,
        matched_endpoint_scope: str = "exact",
    ) -> List[GraphRetrievedLesson]:
        """Build one endpoint-matched CF treatment plan for MEMCF-J.

        Exact replay is removed before source budgets are assigned. The true,
        shuffled, and random treatments are paired on the same candidate
        direction (``candidate_role``). By default (``matched_endpoint_scope
        ="exact"``) they are additionally paired on the exact same candidate
        item, matching the original MEMCF-J design. This is a very narrow
        pool: only sources whose own lesson happens to reference that one
        specific item can serve as a shuffled/random partner, which is why
        the original 100-user Software pilot found matched-eligible triplets
        for only 10/100 users (see
        reports/MEMCF_J_priority1_diagnostic_20260731.md). Setting
        ``matched_endpoint_scope="candidate_pool"`` relaxes the pairing to
        "same candidate_role, any item in the current candidate set" so more
        sources qualify as shuffled/random partners, while every selected
        true/shuffled/random suggestion remains grounded in an item the LLM
        is actually choosing among for this query (not an arbitrary,
        off-candidate-set item). The tradeoff: unlike "exact", the specific
        item being praised/avoided (not just the source) can now differ
        between true/shuffled/random, which is a real confound (see
        reports/MEMCF_K_pilot_20260802_results_and_fix.md §recommendations).
        ``matched_endpoint_scope="category_pool"`` is a middle ground: pairs
        on "same candidate_role, same item category" rather than the exact
        item (higher coverage than "exact") or the whole candidate set
        (tighter than "candidate_pool") -- true/shuffled/random still praise
        or avoid topically comparable items, only the source differs. If all
        three treatments are not available, the matched cross slot is empty
        for every causal-control variant.
        """
        started = time.perf_counter()
        user_id = str(user_id)
        scope = str(scope).strip().lower()
        matched_endpoint_scope = str(matched_endpoint_scope or "exact").strip().lower()
        if matched_endpoint_scope not in {"exact", "candidate_pool", "category_pool"}:
            raise ValueError(
                f"Unsupported matched_endpoint_scope={matched_endpoint_scope}"
            )
        candidate_ids = [str(x) for x in candidate_ids]
        candidate_set = set(candidate_ids)
        recent_history = {str(x) for x in recent_history_ids}
        target_history = set(self.items_by_user.get(user_id, set()))
        same_k = max(0, int(same_k))
        cross_k = max(0, int(cross_k))

        def is_temporal(lesson: Optional[FailureLesson]) -> bool:
            return bool(lesson and lesson.memory_type == "temporal_failure_contrast")

        def make_row(
            lesson: FailureLesson,
            *,
            sources: List[str],
            exposure_type: str,
            candidate_role: str,
            shared_items: Optional[Set[str]] = None,
            control_mode: str = "true",
        ) -> GraphRetrievedLesson:
            candidate_id = (
                str(lesson.observed_next_item_id or lesson.correct_item_id or "")
                if candidate_role == "observed_next"
                else str(lesson.base_selected_item_id or lesson.wrong_item_id or "")
                if candidate_role == "base_selected"
                else ""
            )
            path = (
                f"temporal:{scope}:user:{user_id}->source:{lesson.source_user_id}"
                f"->memory:{lesson.memory_id}"
            )
            if candidate_id:
                path += f"->candidate:{candidate_id}:{candidate_role}"
            return GraphRetrievedLesson(
                lesson=lesson,
                score=0.0,
                sources=sources,
                paths=[path],
                matched_evidence_terms=[],
                exposure_type=exposure_type,
                candidate_role=candidate_role,
                shared_history_items=sorted(shared_items or set()),
                control_mode=control_mode,
            )

        def endpoint_key(row: GraphRetrievedLesson) -> Tuple[str, str]:
            lesson = row.lesson
            candidate_id = (
                str(lesson.observed_next_item_id or lesson.correct_item_id or "")
                if row.candidate_role == "observed_next"
                else str(lesson.base_selected_item_id or lesson.wrong_item_id or "")
            )
            return candidate_id, row.candidate_role

        def pairing_key(row: GraphRetrievedLesson) -> Tuple[str, str]:
            if matched_endpoint_scope == "candidate_pool":
                return ("", row.candidate_role)
            if matched_endpoint_scope == "category_pool":
                item_id, role = endpoint_key(row)
                category = self.item_category_by_id.get(item_id, "unknown")
                if category == "unknown":
                    # Do not let a missing/uninformative category silently
                    # become a shared bucket: for catalogs where category
                    # metadata is absent (e.g. Prime Pantry, where every item
                    # resolves to "unknown"), treating "unknown" as a real
                    # category would match almost any two items, degenerating
                    # category_pool into unconstrained role-only matching
                    # without anyone noticing. Fall back to exact item
                    # identity so an unknown-category endpoint only pairs
                    # with literally the same item, same as "exact" scope.
                    return (f"unknown_item::{item_id}", role)
                return (category, role)
            return endpoint_key(row)

        def source_distance(reference: str, source: str) -> Tuple[float, str]:
            ref_items = len(self.items_by_user.get(reference, set()))
            src_items = len(self.items_by_user.get(source, set()))
            ref_memories = len(self.memories_by_user.get(reference, set()))
            src_memories = len(self.memories_by_user.get(source, set()))
            activity = abs(src_items - ref_items) / max(1, src_items, ref_items)
            memory_count = abs(src_memories - ref_memories) / max(
                1, src_memories, ref_memories
            )
            category = self._counter_l1(
                self._user_category_profile(reference),
                self._user_category_profile(source),
            )
            return activity + memory_count + category, source

        same_rows: List[GraphRetrievedLesson] = []
        for memory_id in sorted(self.memories_by_user.get(user_id, set())):
            lesson = self.lessons.get(memory_id)
            if not is_temporal(lesson):
                continue
            exposure_type, candidate_role = self._temporal_exposure_type(
                lesson, candidate_set
            )
            if exposure_type == "exact_replay":
                continue
            prefix_overlap = recent_history & {
                str(x) for x in lesson.history_item_ids
            }
            if exposure_type == "abstract_transfer" and not prefix_overlap:
                continue
            same_rows.append(make_row(
                lesson,
                sources=["same_user", "temporal_j_matched"],
                exposure_type=exposure_type,
                candidate_role=candidate_role,
                shared_items=prefix_overlap,
            ))
        same_rows.sort(key=lambda row: (
            0 if row.exposure_type == "candidate_linked_transfer" else 1,
            0 if row.candidate_role == "observed_next" else 1,
            row.lesson.memory_id,
        ))
        safe_same = same_rows[:same_k]

        candidate_memory_ids: Set[str] = set()
        for candidate_id in candidate_set:
            candidate_memory_ids.update(
                self.memories_by_correct_item.get(candidate_id, set())
            )
            candidate_memory_ids.update(
                self.memories_by_wrong_item.get(candidate_id, set())
            )

        # Rows are indexed by endpoint and source only after exact replay has
        # been removed, so an invalid row cannot consume a source slot.
        rows_by_endpoint_source: Dict[
            Tuple[str, str], Dict[str, List[GraphRetrievedLesson]]
        ] = defaultdict(lambda: defaultdict(list))
        exact_replay_filtered = 0
        for memory_id in sorted(candidate_memory_ids):
            lesson = self.lessons.get(memory_id)
            if not is_temporal(lesson) or str(lesson.source_user_id) == user_id:
                continue
            exposure_type, candidate_role = self._temporal_exposure_type(
                lesson, candidate_set
            )
            if exposure_type == "exact_replay":
                exact_replay_filtered += 1
                continue
            if exposure_type != "candidate_linked_transfer":
                continue
            source_user = str(lesson.source_user_id)
            shared_items = target_history & self.items_by_user.get(
                source_user, set()
            )
            sources = ["candidate_item", "temporal_j_matched"]
            if shared_items:
                sources.append("neighbor_user")
            sources.append(
                "candidate_correct"
                if candidate_role == "observed_next"
                else "candidate_wrong"
            )
            row = make_row(
                lesson,
                sources=sources,
                exposure_type=exposure_type,
                candidate_role=candidate_role,
                shared_items=shared_items,
            )
            rows_by_endpoint_source[pairing_key(row)][source_user].append(row)

        for source_rows in rows_by_endpoint_source.values():
            for rows in source_rows.values():
                rows.sort(key=lambda row: row.lesson.memory_id)

        shuffled_items = self._degree_preserving_shuffled_items(control_seed)
        shuffled_target = shuffled_items.get(user_id, target_history)
        real_overlap = {
            source: len(target_history & self.items_by_user.get(source, set()))
            for sources in rows_by_endpoint_source.values()
            for source in sources
        }
        shuffled_overlap = {
            source: len(shuffled_target & shuffled_items.get(source, set()))
            for sources in rows_by_endpoint_source.values()
            for source in sources
        }

        true_candidates: List[
            Tuple[int, int, str, Tuple[str, str], GraphRetrievedLesson]
        ] = []
        for key, source_rows in rows_by_endpoint_source.items():
            for source, rows in source_rows.items():
                if real_overlap.get(source, 0) <= 0:
                    continue
                role_order = 0 if key[1] == "observed_next" else 1
                true_candidates.append(
                    (-real_overlap[source], role_order, source, key, rows[0])
                )
        true_candidates.sort(
            key=lambda value: (
                value[0], value[1], value[3][0], value[2],
                value[4].lesson.memory_id,
            )
        )

        matched_triplets: List[Dict[str, Any]] = []
        used_true: Set[str] = set()
        used_shuffled: Set[str] = set()
        used_random: Set[str] = set()
        for _, _, true_source, key, true_row in true_candidates:
            if len(matched_triplets) >= cross_k:
                break
            if true_source in used_true:
                continue
            endpoint_sources = rows_by_endpoint_source[key]
            shuffled_pool = [
                source for source in endpoint_sources
                if source != true_source
                and source not in used_shuffled
                and shuffled_overlap.get(source, 0) > 0
            ]
            random_pool = [
                source for source in endpoint_sources
                if source != true_source
                and source not in used_random
                and real_overlap.get(source, 0) == 0
                and shuffled_overlap.get(source, 0) == 0
            ]
            if not shuffled_pool or not random_pool:
                continue
            shuffled_source = min(
                shuffled_pool,
                key=lambda source: source_distance(true_source, source),
            )
            random_source = min(
                random_pool,
                key=lambda source: source_distance(true_source, source),
            )
            shuffled_row = endpoint_sources[shuffled_source][0]
            random_row = endpoint_sources[random_source][0]
            shuffled_row.control_mode = "degree_preserving_shuffled_matched"
            shuffled_row.sources = [
                source for source in shuffled_row.sources
                if source != "neighbor_user"
            ] + ["degree_preserving_shuffled_matched"]
            random_row.control_mode = "endpoint_degree_matched_random"
            random_row.sources = [
                source for source in random_row.sources
                if source != "neighbor_user"
            ] + ["endpoint_degree_matched_random"]
            true_row.control_mode = "true_matched"
            matched_triplets.append({
                "endpoint_item_id": (
                    key[0] if matched_endpoint_scope == "exact"
                    else endpoint_key(true_row)[0]
                ),
                "candidate_role": key[1],
                "matched_endpoint_scope": matched_endpoint_scope,
                "true_source_user_id": true_source,
                "shuffled_source_user_id": shuffled_source,
                "random_source_user_id": random_source,
                "true_memory_id": true_row.lesson.memory_id,
                "shuffled_memory_id": shuffled_row.lesson.memory_id,
                "random_memory_id": random_row.lesson.memory_id,
                "shuffled_endpoint_item_id": endpoint_key(shuffled_row)[0],
                "random_endpoint_item_id": endpoint_key(random_row)[0],
                "true_shared_items": real_overlap.get(true_source, 0),
                "shuffled_graph_shared_items": shuffled_overlap.get(
                    shuffled_source, 0
                ),
                "shuffled_real_shared_items": real_overlap.get(
                    shuffled_source, 0
                ),
                "random_real_shared_items": real_overlap.get(random_source, 0),
                "random_shuffled_shared_items": shuffled_overlap.get(
                    random_source, 0
                ),
                "true_source_degree": len(
                    self.items_by_user.get(true_source, set())
                ),
                "shuffled_source_degree": len(
                    self.items_by_user.get(shuffled_source, set())
                ),
                "random_source_degree": len(
                    self.items_by_user.get(random_source, set())
                ),
                "_true_row": true_row,
                "_shuffled_row": shuffled_row,
                "_random_row": random_row,
            })
            used_true.add(true_source)
            used_shuffled.add(shuffled_source)
            used_random.add(random_source)

        production_rows: List[GraphRetrievedLesson] = []
        production_sources: Set[str] = set()
        for _, _, source, _, row in true_candidates:
            if source in production_sources:
                continue
            production_rows.append(row)
            production_sources.add(source)
            if len(production_rows) >= cross_k:
                break

        if scope == "temporal_j_same":
            selected_cross: List[GraphRetrievedLesson] = []
            control_mode = "same_only"
        elif scope == "temporal_j_full":
            selected_cross = production_rows
            control_mode = "true_production"
        elif scope == "temporal_j_true_matched":
            selected_cross = [
                triplet["_true_row"] for triplet in matched_triplets
            ]
            control_mode = "true_matched"
        elif scope == "temporal_j_shuffled_matched":
            selected_cross = [
                triplet["_shuffled_row"] for triplet in matched_triplets
            ]
            control_mode = "degree_preserving_shuffled_matched"
        elif scope == "temporal_j_random_matched":
            selected_cross = [
                triplet["_random_row"] for triplet in matched_triplets
            ]
            control_mode = "endpoint_degree_matched_random"
        else:
            raise ValueError(f"Unsupported MEMCF-J temporal scope={scope}")

        selected = safe_same + selected_cross[:cross_k]
        serializable_triplets = [
            {
                key: value
                for key, value in triplet.items()
                if not key.startswith("_")
            }
            for triplet in matched_triplets
        ]
        plan_payload = {
            "same_memory_ids": [row.lesson.memory_id for row in safe_same],
            "matched_triplets": serializable_triplets,
        }
        plan_signature = hashlib.sha256(
            json.dumps(plan_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        expected_cross_count = (
            0 if scope == "temporal_j_same"
            else len(production_rows[:cross_k])
            if scope == "temporal_j_full"
            else len(matched_triplets)
        )
        self.last_temporal_retrieval_audit = {
            "protocol": "memcf_j_matched_cf_v1",
            "scope": scope,
            "user_id": user_id,
            "candidate_ids": candidate_ids,
            "same_budget": same_k,
            "cross_budget": cross_k,
            "same_memory_ids": plan_payload["same_memory_ids"],
            "matched_cf_eligible": bool(matched_triplets),
            "matched_triplets": serializable_triplets,
            "matched_plan_signature": plan_signature,
            "production_source_user_ids": [
                row.lesson.source_user_id for row in production_rows
            ],
            "selected_cross_source_user_ids": [
                row.lesson.source_user_id for row in selected_cross[:cross_k]
            ],
            "selected_memory_ids": [row.lesson.memory_id for row in selected],
            "expected_cross_count": expected_cross_count,
            "selected_cross_count_pre_pack": len(selected_cross[:cross_k]),
            "equal_pre_pack_budget": (
                len(matched_triplets)
                == len([triplet["_true_row"] for triplet in matched_triplets])
                == len([triplet["_shuffled_row"] for triplet in matched_triplets])
                == len([triplet["_random_row"] for triplet in matched_triplets])
            ),
            "exact_replay_filtered_before_budget": exact_replay_filtered,
            "control_mode": control_mode,
            "retrieval_latency_ms": (time.perf_counter() - started) * 1000.0,
        }
        return selected

    def retrieve(
        self,
        user_id: str,
        recent_history_ids: List[str],
        candidate_ids: List[str],
        current_context_text: str,
        top_k: int = 3,
        neighbor_k: int = 10,
        dense_pool_per_signal: int = 15,
        consensus_top_n_users: int = 15,
        consensus_min_users: int = 2,
        fmrec_top_k_neighbors: int = 3,
        min_evidence_terms: int = 1,
        retrieval_scope: str = "full",
        shuffle_salt: str = "",
        temporal_same_k: int = 2,
        temporal_cross_k: int = 1,
        temporal_control_seed: int = 2027,
        matched_endpoint_scope: str = "exact",
    ) -> List[GraphRetrievedLesson]:
        user_id = str(user_id)
        scores: Dict[str, float] = defaultdict(float)
        sources: Dict[str, Set[str]] = defaultdict(set)
        paths: Dict[str, List[str]] = defaultdict(list)
        candidate_set = set(str(x) for x in candidate_ids)
        history_set = set(str(x) for x in recent_history_ids)
        retrieval_scope = str(retrieval_scope or "full").strip().lower()
        if retrieval_scope in {"graph", "graph_only", "fail_graph"}:
            retrieval_scope = "full"
        if retrieval_scope not in {
            "full", "same_user", "candidate_item", "history_item",
            "neighbor_user", "same_user_first", "candidate_strict",
            "safe_residual", "directional_residual",
            "cross_user_only", "cluster_user", "cluster_full",
            "hybrid_cluster", "hybrid_cluster_strict",
            "random_memory", "shuffled_memory", "random_memory_clean", "shuffled_memory_clean",
            "random_cluster", "shuffled_cluster",
            "full_lgcn", "full_lgcn_cluster", "full_lgcn_random",
            "dense_lgcn", "dense_lgcn_agree", "dense_lgcn_consensus", "dense_lgcn_random",
            "dense_lgcn_consensus_anchored", "dense_lgcn_consensus_anchored_random",
            "dense_lgcn_consensus_maxsim", "dense_lgcn_consensus_maxsim_random",
            "dense_lgcn_decomposed", "dense_lgcn_decomposed_random",
            "dense_lgcn_decomposed_cross_only", "dense_lgcn_decomposed_cross_only_random",
            "dense_lgcn_decomposed_scored", "dense_lgcn_decomposed_scored_random",
            "dense_lgcn_userscore_consensus", "dense_lgcn_userscore_consensus_random",
            "dense_lgcn_userscore_consensus_cross_only", "dense_lgcn_userscore_consensus_cross_only_random",
            "dense_lgcn_fmrec_topk", "dense_lgcn_fmrec_topk_random",
            "dense_lgcn_fmrec_topk_shared", "dense_lgcn_fmrec_topk_popular",
            "dense_lgcn_fmrec_topk_noself", "dense_lgcn_fmrec_topk_noself_random",
            "dense_lgcn_fmrec_pool", "dense_lgcn_fmrec_pool_random",
            "dense_lgcn_rrf", "dense_lgcn_rrf_random",
            "temporal_same", "temporal_exact", "temporal_abstract",
            "temporal_full", "temporal_cross_only", "temporal_shuffled",
            "temporal_matched_random",
            "temporal_cluster_only", "temporal_cluster_residual",
            "temporal_cluster_global", "temporal_cluster_random",
            "temporal_cluster_shuffled",
            "temporal_j_same", "temporal_j_full",
            "temporal_j_true_matched", "temporal_j_shuffled_matched",
            "temporal_j_random_matched",
            "oracle_cross_candidate",
        }:
            raise ValueError(f"Unsupported graph_retrieval_scope={retrieval_scope}")

        if retrieval_scope in {"dense_lgcn", "dense_lgcn_agree", "dense_lgcn_random"}:
            return self.retrieve_dense_lgcn(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                top_k=top_k,
                require_lgcn_agreement=(retrieval_scope == "dense_lgcn_agree"),
                randomize=(retrieval_scope == "dense_lgcn_random"),
                shuffle_salt=shuffle_salt,
            )

        if retrieval_scope == "dense_lgcn_consensus":
            return self.retrieve_dense_lgcn_consensus(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                top_k=top_k,
            )

        if retrieval_scope in {"dense_lgcn_consensus_anchored", "dense_lgcn_consensus_anchored_random"}:
            return self.retrieve_dense_lgcn_consensus_anchored(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                top_k=top_k,
                randomize=(retrieval_scope == "dense_lgcn_consensus_anchored_random"),
                shuffle_salt=shuffle_salt,
            )

        if retrieval_scope in {"dense_lgcn_consensus_maxsim", "dense_lgcn_consensus_maxsim_random"}:
            return self.retrieve_dense_lgcn_consensus_maxsim(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                top_k=top_k,
                randomize=(retrieval_scope == "dense_lgcn_consensus_maxsim_random"),
                shuffle_salt=shuffle_salt,
            )

        if retrieval_scope in {"dense_lgcn_decomposed", "dense_lgcn_decomposed_random"}:
            return self.retrieve_dense_lgcn_decomposed(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                top_k=top_k,
                pool_per_signal=dense_pool_per_signal,
                randomize=(retrieval_scope == "dense_lgcn_decomposed_random"),
                shuffle_salt=shuffle_salt,
            )

        if retrieval_scope in {"dense_lgcn_decomposed_cross_only", "dense_lgcn_decomposed_cross_only_random"}:
            return self.retrieve_dense_lgcn_decomposed_cross_only(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                top_k=top_k,
                pool_per_signal=dense_pool_per_signal,
                randomize=(retrieval_scope == "dense_lgcn_decomposed_cross_only_random"),
                shuffle_salt=shuffle_salt,
            )

        if retrieval_scope in {"dense_lgcn_decomposed_scored", "dense_lgcn_decomposed_scored_random"}:
            return self.retrieve_dense_lgcn_decomposed_scored(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                top_k=top_k,
                pool_per_signal=dense_pool_per_signal,
                randomize=(retrieval_scope == "dense_lgcn_decomposed_scored_random"),
                shuffle_salt=shuffle_salt,
            )

        if retrieval_scope in {"dense_lgcn_userscore_consensus", "dense_lgcn_userscore_consensus_random"}:
            return self.retrieve_dense_lgcn_userscore_consensus(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                top_k=top_k,
                top_n_similar_users=consensus_top_n_users,
                min_consensus_users=consensus_min_users,
                randomize=(retrieval_scope == "dense_lgcn_userscore_consensus_random"),
                shuffle_salt=shuffle_salt,
            )

        if retrieval_scope in {
            "dense_lgcn_userscore_consensus_cross_only",
            "dense_lgcn_userscore_consensus_cross_only_random",
        }:
            return self.retrieve_dense_lgcn_userscore_consensus_cross_only(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                top_k=top_k,
                top_n_similar_users=consensus_top_n_users,
                min_consensus_users=consensus_min_users,
                randomize=(retrieval_scope == "dense_lgcn_userscore_consensus_cross_only_random"),
                shuffle_salt=shuffle_salt,
            )

        if retrieval_scope == "dense_lgcn_fmrec_topk_noself":
            return self.retrieve_dense_lgcn_fmrec_topk(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                top_k=top_k,
                top_k_neighbors=fmrec_top_k_neighbors,
                include_self=False,
                randomize=False,
                shuffle_salt=shuffle_salt,
            )
        if retrieval_scope == "dense_lgcn_fmrec_topk_noself_random":
            return self.retrieve_dense_lgcn_fmrec_topk(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                top_k=top_k,
                top_k_neighbors=fmrec_top_k_neighbors,
                include_self=False,
                randomize=True,
                shuffle_salt=shuffle_salt,
            )
        if retrieval_scope in {"dense_lgcn_fmrec_topk_shared", "dense_lgcn_fmrec_topk_popular"}:
            return self.retrieve_dense_lgcn_fmrec_topk(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                top_k=top_k,
                top_k_neighbors=fmrec_top_k_neighbors,
                include_self=True,
                randomize=False,
                shuffle_salt=shuffle_salt,
                neighbor_mode=("shared" if retrieval_scope.endswith("_shared") else "popular"),
            )
        if retrieval_scope in {"dense_lgcn_fmrec_topk", "dense_lgcn_fmrec_topk_random"}:
            return self.retrieve_dense_lgcn_fmrec_topk(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                top_k=top_k,
                top_k_neighbors=fmrec_top_k_neighbors,
                include_self=True,
                randomize=(retrieval_scope == "dense_lgcn_fmrec_topk_random"),
                shuffle_salt=shuffle_salt,
            )
        if retrieval_scope in {"dense_lgcn_fmrec_pool", "dense_lgcn_fmrec_pool_random"}:
            return self.retrieve_dense_lgcn_fmrec_topk(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                top_k=top_k,
                top_k_neighbors=fmrec_top_k_neighbors,
                include_self=True,
                randomize=(retrieval_scope == "dense_lgcn_fmrec_pool_random"),
                shuffle_salt=shuffle_salt,
                pool_all=True,
            )

        if retrieval_scope in {"dense_lgcn_rrf", "dense_lgcn_rrf_random"}:
            return self.retrieve_dense_lgcn_rrf(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                top_k=top_k,
                randomize=(retrieval_scope == "dense_lgcn_rrf_random"),
                shuffle_salt=shuffle_salt,
            )

        if retrieval_scope.startswith("temporal_j_"):
            return self.retrieve_temporal_matched_j(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                scope=retrieval_scope,
                same_k=temporal_same_k,
                cross_k=temporal_cross_k,
                control_seed=temporal_control_seed,
                matched_endpoint_scope=matched_endpoint_scope,
            )

        if retrieval_scope.startswith("temporal_cluster_"):
            return self.retrieve_temporal_cluster_consensus(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                scope=retrieval_scope,
                same_k=temporal_same_k,
                cross_k=temporal_cross_k,
                control_seed=temporal_control_seed,
                shuffle_salt=shuffle_salt,
            )

        if retrieval_scope.startswith("temporal_"):
            return self.retrieve_temporal_lexicographic(
                user_id=user_id,
                recent_history_ids=recent_history_ids,
                candidate_ids=candidate_ids,
                scope=retrieval_scope,
                same_k=temporal_same_k,
                cross_k=temporal_cross_k,
                control_seed=temporal_control_seed,
                shuffle_salt=shuffle_salt,
            )

        if retrieval_scope in {"random_memory", "random_memory_clean", "random_cluster"}:
            mids = deterministic_shuffle(list(self.lessons.keys()), salt=f"random_memory::{user_id}::{shuffle_salt}")
            if retrieval_scope == "random_memory_clean":
                # Clean random control: prefer memories with no evidence-term overlap
                # with the current user/candidate context. This avoids measuring
                # accidental in-domain transfer as a valid random-memory gain.
                nonoverlap = []
                overlap = []
                for mid in mids:
                    lesson = self.lessons[mid]
                    evidence_terms = normalize_evidence_terms(
                        list(lesson.evidence_terms or [])
                        + list(lesson.applies_if or [])
                        + [lesson.prefer, lesson.avoid, lesson.correct_item_title, lesson.wrong_item_title]
                    )
                    has_overlap = any(term_matches_context(term, current_context_text) for term in evidence_terms)
                    (overlap if has_overlap else nonoverlap).append(mid)
                mids = nonoverlap + overlap
            source = (
                "random_cluster" if retrieval_scope == "random_cluster"
                else "random_memory_clean" if retrieval_scope == "random_memory_clean"
                else "random_memory"
            )
            return [
                GraphRetrievedLesson(
                    lesson=self.lessons[mid],
                    score=0.0,
                    sources=[source],
                    paths=[f"{retrieval_scope}_control:{mid}"],
                    matched_evidence_terms=[],
                )
                for mid in mids[:top_k]
            ]

        if retrieval_scope == "directional_residual":
            # H-family retrieval keeps the polarity of failure edges. The legacy
            # item index mixes history/correct/wrong links and therefore cannot
            # tell whether a candidate should be supported or avoided.
            directional_scores: Dict[str, float] = {}
            directional_sources: Dict[str, Set[str]] = defaultdict(set)
            directional_paths: Dict[str, List[str]] = defaultdict(list)

            def add_path(mid: str, source: str, path: str) -> None:
                lesson = self.lessons.get(mid)
                if lesson is None:
                    return
                confidence = min(1.0, max(0.0, _safe_float(lesson.confidence, 0.5)))
                risk = min(1.0, max(0.0, _safe_float(lesson.overgeneralization_risk, 0.5)))
                directional_scores[mid] = max(
                    directional_scores.get(mid, 0.0), confidence * (1.0 - risk)
                )
                directional_sources[mid].add(source)
                directional_paths[mid].append(path)

            for mid in self.memories_by_user.get(user_id, set()):
                add_path(mid, "same_user", f"user:{user_id}->memory:{mid}")

            for item_id in candidate_set:
                for mid in self.memories_by_correct_item.get(item_id, set()):
                    add_path(mid, "candidate_item", f"candidate_correct:{item_id}->memory:{mid}")
                    directional_sources[mid].add("candidate_correct")
                for mid in self.memories_by_wrong_item.get(item_id, set()):
                    add_path(mid, "candidate_item", f"candidate_wrong:{item_id}->memory:{mid}")
                    directional_sources[mid].add("candidate_wrong")

            for item_id in history_set:
                for mid in self.memories_by_context_item.get(item_id, set()):
                    add_path(mid, "history_item", f"history_context:{item_id}->memory:{mid}")

            for other_user, shared_count in self.similar_users(user_id, top_k=neighbor_k):
                for mid in self.memories_by_user.get(other_user, set()):
                    add_path(
                        mid,
                        "neighbor_user",
                        f"user:{user_id}->shared_items:{shared_count}->user:{other_user}->memory:{mid}",
                    )

            directional_rows: List[GraphRetrievedLesson] = []
            for mid, quality in directional_scores.items():
                lesson = self.lessons[mid]
                evidence_terms = normalize_evidence_terms(
                    list(lesson.evidence_terms or [])
                    + list(lesson.applies_if or [])
                    + [lesson.prefer, lesson.avoid]
                )
                matched_terms = [
                    term for term in evidence_terms
                    if term_matches_context(term, current_context_text)
                ]
                source_set = directional_sources[mid]
                is_same = "same_user" in source_set
                has_candidate = "candidate_item" in source_set
                has_shared_context = bool({"neighbor_user", "history_item"} & source_set)
                # Keep same-user fallback and candidate+shared-context cross-user
                # evidence. Other rows cannot pass collaborative consensus.
                if not is_same and not (has_candidate and has_shared_context):
                    continue
                directional_rows.append(GraphRetrievedLesson(
                    lesson=lesson,
                    score=quality,
                    sources=sorted(source_set),
                    paths=directional_paths[mid],
                    matched_evidence_terms=matched_terms,
                ))

            directional_rows.sort(key=lambda row: (
                0 if "same_user" in row.sources else 1,
                0 if "candidate_correct" in row.sources else 1,
                -len(row.matched_evidence_terms),
                -row.score,
                row.lesson.memory_id,
            ))
            return directional_rows[:top_k]

        for mid in self.memories_by_user.get(user_id, set()):
            if retrieval_scope in {
                "full", "same_user", "same_user_first", "candidate_strict", "safe_residual",
                "shuffled_memory", "shuffled_memory_clean",
                "cluster_full", "hybrid_cluster", "hybrid_cluster_strict", "shuffled_cluster",
                "full_lgcn", "full_lgcn_cluster", "full_lgcn_random",
            }:
                scores[mid] += 3.0
                sources[mid].add("same_user")
                paths[mid].append(f"user:{user_id}->memory:{mid}")

        if retrieval_scope in {
            "full", "same_user_first", "candidate_strict", "safe_residual", "candidate_item",
            "cross_user_only", "shuffled_memory", "shuffled_memory_clean",
            "cluster_full", "hybrid_cluster", "hybrid_cluster_strict", "shuffled_cluster",
            "full_lgcn", "full_lgcn_cluster", "full_lgcn_random",
            "oracle_cross_candidate",
        }:
            for item_id in candidate_set:
                for mid in self.memories_by_item.get(item_id, set()):
                    # Oracle arm: surface ONLY other users' lessons that actually
                    # concern an item this user has to rank. Measures the ceiling
                    # of cross-user transfer when relevance is guaranteed --
                    # naturally this happens for ~1% of retrievals, far too few to
                    # test observationally.
                    if retrieval_scope == "oracle_cross_candidate":
                        les = self.lessons.get(mid)
                        if les is None or str(les.source_user_id) == str(user_id):
                            continue
                    scores[mid] += 2.0
                    sources[mid].add("candidate_item")
                    paths[mid].append(f"candidate_item:{item_id}->memory:{mid}")

        if retrieval_scope in {
            "full", "same_user_first", "candidate_strict", "safe_residual", "history_item",
            "cross_user_only", "shuffled_memory", "shuffled_memory_clean", "hybrid_cluster", "hybrid_cluster_strict",
            "full_lgcn", "full_lgcn_cluster", "full_lgcn_random",
        }:
            for item_id in history_set:
                for mid in self.memories_by_item.get(item_id, set()):
                    scores[mid] += 1.5
                    sources[mid].add("history_item")
                    paths[mid].append(f"history_item:{item_id}->memory:{mid}")

        if retrieval_scope in {
            "full", "same_user_first", "candidate_strict", "safe_residual", "neighbor_user",
            "cross_user_only", "shuffled_memory", "shuffled_memory_clean", "hybrid_cluster", "hybrid_cluster_strict",
        }:
            for other_user, shared_count in self.similar_users(user_id, top_k=neighbor_k):
                for mid in self.memories_by_user.get(other_user, set()):
                    scores[mid] += 1.0 + 0.2 * min(shared_count, 5)
                    sources[mid].add("neighbor_user")
                    paths[mid].append(f"user:{user_id}->shared_items:{shared_count}->user:{other_user}->memory:{mid}")

        if retrieval_scope in {"cluster_user", "cluster_full", "hybrid_cluster", "hybrid_cluster_strict", "shuffled_cluster"}:
            cluster_id = self.cluster_by_user.get(user_id)
            for other_user, jaccard in self.cluster_users(user_id, top_k=neighbor_k):
                for mid in self.memories_by_user.get(other_user, set()):
                    scores[mid] += 1.0 + jaccard
                    sources[mid].add("cluster_user")
                    paths[mid].append(
                        f"user:{user_id}->cluster:{cluster_id}->user:{other_user}"
                        f"(jaccard={jaccard:.3f})->memory:{mid}"
                    )

        if retrieval_scope == "full_lgcn":
            # Same structural role/range as the neighbor_user bonus above ([1.0, 2.0]),
            # but driven by LightGCN propagated-embedding cosine similarity instead of a
            # raw co-interaction count -- isolates the similarity metric as the sole
            # manipulated variable, holding same_user/candidate_item/history_item fixed.
            for other_user, cos_sim in self.similar_users_lgcn(user_id, top_k=neighbor_k):
                for mid in self.memories_by_user.get(other_user, set()):
                    scores[mid] += 1.0 + max(0.0, cos_sim)
                    sources[mid].add("neighbor_user_lgcn")
                    paths[mid].append(
                        f"user:{user_id}->lgcn_cosine:{cos_sim:.3f}->user:{other_user}->memory:{mid}"
                    )

        if retrieval_scope == "full_lgcn_cluster":
            # Replaces neighbor_user's raw-count similarity with hard cluster membership
            # (KMeans over LightGCN propagated embeddings, see _build_lgcn_clusters) --
            # only users in the same LightGCN cluster as the query user are eligible,
            # ranked within-cluster by cosine similarity.
            lgcn_cluster_id = self.lgcn_cluster_by_user.get(user_id)
            for other_user, cos_sim in self.cluster_users_lgcn(user_id, top_k=neighbor_k):
                for mid in self.memories_by_user.get(other_user, set()):
                    scores[mid] += 1.0 + max(0.0, cos_sim)
                    sources[mid].add("cluster_user_lgcn")
                    paths[mid].append(
                        f"user:{user_id}->lgcn_cluster:{lgcn_cluster_id}->user:{other_user}"
                        f"(cos={cos_sim:.3f})->memory:{mid}"
                    )

        if retrieval_scope == "full_lgcn_random":
            # Ablation control for full_lgcn: identical same_user/candidate_item/
            # history_item terms and identical score formula for the cross-user
            # bonus, but the neighbor SELECTION is randomized (deterministically
            # seeded) instead of ranked by cosine similarity. If this performs
            # comparably to full_lgcn, similarity-based selection isn't earning
            # its keep; if full_lgcn beats this, the selection criterion itself
            # is adding value, not just "having more cross-user facts."
            for other_user, cos_sim in self.random_users_lgcn(user_id, top_k=neighbor_k, shuffle_salt=shuffle_salt):
                for mid in self.memories_by_user.get(other_user, set()):
                    scores[mid] += 1.0 + max(0.0, cos_sim)
                    sources[mid].add("neighbor_user_lgcn_random")
                    paths[mid].append(
                        f"user:{user_id}->lgcn_random_pick->user:{other_user}"
                        f"(cos={cos_sim:.3f})->memory:{mid}"
                    )

        retrieved: List[GraphRetrievedLesson] = []
        for mid, score in scores.items():
            lesson = self.lessons.get(mid)
            if lesson is None:
                continue
            evidence_terms = normalize_evidence_terms(
                list(lesson.evidence_terms or [])
                + list(lesson.applies_if or [])
                + [lesson.prefer, lesson.avoid]
            )
            matched_terms = [
                term for term in evidence_terms
                if term_matches_context(term, current_context_text)
            ]
            same_user = "same_user" in sources[mid]
            candidate_item = "candidate_item" in sources[mid]
            cluster_user = "cluster_user" in sources[mid]
            if not same_user and not candidate_item and not cluster_user and len(matched_terms) < min_evidence_terms:
                continue
            score += 0.25 * len(matched_terms)
            score += 0.2 * _safe_float(lesson.confidence, 0.5)
            score -= 0.5 * _safe_float(lesson.overgeneralization_risk, 0.5)
            retrieved.append(GraphRetrievedLesson(
                lesson=lesson,
                score=score,
                sources=sorted(sources[mid]),
                paths=paths[mid],
                matched_evidence_terms=matched_terms,
            ))
        if retrieval_scope in {"same_user_first", "safe_residual"}:
            retrieved.sort(key=lambda r: (
                0 if "same_user" in r.sources else 1,
                -r.score,
                r.lesson.memory_id,
            ))
        else:
            retrieved.sort(key=lambda r: (-r.score, r.lesson.memory_id))
        if retrieval_scope in {"shuffled_memory", "shuffled_memory_clean", "shuffled_cluster"} and retrieved:
            candidate_mids = deterministic_shuffle(list(self.lessons.keys()), salt=f"shuffled_memory::{user_id}::{shuffle_salt}")
            shuffled: List[GraphRetrievedLesson] = []
            for row, replacement_mid in zip(retrieved[:top_k], candidate_mids):
                if retrieval_scope == "shuffled_memory_clean":
                    # Clean shuffled control: keep only the count of retrieved rows,
                    # but remove all real graph provenance from the replacement
                    # memory. The old shuffled control inherited same_user /
                    # candidate_item / matched terms, which made it biased high.
                    shuffled.append(GraphRetrievedLesson(
                        lesson=self.lessons[replacement_mid],
                        score=0.0,
                        sources=["shuffled_memory_clean"],
                        paths=[f"shuffled_memory_clean_control:{replacement_mid}"],
                        matched_evidence_terms=[],
                    ))
                else:
                    shuffled.append(GraphRetrievedLesson(
                        lesson=self.lessons[replacement_mid],
                        score=row.score,
                        sources=sorted(set(row.sources + [retrieval_scope])),
                        paths=row.paths + [f"shuffled_control_replacement:{replacement_mid}"],
                        matched_evidence_terms=row.matched_evidence_terms,
                    ))
            return shuffled
        return retrieved[:top_k]

    def stats(self) -> Dict[str, Any]:
        cluster_sizes = [len(users) for users in self.users_by_cluster.values()]
        memory_item_edges = sum(len(v) for v in self.memories_by_item.values())
        memory_user_edges = sum(len(v) for v in self.memories_by_user.values())
        return {
            "num_lessons": len(self.lessons),
            "num_users": len(self.items_by_user),
            "num_items": len(self.users_by_item),
            "num_clusters": len(self.users_by_cluster),
            "num_memory_source_users": len(self.memories_by_user),
            "avg_users_per_cluster": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
            "max_users_per_cluster": max(cluster_sizes) if cluster_sizes else 0,
            "min_users_per_cluster": min(cluster_sizes) if cluster_sizes else 0,
            "num_memory_user_edges": memory_user_edges,
            "num_memory_item_edges": memory_item_edges,
            "num_users_with_memories": len(self.memories_by_user),
            "num_items_with_memories": len(self.memories_by_item),
            "num_cluster_corrective_lessons": len(self.cluster_corrective_lessons),
            "num_cluster_corrective_global_lessons": len(self.cluster_corrective_global),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lessons": [asdict(lesson) for lesson in self.lessons.values()],
            "cluster_by_user": self.cluster_by_user,
            "cluster_stats": self.stats(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        user_sequences: Dict[str, Dict[str, Any]],
        build_clusters: bool = True,
    ) -> "MemoryGraphIndex":
        graph = cls(user_sequences, build_clusters=build_clusters)
        for row in data.get("lessons", []):
            graph.add_lesson(FailureLesson(**row))
        return graph


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


def build_compact_safe_residual_score_prompt(
    history_items: List[Dict[str, Any]],
    aliased_candidates: List[Dict[str, Any]],
    memory_facts: List[str],
    user_profile_payload: Optional[Dict[str, Any]] = None,
    prompt_sample: str = "",
) -> str:
    """Strong A-style scorer with bounded, explicitly routed graph evidence."""
    same_user = [fact for fact in memory_facts if fact.startswith("[SAME-USER")]
    support = [fact for fact in memory_facts if fact.startswith("[COLLABORATIVE SUPPORT")]
    avoid = [fact for fact in memory_facts if fact.startswith("[COLLABORATIVE AVOID")]
    return f"""
You are a deterministic recommendation scorer. Rank candidates from the target user's profile, recent positive history, candidate facts, and optional failure-derived corrections.

{str(prompt_sample).strip()}

Target User Profile:
{json.dumps(user_profile_payload or {}, ensure_ascii=False, separators=(",", ":"))}

Recent Positive History:
{_format_compact_history_lines(history_items)}

Same-User Corrections (primary personalized evidence):
{json.dumps(same_user, ensure_ascii=False, separators=(",", ":")) if same_user else "[]"}

Collaborative Support (weak residual evidence):
{json.dumps(support, ensure_ascii=False, separators=(",", ":")) if support else "[]"}

Collaborative Avoid (weak residual evidence):
{json.dumps(avoid, ensure_ascii=False, separators=(",", ":")) if avoid else "[]"}

Candidate Items:
{_format_compact_candidate_lines(aliased_candidates)}

Scoring policy:
- Establish the base ranking from the target user's profile, recent history, and candidate facts.
- Same-user corrections may refine that base when their concrete titles or attributes match.
- Collaborative support/avoid is only a small residual tie-breaker; never let it override a clear target-user mismatch.
- Apply support or avoid only to the explicitly named current candidate. Do not transfer it to unrelated candidates.
- If collaborative evidence conflicts with target-user evidence, follow the target user.
- Do not infer new preferences or causal explanations.

Output requirements:
- Return ONLY valid compact JSON. No markdown.
- Output one score row for every candidate_id exactly once.
- Score is a number from 0.0 to 1.0.
- Rationale must be <= 8 words.

JSON format:
{{
  "scores": [
    {{"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}}
  ]
}}
"""


def _token_set_for_prompt(text: Any) -> Set[str]:
    """Small lexical set for deterministic memory-candidate evidence mapping."""
    stop = {
        "the", "and", "for", "with", "from", "into", "that", "this", "user",
        "item", "items", "game", "games", "video", "unknown", "preferred",
        "bought", "instead", "preference", "history", "candidate",
    }
    return {
        tok for tok in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(tok) >= 4 and tok not in stop
    }


def _extract_prefer_avoid_titles_from_fact(fact: str) -> Tuple[List[str], List[str]]:
    """Parse MEMCF safe facts: preferred/bought 'correct' instead of 'wrong'."""
    text = str(fact or "")
    # Prefer the explicit contrastive pattern. The preceding user-history quote
    # can be very long and may be truncated, which can confuse generic quote
    # extraction and accidentally capture "preferred/bought" as a title.
    explicit = re.search(
        r"(?:preferred/bought|preferred|bought|selected|chose)\s+['\"]([^'\"]{2,220})['\"]\s+instead of\s+['\"]([^'\"]{2,220})['\"]",
        text,
        flags=re.IGNORECASE,
    )
    if explicit:
        return [explicit.group(1).strip()], [explicit.group(2).strip()]

    quoted = re.findall(r"'([^']{2,160})'", text)
    prefer: List[str] = []
    avoid: List[str] = []
    if " instead of " in text:
        before, after = text.split(" instead of ", 1)
        prefer = re.findall(r"'([^']{2,160})'", before)[-1:]
        avoid = re.findall(r"'([^']{2,160})'", after)[:1]
    elif len(quoted) >= 2:
        prefer = quoted[-2:-1]
        avoid = quoted[-1:]
    elif quoted:
        prefer = quoted[:1]
    return prefer, avoid


def build_memory_candidate_evidence(
    memory_facts: List[str],
    aliased_candidates: List[Dict[str, Any]],
    max_facts: int = 3,
) -> Dict[str, Any]:
    """Curate raw corrective memory into candidate-level evidence.

    This is a deterministic MEMCF counterpart to MemRec's Stage-R/Packer:
    it does not use target-user history/profile, but it makes failure memories
    actionable by mapping prefer/avoid patterns to current candidate IDs.
    """
    facts = [shorten_words(x, 45) for x in memory_facts[:max_facts] if str(x).strip()]
    prefer_patterns: List[str] = []
    avoid_patterns: List[str] = []
    fact_rows: List[Dict[str, Any]] = []
    for idx, fact in enumerate(facts, 1):
        prefer_titles, avoid_titles = _extract_prefer_avoid_titles_from_fact(fact)
        prefer_patterns.extend(prefer_titles)
        avoid_patterns.extend(avoid_titles)
        fact_rows.append({
            "memory_id": f"M{idx:02d}",
            "fact": fact,
            "prefer_titles": prefer_titles,
            "avoid_titles": avoid_titles,
        })

    evidence_rows: List[Dict[str, Any]] = []
    for cand in aliased_candidates:
        cid = str(cand.get("candidate_id", ""))
        title = str(cand.get("title", ""))
        category = str(cand.get("category", ""))
        desc = str(cand.get("description", ""))
        cand_text = " ".join([title, category, shorten_words(desc, 35)])
        cand_tokens = _token_set_for_prompt(cand_text)
        supports: List[str] = []
        avoids: List[str] = []
        support_terms: List[str] = []

        for row in fact_rows:
            fact_tokens = _token_set_for_prompt(row["fact"])
            overlap = sorted(cand_tokens & fact_tokens)
            title_low = title.lower()
            prefer_hit = any(p and (p.lower() in title_low or title_low in p.lower()) for p in row["prefer_titles"])
            avoid_hit = any(a and (a.lower() in title_low or title_low in a.lower()) for a in row["avoid_titles"])
            # Require either direct title evidence or at least two concrete
            # lexical overlaps to avoid random-text effects.
            if prefer_hit or len(overlap) >= 2:
                supports.append(row["memory_id"])
                support_terms.extend(overlap[:4])
            if avoid_hit:
                avoids.append(row["memory_id"])

        label = "neutral"
        if supports and not avoids:
            label = "support"
        elif avoids and not supports:
            label = "avoid"
        elif supports and avoids:
            label = "mixed"
        evidence_rows.append({
            "candidate_id": cid,
            "title": title[:90],
            "memory_signal": label,
            "supporting_memories": sorted(set(supports))[:3],
            "avoid_memories": sorted(set(avoids))[:3],
            "matched_terms": sorted(set(support_terms))[:8],
        })

    anchor_bits: List[str] = []
    if prefer_patterns:
        anchor_bits.append("Prefer candidates similar to: " + "; ".join(shorten_words(x, 8) for x in prefer_patterns[:4]))
    if avoid_patterns:
        anchor_bits.append("Avoid candidates similar to past wrong choices: " + "; ".join(shorten_words(x, 8) for x in avoid_patterns[:4]))
    if not anchor_bits:
        anchor_bits.append("No direct prefer/avoid title pattern was parsed; use only candidate evidence rows.")

    return {
        "memory_anchor": anchor_bits,
        "memory_facts": fact_rows,
        "candidate_evidence": evidence_rows,
    }


def _compact_candidate_rows_for_router(aliased_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Candidate facts for weak memory-router prompts.

    The router variant intentionally keeps candidate facts short so corrective
    memory evidence is not drowned out by long product descriptions.
    """
    rows: List[Dict[str, Any]] = []
    for cand in aliased_candidates:
        rows.append({
            "candidate_id": str(cand.get("candidate_id", "")),
            "title": shorten_words(cand.get("title", ""), 14),
            "category": shorten_words(cand.get("category", ""), 8),
            "description": shorten_words(cand.get("description", ""), 22),
        })
    return rows


def build_failure_evidence_router(
    memory_facts: List[str],
    aliased_candidates: List[Dict[str, Any]],
    max_facts: int = 3,
    max_evidence_rows: int = 8,
) -> Dict[str, Any]:
    """Route failure memories to current candidate IDs without another LLM call.

    This is deliberately not MemRec's generic facet synthesis. It keeps MEMCF's
    failure provenance: each memory says a prior correct item was preferred over
    a prior wrong item, then routes prefer/avoid title terms to candidates.
    """
    fact_rows: List[Dict[str, Any]] = []
    prefer_terms_all: List[str] = []
    avoid_terms_all: List[str] = []
    for idx, fact in enumerate([x for x in memory_facts[:max_facts] if str(x).strip()], 1):
        prefer_titles, avoid_titles = _extract_prefer_avoid_titles_from_fact(str(fact))
        prefer_terms = sorted(set().union(*(_token_set_for_prompt(x) for x in prefer_titles))) if prefer_titles else []
        avoid_terms = sorted(set().union(*(_token_set_for_prompt(x) for x in avoid_titles))) if avoid_titles else []
        prefer_terms_all.extend(prefer_terms)
        avoid_terms_all.extend(avoid_terms)
        fact_rows.append({
            "memory_id": f"M{idx:02d}",
            "prefer": [shorten_words(x, 10) for x in prefer_titles[:2]],
            "avoid": [shorten_words(x, 10) for x in avoid_titles[:2]],
            "prefer_terms": prefer_terms[:8],
            "avoid_terms": avoid_terms[:8],
        })

    candidate_rows = _compact_candidate_rows_for_router(aliased_candidates)
    evidence_rows: List[Dict[str, Any]] = []
    for cand in candidate_rows:
        cid = str(cand.get("candidate_id", ""))
        title = str(cand.get("title", ""))
        cand_tokens = _token_set_for_prompt(" ".join([cand.get("title", ""), cand.get("category", ""), cand.get("description", "")]))
        support_ids: List[str] = []
        avoid_ids: List[str] = []
        matched_prefer_terms: List[str] = []
        matched_avoid_terms: List[str] = []
        for row in fact_rows:
            prefer_overlap = sorted(cand_tokens & set(row.get("prefer_terms", [])))
            avoid_overlap = sorted(cand_tokens & set(row.get("avoid_terms", [])))
            title_low = title.lower()
            direct_prefer = any(x and (x.lower() in title_low or title_low in x.lower()) for x in row.get("prefer", []))
            direct_avoid = any(x and (x.lower() in title_low or title_low in x.lower()) for x in row.get("avoid", []))
            if direct_prefer or len(prefer_overlap) >= 1:
                support_ids.append(row["memory_id"])
                matched_prefer_terms.extend(prefer_overlap[:4])
            if direct_avoid or len(avoid_overlap) >= 1:
                avoid_ids.append(row["memory_id"])
                matched_avoid_terms.extend(avoid_overlap[:4])

        if support_ids or avoid_ids:
            if support_ids and avoid_ids:
                signal = "mixed"
            elif support_ids:
                signal = "support"
            else:
                signal = "avoid"
            evidence_rows.append({
                "candidate_id": cid,
                "signal": signal,
                "supporting_memories": sorted(set(support_ids))[:3],
                "avoid_memories": sorted(set(avoid_ids))[:3],
                "matched_prefer_terms": sorted(set(matched_prefer_terms))[:6],
                "matched_avoid_terms": sorted(set(matched_avoid_terms))[:6],
            })

    # Keep the evidence table compact and ranked by usefulness: support/mixed
    # rows first, then avoid rows, then stronger lexical matches.
    signal_rank = {"support": 0, "mixed": 1, "avoid": 2}
    evidence_rows.sort(key=lambda r: (signal_rank.get(r["signal"], 9), -len(r.get("matched_prefer_terms", [])), r["candidate_id"]))
    evidence_rows = evidence_rows[:max_evidence_rows]

    anchor = {
        "prefer_terms": sorted(set(prefer_terms_all))[:12],
        "avoid_terms": sorted(set(avoid_terms_all))[:12],
        "policy": "Use support/avoid candidate evidence when present; otherwise fall back to compact candidate facts.",
    }
    return {
        "candidate_rows": candidate_rows,
        "memory_routes": fact_rows,
        "candidate_evidence": evidence_rows,
        "anchor": anchor,
    }


def build_stage_r_corrective_rules(
    memory_facts: List[str],
    max_rules: int = 3,
    max_words: int = 26,
) -> List[Dict[str, Any]]:
    """Summarize failure facts into short ranker-safe corrective rules.

    This is not raw chain-of-thought replay. It extracts only the contrastive
    prefer-vs-avoid relation that can be safely used as weak ranking evidence.
    """
    rules: List[Dict[str, Any]] = []
    for idx, fact in enumerate([x for x in memory_facts if str(x).strip()], 1):
        prefer_titles, avoid_titles = _extract_prefer_avoid_titles_from_fact(str(fact))
        prefer = shorten_words(prefer_titles[0], 10) if prefer_titles else ""
        avoid = shorten_words(avoid_titles[0], 10) if avoid_titles else ""
        if not prefer and not avoid:
            continue
        prefer_terms = sorted(set().union(*(_token_set_for_prompt(x) for x in prefer_titles))) if prefer_titles else []
        avoid_terms = sorted(set().union(*(_token_set_for_prompt(x) for x in avoid_titles))) if avoid_titles else []
        if prefer and avoid:
            rule = f"Prefer candidates like {prefer} over candidates like {avoid} only when current history and candidates support the contrast."
        elif prefer:
            rule = f"Prefer candidates like {prefer} only when current history and candidates support the match."
        else:
            rule = f"Avoid candidates like {avoid} unless current candidate facts strongly support them."
        rules.append({
            "rule_id": f"R{idx:02d}",
            "corrective_rule": shorten_words(rule, max_words),
            "prefer_terms": prefer_terms[:8],
            "avoid_terms": avoid_terms[:8],
        })
        if len(rules) >= max_rules:
            break
    return rules


def build_compact_stage_r_prompt(
    history_items: List[Dict[str, Any]],
    aliased_candidates: List[Dict[str, Any]],
    user_profile_payload: Optional[Dict[str, Any]],
    memory_facts: List[str],
    include_reasoning_rules: bool = False,
    prompt_sample: str = "",
) -> str:
    """A-style prompt with MemRec-like memory packing.

    The ranker still gets MEMCF-A's user history/profile and candidate facts.
    Memory is transformed into compact Stage-R-style evidence tables so Qwen 7B
    is not asked to interpret long raw memory text.
    """
    router = build_failure_evidence_router(memory_facts, aliased_candidates, max_facts=5)
    rules = build_stage_r_corrective_rules(memory_facts, max_rules=3) if include_reasoning_rules else []

    parts = [
        "You are scoring candidate items for a recommender system based on user history, candidate facts, and optional packed memory evidence.\n"
    ]
    if prompt_sample:
        parts.append(str(prompt_sample).strip() + "\n")
    parts.append("Inputs:\n")
    if user_profile_payload:
        parts.append("User Memory Profile:\n")
        parts.append(json.dumps(user_profile_payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    parts.append("User recent history:\n")
    parts.append(_format_compact_history_lines(history_items) + "\n")
    parts.append("Candidate Items:\n")
    parts.append(_format_compact_candidate_lines(aliased_candidates) + "\n")
    parts.append("Packed Memory Evidence:\n")
    parts.append("Failure Memory Routes:\n")
    parts.append(json.dumps(router["memory_routes"], ensure_ascii=False, separators=(",", ":")) + "\n")
    parts.append("Candidate Evidence Router:\n")
    parts.append(json.dumps(router["candidate_evidence"], ensure_ascii=False, separators=(",", ":")) + "\n")
    parts.append("Router Anchor:\n")
    parts.append(json.dumps(router["anchor"], ensure_ascii=False, separators=(",", ":")) + "\n")
    if include_reasoning_rules:
        parts.append("Short Corrective Rules:\n")
        parts.append(json.dumps(rules, ensure_ascii=False, separators=(",", ":")) + "\n")
    parts.append(
        "\nMemory policy:\n"
        "- Packed memory evidence is weak corrective evidence, not a hard rule.\n"
        "- Use memory only when it maps to current candidate_id evidence and is consistent with user history.\n"
        "- Ignore generic platform/category-only matches if they conflict with stronger history or candidate facts.\n"
        "- If packed evidence is empty or irrelevant, score from history and candidate facts only.\n"
        "\nOutput requirements:\n"
        "- Return ONLY valid compact JSON. No markdown.\n"
        "- Output one score row for every candidate_id exactly once.\n"
        "- Score is a number from 0.0 to 1.0.\n"
        "- Rationale must be <= 8 words.\n"
        "\nJSON format:\n"
        "{\n"
        '  "scores": [\n'
        '    {"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}\n'
        "  ]\n"
        "}\n"
    )
    return "".join(parts)


def build_compact_curated_score_prompt(
    history_items: List[Dict[str, Any]],
    aliased_candidates: List[Dict[str, Any]],
    user_profile_payload: Optional[Dict[str, Any]],
    memory_facts: List[str],
    prompt_sample: str = "",
) -> str:
    """A-style ranker prompt with curated candidate-level memory evidence only.

    Unlike ``compact_score``, this does not expose raw memory text. Unlike the
    Stage-R prompt, it also hides memory-route internals. The ranker only sees:
    current user evidence, current candidate facts, candidate-level support/avoid
    evidence, and short corrective rules. This keeps MEMCF-A strong while
    reducing the distraction from wrong-item titles in failure memories.
    """
    router = build_failure_evidence_router(memory_facts, aliased_candidates, max_facts=5, max_evidence_rows=10)
    corrective_rules = build_stage_r_corrective_rules(memory_facts, max_rules=3, max_words=24)
    candidate_evidence = router.get("candidate_evidence", [])
    anchor = router.get("anchor", {})

    parts = [
        "You are scoring candidate items for a recommender system.\n"
        "Use user history/profile as primary evidence and curated failure memory as weak corrective evidence.\n"
    ]
    if prompt_sample:
        parts.append(str(prompt_sample).strip() + "\n")
    parts.append("Inputs:\n")
    if user_profile_payload:
        parts.append("User Memory Profile:\n")
        parts.append(json.dumps(user_profile_payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    parts.append("User recent history:\n")
    parts.append(_format_compact_history_lines(history_items) + "\n")
    parts.append("Candidate Items:\n")
    parts.append(_format_compact_candidate_lines(aliased_candidates) + "\n")
    parts.append("Curated Candidate Memory Evidence:\n")
    parts.append(json.dumps(candidate_evidence, ensure_ascii=False, separators=(",", ":")) + "\n")
    parts.append("Short Corrective Rules:\n")
    parts.append(json.dumps(corrective_rules, ensure_ascii=False, separators=(",", ":")) + "\n")
    parts.append("Memory Anchor Terms:\n")
    parts.append(json.dumps(anchor, ensure_ascii=False, separators=(",", ":")) + "\n")
    parts.append(
        "\nScoring policy:\n"
        "- User history/profile and current candidate facts are primary evidence.\n"
        "- Curated memory evidence is weak correction, not a hard rule.\n"
        "- A support signal should help only if it maps to a current candidate_id and matches the user evidence.\n"
        "- An avoid signal should hurt only if it maps to a current candidate_id and the candidate is otherwise comparable.\n"
        "- Ignore memory terms that are generic, conflict with candidate facts, or do not map to current candidates.\n"
        "- If memory evidence is empty or ambiguous, score from user history/profile and candidate facts only.\n"
        "\nOutput requirements:\n"
        "- Return ONLY valid compact JSON. No markdown.\n"
        "- Output one score row for every candidate_id exactly once.\n"
        "- Score is a number from 0.0 to 1.0.\n"
        "- Rationale must be <= 8 words.\n"
        "\nJSON format:\n"
        "{\n"
        '  "scores": [\n'
        '    {"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}\n'
        "  ]\n"
        "}\n"
    )
    return "".join(parts)


def build_deterministic_user_anchor(train_items: List[Dict[str, Any]], max_items: int = 10) -> Dict[str, Any]:
    """Build a compact non-LLM user anchor from recent positive history.

    This is intentionally weaker than the full MEMCF prompt: it exposes only
    category/platform/keyword summaries, not raw history rows or a generated
    profile. It is meant for B-style controls where the vanilla prompt is too
    weak for memories to be interpreted consistently.
    """
    stop_terms = {
        "unknown", "item", "items", "product", "products", "amazon", "edition",
        "new", "used", "pack", "set", "collection", "series", "vol", "volume",
        "digital", "music", "video", "game", "games", "cd", "vinyl", "beauty",
        "the", "and", "for", "with", "from", "into", "this", "that",
    }
    platform_vocab = {
        "nintendo", "wii", "gamecube", "playstation", "ps2", "ps3", "ps4",
        "xbox", "vita", "ds", "3ds", "pc", "windows", "mac", "switch",
        "sega", "nes", "snes", "n64",
    }
    category_counts: Counter[str] = Counter()
    term_counts: Counter[str] = Counter()
    platform_terms: Counter[str] = Counter()

    for item in (train_items or [])[-max_items:]:
        category = str(item.get("category", "")).strip()
        if category and category.lower() != "unknown":
            category_counts[shorten_words(category, 8)] += 1
        text = " ".join([
            str(item.get("title", "")),
            str(item.get("category", "")),
            shorten_words(item.get("description", ""), 30),
        ])
        for tok in _token_set_for_prompt(text):
            if tok in platform_vocab:
                platform_terms[tok] += 1
            elif tok not in stop_terms and len(tok) >= 3:
                term_counts[tok] += 1

    return {
        "source": "deterministic_recent_history_anchor",
        "recent_history_count": min(len(train_items or []), max_items),
        "top_categories": [
            {"category": cat, "count": count}
            for cat, count in category_counts.most_common(3)
        ],
        "platform_terms": [term for term, _ in platform_terms.most_common(8)],
        "keyword_terms": [term for term, _ in term_counts.most_common(12)],
        "policy": "Use this anchor as weak user evidence; do not infer preferences beyond these terms.",
    }


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


def llm_ranking_v2(
    memory_system: RecommendationMemorySystem,
    train_items: List[Dict[str, Any]],
    candidate_items: List[Dict[str, Any]],
    user_profile: Optional[UserMemoryProfile],
    memory_facets: Optional[List[str]],
    prompt_sample: str = "",
    ranking_prompt_style: str = "memcf",
    trace_context: Optional[Dict[str, Any]] = None,
    pairwise_cf_rerank: bool = False,
    pairwise_memory_rows: Optional[List[Dict[str, Any]]] = None,
    pairwise_cf_alpha: float = 0.04,
    pairwise_cf_beta: float = 0.04,
    failure_constraint_mode: str = "none",
    failure_constraint_evidence: Optional[List[Dict[str, Any]]] = None,
    failure_constraint_tie_epsilon: float = 0.0,
    failure_constraint_min_cross_support: int = 2,
    failure_constraint_max_cross_corrections: int = 3,
    failure_constraint_candidate_popularity: Optional[Dict[str, int]] = None,
    ranking_score_cache_dir: Optional[str] = None,
) -> List[str]:
    candidate_info = [
        {
            "item_id": str(item["item_id"]),
            "title": item["title"],
            "category": item["category"],
            "description": item.get("description", ""),
        }
        for item in candidate_items
    ]
    aliased_candidates, alias_to_item_id = add_candidate_aliases(candidate_info)
    valid_candidate_aliases = list(alias_to_item_id.keys())
    profile_block = user_profile.to_prompt_dict() if user_profile else {}
    facets = [str(x) for x in (memory_facets or []) if str(x).strip()]

    if ranking_prompt_style == "memrec_vanilla":
        # MemRec-style vanilla LLM baseline: candidate metadata only.
        # It intentionally omits user history, profile, and memory facts so the
        # baseline/control strength matches the vanilla setting used by MemRec.
        prompt = f"""
You are an intelligent recommendation scoring system. Your task is to evaluate how well each candidate item matches the target user's preferences.

Target User:
No specific user profile provided.

Candidate Items (use candidate_id only in output):
{json.dumps(aliased_candidates, ensure_ascii=False, indent=2)}

Your Task:
For each candidate item, provide a relevance score between 0 and 1:
- 1.0 = Excellent match, highly aligned with the user's preferences
- 0.5 = Moderate match, partially relevant
- 0.0 = Poor match, not aligned with the user's interests

Output requirements:
- Return ONLY valid JSON. No markdown.
- Output one score row for every candidate_id exactly once.
- Score is a number from 0.0 to 1.0.
- Rationale must be <= 8 words.

JSON format:
{{
  "scores": [
    {{"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}}
  ]
}}
"""
    elif ranking_prompt_style == "weak_anchor_score":
        # Weak baseline with a deterministic compact user anchor. This keeps the
        # B-style prompt controlled, but avoids a completely user-agnostic ranker.
        user_anchor = build_deterministic_user_anchor(train_items[-10:])
        candidate_rows = _compact_candidate_rows_for_router(aliased_candidates)
        prompt = f"""
You are a deterministic recommendation scorer. Score candidates using compact candidate facts and a short deterministic user anchor.

Target User Anchor:
{json.dumps(user_anchor, ensure_ascii=False, separators=(",", ":"))}

Compact Candidate Facts:
{json.dumps(candidate_rows, ensure_ascii=False, separators=(",", ":"))}

Scoring policy:
- Use the user anchor as weak evidence from recent positive history.
- Prefer candidates matching anchor categories, platform terms, or keyword terms.
- Do not invent preferences beyond the anchor and candidate facts.
- If the anchor is sparse, rely on candidate facts only.

Output requirements:
- Return ONLY valid JSON. No markdown.
- Output one score row for every candidate_id exactly once.
- Score is a number from 0.0 to 1.0.
- Rationale must be <= 8 words.

JSON format:
{{
  "scores": [
    {{"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}}
  ]
}}
"""
    elif ranking_prompt_style == "weak_memory_score":
        # Weak memory-control prompt: candidate metadata plus selected memory
        # snippets only. This removes MEMCF's user-history/profile advantage and
        # makes random/shuffled/profile controls closer to MemRec's vanilla LLM.
        prompt = f"""
You are an intelligent recommendation scoring system. Your task is to evaluate how well each candidate item matches the target user's preferences.

Target User:
No specific user profile provided.

Candidate Items (use candidate_id only in output):
{json.dumps(aliased_candidates, ensure_ascii=False, indent=2)}

Optional Retrieved Memory Snippets:
{json.dumps(facets, ensure_ascii=False, indent=2) if facets else "No retrieved memory snippets."}

Memory policy:
- Memory snippets are weak evidence.
- Use a snippet only when it directly matches candidate facts.
- If snippets are irrelevant or conflict with item facts, ignore them.

Your Task:
For each candidate item, provide a relevance score between 0 and 1:
- 1.0 = Excellent match, highly aligned with the available evidence
- 0.5 = Moderate match, partially relevant
- 0.0 = Poor match, not aligned with the available evidence

Output requirements:
- Return ONLY valid JSON. No markdown.
- Output one score row for every candidate_id exactly once.
- Score is a number from 0.0 to 1.0.
- Rationale must be <= 8 words.

JSON format:
{{
  "scores": [
    {{"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}}
  ]
}}
"""
    elif ranking_prompt_style == "weak_memory_evidence_score":
        # MemRec-inspired weak prompt: no raw user history/profile, but selected
        # failure memories are curated into compact candidate-level evidence.
        # This keeps the controlled B-style setting while making memory usable
        # for smaller local instruction models.
        evidence = build_memory_candidate_evidence(facets, aliased_candidates, max_facts=5)
        prompt = f"""
You are an intelligent recommendation scoring system. Your task is to score candidate items using only candidate facts and memory-derived corrective evidence.

Target User:
No raw user history or user profile is provided. Infer only from the memory-derived evidence below.

Candidate Items (use candidate_id only in output):
{json.dumps(aliased_candidates, ensure_ascii=False, indent=2)}

Memory-Derived Target Anchor:
{json.dumps(evidence["memory_anchor"], ensure_ascii=False, indent=2)}

Corrective Memory Facts:
{json.dumps(evidence["memory_facts"], ensure_ascii=False, indent=2)}

Candidate-Memory Evidence Table:
{json.dumps(evidence["candidate_evidence"], ensure_ascii=False, indent=2)}

Memory policy:
- Candidate-memory evidence is weak but actionable when it maps to candidate_id.
- Prefer candidates marked "support" over candidates marked "neutral" when item facts are plausible.
- Penalize candidates marked "avoid" unless candidate facts strongly contradict the memory.
- If every candidate is neutral, score using candidate facts only.
- Do not use raw user history; it is intentionally not provided.

Your Task:
For each candidate item, provide a relevance score between 0 and 1:
- 1.0 = Excellent match to candidate facts and memory-derived evidence
- 0.5 = Moderate match or uncertain evidence
- 0.0 = Poor match or avoid-pattern match

Output requirements:
- Return ONLY valid JSON. No markdown.
- Output one score row for every candidate_id exactly once.
- Score is a number from 0.0 to 1.0.
- Rationale must be <= 8 words.

JSON format:
{{
  "scores": [
    {{"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}}
  ]
}}
"""
    elif ranking_prompt_style == "weak_memory_router_score":
        # Failure Evidence Router: a more compact, MEMCF-specific version of the
        # correction prompt. It routes wrong-vs-correct memory provenance to
        # candidate IDs without exposing raw user history/profile.
        router = build_failure_evidence_router(facets, aliased_candidates, max_facts=5)
        prompt = f"""
You are a deterministic recommendation scorer. Score candidates using compact candidate facts and failure-derived candidate evidence.

Target User:
No raw user history or user profile is provided. Use only the routed memory evidence and candidate facts.

Compact Candidate Facts:
{json.dumps(router["candidate_rows"], ensure_ascii=False, separators=(",", ":"))}

Failure Memory Routes:
{json.dumps(router["memory_routes"], ensure_ascii=False, separators=(",", ":"))}

Candidate Evidence Router:
{json.dumps(router["candidate_evidence"], ensure_ascii=False, separators=(",", ":"))}

Router Anchor:
{json.dumps(router["anchor"], ensure_ascii=False, separators=(",", ":"))}

Scoring policy:
- Candidates with signal="support" should usually score above neutral candidates when candidate facts are plausible.
- Candidates with signal="avoid" should usually score below neutral candidates.
- Candidates with signal="mixed" need conservative middle scores unless support evidence is stronger than avoid evidence.
- If a candidate has no router row, score it from compact candidate facts only.
- Do not invent user preferences beyond the router evidence.

Output requirements:
- Return ONLY valid JSON. No markdown.
- Output one score row for every candidate_id exactly once.
- Score is a number from 0.0 to 1.0.
- Rationale must be <= 8 words.

JSON format:
{{
  "scores": [
    {{"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}}
  ]
}}
"""
    elif ranking_prompt_style == "weak_anchor_router_score":
        # Anchor + Failure Evidence Router: a stronger B-style prompt that still
        # avoids raw history/full profile. It tests whether concise user evidence
        # lets failure memories help without reverting to the main A prompt.
        user_anchor = build_deterministic_user_anchor(train_items[-10:])
        router = build_failure_evidence_router(facets, aliased_candidates, max_facts=5)
        prompt = f"""
You are a deterministic recommendation scorer. Score candidates using a short user anchor, compact candidate facts, and failure-derived candidate evidence.

Target User Anchor:
{json.dumps(user_anchor, ensure_ascii=False, separators=(",", ":"))}

Compact Candidate Facts:
{json.dumps(router["candidate_rows"], ensure_ascii=False, separators=(",", ":"))}

Failure Memory Routes:
{json.dumps(router["memory_routes"], ensure_ascii=False, separators=(",", ":"))}

Candidate Evidence Router:
{json.dumps(router["candidate_evidence"], ensure_ascii=False, separators=(",", ":"))}

Router Anchor:
{json.dumps(router["anchor"], ensure_ascii=False, separators=(",", ":"))}

Scoring policy:
- First use the Target User Anchor to identify plausible candidates.
- Then use Candidate Evidence Router as corrective evidence.
- Candidates with signal="support" should usually score above similar neutral candidates.
- Candidates with signal="avoid" should usually score below similar neutral candidates.
- Ignore memory routes that do not map to current candidate_id evidence.
- Do not invent user preferences beyond the anchor, routed evidence, and candidate facts.

Output requirements:
- Return ONLY valid JSON. No markdown.
- Output one score row for every candidate_id exactly once.
- Score is a number from 0.0 to 1.0.
- Rationale must be <= 8 words.

JSON format:
{{
  "scores": [
    {{"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}}
  ]
}}
"""
    elif ranking_prompt_style == "compact_score":
        prompt = build_compact_score_prompt(
            history_items=train_items[-10:],
            aliased_candidates=aliased_candidates,
            prompt_sample=prompt_sample,
            memory_payload=facets if facets else None,
            user_profile_payload=profile_block if profile_block else None,
        )
    elif ranking_prompt_style == "compact_safe_residual_score":
        has_collaborative_residual = any(
            fact.startswith("[COLLABORATIVE ") for fact in facets
        )
        if has_collaborative_residual:
            prompt = build_compact_safe_residual_score_prompt(
                history_items=train_items[-10:],
                aliased_candidates=aliased_candidates,
                memory_facts=facets,
                user_profile_payload=profile_block if profile_block else None,
                prompt_sample=prompt_sample,
            )
        else:
            # Preserve the proven A1/A4 scorer when no collaborative residual
            # survived. This makes safe-residual an additive intervention rather
            # than a global prompt rewrite.
            prompt = build_compact_score_prompt(
                history_items=train_items[-10:],
                aliased_candidates=aliased_candidates,
                prompt_sample=prompt_sample,
                memory_payload=facets if facets else None,
                user_profile_payload=profile_block if profile_block else None,
            )
    elif ranking_prompt_style == "compact_stage_r":
        prompt = build_compact_stage_r_prompt(
            history_items=train_items[-10:],
            aliased_candidates=aliased_candidates,
            user_profile_payload=profile_block if profile_block else None,
            memory_facts=facets,
            include_reasoning_rules=False,
            prompt_sample=prompt_sample,
        )
    elif ranking_prompt_style == "compact_stage_r_reasoning":
        prompt = build_compact_stage_r_prompt(
            history_items=train_items[-10:],
            aliased_candidates=aliased_candidates,
            user_profile_payload=profile_block if profile_block else None,
            memory_facts=facets,
            include_reasoning_rules=True,
            prompt_sample=prompt_sample,
        )
    elif ranking_prompt_style == "compact_curated_score":
        prompt = build_compact_curated_score_prompt(
            history_items=train_items[-10:],
            aliased_candidates=aliased_candidates,
            user_profile_payload=profile_block if profile_block else None,
            memory_facts=facets,
            prompt_sample=prompt_sample,
        )
    elif facets:
        prompt = f"""
You are scoring candidate items for a recommender system.

Inputs:
User Memory Profile initialized from observed history:
{json.dumps(profile_block, ensure_ascii=False, indent=2)}

User Recent History:
{json.dumps(train_items[-10:], ensure_ascii=False, indent=2)}

Safe Graph Memory Facts (factual snippets from prior failures):
{json.dumps(facets, ensure_ascii=False, indent=2)}

Candidate Items (use candidate_id only in output):
{json.dumps(aliased_candidates, ensure_ascii=False, indent=2)}

Memory policy:
- The graph memory facts are weak evidence from prior observed failures.
- Use a memory fact only when it matches current history or candidate facts.
- If a memory fact conflicts with item facts, ignore it.
- Do not overgeneralize from a single failure memory.

Output requirements:
- Return ONLY valid JSON. No markdown.
- Output one score row for every candidate_id exactly once.
- Score is a number from 0.0 to 1.0.
- Rationale must be <= 8 words.

JSON format:
{{
  "scores": [
    {{"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}}
  ],
  "reasoning": "one short sentence"
}}
"""
    else:
        prompt = f"""
You are scoring candidate items for a recommender system based only on user history and candidate facts.
{prompt_sample}

Inputs:
User Memory Profile initialized from observed history:
{json.dumps(profile_block, ensure_ascii=False, indent=2)}

User Recent History:
{json.dumps(train_items[-10:], ensure_ascii=False, indent=2)}

Candidate Items (use candidate_id only in output):
{json.dumps(aliased_candidates, ensure_ascii=False, indent=2)}

Output requirements:
- Return ONLY valid JSON. No markdown.
- Output one score row for every candidate_id exactly once.
- Score is a number from 0.0 to 1.0.
- Rationale must be <= 8 words.

JSON format:
{{
  "scores": [
    {{"candidate_id": "C01", "score": 0.0, "rationale": "short reason"}}
  ],
  "reasoning": "one short sentence"
}}
"""

    score_json_schema = {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string", "enum": valid_candidate_aliases},
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"},
                    },
                    "required": ["candidate_id", "score", "rationale"],
                    "additionalProperties": False,
                },
                "minItems": len(valid_candidate_aliases),
                "maxItems": len(valid_candidate_aliases),
            },
        },
        "required": ["scores"],
        "additionalProperties": False,
    }
    compact_schema_styles = {
        "compact_score",
        "compact_safe_residual_score",
        "memrec_vanilla",
        "weak_memory_score",
        "weak_memory_evidence_score",
        "weak_memory_router_score",
        "weak_anchor_score",
        "weak_anchor_router_score",
        "compact_stage_r",
        "compact_stage_r_reasoning",
        "compact_curated_score",
    }
    if ranking_prompt_style not in compact_schema_styles:
        score_json_schema["properties"]["reasoning"] = {"type": "string"}
        score_json_schema["required"] = ["scores", "reasoning"]

    max_retries = int(os.getenv("MEMCF_RANK_RETRIES", "1"))
    current_prompt = prompt
    base_cache_path = None
    if ranking_score_cache_dir:
        base_cache_key = hashlib.sha256(
            (ranking_prompt_style + "\n" + prompt).encode("utf-8")
        ).hexdigest()
        base_cache_path = os.path.join(ranking_score_cache_dir, f"{base_cache_key}.json")
    for attempt in range(max_retries + 1):
        try:
            cache_path = None
            raw_response = None
            if ranking_score_cache_dir:
                cache_key = hashlib.sha256(
                    (ranking_prompt_style + "\n" + current_prompt).encode("utf-8")
                ).hexdigest()
                cache_path = os.path.join(ranking_score_cache_dir, f"{cache_key}.json")
                if os.path.isfile(cache_path):
                    with open(cache_path, "r", encoding="utf-8") as cache_file:
                        cached = json.load(cache_file)
                    # Legacy cache entries may contain malformed first-attempt
                    # responses. Only replay outputs that passed validation.
                    if cached.get("is_valid") is True:
                        raw_response = str(cached["raw_response"])
                        memory_system._trace("ranking_score_cache_hit", {
                            **(trace_context or {}),
                            "cache_path": cache_path,
                            "prompt_hash": cache_key,
                        })
            if raw_response is None:
                raw_response = memory_system.qwen_generate(
                    prompt=current_prompt,
                    role_prompt=(
                        "You are a deterministic recommender scorer. "
                        "Return JSON only and follow the provided JSON schema exactly."
                    ),
                    max_new_tokens=int(os.getenv("MEMCF_RANK_MAX_TOKENS", "1400")),
                    json_schema=score_json_schema,
                    json_mode=True,
                    call_type="ranking",
                )
            try:
                result = extract_json_object(raw_response)
                raw_scores = result.get("scores", [])
            except Exception:
                result = {
                    "scores": parse_score_entries_from_text(raw_response, alias_to_item_id),
                    "reasoning": "Recovered score rows from malformed JSON",
                }
                raw_scores = result.get("scores", [])
            ranked_ids, validation = score_entries_to_ranking(raw_scores, alias_to_item_id)
            pairwise_cf_audit = {"enabled": False}
            if pairwise_cf_rerank:
                corrections = build_pairwise_cf_corrections(
                    pairwise_memory_rows or [],
                    aliased_candidates,
                    alias_to_item_id=alias_to_item_id,
                    max_corrections=int(os.getenv("MEMCF_PAIRWISE_CF_MAX_CORRECTIONS", "8")),
                )
                adjusted_scores, pairwise_cf_audit = apply_pairwise_cf_score_adjustments(
                    validation.get("parsed_scores", []),
                    corrections,
                    alpha=pairwise_cf_alpha,
                    beta=pairwise_cf_beta,
                )
                if pairwise_cf_audit.get("num_applied_corrections", 0) > 0:
                    adjusted_ranked_ids, adjusted_validation = score_entries_to_ranking(adjusted_scores, alias_to_item_id)
                    ranked_ids = adjusted_ranked_ids
                    validation["pairwise_cf_adjusted_validation"] = adjusted_validation
                    validation["pairwise_cf_adjusted_scores"] = adjusted_scores
            failure_constraint_audit = {"enabled": False, "mode": failure_constraint_mode}
            if failure_constraint_mode != "none":
                constrained_ranked_ids, failure_constraint_audit = apply_typed_failure_constraints(
                    parsed_scores=validation.get("parsed_scores", []),
                    evidence_rows=failure_constraint_evidence or [],
                    mode=failure_constraint_mode,
                    tie_epsilon=failure_constraint_tie_epsilon,
                    min_cross_support=failure_constraint_min_cross_support,
                    candidate_popularity=failure_constraint_candidate_popularity,
                    max_cross_corrections=failure_constraint_max_cross_corrections,
                )
                ranked_ids = constrained_ranked_ids
                memory_system.memory_diagnostics["failure_constraint_users"] += 1
                memory_system.memory_diagnostics["failure_constraint_evidence"] += int(
                    failure_constraint_audit.get("num_evidence_rows", 0)
                )
                if failure_constraint_audit.get("num_moved_candidates", 0) > 0:
                    memory_system.memory_diagnostics["failure_constraint_changed_users"] += 1
                memory_system.memory_diagnostics["failure_constraint_moved_candidates"] += int(
                    failure_constraint_audit.get("num_moved_candidates", 0)
                )
            memory_system.memory_diagnostics["rank_score_calls"] += 1
            if validation["is_valid"]:
                memory_system.memory_diagnostics["rank_valid_score_outputs"] += 1
            else:
                memory_system.memory_diagnostics["rank_invalid_score_outputs"] += 1
            memory_system._trace("ranking_llm", {
                **(trace_context or {}),
                "attempt": attempt,
                "memcf_graph": True,
                "ranking_mode": "score_based_candidate_alias_graph_facets",
                "prompt": current_prompt,
                "answer": raw_response,
                "parsed": result,
                "score_validation": validation,
                "pairwise_cf_audit": pairwise_cf_audit,
                "failure_constraint_audit": failure_constraint_audit,
                "cleaned_ranked_item_ids": ranked_ids,
                "candidate_items": candidate_info,
                "aliased_candidate_items": aliased_candidates,
                "alias_to_item_id": alias_to_item_id,
                "train_items": train_items[-10:],
                "user_memory_profile": asdict(user_profile) if user_profile else None,
                "memory_facts": facets,
                "use_graph_memory_facts": bool(facets),
            })
            if validation["is_valid"] and base_cache_path:
                # Cache the final valid response under the original prompt key.
                # This makes replay variants exactly paired even when the base
                # run needed a repair prompt on an earlier attempt.
                os.makedirs(ranking_score_cache_dir, exist_ok=True)
                tmp_path = f"{base_cache_path}.{os.getpid()}.tmp"
                with open(tmp_path, "w", encoding="utf-8") as cache_file:
                    json.dump(
                        {"raw_response": raw_response, "is_valid": True},
                        cache_file,
                        ensure_ascii=False,
                    )
                os.replace(tmp_path, base_cache_path)
            if validation["is_valid"] or attempt >= max_retries:
                return ranked_ids
            current_prompt = f"""{prompt}

The previous answer was invalid:
{json.dumps(validation, ensure_ascii=False, indent=2)}

Retry now. Return ONLY valid JSON with exactly one score row for every candidate_id.
"""
        except Exception as e:
            memory_system.memory_diagnostics["rank_attempt_errors"] += 1
            memory_system._trace("ranking_attempt_error", {
                **(trace_context or {}),
                "attempt": attempt,
                "error": str(e),
                "prompt": current_prompt,
                "candidate_items": candidate_info,
                "memory_facts": facets,
            })
            if attempt >= max_retries:
                break
    memory_system.memory_diagnostics["rank_fallbacks"] += 1
    return [str(item["item_id"]) for item in candidate_items]


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


def evaluate_user_v2(
    user_id: str,
    user_data: Dict[str, Any],
    negative_data: Dict[str, Any],
    items_meta: Dict[str, Dict[str, Any]],
    memory_system: RecommendationMemorySystem,
    graph: MemoryGraphIndex,
    user_profiles: Dict[str, UserMemoryProfile],
    eval_type: str = "test",
    use_memory: bool = True,
    graph_memory_k: int = 3,
    neighbor_k: int = 10,
    dense_pool_per_signal: int = 15,
    consensus_top_n_users: int = 15,
    consensus_min_users: int = 2,
    fmrec_top_k_neighbors: int = 3,
    min_evidence_terms: int = 1,
    max_positive_interactions: Optional[int] = None,
    max_negative_candidates: Optional[int] = None,
    no_harm_arbitration: bool = False,
    ranking_prompt_style: str = "memcf",
    graph_retrieval_scope: str = "full",
    max_memory_facts: int = 3,
    max_memory_fact_words: int = 55,
    memory_token_budget: int = 420,
    strict_memory_applicability: bool = False,
    min_candidate_matches: int = 1,
    allow_same_user_without_candidate_match: bool = True,
    allow_random_memory_injection: bool = False,
    reject_wrong_only_memory: bool = False,
    disable_user_profile_in_eval_prompt: bool = False,
    memory_selector: str = "none",
    memory_selector_top_m: int = 12,
    memory_selector_top_k: int = 3,
    memory_selector_min_relevance: float = 0.60,
    memory_selector_neutral_cross: bool = False,
    pairwise_cf_rerank: bool = False,
    pairwise_cf_alpha: float = 0.04,
    pairwise_cf_beta: float = 0.04,
    pairwise_cf_hide_memory_prompt: bool = False,
    failure_constraint_mode: str = "none",
    failure_constraint_tie_epsilon: float = 0.0,
    failure_constraint_min_cross_support: int = 2,
    failure_constraint_min_context_terms: int = 1,
    failure_constraint_same_budget: int = 32,
    failure_constraint_cross_budget: int = 128,
    failure_constraint_min_shared_items: int = 1,
    failure_constraint_max_cross_corrections: int = 3,
    cf_source_budget: int = 2,
    cf_control_seed: int = 2027,
    ranking_score_cache_dir: Optional[str] = None,
    failure_constraint_with_prompt_memory: bool = False,
    safe_residual_pool_size: int = 32,
    safe_residual_min_cross_users: int = 2,
    safe_residual_max_cross_facts: int = 1,
    safe_residual_max_same_facts: int = 2,
    safe_residual_semantic_consensus: bool = False,
    safe_residual_semantic_min_terms: int = 2,
    safe_residual_min_vote_margin: int = 0,
    safe_residual_verify_cross: bool = False,
    safe_residual_verify_min_confidence: float = 0.65,
    temporal_same_k: int = 2,
    temporal_cross_k: int = 1,
    temporal_control_seed: int = 2027,
    matched_endpoint_scope: str = "exact",
) -> Tuple[Dict[str, float], Dict[str, float], List[str], List[str], List[str]]:
    if eval_type == "val":
        ground_truth = [str(x) for x in user_data.get("val", [])]
        negatives = [str(x) for x in negative_data.get("val_neg", [])]
    else:
        ground_truth = [str(x) for x in user_data.get("test", [])]
        negatives = [str(x) for x in negative_data.get("test_neg", [])]
    if max_negative_candidates and max_negative_candidates > 0:
        negatives = negatives[:max_negative_candidates]
    candidates = deterministic_shuffle(ground_truth + negatives, salt=f"{eval_type}_candidates")

    train_history = [str(x) for x in user_data.get("train", [])]
    if max_positive_interactions and max_positive_interactions > 0:
        train_history_for_prompt = train_history[-max_positive_interactions:]
    else:
        train_history_for_prompt = train_history[-10:]
    train_items_info = _history_item_infos(train_history_for_prompt, items_meta)
    candidate_items_info = [_item_info_for_prompt(item_id, items_meta) for item_id in candidates]
    user_profile = user_profiles.get(str(user_id))
    eval_user_profile = None if disable_user_profile_in_eval_prompt else user_profile

    retrieved_graph_lessons: List[GraphRetrievedLesson] = []
    failure_constraint_evidence: List[Dict[str, Any]] = []
    memory_reader_result = {"use_memory": False, "facets": [], "reason": "memory disabled"}
    if use_memory:
        user_context_text = (
            _context_text_from_items(train_items_info)
            + " "
            + (eval_user_profile.profile.lower() if eval_user_profile else "")
        )
        current_context_text = (
            user_context_text
            + " "
            + _context_text_from_items(candidate_items_info)
        )
        if failure_constraint_mode != "none":
            failure_constraint_evidence = graph.typed_failure_evidence(
                user_id=str(user_id),
                candidate_ids=candidates,
                user_context_text=user_context_text,
                mode=failure_constraint_mode,
                min_context_terms=failure_constraint_min_context_terms,
                max_same_evidence=failure_constraint_same_budget,
                max_cross_evidence=failure_constraint_cross_budget,
                shuffle_salt=f"{eval_type}:{user_id}:{','.join(candidates)}",
                min_shared_items=failure_constraint_min_shared_items,
                cf_source_budget=cf_source_budget,
                cf_control_seed=cf_control_seed,
            )
            memory_system.record_memory_diagnostics(
                retrieved=len(failure_constraint_evidence),
                kept=len(failure_constraint_evidence),
                skipped=0,
            )
            memory_system._trace("typed_failure_evidence", {
                "user_id": user_id,
                "eval_type": eval_type,
                "failure_constraint_mode": failure_constraint_mode,
                "candidate_item_ids": candidates,
                "recent_history_ids": train_history_for_prompt,
                "evidence": failure_constraint_evidence,
                "cf_control_audit": dict(graph.last_cf_control_audit),
            })
            memory_reader_result = {
                "use_memory": bool(failure_constraint_evidence) or failure_constraint_mode == "popularity",
                "facets": [],
                "memory_facts": [],
                "selected_rows": failure_constraint_evidence,
                "rejected_rows": [],
                "reason": "typed failure constraints are applied after clean LLM scoring",
                "failure_constraint_mode": failure_constraint_mode,
            }
        if failure_constraint_mode == "none" or failure_constraint_with_prompt_memory:
            retrieval_top_k = graph_memory_k
            if memory_selector in {"llm", "heuristic"} or graph_retrieval_scope.startswith("dense_lgcn_fmrec_pool"):
                # A pool scope must retrieve the whole pool even with no
                # selector, otherwise it collapses back to top-3-by-confidence
                # of a 3-row pool, i.e. the old one-per-donor behaviour.
                retrieval_top_k = max(graph_memory_k, int(memory_selector_top_m or 0))
            if graph_retrieval_scope in {"safe_residual", "directional_residual"}:
                retrieval_top_k = max(graph_memory_k, int(safe_residual_pool_size or 0))
            retrieved_graph_lessons = graph.retrieve(
                user_id=str(user_id),
                recent_history_ids=train_history_for_prompt,
                candidate_ids=candidates,
                current_context_text=current_context_text,
                top_k=retrieval_top_k,
                neighbor_k=neighbor_k,
                dense_pool_per_signal=dense_pool_per_signal,
                consensus_top_n_users=consensus_top_n_users,
                consensus_min_users=consensus_min_users,
                fmrec_top_k_neighbors=fmrec_top_k_neighbors,
                min_evidence_terms=min_evidence_terms,
                retrieval_scope=graph_retrieval_scope,
                shuffle_salt=f"{eval_type}:{user_id}:{','.join(candidates)}",
                temporal_same_k=temporal_same_k,
                temporal_cross_k=temporal_cross_k,
                temporal_control_seed=temporal_control_seed,
                matched_endpoint_scope=matched_endpoint_scope,
            )
            if not graph_retrieval_scope.startswith("temporal_"):
                memory_system.record_memory_diagnostics(
                    retrieved=len(retrieved_graph_lessons),
                    kept=len(retrieved_graph_lessons),
                    skipped=0,
                )
            memory_system._trace("graph_memory_retrieval", {
                "user_id": user_id,
                "eval_type": eval_type,
                "graph_memory_k": graph_memory_k,
                "retrieval_top_k": retrieval_top_k,
                "neighbor_k": neighbor_k,
                "min_evidence_terms": min_evidence_terms,
                "graph_retrieval_scope": graph_retrieval_scope,
                "memory_selector": memory_selector,
                "user_cluster_id": graph.cluster_by_user.get(str(user_id)),
                "candidate_item_ids": candidates,
                "recent_history_ids": train_history_for_prompt,
                "retrieved": [
                    {
                        "lesson": asdict(r.lesson),
                        "score": r.score,
                        "sources": r.sources,
                        "paths": r.paths,
                        "matched_evidence_terms": r.matched_evidence_terms,
                        "exposure_type": r.exposure_type,
                        "candidate_role": r.candidate_role,
                        "shared_history_items": r.shared_history_items,
                        "control_mode": r.control_mode,
                    }
                    for r in retrieved_graph_lessons
                ],
            })
            if graph_retrieval_scope.startswith("temporal_j_"):
                memory_system._trace("temporal_treatment_plan", {
                    "user_id": user_id,
                    "eval_type": eval_type,
                    "graph_retrieval_scope": graph_retrieval_scope,
                    "audit": dict(graph.last_temporal_retrieval_audit),
                })
            if graph_retrieval_scope.startswith("temporal_"):
                memory_reader_result = read_temporal_lexicographic_memory_v2(
                    memory_system=memory_system,
                    user_profile=eval_user_profile,
                    train_items=train_items_info,
                    candidate_items=candidate_items_info,
                    retrieved_lessons=retrieved_graph_lessons,
                    trace_context={
                        "user_id": user_id,
                        "eval_type": eval_type,
                        "graph_retrieval_scope": graph_retrieval_scope,
                    },
                    max_memory_facts=max_memory_facts,
                    max_memory_fact_words=max_memory_fact_words,
                    memory_token_budget=memory_token_budget,
                    retrieval_audit=graph.last_temporal_retrieval_audit,
                    memory_selector=memory_selector,
                    memory_selector_top_k=memory_selector_top_k,
                    memory_selector_min_relevance=memory_selector_min_relevance,
                )
            else:
                memory_reader_result = read_graph_lessons_as_facets_v2(
                    memory_system=memory_system,
                    user_profile=eval_user_profile,
                    train_items=train_items_info,
                    candidate_items=candidate_items_info,
                    retrieved_lessons=retrieved_graph_lessons,
                    trace_context={"user_id": user_id, "eval_type": eval_type, "graph_retrieval_scope": graph_retrieval_scope},
                    max_memory_facts=max_memory_facts,
                    max_memory_fact_words=max_memory_fact_words,
                    memory_token_budget=memory_token_budget,
                    strict_candidate_applicability=strict_memory_applicability,
                    min_candidate_matches=min_candidate_matches,
                    allow_same_user_without_candidate_match=allow_same_user_without_candidate_match,
                    allow_random_memory_injection=allow_random_memory_injection,
                    reject_wrong_only_memory=reject_wrong_only_memory,
                    memory_selector=memory_selector,
                    memory_selector_top_k=memory_selector_top_k,
                    memory_selector_min_relevance=memory_selector_min_relevance,
                    memory_selector_top_m=memory_selector_top_m,
                    memory_selector_neutral_cross=memory_selector_neutral_cross,
                    safe_residual_mode=(graph_retrieval_scope in {"safe_residual", "directional_residual"}),
                    safe_residual_min_cross_users=safe_residual_min_cross_users,
                    safe_residual_max_cross_facts=safe_residual_max_cross_facts,
                    safe_residual_max_same_facts=safe_residual_max_same_facts,
                    safe_residual_semantic_consensus=safe_residual_semantic_consensus,
                    safe_residual_semantic_min_terms=safe_residual_semantic_min_terms,
                    safe_residual_min_vote_margin=safe_residual_min_vote_margin,
                    safe_residual_verify_cross=safe_residual_verify_cross,
                    safe_residual_verify_min_confidence=safe_residual_verify_min_confidence,
                )

    memory_facets = memory_reader_result.get("facets", []) if memory_reader_result.get("use_memory") else []
    pairwise_memory_rows = memory_reader_result.get("selected_rows", []) if (use_memory and pairwise_cf_rerank) else []
    ranking_memory_facets = [] if (pairwise_cf_rerank and pairwise_cf_hide_memory_prompt) else memory_facets
    selected_ranking_source = (
        "graph_memory_plus_typed_constraints"
        if failure_constraint_mode != "none" and memory_facets
        else "typed_failure_constraints"
        if failure_constraint_mode != "none"
        else "graph_memory_facts" if memory_facets else "no_memory"
    )
    no_memory_predictions = None
    memory_predictions = None
    arbitration = {"enabled": False, "selected_ranking_source": selected_ranking_source}
    if use_memory and no_harm_arbitration:
        memory_system.memory_diagnostics["no_harm_users"] += 1
        no_memory_predictions = llm_ranking_v2(
            memory_system, train_items_info, candidate_items_info, eval_user_profile, [],
            ranking_prompt_style=ranking_prompt_style,
            trace_context={"user_id": user_id, "eval_type": eval_type, "ranking_path": "v2_no_harm_no_memory"},
        )
        if memory_facets:
            memory_predictions = llm_ranking_v2(
                memory_system, train_items_info, candidate_items_info, eval_user_profile, ranking_memory_facets,
                ranking_prompt_style=ranking_prompt_style,
                trace_context={"user_id": user_id, "eval_type": eval_type, "ranking_path": "v2_no_harm_memory"},
                pairwise_cf_rerank=pairwise_cf_rerank,
                pairwise_memory_rows=pairwise_memory_rows,
                pairwise_cf_alpha=pairwise_cf_alpha,
                pairwise_cf_beta=pairwise_cf_beta,
            )
            # Conservative rule: use memory only if the selected facts carry
            # candidate-applicable evidence. In non-strict mode this reduces to
            # the historical same_user/candidate_item path rule.
            selected_rows_for_no_harm = memory_reader_result.get("selected_rows", [])
            strong_path = any(
                ("same_user" in r.sources or "candidate_item" in r.sources or "cluster_user" in r.sources)
                for r in retrieved_graph_lessons
            )
            candidate_supported = any(
                bool(row.get("candidate_support") or row.get("direct_candidate_match") or row.get("candidate_matches"))
                for row in selected_rows_for_no_harm
            )
            use_memory_ranking = strong_path and (candidate_supported or not strict_memory_applicability)
            if use_memory_ranking:
                predictions = memory_predictions
                selected_ranking_source = "graph_memory_facts"
                memory_system.memory_diagnostics["no_harm_used_memory"] += 1
            else:
                predictions = no_memory_predictions
                selected_ranking_source = "no_memory"
                memory_system.memory_diagnostics["no_harm_fallback_no_memory"] += 1
            arbitration = {
                "enabled": True,
                "selected_ranking_source": selected_ranking_source,
                "strong_graph_path": strong_path,
                "candidate_supported": candidate_supported,
                "strict_memory_applicability": strict_memory_applicability,
                "reason": (
                    "use memory ranking: graph path and candidate evidence passed"
                    if use_memory_ranking
                    else "fallback to no-memory: insufficient candidate-supported memory evidence"
                ),
            }
        else:
            predictions = no_memory_predictions
            selected_ranking_source = "no_memory"
            memory_system.memory_diagnostics["no_harm_fallback_no_memory"] += 1
            arbitration = {
                "enabled": True,
                "selected_ranking_source": selected_ranking_source,
                "reason": "fallback to no-memory: no accepted facets",
            }
        memory_system._trace("no_harm_arbitration", {
            "user_id": user_id,
            "eval_type": eval_type,
            "decision": arbitration,
            "no_memory_predictions": no_memory_predictions,
            "memory_predictions": memory_predictions,
            "selected_predictions": predictions,
            "memory_reader_result": memory_reader_result,
        })
    else:
        predictions = llm_ranking_v2(
            memory_system, train_items_info, candidate_items_info, eval_user_profile, ranking_memory_facets,
            ranking_prompt_style=ranking_prompt_style,
            trace_context={
                "user_id": user_id,
                "eval_type": eval_type,
                "ranking_path": "v2_single_path",
                "use_memory": use_memory,
                "selected_ranking_source": selected_ranking_source,
                "pairwise_cf_rerank": pairwise_cf_rerank,
                "pairwise_cf_hide_memory_prompt": pairwise_cf_hide_memory_prompt,
            },
            pairwise_cf_rerank=pairwise_cf_rerank,
            pairwise_memory_rows=pairwise_memory_rows,
            pairwise_cf_alpha=pairwise_cf_alpha,
            pairwise_cf_beta=pairwise_cf_beta,
            failure_constraint_mode=failure_constraint_mode,
            failure_constraint_evidence=failure_constraint_evidence,
            failure_constraint_tie_epsilon=failure_constraint_tie_epsilon,
            failure_constraint_min_cross_support=failure_constraint_min_cross_support,
            failure_constraint_max_cross_corrections=failure_constraint_max_cross_corrections,
            failure_constraint_candidate_popularity={
                item_id: len(graph.users_by_item.get(str(item_id), set()))
                for item_id in candidates
            },
            ranking_score_cache_dir=ranking_score_cache_dir,
        )

    baseline_metric = calculate_paper_ranking_metrics(candidates, ground_truth)
    metrics = calculate_paper_ranking_metrics(predictions, ground_truth)
    memory_system._trace("ranking_result", {
        "user_id": user_id,
        "eval_type": eval_type,
        "use_memory": use_memory,
        "ground_truth": ground_truth,
        "candidate_item_ids": candidates,
        "ranked_item_ids": predictions,
        "metrics": metrics,
        "baseline_metrics": baseline_metric,
        "selected_ranking_source": selected_ranking_source,
        "ranking_prompt_style": ranking_prompt_style,
        "pairwise_cf_rerank": pairwise_cf_rerank,
        "pairwise_cf_hide_memory_prompt": pairwise_cf_hide_memory_prompt,
        "pairwise_cf_alpha": pairwise_cf_alpha,
        "pairwise_cf_beta": pairwise_cf_beta,
        "failure_constraint_mode": failure_constraint_mode,
        "failure_constraint_tie_epsilon": failure_constraint_tie_epsilon,
        "failure_constraint_min_cross_support": failure_constraint_min_cross_support,
        "failure_constraint_min_shared_items": failure_constraint_min_shared_items,
        "failure_constraint_max_cross_corrections": failure_constraint_max_cross_corrections,
        "failure_constraint_with_prompt_memory": failure_constraint_with_prompt_memory,
        "failure_constraint_evidence": failure_constraint_evidence,
        "memory_reader_result": memory_reader_result,
        "retrieved_graph_lessons": [
            {
                "lesson": asdict(r.lesson),
                "score": r.score,
                "sources": r.sources,
                "paths": r.paths,
                "matched_evidence_terms": r.matched_evidence_terms,
                "exposure_type": r.exposure_type,
                "candidate_role": r.candidate_role,
                "shared_history_items": r.shared_history_items,
                "control_mode": r.control_mode,
            }
            for r in retrieved_graph_lessons
        ],
        "no_memory_predictions": no_memory_predictions,
        "memory_predictions": memory_predictions,
    })
    return baseline_metric, metrics, candidates, predictions, ground_truth


def parse_args_v2():
    parser = argparse.ArgumentParser(description="MEMCF graph-memory experiment")
    parser.add_argument("--data_name", type=str, default="Video_Game")
    parser.add_argument("--use_memory", action="store_true", default=True)
    parser.add_argument("--no_use_memory", action="store_false", dest="use_memory")
    parser.add_argument("--LOAD_SAVED_MEMORY", action="store_true", default=False)
    parser.add_argument("--max_iterations", type=int, default=1)
    parser.add_argument("--number_of_users", type=int, default=100)
    parser.add_argument("--max_positive_interactions", type=int, default=5)
    parser.add_argument("--max_negative_candidates", type=int, default=19)
    parser.add_argument("--graph_memory_k", type=int, default=3)
    parser.add_argument(
        "--k_memories",
        type=int,
        default=None,
        help="Legacy alias for --graph_memory_k.",
    )
    parser.add_argument("--neighbor_k", type=int, default=10)
    parser.add_argument(
        "--dense_pool_per_signal", type=int, default=15,
        help=(
            "dense_lgcn_decomposed only: size of the top-N candidate pool taken "
            "independently from EACH similarity signal before union/scoring. "
            "Exposed for the k-sensitivity ablation (RQ3)."
        ),
    )
    parser.add_argument(
        "--consensus_top_n_users", type=int, default=15,
        help=(
            "dense_lgcn_userscore_consensus[_cross_only] only: size of the "
            "similar-user pool (ranked by LGCN user-embedding cosine only) "
            "considered for cross-user consensus voting."
        ),
    )
    parser.add_argument(
        "--consensus_min_users", type=int, default=2,
        help=(
            "dense_lgcn_userscore_consensus[_cross_only] only: minimum number "
            "of distinct similar source users that must agree on the same "
            "candidate item (prefer/avoid) before it is surfaced."
        ),
    )
    parser.add_argument(
        "--fmrec_top_k_neighbors", type=int, default=3,
        help=(
            "dense_lgcn_fmrec_topk only: number of top similar-by-LGCN-"
            "cosine other users whose single best (highest-confidence) "
            "lesson each gets surfaced unconditionally, no gate. Mirrors "
            "github.com/dangkh/FMRec's default top_k_neighbors=3."
        ),
    )
    parser.add_argument(
        "--memory_retrieval_mode",
        type=str,
        default="graph",
        help="Legacy compatibility flag. Only graph retrieval is supported by the v2 runner.",
    )
    parser.add_argument("--min_evidence_terms", type=int, default=1)
    parser.add_argument("--no_harm_arbitration", action="store_true", default=False)
    parser.add_argument("--candidate_negative_mode", type=str, default="candidate_hard",
                        choices=["random", "candidate_hard"])
    parser.add_argument(
        "--training_negative_source",
        type=str,
        default="legacy_runtime",
        choices=["legacy_runtime", "train_catalog"],
        help=(
            "Negative source for failure-memory creation. train_catalog never "
            "reads val_neg/test_neg; legacy_runtime preserves old runs."
        ),
    )
    parser.add_argument(
        "--failure_memory_protocol",
        type=str,
        default="legacy",
        choices=["legacy", "temporal_factual"],
        help="Failure collection protocol. temporal_factual uses strict train prefixes and fixed factual lessons.",
    )
    parser.add_argument("--min_lesson_confidence", type=float, default=0.25)
    parser.add_argument("--max_lesson_risk", type=float, default=0.85)
    parser.add_argument("--max_failure_lessons_per_user", type=int, default=3)
    parser.add_argument("--ranking_prompt_style", type=str, default="compact_score",
                        choices=[
                            "memcf", "compact_score", "memrec_old",
                            "memrec_vanilla", "weak_memory_score",
                            "weak_memory_evidence_score", "weak_memory_router_score",
                            "weak_anchor_score", "weak_anchor_router_score",
                            "compact_stage_r", "compact_stage_r_reasoning",
                            "compact_curated_score", "compact_safe_residual_score",
                        ])
    parser.add_argument("--graph_retrieval_scope", type=str, default="full",
                        choices=[
                            "full", "same_user", "candidate_item", "history_item",
                            "neighbor_user", "same_user_first", "candidate_strict",
                            "safe_residual", "directional_residual",
                            "temporal_same", "temporal_exact", "temporal_abstract",
                            "temporal_full", "temporal_cross_only", "temporal_shuffled",
                            "temporal_matched_random",
                            "temporal_cluster_only", "temporal_cluster_residual",
                            "temporal_cluster_global", "temporal_cluster_random",
                            "temporal_cluster_shuffled",
                            "temporal_j_same", "temporal_j_full",
                            "temporal_j_true_matched", "temporal_j_shuffled_matched",
                            "temporal_j_random_matched",
                            "cross_user_only", "cluster_user", "cluster_full",
                            "hybrid_cluster", "hybrid_cluster_strict",
                            "random_memory", "shuffled_memory", "random_memory_clean", "shuffled_memory_clean",
                            "random_cluster", "shuffled_cluster",
                            "full_lgcn", "full_lgcn_cluster", "full_lgcn_random",
                            "dense_lgcn", "dense_lgcn_agree", "dense_lgcn_consensus", "dense_lgcn_random",
                            "dense_lgcn_consensus_anchored", "dense_lgcn_consensus_anchored_random",
                            "dense_lgcn_consensus_maxsim", "dense_lgcn_consensus_maxsim_random",
                            "dense_lgcn_decomposed", "dense_lgcn_decomposed_random",
                            "dense_lgcn_decomposed_cross_only", "dense_lgcn_decomposed_cross_only_random",
                            "dense_lgcn_decomposed_scored", "dense_lgcn_decomposed_scored_random",
                            "dense_lgcn_userscore_consensus", "dense_lgcn_userscore_consensus_random",
                            "dense_lgcn_userscore_consensus_cross_only", "dense_lgcn_userscore_consensus_cross_only_random",
                            "dense_lgcn_fmrec_topk", "dense_lgcn_fmrec_topk_random",
            "dense_lgcn_fmrec_topk_shared", "dense_lgcn_fmrec_topk_popular",
            "dense_lgcn_fmrec_topk_noself", "dense_lgcn_fmrec_topk_noself_random",
            "dense_lgcn_fmrec_pool", "dense_lgcn_fmrec_pool_random",
                            "oracle_cross_candidate",
                            "dense_lgcn_rrf", "dense_lgcn_rrf_random",
                        ],
                        help="Failure-graph retrieval ablation scope. Distinct from MemRec neighbor pruning.")
    parser.add_argument("--temporal_same_k", type=int, default=2,
                        help="Reserved same-user slots for temporal lexicographic retrieval.")
    parser.add_argument("--temporal_cross_k", type=int, default=1,
                        help="Reserved cross-user slots for temporal lexicographic retrieval.")
    parser.add_argument("--eval_workers", type=int, default=1,
                         help="Number of parallel worker threads for Phase 2 "
                              "(read-only) evaluation. Safe for eval_only runs "
                              "with a precomputed --memory_file, since no lessons "
                              "are written during evaluation. Diagnostic counters "
                              "(memory_diagnostics dict) are best-effort under "
                              "concurrency and may undercount slightly; the "
                              "primary ndcg/recall metrics are unaffected since "
                              "each user's result is computed independently and "
                              "collected back on the main thread.")
    parser.add_argument("--temporal_control_seed", type=int, default=2027,
                        help="Seed for degree-preserving shuffled and matched-random temporal controls.")
    parser.add_argument(
        "--matched_endpoint_scope", type=str, default="exact",
        choices=["exact", "candidate_pool", "category_pool"],
        help=(
            "MEMCF-J/K matched-triplet pairing granularity. 'exact' (default) "
            "requires the shuffled/random control source to reference the "
            "exact same candidate item as the true source, matching the "
            "original MEMCF-J design (narrow pool, ~10%% matched coverage in "
            "the Software 100-user pilot). 'candidate_pool' relaxes pairing "
            "to same candidate_role within the current candidate set "
            "(~36%% coverage), but lets the specific praised/avoided item "
            "differ across true/shuffled/random -- a real confound. "
            "'category_pool' is a middle ground: pairs on same "
            "candidate_role + same item category, keeping true/shuffled/"
            "random topically comparable while still raising coverage above "
            "'exact'. See reports/MEMCF_J_priority1_diagnostic_20260731.md "
            "and reports/MEMCF_K_pilot_20260802_results_and_fix.md."
        ),
    )
    parser.add_argument(
        "--cluster_memory_file", type=str, default=None,
        help=(
            "Optional C-MEMCF cluster consensus artifact. It is built offline from "
            "train-only CF embeddings and temporal failure lessons. Required by "
            "temporal_cluster_* retrieval scopes."
        ),
    )
    parser.add_argument(
        "--lightgcn_embeddings_json", type=str, default=None,
        help=(
            "Optional raw per-user LightGCN propagated-embedding dump "
            "(dump_lightgcn_embeddings.py). Required by full_lgcn / "
            "full_lgcn_cluster retrieval scopes; unused otherwise."
        ),
    )
    parser.add_argument("--max_memory_facts", type=int, default=3)
    parser.add_argument("--max_memory_fact_words", type=int, default=55)
    parser.add_argument("--memory_token_budget", type=int, default=420)
    parser.add_argument("--strict_memory_applicability", action="store_true", default=False,
                        help="Require selected graph memories to have current-candidate support before entering ranking prompts.")
    parser.add_argument("--min_candidate_matches", type=int, default=1,
                        help="Minimum candidate-side evidence terms for strict memory applicability.")
    parser.add_argument("--require_same_user_candidate_match", action="store_true", default=False,
                        help="In strict mode, same-user memories also need current-candidate support.")
    parser.add_argument("--allow_random_memory_injection", action="store_true", default=False,
                        help="For random-memory controls, inject random facts even under strict applicability gates.")
    parser.add_argument("--reject_wrong_only_memory", action="store_true", default=False,
                        help="Reject memory facts whose only direct current-candidate match is the past wrong item.")
    parser.add_argument("--profile_only", action="store_true", default=False,
                        help="Initialize/load user profiles but disable graph-memory retrieval during evaluation.")
    parser.add_argument("--disable_user_profile_in_eval_prompt", action="store_true", default=False,
                        help="Do not include the user profile block in evaluation ranking prompts or memory-fact selection.")
    parser.add_argument("--memory_selector", type=str, default="none", choices=["none", "llm", "heuristic"],
                        help="Optional memory applicability selector before ranking. It selects memory facts only, not items. "
                             "llm: one extra LLM call judges each pooled memory against history+candidates. "
                             "heuristic: no LLM call; rank pooled memories by term overlap with history/candidates.")
    parser.add_argument("--memory_selector_top_m", type=int, default=12,
                        help="When --memory_selector is llm or heuristic (or the scope is a *_pool scope), retrieve at least this many graph memories before selection.")
    parser.add_argument("--memory_selector_neutral_cross", action="store_true", default=False,
                        help="Drop the LLM selector's rule that cross-user memories need stronger evidence than same-user ones. "
                             "Required for a fair test of cross-user pooling; default off preserves earlier runs.")
    parser.add_argument("--memory_selector_top_k", type=int, default=3,
                        help="When --memory_selector=llm, keep at most this many selected memory facts.")
    parser.add_argument("--memory_selector_min_relevance", type=float, default=0.60,
                        help="Minimum selector relevance score for using a memory fact.")
    parser.add_argument("--safe_residual_pool_size", type=int, default=32,
                        help="Candidate graph facts inspected before safe-residual consensus selection.")
    parser.add_argument("--safe_residual_min_cross_users", type=int, default=2,
                        help="Distinct shared-history source users required for a cross-user residual fact.")
    parser.add_argument("--safe_residual_max_cross_facts", type=int, default=1,
                        help="Maximum collaborative residual facts added to one ranking prompt.")
    parser.add_argument("--safe_residual_max_same_facts", type=int, default=2,
                        help="Maximum same-user facts kept when a collaborative residual fact is present.")
    parser.add_argument("--safe_residual_semantic_consensus", action="store_true", default=False,
                        help="Allow cross-user consensus through unambiguous title/category matches, not only exact item IDs.")
    parser.add_argument("--safe_residual_semantic_min_terms", type=int, default=2,
                        help="Minimum shared item terms for semantic candidate grounding; one distinctive long term may suffice.")
    parser.add_argument("--safe_residual_min_vote_margin", type=int, default=0,
                        help="Required source-user margin between the selected support/avoid direction and its opposite.")
    parser.add_argument("--safe_residual_verify_cross", action="store_true", default=False,
                        help="Run a fail-closed LLM applicability check only when a cross-user residual survives consensus.")
    parser.add_argument("--safe_residual_verify_min_confidence", type=float, default=0.65,
                        help="Minimum verifier confidence required to inject a cross-user residual.")
    parser.add_argument("--pairwise_cf_rerank", action="store_true", default=False,
                        help="Apply candidate-pair corrective score deltas from selected failure memories after LLM scoring.")
    parser.add_argument("--pairwise_cf_alpha", type=float, default=0.04,
                        help="Maximum boost delta for pairwise CF corrective evidence.")
    parser.add_argument("--pairwise_cf_beta", type=float, default=0.04,
                        help="Maximum demotion delta for pairwise CF corrective evidence.")
    parser.add_argument("--pairwise_cf_hide_memory_prompt", action="store_true", default=False,
                        help="Use selected graph memories only for pairwise score correction, not as raw prompt text.")
    parser.add_argument(
        "--failure_constraint_mode",
        type=str,
        default="none",
        choices=sorted(FAILURE_CONSTRAINT_MODES),
        help="D-family typed failure constraint applied to clean LLM score ties.",
    )
    parser.add_argument("--failure_constraint_tie_epsilon", type=float, default=0.0,
                        help="Only reorder candidates whose base scores differ by at most this value.")
    parser.add_argument("--failure_constraint_min_cross_support", type=int, default=2,
                        help="Distinct cross-user support required by full_consensus mode.")
    parser.add_argument("--failure_constraint_min_context_terms", type=int, default=1,
                        help="User-history/profile evidence terms required for cross-user typed edges.")
    parser.add_argument("--failure_constraint_same_budget", type=int, default=32,
                        help="Maximum exact same-user evidence rows retained per query; <=0 keeps all.")
    parser.add_argument("--failure_constraint_cross_budget", type=int, default=128,
                        help="Maximum exact cross-user evidence rows retained per query; <=0 keeps all.")
    parser.add_argument("--failure_constraint_min_shared_items", type=int, default=1,
                        help="Minimum shared training items for an F-family collaborative source user.")
    parser.add_argument("--failure_constraint_max_cross_corrections", type=int, default=3,
                        help="Maximum cross-user candidate corrections per query in F-family modes.")
    parser.add_argument("--cf_source_budget", type=int, default=2,
                        help="Cross-user source count used by matched-budget G-family controls.")
    parser.add_argument("--cf_control_seed", type=int, default=2027,
                        help="Deterministic graph-shuffle/random-control seed for G-family controls.")
    parser.add_argument("--ranking_score_cache_dir", type=str, default=None,
                        help="Optional cache for clean LLM score responses shared by eval-only ablations.")
    parser.add_argument("--failure_constraint_with_prompt_memory", action="store_true", default=False,
                        help="Hybrid AF mode: inject graph memory facts into the prompt and apply typed cross-user constraints afterward.")
    parser.add_argument("--skip_user_clusters", action="store_true", default=False,
                        help="Skip legacy user clusters when the selected retrieval mode does not use them.")
    parser.add_argument("--phase", type=str, default="all", choices=["all", "train_only", "eval_only"])
    parser.add_argument("--eval_split", type=str, default="test", choices=["val", "test"],
                        help="Evaluation split. Use val for selection/tuning and test once after freezing settings.")
    parser.add_argument("--memory_file", type=str, default=None, help="Optional explicit memory artifact path for train/eval reuse.")
    parser.add_argument("--artifact_root", type=str, default=None, help="Optional artifact root for failure-graph memory files.")
    parser.add_argument("--user_shard_id", type=int, default=0)
    parser.add_argument("--num_user_shards", type=int, default=1)
    parser.add_argument("--run_name_suffix", type=str, default="")
    parser.add_argument("--trace_dir", type=str, default=None)
    parser.add_argument("--disable_trace", action="store_false", dest="trace_enabled")
    parser.set_defaults(trace_enabled=True)
    args = parser.parse_args()

    if args.k_memories is not None:
        args.graph_memory_k = args.k_memories

    retrieval_mode = str(args.memory_retrieval_mode).strip().lower()
    if retrieval_mode not in {"graph", "graph_only", "fail_graph"}:
        raise ValueError(
            "MEMCF v2 only supports graph retrieval. "
            f"Received --memory_retrieval_mode={args.memory_retrieval_mode}."
        )
    args.memory_retrieval_mode = "graph"

    if args.ranking_prompt_style == "memrec_old":
        # Historical 100-strong runs used the legacy name in configs, but the
        # actual prompt shape was the compact candidate-alias scorer.
        args.ranking_prompt_style = "compact_score"

    if args.failure_constraint_tie_epsilon < 0:
        raise ValueError("--failure_constraint_tie_epsilon must be >= 0")
    if args.failure_constraint_min_cross_support < 1:
        raise ValueError("--failure_constraint_min_cross_support must be >= 1")
    if args.failure_constraint_min_shared_items < 1:
        raise ValueError("--failure_constraint_min_shared_items must be >= 1")
    if args.cf_source_budget < 1:
        raise ValueError("--cf_source_budget must be >= 1")
    if args.failure_constraint_max_cross_corrections < 1:
        raise ValueError("--failure_constraint_max_cross_corrections must be >= 1")
    if args.safe_residual_pool_size < 1:
        raise ValueError("--safe_residual_pool_size must be >= 1")
    if args.safe_residual_min_cross_users < 1:
        raise ValueError("--safe_residual_min_cross_users must be >= 1")
    if args.safe_residual_max_cross_facts < 0:
        raise ValueError("--safe_residual_max_cross_facts must be >= 0")
    if args.safe_residual_max_same_facts < 0:
        raise ValueError("--safe_residual_max_same_facts must be >= 0")
    if args.safe_residual_semantic_min_terms < 1:
        raise ValueError("--safe_residual_semantic_min_terms must be >= 1")
    if args.safe_residual_min_vote_margin < 0:
        raise ValueError("--safe_residual_min_vote_margin must be >= 0")
    if args.temporal_same_k < 0 or args.temporal_cross_k < 0:
        raise ValueError("--temporal_same_k and --temporal_cross_k must be >= 0")
    if not 0.0 <= args.safe_residual_verify_min_confidence <= 1.0:
        raise ValueError("--safe_residual_verify_min_confidence must be in [0, 1]")

    return args


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


def main_v2():
    run_started_at = datetime.now()
    run_start_time = time.time()
    args = parse_args_v2()
    random.seed(2020)
    np.random.seed(2020)

    data_name = args.data_name
    use_memory = args.use_memory
    profile_only = args.profile_only
    need_user_profiles = use_memory or profile_only
    retrieve_memory_for_eval = use_memory and not profile_only
    number_of_users = args.number_of_users
    max_positive_interactions = args.max_positive_interactions
    max_negative_candidates = args.max_negative_candidates
    candidate_negative_mode = args.candidate_negative_mode
    training_negative_source = args.training_negative_source
    failure_memory_protocol = args.failure_memory_protocol
    if failure_memory_protocol == "temporal_factual":
        training_negative_source = "train_catalog"
    min_lesson_confidence = args.min_lesson_confidence
    max_lesson_risk = args.max_lesson_risk
    max_failure_lessons_per_user = args.max_failure_lessons_per_user
    ranking_prompt_style = args.ranking_prompt_style

    base_dir = os.getenv(
        "MEMCF_ROOT",
        os.getenv("AGENTICREC_CFMEMORY_ROOT", os.path.dirname(os.path.abspath(__file__))),
    )
    data_root = os.getenv(
        "MEMCF_DATA_ROOT",
        os.getenv("AGENTICREC_DATA_ROOT", os.path.join(base_dir, "data")),
    )
    eval_root = os.getenv(
        "MEMCF_EVAL_ROOT",
        os.getenv("AGENTICREC_EVAL_ROOT", os.path.join(base_dir, "evaluation_results")),
    )
    memory_root = os.getenv(
        "MEMCF_MEMORY_ROOT",
        os.getenv("AGENTICREC_MEMORY_ROOT", os.path.join(base_dir, "agent_memory")),
    )
    data_dir = os.path.join(data_root, data_name)
    eval_dir = os.path.join(eval_root, data_name)
    memory_dir = os.path.join(memory_root, data_name)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(memory_dir, exist_ok=True)

    items_path = os.path.join(data_dir, "items.json")
    sequences_path = os.path.join(data_dir, "user_sequences_10.json")
    negatives_path = os.path.join(data_dir, "user_negatives_10.json")
    items_meta, user_sequences, user_negatives = load_data(items_path, sequences_path, negatives_path)
    all_selected_user_ids = list(user_sequences.keys())[:number_of_users]
    user_ids = stable_shard_filter(all_selected_user_ids, args.user_shard_id, args.num_user_shards)
    print(f"Total users loaded: {len(user_sequences)}")
    print(f"MEMCF selected users before shard: {len(all_selected_user_ids)}")
    print(f"MEMCF active users after shard {args.user_shard_id}/{args.num_user_shards}: {len(user_ids)}")

    prompt_tag = slugify(ranking_prompt_style)
    negative_tag = slugify(candidate_negative_mode)
    training_negative_tag = (
        "" if training_negative_source == "legacy_runtime"
        else f"_trneg{slugify(training_negative_source)}"
    )
    protocol_tag = f"_fm{slugify(failure_memory_protocol)}"
    quality_tag = (
        f"conf{float_tag(min_lesson_confidence)}"
        f"_risk{float_tag(max_lesson_risk)}"
        f"_maxless{max_failure_lessons_per_user}"
    )
    no_harm_tag = "noharm1" if args.no_harm_arbitration else "noharm0"
    scope_tag = slugify(args.graph_retrieval_scope)
    cluster_tag = f"_cmem{hashlib.sha256(args.cluster_memory_file.encode('utf-8')).hexdigest()[:8]}" if args.cluster_memory_file else ""
    pack_tag = f"mf{args.max_memory_facts}_mw{args.max_memory_fact_words}_tb{args.memory_token_budget}"
    strict_tag = "strictcand1" if args.strict_memory_applicability else "strictcand0"
    if args.strict_memory_applicability:
        strict_tag += f"_cm{args.min_candidate_matches}"
        if args.require_same_user_candidate_match:
            strict_tag += "_sucand1"
    if args.allow_random_memory_injection:
        strict_tag += "_randinj1"
    if args.reject_wrong_only_memory:
        strict_tag += "_rejwrong1"
    if args.disable_user_profile_in_eval_prompt:
        strict_tag += "_noprofile1"
    selector_tag = ""
    if args.memory_selector != "none":
        selector_tag = (
            f"_sel{slugify(args.memory_selector)}"
            f"_sm{args.memory_selector_top_m}"
            f"_sk{args.memory_selector_top_k}"
            f"_sr{float_tag(args.memory_selector_min_relevance)}"
        )
        if args.memory_selector_neutral_cross:
            selector_tag += "_nc1"
    elif args.graph_retrieval_scope.startswith("dense_lgcn_fmrec_pool"):
        # pool scope with no selector still widens retrieval; record it
        selector_tag = f"_selnone_sm{args.memory_selector_top_m}"
    safe_residual_tag = ""
    if args.graph_retrieval_scope in {"safe_residual", "directional_residual"}:
        safe_residual_tag = (
            f"_srpool{args.safe_residual_pool_size}"
            f"_sru{args.safe_residual_min_cross_users}"
            f"_src{args.safe_residual_max_cross_facts}"
            f"_srs{args.safe_residual_max_same_facts}"
        )
        if args.safe_residual_semantic_consensus:
            safe_residual_tag += (
                f"_sem1_st{args.safe_residual_semantic_min_terms}"
                f"_vm{args.safe_residual_min_vote_margin}"
            )
        if args.safe_residual_verify_cross:
            safe_residual_tag += f"_verify1_vc{float_tag(args.safe_residual_verify_min_confidence)}"
    failure_constraint_tag = ""
    if args.failure_constraint_mode != "none":
        mode_tags = {
            "same_exact": "d1same",
            "cross_exact": "d2cross",
            "full_partitioned": "d3full",
            "full_consensus": "d4cons",
            "polarity_swapped": "d5swap",
            "popularity": "d6pop",
            "shuffled_provenance": "d7shuf",
            "cf_shared_cross": "f2shared",
            "cf_same_plus_shared": "f3full",
            "cf_shuffled_neighbors": "f4shuf",
            "cf_random_neighbors": "f5rand",
            "cf_polarity_swapped": "f6swap",
            "g_true_neighbor": "g1true",
            "g_shuffled_graph": "g2shufgraph",
            "g_random_neighbor": "g3random",
            "g_matched_random": "g4matched",
        }
        failure_constraint_tag = (
            f"_{mode_tags[args.failure_constraint_mode]}"
            f"_te{float_tag(args.failure_constraint_tie_epsilon)}"
            f"_cs{args.failure_constraint_min_cross_support}"
            f"_si{args.failure_constraint_min_shared_items}"
            f"_mc{args.failure_constraint_max_cross_corrections}"
        )
        if args.failure_constraint_mode.startswith("g_"):
            failure_constraint_tag += f"_sb{args.cf_source_budget}_seed{args.cf_control_seed}"
    shard_tag = f"shard{args.user_shard_id}of{args.num_user_shards}" if args.num_user_shards > 1 else "fullusers"
    suffix_tag = f"_{slugify(args.run_name_suffix)}" if args.run_name_suffix else ""
    run_name = (
        f"memcf_graph_nuser{number_of_users}_{shard_tag}_iter{args.max_iterations}"
        f"_scope{scope_tag}_gk{args.graph_memory_k}_nk{args.neighbor_k}_ev{args.min_evidence_terms}"
        f"_{no_harm_tag}_neg{negative_tag}{training_negative_tag}_prompt{prompt_tag}_{quality_tag}_{pack_tag}_{strict_tag}{cluster_tag}"
        f"{protocol_tag}{selector_tag}{safe_residual_tag}{failure_constraint_tag}{suffix_tag}"
    )
    if not use_memory:
        run_name = f"memcf_nomemory_nuser{number_of_users}_{shard_tag}_neg{negative_tag}{training_negative_tag}_prompt{prompt_tag}{suffix_tag}"
    if args.profile_only:
        run_name = f"memcf_profileonly_nuser{number_of_users}_{shard_tag}_neg{negative_tag}{training_negative_tag}_prompt{prompt_tag}{suffix_tag}"

    trace_component = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if len(trace_component) > 220:
        run_digest = hashlib.sha256(run_name.encode("utf-8")).hexdigest()[:10]
        trace_component = f"{run_name[:180]}_{run_digest}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trace_dir = args.trace_dir or os.path.join(eval_dir, "traces", trace_component)
    trace_recorder = TraceRecorder(trace_dir, enabled=args.trace_enabled)
    if args.trace_enabled:
        print(f"✓ Trace enabled: {trace_dir}")

    memory_system = RecommendationMemorySystem(use_gemini_embeddings=True)
    memory_system.trace_recorder = trace_recorder
    item_states = init_pairwise_item_states(items_meta)
    user_states: Dict[str, PairwiseUserState] = {}
    graph = MemoryGraphIndex(user_sequences, build_clusters=not args.skip_user_clusters)
    user_profiles: Dict[str, UserMemoryProfile] = {}
    artifact_root = args.artifact_root or memory_dir
    os.makedirs(artifact_root, exist_ok=True)
    memory_file_path = args.memory_file or os.path.join(artifact_root, f"{run_name}.memory.json")

    if need_user_profiles:
        should_load_memory = (args.LOAD_SAVED_MEMORY or args.phase == "eval_only") and os.path.exists(memory_file_path)
        if should_load_memory:
            print(f"Loading MEMCF graph memory from {memory_file_path}")
            graph, user_profiles = load_v2_memory(
                memory_file_path,
                user_sequences,
                build_clusters=not args.skip_user_clusters,
            )
            if not args.skip_user_clusters:
                graph.rebuild_clusters_from_memory_users()
            print(f"MEMCF memory-source cluster stats: {graph.stats()}")
            for uid, profile in user_profiles.items():
                user_states[uid] = PairwiseUserState(user_id=uid, short_term_memory=profile.profile)
        elif args.phase == "eval_only":
            raise FileNotFoundError(
                f"MEMCF eval_only requires an existing memory file: {memory_file_path}"
            )
        else:
            print("\n" + "=" * 80)
            print("PHASE 0: INITIALIZE USER MEMORY FROM HISTORY")
            print("=" * 80)
            for user_id in tqdm(user_ids, desc="Init user memories"):
                profile = initialize_user_memory_from_history_v2(
                    memory_system=memory_system,
                    user_id=str(user_id),
                    user_data=user_sequences[user_id],
                    items_meta=items_meta,
                    max_positive_interactions=max_positive_interactions,
                )
                user_profiles[str(user_id)] = profile
                user_states[str(user_id)] = PairwiseUserState(
                    user_id=str(user_id),
                    short_term_memory=profile.profile,
                )
                memory_system._trace("user_memory_initialized", {
                    "user_id": user_id,
                    "profile": asdict(profile),
                })

            if retrieve_memory_for_eval:
                print("\n" + "=" * 80)
                print("PHASE 1: PAIRWISE FAILURE TRAINING -> GRAPH LESSONS")
                print("=" * 80)
                total_lessons = 0
                for user_id in tqdm(user_ids, desc="Graph failure training"):
                    print(f"\nProcessing user {user_id}")
                    if failure_memory_protocol == "temporal_factual":
                        lessons = train_temporal_factual_memory_v2(
                            user_id=str(user_id),
                            user_data=user_sequences[user_id],
                            negative_data=None,
                            items_meta=items_meta,
                            memory_system=memory_system,
                            item_states=item_states,
                            graph=graph,
                            max_iterations=args.max_iterations,
                            max_positive_interactions=max_positive_interactions,
                            candidate_negative_mode=candidate_negative_mode,
                            max_failure_lessons_per_user=max_failure_lessons_per_user,
                        )
                    else:
                        lessons = train_memory_graph_from_fail_interactions_v2(
                            user_id=str(user_id),
                            user_data=user_sequences[user_id],
                            negative_data=user_negatives.get(user_id, {}),
                            items_meta=items_meta,
                            memory_system=memory_system,
                            user_states=user_states,
                            item_states=item_states,
                            graph=graph,
                            max_iterations=args.max_iterations,
                            max_positive_interactions=max_positive_interactions,
                            candidate_negative_mode=candidate_negative_mode,
                            min_lesson_confidence=min_lesson_confidence,
                            max_lesson_risk=max_lesson_risk,
                            max_failure_lessons_per_user=max_failure_lessons_per_user,
                            training_negative_source=training_negative_source,
                        )
                    total_lessons += len(lessons)
                    print(f"  → Generated {len(lessons)} graph failure lessons")
                print(f"Total MEMCF graph lessons: {total_lessons}")
                graph.rebuild_clusters_from_memory_users()
                print(f"MEMCF memory-source cluster stats: {graph.stats()}")
            else:
                print("MEMCF profile-only run: initialized/loaded user profiles; skipping failure-memory training.")
            save_v2_memory(
                memory_file_path,
                graph,
                user_profiles,
                training_metadata={
                    "training_negative_source": training_negative_source,
                    "failure_memory_protocol": failure_memory_protocol,
                    "candidate_negative_mode": candidate_negative_mode,
                    "number_of_users": number_of_users,
                    "max_positive_interactions": max_positive_interactions,
                    "min_lesson_confidence": min_lesson_confidence,
                    "max_lesson_risk": max_lesson_risk,
                    "max_failure_lessons_per_user": max_failure_lessons_per_user,
                },
            )
            if args.phase == "train_only":
                trace_recorder.write_manifest({
                    "run_name": run_name,
                    "memory_file": memory_file_path,
                    "completed_at": datetime.now().isoformat(),
                    "phase": args.phase,
                    "profile_only": profile_only,
                    "num_graph_lessons": len(graph.lessons),
                    "memory_artifact": memory_artifact_stats(memory_file_path, graph, user_profiles),
                    "llm_usage": memory_system.get_llm_usage_summary(),
                })
                print("MEMCF train_only complete; skipping evaluation.")
                return
    else:
        print("MEMCF no-memory run: skipping user-memory init and failure-memory training.")

    # Matching controls use only coarse train-item category distributions.
    # This metadata is query-independent and never includes held-out labels.
    graph.configure_item_metadata(items_meta)
    if args.cluster_memory_file:
        if not os.path.exists(args.cluster_memory_file):
            raise FileNotFoundError(
                f"C-MEMCF cluster artifact does not exist: {args.cluster_memory_file}"
            )
        graph.load_cluster_corrective_memory(args.cluster_memory_file)
        print(
            "Loaded C-MEMCF cluster corrective memory: "
            f"{args.cluster_memory_file} ({len(graph.cluster_corrective_lessons)} lessons)"
        )
    elif args.graph_retrieval_scope.startswith("temporal_cluster_"):
        raise ValueError("temporal_cluster_* requires --cluster_memory_file")

    if args.lightgcn_embeddings_json:
        if not os.path.exists(args.lightgcn_embeddings_json):
            raise FileNotFoundError(
                f"LightGCN embeddings artifact does not exist: {args.lightgcn_embeddings_json}"
            )
        graph.load_lightgcn_embeddings(args.lightgcn_embeddings_json)
        print(
            "Loaded LightGCN embeddings: "
            f"{args.lightgcn_embeddings_json} ({len(graph.lgcn_embeddings)} users, "
            f"{len(graph.users_by_lgcn_cluster)} clusters)"
        )
    elif args.graph_retrieval_scope in {"full_lgcn", "full_lgcn_cluster"}:
        raise ValueError(f"{args.graph_retrieval_scope} requires --lightgcn_embeddings_json")

    print("\n" + "=" * 80)
    print(f"PHASE 2: {args.eval_split.upper()} SET EVALUATION")
    print("=" * 80)
    all_user_results = []
    val_metrics = defaultdict(list)
    baseline_metrics = defaultdict(list)
    def _eval_one_user_v2(uid):
        return evaluate_user_v2(
            user_id=str(uid),
            user_data=user_sequences[uid],
            negative_data=user_negatives[uid],
            items_meta=items_meta,
            memory_system=memory_system,
            graph=graph,
            user_profiles=user_profiles,
            eval_type=args.eval_split,
            use_memory=retrieve_memory_for_eval,
            graph_memory_k=args.graph_memory_k,
            neighbor_k=args.neighbor_k,
            dense_pool_per_signal=args.dense_pool_per_signal,
            consensus_top_n_users=args.consensus_top_n_users,
            consensus_min_users=args.consensus_min_users,
            fmrec_top_k_neighbors=args.fmrec_top_k_neighbors,
            min_evidence_terms=args.min_evidence_terms,
            max_positive_interactions=max_positive_interactions,
            max_negative_candidates=max_negative_candidates,
            no_harm_arbitration=args.no_harm_arbitration,
            ranking_prompt_style=ranking_prompt_style,
            graph_retrieval_scope=args.graph_retrieval_scope,
            max_memory_facts=args.max_memory_facts,
            max_memory_fact_words=args.max_memory_fact_words,
            memory_token_budget=args.memory_token_budget,
            strict_memory_applicability=args.strict_memory_applicability,
            min_candidate_matches=args.min_candidate_matches,
            allow_same_user_without_candidate_match=(not args.require_same_user_candidate_match),
            allow_random_memory_injection=args.allow_random_memory_injection,
            reject_wrong_only_memory=args.reject_wrong_only_memory,
            disable_user_profile_in_eval_prompt=args.disable_user_profile_in_eval_prompt,
            memory_selector=args.memory_selector,
            memory_selector_top_m=args.memory_selector_top_m,
            memory_selector_top_k=args.memory_selector_top_k,
            memory_selector_min_relevance=args.memory_selector_min_relevance,
            memory_selector_neutral_cross=args.memory_selector_neutral_cross,
            pairwise_cf_rerank=args.pairwise_cf_rerank,
            pairwise_cf_alpha=args.pairwise_cf_alpha,
            pairwise_cf_beta=args.pairwise_cf_beta,
            pairwise_cf_hide_memory_prompt=args.pairwise_cf_hide_memory_prompt,
            failure_constraint_mode=args.failure_constraint_mode,
            failure_constraint_tie_epsilon=args.failure_constraint_tie_epsilon,
            failure_constraint_min_cross_support=args.failure_constraint_min_cross_support,
            failure_constraint_min_context_terms=args.failure_constraint_min_context_terms,
            failure_constraint_same_budget=args.failure_constraint_same_budget,
            failure_constraint_cross_budget=args.failure_constraint_cross_budget,
            failure_constraint_min_shared_items=args.failure_constraint_min_shared_items,
            failure_constraint_max_cross_corrections=args.failure_constraint_max_cross_corrections,
            cf_source_budget=args.cf_source_budget,
            cf_control_seed=args.cf_control_seed,
            ranking_score_cache_dir=args.ranking_score_cache_dir,
            failure_constraint_with_prompt_memory=args.failure_constraint_with_prompt_memory,
            safe_residual_pool_size=args.safe_residual_pool_size,
            safe_residual_min_cross_users=args.safe_residual_min_cross_users,
            safe_residual_max_cross_facts=args.safe_residual_max_cross_facts,
            safe_residual_max_same_facts=args.safe_residual_max_same_facts,
            safe_residual_semantic_consensus=args.safe_residual_semantic_consensus,
            safe_residual_semantic_min_terms=args.safe_residual_semantic_min_terms,
            safe_residual_min_vote_margin=args.safe_residual_min_vote_margin,
            safe_residual_verify_cross=args.safe_residual_verify_cross,
            safe_residual_verify_min_confidence=args.safe_residual_verify_min_confidence,
            temporal_same_k=args.temporal_same_k,
            temporal_cross_k=args.temporal_cross_k,
            temporal_control_seed=args.temporal_control_seed,
            matched_endpoint_scope=args.matched_endpoint_scope,
        )

    def _collect_result(uid, baseline_metric, metrics, candidates, predictions, ground_truth):
        for metric_name, value in metrics.items():
            val_metrics[metric_name].append(value)
        for metric_name, value in baseline_metric.items():
            baseline_metrics[metric_name].append(value)
        all_user_results.append({
            "user_id": uid,
            "candidates": candidates,
            "predictions": predictions,
            "ground_truth": ground_truth,
            "metrics": metrics,
            "baseline_metrics": baseline_metric,
        })

    eval_user_ids = [uid for uid in user_ids if uid in user_sequences and uid in user_negatives]

    if args.eval_workers and args.eval_workers > 1:
        # Phase 2 is read-only against `memory_system`/`graph` for eval_only-style
        # runs with a precomputed --memory_file (no lessons are written during
        # evaluation), so per-user calls are safe to run concurrently. The only
        # shared-mutable state touched inside evaluate_user_v2 is the
        # `memory_system.memory_diagnostics` counters and TraceRecorder JSONL
        # appends, which are best-effort/non-atomic under threads and may
        # undercount slightly -- this does not affect the primary ndcg/recall
        # metrics below, since each user's result is computed independently and
        # only merged back on the main thread via _collect_result.
        print(
            f"Running Phase 2 evaluation with {args.eval_workers} parallel worker "
            f"threads (read-only eval_only mode)."
        )
        with ThreadPoolExecutor(max_workers=args.eval_workers) as pool:
            futures = {pool.submit(_eval_one_user_v2, uid): uid for uid in eval_user_ids}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Validation"):
                uid = futures[fut]
                baseline_metric, metrics, candidates, predictions, ground_truth = fut.result()
                _collect_result(uid, baseline_metric, metrics, candidates, predictions, ground_truth)
    else:
        for user_id in tqdm(eval_user_ids, desc="Validation"):
            baseline_metric, metrics, candidates, predictions, ground_truth = _eval_one_user_v2(user_id)
            _collect_result(user_id, baseline_metric, metrics, candidates, predictions, ground_truth)

    output_stem = run_name
    # Keep room for both `.json` and `.summary.json` on filesystems with the
    # usual 255-byte component limit. The full run name remains in the summary.
    if len(output_stem.encode("utf-8")) > 220:
        output_digest = hashlib.sha256(output_stem.encode("utf-8")).hexdigest()[:12]
        output_stem = f"{output_stem[:190]}_{output_digest}"
    output_file = os.path.join(eval_dir, f"{output_stem}.json")
    save_all_users_ranking_results(all_user_results, items_meta, output_file)

    print("\nValidation Results:")
    print("-" * 80)
    report_metrics = [
        metric
        for k in PAPER_METRIC_KS
        for metric in (f"hit@{k}", f"ndcg@{k}")
    ]
    for metric in report_metrics:
        if baseline_metrics[metric]:
            print(f"Baseline {metric:10s}: {np.mean(baseline_metrics[metric]):.4f}")
        else:
            print(f"Baseline {metric:10s}: N/A")
    print("-" * 80)
    for metric in report_metrics:
        if val_metrics[metric]:
            print(f"{metric:12s}: {np.mean(val_metrics[metric]):.4f}")
        else:
            print(f"{metric:12s}: N/A")

    diag = getattr(memory_system, "memory_diagnostics", defaultdict(float))
    summary = {
        "model": "MEMCF",
        "dataset": data_name,
        "number_of_users_requested": number_of_users,
        "number_of_users_evaluated": len(all_user_results),
        "use_memory": use_memory,
        "retrieve_memory_for_eval": retrieve_memory_for_eval,
        "load_saved_memory": args.LOAD_SAVED_MEMORY,
        "max_iterations": args.max_iterations,
        "max_positive_interactions": max_positive_interactions,
        "max_negative_candidates": max_negative_candidates,
        "candidate_negative_mode": candidate_negative_mode,
        "training_negative_source": training_negative_source,
        "failure_memory_protocol": failure_memory_protocol,
        "memory_training_metadata": dict(getattr(graph, "artifact_metadata", {})),
        "min_lesson_confidence": min_lesson_confidence,
        "max_lesson_risk": max_lesson_risk,
        "max_failure_lessons_per_user": max_failure_lessons_per_user,
        "ranking_prompt_style": ranking_prompt_style,
        "phase": args.phase,
        "eval_split": args.eval_split,
        "user_shard_id": args.user_shard_id,
        "num_user_shards": args.num_user_shards,
        "active_user_count": len(user_ids),
        "graph_retrieval_scope": args.graph_retrieval_scope,
        "temporal_same_k": args.temporal_same_k,
        "temporal_cross_k": args.temporal_cross_k,
        "temporal_control_seed": args.temporal_control_seed,
        "matched_endpoint_scope": args.matched_endpoint_scope,
        "cluster_memory_file": args.cluster_memory_file,
        "cluster_memory_metadata": dict(getattr(graph, "cluster_corrective_metadata", {})),
        "max_memory_facts": args.max_memory_facts,
        "max_memory_fact_words": args.max_memory_fact_words,
        "memory_token_budget": args.memory_token_budget,
        "strict_memory_applicability": args.strict_memory_applicability,
        "min_candidate_matches": args.min_candidate_matches,
        "require_same_user_candidate_match": args.require_same_user_candidate_match,
        "allow_random_memory_injection": args.allow_random_memory_injection,
        "reject_wrong_only_memory": args.reject_wrong_only_memory,
        "profile_only": args.profile_only,
        "disable_user_profile_in_eval_prompt": args.disable_user_profile_in_eval_prompt,
        "memory_selector": args.memory_selector,
        "memory_selector_top_m": args.memory_selector_top_m,
        "memory_selector_top_k": args.memory_selector_top_k,
        "memory_selector_min_relevance": args.memory_selector_min_relevance,
        "memory_selector_neutral_cross": args.memory_selector_neutral_cross,
        "safe_residual_pool_size": args.safe_residual_pool_size,
        "safe_residual_min_cross_users": args.safe_residual_min_cross_users,
        "safe_residual_max_cross_facts": args.safe_residual_max_cross_facts,
        "safe_residual_max_same_facts": args.safe_residual_max_same_facts,
        "safe_residual_semantic_consensus": args.safe_residual_semantic_consensus,
        "safe_residual_semantic_min_terms": args.safe_residual_semantic_min_terms,
        "safe_residual_min_vote_margin": args.safe_residual_min_vote_margin,
        "safe_residual_verify_cross": args.safe_residual_verify_cross,
        "safe_residual_verify_min_confidence": args.safe_residual_verify_min_confidence,
        "failure_constraint_mode": args.failure_constraint_mode,
        "failure_constraint_tie_epsilon": args.failure_constraint_tie_epsilon,
        "failure_constraint_min_cross_support": args.failure_constraint_min_cross_support,
        "failure_constraint_min_context_terms": args.failure_constraint_min_context_terms,
        "failure_constraint_same_budget": args.failure_constraint_same_budget,
        "failure_constraint_cross_budget": args.failure_constraint_cross_budget,
        "failure_constraint_min_shared_items": args.failure_constraint_min_shared_items,
        "failure_constraint_max_cross_corrections": args.failure_constraint_max_cross_corrections,
        "cf_source_budget": args.cf_source_budget,
        "cf_control_seed": args.cf_control_seed,
        "ranking_score_cache_dir": args.ranking_score_cache_dir,
        "failure_constraint_with_prompt_memory": args.failure_constraint_with_prompt_memory,
        "artifact_root": artifact_root,
        "graph_memory_k": args.graph_memory_k,
        "neighbor_k": args.neighbor_k,
        "min_evidence_terms": args.min_evidence_terms,
        "no_harm_arbitration": args.no_harm_arbitration,
        "trace_enabled": args.trace_enabled,
        "trace_dir": trace_dir if args.trace_enabled else None,
        "memory_file": memory_file_path if need_user_profiles else None,
        "num_graph_lessons": len(graph.lessons),
        "num_user_profiles": len(user_profiles),
        "memory_artifact": memory_artifact_stats(memory_file_path if need_user_profiles else None, graph, user_profiles),
        "runtime": {
            "started_at": run_started_at.isoformat(),
            "completed_at": datetime.now().isoformat(),
            "total_seconds": time.time() - run_start_time,
            "seconds_per_evaluated_user": (
                (time.time() - run_start_time) / len(all_user_results)
                if all_user_results else None
            ),
        },
        "llm_usage": memory_system.get_llm_usage_summary(),
        "baseline_metrics": {
            metric: (float(np.mean(baseline_metrics[metric])) if baseline_metrics[metric] else None)
            for metric in PAPER_METRIC_NAMES
        },
        "metrics": {
            metric: (float(np.mean(val_metrics[metric])) if val_metrics[metric] else None)
            for metric in PAPER_METRIC_NAMES
        },
        "memory_diagnostics": {
            "eval_users_with_memory_retrieval": int(diag.get("eval_users", 0.0)),
            "retrieved_total": int(diag.get("retrieved_total", 0.0)),
            "kept_total": int(diag.get("kept_total", 0.0)),
            "users_with_kept_memory": int(diag.get("users_with_kept_memory", 0.0)),
            "avg_retrieved_memories": (
                float(diag.get("retrieved_total", 0.0)) / float(diag.get("eval_users", 0.0))
                if float(diag.get("eval_users", 0.0)) else 0.0
            ),
            "rank_score_calls": int(diag.get("rank_score_calls", 0.0)),
            "rank_valid_score_outputs": int(diag.get("rank_valid_score_outputs", 0.0)),
            "rank_invalid_score_outputs": int(diag.get("rank_invalid_score_outputs", 0.0)),
            "rank_fallbacks": int(diag.get("rank_fallbacks", 0.0)),
            "no_harm_users": int(diag.get("no_harm_users", 0.0)),
            "no_harm_used_memory": int(diag.get("no_harm_used_memory", 0.0)),
            "no_harm_fallback_no_memory": int(diag.get("no_harm_fallback_no_memory", 0.0)),
            "no_harm_memory_use_rate": (
                float(diag.get("no_harm_used_memory", 0.0)) / float(diag.get("no_harm_users", 0.0))
                if float(diag.get("no_harm_users", 0.0)) else 0.0
            ),
            "selected_memory_facts_total": int(diag.get("selected_memory_facts_total", 0.0)),
            "rejected_memory_facts_total": int(diag.get("rejected_memory_facts_total", 0.0)),
            "selected_memory_facts_noisy": int(diag.get("selected_memory_facts_noisy", 0.0)),
            "selected_wrong_only_memory_facts": int(diag.get("selected_wrong_only_memory_facts", 0.0)),
            "rejected_wrong_only_memory_facts": int(diag.get("rejected_wrong_only_memory_facts", 0.0)),
            "selected_memory_facts_noise_rate": (
                float(diag.get("selected_memory_facts_noisy", 0.0)) / float(diag.get("selected_memory_facts_total", 0.0))
                if float(diag.get("selected_memory_facts_total", 0.0)) else 0.0
            ),
            "selected_source_same_user": int(diag.get("selected_source_same_user", 0.0)),
            "selected_source_candidate_item": int(diag.get("selected_source_candidate_item", 0.0)),
            "selected_source_history_item": int(diag.get("selected_source_history_item", 0.0)),
            "selected_source_neighbor_user": int(diag.get("selected_source_neighbor_user", 0.0)),
            "selected_source_cluster_user": int(diag.get("selected_source_cluster_user", 0.0)),
            "selected_source_random_memory": int(diag.get("selected_source_random_memory", 0.0)),
            "selected_source_random_memory_clean": int(diag.get("selected_source_random_memory_clean", 0.0)),
            "selected_source_random_cluster": int(diag.get("selected_source_random_cluster", 0.0)),
            "selected_source_shuffled_memory": int(diag.get("selected_source_shuffled_memory", 0.0)),
            "selected_source_shuffled_memory_clean": int(diag.get("selected_source_shuffled_memory_clean", 0.0)),
            "selected_source_shuffled_cluster": int(diag.get("selected_source_shuffled_cluster", 0.0)),
            "memory_selector_calls": int(diag.get("memory_selector_calls", 0.0)),
            "memory_selector_errors": int(diag.get("memory_selector_errors", 0.0)),
            "memory_selector_fallbacks": int(diag.get("memory_selector_fallbacks", 0.0)),
            "memory_selector_selected": int(diag.get("memory_selector_selected", 0.0)),
            "memory_selector_rejected": int(diag.get("memory_selector_rejected", 0.0)),
            "memory_selector_selected_source_same_user": int(diag.get("memory_selector_selected_source_same_user", 0.0)),
            "memory_selector_selected_source_candidate_item": int(diag.get("memory_selector_selected_source_candidate_item", 0.0)),
            "memory_selector_selected_source_history_item": int(diag.get("memory_selector_selected_source_history_item", 0.0)),
            "memory_selector_selected_source_neighbor_user": int(diag.get("memory_selector_selected_source_neighbor_user", 0.0)),
            "memory_selector_selected_source_cluster_user": int(diag.get("memory_selector_selected_source_cluster_user", 0.0)),
            "safe_residual_users": int(diag.get("safe_residual_users", 0.0)),
            "safe_residual_users_with_cross": int(diag.get("safe_residual_users_with_cross", 0.0)),
            "safe_residual_cross_use_rate": (
                float(diag.get("safe_residual_users_with_cross", 0.0))
                / float(diag.get("safe_residual_users", 0.0))
                if float(diag.get("safe_residual_users", 0.0)) else 0.0
            ),
            "safe_residual_same_facts": int(diag.get("safe_residual_same_facts", 0.0)),
            "safe_residual_cross_facts": int(diag.get("safe_residual_cross_facts", 0.0)),
            "safe_residual_cross_source_users": int(diag.get("safe_residual_cross_source_users", 0.0)),
            "safe_residual_verifier_calls": int(diag.get("safe_residual_verifier_calls", 0.0)),
            "safe_residual_verifier_accepted": int(diag.get("safe_residual_verifier_accepted", 0.0)),
            "safe_residual_verifier_rejected": int(diag.get("safe_residual_verifier_rejected", 0.0)),
            "safe_residual_verifier_errors": int(diag.get("safe_residual_verifier_errors", 0.0)),
            "temporal_memory_users": int(diag.get("temporal_memory_users", 0.0)),
            "temporal_memory_users_with_cross": int(diag.get("temporal_memory_users_with_cross", 0.0)),
            "temporal_cross_use_rate": (
                float(diag.get("temporal_memory_users_with_cross", 0.0))
                / float(diag.get("temporal_memory_users", 0.0))
                if float(diag.get("temporal_memory_users", 0.0)) else 0.0
            ),
            "temporal_exposure_exact_replay": int(diag.get("temporal_exposure_exact_replay", 0.0)),
            "temporal_exposure_candidate_linked_transfer": int(diag.get("temporal_exposure_candidate_linked_transfer", 0.0)),
            "temporal_exposure_abstract_transfer": int(diag.get("temporal_exposure_abstract_transfer", 0.0)),
            "temporal_control_true": int(diag.get("temporal_control_true", 0.0)),
            "temporal_control_degree_preserving_shuffled": int(diag.get("temporal_control_degree_preserving_shuffled", 0.0)),
            "temporal_control_matched_random": int(diag.get("temporal_control_matched_random", 0.0)),
            "temporal_j_queries": int(diag.get("temporal_j_queries", 0.0)),
            "temporal_j_matched_eligible": int(diag.get("temporal_j_matched_eligible", 0.0)),
            "temporal_j_matched_eligible_rate": (
                float(diag.get("temporal_j_matched_eligible", 0.0))
                / float(diag.get("temporal_j_queries", 0.0))
                if float(diag.get("temporal_j_queries", 0.0)) else 0.0
            ),
            "temporal_j_equal_post_gate_budget": int(
                diag.get("temporal_j_equal_post_gate_budget", 0.0)
            ),
            "temporal_j_equal_post_gate_budget_rate": (
                float(diag.get("temporal_j_equal_post_gate_budget", 0.0))
                / float(diag.get("temporal_j_queries", 0.0))
                if float(diag.get("temporal_j_queries", 0.0)) else 0.0
            ),
            "temporal_j_selected_cross_facts": int(
                diag.get("temporal_j_selected_cross_facts", 0.0)
            ),
            "temporal_j_exact_replay_filtered": int(
                diag.get("temporal_j_exact_replay_filtered", 0.0)
            ),
            "temporal_j_retrieval_latency_ms_total": float(
                diag.get("temporal_j_retrieval_latency_ms", 0.0)
            ),
            "temporal_j_retrieval_latency_ms_mean": (
                float(diag.get("temporal_j_retrieval_latency_ms", 0.0))
                / float(diag.get("temporal_j_queries", 0.0))
                if float(diag.get("temporal_j_queries", 0.0)) else 0.0
            ),
            "failure_constraint_users": int(diag.get("failure_constraint_users", 0.0)),
            "failure_constraint_changed_users": int(diag.get("failure_constraint_changed_users", 0.0)),
            "failure_constraint_evidence": int(diag.get("failure_constraint_evidence", 0.0)),
            "failure_constraint_moved_candidates": int(diag.get("failure_constraint_moved_candidates", 0.0)),
            "rejected_source_same_user": int(diag.get("rejected_source_same_user", 0.0)),
            "rejected_source_candidate_item": int(diag.get("rejected_source_candidate_item", 0.0)),
            "rejected_source_history_item": int(diag.get("rejected_source_history_item", 0.0)),
            "rejected_source_neighbor_user": int(diag.get("rejected_source_neighbor_user", 0.0)),
            "rejected_source_cluster_user": int(diag.get("rejected_source_cluster_user", 0.0)),
            "rejected_source_random_memory": int(diag.get("rejected_source_random_memory", 0.0)),
            "rejected_source_random_memory_clean": int(diag.get("rejected_source_random_memory_clean", 0.0)),
            "rejected_source_random_cluster": int(diag.get("rejected_source_random_cluster", 0.0)),
            "rejected_source_shuffled_memory": int(diag.get("rejected_source_shuffled_memory", 0.0)),
            "rejected_source_shuffled_memory_clean": int(diag.get("rejected_source_shuffled_memory_clean", 0.0)),
            "rejected_source_shuffled_cluster": int(diag.get("rejected_source_shuffled_cluster", 0.0)),
        },
    }
    summary_file = output_file.replace(".json", ".summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(make_jsonable(summary), f, ensure_ascii=False, indent=2)
    print(f"✓ Saved MEMCF summary to {summary_file}")
    trace_recorder.write_manifest({
        "run_name": run_name,
        "output_file": output_file,
        "summary_file": summary_file,
        "memory_file": memory_file_path if need_user_profiles else None,
        "completed_at": datetime.now().isoformat(),
        "summary": summary,
    })


if __name__ == "__main__":
    main_v2()
