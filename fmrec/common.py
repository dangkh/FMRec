"""Shared helpers: text normalisation, JSON repair, seeding, item accessors."""

import os
import json
import numpy as np
from typing import List, Dict, Any, Set
from dataclasses import asdict
import random
import re
import html
import hashlib


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


def deterministic_shuffle(values: List[Any], salt: str = "") -> List[Any]:
    """Shuffle reproducibly without depending on global random state consumed during training."""
    values = list(values)
    seed_material = salt + "||" + "||".join(str(v) for v in values)
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    rng.shuffle(values)
    return values


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


def term_matches_context(term: str, context: str) -> bool:
    term = re.sub(r"\s+", " ", str(term).lower()).strip()
    if not term:
        return False
    if " " in term:
        return term in context
    return re.search(rf"\b{re.escape(term)}\b", context) is not None


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


# Imported last: these names are only used inside function bodies, and importing them
# at the top would create a circular import between modules.
from fmrec.records import BehaviorMemory, PairwiseItemState, PairwiseUserState, UserInteraction
