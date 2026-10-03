"""Ranking prompt styles other than compact_score (safe-residual, stage-R, curated, routers)."""

import json
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import Counter
import re

from fmrec.common import shorten_words
from fmrec.prompts import _format_compact_candidate_lines, _format_compact_history_lines


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
