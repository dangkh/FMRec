"""LLM ranking step (llm_ranking_v2)."""

import os
import json
from typing import List, Dict, Optional, Any
from dataclasses import asdict
import hashlib

from fmrec.common import extract_json_object
from fmrec.llm_client import RecommendationMemorySystem
from fmrec.prompts import (
    add_candidate_aliases,
    build_compact_score_prompt,
    parse_score_entries_from_text,
    score_entries_to_ranking,
)
from fmrec.records import UserMemoryProfile
from fmrec.variants.failure_constraints import apply_typed_failure_constraints
from fmrec.variants.pairwise_cf import (
    apply_pairwise_cf_score_adjustments,
    build_pairwise_cf_corrections,
)
from fmrec.variants.prompt_styles import (
    _compact_candidate_rows_for_router,
    build_compact_curated_score_prompt,
    build_compact_safe_residual_score_prompt,
    build_compact_stage_r_prompt,
    build_deterministic_user_anchor,
    build_failure_evidence_router,
    build_memory_candidate_evidence,
)


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
