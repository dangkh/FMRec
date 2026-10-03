"""Per-user evaluation (evaluate_user_v2)."""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import asdict

from fmrec.common import deterministic_shuffle
from fmrec.data_metrics import calculate_paper_ranking_metrics
from fmrec.graph_index import MemoryGraphIndex
from fmrec.llm_client import RecommendationMemorySystem
from fmrec.prompts import _context_text_from_items, _history_item_infos, _item_info_for_prompt
from fmrec.ranking import llm_ranking_v2
from fmrec.records import GraphRetrievedLesson, UserMemoryProfile
from fmrec.selector import read_graph_lessons_as_facets_v2
from fmrec.variants.temporal_reading import read_temporal_lexicographic_memory_v2


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
