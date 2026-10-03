"""Command-line arguments and the main experiment driver (main_v2)."""

import os
import json
import numpy as np
from datetime import datetime
from typing import Dict
from dataclasses import asdict
from collections import defaultdict
from tqdm import tqdm
import random
import time
import hashlib
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from fmrec.common import float_tag, make_jsonable, slugify, stable_shard_filter
from fmrec.data_metrics import (
    PAPER_METRIC_KS,
    PAPER_METRIC_NAMES,
    load_data,
    save_all_users_ranking_results,
)
from fmrec.evaluation import evaluate_user_v2
from fmrec.graph_index import MemoryGraphIndex
from fmrec.llm_client import RecommendationMemorySystem
from fmrec.memory_build import (
    init_pairwise_item_states,
    initialize_user_memory_from_history_v2,
    load_v2_memory,
    memory_artifact_stats,
    save_v2_memory,
    train_temporal_factual_memory_v2,
)
from fmrec.records import PairwiseUserState, TraceRecorder, UserMemoryProfile
from fmrec.variants.failure_constraints import FAILURE_CONSTRAINT_MODES
from fmrec.variants.legacy_lessons import train_memory_graph_from_fail_interactions_v2


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
        os.getenv("AGENTICREC_CFMEMORY_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
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
