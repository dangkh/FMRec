"""MemoryGraphIndex temporal / typed-evidence retrieval scopes."""

import json
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import defaultdict
import time
import hashlib

from fmrec.common import (
    _safe_float,
    deterministic_shuffle,
    normalize_evidence_terms,
    term_matches_context,
)
from fmrec.records import FailureLesson, GraphRetrievedLesson


class TemporalScopesMixin:
    """MemoryGraphIndex methods for: MemoryGraphIndex temporal / typed-evidence retrieval scopes."""

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
