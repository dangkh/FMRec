"""MemoryGraphIndex: lesson store + LightGCN neighbours + the FMRec pool retrieval."""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Any, Set
from dataclasses import asdict
from collections import defaultdict

from fmrec.common import (
    _safe_float,
    deterministic_shuffle,
    item_category,
    normalize_evidence_terms,
    term_matches_context,
)
from fmrec.records import FailureLesson, GraphRetrievedLesson
from fmrec.variants.graph_cluster import ClusterScopesMixin
from fmrec.variants.graph_lgcn import DenseLgcnScopesMixin
from fmrec.variants.graph_temporal import TemporalScopesMixin


class MemoryGraphIndex(ClusterScopesMixin, DenseLgcnScopesMixin, TemporalScopesMixin):
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
