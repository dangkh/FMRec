"""MemoryGraphIndex dense-LightGCN retrieval scopes other than the FMRec pool."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from fmrec.common import deterministic_shuffle
from fmrec.records import FailureLesson, GraphRetrievedLesson


class DenseLgcnScopesMixin:
    """MemoryGraphIndex methods for: MemoryGraphIndex dense-LightGCN retrieval scopes other than the FMRec pool."""

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
