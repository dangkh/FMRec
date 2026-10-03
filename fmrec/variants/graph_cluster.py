"""MemoryGraphIndex retrieval scopes based on user clusters."""

import json
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict, Counter
import random
import time
import hashlib

from fmrec.records import FailureLesson, GraphRetrievedLesson


class ClusterScopesMixin:
    """MemoryGraphIndex methods for: MemoryGraphIndex retrieval scopes based on user clusters."""

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

    def similar_users(self, user_id: str, top_k: int = 10) -> List[Tuple[str, int]]:
        user_id = str(user_id)
        counts: Dict[str, int] = defaultdict(int)
        for item_id in self.items_by_user.get(user_id, set()):
            for other_user in self.users_by_item.get(item_id, set()):
                if other_user != user_id:
                    counts[other_user] += 1
        return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top_k]
