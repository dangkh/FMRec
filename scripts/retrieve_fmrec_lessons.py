#!/usr/bin/env python3
"""
Retrieve FMRec failure lessons with pretrained LightGCN user embeddings.

For each evaluation user u:
    1) keep at most ONE own failure lesson, if available;
    2) find top-K similar OTHER users by cosine similarity;
    3) keep at most ONE lesson from each similar user.

Default:
    1 personal lesson + 3 collaborative lessons = at most 4 lessons/user.

Inputs:
    --lightgcn_embeddings
        JSON from train_lightgcn.py:
        {
          "users": {"0": [...], ...},
          "items": {...},
          "meta": {...}
        }

    --failure_memory
        JSONL from build_fmrec_lessons.py.
        Each row contains at least:
          source_user_id, lesson, confidence, memory_id

    --eval_user_list
        Same evaluation user list used by llmrank.py.

Outputs:
    retrieved_lessons_by_user.jsonl
        Directly consumable by llmrank.py --lessons_file
        Example:
        {
          "user_id": 12,
          "memory_facts": [
            "Personal lesson: ...",
            "Collaborative lesson: ...",
            ...
          ]
        }

    retrieval_audit.jsonl
        Full source/similarity/memory trace.

    summary.json
        Retrieval coverage statistics.

No LLM call is made here.
No LightGCN training is performed here.
No test interaction is used here.
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_eval_users(path: str, max_users: Optional[int] = None) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    ids = obj.get("user_ids") if isinstance(obj, dict) else obj
    if ids is None:
        raise ValueError(f"No user_ids found in {path}")

    users = [int(x) for x in ids]

    if max_users is not None and max_users > 0:
        users = users[:max_users]

    return users


def load_user_embeddings(path: str) -> Dict[int, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_users = payload.get("users", {})
    if not raw_users:
        raise ValueError(f"No user embeddings found in {path}")

    out: Dict[int, np.ndarray] = {}

    for uid, vec in raw_users.items():
        arr = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))

        if norm <= 1e-12:
            continue

        # Normalize again for robustness, even if train_lightgcn.py already did.
        out[int(uid)] = arr / norm

    print(f"✓ Loaded LightGCN embeddings for {len(out)} users")
    return out


def load_failure_memory(path: str) -> Dict[int, List[Dict[str, Any]]]:
    by_user: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            source_uid = int(row["source_user_id"])
            lesson = str(row.get("lesson", "") or "").strip()

            # Retrieval uses only generalized lessons.
            if not lesson or lesson.upper() == "NONE":
                continue

            row["source_user_id"] = source_uid
            row["lesson"] = lesson
            row["confidence"] = float(row.get("confidence", 0.0) or 0.0)

            by_user[source_uid].append(row)

    # Deterministic ordering: strongest lesson first.
    for uid in by_user:
        by_user[uid].sort(
            key=lambda x: (
                -float(x.get("confidence", 0.0)),
                str(x.get("memory_id", "")),
            )
        )

    n_lessons = sum(len(v) for v in by_user.values())
    print(
        f"✓ Loaded {n_lessons} usable lessons "
        f"from {len(by_user)} source users"
    )

    return dict(by_user)


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Lesson selection
# ---------------------------------------------------------------------------

def best_lesson_for_user(
    source_uid: int,
    lessons_by_user: Dict[int, List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    rows = lessons_by_user.get(source_uid, [])
    return rows[0] if rows else None


def retrieve_neighbor_users(
    query_uid: int,
    embeddings: Dict[int, np.ndarray],
    lessons_by_user: Dict[int, List[Dict[str, Any]]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-K OTHER users by LightGCN cosine similarity.

    Only users with at least one usable generalized lesson are eligible,
    because users without memory cannot contribute to FMRec retrieval.
    """
    query_vec = embeddings.get(query_uid)
    if query_vec is None:
        return []

    candidates = []

    for source_uid, source_lessons in lessons_by_user.items():
        if source_uid == query_uid or not source_lessons:
            continue

        source_vec = embeddings.get(source_uid)
        if source_vec is None:
            continue

        cosine = float(np.dot(query_vec, source_vec))

        candidates.append({
            "source_user_id": source_uid,
            "similarity": cosine,
        })

    candidates.sort(
        key=lambda x: (
            -x["similarity"],
            x["source_user_id"],
        )
    )

    return candidates[:top_k]


def retrieve_for_user(
    uid: int,
    embeddings: Dict[int, np.ndarray],
    lessons_by_user: Dict[int, List[Dict[str, Any]]],
    top_k_neighbors: int,
    include_self: bool,
    label_sources: bool,
) -> Dict[str, Any]:
    selected: List[Dict[str, Any]] = []

    # 1) Personal anchor: never let collaborative retrieval replace it.
    if include_self:
        own = best_lesson_for_user(uid, lessons_by_user)

        if own is not None:
            selected.append({
                "source_type": "personal",
                "source_user_id": uid,
                "similarity": 1.0,
                "memory_id": own.get("memory_id", ""),
                "confidence": float(own.get("confidence", 0.0)),
                "lesson": own["lesson"],
            })

    # 2) One lesson from each of top-K collaborative neighbors.
    neighbors = retrieve_neighbor_users(
        query_uid=uid,
        embeddings=embeddings,
        lessons_by_user=lessons_by_user,
        top_k=top_k_neighbors,
    )

    for neighbor in neighbors:
        source_uid = int(neighbor["source_user_id"])
        memory = best_lesson_for_user(source_uid, lessons_by_user)

        if memory is None:
            continue

        selected.append({
            "source_type": "collaborative",
            "source_user_id": source_uid,
            "similarity": float(neighbor["similarity"]),
            "memory_id": memory.get("memory_id", ""),
            "confidence": float(memory.get("confidence", 0.0)),
            "lesson": memory["lesson"],
        })

    memory_facts = []

    for row in selected:
        lesson = row["lesson"]

        if label_sources:
            if row["source_type"] == "personal":
                lesson = f"Personal failure lesson: {lesson}"
            else:
                lesson = f"Collaborative failure lesson: {lesson}"

        memory_facts.append(lesson)

    return {
        "user_id": uid,
        "memory_facts": memory_facts,
        "selected": selected,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    users = load_eval_users(args.eval_user_list, args.max_users)
    embeddings = load_user_embeddings(args.lightgcn_embeddings)
    lessons_by_user = load_failure_memory(args.failure_memory)

    os.makedirs(args.output_dir, exist_ok=True)

    lessons_path = os.path.join(
        args.output_dir,
        "retrieved_lessons_by_user.jsonl",
    )
    audit_path = os.path.join(
        args.output_dir,
        "retrieval_audit.jsonl",
    )
    summary_path = os.path.join(
        args.output_dir,
        "summary.json",
    )

    for path in (lessons_path, audit_path):
        if os.path.exists(path):
            os.remove(path)

    n_with_self = 0
    n_with_collab = 0
    n_with_any = 0
    total_collab = 0
    total_lessons = 0
    missing_embedding = 0

    pbar = tqdm(users, desc="Retrieve FMRec lessons")

    for uid in pbar:
        result = retrieve_for_user(
            uid=uid,
            embeddings=embeddings,
            lessons_by_user=lessons_by_user,
            top_k_neighbors=args.top_k_neighbors,
            include_self=args.include_self,
            label_sources=args.label_sources,
        )

        selected = result["selected"]
        personal = [
            x for x in selected
            if x["source_type"] == "personal"
        ]
        collaborative = [
            x for x in selected
            if x["source_type"] == "collaborative"
        ]

        if uid not in embeddings:
            missing_embedding += 1

        if personal:
            n_with_self += 1

        if collaborative:
            n_with_collab += 1

        if selected:
            n_with_any += 1

        total_collab += len(collaborative)
        total_lessons += len(selected)

        append_jsonl(
            lessons_path,
            [{
                "user_id": uid,
                "memory_facts": result["memory_facts"],
            }],
        )

        append_jsonl(
            audit_path,
            [{
                "user_id": uid,
                "n_personal": len(personal),
                "n_collaborative": len(collaborative),
                "selected_lessons": selected,
            }],
        )

        pbar.set_postfix({
            "self": n_with_self,
            "collab": n_with_collab,
        })

    n = len(users)

    summary = {
        "method": "FMRec LightGCN lesson retrieval",
        "n_eval_users": n,
        "lightgcn_embeddings": args.lightgcn_embeddings,
        "failure_memory": args.failure_memory,
        "top_k_neighbors": args.top_k_neighbors,
        "include_self": args.include_self,
        "label_sources": args.label_sources,
        "max_lessons_per_user": (
            (1 if args.include_self else 0) + args.top_k_neighbors
        ),
        "users_with_personal_lesson": n_with_self,
        "personal_coverage": n_with_self / n if n else 0.0,
        "users_with_collaborative_lesson": n_with_collab,
        "collaborative_coverage": n_with_collab / n if n else 0.0,
        "users_with_any_lesson": n_with_any,
        "memory_coverage": n_with_any / n if n else 0.0,
        "avg_collaborative_lessons": total_collab / n if n else 0.0,
        "avg_total_lessons": total_lessons / n if n else 0.0,
        "users_missing_lightgcn_embedding": missing_embedding,
        "outputs": {
            "llmrank_lessons": lessons_path,
            "retrieval_audit": audit_path,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("FMRec retrieval complete")
    print("=" * 80)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nUse directly with llmrank.py:")
    print(
        f"python scripts/llmrank.py "
        f"--lessons_file {lessons_path} "
        f"--output_dir results/llmrank_fmrec"
    )


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--lightgcn_embeddings",
        default="results/lightgcn_books/lightgcn_embeddings.json",
    )
    p.add_argument(
        "--failure_memory",
        default="results/fmrec_lessons_books_r2/failure_memory.jsonl",
    )
    p.add_argument(
        "--eval_user_list",
        default="data/eval_user_samples/eval_user_sample_10_instructrec-books.json",
    )
    p.add_argument(
        "--output_dir",
        default="results/fmrec_retrieval_books",
    )

    p.add_argument("--top_k_neighbors", type=int, default=3)
    p.add_argument(
        "--include_self",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--label_sources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Prefix facts with Personal/Collaborative so the final LLM "
            "knows which evidence is direct vs transferred."
        ),
    )
    p.add_argument("--max_users", type=int, default=None)

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
