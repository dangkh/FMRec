#!/usr/bin/env python3
"""
Build FMRec failure lessons from MemRec's TRAIN/VALID split.

For each user:
    train history + instruction
    + validation item (positive)
    + up to R fixed random negatives
        -> LLM pairwise decisions

The user is probed sequentially:
    probe 1 -> if failure, STOP and keep this failure
    probe 2 -> only if probe 1 was not a failure
    ...
Thus each user contributes at most ONE personal failure lesson.

If the LLM chooses a negative:
    1) store a factual failure record;
    2) ask the LLM to abstract ONE transferable/generalizable lesson.

Outputs:
    failure_memory.jsonl
    lessons_by_user.jsonl
    audit.jsonl
    summary.json

No LightGCN is trained or used here.
No test item is used to construct a lesson.
No warm-up or test-time reflection is performed.
"""

import argparse
import hashlib
import html
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.data import RecDataset
except ModuleNotFoundError as e:
    raise RuntimeError(
        "Cannot import MemRec `src` package.\n"
        "Place this file inside <MemRec>/scripts/."
    ) from e


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def user_seed(seed: int, user_id: int, salt: str) -> int:
    raw = f"{seed}::{user_id}::{salt}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2 ** 32)


def clean_text(value: Any, max_chars: int = 0) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(x) for x in value if str(x).strip())
    elif isinstance(value, dict):
        value = " ".join(str(x) for x in value.values() if str(x).strip())

    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def extract_json(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()

    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]

    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    return json.loads(text)


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_users(
    dataset: RecDataset,
    scope: str,
    eval_user_list: Optional[str],
    max_users: Optional[int],
) -> List[int]:
    if scope == "eval":
        if not eval_user_list:
            raise ValueError("--eval_user_list is required when --user_scope eval")

        with open(eval_user_list, "r", encoding="utf-8") as f:
            obj = json.load(f)

        ids = obj.get("user_ids") if isinstance(obj, dict) else obj
        if ids is None:
            raise ValueError(f"No user_ids found in {eval_user_list}")

        users = [int(x) for x in ids]
    else:
        users = list(dataset.train_data.keys())

    users = [
        u for u in users
        if dataset.train_data.get(u) and dataset.valid_data.get(u) is not None
    ]

    if max_users is not None and max_users > 0:
        users = users[:max_users]

    return users


def get_instruction(dataset: RecDataset, uid: int, use_instruction: bool) -> str:
    if not use_instruction or not dataset.instructions:
        return ""

    return clean_text(
        dataset.instructions.get(uid, {}).get("instruction", ""),
        1200,
    )


def item_info(
    dataset: RecDataset,
    item_id: int,
    description_chars: int = 200,
) -> Dict[str, Any]:
    meta = (dataset.item_metadata or {}).get(int(item_id), {})

    return {
        "item_id": int(item_id),
        "title": (
            clean_text(meta.get("title", f"Item {item_id}"), 300)
            or f"Item {item_id}"
        ),
        "description": clean_text(
            meta.get("description", ""),
            description_chars,
        ),
    }


def format_history(dataset: RecDataset, item_ids: List[int]) -> str:
    """
    Full observed TRAIN history. No arbitrary last-5/last-10 cutoff.
    """
    lines = []

    for idx, item_id in enumerate(item_ids, 1):
        info = item_info(dataset, int(item_id), description_chars=0)
        lines.append(f"{idx}. {info['title']}")

    return "\n".join(lines) if lines else "No observed history."


def sample_validation_negative(
    dataset: RecDataset,
    uid: int,
    seed: int,
    probe_index: int,
    max_failure_probes: int,
) -> int:
    """
    Return a deterministic DISTINCT negative for this probe.

    probe_index is zero-based.

    We sample an ordered list of negatives once deterministically:
        [neg_probe_1, neg_probe_2, ..., neg_probe_R]

    and return the requested position.

    Every known positive is excluded:
        train + valid + test
    """
    positives = set(int(x) for x in dataset.get_user_all_items(uid))

    if dataset.item_metadata:
        catalog = [int(x) for x in dataset.item_metadata.keys()]
    else:
        catalog = list(range(dataset.n_items))

    pool = [iid for iid in catalog if iid not in positives]

    if len(pool) <= probe_index:
        raise ValueError(
            f"User {uid}: only {len(pool)} negatives available, "
            f"cannot run probe {probe_index + 1}"
        )

    rng = random.Random(
        user_seed(seed, uid, "validation_negative_sequence")
    )

    n_draws = min(max_failure_probes, len(pool))
    negatives = rng.sample(pool, n_draws)

    return int(negatives[probe_index])


def build_failure_detection_prompt(
    dataset: RecDataset,
    uid: int,
    positive_id: int,
    negative_id: int,
    instruction: str,
    seed: int,
    probe_index: int,
) -> Dict[str, Any]:
    history_ids = [int(x) for x in dataset.train_data[uid]]
    history_text = format_history(dataset, history_ids)

    positive = item_info(dataset, positive_id)
    negative = item_info(dataset, negative_id)

    # Different deterministic A/B ordering per probe.
    rng = random.Random(
        user_seed(seed, uid, f"pair_order_probe_{probe_index}")
    )

    pair = [
        ("positive", positive),
        ("negative", negative),
    ]
    rng.shuffle(pair)

    label_to_kind = {
        "A": pair[0][0],
        "B": pair[1][0],
    }
    label_to_item = {
        "A": pair[0][1],
        "B": pair[1][1],
    }

    prompt = f"""You are making a recommendation decision from observed user evidence.

Observed training history (oldest to newest):
{history_text}

Current user instruction:
{instruction or "No additional instruction was provided."}

Candidate A:
Title: {label_to_item["A"]["title"]}
Description: {label_to_item["A"]["description"] or "N/A"}

Candidate B:
Title: {label_to_item["B"]["title"]}
Description: {label_to_item["B"]["description"] or "N/A"}

Choose the ONE candidate that better matches the user's preference evidenced by
the observed history and instruction.

Return ONLY valid JSON:
{{"choice": "A", "reason": "brief evidence-based reason"}}

Rules:
- choice must be exactly "A" or "B";
- do not use knowledge of which item is the observed validation item;
- ground the decision only in the provided user evidence and candidate facts.
"""

    return {
        "prompt": prompt,
        "probe_index": probe_index + 1,
        "history_ids": history_ids,
        "positive": positive,
        "negative": negative,
        "label_to_kind": label_to_kind,
        "label_to_item": label_to_item,
    }


def build_generalization_prompt(
    dataset: RecDataset,
    instruction: str,
    history_ids: List[int],
    positive: Dict[str, Any],
    negative: Dict[str, Any],
    decision_reason: str,
) -> str:
    history_text = format_history(dataset, history_ids)

    return f"""A recommendation ranker made the following mistake.

Observed training history (oldest to newest):
{history_text}

Current user instruction:
{instruction or "No additional instruction was provided."}

Recorded validation item:
Title: {positive["title"]}
Description: {positive["description"] or "N/A"}

Incorrectly preferred negative item:
Title: {negative["title"]}
Description: {negative["description"] or "N/A"}

Ranker's reason:
{decision_reason or "No reliable reason was produced."}

Derive ONE transferable failure lesson that could help rank future candidate
sets for this user or collaboratively similar users.

Return ONLY valid JSON:
{{"lesson": "one concise transferable lesson", "confidence": 0.0}}

Rules:
- use only evidence supported by the history, instruction, and item facts;
- abstract away from item IDs and exact item titles;
- describe transferable preference attributes or ranking behavior;
- do not infer demographics or unsupported preferences;
- do not merely restate that the recorded item was correct;
- keep the lesson under 45 words;
- confidence must be between 0 and 1;
- if evidence is insufficient for a meaningful transferable lesson,
  return {{"lesson": "NONE", "confidence": 0.0}}.
"""


class UnslothBatchLLM:
    def __init__(
        self,
        model_name: str,
        max_seq_length: int,
        load_in_4bit: bool,
    ):
        from unsloth import FastModel

        print(f"Loading model once: {model_name}")

        self.model, self.processor = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
        FastModel.for_inference(self.model)

        self.tokenizer = getattr(
            self.processor,
            "tokenizer",
            self.processor,
        )
        self.tokenizer.padding_side = "left"

        try:
            self.tokenizer.truncation_side = "left"
        except Exception:
            pass

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_seq_length = int(max_seq_length)
        self.calls = 0
        self.examples = 0
        self.seconds = 0.0

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int,
    ) -> List[str]:
        texts = []

        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]

            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        inp = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )

        inp = {
            k: v.to(self.model.device)
            for k, v in inp.items()
        }

        width = int(inp["input_ids"].shape[1])

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():
            out = self.model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        self.calls += 1
        self.examples += len(prompts)
        self.seconds += elapsed

        return [
            self.tokenizer.decode(
                row[width:],
                skip_special_tokens=True,
            ).strip()
            for row in out
        ]

    def stats(self) -> Dict[str, Any]:
        return {
            "generate_calls": self.calls,
            "examples": self.examples,
            "seconds": self.seconds,
            "examples_per_second": (
                self.examples / self.seconds
                if self.seconds else 0.0
            ),
        }


def parse_pairwise_choice(raw: str) -> Dict[str, Any]:
    try:
        obj = extract_json(raw)
        choice = str(obj.get("choice", "")).strip().upper()
        reason = clean_text(obj.get("reason", ""), 500)
    except Exception:
        choice, reason = "", ""

    if choice not in {"A", "B"}:
        m = re.search(
            r'\bchoice\b[^AB]{0,20}\b([AB])\b',
            str(raw),
            re.I,
        )
        choice = m.group(1).upper() if m else ""

    return {
        "valid": choice in {"A", "B"},
        "choice": choice,
        "reason": reason,
    }


def parse_generalized_lesson(raw: str) -> Dict[str, Any]:
    try:
        obj = extract_json(raw)
        lesson = clean_text(obj.get("lesson", ""), 700)
        confidence = float(obj.get("confidence", 0.0))
    except Exception:
        return {
            "valid": False,
            "lesson": "",
            "confidence": 0.0,
        }

    confidence = max(0.0, min(1.0, confidence))
    valid = bool(lesson) and lesson.upper() != "NONE"

    return {
        "valid": valid,
        "lesson": lesson if valid else "",
        "confidence": confidence if valid else 0.0,
    }


def build_lessons(args) -> None:
    seed_all(args.seed)

    dataset = RecDataset(args.data_path, seed=args.seed)
    dataset.load_item_metadata()
    dataset.load_instructions()

    users = load_users(
        dataset,
        args.user_scope,
        args.eval_user_list,
        args.max_users,
    )

    print(
        f"Lesson-source users: {len(users)} "
        f"(scope={args.user_scope})"
    )
    print(
        f"Failure mining budget: "
        f"up to {args.max_failure_probes} probes/user"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    memory_path = os.path.join(
        args.output_dir,
        "failure_memory.jsonl",
    )
    direct_path = os.path.join(
        args.output_dir,
        "lessons_by_user.jsonl",
    )
    audit_path = os.path.join(
        args.output_dir,
        "audit.jsonl",
    )
    summary_path = os.path.join(
        args.output_dir,
        "summary.json",
    )

    for path in (
        memory_path,
        direct_path,
        audit_path,
    ):
        if os.path.exists(path):
            os.remove(path)

    llm = UnslothBatchLLM(
        args.model_name,
        args.max_seq_length,
        args.load_in_4bit,
    )

    failures: List[Dict[str, Any]] = []
    user_output: Dict[int, List[str]] = {
        uid: [] for uid in users
    }

    # ================================================================
    # Phase 1: sequential failure mining.
    # ================================================================
    unresolved_users = list(users)

    failures_by_probe: Dict[str, int] = {}
    attempts_by_probe: Dict[str, int] = {}
    total_detection_probes = 0

    for probe_index in range(args.max_failure_probes):
        if not unresolved_users:
            break

        round_number = probe_index + 1
        round_users = list(unresolved_users)
        next_unresolved: List[int] = []

        round_failures_before = len(failures)

        print(
            f"\nFailure probe {round_number}/"
            f"{args.max_failure_probes}: "
            f"{len(round_users)} unresolved users"
        )

        pbar = tqdm(
            total=len(round_users),
            desc=f"Failure probe {round_number}",
        )

        attempts_by_probe[str(round_number)] = len(round_users)
        total_detection_probes += len(round_users)

        for start in range(
            0,
            len(round_users),
            args.batch_size,
        ):
            batch_users = round_users[
                start:start + args.batch_size
            ]

            requests = []

            for uid in batch_users:
                positive_id = int(
                    dataset.valid_data[uid]
                )

                negative_id = sample_validation_negative(
                    dataset=dataset,
                    uid=uid,
                    seed=args.seed,
                    probe_index=probe_index,
                    max_failure_probes=args.max_failure_probes,
                )

                instruction = get_instruction(
                    dataset,
                    uid,
                    args.use_instruction,
                )

                req = build_failure_detection_prompt(
                    dataset=dataset,
                    uid=uid,
                    positive_id=positive_id,
                    negative_id=negative_id,
                    instruction=instruction,
                    seed=args.seed,
                    probe_index=probe_index,
                )

                req["uid"] = uid
                req["instruction"] = instruction
                requests.append(req)

            raw_outputs = llm.generate_batch(
                [r["prompt"] for r in requests],
                args.decision_max_new_tokens,
            )

            parsed = [
                parse_pairwise_choice(raw)
                for raw in raw_outputs
            ]

            # Retry malformed pairwise outputs on the SAME probe.
            bad = [
                i for i, x in enumerate(parsed)
                if not x["valid"]
            ]

            if bad and args.max_retries > 0:
                retry_prompts = [
                    requests[i]["prompt"]
                    + '\nYour previous output was invalid. '
                      'Return ONLY '
                      '{"choice":"A","reason":"brief reason"} '
                      'or '
                      '{"choice":"B","reason":"brief reason"}.'
                    for i in bad
                ]

                retry_outputs = llm.generate_batch(
                    retry_prompts,
                    args.decision_max_new_tokens,
                )

                for local_i, global_i in enumerate(bad):
                    parsed[global_i] = (
                        parse_pairwise_choice(
                            retry_outputs[local_i]
                        )
                    )

            audit_rows = []

            for req, pred, raw in zip(
                requests,
                parsed,
                raw_outputs,
            ):
                uid = req["uid"]

                chosen_kind = req[
                    "label_to_kind"
                ].get(pred["choice"], "")

                chosen_item = req[
                    "label_to_item"
                ].get(pred["choice"], {})

                is_failure = (
                    pred["valid"]
                    and chosen_kind == "negative"
                )

                audit_rows.append({
                    "user_id": uid,
                    "stage": "failure_detection",
                    "probe_index": round_number,
                    "valid_output": pred["valid"],
                    "choice": pred["choice"],
                    "chosen_kind": chosen_kind,
                    "chosen_item_id": (
                        chosen_item.get("item_id")
                    ),
                    "validation_item_id": (
                        req["positive"]["item_id"]
                    ),
                    "negative_item_id": (
                        req["negative"]["item_id"]
                    ),
                    "is_failure": is_failure,
                    "reason": pred["reason"],
                    "raw_output": raw,
                })

                if is_failure:
                    # FIRST failure only. User is removed from future rounds.
                    failures.append({
                        "uid": uid,
                        "probe_index": round_number,
                        "instruction": req["instruction"],
                        "history_ids": req["history_ids"],
                        "positive": req["positive"],
                        "negative": req["negative"],
                        "decision_reason": pred["reason"],
                    })
                else:
                    # Correct or still-invalid output:
                    # user remains eligible for the next probe.
                    next_unresolved.append(uid)

            append_jsonl(
                audit_path,
                audit_rows,
            )

            pbar.update(len(batch_users))
            pbar.set_postfix({
                "total_failures": len(failures),
            })

        pbar.close()

        round_failures = (
            len(failures) - round_failures_before
        )

        failures_by_probe[str(round_number)] = (
            round_failures
        )

        unresolved_users = next_unresolved

        discovered = len({
            int(x["uid"])
            for x in failures
        })

        coverage = (
            discovered / len(users)
            if users else 0.0
        )

        print(
            f"Probe {round_number}: "
            f"{round_failures} new failures; "
            f"cumulative coverage={coverage:.4f}; "
            f"remaining={len(unresolved_users)}"
        )

    # ================================================================
    # Phase 2: generalize FIRST discovered failure per user.
    # ================================================================
    pbar = tqdm(
        total=len(failures),
        desc="Generalize failure lessons",
    )

    n_generalized = 0

    for start in range(
        0,
        len(failures),
        args.batch_size,
    ):
        batch = failures[
            start:start + args.batch_size
        ]

        prompts = [
            build_generalization_prompt(
                dataset=dataset,
                instruction=row["instruction"],
                history_ids=row["history_ids"],
                positive=row["positive"],
                negative=row["negative"],
                decision_reason=row["decision_reason"],
            )
            for row in batch
        ]

        raw_outputs = llm.generate_batch(
            prompts,
            args.lesson_max_new_tokens,
        )

        parsed = [
            parse_generalized_lesson(raw)
            for raw in raw_outputs
        ]

        bad = [
            i for i, x in enumerate(parsed)
            if not x["valid"]
        ]

        if bad and args.max_retries > 0:
            retry_prompts = [
                prompts[i]
                + "\nThe previous answer was unusable. "
                  "If evidence supports a transferable lesson, "
                  "return exactly the requested JSON. "
                  'Otherwise return '
                  '{"lesson":"NONE","confidence":0.0}.'
                for i in bad
            ]

            retry_outputs = llm.generate_batch(
                retry_prompts,
                args.lesson_max_new_tokens,
            )

            for local_i, global_i in enumerate(bad):
                parsed[global_i] = (
                    parse_generalized_lesson(
                        retry_outputs[local_i]
                    )
                )

        memory_rows = []
        audit_rows = []

        for row, lesson_row, raw in zip(
            batch,
            parsed,
            raw_outputs,
        ):
            uid = int(row["uid"])
            positive = row["positive"]
            negative = row["negative"]

            factual_statement = (
                "Given the source user's observed training history, "
                f"the recorded validation item was "
                f"'{positive['title']}', "
                f"while the ranker selected "
                f"'{negative['title']}'."
            )

            generalized_lesson = (
                lesson_row["lesson"]
                if lesson_row["valid"]
                else ""
            )

            event_key = (
                f"{uid}|"
                f"{positive['item_id']}|"
                f"{negative['item_id']}|"
                f"probe={row['probe_index']}|"
                f"{args.seed}"
            )

            memory_id = hashlib.sha256(
                event_key.encode()
            ).hexdigest()[:16]

            memory_rows.append({
                "memory_id": memory_id,
                "source_user_id": uid,
                "memory_type": (
                    "validation_failure_lesson"
                ),
                "probe_index": row["probe_index"],
                "factual_statement": factual_statement,
                "lesson": generalized_lesson,
                "confidence": lesson_row["confidence"],
                "history_item_ids": row["history_ids"],
                "correct_item_id": (
                    positive["item_id"]
                ),
                "correct_item_title": (
                    positive["title"]
                ),
                "wrong_item_id": (
                    negative["item_id"]
                ),
                "wrong_item_title": (
                    negative["title"]
                ),
                "instruction": row["instruction"],
                "decision_reason": (
                    row["decision_reason"]
                ),
                "seed": args.seed,
            })

            if generalized_lesson:
                user_output[uid].append(
                    generalized_lesson
                )
                n_generalized += 1

            audit_rows.append({
                "user_id": uid,
                "stage": "lesson_generalization",
                "probe_index": row["probe_index"],
                "memory_id": memory_id,
                "factual_statement": factual_statement,
                "lesson": generalized_lesson,
                "confidence": lesson_row["confidence"],
                "valid_lesson": bool(
                    generalized_lesson
                ),
                "raw_output": raw,
            })

        append_jsonl(
            memory_path,
            memory_rows,
        )
        append_jsonl(
            audit_path,
            audit_rows,
        )

        pbar.update(len(batch))
        pbar.set_postfix({
            "usable_lessons": n_generalized,
        })

    pbar.close()

    # Direct file for llmrank.py.
    direct_rows = [
        {
            "user_id": uid,
            "memory_facts": (
                user_output.get(uid, [])
                [:args.max_lessons_per_user]
            ),
        }
        for uid in users
    ]

    append_jsonl(
        direct_path,
        direct_rows,
    )

    users_with_failure = len({
        int(x["uid"])
        for x in failures
    })

    users_with_lesson = sum(
        bool(user_output.get(uid))
        for uid in users
    )

    summary = {
        "method": "FMRec lesson construction",
        "data_path": args.data_path,
        "user_scope": args.user_scope,
        "n_users": len(users),

        "failure_definition": (
            "First probe in which the LLM prefers a sampled "
            "negative over the validation item"
        ),
        "failure_mining_policy": (
            "Sequential probes; stop after first detected failure"
        ),

        "uses_train_history": True,
        "uses_instruction": args.use_instruction,
        "uses_validation_as_positive": True,
        "uses_test_for_lesson_creation": False,

        "max_failure_probes": args.max_failure_probes,
        "total_detection_probes_executed": total_detection_probes,
        "attempts_by_probe": attempts_by_probe,
        "failures_by_probe": failures_by_probe,

        "n_detected_failures": len(failures),
        "users_with_failure": users_with_failure,

        # Important: with R>1 this is discovery coverage, not single-trial error rate.
        "failure_discovery_coverage": (
            users_with_failure / len(users)
            if users else 0.0
        ),

        "users_without_detected_failure": (
            len(users) - users_with_failure
        ),

        "n_generalized_lessons": n_generalized,
        "users_with_usable_lesson": users_with_lesson,
        "lesson_coverage": (
            users_with_lesson / len(users)
            if users else 0.0
        ),

        "llm_stats": llm.stats(),

        "outputs": {
            "failure_memory": memory_path,
            "llmrank_lessons": direct_path,
            "audit": audit_path,
        },
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 80)
    print("FMRec lesson construction complete")
    print("=" * 80)
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data_path",
        default=(
            "data/processed/instructrec-books/"
            "instructrec-books.inter"
        ),
    )
    p.add_argument(
        "--output_dir",
        default="results/fmrec_lessons_books_r2",
    )
    p.add_argument(
        "--eval_user_list",
        default=(
            "data/eval_user_samples/"
            "eval_user_sample_1k_instructrec-books.json"
        ),
    )

    p.add_argument(
        "--user_scope",
        choices=["eval", "all"],
        default="all",
    )
    p.add_argument(
        "--max_users",
        type=int,
        default=None,
    )

    p.add_argument(
        "--model_name",
        default=(
            "unsloth/"
            "gemma-3-4b-it-unsloth-bnb-4bit"
        ),
    )
    p.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
    )
    p.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    p.add_argument(
        "--decision_max_new_tokens",
        type=int,
        default=128,
    )
    p.add_argument(
        "--lesson_max_new_tokens",
        type=int,
        default=160,
    )
    p.add_argument(
        "--max_retries",
        type=int,
        default=1,
    )

    # Main change: R=2 by default.
    p.add_argument(
        "--max_failure_probes",
        type=int,
        default=2,
        help=(
            "Maximum random-negative probes per user. "
            "Stop after the first detected failure."
        ),
    )

    p.add_argument(
        "--use_instruction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Still at most one personal lesson per user.
    p.add_argument(
        "--max_lessons_per_user",
        type=int,
        default=1,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return p.parse_args()


if __name__ == "__main__":
    build_lessons(parse_args())
