#!/usr/bin/env python3
"""
LLMRank / FMRec reranker aligned with MemRec vanilla Stage-ReRank.

Vanilla mode:
    instruction + full TRAIN history + candidate items
    -> LLM scores/ranks candidates

FMRec mode:
    same exact vanilla input + retrieved failure lessons
    -> LLM scores/ranks candidates

Important:
- Full TRAIN history is passed to both Vanilla and FMRec.
- This reranker receives only already-existing/retrieved lesson facts.
- With --lessons_file omitted, the prompt follows MemRec's vanilla
  reranker prompt as closely as possible.
- Uses raw candidate item IDs (not C01/C02 aliases), like MemRec.
- Test candidates follow MemRec final-test candidate construction.
- No warm-up, no reflection, and no lesson generation occur in this script.
- Failure lessons are optional external inputs; without them the script is Vanilla.
"""

import argparse
import html
import json
import math
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

# ---------------------------------------------------------------------------
# MemRec import
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.data import RecDataset
except ModuleNotFoundError as e:
    raise RuntimeError(
        f"Cannot import MemRec `src` package.\n"
        f"Place this file inside <MemRec>/scripts/.\n"
        f"Expected repository root: {REPO_ROOT}\n"
        f"Expected src directory: {REPO_ROOT / 'src'}"
    ) from e


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

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


def get_candidate_item(dataset: RecDataset, item_id: int, use_description: bool = False,) -> Dict[str, Any]:
    """
    Match MemRec's _prepare_candidate_list() by default:
        id + title + tags

    Stock MemRec currently does NOT pass description from memrec_agent.py
    into reranker_llm.py. If you patched MemRec to include descriptions,
    run this script with --use_description.
    """
    meta = (dataset.item_metadata or {}).get(int(item_id), {})

    tags = meta.get("tags", [])
    if tags is None:
        tags = []
    if isinstance(tags, str):
        tags = [tags]
    elif not isinstance(tags, list):
        tags = list(tags) if isinstance(tags, (tuple, set)) else []

    row = {
        "id": int(item_id),
        "title": meta.get("title", "N/A"),
        "tags": [str(x) for x in tags],
    }

    if use_description:
        # MemRec reranker itself truncates descriptions at 200 chars.
        row["description"] = clean_text(meta.get("description", ""), 200)

    return row


def get_history_items(dataset: RecDataset, uid: int) -> List[Dict[str, Any]]:
    """Full observed TRAIN history, oldest to newest."""
    item_ids = dataset.get_user_train_items(uid)[-10:]
    rows = []
    for item_id in dataset.get_user_train_items(uid):
        item_id = int(item_id)
        meta = (dataset.item_metadata or {}).get(item_id, {})
        title = clean_text(meta.get("title", ""), 300) or f"Item {item_id}"
        rows.append({"id": item_id, "title": title})
    return rows


# ---------------------------------------------------------------------------
# Users / candidates
# ---------------------------------------------------------------------------

def load_eval_users(
    path: Optional[str],
    dataset: RecDataset,
    max_users: Optional[int],
    seed: int,
) -> List[int]:
    """
    Match MemRec evaluation user selection.

    Priority in MemRec:
      eval_user_list > n_eval_users > all test users.

    With a fixed eval_user_list (the standard 1K setup), no RNG is consumed.
    If max_users is used without a list, MemRec reseeds Python random and
    random.sample()s the users.
    """
    if path:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        ids = obj.get("user_ids") if isinstance(obj, dict) else obj
        if ids is None:
            raise ValueError(f"No user_ids found in {path}")

        users = [int(x) for x in ids]
        users = [u for u in users if u in dataset.test_data]
        return users

    all_users = list(dataset.test_data.keys())

    if max_users is not None and max_users > 0 and max_users < len(all_users):
        # Exact behavior of MemRecTrainer.evaluate() when n_eval_users is used.
        random.seed(seed)
        return random.sample(all_users, max_users)

    return all_users


def make_memrec_test_candidates(
    dataset: RecDataset,
    users: List[int],
    n_candidates: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Candidate construction for the FINAL test only.

    This matches MemRecTrainer.evaluate() test-candidate logic:
        target = dataset.test_data[user_id]
        all_items = set(range(dataset.n_items))
        train_history = set(dataset.get_user_train_items(user_id))
        negative_pool = all_items - train_history - {target}
        negatives = random.sample(negative_pool, n_candidates - 1)
        candidates = [target] + negatives
        random.shuffle(candidates)

    There is deliberately NO warm-up, NO reflection, and NO memory update here.
    """
    out: Dict[int, Dict[str, Any]] = {}

    for uid in users:
        target = int(dataset.test_data[uid])

        all_items = set(range(dataset.n_items))
        train_history = set(dataset.get_user_train_items(uid))
        negative_pool = list(
            all_items - train_history - {target}
        )

        if len(negative_pool) < n_candidates - 1:
            raise ValueError(
                f"User {uid}: only {len(negative_pool)} MemRec test negatives "
                f"available; need {n_candidates - 1}"
            )

        negative_items = random.sample(
            negative_pool,
            n_candidates - 1,
        )

        candidate_ids = [target] + [
            int(x) for x in negative_items
        ]
        random.shuffle(candidate_ids)

        out[uid] = {
            "target": target,
            "candidates": candidate_ids,
        }

    return out


def load_or_make_candidates(
    dataset: RecDataset,
    users: List[int],
    n_candidates: int,
    seed: int,
    path: Optional[str],
) -> Dict[int, Dict[str, Any]]:
    """
    Load a frozen candidate file, or create it once using MemRec's FINAL-test
    candidate construction.

    The generated file is then reused by Vanilla and FMRec so both methods see
    exactly the same users, target item, negatives, and candidate order.
    """
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, dict) and "candidate_source" in obj:
            source = obj.get("candidate_source")
            if source != "memrec_test":
                raise ValueError(
                    f"{path} has candidate_source={source!r}. "
                    "Use a MemRec-test candidate file or delete/rename this file "
                    "so it can be regenerated."
                )
        elif isinstance(obj, dict) and "users" in obj:
            raise ValueError(
                f"{path} has no candidate_source marker. It may be an older "
                "FMRec candidate file. Delete/rename it and let this script "
                "regenerate a clean MemRec-test candidate file."
            )

        raw = obj.get("users", obj)
        out = {
            int(uid): {
                "target": int(row["target"]),
                "candidates": [int(x) for x in row["candidates"]],
            }
            for uid, row in raw.items()
        }

        for uid in users:
            if uid not in out:
                raise KeyError(f"Candidate file missing user {uid}")
            if out[uid]["target"] != int(dataset.test_data[uid]):
                raise ValueError(f"Target mismatch for user {uid}")
            if len(out[uid]["candidates"]) != n_candidates:
                raise ValueError(f"Candidate count mismatch for user {uid}")

        print(f"✓ Loaded frozen MemRec-test candidates: {path}")
        return out

    # seed_all(seed) has already seeded Python random before this function.
    out = make_memrec_test_candidates(
        dataset=dataset,
        users=users,
        n_candidates=n_candidates,
    )

    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "candidate_source": "memrec_test",
                    "seed": seed,
                    "n_candidates": n_candidates,
                    "users": {
                        str(uid): row
                        for uid, row in out.items()
                    },
                },
                f,
                indent=2,
            )
        print(f"✓ Saved frozen MemRec-test candidates: {path}")

    return out


# ---------------------------------------------------------------------------
# Failure lessons
# ---------------------------------------------------------------------------

def _ground_raw_item_id(
    dataset: RecDataset,
    text: str,
) -> str:
    """
    Deterministic metadata grounding only; this is NOT reflection.

    If an already-provided lesson is nothing but a raw item ID, replace it with:
        [Item-<id>] <title>

    Otherwise keep the lesson text unchanged.
    """
    s = clean_text(text, 700)
    if not s:
        return ""

    m = re.fullmatch(
        r"\[?\s*Item[-\s:]*(\d+)\s*\]?",
        s,
        flags=re.I,
    )
    if not m and re.fullmatch(r"\d+", s):
        item_id = int(s)
    elif m:
        item_id = int(m.group(1))
    else:
        return s

    meta = (dataset.item_metadata or {}).get(item_id, {})
    title = clean_text(meta.get("title", ""), 300)

    if title:
        return f"[Item-{item_id}] {title}"
    return f"[Item-{item_id}]"


def normalize_existing_lesson(
    value: Any,
    dataset: RecDataset,
) -> Optional[str]:
    """
    Accept ONLY an already-existing lesson/fact.

    No Stage-W.
    No reflection.
    No LLM call.
    No automatic construction from correct/wrong item IDs.
    """
    if isinstance(value, str):
        text = clean_text(value, 700)
    elif isinstance(value, dict):
        text = ""
        for key in (
            "safe_fact",
            "factual_statement",
            "lesson",
            "fact",
            "memory",
            "memory_update",
        ):
            if value.get(key):
                text = clean_text(value[key], 700)
                break
    else:
        return None

    if not text:
        return None

    if text.lower() in {
        "nan",
        "[nan]",
        "['']",
        "none",
        "null",
        "n/a",
    }:
        return None

    return _ground_raw_item_id(dataset, text) or None


def normalize_facts(
    value: Any,
    max_facts: int,
    dataset: RecDataset,
) -> List[str]:
    if value is None:
        return []

    if isinstance(value, dict):
        if "memory_facts" in value:
            value = value["memory_facts"]
        elif "lessons" in value:
            value = value["lessons"]

    if isinstance(value, (str, dict)):
        value = [value]

    if not isinstance(value, list):
        return []

    facts: List[str] = []
    seen = set()

    for row in value:
        fact = normalize_existing_lesson(row, dataset)
        if fact and fact not in seen:
            facts.append(fact)
            seen.add(fact)

        if max_facts > 0 and len(facts) >= max_facts:
            break

    return facts


def load_lessons(
    path: Optional[str],
    max_facts: int,
    dataset: RecDataset,
) -> Dict[int, List[str]]:
    """
    Failure lessons are intentionally external/precomputed.

    If path is omitted:
        {}  -> Vanilla mode.

    If path is provided:
        only the existing natural-language lesson strings are loaded.
        This script NEVER reflects on failures and NEVER generates lessons.
    """
    if not path:
        return {}

    out: Dict[int, List[str]] = {}

    if Path(path).suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                uid = int(row["user_id"])
                out[uid] = normalize_facts(
                    row,
                    max_facts,
                    dataset,
                )
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, dict):
            obj = obj.get("users", obj)
            for uid, value in obj.items():
                out[int(uid)] = normalize_facts(
                    value,
                    max_facts,
                    dataset,
                )
        else:
            for row in obj:
                uid = int(row["user_id"])
                out[uid] = normalize_facts(
                    row,
                    max_facts,
                    dataset,
                )

    print(
        f"✓ Existing lessons loaded: "
        f"{sum(bool(v) for v in out.values())}/{len(out)} users"
    )

    return out


# ---------------------------------------------------------------------------
# Prompt: aligned with MemRec vanilla reranker_llm.py
# ---------------------------------------------------------------------------

def build_memrec_vanilla_prompt(
    user_id: int,
    instruction: Optional[str],
    history_items: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    failure_lessons: Optional[List[str]] = None,
) -> str:
    parts = [
        "You are an intelligent recommendation scoring system. "
        "Evaluate how well each candidate matches the target user's preferences.",
        f"\n\n**Target User:** User {user_id}",
    ]

    if instruction:
        parts.append(f"\n\n**User Instruction:**\n{instruction}")
    else:
        parts.append("\n\n**User Instruction:**\nNo specific instruction provided.")

    parts.append("\n\n**Observed Training History (oldest to newest):**")
    if history_items:
        for idx, item in enumerate(history_items, 1):
            parts.append(f"\n  {idx}. {item['title']}")
    else:
        parts.append("\n  No observed training history.")

    if failure_lessons:
        parts.append("\n\n**Retrieved Failure Lessons:**")
        for idx, fact in enumerate(failure_lessons, 1):
            parts.append(f"\n  {idx}. {fact}")
        parts.append(
            "\nUse these lessons only when relevant to the current user's "
            "instruction, history, and candidate items."
        )

    parts.append("\n\n**Candidate Items:**")
    for c in candidates:
        cid = int(c["id"])
        title = c.get("title", f"Item {cid}")
        description = c.get("description", "")
        tags = c.get("tags", [])

        if description:
            if len(description) > 200:
                description = description[:200] + "..."
            parts.append(f"\n  • Item {cid}: {title}. {description}")
        elif tags:
            tags_str = ", ".join(str(x) for x in tags[:5])
            parts.append(f"\n  • Item {cid}: {title} (Tags: {tags_str})")
        else:
            parts.append(f"\n  • Item {cid}: {title}")

    parts.append("""

**Your Task:**
For every candidate item above, provide a relevance score from 0 to 1:
  • 1.0 = excellent match
  • 0.5 = moderate match
  • 0.0 = poor match

Base the ranking on the user's instruction, observed training history, and
candidate characteristics. If failure lessons are provided, use them only as
additional evidence.

Return a JSON object with exactly one score row for every candidate:
{
  "scores": [
    {
      "item_id": 123,
      "score": 0.8,
      "rationale": "brief reason"
    }
  ]
}
""")

    return "".join(parts)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()

    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]

    start = text.find("{")
    end = text.rfind("}") + 1

    if start >= 0 and end > start:
        text = text[start:end]

    return json.loads(text)


def recover_score_rows(text: str) -> List[Dict[str, Any]]:
    """
    Recover rows from malformed JSON using MemRec's raw item_id schema.
    """
    pattern = re.compile(
        r'"item_id"\s*:\s*"?(\d+)"?'
        r'[^{}]{0,120}?'
        r'"score"\s*:\s*"?([0-9]*\.?[0-9]+)"?',
        re.I,
    )

    return [
        {
            "item_id": int(match.group(1)),
            "score": float(match.group(2)),
            "rationale": "Recovered from malformed JSON",
        }
        for match in pattern.finditer(str(text or ""))
    ]


def parse_ranking(
    raw: str,
    candidate_ids: List[int],
):
    try:
        rows = extract_json(raw).get("scores", [])
    except Exception:
        rows = recover_score_rows(raw)

    candidate_set = set(int(x) for x in candidate_ids)
    original_index = {
        int(item_id): idx
        for idx, item_id in enumerate(candidate_ids)
    }

    seen = set()
    parsed = []

    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue

        try:
            item_id = int(row.get("item_id"))
        except Exception:
            continue

        if item_id not in candidate_set or item_id in seen:
            continue

        try:
            score = float(row.get("score", 0.0))
        except Exception:
            score = 0.0

        parsed.append({
            "item_id": item_id,
            "score": max(0.0, min(1.0, score)),
            "rationale": str(row.get("rationale", "")),
        })
        seen.add(item_id)

    valid = len(seen) == len(candidate_ids)

    # Match MemRec behavior as closely as practical:
    # missing outputs are retained at the end rather than disappearing.
    for item_id in candidate_ids:
        item_id = int(item_id)
        if item_id not in seen:
            parsed.append({
                "item_id": item_id,
                "score": -1.0,
                "rationale": "Missing from LLM output",
            })

    parsed.sort(
        key=lambda x: (
            -x["score"],
            original_index[x["item_id"]],
        )
    )

    ranking = [
        row["item_id"]
        for row in parsed
    ]

    return ranking, valid, parsed


# ---------------------------------------------------------------------------
# Local Unsloth Gemma batch generation
# ---------------------------------------------------------------------------

class UnslothBatchLLM:
    def __init__(
        self,
        model_name: str,
        max_seq_length: int,
        load_in_4bit: bool,
    ):
        from unsloth import FastModel

        print(f"Loading model ONCE: {model_name}")

        self.model, self.processor = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
        FastModel.for_inference(self.model)

        # Gemma-3 returns a Processor. Use its inner tokenizer for text-only
        # batched tensorization, while keeping Processor for chat templates.
        self.tokenizer = getattr(
            self.processor,
            "tokenizer",
            self.processor,
        )

        self.max_seq_length = int(max_seq_length)

        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.model.generation_config.pad_token_id = (
                self.tokenizer.pad_token_id
            )
        except Exception:
            pass

        self.calls = 0
        self.examples = 0
        self.seconds = 0.0

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int,
    ) -> List[str]:

        # IMPORTANT: MemRec's reranker returns ONE user message.
        # No system message is added here.
        batch_messages = [
            [{"role": "user", "content": prompt}]
            for prompt in prompts
        ]

        texts = []

        for messages in batch_messages:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)

        inp = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )

        inp = {
            key: value.to(self.model.device)
            for key, value in inp.items()
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


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------

def make_request(
    dataset: RecDataset,
    uid: int,
    candidate_row: Dict[str, Any],
    lessons: Dict[int, List[str]],
    use_instruction: bool,
    use_description: bool,
) -> Dict[str, Any]:

    instruction = ""
    if use_instruction and dataset.instructions and uid in dataset.instructions:
        instruction = str(dataset.instructions[uid].get("instruction", "") or "")

    history_items = get_history_items(dataset, uid)
    candidate_ids = [int(x) for x in candidate_row["candidates"]]
    candidate_items = [
        get_candidate_item(dataset, item_id, use_description=use_description)
        for item_id in candidate_ids
    ]
    facts = lessons.get(uid, [])

    prompt = build_memrec_vanilla_prompt(
        user_id=uid,
        instruction=instruction,
        history_items=history_items,
        candidates=candidate_items,
        failure_lessons=facts,
    )

    return {
        "uid": uid,
        "target": int(candidate_row["target"]),
        "candidates": candidate_ids,
        "n_history_items": len(history_items),
        "facts": facts,
        "prompt": prompt,
    }


# ---------------------------------------------------------------------------
# Metrics / persistence
# ---------------------------------------------------------------------------

def metrics(positions: List[int]) -> Dict[str, float]:
    if not positions:
        return {}

    n = len(positions)
    out = {}

    for k in (1, 3, 5, 10):
        out[f"Hit@{k}"] = (
            sum(pos < k for pos in positions)
            / n
        )
        out[f"NDCG@{k}"] = (
            sum(
                (1.0 / math.log2(pos + 2))
                if pos < k else 0.0
                for pos in positions
            )
            / n
        )

    out["MRR"] = (
        sum(
            1.0 / (pos + 1)
            for pos in positions
        )
        / n
    )

    return out


def append_jsonl(
    path: str,
    rows: List[Dict[str, Any]],
) -> None:
    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True,
    )

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_resume(
    path: str,
):
    done = set()
    rows = []

    if os.path.exists(path):
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                done.add(int(row["user_id"]))
                rows.append(row)

    return done, rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> None:
    seed_all(args.seed)

    dataset = RecDataset(args.data_path, seed=args.seed)
    dataset.load_item_metadata()
    dataset.load_instructions()

    users = load_eval_users(
        args.eval_user_list, dataset, args.max_users, args.seed
    )

    candidates = load_or_make_candidates(
        dataset, users, args.n_candidates, args.seed, args.candidate_file
    )

    lessons = load_lessons(
        args.lessons_file, args.max_lesson_facts, dataset
    )

    mode = "FMRec" if args.lessons_file else "Vanilla-MemRec-aligned LLMRank"

    print(
        f"Mode={mode}; users={len(users)}; candidates={args.n_candidates}; "
        f"batch={args.batch_size}"
    )

    if args.lessons_file:
        print(
            "Final ranker input: Instruction + TRAIN History + Failure Lessons + Candidate Items "
            "(no warm-up, no reflection)"
        )
    else:
        print(
            "Final ranker input: Instruction + TRAIN History + Candidate Items "
            "(Vanilla; no warm-up, no reflection)"
        )

    metadata = (
        "title + description"
        if args.use_description
        else "title + tags (stock MemRec)"
    )
    print(f"Candidate metadata: {metadata}")

    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, "per_user_results.jsonl")

    if args.resume:
        done, old_rows = load_resume(result_path)
    else:
        done, old_rows = set(), []

        # Prevent accidentally mixing old results from a different mode.
        if os.path.exists(result_path):
            os.remove(result_path)

    pending = [uid for uid in users if uid not in done]

    llm = UnslothBatchLLM(
        args.model_name, args.max_seq_length, args.load_in_4bit
    )

    new_rows = []

    pbar = tqdm(total=len(pending), desc=f"{mode} test")

    for start in range(0, len(pending), args.batch_size):
        batch_users = pending[start:start + args.batch_size]

        requests = [
            make_request(
                dataset, uid, candidates[uid], lessons,
                args.use_instruction, args.use_description
            )
            for uid in batch_users
        ]

        raw_outputs = llm.generate_batch(
            [r["prompt"] for r in requests], args.max_new_tokens
        )

        parsed = [
            parse_ranking(raw, req["candidates"])
            for raw, req in zip(raw_outputs, requests)
        ]

        # Retry malformed outputs, still batched.
        bad = [
            idx for idx, result in enumerate(parsed)
            if not result[1]
        ]

        for _ in range(args.max_retries):
            if not bad:
                break

            retry_prompts = [
                requests[idx]["prompt"]
                + "\n\nThe previous response was invalid. "
                "Return exactly one score object for every candidate item, "
                "using the original integer item_id values."
                for idx in bad
            ]

            retry_outputs = llm.generate_batch(retry_prompts, args.max_new_tokens)
            new_bad = []

            for local_idx, global_idx in enumerate(bad):
                parsed[global_idx] = parse_ranking(
                    retry_outputs[local_idx], requests[global_idx]["candidates"]
                )

                if not parsed[global_idx][1]:
                    new_bad.append(global_idx)

            bad = new_bad

        batch_rows = []

        for req, parsed_result in zip(requests, parsed):
            ranking, valid, score_rows = parsed_result
            target = req["target"]

            position = ranking.index(target) if target in ranking else len(ranking)

            batch_rows.append({
                "user_id": req["uid"],
                "target": target,
                "candidates": req["candidates"],
                "ranking": ranking,
                "rank_position": position,
                "valid_output": valid,
                "n_history_items": req["n_history_items"],
                "n_memory_facts": len(req["facts"]),
                "memory_facts": req["facts"],
                "scores": score_rows,
            })

        append_jsonl(result_path, batch_rows)
        new_rows.extend(batch_rows)

        pbar.update(len(batch_users))
        pbar.set_postfix({ "calls": llm.calls,
            "user/s": (f"{llm.stats()['examples_per_second']:.2f}"),
        })

    pbar.close()

    rows = old_rows + new_rows
    user_set = set(users)

    rows = [row for row in rows if int(row["user_id"]) in user_set]
    positions = [int(row["rank_position"]) for row in rows]

    summary = {
        "mode": mode,
        "model": args.model_name,
        "n_users": len(rows),
        "n_candidates": args.n_candidates,
        "candidate_source": "memrec_test",
        "warmup": False,
        "reflection": False,
        "use_instruction": args.use_instruction,
        "raw_history_in_final_prompt": True,
        "use_description": args.use_description,
        "memory_coverage": (
            sum(row.get("n_memory_facts", 0) > 0 for row in rows) / len(rows)
            if rows else 0.0
        ),
        "valid_output_rate": (
            sum(bool(row.get("valid_output")) for row in rows) / len(rows)
            if rows else 0.0
        ),
        "metrics": metrics(positions),
        "llm_stats": llm.stats(),
    }

    summary_path = os.path.join(args.output_dir, "summary.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_path",
                   default="data/processed/instructrec-books/instructrec-books.inter")
    p.add_argument("--output_dir", default="results/llmrank_books")
    p.add_argument("--eval_user_list",
                   default="data/eval_user_samples/eval_user_sample_10_instructrec-books.json")
    p.add_argument("--candidate_file",
                   default="results/books_candidates_1k_memrec_test.json")
    p.add_argument("--lessons_file", default=None)

    p.add_argument("--model_name",
                   default="unsloth/gemma-3-4b-it-unsloth-bnb-4bit")
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--load_in_4bit",
                   action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=1000)
    p.add_argument("--max_retries", type=int, default=1)
    p.add_argument("--n_candidates", type=int, default=10)


    p.add_argument("--use_instruction",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_description",
                   action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--max_lesson_facts", type=int, default=3)
    p.add_argument("--max_users", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")

    return p.parse_args()

if __name__ == "__main__":
    main(parse_args())
