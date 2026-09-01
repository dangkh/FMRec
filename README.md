# FMRec: Failure Memory for LLM-based Recommendation

FMRec is an LLM-based recommendation framework that learns **corrective knowledge from previous ranking failures** and transfers these failure lessons across users through **collaborative user similarity**.

For a target user, FMRec retrieves an optional personal failure lesson from the same user and failure lessons from the **Top-K similar users**, where user similarity is computed from pretrained LightGCN embeddings. The retrieved lessons are added to the final LLM reranking prompt together with the user's instruction, training history, and candidate items.

This repository contains the standalone FMRec implementation used for local-LLM experiments.

## Project Structure

```text
FMRec/
├── README.md
├── requirements.txt
│
├── data/
│   ├── processed/
│   │   └── instructrec-books/
│   │       ├── instructrec-books.inter
│   │       ├── instructrec-books.instruction
│   │       └── instructrec-books.meta
│   └── eval_user_samples/
│       ├── eval_user_sample_10_instructrec-books.json
│       └── eval_user_sample_1k_instructrec-books.json
│
├── src/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── dataset_base.py
│       └── ...
│
├── scripts/
│   ├── train_lightgcn.py
│   ├── llmrank_vanilla.py
│   ├── build_fmrec_lessons.py
│   ├── retrieve_fmrec_lessons.py
│   └── llmrank_fmrec.py
│
├── notebooks/
│   └── fmrec_retrieve_and_rank.ipynb
│
└── results/
    ├── books_candidates_1k_memrec_test.json
    ├── lightgcn_books/
    │   └── lightgcn_embeddings.json
    └── fmrec_lessons_books_r2/
        └── failure_memory.jsonl
```

For the current FMRec retrieval and reranking pipeline, only `src/data/` from the original MemRec `src/` directory is required. The original MemRec folders `src/memory/`, `src/models/`, and `src/train/` are not required by the standalone FMRec code.

## Environment Setup

Create a Python environment:

```bash
conda create -n fmrec python=3.10
conda activate fmrec
pip install -r requirements.txt
```

The current local-LLM implementation uses PyTorch and Unsloth. A CUDA-capable GPU is strongly recommended. Typical dependencies include:

```text
torch
numpy
pandas
tqdm
unsloth
transformers
```

The default local LLM is:

```text
unsloth/gemma-3-4b-it-unsloth-bnb-4bit
```

## Required Dataset Files

For InstructRec-Books, keep:

```text
data/processed/instructrec-books/
├── instructrec-books.inter
├── instructrec-books.instruction
└── instructrec-books.meta
```

Their roles are:

- `*.inter`: user-item interactions used to construct train/validation/test histories.
- `*.instruction`: user instructions used by the LLM reranker.
- `*.meta`: item title, tags, description, and other metadata.

The evaluation-user file is also required, for example:

```text
data/eval_user_samples/
└── eval_user_sample_1k_instructrec-books.json
```

Use the same evaluation-user list across Vanilla, MemRec, FMRec, and ablation experiments.

## Required Precomputed Files

To run the retrieval + reranking notebook directly, only three precomputed result files are required.

### 1. Frozen candidate file

```text
results/books_candidates_1k_memrec_test.json
```

This contains the target item and frozen candidate set for each evaluation user. In the current protocol:

```text
1 target + 9 negatives = 10 candidates
```

Keep this file fixed across compared methods so Vanilla and FMRec see exactly the same target, negatives, and candidate order.

### 2. LightGCN user embeddings

```text
results/lightgcn_books/
└── lightgcn_embeddings.json
```

FMRec uses these pretrained user embeddings for collaborative retrieval:

```text
Target user
    ↓
LightGCN user embedding
    ↓
Cosine similarity
    ↓
Top-K similar users
```

Only the saved embeddings are needed for the notebook. A LightGCN checkpoint is not required once the embeddings have been generated.

### 3. Failure memory

```text
results/fmrec_lessons_books_r2/
└── failure_memory.jsonl
```

Each row contains a generalized lesson learned from a previous LLM ranking failure, for example:

```json
{
  "source_user_id": 3364,
  "memory_id": "522f3245bbd15565",
  "confidence": 0.8,
  "lesson": "Prioritize narrative and experiential elements when seeking information."
}
```

## Recommended Execution: Jupyter Notebook

The recommended experiment driver is:

```text
notebooks/fmrec_retrieve_and_rank.ipynb
```

The notebook is stateful and avoids repeatedly loading expensive components. It loads the dataset, candidates, histories, candidate metadata, LightGCN embeddings, failure memory, and local LLM once, then reuses them across experiments.

### Step 1 — Start Jupyter

Run from the FMRec project root:

```bash
conda activate fmrec
cd /path/to/FMRec
jupyter lab
```

or:

```bash
jupyter notebook
```

Open:

```text
notebooks/fmrec_retrieve_and_rank.ipynb
```

### Step 2 — Configure paths

At the beginning of the notebook:

```python
DATA_PATH = "data/processed/instructrec-books/instructrec-books.inter"

EVAL_USER_LIST = (
    "data/eval_user_samples/"
    "eval_user_sample_1k_instructrec-books.json"
)

CANDIDATE_FILE = "results/books_candidates_1k_memrec_test.json"

LIGHTGCN_EMBEDDINGS = (
    "results/lightgcn_books/lightgcn_embeddings.json"
)

FAILURE_MEMORY = (
    "results/fmrec_lessons_books_r2/failure_memory.jsonl"
)
```

For a quick sanity check, use the 10-user evaluation list instead.

### Step 3 — Configure retrieval

Default FMRec retrieval:

```python
TOP_K_NEIGHBORS = 3
INCLUDE_SELF = True
LABEL_SOURCES = True
```

This allows at most:

```text
1 personal lesson
+
3 collaborative lessons
=
4 retrieved lessons
```

The current reranker additionally uses:

```python
MAX_LESSON_FACTS = 3
```

which reproduces the behavior of the current ranking script. If all four retrieved lessons should be included in the final prompt, set:

```python
MAX_LESSON_FACTS = 4
```

### Step 4 — Load dataset and candidates once

Run the dataset-loading cell. It loads `RecDataset`, metadata, instructions, evaluation users, and frozen candidates, and precomputes reusable history/candidate metadata caches.

Do not rerun this cell when only changing retrieval settings such as `TOP_K_NEIGHBORS`.

### Step 5 — Load retrieval state once

Run:

```python
user_embeddings = load_user_embeddings(LIGHTGCN_EMBEDDINGS)
failure_memory_by_user = load_failure_memory(FAILURE_MEMORY)
```

These objects remain in memory while the notebook kernel is active.

### Step 6 — Retrieve FMRec lessons

Run:

```python
retrieved_lessons, retrieval_audit, retrieval_summary = retrieve_all(
    users=users,
    embeddings=user_embeddings,
    lessons_by_user=failure_memory_by_user,
    top_k_neighbors=TOP_K_NEIGHBORS,
    include_self=INCLUDE_SELF,
    label_sources=LABEL_SOURCES,
)
```

Retrieval outputs are also saved to:

```text
results/fmrec_retrieval_books/
├── retrieved_lessons_by_user.jsonl
├── retrieval_audit.jsonl
└── summary.json
```

These files are useful for audit/analysis, but the ranking step in the same notebook uses `retrieved_lessons` directly from RAM.

### Step 7 — Load the local LLM once

Run:

```python
llm = UnslothBatchLLM(
    MODEL_NAME,
    MAX_SEQ_LENGTH,
    LOAD_IN_4BIT,
)
```

Default configuration:

```python
MODEL_NAME = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True
```

This is the expensive initialization step. Once `llm` is loaded, do not rerun this cell unless the model or notebook kernel must be restarted.

### Step 8 — Run FMRec ranking

Run:

```python
fmrec_summary, fmrec_rows = run_rank(
    lessons_by_user=retrieved_lessons,
    output_dir="results/llmrank_fmrec_history",
    resume=False,
)
```

The final FMRec prompt contains:

```text
User Instruction
+
Full Training History
+
Retrieved Failure Lessons
+
Candidate Items
```

No warm-up or reflection is performed during final reranking.

Outputs:

```text
results/llmrank_fmrec_history/
├── per_user_results.jsonl
└── summary.json
```

## Run Vanilla Without Reloading the LLM

Vanilla can be evaluated in the same notebook session:

```python
vanilla_summary, vanilla_rows = run_rank(
    lessons_by_user=None,
    output_dir="results/llmrank_vanilla_history",
    resume=False,
)
```

Vanilla receives:

```text
User Instruction
+
Full Training History
+
Candidate Items
```

FMRec receives the same information plus retrieved failure lessons. Both methods therefore use the same users, target items, negative items, candidate order, local LLM, and decoding configuration.

## Sensitivity Analysis Without Reloading the LLM

For retrieval sensitivity, load the dataset, embeddings, failure memory, and LLM once, then only rerun retrieval and ranking.

Example for `K=5`:

```python
retrieved_k5, audit_k5, retrieval_summary_k5 = retrieve_all(
    users,
    user_embeddings,
    failure_memory_by_user,
    top_k_neighbors=5,
    include_self=True,
    label_sources=True,
)

summary_k5, rows_k5 = run_rank(
    retrieved_k5,
    output_dir="results/llmrank_fmrec_k5",
)
```

The same loaded `llm` object is reused. This is the recommended way to test `K=1,2,3,5,...` without paying the model-loading cost for every experiment.

## FMRec Pipeline

```text
Interaction Data
       │
       ├─────────────────────────────┐
       │                             │
       ▼                             ▼
  LightGCN                     Vanilla LLMRank
       │                             │
       ▼                             ▼
User Embeddings                Ranking Failures
                                     │
                                     ▼
                               Lesson Builder
                                     │
                                     ▼
                               Failure Memory
       │                             │
       └──────────────┬──────────────┘
                      ▼
               FMRec Retrieval
                      │
             Personal + Top-K
             Similar-user Lessons
                      │
                      ▼
              LLM Re-ranking
                      │
                      ▼
               Hit / NDCG / MRR
```

## Optional: Rebuild Components from Scratch

The notebook assumes that LightGCN embeddings, frozen candidates, and failure memory already exist. If needed, regenerate them using the standalone scripts.

### Train LightGCN

```bash
python scripts/train_lightgcn.py
```

Expected output:

```text
results/lightgcn_books/
└── lightgcn_embeddings.json
```

### Run Vanilla LLMRank

```bash
python scripts/llmrank_vanilla.py
```

This produces the baseline ranking results used to identify ranking failures.

### Build failure lessons

```bash
python scripts/build_fmrec_lessons.py
```

Expected output:

```text
results/fmrec_lessons_books_r2/
└── failure_memory.jsonl
```

### Standalone retrieval

```bash
python scripts/retrieve_fmrec_lessons.py
```

Expected outputs:

```text
results/fmrec_retrieval_books/
├── retrieved_lessons_by_user.jsonl
├── retrieval_audit.jsonl
└── summary.json
```

For repeated experiments, however, the notebook workflow is recommended because the LLM and other shared objects remain loaded.

## Evaluation Metrics

The current evaluator reports:

- Hit@1, Hit@3, Hit@5, Hit@10
- NDCG@1, NDCG@3, NDCG@5, NDCG@10
- MRR

## Reproducibility

For a fair Vanilla–FMRec comparison:

1. Use the same evaluation-user list.
2. Use the same frozen candidate file.
3. Use the same target and negative items.
4. Use the same candidate ordering.
5. Use the same local LLM.
6. Use the same decoding configuration.
7. Disable warm-up and reflection during final reranking.
8. Change only the failure-memory input between Vanilla and FMRec.

Default settings:

```python
SEED = 42
N_CANDIDATES = 10
```

## Typical Output Structure

After a complete run:

```text
results/
├── books_candidates_1k_memrec_test.json
│
├── lightgcn_books/
│   └── lightgcn_embeddings.json
│
├── fmrec_lessons_books_r2/
│   └── failure_memory.jsonl
│
├── fmrec_retrieval_books/
│   ├── retrieved_lessons_by_user.jsonl
│   ├── retrieval_audit.jsonl
│   └── summary.json
│
├── llmrank_vanilla_history/
│   ├── per_user_results.jsonl
│   └── summary.json
│
└── llmrank_fmrec_history/
    ├── per_user_results.jsonl
    └── summary.json
```

## Troubleshooting

### `No module named src`

Start Jupyter from the FMRec repository root and make sure these files exist:

```text
src/__init__.py
src/data/__init__.py
src/data/dataset_base.py
```

### Negative-item preprocessing is slow

Prefer the frozen candidate file:

```text
results/books_candidates_1k_memrec_test.json
```

If the standalone `RecDataset` still automatically precomputes `user_negatives`, this can be disabled because the final FMRec reranker does not use that cache when frozen candidates are supplied.

### LLM is loaded repeatedly

Use `fmrec_retrieve_and_rank.ipynb`, load `UnslothBatchLLM` once, and reuse the same object for Vanilla, FMRec, different Top-K settings, and ablations.

### CUDA out of memory

Reduce:

```python
BATCH_SIZE = 4
```

or:

```python
BATCH_SIZE = 2
```

Keep:

```python
LOAD_IN_4BIT = True
```

### Resume an interrupted run

Use:

```python
fmrec_summary, fmrec_rows = run_rank(
    lessons_by_user=retrieved_lessons,
    output_dir="results/llmrank_fmrec_history",
    resume=True,
)
```

Users already present in `per_user_results.jsonl` are skipped.

## Recommended Workflow

```text
1. Start notebook
2. Load dataset/candidates once
3. Load LightGCN embeddings once
4. Load failure memory once
5. Load local LLM once
6. Retrieve lessons
7. Run FMRec
8. Change retrieval settings
9. Retrieve again
10. Run FMRec again
```

Only the retrieval and ranking steps need to be repeated for sensitivity experiments.
