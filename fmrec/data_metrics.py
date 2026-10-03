"""Data loading and ranking metrics (Hit/Recall/NDCG)."""

import os
import json
import numpy as np
from typing import List, Dict

from fmrec.common import item_category, item_title


def save_all_users_ranking_results(all_results: List[Dict],
                                  items_meta: Dict,
                                  output_file: str = "all_users_ranking_results.json"):
    """
    Lưu toàn bộ kết quả ranking của tất cả users vào 1 file JSON duy nhất.
    
    Args:
        all_results: List các dict chứa thông tin của từng user
        items_meta: Metadata items để lấy title, category,...
        output_file: Tên file output (sẽ tự tạo thư mục nếu cần)
    """
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    def get_item_info(item_id: str) -> Dict:
        if item_id in items_meta:
            info = items_meta[item_id]
            return {
                "item_id": item_id,
                "title": item_title(info, item_id),
                "category": item_category(info),
                # "brand": info.get("brand", ""),
                # "price": info.get("price", None)
            }
        else:
            return {
                "item_id": item_id,
                "title": f"Unknown Item {item_id}",
                "category": "Unknown",
                # "brand": "",
                # "price": None
            }
    
    # Chuyển đổi chi tiết items cho tất cả users
    final_results = []
    for res in all_results:
        user_result = {
            "user_id": res["user_id"],
            "num_candidates": len(res["candidates"]),
            "ground_truth_item_ids": res["ground_truth"],
            "candidate_item_ids": res["candidates"],
            "reranked_item_ids": res["predictions"],
            # "ground_truth_items": [get_item_info(iid) for iid in res["ground_truth"]],
            "candidate_items": [get_item_info(iid) for iid in res["candidates"]],
            "reranked_items": [get_item_info(iid) for iid in res["predictions"]],
            "metrics": res["metrics"],  # thêm metrics của user này
            "baseline_metrics": res["baseline_metrics"]
        }
        final_results.append(user_result)
    
    # Lưu vào 1 file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved ranking results of {len(final_results)} users to {output_file}")
    print(f"   File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")


def load_data(items_path: str, sequences_path: str, negatives_path: str):
    """Load Amazon dataset"""
    print("Loading data...")
    
    with open(items_path, 'r', encoding="utf-8") as f:
        items_meta = json.load(f)
    for item_id, item_info in list(items_meta.items()):
        if not isinstance(item_info, dict):
            item_info = {"title": str(item_info)}
            items_meta[item_id] = item_info
        item_info["title"] = item_title(item_info, str(item_id))
        item_info["main_cat"] = item_category(item_info)
        item_info["category_normalized"] = True
    
    with open(sequences_path, 'r', encoding="utf-8") as f:
        user_sequences = json.load(f)
    
    with open(negatives_path, 'r', encoding="utf-8") as f:
        user_negatives = json.load(f)
    
    print(f"Loaded {len(items_meta)} items")
    print(f"Loaded {len(user_sequences)} users")
    
    return items_meta, user_sequences, user_negatives


def calculate_recall_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
    """Calculate Recall@K"""
    top_k = predictions[:k]
    hits = len(set(top_k) & set(ground_truth))
    return hits / len(ground_truth) if ground_truth else 0.0


def calculate_ndcg_at_k(predictions: List[str], ground_truth: List[str], k: int) -> float:
    """Calculate NDCG@K"""
    top_k = predictions[:k]
    
    # DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    
    # IDCG
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k))])
    
    return dcg / idcg if idcg > 0 else 0.0


PAPER_METRIC_KS = (1, 3, 5, 10, 20)


PAPER_METRIC_NAMES = tuple(
    metric_name
    for k in PAPER_METRIC_KS
    for metric_name in (f"hit@{k}", f"recall@{k}", f"ndcg@{k}")
)


def calculate_paper_ranking_metrics(
    predictions: List[str],
    ground_truth: List[str],
) -> Dict[str, float]:
    """Return paper metrics without requiring another ranking run."""
    metrics: Dict[str, float] = {}
    for k in PAPER_METRIC_KS:
        recall = calculate_recall_at_k(predictions, ground_truth, k)
        metrics[f"hit@{k}"] = 1.0 if recall > 0.0 else 0.0
        metrics[f"recall@{k}"] = recall
        metrics[f"ndcg@{k}"] = calculate_ndcg_at_k(predictions, ground_truth, k)
    return metrics
