#!/usr/bin/env python3
"""Standard LightGCN pretraining for FMRec collaborative retrieval."""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.data import RecDataset
except ModuleNotFoundError as e:
    raise RuntimeError(
        "Place train_lightgcn.py inside <MemRec>/scripts/."
    ) from e


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_train_edges(
    dataset: RecDataset,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Set[int]]]:
    users, items = [], []
    train_pos: Dict[int, Set[int]] = {}

    for uid, item_list in dataset.train_data.items():
        uid = int(uid)
        train_pos[uid] = {int(i) for i in item_list}
        for iid in item_list:
            users.append(uid)
            items.append(int(iid))

    if not users:
        raise ValueError("No edges found in dataset.train_data")

    return (
        np.asarray(users, dtype=np.int64),
        np.asarray(items, dtype=np.int64),
        train_pos,
    )


def build_normalized_adj(
    n_users: int,
    n_items: int,
    users: np.ndarray,
    items: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """D^{-1/2} A D^{-1/2} on the user-item bipartite graph."""
    u = torch.from_numpy(users).long()
    i = torch.from_numpy(items).long() + n_users

    row = torch.cat([u, i])
    col = torch.cat([i, u])
    n_nodes = n_users + n_items

    degree = torch.bincount(row, minlength=n_nodes).float()
    inv_sqrt = degree.clamp_min(1.0).pow(-0.5)
    values = inv_sqrt[row] * inv_sqrt[col]

    adj = torch.sparse_coo_tensor(
        torch.stack([row, col]), values,
        size=(n_nodes, n_nodes), dtype=torch.float32,
    ).coalesce()

    return adj.to(device)


def sample_negatives(
    edge_users: np.ndarray,
    train_pos: Dict[int, Set[int]],
    n_items: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One negative per positive edge; exclude TRAIN positives only."""
    negatives = np.empty(len(edge_users), dtype=np.int64)
    user_to_indices: Dict[int, List[int]] = {}

    for idx, uid in enumerate(edge_users):
        user_to_indices.setdefault(int(uid), []).append(idx)

    for uid, indices in user_to_indices.items():
        positives = train_pos[uid]
        if len(positives) >= n_items:
            raise ValueError(f"User {uid} has no available negative item")

        sampled = []
        while len(sampled) < len(indices):
            need = len(indices) - len(sampled)
            draws = rng.integers(0, n_items, size=max(16, need * 2))
            for iid in draws:
                iid = int(iid)
                if iid not in positives:
                    sampled.append(iid)
                    if len(sampled) == len(indices):
                        break

        negatives[np.asarray(indices)] = np.asarray(sampled, dtype=np.int64)

    return negatives


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, n_layers: int):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim)

        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def propagate(self, adj: torch.Tensor):
        x = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        layers = [x]

        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            layers.append(x)

        final = torch.stack(layers, dim=0).mean(dim=0)
        return final[:self.n_users], final[self.n_users:]


def train(args) -> None:
    seed_all(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    dataset = RecDataset(args.data_path, seed=args.seed)
    edge_users, edge_items, train_pos = collect_train_edges(dataset)

    n_users, n_items = int(dataset.n_users), int(dataset.n_items)
    print(
        f"Device={device}; users={n_users}; items={n_items}; "
        f"train_edges={len(edge_users)}"
    )

    adj = build_normalized_adj(
        n_users, n_items, edge_users, edge_items, device
    )

    model = LightGCN(
        n_users, n_items, args.embedding_dim, args.n_layers
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    users_t = torch.from_numpy(edge_users).long().to(device)
    pos_items_t = torch.from_numpy(edge_items).long().to(device)
    rng = np.random.default_rng(args.seed)

    pbar = tqdm(range(1, args.epochs + 1), desc="Train LightGCN")

    for _epoch in pbar:
        model.train()
        optimizer.zero_grad()

        neg_items = sample_negatives(
            edge_users, train_pos, n_items, rng
        )
        neg_items_t = torch.from_numpy(neg_items).long().to(device)

        user_emb, item_emb = model.propagate(adj)
        u = user_emb[users_t]
        pos = item_emb[pos_items_t]
        neg = item_emb[neg_items_t]

        pos_score = (u * pos).sum(dim=1)
        neg_score = (u * neg).sum(dim=1)
        bpr = F.softplus(neg_score - pos_score).mean()

        ego_u = model.user_embedding(users_t)
        ego_pos = model.item_embedding(pos_items_t)
        ego_neg = model.item_embedding(neg_items_t)
        reg = (
            ego_u.pow(2).sum(dim=1)
            + ego_pos.pow(2).sum(dim=1)
            + ego_neg.pow(2).sum(dim=1)
        ).mean()

        loss = bpr + args.reg * reg
        loss.backward()
        optimizer.step()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}", bpr=f"{bpr.item():.4f}"
        )

    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.propagate(adj)
        user_emb = F.normalize(user_emb, p=2, dim=1).cpu()
        item_emb = F.normalize(item_emb, p=2, dim=1).cpu()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    payload = {
        "users": {
            str(uid): user_emb[uid].tolist()
            for uid in range(n_users)
        },
        "items": {
            str(iid): item_emb[iid].tolist()
            for iid in range(n_items)
        },
        "meta": {
            "method": "LightGCN",
            "graph_split": "train_only",
            "negative_exclusion": "train_positives_only",
            "embedding_dim": args.embedding_dim,
            "n_layers": args.n_layers,
            "epochs": args.epochs,
            "lr": args.lr,
            "reg": args.reg,
            "seed": args.seed,
            "final_embedding": "mean(layer_0,...,layer_L)",
            "l2_normalized": True,
        },
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    if args.checkpoint_path:
        os.makedirs(os.path.dirname(args.checkpoint_path) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "meta": payload["meta"],
            },
            args.checkpoint_path,
        )

    print(f"Saved embeddings: {args.output_path}")
    if args.checkpoint_path:
        print(f"Saved checkpoint: {args.checkpoint_path}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_path",
                   default="data/processed/instructrec-books/instructrec-books.inter")
    p.add_argument("--output_path",
                   default="results/lightgcn_books/lightgcn_embeddings.json")
    p.add_argument("--checkpoint_path",
                   default="results/lightgcn_books/lightgcn.pt")

    p.add_argument("--embedding_dim", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reg", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
