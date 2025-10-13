"""
Two-stage cascade training on personalization dataset with rich logging.

Stage 1: FIT-inspired Two-Tower Retrieval
- Retrieves top candidates using learnable interaction modules instead of simple dot products
- Pre-computes nav link embeddings offline for speed (during serving; computed on-the-fly here)
- Trained with in-batch negatives (InfoNCE)

Stage 2: HoME-style Multi-Task Ranker
- Ranks top candidates with rich features
- Multi-task: CTR, task completion, cross-sell
- Hierarchical experts with learnable gating to reduce task interference

This script adds a full training pipeline that consumes:
  data/personalization/training_dataset.json
and trains both stages end-to-end with simple text/user encoders.
"""

import json
import math
import os
import re
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# rich logging
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

console = Console()

# -----------------------------
# Utility: device & helpers
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------
# Simple text processing
# -----------------------------

def simple_tokenize(text: str) -> List[str]:
    # Lowercase and split on non-alphanumeric boundaries
    text = text.lower()
    return [t for t in re.split(r"[^a-z0-9]+", text) if t]


class Vocabulary:
    def __init__(self, min_freq: int = 1, max_size: int = 20000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.token_to_id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.freq: Dict[str, int] = {}
        self.finalized = False

    def add_tokens(self, tokens: List[str]):
        if self.finalized:
            return
        for t in tokens:
            self.freq[t] = self.freq.get(t, 0) + 1

    def finalize(self):
        if self.finalized:
            return
        # sort by frequency then lexicographically
        items = [(t, c) for t, c in self.freq.items() if c >= self.min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))
        for t, _ in items[: self.max_size - len(self.token_to_id)]:
            self.token_to_id[t] = len(self.token_to_id)
        self.finalized = True

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.token_to_id.get("<UNK>", 1)
        return [self.token_to_id.get(t, unk) for t in tokens]

    @property
    def size(self) -> int:
        return len(self.token_to_id)


# -----------------------------
# Dataset
# -----------------------------

class PersonalizationDataset(Dataset):
    def __init__(self, json_path: str, max_query_len: int = 16, max_item_len: int = 16):
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Build vocab from queries and nav_link URL tokens
        self.query_vocab = Vocabulary()
        self.item_vocab = Vocabulary()
        self.user_to_id: Dict[str, int] = {}

        for ex in raw:
            q_tokens = simple_tokenize(ex.get("query", ""))
            self.query_vocab.add_tokens(q_tokens)
            # tokenize URL path components
            item_tokens = simple_tokenize(ex.get("nav_link", ""))
            self.item_vocab.add_tokens(item_tokens)
            # user mapping
            uid = ex.get("user_id", "unknown")
            if uid not in self.user_to_id:
                self.user_to_id[uid] = len(self.user_to_id)

        self.query_vocab.finalize()
        self.item_vocab.finalize()

        self.max_query_len = max_query_len
        self.max_item_len = max_item_len
        self.examples = []

        for ex in raw:
            q_tokens = simple_tokenize(ex.get("query", ""))
            i_tokens = simple_tokenize(ex.get("nav_link", ""))
            q_ids = self._pad(self.query_vocab.encode(q_tokens), self.max_query_len)
            i_ids = self._pad(self.item_vocab.encode(i_tokens), self.max_item_len)
            uid = self.user_to_id[ex.get("user_id", "unknown")]

            label = float(ex.get("label", 0))
            # derive completion from dwell_time if present
            dwell = float(ex.get("dwell_time", 0.0) or 0.0)
            completed = 1.0 if (label > 0.5 and dwell >= 60.0) else 0.0
            # cross-sell not available -> assume 0
            crosssell = 0.0

            propensity = float(ex.get("propensity", 0.5) or 0.5)
            position = float(ex.get("position", 1) or 1)

            self.examples.append({
                "user_id": uid,
                "query_ids": q_ids,
                "item_ids": i_ids,
                "clicked": label,
                "completed": completed,
                "crosssell": crosssell,
                "propensity": propensity,
                "position": position,
                "dwell_time": dwell,
            })

        # dataset summary table
        table = Table(title="Dataset Summary", title_style="bold cyan")
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("Examples", str(len(self.examples)))
        table.add_row("Users", str(len(self.user_to_id)))
        table.add_row("Query Vocab Size", str(self.query_vocab.size))
        table.add_row("Item Vocab Size", str(self.item_vocab.size))
        table.add_row("Max Query Len", str(self.max_query_len))
        table.add_row("Max Item Len", str(self.max_item_len))
        console.print(table)

    def _pad(self, ids: List[int], max_len: int) -> List[int]:
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [0] * (max_len - len(ids))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "user_id": torch.tensor(ex["user_id"], dtype=torch.long),
            "query_ids": torch.tensor(ex["query_ids"], dtype=torch.long),
            "item_ids": torch.tensor(ex["item_ids"], dtype=torch.long),
            "clicked": torch.tensor([ex["clicked"]], dtype=torch.float32),
            "completed": torch.tensor([ex["completed"]], dtype=torch.float32),
            "crosssell": torch.tensor([ex["crosssell"]], dtype=torch.float32),
            "propensity": torch.tensor([ex["propensity"]], dtype=torch.float32),
            "position": torch.tensor([ex["position"]], dtype=torch.float32),
            "dwell_time": torch.tensor([ex["dwell_time"]], dtype=torch.float32),
        }


# -----------------------------
# Encoders
# -----------------------------

class MeanEmbedding(nn.Module):
    """Average pooling over token embeddings with padding mask (id=0)."""
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, L]
        embs = self.emb(token_ids)  # [B, L, D]
        mask = (token_ids != 0).float()  # [B, L]
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
        pooled = (embs * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, D]
        return pooled


class TransformerBlock(nn.Module):
    """A lightweight MLP-based block used as a stand-in for a transformer on vector inputs."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Stage 1: Two-Tower Retrieval
# -----------------------------

class TwoTowerRetrieval(nn.Module):
    """
    Retrieves candidates with a learnable interaction scorer (FIT-inspired).
    """
    def __init__(self, user_dim=512, item_dim=256, embedding_dim=128):
        super().__init__()
        # User Tower
        self.user_encoder = nn.Sequential(
            nn.Linear(user_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )
        # Query Encoder
        self.query_encoder = TransformerBlock(
            input_dim=768, hidden_dim=256, output_dim=128, num_heads=4
        )
        # Item Tower
        self.item_encoder = nn.Sequential(
            nn.Linear(item_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )
        # Learnable interaction
        self.row_fc = nn.Linear(embedding_dim, embedding_dim)
        self.col_fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, user_features, query_features, item_features):
        # Pairwise matching for aligned triples (same index)
        user_emb = self.user_encoder(user_features)
        query_emb = self.query_encoder(query_features)
        combined_emb = user_emb + 0.5 * query_emb
        item_emb = self.item_encoder(item_features)
        row_transformed = self.row_fc(combined_emb)
        col_transformed = self.col_fc(item_emb)
        scores = torch.sum(row_transformed * col_transformed, dim=-1)
        return scores  # [B]

    def get_user_embedding(self, user_features, query_features):
        user_emb = self.user_encoder(user_features)
        query_emb = self.query_encoder(query_features)
        return user_emb + 0.5 * query_emb  # [B, D]

    def get_item_embeddings(self, all_items):
        return self.item_encoder(all_items)  # [N, D]

    def pairwise_scores(self, user_query_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """Compute all-pairs scores between user_query_emb [B, D] and item_emb [N, D]."""
        r = self.row_fc(user_query_emb)  # [B, D]
        c = self.col_fc(item_emb)        # [N, D]
        return torch.matmul(r, c.transpose(0, 1))  # [B, N]


# -----------------------------
# Stage 2: HoME-style Ranker
# -----------------------------

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)  # [B, E]
        return F.softmax(logits, dim=-1)  # [B, E]


class TaskTower(nn.Module):
    def __init__(self, input_dim: int, num_experts: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class MultiTaskRanker(nn.Module):
    def __init__(self, input_dim=1024, num_experts=6):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.global_experts = nn.ModuleList([Expert(512, 256) for _ in range(2)])
        self.nav_experts = nn.ModuleList([Expert(512, 256) for _ in range(2)])
        self.conversion_experts = nn.ModuleList([Expert(512, 256) for _ in range(2)])

        self.ctr_tower = TaskTower(256, num_experts=num_experts)
        self.completion_tower = TaskTower(256, num_experts=num_experts)
        self.crosssell_tower = TaskTower(256, num_experts=num_experts)

        self.ctr_gate = GatingNetwork(512, num_experts=num_experts)
        self.completion_gate = GatingNetwork(512, num_experts=num_experts)
        self.crosssell_gate = GatingNetwork(512, num_experts=num_experts)

    def forward(self, features):
        shared_features = self.feature_extractor(features)  # [B, 512]
        global_outputs = [exp(shared_features) for exp in self.global_experts]
        nav_outputs = [exp(shared_features) for exp in self.nav_experts]
        conv_outputs = [exp(shared_features) for exp in self.conversion_experts]
        all_expert_outputs = torch.stack(global_outputs + nav_outputs + conv_outputs)  # [E, B, 256]

        # CTR
        ctr_weights = self.ctr_gate(shared_features)  # [B, E]
        ctr_combined = torch.einsum('be,ebd->bd', ctr_weights, all_expert_outputs)  # [B, 256]
        ctr_pred = self.ctr_tower(ctr_combined)
        # Completion
        completion_weights = self.completion_gate(shared_features)  # [B, E]
        completion_combined = torch.einsum('be,ebd->bd', completion_weights, all_expert_outputs)
        completion_pred = self.completion_tower(completion_combined)
        # Cross-sell
        crosssell_weights = self.crosssell_gate(shared_features)  # [B, E]
        crosssell_combined = torch.einsum('be,ebd->bd', crosssell_weights, all_expert_outputs)
        crosssell_pred = self.crosssell_tower(crosssell_combined)

        return {
            "ctr": ctr_pred,  # [B,1]
            "completion": completion_pred,  # [B,1]
            "crosssell": crosssell_pred,  # [B,1]
        }


# -----------------------------
# Losses with debiasing
# -----------------------------

class UnbiasedRankingLoss:
    def __init__(self, task_weights: Dict[str, float] = None):
        if task_weights is None:
            task_weights = {"ctr": 1.0, "completion": 0.5, "crosssell": 0.3}
        self.task_weights = task_weights

    def compute_loss(self, predictions: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor], propensity_scores: torch.Tensor):
        losses = {}
        ctr_loss = self.snips_bce_loss(predictions["ctr"], labels["clicked"], propensity_scores)
        losses["ctr"] = ctr_loss

        # Only compute completion for clicked examples
        mask = (labels["clicked"].squeeze(-1) >= 0.5)
        if mask.any():
            completion_loss = F.binary_cross_entropy(
                predictions["completion"][mask], labels["completed"][mask]
            )
            losses["completion"] = completion_loss
        else:
            losses["completion"] = torch.tensor(0.0, device=predictions["ctr"].device)

        crosssell_loss = self.ips_bce_loss(predictions["crosssell"], labels["crosssell_converted"], propensity_scores)
        losses["crosssell"] = crosssell_loss

        total_loss = sum(self.task_weights[k] * v for k, v in losses.items())
        return total_loss, losses

    def snips_bce_loss(self, pred: torch.Tensor, target: torch.Tensor, propensity: torch.Tensor) -> torch.Tensor:
        weights = 1.0 / propensity.clamp(min=1e-3)
        normalized_weights = weights / weights.sum().clamp(min=1e-6)
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        return (normalized_weights * bce).sum()

    def ips_bce_loss(self, pred: torch.Tensor, target: torch.Tensor, propensity: torch.Tensor) -> torch.Tensor:
        weights = 1.0 / propensity.clamp(min=1e-3)
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        return (weights * bce).mean()


# -----------------------------
# Offline evaluation utilities
# -----------------------------

@torch.no_grad()
def evaluate_offline(
    dataset: PersonalizationDataset,
    user_table: nn.Embedding,
    query_text: nn.Module,
    item_text: nn.Module,
    retrieval: TwoTowerRetrieval,
    user_to_256: nn.Module,
    query_to_256: nn.Module,
    ranker_proj: nn.Module,
    ranker: MultiTaskRanker,
    topk_list: List[int] = [10, 50, 100],
    batch_eval_size: int = 512,
):
    # switch to eval
    user_table.eval(); query_text.eval(); item_text.eval()
    retrieval.eval(); ranker.eval(); ranker_proj.eval(); user_to_256.eval(); query_to_256.eval()

    retr_metrics = _evaluate_retrieval(dataset, user_table, query_text, item_text, retrieval, topk_list)
    rank_metrics = _evaluate_ranker(dataset, user_table, query_text, item_text, user_to_256, query_to_256, ranker_proj, ranker, batch_eval_size)

    # Retrieval table
    rtable = Table(title="Retrieval Metrics", title_style="bold green")
    rtable.add_column("Metric", style="bold")
    rtable.add_column("Value")
    for k in topk_list:
        rtable.add_row(f"Recall@{k}", f"{retr_metrics.get(f'recall@{k}', float('nan')):.4f}")
    rtable.add_row("MRR", f"{retr_metrics.get('mrr', float('nan')):.4f}")
    rtable.add_row("NDCG@10", f"{retr_metrics.get('ndcg@10', float('nan')):.4f}")
    console.print(rtable)

    # Ranker (CTR)
    ctable = Table(title="Ranker Metrics (CTR)", title_style="bold cyan")
    ctable.add_column("Metric", style="bold")
    ctable.add_column("Value")
    ctable.add_row("AUC", f"{rank_metrics['ctr']['auc']:.4f}")
    ctable.add_row("LogLoss", f"{rank_metrics['ctr']['logloss']:.4f}")
    ctable.add_row("Accuracy@0.5", f"{rank_metrics['ctr']['acc']:.4f}")
    console.print(ctable)

    # Completion (clicked subset)
    ptable = Table(title="Ranker Metrics (Completion, clicked subset)", title_style="bold magenta")
    ptable.add_column("Metric", style="bold")
    ptable.add_column("Value")
    ptable.add_row("AUC", f"{rank_metrics['completion']['auc']:.4f}")
    ptable.add_row("LogLoss", f"{rank_metrics['completion']['logloss']:.4f}")
    ptable.add_row("Accuracy@0.5", f"{rank_metrics['completion']['acc']:.4f}")
    console.print(ptable)

    # switch back to train
    user_table.train(); query_text.train(); item_text.train()
    retrieval.train(); ranker.train(); ranker_proj.train(); user_to_256.train(); query_to_256.train()


def _evaluate_retrieval(
    dataset: PersonalizationDataset,
    user_table: nn.Embedding,
    query_text: nn.Module,
    item_text: nn.Module,
    retrieval: TwoTowerRetrieval,
    topk_list: List[int],
):
    # unique items
    key_to_idx: Dict[Tuple[int, ...], int] = {}
    unique_item_ids: List[List[int]] = []
    for ex in dataset.examples:
        key = tuple(ex["item_ids"]) if isinstance(ex["item_ids"], list) else tuple(ex["item_ids"])  # ensure tuple
        if key not in key_to_idx:
            key_to_idx[key] = len(unique_item_ids)
            unique_item_ids.append(list(key))
    if len(unique_item_ids) == 0:
        return {m: float("nan") for m in ["mrr", "ndcg@10"] + [f"recall@{k}" for k in topk_list]}

    item_ids_tensor = torch.tensor(unique_item_ids, dtype=torch.long, device=device)  # [M, L]
    item_feat = item_text(item_ids_tensor)  # [M, 256]
    item_emb = retrieval.get_item_embeddings(item_feat)  # [M, D]

    # positives
    pos_user_ids = []
    pos_query_ids = []
    target_indices = []
    for ex in dataset.examples:
        if ex["clicked"] >= 0.5:
            pos_user_ids.append(ex["user_id"])  # int
            pos_query_ids.append(ex["query_ids"])  # list[int]
            key = tuple(ex["item_ids"]) if isinstance(ex["item_ids"], list) else tuple(ex["item_ids"])  # ensure
            target_indices.append(key_to_idx[key])

    if len(pos_user_ids) == 0:
        return {m: float("nan") for m in ["mrr", "ndcg@10"] + [f"recall@{k}" for k in topk_list]}

    pos_user_ids_t = torch.tensor(pos_user_ids, dtype=torch.long, device=device)
    pos_query_ids_t = torch.tensor(pos_query_ids, dtype=torch.long, device=device)
    targets_t = torch.tensor(target_indices, dtype=torch.long, device=device)

    B = 256
    ranks: List[int] = []
    recalls = {k: 0 for k in topk_list}
    for i0 in range(0, pos_user_ids_t.size(0), B):
        i1 = min(i0 + B, pos_user_ids_t.size(0))
        u_feat = user_table(pos_user_ids_t[i0:i1])  # [b,512]
        q_feat = query_text(pos_query_ids_t[i0:i1])  # [b,768]
        uq = retrieval.get_user_embedding(u_feat, q_feat)  # [b,D]
        scores = retrieval.pairwise_scores(uq, item_emb)  # [b, M]
        sorted_idx = torch.argsort(scores, dim=1, descending=True)
        for row in range(sorted_idx.size(0)):
            tgt = targets_t[i0 + row].item()
            pos_in_row = (sorted_idx[row] == tgt).nonzero(as_tuple=False)
            if pos_in_row.numel() == 0:
                r = scores.size(1)
            else:
                r = int(pos_in_row.item()) + 1  # 1-based
            ranks.append(r)
            for k in topk_list:
                if r <= k:
                    recalls[k] += 1

    n = len(ranks)
    mrr = sum(1.0 / r for r in ranks) / max(1, n)
    ndcg10 = sum((1.0 / math.log2(r + 1)) if r <= 10 else 0.0 for r in ranks) / max(1, n)
    metrics = {f"recall@{k}": recalls[k] / max(1, n) for k in topk_list}
    metrics.update({"mrr": mrr, "ndcg@10": ndcg10})
    return metrics


def _binary_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    y_true = y_true.detach().cpu().flatten()
    y_score = y_score.detach().cpu().flatten()
    n = y_true.numel()
    n_pos = int((y_true > 0.5).sum().item())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(y_score)  # ascending
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, n + 1, dtype=torch.float)
    sum_pos = ranks[y_true > 0.5].sum().item()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _logloss(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.detach().cpu().flatten()
    p = y_pred.detach().cpu().flatten().clamp(1e-7, 1 - 1e-7)
    loss = - (y_true * torch.log(p) + (1 - y_true) * torch.log(1 - p)).mean()
    return float(loss.item())


@torch.no_grad()
def _evaluate_ranker(
    dataset: PersonalizationDataset,
    user_table: nn.Embedding,
    query_text: nn.Module,
    item_text: nn.Module,
    user_to_256: nn.Module,
    query_to_256: nn.Module,
    ranker_proj: nn.Module,
    ranker: MultiTaskRanker,
    batch_eval_size: int = 512,
):
    loader = DataLoader(dataset, batch_size=batch_eval_size, shuffle=False, drop_last=False)
    ctr_scores = []
    ctr_labels = []
    comp_scores = []
    comp_labels = []
    for batch in loader:
        user_ids = batch["user_id"].to(device)
        q_ids = batch["query_ids"].to(device)
        i_ids = batch["item_ids"].to(device)
        clicked = batch["clicked"].to(device)
        completed = batch["completed"].to(device)
        propensity = batch["propensity"].to(device)
        position = batch["position"].to(device)
        dwell = batch["dwell_time"].to(device)

        user_feat = user_table(user_ids)
        query_feat = query_text(q_ids)
        item_feat = item_text(i_ids)

        u256 = user_to_256(user_feat)
        q256 = query_to_256(query_feat)
        i256 = item_feat
        pos_norm = (position - 1.0) / 9.0
        prop = propensity
        dwell_norm = torch.clamp(dwell / 180.0, 0.0, 1.0)
        numerics = torch.cat([pos_norm, prop, dwell_norm], dim=1)

        ranker_in = torch.cat([u256, q256, i256, numerics], dim=1)
        ranker_feat = ranker_proj(ranker_in)
        preds = ranker(ranker_feat)

        ctr_scores.append(preds["ctr"].detach().cpu())
        ctr_labels.append(clicked.detach().cpu())

        mask = (clicked >= 0.5).squeeze(1)
        if mask.any():
            comp_scores.append(preds["completion"][mask].detach().cpu())
            comp_labels.append(completed[mask].detach().cpu())

    ctr_scores_t = torch.cat(ctr_scores, dim=0)
    ctr_labels_t = torch.cat(ctr_labels, dim=0)
    ctr_auc = _binary_auc(ctr_labels_t, ctr_scores_t)
    ctr_logloss = _logloss(ctr_labels_t, ctr_scores_t)
    ctr_acc = float(((ctr_scores_t >= 0.5).float() == ctr_labels_t).float().mean().item())

    if len(comp_scores) > 0:
        comp_scores_t = torch.cat(comp_scores, dim=0)
        comp_labels_t = torch.cat(comp_labels, dim=0)
        comp_auc = _binary_auc(comp_labels_t, comp_scores_t)
        comp_logloss = _logloss(comp_labels_t, comp_scores_t)
        comp_acc = float(((comp_scores_t >= 0.5).float() == comp_labels_t).float().mean().item())
    else:
        comp_auc = float('nan'); comp_logloss = float('nan'); comp_acc = float('nan')

    return {
        "ctr": {"auc": ctr_auc, "logloss": ctr_logloss, "acc": ctr_acc},
        "completion": {"auc": comp_auc, "logloss": comp_logloss, "acc": comp_acc},
    }


# -----------------------------
# Training harness
# -----------------------------

class TwoStageTrainer:
    def __init__(
        self,
        dataset: PersonalizationDataset,
        batch_size: int = 128,
        lr_retrieval: float = 2e-3,
        lr_ranker: float = 2e-3,
        temperature: float = 0.05,
        verbose: bool = True,
    ):
        self.dataset = dataset
        self.temperature = temperature
        self.verbose = verbose

        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # Encoders for inputs
        self.user_table = nn.Embedding(len(dataset.user_to_id), 512).to(device)
        self.query_text = MeanEmbedding(dataset.query_vocab.size, 768).to(device)
        self.item_text = MeanEmbedding(dataset.item_vocab.size, 256).to(device)

        # Stage 1 retrieval model
        self.retrieval = TwoTowerRetrieval(user_dim=512, item_dim=256, embedding_dim=128).to(device)

        # Feature projection for ranker
        self.user_to_256 = nn.Linear(512, 256).to(device)
        self.query_to_256 = nn.Linear(768, 256).to(device)
        # items already 256 via item_text
        ranker_in_dim = 256 + 256 + 256 + 3  # user + query + item + 3 numeric features
        self.ranker_proj = nn.Sequential(
            nn.Linear(ranker_in_dim, 1024),
            nn.ReLU(),
        ).to(device)
        self.ranker = MultiTaskRanker(input_dim=1024, num_experts=6).to(device)

        # Optimizers
        self.opt_retrieval = torch.optim.Adam(
            list(self.user_table.parameters())
            + list(self.query_text.parameters())
            + list(self.item_text.parameters())
            + list(self.retrieval.parameters()),
            lr=lr_retrieval,
        )
        self.opt_ranker = torch.optim.Adam(
            list(self.user_table.parameters())
            + list(self.query_text.parameters())
            + list(self.item_text.parameters())
            + list(self.user_to_256.parameters())
            + list(self.query_to_256.parameters())
            + list(self.ranker_proj.parameters())
            + list(self.ranker.parameters()),
            lr=lr_ranker,
        )

        self.ranking_loss = UnbiasedRankingLoss()

        # Model summary
        if self.verbose:
            table = Table(title="Model Components", title_style="bold magenta")
            table.add_column("Component", style="bold")
            table.add_column("Params")
            table.add_row("user_table", f"{count_parameters(self.user_table):,}")
            table.add_row("query_text", f"{count_parameters(self.query_text):,}")
            table.add_row("item_text", f"{count_parameters(self.item_text):,}")
            table.add_row("retrieval", f"{count_parameters(self.retrieval):,}")
            table.add_row("user_to_256", f"{count_parameters(self.user_to_256):,}")
            table.add_row("query_to_256", f"{count_parameters(self.query_to_256):,}")
            table.add_row("ranker_proj", f"{count_parameters(self.ranker_proj):,}")
            table.add_row("ranker", f"{count_parameters(self.ranker):,}")
            total = sum(
                count_parameters(m)
                for m in (
                    self.user_table,
                    self.query_text,
                    self.item_text,
                    self.retrieval,
                    self.user_to_256,
                    self.query_to_256,
                    self.ranker_proj,
                    self.ranker,
                )
            )
            console.print(table)
            console.print(Panel.fit(f"[bold green]Total Trainable Parameters:[/] {total:,}", title="Summary"))

    def train(self, epochs: int = 3):
        for epoch in range(1, epochs + 1):
            total_retrieval_loss = 0.0
            total_ranker_loss = 0.0

            for step, batch in enumerate(track(self.loader, description=f"Epoch {epoch}/{epochs}"), 1):
                # Move batch
                user_ids = batch["user_id"].to(device)
                q_ids = batch["query_ids"].to(device)
                i_ids = batch["item_ids"].to(device)
                clicked = batch["clicked"].to(device)
                completed = batch["completed"].to(device)
                crosssell = batch["crosssell"].to(device)
                propensity = batch["propensity"].to(device)
                position = batch["position"].to(device)
                dwell = batch["dwell_time"].to(device)

                # ---------------------
                # Stage 1: Retrieval
                # ---------------------
                self.opt_retrieval.zero_grad(set_to_none=True)
                # Build features
                user_feat = self.user_table(user_ids)  # [B,512]
                query_feat = self.query_text(q_ids)    # [B,768]
                item_feat = self.item_text(i_ids)      # [B,256]

                # Compute pairwise logits for in-batch negatives (InfoNCE)
                uq_emb = self.retrieval.get_user_embedding(user_feat, query_feat)  # [B, D]
                item_emb = self.retrieval.get_item_embeddings(item_feat)           # [B, D]

                logits = self.retrieval.pairwise_scores(uq_emb, item_emb) / self.temperature  # [B, B]
                targets = torch.arange(logits.size(0), device=logits.device)
                retrieval_loss = F.cross_entropy(logits, targets)
                retrieval_loss.backward()
                self.opt_retrieval.step()

                # Recompute features to build a fresh graph for the ranker stage
                user_feat = self.user_table(user_ids)  # [B,512]
                query_feat = self.query_text(q_ids)    # [B,768]
                item_feat = self.item_text(i_ids)      # [B,256]

                # ---------------------
                # Stage 2: Ranker
                # ---------------------
                self.opt_ranker.zero_grad(set_to_none=True)
                # Feature assembly for rich ranker input
                u256 = self.user_to_256(user_feat)
                q256 = self.query_to_256(query_feat)
                i256 = item_feat  # already 256
                # Normalize numeric features (rough scaling)
                pos_norm = (position - 1.0) / 9.0  # position 1-10 -> 0..1
                prop = propensity  # already probability-like
                dwell_norm = torch.clamp(dwell / 180.0, 0.0, 1.0)
                numerics = torch.cat([pos_norm, prop, dwell_norm], dim=1)  # [B,3]

                ranker_in = torch.cat([u256, q256, i256, numerics], dim=1)  # [B, 771]
                ranker_feat = self.ranker_proj(ranker_in)  # [B, 1024]
                preds = self.ranker(ranker_feat)

                labels = {
                    "clicked": clicked,
                    "completed": completed,
                    "crosssell_converted": crosssell,
                }
                ranker_loss, _ = self.ranking_loss.compute_loss(preds, labels, propensity)
                ranker_loss.backward()
                self.opt_ranker.step()

                total_retrieval_loss += retrieval_loss.item()
                total_ranker_loss += ranker_loss.item()

                # occasional step logs
                if self.verbose and (step == 1 or step % 50 == 0):
                    console.log(
                        f"[epoch {epoch} step {step}] retrieval_loss={retrieval_loss.item():.4f} | ranker_loss={ranker_loss.item():.4f}"
                    )

            n_batches = len(self.loader)
            avg_retrieval = total_retrieval_loss / n_batches
            avg_ranker = total_ranker_loss / n_batches
            console.print(
                Panel.fit(
                    f"[bold]Epoch {epoch} Summary[/]\n"
                    f"[green]Retrieval loss[/]: {avg_retrieval:.4f}\n"
                    f"[cyan]Ranker loss[/]: {avg_ranker:.4f}",
                    title=f"Epoch {epoch}",
                    border_style="blue",
                )
            )

            # Offline evaluation on current model state
            evaluate_offline(
                dataset=self.dataset,
                user_table=self.user_table,
                query_text=self.query_text,
                item_text=self.item_text,
                retrieval=self.retrieval,
                user_to_256=self.user_to_256,
                query_to_256=self.query_to_256,
                ranker_proj=self.ranker_proj,
                ranker=self.ranker,
                topk_list=[10, 50, 100],
                batch_eval_size=512,
            )


# -----------------------------
# Entry point
# -----------------------------

def train_from_json(json_path: str, epochs: int = 3, batch_size: int = 128):
    console.print(Panel.fit(f"Training dataset path:\n[bold]{json_path}[/]", title="Config", border_style="cyan"))
    dataset = PersonalizationDataset(json_path)
    trainer = TwoStageTrainer(dataset=dataset, batch_size=batch_size)
    trainer.train(epochs=epochs)


if __name__ == "__main__":
    # Default path as requested
    default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "personalization", "training_dataset.json")
    # Fallback to relative path if resolution fails
    if not os.path.exists(default_path):
        default_path = os.path.join("data", "personalization", "training_dataset.json")
    console.print(Panel.fit(f"[bold green]Training with dataset[/]: {default_path}", title="Start", border_style="green"))
    train_from_json(default_path, epochs=3, batch_size=128)
