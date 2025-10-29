import numpy as np
from typing import List, Set

def dcg_at_k(ranked_list: List[str], ground_truth: List[str], k: int) -> float:
    """Calculate DCG@K"""
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        if item in ground_truth:
            # Relevance score based on position in ground truth
            # Higher position in ground truth = higher relevance
            relevance = len(ground_truth) - ground_truth.index(item)
            dcg += relevance / np.log2(i + 2)  # i+2 because i is 0-indexed
    return dcg

def ndcg_at_k(ranked_list: List[str], ground_truth: List[str], k: int) -> float:
    """Calculate NDCG@K"""
    dcg = dcg_at_k(ranked_list, ground_truth, k)
    idcg = dcg_at_k(ground_truth, ground_truth, k)  # Ideal DCG
    return dcg / idcg if idcg > 0 else 0.0

def precision_at_k(ranked_list: List[str], ground_truth: Set[str], k: int) -> float:
    """Calculate Precision@K"""
    top_k = ranked_list[:k]
    relevant_in_top_k = sum(1 for item in top_k if item in ground_truth)
    return relevant_in_top_k / k if k > 0 else 0.0

def recall_at_k(ranked_list: List[str], ground_truth: Set[str], k: int) -> float:
    """Calculate Recall@K"""
    top_k = ranked_list[:k]
    relevant_in_top_k = sum(1 for item in top_k if item in ground_truth)
    return relevant_in_top_k / len(ground_truth) if len(ground_truth) > 0 else 0.0

def average_precision(ranked_list: List[str], ground_truth: Set[str]) -> float:
    """Calculate Average Precision"""
    relevant_count = 0
    precision_sum = 0.0
    
    for i, item in enumerate(ranked_list):
        if item in ground_truth:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i
    
    return precision_sum / len(ground_truth) if len(ground_truth) > 0 else 0.0

# Example usage
ground_truth_ranked = ["item_A", "item_B", "item_C", "item_D"]  # True relevant items
model_ranked = ["item_B", "item_X", "item_A", "item_Y", "item_C"]  # Model predictions

ground_truth_set = set(ground_truth_ranked)

# Calculate metrics
k = 5
print(f"NDCG@{k}: {ndcg_at_k(model_ranked, ground_truth_ranked, k):.3f}")
print(f"Precision@{k}: {precision_at_k(model_ranked, ground_truth_set, k):.3f}")
print(f"Recall@{k}: {recall_at_k(model_ranked, ground_truth_set, k):.3f}")
print(f"Average Precision: {average_precision(model_ranked, ground_truth_set):.3f}")