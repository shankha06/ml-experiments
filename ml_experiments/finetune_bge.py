import json
import logging
from typing import Dict, List

import torch
from FlagEmbedding import FlagLLM, FlagLLM_TrainingArguments
from torch.utils.data import Dataset
from transformers import HfArgumentParser
from tqdm import tqdm
import numpy as np


logger = logging.getLogger(__name__)


class TripletsDataset(Dataset):
    def __init__(self, data_path: str):
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        query = " ".join(self.data[item]["query"])
        pos = self.data[item]["similar_tags"]
        neg = self.data[item]["negative_tags"]
        return {"query": query, "pos": pos, "neg": neg}


def main():
    parser = HfArgumentParser((FlagLLM_TrainingArguments,))
    (training_args,) = parser.parse_args_into_dataclasses()
    training_args.n_gpu = 4

    model = FlagLLM(
        training_args.model_name_or_path,
        use_fp16=training_args.fp16,
    )

    train_dataset = TripletsDataset(data_path="triplets_dataset.json")

    model.train(
        dataset=train_dataset,
        training_args=training_args
    )

    if training_args.local_rank == 0:
        evaluate(model, train_dataset, training_args)


def evaluate(model: FlagLLM, dataset: TripletsDataset, training_args: FlagLLM_TrainingArguments, k_values: List[int] = [1, 3, 5, 10]):
    corpus: Dict[int, str] = {}
    queries: Dict[int, str] = {}
    relevant_docs: Dict[int, List[int]] = {}

    all_tags = set()
    for item in dataset.data:
        all_tags.update(item["similar_tags"])
        all_tags.update(item["negative_tags"])

    corpus = {i: tag for i, tag in enumerate(all_tags)}
    tag_to_id = {tag: i for i, tag in corpus.items()}
    id_to_tag = {i: tag for tag, i in tag_to_id.items()}

    for i, item in enumerate(dataset.data):
        query = " ".join(item["query"])
        queries[i] = query
        relevant_docs[i] = [
            tag_to_id[tag] for tag in item["similar_tags"] if tag in tag_to_id
        ]

    query_embeddings = model.encode([q for q in queries.values()])
    corpus_embeddings = model.encode([c for c in corpus.values()])

    cos_sim = torch.mm(torch.from_numpy(query_embeddings), torch.from_numpy(corpus_embeddings).T)

    results = {}
    for k in k_values:
        precision_at_k = []
        recall_at_k = []
        hitrate_at_k = []

        top_k = torch.topk(cos_sim, k=k, dim=1).indices.tolist()

        for i in range(len(queries)):
            retrieved_docs = top_k[i]
            relevant = relevant_docs[i]
            
            retrieved_relevant = len(set(retrieved_docs) & set(relevant))
            
            precision_at_k.append(retrieved_relevant / k)
            recall_at_k.append(retrieved_relevant / len(relevant) if relevant else 0)
            hitrate_at_k.append(1.0 if retrieved_relevant > 0 else 0.0)

        results[f"precision@{k}"] = np.mean(precision_at_k)
        results[f"recall@{k}"] = np.mean(recall_at_k)
        results[f"hitrate@{k}"] = np.mean(hitrate_at_k)

    logger.info("Evaluation results:")
    for metric, value in results.items():
        logger.info(f"{metric}: {value:.4f}")

    with open(f"{training_args.output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
