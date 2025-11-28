import json
import random
import logging
from typing import List, Dict
from datetime import datetime

# ---------------------------------------------------------
# Libraries
# pip install sentence-transformers datasets
# ---------------------------------------------------------
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
from sentence_transformers.util import cos_sim
from sentence_transformers.datasets import NoDuplicatesBatchSampler # Added import
from torch.utils.data import DataLoader
import torch

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_NAME = "BAAI/bge-base-en-v1.5"
TRAIN_BATCH_SIZE = 10 # Higher is better for MultipleNegativesRankingLoss (provides more in-batch negatives)
NUM_EPOCHS = 1
OUTPUT_PATH = "./output/fine-tuned-bge-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATA_PATH = "my_dataset.json" # Path to your json file

# BGE-specific instruction (Only add to queries/anchors, not passages)
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# ---------------------------------------------------------
# 1. Data Loading & Processing
# ---------------------------------------------------------
def load_and_prepare_data(filepath: str, split_ratio: float = 0.8):
    """
    Loads JSON data and splits it into training InputExamples and 
    an InformationRetrievalEvaluator for testing.
    """
    logger.info(f"Loading data from {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Shuffle data
    random.shuffle(data)

    # Split index
    split_idx = int(len(data) * split_ratio)
    train_raw = data[:split_idx]
    test_raw = data[split_idx:]

    logger.info(f"Data split: {len(train_raw)} training samples, {len(test_raw)} test samples.")

    # --- Prepare Training Data ---
    # We flatten the list: If an anchor has multiple positives/negatives, 
    # we create multiple triplets or select 1 positive and 1 hard negative per anchor.
    # MultipleNegativesRankingLoss expects: [anchor, positive, negative]
    
    train_examples = []
    for item in train_raw:
        anchor = QUERY_INSTRUCTION + item['anchor'] # Add BGE instruction
        
        # Strategy: Create a triplet for every combination of pos/neg 
        # (Be careful if lists are huge, you might want to sample instead)
        for pos in item['positives']:
            for neg in item['negatives']:
                train_examples.append(InputExample(texts=[anchor, pos, neg]))

    logger.info(f"Generated {len(train_examples)} training triplets.")

    # --- Prepare Evaluation Data ---
    # InformationRetrievalEvaluator requires dictionaries mapping IDs to texts
    
    queries = {}       # qid -> query_text
    corpus = {}        # doc_id -> doc_text
    relevant_docs = {} # qid -> set([doc_id_1, doc_id_2])
    
    for idx, item in enumerate(test_raw):
        qid = f"q_{idx}"
        query_text = QUERY_INSTRUCTION + item['anchor'] # Add BGE instruction
        
        queries[qid] = query_text
        relevant_docs[qid] = set()

        # Process Positives
        for i, pos in enumerate(item['positives']):
            doc_id = f"doc_{idx}_p_{i}"
            corpus[doc_id] = pos
            relevant_docs[qid].add(doc_id)
        
        # Process Negatives (Add to corpus so the model has 'distractors' to choose from)
        for i, neg in enumerate(item['negatives']):
            doc_id = f"doc_{idx}_n_{i}"
            corpus[doc_id] = neg
            # Do NOT add to relevant_docs

    logger.info(f"Evaluation set prepared: {len(queries)} queries, {len(corpus)} corpus documents.")
    
    return train_examples, queries, corpus, relevant_docs

# ---------------------------------------------------------
# 2. Main Training Pipeline
# ---------------------------------------------------------
def main():
    # 1. Load Data
    train_examples, queries, corpus, relevant_docs = load_and_prepare_data(DATA_PATH)
    
    # 2. Create DataLoader with NoDuplicatesBatchSampler
    # We use a special sampler to ensure that if an anchor has multiple positives (different triplets),
    # they DO NOT appear in the same batch. This prevents the model from treating a valid positive
    # as a negative (False Negative) during in-batch negative calculation.
    train_batch_sampler = NoDuplicatesBatchSampler(train_examples, batch_size=TRAIN_BATCH_SIZE)
    train_dataloader = DataLoader(train_examples, batch_sampler=train_batch_sampler)

    # 3. Initialize Model
    logger.info(f"Loading base model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 4. Define Loss
    # MultipleNegativesRankingLoss is ideal here. 
    # It calculates loss = -log(exp(sim(a,p)) / (exp(sim(a,p)) + sum(exp(sim(a,n_i)))))
    # It uses the provided 'negative' PLUS all other positives in the batch as negatives.
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 5. Define Evaluator
    # name argument helps identify the file in output
    # This calculates Recall@k, Precision@k, MRR@k, NDCG@k
    evaluator = evaluation.InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name='bge-finetune-eval',
        show_progress_bar=True,
        main_score_function='cosine'
    )

    # 6. Train
    logger.info("Starting training...")
    
    # Warmup steps usually 10% of training data
    warmup_steps = int(len(train_dataloader) * NUM_EPOCHS * 0.1)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=NUM_EPOCHS,
        evaluation_steps=50, # Evaluate every 50 steps (adjust based on dataset size)
        warmup_steps=warmup_steps,
        output_path=OUTPUT_PATH,
        save_best_model=True,
        show_progress_bar=True
    )

    logger.info(f"Training finished. Model saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
        
    main()