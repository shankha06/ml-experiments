import json
import random
import logging
from typing import List, Dict
from datetime import datetime
import os

# ---------------------------------------------------------
# Libraries
# pip install sentence-transformers datasets
# ---------------------------------------------------------
from sentence_transformers import SentenceTransformer, InputExample, losses, models, evaluation
from sentence_transformers.util import cos_sim
from sentence_transformers.datasets import NoDuplicatesDataLoader
from torch.utils.data import DataLoader, DistributedSampler
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_NAME = "BAAI/bge-base-en-v1.5"
TRAIN_BATCH_SIZE = 10  # Per GPU batch size
NUM_EPOCHS = 1
OUTPUT_PATH = "./output/fine-tuned-bge-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DATA_PATH = "my_dataset.json"

# BGE-specific instruction (Only add to queries/anchors, not passages)
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# ---------------------------------------------------------
# Distributed Training Setup
# ---------------------------------------------------------
def setup_distributed(rank, world_size):
    """Initialize distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',  # Use 'gloo' for CPU
        rank=rank,
        world_size=world_size
    )
    
    # Set device for this process
    torch.cuda.set_device(rank)
    
    if rank == 0:
        logger.info(f"Initialized DDP with {world_size} GPUs")

def cleanup_distributed():
    """Clean up distributed training environment."""
    dist.destroy_process_group()

# ---------------------------------------------------------
# 1. Data Loading & Processing
# ---------------------------------------------------------
def load_and_prepare_data(filepath: str, split_ratio: float = 0.8, rank: int = 0):
    """
    Loads JSON data and splits it into training InputExamples and 
    an InformationRetrievalEvaluator for testing.
    """
    if rank == 0:
        logger.info(f"Loading data from {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Shuffle data with a fixed seed for reproducibility across processes
    random.seed(42)
    random.shuffle(data)

    # Split index
    split_idx = int(len(data) * split_ratio)
    train_raw = data[:split_idx]
    test_raw = data[split_idx:]

    if rank == 0:
        logger.info(f"Data split: {len(train_raw)} training samples, {len(test_raw)} test samples.")

    # --- Prepare Training Data ---
    train_examples = []
    for item in train_raw:
        anchor = QUERY_INSTRUCTION + item['anchor']
        
        for pos in item['positives']:
            for neg in item['negatives']:
                train_examples.append(InputExample(texts=[anchor, pos, neg]))

    if rank == 0:
        logger.info(f"Generated {len(train_examples)} training triplets.")

    # --- Prepare Evaluation Data ---
    queries = {}
    corpus = {}
    relevant_docs = {}
    
    for idx, item in enumerate(test_raw):
        qid = f"q_{idx}"
        query_text = QUERY_INSTRUCTION + item['anchor']
        
        queries[qid] = query_text
        relevant_docs[qid] = set()

        # Process Positives
        for i, pos in enumerate(item['positives']):
            doc_id = f"doc_{idx}_p_{i}"
            corpus[doc_id] = pos
            relevant_docs[qid].add(doc_id)
        
        # Process Negatives
        for i, neg in enumerate(item['negatives']):
            doc_id = f"doc_{idx}_n_{i}"
            corpus[doc_id] = neg

    if rank == 0:
        logger.info(f"Evaluation set prepared: {len(queries)} queries, {len(corpus)} corpus documents.")
    
    return train_examples, queries, corpus, relevant_docs

# ---------------------------------------------------------
# 2. Main Training Pipeline with DDP
# ---------------------------------------------------------
def train_on_gpu(rank, world_size):
    """Training function for each GPU process."""
    
    # Setup distributed environment
    setup_distributed(rank, world_size)
    
    try:
        # 1. Load Data
        train_examples, queries, corpus, relevant_docs = load_and_prepare_data(
            DATA_PATH, rank=rank
        )
        
        # 2. Create DataLoader with DistributedSampler
        # DistributedSampler ensures each GPU gets different data
        train_sampler = DistributedSampler(
            train_examples,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42
        )
        
        # Use standard DataLoader with DistributedSampler
        train_dataloader = DataLoader(
            train_examples,
            batch_size=TRAIN_BATCH_SIZE,
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True
        )

        # 3. Initialize Model
        if rank == 0:
            logger.info(f"Loading base model: {MODEL_NAME}")
        
        model = SentenceTransformer(MODEL_NAME)
        
        # Move model to GPU
        model = model.to(rank)
        
        # Wrap model with DDP
        # find_unused_parameters=True may be needed for some models
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

        # 4. Define Loss
        # Pass the DDP-wrapped model's module
        train_loss = losses.MultipleNegativesRankingLoss(model=model.module)

        # 5. Define Evaluator (only on rank 0)
        evaluator = None
        if rank == 0:
            evaluator = evaluation.InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name='bge-finetune-eval',
                show_progress_bar=True,
                main_score_function='cosine'
            )

        # 6. Train
        if rank == 0:
            logger.info("Starting distributed training...")
        
        warmup_steps = int(len(train_dataloader) * NUM_EPOCHS * 0.1)

        # Custom training loop for better DDP control
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        model.train()
        global_step = 0
        
        for epoch in range(NUM_EPOCHS):
            # Set epoch for DistributedSampler to shuffle differently each epoch
            train_sampler.set_epoch(epoch)
            
            epoch_loss = 0
            for step, batch in enumerate(train_dataloader):
                # Move batch to device
                features = [model.module.tokenize(text) for text in batch]
                features = [{k: v.to(rank) for k, v in feature.items()} for feature in features]
                
                # Forward pass
                embeddings = [model.module(feature)['sentence_embedding'] for feature in features]
                
                # Calculate loss
                loss = train_loss(embeddings, labels=None)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if global_step < warmup_steps:
                    scheduler.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                # Log progress (only rank 0)
                if rank == 0 and step % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Step {step}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
                
                # Evaluation (only rank 0, every 50 steps)
                if rank == 0 and evaluator and global_step % 50 == 0:
                    model.eval()
                    eval_score = evaluator(model.module)
                    if rank == 0:
                        logger.info(f"Step {global_step} - Evaluation score: {eval_score}")
                    model.train()
            
            if rank == 0:
                avg_loss = epoch_loss / len(train_dataloader)
                logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # Final evaluation (only rank 0)
        if rank == 0 and evaluator:
            model.eval()
            final_score = evaluator(model.module)
            logger.info(f"Final evaluation score: {final_score}")
        
        # Save model (only rank 0)
        if rank == 0:
            model.module.save(OUTPUT_PATH)
            logger.info(f"Training finished. Model saved to {OUTPUT_PATH}")
    
    finally:
        # Clean up
        cleanup_distributed()

# ---------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------
def main():
    """Main function to launch distributed training."""
    
    # Get number of GPUs
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print(f"Warning: Only {world_size} GPU(s) available. DDP works best with 2+ GPUs.")
        print("Falling back to single GPU training.")
        if world_size == 1:
            train_on_gpu(0, 1)
        else:
            print("No GPU available. Please use GPU for training.")
        return
    
    print(f"Launching distributed training on {world_size} GPUs...")
    
    # Spawn processes for each GPU
    mp.spawn(
        train_on_gpu,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()