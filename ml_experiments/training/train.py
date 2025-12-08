import os
import argparse
import logging
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
import numpy as np
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_ddp():
    """Initialize Distributed Data Parallel"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        # Fallback for single GPU/CPU debugging
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.warning("Not running in DDP mode. Using single device.")
    
    return device, rank, local_rank, world_size

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

class MNRLDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, max_length=512, num_negatives=5):
        self.data = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.bge_query_prefix = "Represent this sentence for searching relevant passages: "
        
        # Simple validation
        if 'anchor' not in self.data.columns or 'positive' not in self.data.columns:
            raise ValueError("Dataset must contain 'anchor' and 'positive' columns")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        anchor_text = self.bge_query_prefix + row['anchor']
        positive_text = row['positive']
        
        # Handle negatives
        negatives = row['negatives'] if 'negatives' in row and isinstance(row['negatives'], (list, np.ndarray)) else []
        
        if isinstance(negatives, np.ndarray):
            negatives = negatives.tolist()
        
        # Pad or truncate negatives to fixed size
        if len(negatives) < self.num_negatives:
            # Cycle negatives if not enough, or use positive if completely empty (shouldn't happen with filtered data)
            if len(negatives) > 0:
                negatives = negatives * math.ceil(self.num_negatives / len(negatives))
                negatives = negatives[:self.num_negatives]
            else:
                negatives = [positive_text] * self.num_negatives # Fallback
                
        elif len(negatives) > self.num_negatives:
            negatives = negatives[:self.num_negatives]
            
            
        return {
            'anchor': anchor_text,
            'positive': positive_text,
            'negatives': negatives # List of strings
        }

def collate_fn(batch, tokenizer, max_length, device):
    """
    Tokenizes and prepares batch.
    Returns tensors on CPU; move to GPU in loop to avoid multiprocessing CUDA fork issues if num_workers > 0.
    """
    anchors = [b['anchor'] for b in batch]
    positives = [b['positive'] for b in batch]
    negatives_list = [b['negatives'] for b in batch] # List of Lists
    
    # Flatten negatives for tokenization: [batch_size * num_negatives]
    flat_negatives = [n for negs in negatives_list for n in negs]
    
    # Check num_negatives consistency per batch (should be fixed by Dataset)
    num_negatives = len(negatives_list[0]) 

    # Tokenize
    def tokenize(texts):
        return tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
    
    anchor_tok = tokenize(anchors)
    positive_tok = tokenize(positives)
    # Tokenize all negatives at once is usually more efficient if batch size strictly controlled
    # If OOM, might need to chunk? With batch_size ~48 and 5 negs, it's like batch 300. Fits on A10G.
    negative_tok = tokenize(flat_negatives)
    
    return {
        'anchor': anchor_tok,
        'positive': positive_tok,
        'negatives': negative_tok, # Flat tokenized
        'num_negatives': num_negatives
    }

class EncoderModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        
        # Enable gradient checkpointing if memory is tight
        self.model.gradient_checkpointing_enable() 

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # BGE uses [CLS] token embedding
        return output.last_hidden_state[:, 0]

def compute_mnrl_loss(anchor_emb, positive_emb, negative_emb, num_negatives, temperature=0.02):
    """
    Args:
        anchor_emb: [B, D]
        positive_emb: [B, D]
        negative_emb: [B * K, D] (Flat negatives)
        num_negatives: K
        
    Multiple Negatives Ranking Loss:
    - In-batch negatives: The positive of other samples are negatives.
    - Hard negatives: Explicit negatives provided.
    
    Standard MNRL creates a big candidate list.
    Scores = Anchor @ Candidates.T
    Candidates = [Positives, Negatives]
    """
    batch_size = anchor_emb.size(0)
    
    # Reshape negatives to [B, K, D]
    negative_emb = negative_emb.view(batch_size, num_negatives, -1)
    
    # Targets: The positive for anchor i is at index i in the candidates
    # We want to maximize sim(a_i, p_i) and minimize sim(a_i, others)
    
    # 1. Similarity with Positives (Batch x Batch)
    # scores_pos[i][j] = sim(anchor_i, positive_j)
    # We want diagonal to be high. Off-diagonal are "easy" negatives (in-batch).
    sim_pos = torch.matmul(anchor_emb, positive_emb.transpose(0, 1)) / temperature # [B, B]
    
    # 2. Similarity with Hard Negatives
    # We need to flatten negatives again to [B*K, D] effectively for matrix mul, OR:
    # We want sim(anchor_i, neg_i_k).
    # But usually MNRL treats ALL negatives as candidates? 
    # Standard implementation: Candidates = [P_0, ..., P_B, N_0_0, ..., N_B_K]
    # This creates a massive (B) x (B + B*K) matrix.
    
    flat_negatives = negative_emb.reshape(-1, anchor_emb.size(1)) # [B*K, D]
    sim_neg = torch.matmul(anchor_emb, flat_negatives.transpose(0, 1)) / temperature # [B, B*K]
    
    # Concatenate scores
    # Final Scores: [B, B + B*K]
    scores = torch.cat([sim_pos, sim_neg], dim=1)
    
    # Target is 0, 1, 2... B-1 (the index of the positive in the first B columns)
    labels = torch.arange(batch_size, device=anchor_emb.device)
    
    loss = nn.CrossEntropyLoss()(scores, labels)
    return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to training data")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--num_negatives", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=500)
    args = parser.parse_args()

    device, rank, local_rank, world_size = setup_ddp()
    
    if rank == 0:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        logger.info(f"Arguments: {args}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Dataset & Dataloader
    dataset = MNRLDataset(args.input, tokenizer, num_negatives=args.num_negatives)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, 512, device)
    )

    # Model
    model = EncoderModel(args.model_name)
    model.to(device)
    
    # Wrap DDP
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Scheduler
    num_training_steps = len(dataloader) * args.epochs // args.accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * num_training_steps), 
        num_training_steps=num_training_steps
    )
    
    start_epoch = 0
    global_step = 0
    
    # Training Loop
    if rank == 0:
        logger.info("Starting training...")
    
    model.train()
    
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        epoch_iterator = tqdm(dataloader, disable=(rank != 0))
        
        for step, batch in enumerate(epoch_iterator):
            # Move to device
            anchor_inputs = {k: v.to(device) for k, v in batch['anchor'].items()}
            positive_inputs = {k: v.to(device) for k, v in batch['positive'].items()}
            negative_inputs = {k: v.to(device) for k, v in batch['negatives'].items()}
            
            # Forward
            # Enable autocast for bf16
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                anchor_emb = model(anchor_inputs['input_ids'], anchor_inputs['attention_mask'])
                positive_emb = model(positive_inputs['input_ids'], positive_inputs['attention_mask'])
                negative_emb = model(negative_inputs['input_ids'], negative_inputs['attention_mask'])
                
                # Normalize embeddings for cosine similarity
                anchor_emb = torch.nn.functional.normalize(anchor_emb, p=2, dim=1)
                positive_emb = torch.nn.functional.normalize(positive_emb, p=2, dim=1)
                negative_emb = torch.nn.functional.normalize(negative_emb, p=2, dim=1)
                
                loss = compute_mnrl_loss(anchor_emb, positive_emb, negative_emb, batch['num_negatives'])
                loss = loss / args.accumulation_steps

            # Backward
            loss.backward()
            
            if (step + 1) % args.accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if rank == 0 and global_step % 100 == 0:
                    # Sync loss for logging? Or just log local for speed
                    # Simple local log usually enough to sanity check
                    epoch_iterator.set_description(f"Loss: {loss.item() * args.accumulation_steps:.4f}")
                
                # Save checkpoint
                if rank == 0 and global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    
                    unwrapped_model = model.module if hasattr(model, "module") else model
                    unwrapped_model.model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    logger.info(f"Saved checkpoint to {save_path}")

    # Final Save
    if rank == 0:
        save_path = os.path.join(args.output, "final")
        os.makedirs(save_path, exist_ok=True)
        unwrapped_model = model.module if hasattr(model, "module") else model
        unwrapped_model.model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"Training complete. Model saved to {save_path}")

    cleanup_ddp()

if __name__ == "__main__":
    main()
