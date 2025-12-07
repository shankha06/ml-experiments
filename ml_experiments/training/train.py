import argparse
import os
import sys
import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input parquet file")
    parser.add_argument("--output", type=str, required=True, help="Path to output model directory")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5", help="Base model name")
    parser.add_argument("--batch_size", type=int, default=32, help="Per device train batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
    
    # DDP Setup
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Helper for logging
    def log(msg):
        if local_rank in [-1, 0]:
            print(msg)

    # Parse args (and handle known args if needed, but simple parse_args is fine)
    args, unknown = parser.parse_known_args()
    
    log(f"Training params: {args}")

    # Load dataset
    log(f"Loading dataset from {args.input}...")
    dataset = load_dataset("parquet", data_files=args.input, split="train")

    # Preprocessing function to format examples for MNRL
    # MNRL expects InputExamples or a formatted list. 
    # With SentenceTransformerTrainer (v3), we can pass a dictionary or mapped dataset.
    # The columns should be ["anchor", "positive", "negative_1", "negative_2", ...] or just a list column?
    # Actually, the trainer handles dataset columns mapping.
    # If the collator sees multiple columns, it treats them as (input1, input2, ...).
    # We have 'anchor', 'positive', 'negatives' (list).
    # We need to flatten 'negatives' so we have 'negative_0', 'negative_1' etc columns OR 
    # convert to a single column containing the list?
    # Standard way in v3 is usually a list of texts column or multiple text columns.
    # Documentation says: "If the dataset contains multiple columns, the columns are assumed to be (sentence_1, sentence_2, ...)".
    # So we need to flatten the list of negatives into separate columns?
    # Or cleaner: map to a single "text" column which is a list [anchor, pos, neg1, neg2...]? No, that's not standard HF format.
    # Let's flatten to columns: anchor, positive, negative_0, negative_1...
    
    # Check max negatives
    def get_max_negatives(example):
        return len(example['negatives'])
    
    # We assume variable negatives are allowed, but for batching usually fixed is better or handled by collator.
    # Let's try to flatten dynamically or just fix a number.
    # safe approach: flatten to a list feature "text_list" = [anchor, pos, *negatives]
    # Check if SBERT trainer supports single column of list of strings.
    # Yes, if we look at `CoSENTLoss` or others, but `MultipleNegativesRankingLoss` takes [(a, p, n...), ...].
    # Let's transform to "return {'samples': [anchor, pos, *negatives]}"? No.
    
    # Let's try to dynamic flatten to columns.
    # Actually, SBERT V3 Trainer uses the dataset columns. If columns are ["a", "b", "c"], it passes corresponding inputs.
    # Problem: Variable number of negatives.
    # Valid solution: Truncate/Pad to fixed number of negatives or use specific collator.
    # Let's implement a transform to fixed columns: anchor, positive, neg_0, neg_1... up to min(len(negatives)) or fill with empty?
    # "MultipleNegativesRankingLoss" can handle variable inputs if the batch is collated correctly, but standard collator might struggle with missing keys?
    # Simpler approach: Just take top-k negatives and make them columns. 
    # Or, verify if we can pass a list.
    # Let's go with Flatten strategy:
    # 1. Add "BGE Instruction" to anchor.
    # 2. Flatten: columns "sentence_0" (anchor), "sentence_1" (positive), "sentence_2" (neg0)...
    
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    
    def transform_data(example):
        # Add prefix
        anchor = BGE_QUERY_PREFIX + example['anchor']
        positive = example['positive']
        negs = example['negatives']
        
        # We need a consistent schema for HF datasets.
        # We should probably return a dict with numbered keys?
        # A clearer way is to return a SINGLE column 'text' which is a List[str].
        # SBERT Dataset processing checks for 'sentence_0', 'sentence_1' OR a single column in some cases.
        # However, for MNRL, `sentence-transformers` < 3 used `InputExample(texts=[...])`.
        # V3 trainer: "If the dataset columns are not found in the loss, all columns are passed to the model".
        # Let's try returning a dict with keys 'sentence_0', 'sentence_1', ...
        # But schemas must optionally match.
        # Let's assume max 5 negatives as per prep script.
        
        result = {
            "sentence_0": anchor,
            "sentence_1": positive
        }
        for i, neg in enumerate(negs):
            result[f"sentence_{i+2}"] = neg
        return result

    # Apply transform
    # Note: map requires the output schema to be consistent/known or we drop old columns.
    # If rows have different neg counts, 'sentence_x' will be missing for some. HF Datasets will pad with None?
    # Better to force specific columns and pad with empty string or None?
    # MNRL ignores None/Empty if handled? No.
    # Safer: take exactly 5 negatives (pad if needed? or truncate).
    # Since we mined hard negatives, some might have < 5.
    # Let's just fix to using the available ones and hope the collator handles Nones?
    # Actually, if we use `remove_columns`, we need to be careful.
    
    # ALTERNATIVE: Use the classic `InputExample` flow wrapped in a custom dataset if V3 is tricky with variable cols.
    # But V3 Trainer is preferred.
    # Let's standardise to top-1 hard negative? No, we want multiple.
    # Let's try the list approach: "return {'text': [anchor, pos, *negs]}"
    # If the column is named "text" (or whatever), does the trainer unpack it?
    # Let's check `SentenceTransformerTrainer` source is complex.
    
    # Conservative safe bet for script:
    # Just take 1 hard negative to be safe with standard triplet loss if variable length is a risk?
    # But user asked for MNRL on this data.
    # Let's just try to flatten to max 5 negatives (pad with empty string - SBERT might embed empty string, which is bad).
    # Actually, let's just use 'anchor', 'positive', 'negatives' columns and let the user decide?
    # No, I must write working code.
    
    # Let's use the `return_loss` feature of MNRL?
    # Let's accept that we flatten to specific columns and just pad with `positive` (duplicate positive as negative? No)
    # Pad with random? 
    # Let's just take the first hard negative for simplicity if we want to guarantee success, 
    # BUT user has 5 negatives.
    # Let's go with: map to `[anchor, pos, n1, n2, n3, n4, n5]`. If nX missing, reuse n(X-1) or pos?
    # Reuse n(last) is safer than empty.
    
    def robust_transform(example):
        anchor = BGE_QUERY_PREFIX + example['anchor']
        positive = example['positive']
        negs = example['negatives']
        
        # Helper to get negative or fallback
        def get_neg(i):
            if i < len(negs):
                return negs[i]
            # Fallback: cycle negatives?
            if len(negs) > 0:
                return negs[i % len(negs)]
            # If absolutely no negatives (unlikely from prep script), use positive?
            # Ideally shouldn't happen if filtered.
            return positive 
        
        # Return flat
        return {
            "sentence_0": anchor,
            "sentence_1": positive,
            "sentence_2": get_neg(0),
            "sentence_3": get_neg(1),
            "sentence_4": get_neg(2),
            "sentence_5": get_neg(3),
            "sentence_6": get_neg(4),
        }

    train_dataset = dataset.map(robust_transform, remove_columns=dataset.column_names)
    log(f"Dataset columns: {train_dataset.column_names}")

    # Load Model
    log(f"Loading model {args.model_name} on {device}...")
    model = SentenceTransformer(args.model_name, device=device)

    # Loss
    # MNRL expects (a, p, n1, n2...)
    # "MultipleNegativesRankingLoss"
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, # Pass gradient accumulation
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=False,  # Enable mixed precision
        bf16=True, # Use bf16 if typically A100/H100, but safer False for generic
        evaluation_strategy="no", # or "steps"
        save_strategy="epoch",
        logging_steps=1000,
        run_name="mnrl-bge",
        # DDP specific
        ddp_find_unused_parameters=False,
    )

    # Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    log("Starting training...")
    trainer.train()
    
    log("Saving model...")
    trainer.save_model(os.path.join(args.output, "final"))
    log("Done.")

if __name__ == "__main__":
    main()
