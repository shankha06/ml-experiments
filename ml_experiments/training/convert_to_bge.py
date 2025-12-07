import argparse
import pandas as pd
import json
import os
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input parquet file")
    parser.add_argument("--output", type=str, required=True, help="Output jsonl file")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading {args.input}...")
    # Using datasets library for efficient loading if parquet is large, or pandas
    # Let's use datasets to be consistent
    dataset = load_dataset("parquet", data_files=args.input, split="train")

    print(f"Converting to BGE format...")
    # BGE format: {"query": str, "pos": List[str], "neg": List[str]}
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for example in dataset:
            query = example['anchor']
            # Ensure "Represent this sentence..." prefix is handled by BGE training script usually via arguments, 
            # BUT BGE documentation often says to add it to query for retrieval tasks.
            # The standard BGE training script `run.py` has `--query_instruction_for_retrieval`.
            # So we typically pass raw query and let script handle it.
            
            # Format
            obj = {
                "query": query,
                "pos": [example['positive']],
                "neg": example['negatives']
            }
            f.write(json.dumps(obj) + "\n")
            
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
