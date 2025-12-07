import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import argparse
import os
import torch

def prepare_mnrl_data(input_path, output_path, model_name='sentence-transformers/all-MiniLM-L6-v2', top_k_search=20, top_k_negatives=5):
    """
    Reads FAQ data, mines hard negatives using batch processing, and saves training data for MNRL.
    """
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return

    # Basic validation
    required_cols = ['articleID', 'content', 'similar_questions']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Dataset missing required columns: {required_cols}")
        return

    # Filter invalid rows early
    initial_len = len(df)
    # Ensure similar_questions is a list and not empty
    df = df[df['similar_questions'].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0)].copy()
    # Select anchor (first question)
    df['anchor'] = df['similar_questions'].apply(lambda x: x[0])
    # Reset index to ensure alignment with embeddings
    df.reset_index(drop=True, inplace=True)
    print(f"Filtered {initial_len - len(df)} invalid rows. Processing {len(df)} rows.")

    print(f"Loading model {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model.to(device)

    # 1. Encode Corpus (Content)
    print("Encoding corpus (content)...")
    corpus_embeddings = model.encode(
        df['content'].tolist(), 
        convert_to_tensor=True, 
        show_progress_bar=True,
        batch_size=32 # Adjustable batch size
    )

    # 2. Encode Anchors
    print("Encoding anchors...")
    anchor_embeddings = model.encode(
        df['anchor'].tolist(), 
        convert_to_tensor=True, 
        show_progress_bar=True,
        batch_size=32
    )
    
    # 3. Compute Anchor-Positive scores
    # Since df is aligned, anchor[i] corresponds to positive[i] which is corpus[i]
    # We want cosine similarity. Embeddings are usually normalized by SBERT by default? 
    # Not always, but usally utils.cos_sim does it.
    # Manual dot product for aligned tensors: (A * B).sum(axis=1)
    # But let's verify normalization. model.encode(normalize_embeddings=True) is safer if we just use dot product.
    # Let's re-encode with normalization just to be safe for dot product, OR use util.cos_sim pairwise.
    # util.pairwise_cos_sim is available in newer versions, scanning docs... `util.paired_cosine_similarity` ? 
    # Let's simple use torch logic:
    # Normalize manually to be safe
    anchor_embeddings = util.normalize_embeddings(anchor_embeddings)
    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)
    
    # Diagonal of the dot product matrix is not efficient to compute full matrix then diag.
    # Just row-wise dot product.
    pos_scores = (anchor_embeddings * corpus_embeddings).sum(dim=1)
    
    # 4. Batch Semantic Search
    print("Performing batch semantic search...")
    # query_chunk_size prevents OOM on large datasets
    search_results = util.semantic_search(
        anchor_embeddings, 
        corpus_embeddings, 
        top_k=top_k_search, 
        query_chunk_size=1000, 
        score_function=util.dot_score # Dot score on normalized = cosine
    )

    training_samples = []

    print("Post-processing results...")
    # Iterate through results (which matches df order)
    for idx, hits in enumerate(search_results):
        row = df.iloc[idx]
        article_id = row['articleID']
        anchor = row['anchor']
        positive = row['content']
        pos_score = pos_scores[idx].item()
        
        hard_negatives = []
        for hit in hits:
            hit_id = hit['corpus_id']
            score = hit['score']
            
            # Condition 1: Different articleID
            # Optimization: Pre-fetch articleIDs? accessing df.iloc might be slow in loop?
            # Accessing scalar from Series is fast enough for 90k, but list lookup is faster.
            # let's rely on df for now, if slow we can convert column to list.
            candidate_article_id = df.at[hit_id, 'articleID'] # .at is faster than .iloc
            
            if candidate_article_id == article_id:
                continue
                
            # Condition 2: Score > anchor_positive_score
            if score > pos_score:
                hard_negatives.append(df.at[hit_id, 'content'])
                
            if len(hard_negatives) >= top_k_negatives:
                break
        
        training_samples.append({
            'anchor': anchor,
            'positive': positive,
            'negatives': hard_negatives
        })

    # Save
    output_df = pd.DataFrame(training_samples)
    print(f"Saving {len(output_df)} samples to {output_path}...")
    output_df.to_parquet(output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/faq_dataset.parquet", help="Path to input parquet file")
    parser.add_argument("--output", type=str, default="data/mnrl_training_data.parquet", help="Path to output parquet file")
    args = parser.parse_args()
    
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
    prepare_mnrl_data(args.input, args.output)
