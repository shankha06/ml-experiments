import json
import random
import itertools
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers
# Enum for distance metric
from sentence_transformers.losses import TripletDistanceMetric

def load_and_prepare_data(json_file_path):
    """
    Parses JSON and creates explicit (Anchor, Positive, Negative) triplets.
    """
    with open(json_file_path, 'r') as f:
        raw_data = json.load(f)

    triplets = []
    
    for entry in raw_data:
        query = entry.get("query")
        positives = entry.get("positive_sentence", [])
        negatives = entry.get("hard_negative_sentence", [])
        
        if not query or not positives or not negatives:
            continue
            
        # Pair every positive with a hard negative (cycling if necessary)
        for pos, neg in zip(positives, itertools.cycle(negatives)):
            triplets.append({
                "anchor": query,
                "positive": pos,
                "negative": neg
            })

    # Shuffle to ensure NO_DUPLICATES sampler doesn't discard data unnecessarily
    random.shuffle(triplets)
    return triplets

def train_model(json_file_path, model_id="all-MiniLM-L6-v2", output_path="./output_cosine_triplet"):
    # 1. Load Data
    triplet_data = load_and_prepare_data(json_file_path)
    train_dataset = Dataset.from_list(triplet_data)
    print(f"Loaded {len(train_dataset)} triplets.")
    
    # 2. Load Model
    model = SentenceTransformer(model_id)
    
    # 3. Define Loss: TripletLoss with COSINE
    # Cosine Distance = 1 - Cosine Similarity.
    # We want: Distance(A, P) < Distance(A, N) - Margin
    # Which implies: Sim(A, P) > Sim(A, N) + Margin
    loss = losses.TripletLoss(
        model=model, 
        distance_metric=TripletDistanceMetric.COSINE, 
        margin=0.3  # Reduced margin (0.1 - 0.5 is typical for Cosine)
    )

    # 4. Define Training Arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES, 
    )

    # 5. Initialize Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    # 6. Train
    trainer.train()
    model.save_pretrained(output_path)
    print(f"Training finished. Model saved to {output_path}")

if __name__ == "__main__":
    # Dummy data generation for testing
    dummy_data = [
        {
            "query": "best running shoes",
            "positive_sentence": ["Nike Pegasus reviews", "Adidas Ultraboost sale"],
            "hard_negative_sentence": ["installing python", "how to bake cake"]
        }
    ]
    with open("dummy_data_cosine.json", "w") as f:
        json.dump(dummy_data, f)
        
    train_model("dummy_data_cosine.json")