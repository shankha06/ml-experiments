import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator
from torch.utils.data import DataLoader
from typing import List, Dict, Optional
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def prepare_dataset(data: List[Dict[str, str]]) -> List[InputExample]:
    """
    Convert dataset from list of dicts to InputExample format.
    
    Args:
        data: List of dicts with 'anchor' and 'positive' keys
        
    Returns:
        List of InputExample objects
    """
    examples = []
    for item in data:
        examples.append(InputExample(
            texts=[item['anchor'], item['positive']]
        ))
    return examples


def train_sentence_transformer(
    model_name: str,
    train_data: List[Dict[str, str]],
    eval_data: Optional[List[Dict[str, str]]] = None,
    output_path: str = './output/sentence-transformer',
    epochs: int = 1,
    batch_size: int = 16,
    warmup_steps: int = 100,
    evaluation_steps: int = 500,
    learning_rate: float = 2e-5,
    use_amp: bool = True,
    save_best_model: bool = True
):
    """
    Train a SentenceTransformer model with Multiple Negatives Ranking Loss.
    
    Args:
        model_name: Name or path of the base model (e.g., 'bert-base-uncased')
        train_data: Training data as list of dicts with 'anchor' and 'positive' keys
        eval_data: Optional evaluation data in same format
        output_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        warmup_steps: Number of warmup steps for learning rate scheduler
        evaluation_steps: Evaluate model every N steps
        learning_rate: Learning rate for optimizer
        use_amp: Use automatic mixed precision (faster training)
        save_best_model: Save the best model based on evaluation
        
    Returns:
        Trained SentenceTransformer model
    """
    
    logging.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Prepare training data
    logging.info(f"Preparing {len(train_data)} training examples")
    train_examples = prepare_dataset(train_data)
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )
    
    # Set up Multiple Negatives Ranking Loss
    # This loss uses in-batch negatives: for each anchor-positive pair,
    # all other positives in the batch serve as negatives
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Prepare evaluation
    evaluator = None
    if eval_data is not None:
        logging.info(f"Preparing {len(eval_data)} evaluation examples")
        eval_examples = prepare_dataset(eval_data)
        
        # Extract sentences for evaluation
        sentences1 = [ex.texts[0] for ex in eval_examples]
        sentences2 = [ex.texts[1] for ex in eval_examples]
        # Scores of 1.0 because these are positive pairs
        scores = [1.0] * len(eval_examples)
        
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1,
            sentences2,
            scores,
            name='eval',
            show_progress_bar=True
        )
    
    # Calculate training steps
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * epochs
    
    logging.info(f"Training configuration:")
    logging.info(f"  Epochs: {epochs}")
    logging.info(f"  Batch size: {batch_size}")
    logging.info(f"  Steps per epoch: {steps_per_epoch}")
    logging.info(f"  Total steps: {total_steps}")
    logging.info(f"  Warmup steps: {warmup_steps}")
    logging.info(f"  Learning rate: {learning_rate}")
    logging.info(f"  Evaluation steps: {evaluation_steps}")
    
    # Train the model
    logging.info("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        evaluation_steps=evaluation_steps,
        optimizer_params={'lr': learning_rate},
        use_amp=use_amp,
        save_best_model=save_best_model,
        show_progress_bar=True
    )
    
    logging.info(f"Training complete! Model saved to {output_path}")
    return model


def evaluate_model(
    model: SentenceTransformer,
    eval_data: List[Dict[str, str]],
    batch_size: int = 32
) -> Dict[str, float]:
    """
    Evaluate a trained SentenceTransformer model.
    
    Args:
        model: Trained SentenceTransformer model
        eval_data: Evaluation data as list of dicts with 'anchor' and 'positive' keys
        batch_size: Batch size for encoding
        
    Returns:
        Dictionary with evaluation metrics
    """
    logging.info(f"Evaluating model on {len(eval_data)} examples")
    
    eval_examples = prepare_dataset(eval_data)
    
    sentences1 = [ex.texts[0] for ex in eval_examples]
    sentences2 = [ex.texts[1] for ex in eval_examples]
    scores = [1.0] * len(eval_examples)
    
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1,
        sentences2,
        scores,
        batch_size=batch_size,
        name='final_eval',
        show_progress_bar=True
    )
    
    result = evaluator(model)
    
    logging.info("Evaluation results:")
    logging.info(f"  Cosine Similarity - Pearson: {result:.4f}")
    
    return {"cosine_pearson": result}


# Example usage
if __name__ == "__main__":
    # Example dataset
    train_data = [
        {
            "anchor": "A man is eating food.",
            "positive": "A man is eating a piece of bread."
        },
        {
            "anchor": "The dog is playing in the garden.",
            "positive": "A dog is playing outside."
        },
        {
            "anchor": "A woman is cutting vegetables.",
            "positive": "Someone is preparing food."
        },
        # Add more examples...
    ]
    
    eval_data = [
        {
            "anchor": "A person is riding a bike.",
            "positive": "Someone is cycling."
        },
        {
            "anchor": "The cat is sleeping on the couch.",
            "positive": "A cat is resting on furniture."
        },
        # Add more examples...
    ]
    
    # Train the model
    model = train_sentence_transformer(
        model_name='sentence-transformers/all-MiniLM-L6-v2',  # or any base model
        train_data=train_data,
        eval_data=eval_data,
        output_path='./trained_model',
        epochs=3,
        batch_size=16,
        warmup_steps=100,
        evaluation_steps=100,
        learning_rate=2e-5
    )
    
    # Evaluate the final model
    results = evaluate_model(model, eval_data)
    print(f"\nFinal evaluation results: {results}")