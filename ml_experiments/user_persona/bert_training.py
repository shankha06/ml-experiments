import torch
import torch.distributed as dist
import os
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import SentenceEvaluator
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
from tqdm import tqdm

# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


class InBatchNegativesEvaluator(SentenceEvaluator):
    """
    Custom evaluator that treats all other samples in the batch as negatives.
    This mimics the Multiple Negatives Ranking Loss evaluation strategy.
    
    For each anchor-positive pair in a batch:
    - The correct positive should rank #1
    - All other positives in the batch are treated as negatives
    
    Metrics computed:
    - Accuracy@1: How often the correct positive ranks first
    - Accuracy@k: How often the correct positive is in top-k
    - MRR (Mean Reciprocal Rank): Average of 1/rank of correct positive
    """
    
    def __init__(
        self,
        anchors: List[str],
        positives: List[str],
        batch_size: int = 32,
        name: str = 'in_batch_negatives_eval',
        show_progress_bar: bool = True,
        top_k_values: List[int] = [1, 3, 5, 10]
    ):
        """
        Args:
            anchors: List of anchor sentences
            positives: List of positive sentences (corresponding to anchors)
            batch_size: Batch size for evaluation
            name: Name for the evaluator
            show_progress_bar: Whether to show progress bar
            top_k_values: List of k values for accuracy@k computation
        """
        assert len(anchors) == len(positives), "Anchors and positives must have same length"
        
        self.anchors = anchors
        self.positives = positives
        self.batch_size = batch_size
        self.name = name
        self.show_progress_bar = show_progress_bar
        self.top_k_values = sorted(top_k_values)
        
    def __call__(self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """
        Evaluate the model.
        
        Returns:
            Primary metric (Accuracy@1)
        """
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        
        logging.info(f"InBatchNegativesEvaluator: Evaluating the model on {self.name} dataset{out_txt}:")
        
        # Compute metrics
        metrics = self._compute_metrics(model)
        
        # Log results
        logging.info(f"Accuracy@1: {metrics['accuracy@1']:.4f}")
        for k in self.top_k_values:
            if k > 1 and k <= self.batch_size:
                logging.info(f"Accuracy@{k}: {metrics[f'accuracy@{k}']:.4f}")
        logging.info(f"MRR (Mean Reciprocal Rank): {metrics['mrr']:.4f}")
        
        # Write to file if output path provided
        if output_path is not None:
            csv_path = os.path.join(output_path, f"{self.name}_results.csv")
            if not os.path.isfile(csv_path):
                with open(csv_path, mode="w", encoding="utf-8") as f:
                    f.write("epoch,steps,accuracy@1,mrr\n")
            
            with open(csv_path, mode="a", encoding="utf-8") as f:
                f.write(f"{epoch},{steps},{metrics['accuracy@1']:.4f},{metrics['mrr']:.4f}\n")
        
        # Return primary metric
        return metrics['accuracy@1']
    
    def _compute_metrics(self, model: SentenceTransformer) -> Dict[str, float]:
        """
        Compute evaluation metrics using in-batch negatives strategy.
        """
        model.eval()
        
        num_batches = (len(self.anchors) + self.batch_size - 1) // self.batch_size
        
        all_ranks = []
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), disable=not self.show_progress_bar, desc="Evaluating"):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(self.anchors))
                
                batch_anchors = self.anchors[start_idx:end_idx]
                batch_positives = self.positives[start_idx:end_idx]
                batch_size = len(batch_anchors)
                
                # Encode anchors and positives
                anchor_embeddings = model.encode(
                    batch_anchors,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                positive_embeddings = model.encode(
                    batch_positives,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                
                # Normalize embeddings for cosine similarity
                anchor_embeddings = torch.nn.functional.normalize(anchor_embeddings, p=2, dim=1)
                positive_embeddings = torch.nn.functional.normalize(positive_embeddings, p=2, dim=1)
                
                # Compute similarity matrix: [batch_size, batch_size]
                # similarity[i, j] = similarity between anchor[i] and positive[j]
                similarity_matrix = torch.mm(anchor_embeddings, positive_embeddings.t())
                
                # For each anchor, the diagonal element is the correct positive
                # All other elements in the row are negatives
                for i in range(batch_size):
                    similarities = similarity_matrix[i]  # Similarities of anchor[i] with all positives
                    
                    # Rank of the correct positive (diagonal element)
                    # Higher similarity = better, so we sort descending
                    sorted_indices = torch.argsort(similarities, descending=True)
                    
                    # Find where the correct positive (index i) ranks
                    rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
                    all_ranks.append(rank)
        
        # Compute metrics
        all_ranks = np.array(all_ranks)
        
        metrics = {}
        
        # Accuracy@k: percentage of times correct positive is in top-k
        for k in self.top_k_values:
            if k <= self.batch_size:
                accuracy_at_k = np.mean(all_ranks <= k)
                metrics[f'accuracy@{k}'] = accuracy_at_k
        
        # MRR: Mean Reciprocal Rank
        mrr = np.mean(1.0 / all_ranks)
        metrics['mrr'] = mrr
        
        return metrics


def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        logging.info(f"Distributed training - Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
        return rank, world_size, local_rank
    else:
        logging.info("Single GPU or CPU training")
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def prepare_dataset(data: List[Dict[str, str]]) -> List[InputExample]:
    """Convert dataset from list of dicts to InputExample format."""
    examples = []
    for item in data:
        examples.append(InputExample(
            texts=[item['anchor'], item['positive']]
        ))
    return examples


def train_sentence_transformer_ddp(
    model_name: str,
    train_data: List[Dict[str, str]],
    eval_data: Optional[List[Dict[str, str]]] = None,
    output_path: str = './output/sentence-transformer',
    epochs: int = 1,
    batch_size: int = 16,
    eval_batch_size: int = 32,
    warmup_steps: int = 100,
    evaluation_steps: int = 500,
    learning_rate: float = 2e-5,
    use_amp: bool = True,
    save_best_model: bool = True,
    max_grad_norm: float = 1.0
):
    """
    Train a SentenceTransformer model with DDP support and custom in-batch negatives evaluation.
    
    Args:
        model_name: Name or path of the base model
        train_data: Training data as list of dicts with 'anchor' and 'positive' keys
        eval_data: Optional evaluation data
        output_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Training batch size per GPU
        eval_batch_size: Evaluation batch size (important: affects number of negatives!)
        warmup_steps: Number of warmup steps
        evaluation_steps: Evaluate model every N steps
        learning_rate: Learning rate
        use_amp: Use automatic mixed precision
        save_best_model: Save the best model based on evaluation
        max_grad_norm: Max gradient norm for clipping
        
    Returns:
        Trained SentenceTransformer model
    """
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)
    
    # Only log on main process
    if not is_main_process:
        logging.getLogger().setLevel(logging.WARNING)
    
    if is_main_process:
        logging.info(f"Loading model: {model_name}")
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Move model to appropriate device
    if torch.cuda.is_available():
        model = model.to(f'cuda:{local_rank}')
    
    # Prepare training data
    if is_main_process:
        logging.info(f"Preparing {len(train_data)} training examples")
    
    train_examples = prepare_dataset(train_data)
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )
    
    # Set up Multiple Negatives Ranking Loss
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Prepare evaluation with custom evaluator (only on main process)
    evaluator = None
    if eval_data is not None and is_main_process:
        logging.info(f"Preparing {len(eval_data)} evaluation examples with InBatchNegativesEvaluator")
        logging.info(f"Evaluation batch size: {eval_batch_size} (each sample compared against {eval_batch_size-1} negatives)")
        
        # Extract anchors and positives
        anchors = [item['anchor'] for item in eval_data]
        positives = [item['positive'] for item in eval_data]
        
        # Use custom in-batch negatives evaluator
        evaluator = InBatchNegativesEvaluator(
            anchors=anchors,
            positives=positives,
            batch_size=eval_batch_size,
            name='in_batch_eval',
            show_progress_bar=True,
            top_k_values=[1, 3, 5, 10]
        )
    
    # Calculate training steps
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * epochs
    
    if is_main_process:
        logging.info(f"Training configuration:")
        logging.info(f"  World Size (GPUs): {world_size}")
        logging.info(f"  Epochs: {epochs}")
        logging.info(f"  Training batch size per GPU: {batch_size}")
        logging.info(f"  Effective training batch size: {batch_size * world_size}")
        logging.info(f"  Evaluation batch size: {eval_batch_size}")
        logging.info(f"  Steps per epoch: {steps_per_epoch}")
        logging.info(f"  Total steps: {total_steps}")
        logging.info(f"  Warmup steps: {warmup_steps}")
        logging.info(f"  Learning rate: {learning_rate}")
    
    # Train with DDP
    if is_main_process:
        logging.info("Starting training with DistributedDataParallel...")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path if is_main_process else None,
        evaluation_steps=evaluation_steps,
        optimizer_params={'lr': learning_rate},
        use_amp=use_amp,
        save_best_model=save_best_model,
        show_progress_bar=is_main_process,
        checkpoint_save_steps=evaluation_steps if is_main_process else 0,
        checkpoint_path=output_path if is_main_process else None
    )
    
    # Cleanup
    cleanup_distributed()
    
    if is_main_process:
        logging.info(f"Training complete! Model saved to {output_path}")
    
    return model


def evaluate_model(
    model: SentenceTransformer,
    eval_data: List[Dict[str, str]],
    batch_size: int = 32,
    top_k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate a trained SentenceTransformer model using in-batch negatives strategy.
    
    Args:
        model: Trained SentenceTransformer model
        eval_data: Evaluation data as list of dicts with 'anchor' and 'positive' keys
        batch_size: Batch size for evaluation (number of negatives = batch_size - 1)
        top_k_values: List of k values for accuracy@k
        
    Returns:
        Dictionary with evaluation metrics
    """
    logging.info(f"Evaluating model on {len(eval_data)} examples")
    logging.info(f"Using batch size {batch_size}: each anchor compared against {batch_size-1} negatives")
    
    # Extract anchors and positives
    anchors = [item['anchor'] for item in eval_data]
    positives = [item['positive'] for item in eval_data]
    
    # Create evaluator
    evaluator = InBatchNegativesEvaluator(
        anchors=anchors,
        positives=positives,
        batch_size=batch_size,
        name='final_eval',
        show_progress_bar=True,
        top_k_values=top_k_values
    )
    
    # Run evaluation
    metrics = evaluator._compute_metrics(model)
    
    logging.info("\nEvaluation results:")
    for metric_name, value in metrics.items():
        logging.info(f"  {metric_name}: {value:.4f}")
    
    return metrics


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
        {
            "anchor": "A child is riding a bicycle.",
            "positive": "A kid is cycling in the park."
        },
        {
            "anchor": "The sun is shining brightly.",
            "positive": "It's a sunny day."
        },
    ] * 100  # Multiply for larger dataset
    
    eval_data = [
        {
            "anchor": "A person is riding a bike.",
            "positive": "Someone is cycling."
        },
        {
            "anchor": "The cat is sleeping on the couch.",
            "positive": "A cat is resting on furniture."
        },
        {
            "anchor": "People are walking in the rain.",
            "positive": "Pedestrians in rainy weather."
        },
        {
            "anchor": "A bird is flying in the sky.",
            "positive": "A bird soars through the air."
        },
    ] * 25
    
    # Train with DDP and custom evaluator
    model = train_sentence_transformer_ddp(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        train_data=train_data,
        eval_data=eval_data,
        output_path='./trained_model',
        epochs=3,
        batch_size=16,  # Per GPU batch size for training
        eval_batch_size=32,  # Evaluation batch size (32 samples = 31 negatives per anchor)
        warmup_steps=100,
        evaluation_steps=100,
        learning_rate=2e-5
    )
    
    # Evaluate the final model
    if int(os.environ.get('RANK', 0)) == 0:
        results = evaluate_model(
            model, 
            eval_data,
            batch_size=64,  # Larger batch = more negatives = harder evaluation
            top_k_values=[1, 3, 5, 10, 20]
        )
        print(f"\nFinal evaluation results: {results}")