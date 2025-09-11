import logging

from datasets import load_dataset

from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator
from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss
from sentence_transformers.training_args import BatchSamplers

from datasets import Dataset

from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses
from sentence_transformers.sparse_encoder.evaluation import SparseTripletEvaluator

model = SparseEncoder("models/bert")
# SparseEncoder(
#   (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'BertModel'})
#   (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
#   (2): SparseAutoEncoder({'input_dim': 1024, 'hidden_dim': 4096, 'k': 256, 'k_aux': 512, 'normalize': False, 'dead_threshold': 30})
# )


train_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
eval_dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")

print(train_dataset)
"""
Dataset({
    features: ['anchor', 'positive', 'negative'],
    num_rows: 557850
})
"""

model = SparseEncoder("distilbert/distilbert-base-uncased")
train_dataset = Dataset.from_dict(
    {
        "anchor": ["It's nice weather outside today.", "He drove to work."],
        "positive": ["It's so sunny.", "He took the car to the office."],
        "negative": ["It's quite rainy, sadly.", "She walked to the store."],
    }
)
loss = losses.SpladeLoss(
    model=model, loss=losses.SparseTripletLoss(model), document_regularizer_weight=3e-5, query_regularizer_weight=5e-5
)
# 5. (Optional) Specify training arguments
run_name = "splade-distilbert-base-uncased-nq"
args = SparseEncoderTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=200,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
)

# 6. (Optional) Create an evaluator & evaluate the base model
dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev[:1000]")

# Initialize the SparseTripletEvaluator
evaluator = SparseTripletEvaluator(
    anchors=dataset[:1000]["anchor"],
    positives=dataset[:1000]["positive"],
    negatives=dataset[:1000]["negative"],
    name="all_nli_dev",
    batch_size=32,
    show_progress_bar=True,
)

# 7. Create a trainer & train
trainer = SparseEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=evaluator,
)
trainer.train()

# 8. Evaluate the model performance again after training
evaluator(model)

# 9. Save the trained model
model.save_pretrained(f"models/{run_name}/final")