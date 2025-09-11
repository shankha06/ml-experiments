import os
import json
import torch
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

# Device setup and capabilities
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
cuda_capability = torch.cuda.get_device_capability() if device == "cuda" else (0, 0)
use_bf16 = device == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
use_fp16 = device == "cuda" and not use_bf16
num_workers = min(8, (os.cpu_count() or 2))

# Optional: helps speed matmul on Ampere+ (also set in TrainingArguments via tf32=True)
if device == "cuda" and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
    torch.backends.cuda.matmul.allow_tf32 = True

# Load dataset
with open("notebooks/synthetic_misspellings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

TASK_PREFIX = "Correct the spelling: "  # align train and inference prompts

def preprocess_function(examples):
    """Tokenizes input/target with dynamic padding handled by the DataCollator."""
    inputs = [TASK_PREFIX + ex for ex in examples["input"]]
    targets = [ex for ex in examples["target"]]

    # No padding here; DataCollator will handle dynamic padding efficiently
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=False)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding=False)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize once up-front
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# Proper train/test split (faster, clearer)
split = tokenized_dataset.train_test_split(test_size=0.25, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# Load model
model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

# Enable gradient checkpointing for speed/memory
model.gradient_checkpointing_enable()
# Disable cache during training to avoid a warning and reduce memory
model.config.use_cache = False

# torch.compile can speed up training on PyTorch 2.x
use_torch_compile = hasattr(torch, "compile")
if use_torch_compile:
    try:
        model = torch.compile(model)  # default backend 'inductor'
        print("Model compiled with torch.compile for potential speedup.")
    except Exception as e:
        print(f"torch.compile failed, continuing without compile. Reason: {e}")

# Data collator with Tensor Core friendly padding
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,  # improves throughput on modern GPUs
    return_tensors="pt",
)

# Training arguments with performance tweaks
training_args = TrainingArguments(
    output_dir="./models/t5-finetuned-spelling",
    learning_rate=1e-3,  # Adafactor works well with this lr
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    # Mixed precision selection
    bf16=use_bf16,
    fp16=use_fp16,
    # tf32=True,  # enables TF32 matmul on Ampere+ for additional speed
    # DataLoader and batching efficiency
    dataloader_num_workers=num_workers,
    group_by_length=True,  # reduces padding waste -> higher throughput
    # Housekeeping
    push_to_hub=False,
    logging_steps=1000,
    save_strategy="epoch",
    # evaluation_strategy="no",  # skip eval during training for speed (can change to 'epoch' if needed)
    report_to="none",  # avoid logging overhead if not needed
    # Trainer-level compile toggle (if available in your Transformers version)
    torch_compile=use_torch_compile,
    torch_compile_backend="inductor" if use_torch_compile else None,
    # Gradient checkpointing flag for Trainer as well
    gradient_checkpointing=True,
    remove_unused_columns=False
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,  # no eval during training for speed; set to eval_dataset if needed
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train
trainer.train()

# Re-enable cache for faster inference and save
model.config.use_cache = True
trainer.save_model("./models/t5-finetuned-spelling-final")
tokenizer.save_pretrained("./models/t5-finetuned-spelling-final")

# Inference (Using the Fine-tuned Model)
from transformers import T5Tokenizer as _T5Tokenizer, T5ForConditionalGeneration as _T5ForConditionalGeneration

# Load the fine-tuned model and tokenizer
model_path = "./models/t5-finetuned-spelling-final"
tokenizer = _T5Tokenizer.from_pretrained(model_path)
model = _T5ForConditionalGeneration.from_pretrained(model_path).to(device)
model.eval()

def correct_spelling(text: str) -> str:
    """Corrects spelling of a given text using the fine-tuned model."""
    input_text = TASK_PREFIX + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=128, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=128,
            num_beams=5,
            early_stopping=True,
            do_sample=False,
        )
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# Example usage
if __name__ == "__main__":
    original_text = "cappechino offeers"
    corrected = correct_spelling(original_text)
    print(f"Original: {original_text}")
    print(f"Corrected: {corrected}")