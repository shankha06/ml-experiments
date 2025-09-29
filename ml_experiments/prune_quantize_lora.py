import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.utils.prune as prune

# --- 1. Load the Original Model ---
model_id = "models/Meta-Llama-3-8B"

print("Loading original model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto" # Use "cpu" if you don't have enough GPU VRAM to load it
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# --- 2. Apply Pruning ---
print("Applying pruning...")
pruning_amount = 0.3

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=pruning_amount)
        # To make the pruning permanent and remove the pruning mask overhead
        prune.remove(module, 'weight')

# --- 3. Save the Pruned Model ---
# This saves the model's state dict with the zeroed-out weights.
pruned_model_path = "./llama-8b-pruned"
print(f"Saving pruned model to {pruned_model_path}...")
model.save_pretrained(pruned_model_path)
tokenizer.save_pretrained(pruned_model_path)

print("Pruning complete.")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Path where you saved the pruned model
pruned_model_path = "./llama-8b-pruned"

# --- 1. Define Quantization Configuration ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # "nf4" is a popular 4-bit format
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- 2. Load the Pruned Model with Quantization ---
print("Loading pruned model and applying quantization...")
model = AutoModelForCausalLM.from_pretrained(
    pruned_model_path,
    quantization_config=quantization_config,
    device_map="auto" # This will load the quantized model onto the GPU
)
tokenizer = AutoTokenizer.from_pretrained(pruned_model_path)

print("Quantization complete. Model is ready for fine-tuning.")
# The model object is now pruned AND quantized.

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments
# Assume you have your formatted dataset loaded, e.g., using the datasets library
# from datasets import load_dataset
# train_dataset = load_dataset(...)

# --- 1. Prepare Model for LoRA ---
# This freezes the base model layers and prepares for k-bit training
model = prepare_model_for_kbit_training(model)

# --- 2. Define LoRA Configuration ---
lora_config = LoraConfig(
    r=16,  # The rank of the update matrices. Higher r = more parameters to train.
    lora_alpha=32, # A scaling factor for the LoRA weights.
    target_modules=["q_proj", "v_proj"], # Target specific layers for adaptation
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# --- 3. Apply LoRA to the Model ---
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# Output will show something like:
# trainable params: 4,194,304 || all params: 4,608,119,808 || trainable%: 0.0908

# --- 4. Set up the Trainer ---
# This is a standard Hugging Face Trainer setup
training_args = TrainingArguments(
    output_dir="./lora-finetune-results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit", # Memory-efficient optimizer
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True, # Use fp16 for training stability
)

# `train_dataset` needs to be defined based on your task (see next section)
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=your_formatted_dataset, # This is your dataset from the next section
)

print("Starting LoRA fine-tuning...")
trainer.train()

# --- 5. Save the LoRA Adapter ---
adapter_path = "./llama-8b-pruned-quant-lora"
peft_model.save_pretrained(adapter_path)
print(f"LoRA adapter saved to {adapter_path}")

from peft import PeftModel

# Load the pruned and quantized base model
base_model = AutoModelForCausalLM.from_pretrained(pruned_model_path, quantization_config=quantization_config)

# Load the trained LoRA adapter and merge it
merged_model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = merged_model.merge_and_unload() # This merges the weights

# Save the final, ready-to-deploy model
merged_model.save_pretrained("./final_deployable_model")