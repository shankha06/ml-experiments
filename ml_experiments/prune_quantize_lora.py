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

import torch
from datasets import Dataset
from transformers import AutoTokenizer

# --- 1. Load the same tokenizer used for your model ---
model_id = "./llama-7b-wanda-pruned" # Path to your pruned model/tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# --- 2. Create your raw data ---
# A mix of classification and ranking examples
raw_data = [
    # Classification Example 1
    {
        "instruction": "Classify the user's request into one of the following categories: [Account Inquiry, Fraud Report, Loan Application].",
        "input": "There is a charge on my credit card from a store I've never been to.",
        "output": "Fraud Report"
    },
    # Classification Example 2
    {
        "instruction": "Classify the user's request into one of the following categories: [Account Inquiry, Fraud Report, Loan Application].",
        "input": "I would like to apply for a mortgage to buy my first home.",
        "output": "Loan Application"
    },
    # Ranking Example
    {
        "instruction": "Rank the following banking services based on their relevance to the user query.",
        "input": {
            "query": "How can I save for a down payment on a house?",
            "items": [
                "A: Open a high-yield savings account.",
                "B: Apply for a new credit card with travel rewards.",
                "C: Schedule a consultation with a mortgage advisor."
            ]
        },
        "output": "C, A, B"
    }
]

# Convert to a Hugging Face Dataset object
dataset = Dataset.from_list(raw_data)


# --- 3. Create the Preprocessing Function ---
def preprocess_function(examples):
    # Combine instruction and input into a prompt
    prompts = []
    for i in range(len(examples['instruction'])):
        # Handle different input formats (string vs. dict)
        if isinstance(examples['input'][i], dict): # For ranking task
            query = examples['input'][i]['query']
            items = "\n".join(examples['input'][i]['items'])
            input_text = f"Query: {query}\nItems:\n{items}"
        else: # For classification task
            input_text = examples['input'][i]

        prompt = f"### Instruction:\n{examples['instruction'][i]}\n\n### Input:\n{input_text}\n\n### Response:\n"
        prompts.append(prompt)

    # Tokenize the full text (prompt + output)
    full_texts = [p + o for p, o in zip(prompts, examples['output'])]
    
    # Tokenize everything
    model_inputs = tokenizer(full_texts, padding="max_length", truncation=True, max_length=512)
    
    # Tokenize prompts separately to find their length for masking
    prompt_tokens = tokenizer(prompts, padding=False, truncation=True)

    labels = torch.tensor(model_inputs["input_ids"])
    
    # Mask the prompt part of the labels
    for i in range(len(prompt_tokens["input_ids"])):
        prompt_len = len(prompt_tokens["input_ids"][i])
        labels[i, :prompt_len] = -100 # -100 is the ignore_index for the loss function

    model_inputs["labels"] = labels.tolist()
    return model_inputs


# --- 4. Apply the function and prepare the dataset ---
processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

print("Dataset is ready for the Trainer.")
print("Example of one processed item:")
print(processed_dataset[0])