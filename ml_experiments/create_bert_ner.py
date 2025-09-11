import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from faker import Faker
from seqeval.metrics import classification_report
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



def generate_synthetic_ner_data(num_samples=100):
    """Generates synthetic NER training data for merchant and location entities.

    Args:
        num_samples (int): The number of synthetic data samples to generate.

    Returns:
        list: A list of spaCy-formatted training examples.
    """
    fake = Faker()
    training_data = []

    data = pd.read_excel("data/taxonomy-with-ids.en-US.xlsx")
    product_categories = data["category"].to_list()
    product_categories = [
        x.lower().strip() for x in product_categories if len(str(x)) > 4
    ]

    # Simple templates to embed entities
    templates = [
        "nearest {} deals to {}",
        "{} near {}",
        "{} near me",
        "{} deals",
        "offers for {} in {}",
        "{} in {}",
        "{} discounts",
        "deals for {}",
        "{}",
        "discounts for {}",
        "{} offers",
        "offers for {}",
    ]

    # Generate a list of fake merchants and locations for diversity
    merchants = [fake.company() for _ in range(5000)]
    locations = [fake.city() for _ in range(5000)]
    merchants.extend(product_categories)

    for _ in range(num_samples):
        # Choose a random template
        template = random.choice(templates)

        # Pick a random merchant and location
        merchant_name = random.choice(merchants)
        location_name = random.choice(locations)

        # Determine which entities to use in the template
        entities_in_query = []
        if "{}" in template:
            entities_in_query.append(("MERCHANT", merchant_name))
        if "{}" in template:
            entities_in_query.append(("LOCATION", location_name))

        # Fill the template with the chosen entities
        if len(entities_in_query) == 2:
            text = template.format(merchant_name, location_name)
        elif len(entities_in_query) == 1:
            if "merchant" in template:
                text = template.format(merchant_name)
            else:
                text = template.format(location_name)
        else:
            continue

        # Create the NER annotations (start and end indices)
        entities = []
        current_text = text

        # Find and annotate the merchant name
        if "MERCHANT" in [e[0] for e in entities_in_query]:
            start = text.find(merchant_name)
            if start != -1:
                end = start + len(merchant_name)
                entities.append((start, end, "MERCHANT"))

        # Find and annotate the location name
        if "LOCATION" in [e[0] for e in entities_in_query]:
            start = text.find(location_name)
            if start != -1:
                end = start + len(location_name)
                entities.append((start, end, "LOCATION"))

        # Append the annotated example to our dataset
        training_data.append((text.lower(), {"entities": entities}))

    return training_data


synthetic_data = generate_synthetic_ner_data(num_samples=5000)
print(synthetic_data[:5])
# Convert the data into a list of words and their labels
# This is a crucial step for NER tasks, as models require token-level labels.
processed_data = []
for text, annotations in synthetic_data:
    words = text.split()
    labels = ["O"] * len(words)  # Initialize with "O" for "Outside" an entity

    for start_char, end_char, entity_type in annotations["entities"]:
        # Find the word indices corresponding to the character spans
        current_char = 0
        start_word_idx, end_word_idx = -1, -1
        for i, word in enumerate(words):
            if current_char == start_char:
                start_word_idx = i
            current_char += len(word)
            if current_char == end_char:
                end_word_idx = i
                break
            current_char += 1  # for the space

        # Apply BIO (Beginning, Inside, Outside) tagging scheme
        if start_word_idx != -1 and end_word_idx != -1:
            labels[start_word_idx] = f"B-{entity_type}"
            for i in range(start_word_idx + 1, end_word_idx + 1):
                labels[i] = f"I-{entity_type}"

    processed_data.append({"tokens": words, "ner_tags": labels})

# Create a Hugging Face Dataset
df = pd.DataFrame(processed_data)
raw_dataset = Dataset.from_pandas(df)

# Load the tokenizer and define the label mapping
model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

labels_present = sorted({tag for d in raw_dataset for tag in d["ner_tags"]})
# Ensure 'O' is index 0 (common convention), followed by the rest of the tags
labels_present = [l for l in labels_present if l != "O"]
label_list = ["O"] + labels_present

id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in id2label.items()}


# Define a function to tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    labels = []
    for i, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Map the exact BIO tag string to its id
                tag = word_labels[word_idx]
                # Safety: fallback to 'O' if something unexpected slips in
                label_ids.append(label2id.get(tag, label2id["O"]))
            else:
                # Ignore subsequent subword tokens
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Re-apply the function to the dataset after fixing mappings
tokenized_dataset = raw_dataset.map(tokenize_and_align_labels, batched=True)

# Recreate/load the model with correct num_labels and mappings
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=len(label2id), id2label=id2label, label2id=label2id
).to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./models/bert-ner-finetuned",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    fp16=True,  # Enable mixed-precision training for faster GPU training
    logging_steps=1000,
    report_to="none",  # Disable logging to external services for simplicity
)


# Define compute metrics function for evaluation (optional but recommended)
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (where label is -100)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label, strict=False) if l != -100]
        for prediction, label in zip(predictions, labels, strict=False)
    ]
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]

    results = classification_report(true_labels, true_predictions, output_dict=True)
    return {
        "precision": results["micro avg"]["precision"],
        "recall": results["micro avg"]["recall"],
        "f1": results["micro avg"]["f1"],
        "accuracy": results["accuracy"],
    }


# Create a data collator to handle padding
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()
MODEL_NAME = "models/bert-ner-finetuned"
# Save the fine-tuned model
trainer.save_model(MODEL_NAME)

# Example of using the fine-tuned model for inference
from transformers import pipeline

ner_pipeline = pipeline(
    "ner",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)
text_to_predict = "video games deals in Washington and california"
prediction = ner_pipeline(text_to_predict)

# Print the predictions
print("\nPrediction on new text:")
print(prediction)
