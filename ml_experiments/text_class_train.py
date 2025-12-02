import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import InputExample, SentenceTransformer, losses, models, util
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Sampler

def create_triplets(df: pd.DataFrame, num_positives: int = 5, num_negatives: int = 20) -> list[dict]:
    """
    Creates a dataset of triplets (anchor, positives, negatives).

    Args:
        df: DataFrame with 'question' and 'navigation' columns.
        num_positives: Number of positive samples per anchor.
        num_negatives: Number of negative samples per anchor.

    Returns:
        A list of dictionaries, where each dictionary contains an 'anchor',
        a list of 'positives', and a list of 'negatives'.
    """
    dataset = []
    grouped_by_nav = df.groupby('navigation')['question'].apply(list)
    nav_classes = list(grouped_by_nav.index)
    all_questions_flat = df['question'].tolist()

    if len(nav_classes) < 2:
        raise ValueError("Need at least two distinct navigation classes to create negatives.")

    for nav_class, questions_in_class in grouped_by_nav.items():
        # Define the pool of potential positive and negative samples for this class
        
        # Positives must come from the same class, excluding the anchor itself.
        # Negatives are all questions NOT in the current class.
        negative_pool = [q for q in all_questions_flat if q not in questions_in_class]

        if len(questions_in_class) <= num_positives:
            print(f"Warning: Class '{nav_class}' has {len(questions_in_class)} samples, which is not more than the required {num_positives} positives. Some positives will be repeated.")
        
        if len(negative_pool) < num_negatives:
             print(f"Warning: Total negative samples available ({len(negative_pool)}) is less than num_negatives ({num_negatives}). Sampling with replacement.")

        for anchor_question in questions_in_class:
            # Sample positives
            # Candidates are all questions in the same class, except for the anchor
            positive_candidates = [q for q in questions_in_class if q != anchor_question]
            if not positive_candidates: # If only one sample in the class
                positive_candidates = questions_in_class # Sample from the single item list

            # Sample with replacement if not enough candidates
            if len(positive_candidates) < num_positives:
                positives = random.choices(positive_candidates, k=num_positives)
            else:
                positives = random.sample(positive_candidates, num_positives)

            # Sample negatives
            # Sample with replacement if not enough candidates
            if len(negative_pool) < num_negatives:
                negatives = random.choices(negative_pool, k=num_negatives)
            else:
                negatives = random.sample(negative_pool, num_negatives)

            dataset.append({
                'anchor': anchor_question,
                'positives': positives,
                'negatives': negatives
            })
            
    return dataset

# ==========================================
# 1. SETUP & DATA PREPARATION
# ==========================================

# Dummy data generation (Replace with your actual data loading)
# df = pd.read_csv("your_data.csv")
data = {
    "question": [
        "Where is the login page?", "How do I sign in?", 
        "Show me my account settings", "Edit profile options",
        "Where can I find the logout button?", "Exit the application"
    ] * 50, 
    "navigation": [
        "Login_Page", "Login_Page",
        "Settings_Page", "Settings_Page",
        "Logout_Action", "Logout_Action"
    ] * 50
}
df = pd.DataFrame(data)

# Create the triplet dataset and print one example
triplet_dataset = create_triplets(df, num_positives=5, num_negatives=20)
if triplet_dataset:
    print("\n--- Example Triplet ---")
    print(f"Total triplets created: {len(triplet_dataset)}")
    example = triplet_dataset[0]
    print(f"Anchor: {example['anchor']}")
    print(f"Positives ({len(example['positives'])}): {example['positives']}")
    print(f"Negatives ({len(example['negatives'])}): {example['negatives']}")
    print("-----------------------\n")

# Encode labels to integers (0 to 67)
le = LabelEncoder()
df["label"] = le.fit_transform(df["navigation"])
num_classes = len(le.classes_)
print(f"Dataset contains {len(df)} samples and {num_classes} classes.")

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# ==========================================
# 2. STAGE 1: CONTRASTIVE FINE-TUNING (MNRL)
# ==========================================

print("\n--- Stage 1: Contrastive Fine-tuning (MNRL) ---")

base_model_name = "sentence-transformers/paraphrase-mpnet-base-v2"
model = SentenceTransformer(base_model_name)

# 1. Format Data for MNRL: (Anchor, Positive)
# Anchor = User Question
# Positive = The Navigation Label text (semantic representation of the class)
train_examples = [
    InputExample(texts=[row['question'], row['navigation']], label=row['label_id']) 
    for _, row in train_df.iterrows()
]


# 3. Create DataLoader with the Custom BatchSampler
# Note: when using batch_sampler, we do NOT use batch_size or shuffle in DataLoader
train_dataloader = DataLoader(train_examples, batch_sampler=sampler)

# 4. Define Loss: MultipleNegativesRankingLoss
# This loss calculates similarities between (Anchor, Positive) and treats 
# all other Positives in the batch as Negatives.
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# Train the body
# 1 epoch is often enough for contrastive learning on simple datasets
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    show_progress_bar=True
)

print("Contrastive fine-tuning complete.")

# ==========================================
# 3. STAGE 2: TRAIN CLASSIFICATION HEAD ONLY (Frozen Body)
# ==========================================
print("\n--- Stage 2: Training Classification Head (Body Frozen) ---")

# Define a custom wrapper model that includes the SBERT body + Head
class SentenceTransformerClassifier(nn.Module):
    def __init__(self, sbert_model, num_classes):
        super(SentenceTransformerClassifier, self).__init__()
        self.sbert = sbert_model
        self.embedding_dim = sbert_model.get_sentence_embedding_dimension()
        
        # --- MODIFICATION: Transformer Encoder Layer Head ---
        # A single Transformer Encoder layer is used to process the sentence embedding
        # before the final classification. We treat the embedding vector as a sequence
        # of length 1 for this layer.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,      # Input/Output dimension
            nhead=8,                         # Number of attention heads
            dim_feedforward=2048,            # Dimension of the feedforward network model
            batch_first=False,               # Expects (Seq, Batch, Embed)
            dropout=0.1,
            activation='relu'
        )
        # We use TransformerEncoder with num_layers=1 to hold the single layer
        self.transformer_layer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # The final Classification Head maps the Transformer output to the number of classes
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_texts):
        # Get features using SBERT's internal tokenizer
        features = self.sbert.tokenize(input_texts)
        # Move to same device as model
        features = {k: v.to(self.sbert.device) for k, v in features.items()}
        
        # Forward pass through SBERT to get sentence embeddings
        out_features = self.sbert(features)
        embeddings = out_features['sentence_embedding'] # Shape: [batch_size, embedding_dim]
        
        # 1. Reshape for Transformer: [1, batch_size, embedding_dim]
        embeddings_reshaped = embeddings.unsqueeze(0) 

        # 2. Pass through Transformer Encoder (outputs [1, batch_size, embedding_dim])
        transformer_output_seq = self.transformer_layer(embeddings_reshaped) 

        # 3. Reshape back to [batch_size, embedding_dim]
        transformer_output = transformer_output_seq.squeeze(0)
        
        # 4. Forward pass through final Classifier
        logits = self.classifier(transformer_output)
        return logits

# Initialize the combined model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
full_model = SentenceTransformerClassifier(model, num_classes).to(device)

# Freeze the SBERT body (Standard SetFit / Transfer Learning procedure)
for param in full_model.sbert.parameters():
    param.requires_grad = False
    
# Freeze the TransformerEncoder and Linear Classifier initially
for param in full_model.transformer_layer.parameters():
    param.requires_grad = True
    
# Optimizer & Loss for the Head only
# Parameters to optimize: only the classifier's weights (Transformer Layer and Linear Layers)
head_only_params = list(full_model.classifier.parameters()) + list(full_model.transformer_layer.parameters())
optimizer_head = torch.optim.Adam(head_only_params, lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Prepare simple batch loader for the strings
train_texts = train_df['question'].tolist()
train_labels = torch.tensor(train_df['label'].tolist()).to(device)

val_texts = val_df['question'].tolist()
val_labels = torch.tensor(val_df['label'].tolist()).to(device)

# Training Loop for the Head Only (3 epochs recommended for initialization)
epochs_head_only = 3
batch_size = 32

for epoch in range(epochs_head_only):
    epoch_loss = 0.0
    
    # --- 1. Training Phase (Head Only) ---
    full_model.train()
    for i in range(0, len(train_texts), batch_size):
        batch_texts = train_texts[i : i + batch_size]
        batch_labels = train_labels[i : i + batch_size]
        
        optimizer_head.zero_grad()
        logits = full_model(batch_texts)
        loss = criterion(logits, batch_labels)
        loss.backward()
        
        # Apply gradient clipping to the classifier parameters 
        torch.nn.utils.clip_grad_norm_(head_only_params, max_norm=1.0)
        
        optimizer_head.step()
        
        epoch_loss += loss.item() * len(batch_texts)
    
    avg_train_loss = epoch_loss / len(train_texts)

    # --- 2. Validation Phase (Head Only) ---
    full_model.eval()
    val_epoch_loss = 0.0
    val_preds_for_epoch = []
    
    with torch.no_grad():
        for i in range(0, len(val_texts), batch_size):
            batch_texts = val_texts[i : i + batch_size]
            batch_labels = val_labels[i : i + batch_size] 

            # Forward pass
            logits = full_model(batch_texts)
            loss = criterion(logits, batch_labels)
            
            val_epoch_loss += loss.item() * len(batch_texts)
            
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds_for_epoch.extend(batch_preds)

    avg_val_loss = val_epoch_loss / len(val_texts)
    epoch_acc = accuracy_score(val_df['label'], val_preds_for_epoch)
    
    print(f"STAGE 2: Epoch {epoch+1}/{epochs_head_only} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {epoch_acc:.4f}")

# ==========================================
# 4. STAGE 3: END-TO-END FINE-TUNING (Unfrozen Body)
# ==========================================
# Unfreeze the Sentence Transformer body for fine-tuning. This is critical 
# for getting high accuracy on domain-specific tasks.
print("\n--- Stage 3: End-to-End Fine-tuning (Unfrozen Body) ---")

# 1. Unfreeze all parameters
for param in full_model.parameters():
    param.requires_grad = True

# 2. Use a NEW optimizer with a VERY low learning rate (e.g., 5e-6) 
# to prevent catastrophic forgetting in the SBERT body.
epochs_full_finetune = 2
low_lr = 5e-6 
optimizer_full = torch.optim.AdamW(full_model.parameters(), lr=low_lr, weight_decay=0.01)

# Training Loop for Full Fine-Tuning
for epoch in range(epochs_full_finetune):
    epoch_loss = 0.0
    
    # --- 1. Training Phase (Full Model) ---
    full_model.train()
    for i in range(0, len(train_texts), batch_size):
        batch_texts = train_texts[i : i + batch_size]
        batch_labels = train_labels[i : i + batch_size]
        
        optimizer_full.zero_grad()
        logits = full_model(batch_texts)
        loss = criterion(logits, batch_labels)
        loss.backward()
        
        # Apply gradient clipping to ALL model parameters
        torch.nn.utils.clip_grad_norm_(full_model.parameters(), max_norm=1.0)
        
        optimizer_full.step()
        
        epoch_loss += loss.item() * len(batch_texts)
    
    avg_train_loss = epoch_loss / len(train_texts)

    # --- 2. Validation Phase (Full Model) ---
    full_model.eval()
    val_epoch_loss = 0.0
    val_preds_for_epoch = []
    
    with torch.no_grad():
        for i in range(0, len(val_texts), batch_size):
            batch_texts = val_texts[i : i + batch_size]
            batch_labels = val_labels[i : i + batch_size] 

            # Forward pass
            logits = full_model(batch_texts)
            loss = criterion(logits, batch_labels)
            
            val_epoch_loss += loss.item() * len(batch_texts)
            
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds_for_epoch.extend(batch_preds)

    avg_val_loss = val_epoch_loss / len(val_texts)
    epoch_acc = accuracy_score(val_df['label'], val_preds_for_epoch)
    
    print(f"STAGE 3: Epoch {epoch+1}/{epochs_full_finetune} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {epoch_acc:.4f}")


# ==========================================
# 5. EVALUATION (Final Accuracy Report)
# ==========================================
# This section remains to report the final performance based on the last model state.
print("\n--- Final Evaluation ---")
full_model.eval()

val_preds = []
batch_size = 32

with torch.no_grad():
    for i in range(0, len(val_texts), batch_size):
        batch_texts = val_texts[i : i + batch_size]
        
        # Pass batch to model
        logits = full_model(batch_texts)
        
        # Get predictions for this batch
        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
        val_preds.extend(batch_preds)

# Calculate accuracy
acc = accuracy_score(val_df['label'], val_preds)
print(f"Final Validation Accuracy: {acc:.4f}")

# ==========================================
# 6. SAVING THE MODEL
# ==========================================
save_path = "./my_manual_model"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 1. Save the SentenceTransformer body (Standard format)
full_model.sbert.save(os.path.join(save_path, "sbert_body"))

# 2. Save the PyTorch Head: Save both the transformer layer and the linear layers
head_state_dict = {
    'transformer_layer': full_model.transformer_layer.state_dict(),
    'classifier': full_model.classifier.state_dict(),
}
torch.save(head_state_dict, os.path.join(save_path, "head.pt"))

# 3. Save the LabelEncoder (Crucial for prediction!)
import pickle
with open(os.path.join(save_path, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

print(f"\nModel saved to {save_path}")
print("To inference, load sbert_body, load head.pt, and wrap them in the class.")

# ==========================================
# 7. INFERENCE EXAMPLE CODE
# ==========================================
print("\n--- Inference Example ---")
# Re-load
loaded_sbert = SentenceTransformer(os.path.join(save_path, "sbert_body"))
loaded_head_state = torch.load(os.path.join(save_path, "head.pt"))

# Re-init Wrapper
inference_model = SentenceTransformerClassifier(loaded_sbert, num_classes)
inference_model.transformer_layer.load_state_dict(loaded_head_state['transformer_layer'])
inference_model.classifier.load_state_dict(loaded_head_state['classifier'])
inference_model.to(device)
inference_model.eval()

# Predict
test_query = "How do I sign out?"
with torch.no_grad():
    logits = inference_model([test_query])
    pred_id = torch.argmax(logits, dim=1).item()
    pred_label = le.inverse_transform([pred_id])[0]

print(f"Query: '{test_query}' -> Predicted: {pred_label}")