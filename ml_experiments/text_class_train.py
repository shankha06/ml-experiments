import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import InputExample, SentenceTransformer, losses, models, util
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

# ==========================================
# 1. SETUP & DATA PREPARATION
# ==========================================

# Dummy data generation (Replace with your actual data loading)
df = pd.read_parquet("your_data.parquet")

# Encode labels to integers (0 to 67)
le = LabelEncoder()
df["label"] = le.fit_transform(df["navigation"])
num_classes = len(le.classes_)
print(f"Dataset contains {len(df)} samples and {num_classes} classes.")

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# ==========================================
# 2. STAGE 1: CONTRASTIVE FINE-TUNING
# ==========================================
# We use BatchHardTripletLoss. It works directly on (text, label) batches.
# It makes the model learn that "login" texts are similar to each other 
# and dissimilar to "logout" texts.

print("\n--- Stage 1: Contrastive Fine-tuning (SentenceTransformer) ---")

# Load base model
base_model_name = "sentence-transformers/paraphrase-mpnet-base-v2"
model = SentenceTransformer(base_model_name)

# Prepare data for SentenceTransformer
# InputExample format: texts=[text], label=label_int
train_examples = [
    InputExample(texts=[row['question']], label=row['label']) 
    for _, row in train_df.iterrows()
]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

# Define Contrastive Loss
# BatchHardTripletLoss is excellent for classification data.
train_loss = losses.BatchHardTripletLoss(model=model)

# Train the body
# 1 epoch is often enough for contrastive learning on simple datasets
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    show_progress_bar=True
)

print("Contrastive fine-tuning complete.")

# ==========================================
# 3. STAGE 2: TRAIN CLASSIFICATION HEAD
# ==========================================
print("\n--- Stage 2: Training Classification Head ---")

# Define a custom wrapper model that includes the SBERT body + Head
class SentenceTransformerClassifier(nn.Module):
    def __init__(self, sbert_model, num_classes):
        super(SentenceTransformerClassifier, self).__init__()
        self.sbert = sbert_model
        self.embedding_dim = sbert_model.get_sentence_embedding_dimension()
        
        # The Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_texts):
        # Get embeddings from SBERT
        # We manually tokenize if we were doing raw pytorch, but sbert has encode()
        # For training integration, we usually want gradients.
        # However, for Stage 2, SetFit FREEZES the body. We will do the same.
        
        # Get features using SBERT's internal tokenizer
        features = self.sbert.tokenize(input_texts)
        # Move to same device as model
        features = {k: v.to(self.sbert.device) for k, v in features.items()}
        
        # Forward pass through SBERT
        out_features = self.sbert(features)
        embeddings = out_features['sentence_embedding']
        
        # Forward pass through Classifier
        logits = self.classifier(embeddings)
        return logits

# Initialize the combined model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
full_model = SentenceTransformerClassifier(model, num_classes).to(device)

# Freeze the SBERT body (Standard SetFit / Transfer Learning procedure)
for param in full_model.sbert.parameters():
    param.requires_grad = False

# Optimizer & Loss for the Head only
optimizer = torch.optim.Adam(full_model.classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Prepare simple batch loader for the strings
train_texts = train_df['question'].tolist()
train_labels = torch.tensor(train_df['label'].tolist()).to(device)

val_texts = val_df['question'].tolist()
val_labels = torch.tensor(val_df['label'].tolist()).to(device)

# Training Loop for the Head
epochs = 5
batch_size = 32

full_model.train()
for epoch in range(epochs):
    epoch_loss = 0
    # Simple batching loop
    for i in range(0, len(train_texts), batch_size):
        batch_texts = train_texts[i : i + batch_size]
        batch_labels = train_labels[i : i + batch_size]
        
        optimizer.zero_grad()
        logits = full_model(batch_texts)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs} | Head Loss: {epoch_loss/len(train_texts):.5f}")

# ==========================================
# 4. EVALUATION
# ==========================================
print("\n--- Evaluation ---")
full_model.eval()
with torch.no_grad():
    val_logits = full_model(val_texts)
    val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()

acc = accuracy_score(val_df['label'], val_preds)
print(f"Validation Accuracy: {acc:.4f}")

# ==========================================
# 5. SAVING THE MODEL
# ==========================================
save_path = "./my_manual_model"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 1. Save the SentenceTransformer body (Standard format)
full_model.sbert.save(os.path.join(save_path, "sbert_body"))

# 2. Save the PyTorch Head
torch.save(full_model.classifier.state_dict(), os.path.join(save_path, "head.pt"))

# 3. Save the LabelEncoder (Crucial for prediction!)
import pickle
with open(os.path.join(save_path, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

print(f"\nModel saved to {save_path}")
print("To inference, load sbert_body, load head.pt, and wrap them in the class.")

# ==========================================
# 6. INFERENCE EXAMPLE CODE
# ==========================================
print("\n--- Inference Example ---")
# Re-load
loaded_sbert = SentenceTransformer(os.path.join(save_path, "sbert_body"))
loaded_head_state = torch.load(os.path.join(save_path, "head.pt"))

# Re-init Wrapper
inference_model = SentenceTransformerClassifier(loaded_sbert, num_classes)
inference_model.classifier.load_state_dict(loaded_head_state)
inference_model.to(device)
inference_model.eval()

# Predict
test_query = "How do I sign out?"
with torch.no_grad():
    logits = inference_model([test_query])
    pred_id = torch.argmax(logits, dim=1).item()
    pred_label = le.inverse_transform([pred_id])[0]

print(f"Query: '{test_query}' -> Predicted: {pred_label}")