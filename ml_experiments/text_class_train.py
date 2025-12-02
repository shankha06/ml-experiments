import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers import util
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import os
from transformers import get_linear_schedule_with_warmup 

# ==========================================
# 0. ADVANCED UTILITIES (Focal Loss)
# ==========================================

class FocalLoss(nn.Module):
    """
    Focal Loss allows the model to focus on hard examples.
    It down-weights easy examples and up-weights hard ones.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # Class weights
        self.gamma = gamma # Focusing parameter (higher = more focus on hard examples)
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # inputs: [N, C] logits
        # targets: [N] class indices
        
        # Apply label smoothing if requested
        if self.label_smoothing > 0:
            num_classes = inputs.size(-1)
            # Create smoothed target distribution
            log_preds = F.log_softmax(inputs, dim=-1)
            # Standard Cross Entropy part (with smoothing)
            # We implement Focal Loss manually on top of probabilities
            ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none', label_smoothing=self.label_smoothing)
            pt = torch.exp(-ce_loss) # accuracy of prediction
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        else:
            ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

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

# Encode labels to integers (0 to 67)
le = LabelEncoder()
df["label"] = le.fit_transform(df["navigation"])
num_classes = len(le.classes_)
class_names = le.classes_
print(f"Dataset contains {len(df)} samples and {num_classes} classes.")

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# --- NEW: Compute Class Weights ---
# This helps if your 67 classes are not perfectly balanced
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(train_df['label']), 
    y=train_df['label']
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# ==========================================
# 2. STAGE 1: CONTRASTIVE FINE-TUNING
# ==========================================
print("\n--- Stage 1: Contrastive Fine-tuning (SentenceTransformer) ---")

# Load base model
base_model_name = "sentence-transformers/paraphrase-mpnet-base-v2"
model = SentenceTransformer(base_model_name)

# Prepare data for SentenceTransformer
train_examples = [
    InputExample(texts=[row['question']], label=row['label']) 
    for _, row in train_df.iterrows()
]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

# Define Contrastive Loss
train_loss = losses.BatchHardTripletLoss(model=model)

# Train the body
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
        
        # Transformer Encoder Layer Head
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,      
            nhead=8,                         
            dim_feedforward=2048,            
            batch_first=False,               
            dropout=0.2, # Increased dropout for regularization
            activation='relu'
        )
        self.transformer_layer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 512), # Increased size
            nn.BatchNorm1d(512), # Added BatchNorm
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_texts):
        features = self.sbert.tokenize(input_texts)
        features = {k: v.to(self.sbert.device) for k, v in features.items()}
        
        out_features = self.sbert(features)
        embeddings = out_features['sentence_embedding'] 
        
        # Transformer Head
        embeddings_reshaped = embeddings.unsqueeze(0) 
        transformer_output_seq = self.transformer_layer(embeddings_reshaped) 
        transformer_output = transformer_output_seq.squeeze(0)
        
        logits = self.classifier(transformer_output)
        return logits

# Initialize the combined model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
full_model = SentenceTransformerClassifier(model, num_classes).to(device)

# Freeze the SBERT body
for param in full_model.sbert.parameters():
    param.requires_grad = False
    
for param in full_model.transformer_layer.parameters():
    param.requires_grad = True
    
# Optimizer & Loss
head_only_params = list(full_model.classifier.parameters()) + list(full_model.transformer_layer.parameters())
optimizer_head = torch.optim.AdamW(head_only_params, lr=1e-3, weight_decay=0.01)

# --- NEW: Use Focal Loss with Class Weights ---
criterion = FocalLoss(alpha=class_weights_tensor.to(device), gamma=2.0, label_smoothing=0.1)

train_texts = train_df['question'].tolist()
train_labels = torch.tensor(train_df['label'].tolist()).to(device)
val_texts = val_df['question'].tolist()
val_labels = torch.tensor(val_df['label'].tolist()).to(device)

epochs_head_only = 3
batch_size = 32

for epoch in range(epochs_head_only):
    full_model.train()
    epoch_loss = 0.0
    for i in range(0, len(train_texts), batch_size):
        batch_texts = train_texts[i : i + batch_size]
        batch_labels = train_labels[i : i + batch_size]
        
        optimizer_head.zero_grad()
        logits = full_model(batch_texts)
        loss = criterion(logits, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_only_params, max_norm=1.0)
        optimizer_head.step()
        epoch_loss += loss.item() * len(batch_texts)
    
    # Validation loop omitted for brevity in Stage 2, focused on initialization

# ==========================================
# 4. STAGE 3: END-TO-END FINE-TUNING (Unfrozen Body)
# ==========================================
print("\n--- Stage 3: End-to-End Fine-tuning (Unfrozen Body) ---")

for param in full_model.parameters():
    param.requires_grad = True

epochs_full_finetune = 5
low_lr = 5e-6 
optimizer_full = torch.optim.AdamW(full_model.parameters(), lr=low_lr, weight_decay=0.01)

total_steps = (len(train_texts) // batch_size) * epochs_full_finetune
warmup_steps = int(0.1 * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer_full, 
    num_warmup_steps=warmup_steps, 
    num_training_steps=total_steps
)

for epoch in range(epochs_full_finetune):
    epoch_loss = 0.0
    
    # --- Training Phase ---
    full_model.train()
    for i in range(0, len(train_texts), batch_size):
        batch_texts = train_texts[i : i + batch_size]
        batch_labels = train_labels[i : i + batch_size]
        
        optimizer_full.zero_grad()
        logits = full_model(batch_texts)
        loss = criterion(logits, batch_labels) # Focal Loss used here too
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(full_model.parameters(), max_norm=1.0)
        
        optimizer_full.step()
        scheduler.step() 
        epoch_loss += loss.item() * len(batch_texts)
    
    avg_train_loss = epoch_loss / len(train_texts)

    # --- Validation Phase ---
    full_model.eval()
    val_epoch_loss = 0.0
    val_preds_for_epoch = []
    
    with torch.no_grad():
        for i in range(0, len(val_texts), batch_size):
            batch_texts = val_texts[i : i + batch_size]
            batch_labels = val_labels[i : i + batch_size] 

            logits = full_model(batch_texts)
            loss = criterion(logits, batch_labels)
            
            val_epoch_loss += loss.item() * len(batch_texts)
            
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds_for_epoch.extend(batch_preds)

    avg_val_loss = val_epoch_loss / len(val_texts)
    epoch_acc = accuracy_score(val_df['label'], val_preds_for_epoch)
    
    print(f"STAGE 3: Epoch {epoch+1}/{epochs_full_finetune} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {epoch_acc:.4f}")


# ==========================================
# 5. DETAILED EVALUATION (Crucial for 67 classes)
# ==========================================
print("\n--- Final Detailed Evaluation ---")
full_model.eval()

val_preds = []
val_probs = []

with torch.no_grad():
    for i in range(0, len(val_texts), batch_size):
        batch_texts = val_texts[i : i + batch_size]
        logits = full_model(batch_texts)
        
        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
        val_preds.extend(batch_preds)

# Calculate global accuracy
acc = accuracy_score(val_df['label'], val_preds)
print(f"Final Validation Accuracy: {acc:.4f}")

# --- NEW: Classification Report ---
# This shows Precision, Recall, and F1-score for EVERY class.
# Look for classes with low F1-scores - those are your bottlenecks.
print("\nClassification Report:")
print(classification_report(val_df['label'], val_preds, target_names=class_names))

# ==========================================
# 6. SAVING THE MODEL (Same as before)
# ==========================================
save_path = "./my_manual_model"
if not os.path.exists(save_path):
    os.makedirs(save_path)

full_model.sbert.save(os.path.join(save_path, "sbert_body"))
head_state_dict = {
    'transformer_layer': full_model.transformer_layer.state_dict(),
    'classifier': full_model.classifier.state_dict(),
}
torch.save(head_state_dict, os.path.join(save_path, "head.pt"))

import pickle
with open(os.path.join(save_path, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

print(f"\nModel saved to {save_path}")