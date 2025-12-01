# pip install sentence-transformers torch torchvision scikit-learn pandas

import os

import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import InputExample, SentenceTransformer, evaluation, util
from sentence_transformers.losses import ContrastiveLoss, MultipleNegativesRankingLoss
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# -------------------------------
# 1. Your Data (example)
# -------------------------------
df = pd.read_csv("your_data.csv")

# Ensure class labels are 0 to 66
df['class'] = pd.factorize(df['class'])[0]
num_classes = df['class'].nunique()
assert num_classes == 67, f"Expected 67 classes, got {num_classes}"

print(f"Classes: {num_classes}, Samples: {len(df)}")

# Train/validation split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=42)

# -------------------------------
# 2. Custom Dataset that returns text + label
# -------------------------------
class TextLabelDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = TextLabelDataset(train_df['text'].tolist(), train_df['class'].tolist())
val_dataset   = TextLabelDataset(val_df['text'].tolist(),   val_df['class'].tolist())

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------
# 3. Model: SentenceTransformer + Classification Head
# -------------------------------
class SentenceTransformerWithHead(nn.Module):
    def __init__(self, base_model_name='all-MiniLM-L6-v2', num_classes=67, fine_tune_encoder=True):
        super().__init__()
        self.encoder = SentenceTransformer(base_model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()  # 384 for MiniLM
        self.classifier = nn.Linear(self.embedding_dim, num_classes)
        
        # Optionally freeze encoder (for pure head training)
        if not fine_tune_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, texts):
        # texts: list of strings or already tokenized
        embeddings = self.encoder.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        logits = self.classifier(embeddings)
        return logits, embeddings  # return both for dual loss

    def encode(self, texts):
        return self.encoder.encode(texts, convert_to_tensor=True, show_progress_bar=False)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformerWithHead(
    base_model_name='all-MiniLM-L6-v2',
    num_classes=67,
    fine_tune_encoder=True  # Set False if you want frozen encoder
).to(device)

# -------------------------------
# 4. Losses: Contrastive + Cross-Entropy
# -------------------------------
ce_loss_fn = nn.CrossEntropyLoss()

# MultipleNegativesRankingLoss treats all samples with same label in batch as positives
contrastive_loss_fn = MultipleNegativesRankingLoss(model=None)  # will use embeddings

# Hyperparameters
contrastive_weight = 0.5   # Tune this: 0.1 to 1.0
ce_weight = 1.0

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
num_epochs = 10

# -------------------------------
# 5. Training Loop with Dual Loss
# -------------------------------
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        texts = batch['text']
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        logits, embeddings = model(texts)  # embeddings: (batch_size, dim)

        # 1. Cross-entropy loss
        ce_loss = ce_loss_fn(logits, labels)

        # 2. Contrastive loss on embeddings (same class = positive)
        # We create pairs: (embedding_i, embedding_j) where label_i == label_j
        contrastive_loss = contrastive_loss_fn(embeddings, labels)

        # Or use this stronger SupCon version (recommended):
        # from sentence_transformers.losses import SupConLoss
        # contrastive_loss = SupConLoss(model=None)(embeddings, labels)

        loss = ce_weight * ce_loss + contrastive_weight * contrastive_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # -------------------------------
    # Validation
    # -------------------------------
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            texts = batch['text']
            labels = batch['label'].to(device)

            logits, _ = model(texts)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model_with_contrastive_head.pt")

print(f"\nBest Validation Accuracy: {best_acc:.4f}")

# -------------------------------
# 6. Final Evaluation + Classification Report
# -------------------------------
model.load_state_dict(torch.load("best_model_with_contrastive_head.pt"))
model.eval()

val_texts = val_df['text'].tolist()
val_labels = val_df['class'].tolist()

with torch.no_grad():
    logits, _ = model(val_texts)
    preds = torch.argmax(logits, dim=1).cpu().numpy()

print("\n=== Final Classification Report ===")
print(classification_report(val_labels, preds, digits=4))

# -------------------------------
# 7. Inference Example
# -------------------------------
def predict(texts):
    model.eval()
    with torch.no_grad():
        logits, _ = model(texts)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds.cpu().numpy(), probs.cpu().numpy()

new_texts = ["This is an amazing product!", "Terrible service, never again."]
preds, probs = predict(new_texts)

for t, p, prob in zip(new_texts, preds, probs):
    print(f"Text: {t}")
    print(f"Predicted class: {p}, Confidence: {prob[p]:.3f}")