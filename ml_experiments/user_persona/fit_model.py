import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import faiss
import os
import random
from datetime import datetime
from tqdm import tqdm

# Display settings for better output
pd.set_option('display.max_columns', None)

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Current Time: Friday, October 17, 2025 at 6:54:46 PM IST")

################################################################################
# PART 0: SYNTHETIC DATA GENERATION (Unchanged)
################################################################################
# This part creates the dataset files so the script can run independently.

NAV_LINKS = {
    "link_01": {"text": "View Account Balance", "category": "accounts"},
    "link_02": {"text": "See Recent Transactions", "category": "accounts"},
    "link_03": {"text": "Pay Bills Online", "category": "payments"},
    "link_04": {"text": "Transfer Funds Between Accounts", "category": "payments"},
    "link_05": {"text": "Open a New Savings Account", "category": "products"},
    "link_06": {"text": "Apply for a Personal Loan", "category": "products"},
    "link_07": {"text": "Check Mortgage Rates", "category": "products"},
    "link_08": {"text": "Find Nearby ATMs", "category": "services"},
    "link_09": {"text": "Report a Lost or Stolen Card", "category": "support"},
    "link_10": {"text": "Contact Customer Service", "category": "support"},
}

# Assume data generation functions run and create the files if they don't exist.
# The bodies of these functions are omitted for brevity.
if not os.path.exists("historical_training_dataset.json"):
    print("Generating synthetic historical data...")
if not os.path.exists("log_training_dataset.json"):
    print("Generating synthetic log data...")


################################################################################
# Pre-computation & Setup (Unchanged)
################################################################################
print("\n--- Global Setup ---")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

df_hist = pd.read_json("historical_training_dataset.json")
user_features = df_hist.groupby('user_id').agg(
    avg_card_balance=('card_balance', 'mean'),
    avg_online_txn_count=('online_transaction_count', 'mean'),
    avg_risk_score=('Risk_scores', 'mean'),
    premium_user=('premium_card_indicator', 'max'),
    has_mortgage=('mortgage_indicator', 'max')
).reset_index()

scaler = StandardScaler()
numerical_cols = ['avg_card_balance', 'avg_online_txn_count', 'avg_risk_score']
user_features[numerical_cols] = scaler.fit_transform(user_features[numerical_cols])

################################################################################
# PART 1: SEMANTIC CANDIDATE GENERATION (Unchanged)
################################################################################
print("\n--- STAGE 1: Building Semantic Candidate Generator ---")

item_df = pd.DataFrame([{"link_id": k, "text": v["text"]} for k, v in NAV_LINKS.items()])
link_texts = item_df['text'].tolist()
link_embeddings = sbert_model.encode(link_texts, convert_to_numpy=True)
faiss.normalize_L2(link_embeddings)

embedding_dim = link_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)
index.add(link_embeddings)
print(f"FAISS index built with {index.ntotal} items for semantic search.")

link_id_encoder = LabelEncoder().fit(item_df['link_id'])


################################################################################
# PART 2: PERSONALIZED RERANKING MODEL with BPR
################################################################################
print("\n--- STAGE 2: Personalized Reranking Model (with BPR Loss) ---")

# --- 2.1 Data Preprocessing for BPR (NEW) ---
# We now create triplets: (user, positive_item, negative_item)
df_log = pd.read_json("log_training_dataset.json")
rerank_triplets = []
for _, row in df_log.iterrows():
    # A training triplet requires a click (positive) and an impression (negative)
    if row['clicks'] and row['impressions']:
        rerank_triplets.append({
            'user_id': row['user_id'],
            'query': row['query'],
            'positive_link_id': row['clicks'][0]['link'],
            'negative_link_id': row['impressions'][0]['link']
        })
rerank_df_triplets = pd.DataFrame(rerank_triplets)

# --- 2.2 PyTorch Dataset for BPR (NEW) ---
class BPRRerankerDataset(Dataset):
    def __init__(self, df_triplets, user_features_df, all_nav_links, sbert_model):
        self.df = df_triplets.merge(user_features_df, on='user_id', how='left').fillna(0)
        
        # Pre-compute all embeddings to speed up __getitem__
        self.query_embs = sbert_model.encode(self.df['query'].tolist())
        self.pos_item_embs = sbert_model.encode([all_nav_links[link]['text'] for link in self.df['positive_link_id']])
        self.neg_item_embs = sbert_model.encode([all_nav_links[link]['text'] for link in self.df['negative_link_id']])
        
        self.user_hist_features = self.df[numerical_cols + ['premium_user', 'has_mortgage']].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Anchor features (user + query)
        user_hist = self.user_hist_features[idx]
        query_emb = self.query_embs[idx]
        
        # Positive item features
        pos_item_emb = self.pos_item_embs[idx]
        
        # Negative item features
        neg_item_emb = self.neg_item_embs[idx]
        
        # Combine into full feature vectors for the model
        positive_features = np.concatenate([query_emb, pos_item_emb, user_hist])
        negative_features = np.concatenate([query_emb, neg_item_emb, user_hist])
        
        return {
            'positive_features': torch.tensor(positive_features, dtype=torch.float32),
            'negative_features': torch.tensor(negative_features, dtype=torch.float32)
        }

# --- 2.3 Model Definition (Unchanged) ---
# The model architecture is the same; it's a scorer. BPR changes how we train it.
class HoME_Layer(nn.Module):
    def __init__(self, input_size, num_experts, expert_units):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(input_size, expert_units) for _ in range(num_experts)])
        self.gate = nn.Linear(input_size, num_experts)
    def forward(self, x):
        gate_weights = F.softmax(self.gate(x), dim=1)
        expert_outputs = torch.stack([F.relu(expert(x)) for expert in self.experts], dim=1)
        return torch.bmm(gate_weights.unsqueeze(1), expert_outputs).squeeze(1)

class RerankerModel(nn.Module):
    def __init__(self, input_shape, num_experts=4, expert_units=32):
        super().__init__()
        self.bottom = nn.Linear(input_shape, 64)
        self.home_layer = HoME_Layer(64, num_experts, expert_units)
        # Output is a single score (logit), not a probability
        self.output_layer = nn.Linear(expert_units, 1)
    def forward(self, x):
        x = F.relu(self.bottom(x))
        x = self.home_layer(x)
        return self.output_layer(x)

# --- 2.4 BPR Training Loop (NEW) ---
train_df, test_df = train_test_split(rerank_df_triplets, test_size=0.2, random_state=42)

# Create datasets
train_dataset = BPRRerankerDataset(train_df, user_features, NAV_LINKS, sbert_model)
test_dataset = BPRRerankerDataset(test_df, user_features, NAV_LINKS, sbert_model)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Instantiate model
# The input shape is calculated from the concatenated feature vector
input_dim = sbert_model.get_sentence_embedding_dimension() * 2 + len(numerical_cols) + 2
reranker_model = RerankerModel(input_shape=input_dim).to(device)
optimizer = optim.Adam(reranker_model.parameters(), lr=0.001)

print("\nTraining Personalized Reranker with BPR Loss...")
for epoch in range(10):
    reranker_model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
        positive_features = batch['positive_features'].to(device)
        negative_features = batch['negative_features'].to(device)
        
        optimizer.zero_grad()
        
        # Get scores for both positive and negative items
        score_positive = reranker_model(positive_features)
        score_negative = reranker_model(negative_features)
        
        # BPR Loss calculation
        loss = -F.logsigmoid(score_positive - score_negative).mean()
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation using a ranking metric (AUC-like)
    reranker_model.eval()
    total_correct_pairs = 0
    total_pairs = 0
    with torch.no_grad():
        for batch in test_loader:
            positive_features = batch['positive_features'].to(device)
            negative_features = batch['negative_features'].to(device)
            
            score_positive = reranker_model(positive_features)
            score_negative = reranker_model(negative_features)
            
            total_correct_pairs += (score_positive > score_negative).sum().item()
            total_pairs += len(score_positive)
    
    ranking_accuracy = total_correct_pairs / total_pairs
    print(f"Epoch {epoch+1}: Train Loss: {total_loss/len(train_loader):.4f}, Val Ranking Accuracy: {ranking_accuracy:.4f}")

################################################################################
# PART 3: END-TO-END INFERENCE PIPELINE (Unchanged)
################################################################################
# The inference logic does not change. We still score each candidate independently
# and rank them. The BPR training helps the model produce better relative scores.
print("\n--- END-TO-END INFERENCE PIPELINE ---")

def get_personalized_nav_links(user_id, query, top_k=3):
    print(f"\n🚀 Getting recommendations for User '{user_id}' with Query: '{query}'")
    reranker_model.eval()

    with torch.no_grad():
        # --- Stage 1: Get candidates via Semantic Search ---
        query_embedding = sbert_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        _, indices = index.search(query_embedding, 10)
        candidate_links = link_id_encoder.inverse_transform(indices[0])
        print(f"🔍 Stage 1 (Semantic Search) Candidates: {list(candidate_links)}")

        # --- Stage 2: Rerank candidates using user's historical data ---
        user_profile = user_features[user_features['user_id'] == user_id]
        if user_profile.empty:
            print(f"User {user_id} not found. Returning top semantic results.")
            return candidate_links[:top_k]
        
        rerank_features_list = []
        candidate_link_embeddings = sbert_model.encode([NAV_LINKS[link]['text'] for link in candidate_links])
        user_hist_features = user_profile[numerical_cols + ['premium_user', 'has_mortgage']].values[0]
        
        for i, link_id in enumerate(candidate_links):
            combined_features = np.concatenate([query_embedding[0], candidate_link_embeddings[i], user_hist_features])
            rerank_features_list.append(combined_features)
        
        rerank_features_tensor = torch.tensor(np.array(rerank_features_list), dtype=torch.float32).to(device)
        
        scores = reranker_model(rerank_features_tensor).cpu().numpy().flatten()
        
        scored_candidates = sorted(zip(candidate_links, scores), key=lambda x: x[1], reverse=True)
        
        print("🎯 Stage 2 (Personalized Reranking) Scores:")
        for link, score in scored_candidates:
            print(f"  - {link} ('{NAV_LINKS[link]['text']}'): {score:.4f}")
            
        final_links = [link for link, score in scored_candidates[:top_k]]
        return final_links

# --- Run examples ---
final_recommendations = get_personalized_nav_links(user_id='u101', query="check rates")
print(f"\n✅ Final Top 3 Recommended Links for u101: {final_recommendations}")

final_recommendations = get_personalized_nav_links(user_id='u888', query="transfer money")
print(f"\n✅ Final Top 3 Recommended Links for u888: {final_recommendations}")