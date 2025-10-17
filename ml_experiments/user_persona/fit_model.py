import json
import os
import random
from datetime import datetime

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


# Display settings for better output
pd.set_option('display.max_columns', None)
# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"PyTorch Version: {torch.__version__}")
print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

################################################################################
# PART 0: SYNTHETIC DATA GENERATION
################################################################################
# This part creates the dataset files so the script can run independently.

# Define our universe of navigation links (our "items")
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

def generate_synthetic_historical_data(num_users=1000, num_records=5000):
    print("Generating synthetic historical data...")
    data = []
    for _ in range(num_records):
        user_id = f"u{random.randint(100, 100+num_users-1)}"
        # Create user personas
        has_mortgage = random.random() > 0.8
        is_premium = random.random() > 0.7
        
        record = {
            "user_id": user_id,
            "month": f"2025-0{random.randint(1,9)}",
            "premium_card_indicator": 1 if is_premium else 0,
            "mortgage_indicator": 1 if has_mortgage else 0,
            "personal_saving_acc_count": random.randint(1, 3),
            "total_calls_to_bank": random.randint(0, 5),
            "online_transaction_count": random.randint(5, 50),
            "card_balance": round(random.uniform(500, 15000), 2),
            "category_retail_spending_amount": round(random.uniform(100, 2000), 2),
            "Risk_scores": random.randint(600, 850),
            # Simulate interaction based on persona
            "interacted_link_id": random.choice(
                ["link_06", "link_07"] if has_mortgage else \
                ["link_03", "link_04"] if not is_premium else \
                ["link_01", "link_02", "link_05"]
            )
        }
        data.append(record)
    with open("historical_training_dataset.json", "w") as f:
        json.dump(data, f)
    print("Done generating historical_training_dataset.json")

def generate_synthetic_log_data(num_logs=2000):
    print("Generating synthetic log data...")
    data = []
    for i in range(num_logs):
        user_id = f"u{random.randint(100, 1099)}"
        query = random.choice(["how to pay bill", "loan status", "transfer", "check balance", "customer support number"])
        
        # Simulate clicks based on query
        if "bill" in query or "transfer" in query:
            clicked_link = random.choice(["link_03", "link_04"])
            impressed_link = random.choice(["link_01", "link_06"])
        elif "loan" in query:
            clicked_link = random.choice(["link_06", "link_07"])
            impressed_link = random.choice(["link_02", "link_08"])
        else:
            clicked_link = random.choice(["link_01", "link_02", "link_10"])
            impressed_link = random.choice(["link_05", "link_07"])
            
        log = {
            "query": query,
            "user_id": user_id,
            "session_id": f"s{i}",
            "timestamp": 1758922149 + i*100,
            "clicks": [
                {"link": clicked_link, "rank": 1, "dwell_seconds": random.uniform(15, 120)}
            ],
            "impressions": [
                {"link": impressed_link, "position": 2, "skip_type": "user_scrolled_past"}
            ]
        }
        data.append(log)
    with open("log_training_dataset.json", "w") as f:
        json.dump(data, f)
    print("Done generating log_training_dataset.json")

# Generate data if files don't exist
if not os.path.exists("historical_training_dataset.json"):
    generate_synthetic_historical_data()
if not os.path.exists("log_training_dataset.json"):
    generate_synthetic_log_data()

################################################################################
# PART 1: CANDIDATE GENERATION (FIT MODEL)
################################################################################
print("\n--- STAGE 1: CANDIDATE GENERATION (FIT) ---")

# --- 1.1 Data Loading and Preprocessing ---
df_hist = pd.read_json("historical_training_dataset.json")

# Aggregate historical data to create one profile per user
user_features = df_hist.groupby('user_id').agg(
    avg_card_balance=('card_balance', 'mean'),
    avg_online_txn_count=('online_transaction_count', 'mean'),
    avg_risk_score=('Risk_scores', 'mean'),
    premium_user=('premium_card_indicator', 'max'),
    has_mortgage=('mortgage_indicator', 'max')
).reset_index()

# Normalize numerical features
scaler = StandardScaler()
numerical_cols = ['avg_card_balance', 'avg_online_txn_count', 'avg_risk_score']
user_features[numerical_cols] = scaler.fit_transform(user_features[numerical_cols])

# Create training data (user_id, interacted_link_id)
training_data = df_hist[['user_id', 'interacted_link_id']].drop_duplicates()
training_data = training_data.merge(user_features, on='user_id', how='left')

# Prepare item data
item_df = pd.DataFrame([
    {"link_id": k, "text": v["text"], "category": v["category"]} for k, v in NAV_LINKS.items()
])

# Encode IDs
user_encoder = LabelEncoder().fit(user_features['user_id'])
item_encoder = LabelEncoder().fit(item_df['link_id'])

training_data['user_idx'] = user_encoder.transform(training_data['user_id'])
training_data['item_idx'] = item_encoder.transform(training_data['interacted_link_id'])
user_features['user_idx'] = user_encoder.transform(user_features['user_id'])

# --- 1.2 Model Definition ---
embedding_dim = 32

class UserTower(nn.Module):
    def __init__(self, num_users, num_numerical_features, embedding_dim):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.dense1 = nn.Linear(embedding_dim + num_numerical_features, 64)
        self.dense2 = nn.Linear(64, embedding_dim)

    def forward(self, user_idx, numerical_features):
        user_emb = self.user_embed(user_idx).squeeze(1)
        combined = torch.cat([user_emb, numerical_features], dim=1)
        x = F.relu(self.dense1(combined))
        return self.dense2(x)

class ItemTower(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super().__init__()
        self.item_embed = nn.Embedding(num_items, embedding_dim)
    
    def forward(self, item_idx):
        return self.item_embed(item_idx).squeeze(1)

# --- 1.3 Training ---
num_users = len(user_encoder.classes_)
num_items = len(item_encoder.classes_)
num_numerical_features = len(numerical_cols) + 2 # premium_user, has_mortgage

user_tower = UserTower(num_users, num_numerical_features, embedding_dim).to(device)
item_tower = ItemTower(num_items, embedding_dim).to(device)

# Prepare data for training
# We merge user features into the training data to align them for the DataLoader
merged_training_data = training_data.merge(user_features, on='user_id', suffixes=('', '_y'))
user_idx_tensor = torch.tensor(merged_training_data['user_idx_y'].values, dtype=torch.long)
numerical_features_tensor = torch.tensor(
    merged_training_data[numerical_cols + ['premium_user_y', 'has_mortgage_y']].values, dtype=torch.float32
)
item_idx_tensor = torch.tensor(merged_training_data['item_idx'].values, dtype=torch.long)

dataset = TensorDataset(user_idx_tensor, numerical_features_tensor, item_idx_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Optimizer and Loss
params = list(user_tower.parameters()) + list(item_tower.parameters())
optimizer = optim.Adam(params, lr=0.01)
loss_fn = nn.CrossEntropyLoss()

print("Training FIT model...")
user_tower.train()
item_tower.train()
for epoch in range(5):
    total_loss = 0
    for user_idx, numerical_feats, item_idx in dataloader:
        user_idx, numerical_feats, item_idx = user_idx.to(device), numerical_feats.to(device), item_idx.to(device)
        
        optimizer.zero_grad()
        
        user_embeddings = user_tower(user_idx, numerical_feats)
        # Get embeddings for all items to use as negatives
        all_item_embeddings = item_tower.item_embed.weight
        
        # Dot product scores between user embeddings and all item embeddings
        scores = torch.matmul(user_embeddings, all_item_embeddings.t())
        
        # The target labels are the indices of the positive items
        loss = loss_fn(scores, item_idx)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/5, Loss: {total_loss/len(dataloader):.4f}")

# --- 1.4 Inference and Candidate Retrieval ---
print("\nDemonstrating FIT inference...")
user_tower.eval()
item_tower.eval()

with torch.no_grad():
    # 1. Get all item embeddings
    all_item_indices = torch.arange(num_items, device=device)
    all_item_embeddings = item_tower(all_item_indices).cpu().numpy()

    # 2. Build FAISS index
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(all_item_embeddings)
    print(f"FAISS index built with {index.ntotal} items.")

    # 3. Get a user's embedding
    sample_user_id = 'u101'
    user_profile = user_features[user_features['user_id'] == sample_user_id]
    user_idx_val = torch.tensor(user_profile['user_idx'].values, dtype=torch.long).to(device)
    user_feats_val = torch.tensor(
        user_profile[numerical_cols + ['premium_user', 'has_mortgage']].values, dtype=torch.float32
    ).to(device)
    
    user_embedding = user_tower(user_idx_val, user_feats_val).cpu().numpy()

    # 4. Query FAISS for top candidates
    k = 5
    distances, indices = index.search(user_embedding, k)
    retrieved_link_ids = item_encoder.inverse_transform(indices[0])
    print(f"Top {k} candidates for user {sample_user_id}: {list(retrieved_link_ids)}")
    candidates = list(retrieved_link_ids)


################################################################################
# PART 2: RERANKING (HoME-STYLE MODEL)
################################################################################
print("\n--- STAGE 2: RERANKING (HoME) ---")

# --- 2.1 Data Loading and Preprocessing ---
df_log = pd.read_json("log_training_dataset.json")

# Create a flat dataset for pointwise ranking
rerank_data = []
for _, row in df_log.iterrows():
    # Positive example
    if row['clicks']:
        rerank_data.append({
            'user_id': row['user_id'],
            'query': row['query'],
            'link_id': row['clicks'][0]['link'],
            'label': 1 # Clicked
        })
    # Negative example
    if row['impressions']:
       rerank_data.append({
            'user_id': row['user_id'],
            'query': row['query'],
            'link_id': row['impressions'][0]['link'],
            'label': 0 # Impressed but not clicked
        })
rerank_df = pd.DataFrame(rerank_data)

# Feature Engineering
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
query_embeddings = sbert_model.encode(rerank_df['query'].tolist())
item_text_embeddings = sbert_model.encode([NAV_LINKS[link]['text'] for link in rerank_df['link_id']], show_progress_bar=False)

rerank_df = rerank_df.merge(user_features, on='user_id', how='left').fillna(0)

X = np.concatenate([
    query_embeddings,
    item_text_embeddings,
    rerank_df[numerical_cols + ['premium_user', 'has_mortgage']].values.astype(np.float32)
], axis=1)
y = rerank_df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2.2 Model Definition (HoME-style) ---

class Expert(nn.Module):
    """A simple MLP expert network."""
    def __init__(self, input_dim, units):
        super().__init__()
        self.dense = nn.Linear(input_dim, units)

    def forward(self, inputs):
        return F.relu(self.dense(inputs))

class HoME_Layer(nn.Module):
    """A simple Mixture-of-Experts layer."""
    def __init__(self, input_dim, num_experts, expert_units):
        super().__init__()
        self.experts = nn.ModuleList([Expert(input_dim, expert_units) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, inputs):
        gate_weights = F.softmax(self.gate(inputs), dim=1)
        expert_outputs = [expert(inputs) for expert in self.experts]
        
        # Stack expert outputs to shape: (batch_size, num_experts, expert_units)
        expert_outputs_stack = torch.stack(expert_outputs, dim=1)
        weighted_outputs = torch.unsqueeze(gate_weights, 2) * expert_outputs_stack
        return torch.sum(weighted_outputs, dim=1)

class RerankerModel(nn.Module):
    def __init__(self, input_shape, num_experts=4, expert_units=32):
        super().__init__()
        self.bottom_dense = nn.Linear(input_shape, 64)
        self.home_layer = HoME_Layer(64, num_experts, expert_units)
        self.output_dense = nn.Linear(expert_units, 1)
        
    def forward(self, inputs):
        x = F.relu(self.bottom_dense(inputs))
        x = self.home_layer(x)
        # Use torch.sigmoid on raw logits from the linear layer
        return torch.sigmoid(self.output_dense(x))

# --- 2.3 Training ---
reranker_model = RerankerModel(input_shape=X_train.shape[1]).to(device)
print(reranker_model)

reranker_optimizer = optim.Adam(reranker_model.parameters())
reranker_loss_fn = nn.BCELoss() # Binary Cross-Entropy for binary classification

# Create DataLoaders
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

print("\nTraining HoME-style reranker model...")
for epoch in range(10):
    reranker_model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        reranker_optimizer.zero_grad()
        outputs = reranker_model(X_batch).squeeze()
        loss = reranker_loss_fn(outputs, y_batch)
        loss.backward()
        reranker_optimizer.step()
        total_loss += loss.item()
    
    # Validation
    reranker_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = reranker_model(X_batch).squeeze()
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/10, Loss: {total_loss/len(train_loader):.4f}, Test Accuracy: {accuracy:.2f}%")


################################################################################
# PART 3: END-TO-END INFERENCE PIPELINE
################################################################################
print("\n--- END-TO-END INFERENCE PIPELINE ---")

def get_personalized_nav_links(user_id, query, top_k=3):
    print(f"\nGetting recommendations for User '{user_id}' with Query: '{query}'")
    
    # Set models to evaluation mode
    user_tower.eval()
    item_tower.eval()
    reranker_model.eval()
    
    with torch.no_grad():
        # --- Stage 1: Candidate Generation ---
        user_profile = user_features[user_features['user_id'] == user_id]
        if user_profile.empty:
            print(f"User {user_id} not found. Returning default links.")
            return ["link_01", "link_03", "link_10"]
            
        user_idx_val = torch.tensor(user_profile['user_idx'].values, dtype=torch.long).to(device)
        user_feats_val = torch.tensor(
            user_profile[numerical_cols + ['premium_user', 'has_mortgage']].values, dtype=torch.float32
        ).to(device)
        user_embedding = user_tower(user_idx_val, user_feats_val).cpu().numpy()
        
        _, indices = index.search(user_embedding, 10) # Get more candidates for reranking
        candidate_links = item_encoder.inverse_transform(indices[0])
        print(f"Stage 1 Candidates: {list(candidate_links)}")

        # --- Stage 2: Reranking ---
        rerank_features = []
        query_embedding = sbert_model.encode([query])
        for link_id in candidate_links:
            link_text_embedding = sbert_model.encode([NAV_LINKS[link_id]['text']])
            user_hist_features = user_profile[numerical_cols + ['premium_user', 'has_mortgage']].values
            
            combined_features = np.concatenate([
                query_embedding,
                link_text_embedding,
                user_hist_features
            ], axis=1)
            rerank_features.append(combined_features)
        
        rerank_features_np = np.vstack(rerank_features)
        rerank_features_tensor = torch.tensor(rerank_features_np, dtype=torch.float32).to(device)
        
        scores = reranker_model(rerank_features_tensor).cpu().numpy().flatten()
        
        scored_candidates = sorted(zip(candidate_links, scores), key=lambda x: x[1], reverse=True)
        
        print("Stage 2 Reranked Scores:")
        for link, score in scored_candidates:
            print(f"  - {link} ({NAV_LINKS[link]['text']}): {score:.4f}")
            
        final_links = [link for link, score in scored_candidates[:top_k]]
        return final_links

# --- Run examples ---
# Example 1: A user with a mortgage who is likely to be interested in loans
final_recommendations = get_personalized_nav_links(user_id='u101', query="check rates")
print(f"\nFinal Top 3 Recommended Links: {final_recommendations}")

# Example 2: A different user with a general query
final_recommendations = get_personalized_nav_links(user_id='u888', query="how to pay")
print(f"\nFinal Top 3 Recommended Links: {final_recommendations}")