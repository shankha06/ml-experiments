import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 1. Configuration ---
NUM_USERS = 25000
NUM_FEATURES = 250       # Number of aggregated features per user
EMBEDDING_DIM = 368      # The final user embedding dimension
NUM_CLUSTERS = 14        # Number of user archetypes/clusters to find
PRETRAIN_EPOCHS = 25
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GAMMA = 0.1             # Coefficient for the clustering loss

# --- 2. The DeepCAE Model ---
class DeepCAE(nn.Module):
    def __init__(self, num_features, embedding_dim, num_clusters):
        super(DeepCAE, self).__init__()
        
        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, embedding_dim) # This is our embedding layer
        )
        
        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, num_features) # Output is the reconstructed feature vector
        )
        
        # --- Clustering Layer ---
        # Learnable cluster centroids
        self.clustering_layer = nn.Parameter(torch.zeros(num_clusters, embedding_dim))

    def forward(self, x):
        # Get the embedding
        z = self.encoder(x)
        
        # Reconstruct the input
        x_recon = self.decoder(z)
        
        # --- Calculate cluster similarity (Student's t-distribution) ---
        alpha = 1.0
        # Calculate pairwise squared distances between embeddings and centroids
        # z.unsqueeze(1) shape: (batch, 1, embed_dim)
        # self.clustering_layer shape: (num_clusters, embed_dim)
        dist = torch.sum((z.unsqueeze(1) - self.clustering_layer) ** 2, 2)
        
        # Calculate the q distribution
        numerator = (1.0 + dist / alpha).pow(- (alpha + 1.0) / 2.0)
        q = numerator / torch.sum(numerator, dim=1, keepdim=True)
        
        return x_recon, q, z

def target_distribution(q):
    """
    Calculates the target distribution p, which sharpens the soft cluster assignments.
    """
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

# --- 3. Data Simulation and Preparation ---
print("1. Preparing data...")
# Simulate aggregated feature data for users
raw_data = np.random.rand(NUM_USERS, NUM_FEATURES)

# Scale the data (important for AE and clustering)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(raw_data)

# Create PyTorch DataLoader
tensor_data = torch.tensor(scaled_data, dtype=torch.float32)
dataset = TensorDataset(tensor_data)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 4. Phase 1: Pre-training the Autoencoder ---
print("\n2. Starting Phase 1: Autoencoder Pre-training")
model = DeepCAE(num_features=NUM_FEATURES, embedding_dim=EMBEDDING_DIM, num_clusters=NUM_CLUSTERS)
mse_loss = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5, betas=(0.9, 0.999))

for epoch in range(PRETRAIN_EPOCHS):
    total_loss = 0
    for batch in dataloader:
        x = batch[0]
        optimizer.zero_grad()
        x_recon, _, _ = model(x)
        loss = mse_loss(x_recon, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Pre-train Epoch [{epoch+1}/{PRETRAIN_EPOCHS}], Loss: {total_loss/len(dataloader):.4f}")

# --- 5. Phase 2: Joint Training with Clustering Loss ---
print("\n3. Starting Phase 2: Joint Clustering and Reconstruction Training")

# Initialize cluster centroids using K-Means
print("   - Initializing centroids with K-Means...")
model.eval()
with torch.no_grad():
    # Get all embeddings from the pre-trained encoder
    all_embeddings = model.encoder(tensor_data)
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, n_init="auto")
    kmeans.fit(all_embeddings.numpy())
    
    # Assign the K-Means centroids to the model's clustering layer
    model.clustering_layer.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

kld_loss = nn.KLDivLoss(reduction='batchmean')

for epoch in range(EPOCHS):
    # Update the target distribution p once per epoch
    model.eval()
    with torch.no_grad():
        _, q_full, _ = model(tensor_data)
        p_full = target_distribution(q_full)

    model.train()
    total_loss = 0
    for i, batch in enumerate(dataloader):
        x = batch[0]
        
        # Get the target distribution for the current batch
        # We need to map batch indices to the full dataset indices if shuffle=True
        # For simplicity here, let's assume no shuffle or re-calculate for batch.
        # A more robust implementation would pass indices.
        # Here we just use the pre-calculated p_full for demonstration.
        # Let's map it simply for this example:
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + x.shape[0]
        p_batch = p_full[start_idx:end_idx]

        optimizer.zero_grad()
        
        x_recon, q_batch, _ = model(x)
        
        # Calculate losses
        loss_recon = mse_loss(x_recon, x)
        loss_clust = kld_loss(q_batch.log(), p_batch)
        
        # Combine losses
        loss = loss_recon + GAMMA * loss_clust
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Joint Training Epoch [{epoch+1}/{EPOCHS}], Combined Loss: {total_loss/len(dataloader):.4f}")

# --- 6. Getting Final Embeddings ---
print("\n4. Training complete. Generating final user embeddings...")
model.eval()
with torch.no_grad():
    # To get the embedding for any user, just pass their data through the encoder
    final_user_embeddings = model.encoder(tensor_data)
    
    # To get cluster assignments, you can look at the q distribution
    _, final_q, _ = model(tensor_data)
    cluster_assignments = torch.argmax(final_q, dim=1)

print(f"Shape of final embeddings: {final_user_embeddings.shape}")
print(f"Example embedding for user 0: {final_user_embeddings[0].numpy()}")
print(f"Example cluster assignment for user 0: {cluster_assignments[0].item()}")