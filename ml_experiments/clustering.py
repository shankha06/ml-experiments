import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score

# 1. Setup Dummy Data (3 distinct topics: Space, Food, Animals)
sentences = [
    # Space
    "The sun is a huge ball of plasma.",
    "Planets orbit around stars in the galaxy.",
    "The moon controls the tides on Earth.",
    "Astronomers used a telescope to view the nebula.",
    "Mars is known as the Red Planet.",
    "NASA launched a new rover to explore the surface.",
    "Black holes have gravity so strong light cannot escape.",
    "The Milky Way is just one of many galaxies.",
    
    # Food
    "Pizza is a popular dish in Italy.",
    "I prefer apples over oranges for a snack.",
    "The chef prepared a delicious pasta sauce.",
    "Vegetables are essential for a healthy diet.",
    "Chocolate cake is my favorite dessert.",
    "Sushi is made with vinegared rice and fish.",
    "Baking bread requires flour, water, and yeast.",
    "Spicy tacos are best served with salsa.",
    
    # Animals
    "The cheetah is the fastest land animal.",
    "Elephants have large trunks and tusks.",
    "Dogs are often called man's best friend.",
    "The eagle soared high above the mountains.",
    "Whales are the largest mammals in the ocean.",
    "Cats are known for their agility and independence.",
    "The zoo has a new exhibit for tigers.",
    "Dolphins are highly intelligent marine creatures."
]

def generate_embeddings(text_data):
    print("Loading model and generating embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Normalize embeddings is crucial for Euclidean-based steps later
    embeddings = model.encode(text_data, normalize_embeddings=True)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings

def optimize_clustering(embeddings):
    """
    Grid search to find the best UMAP + HDBSCAN hyperparameters.
    We use Silhouette Score to judge 'optimality'.
    """
    print("\nStarting Hyperparameter Grid Search...")
    
    # Grid of parameters to test
    # n_neighbors: Controls how UMAP balances local vs global structure (Low = local, High = global)
    # min_cluster_size: The smallest size grouping that we consider a valid cluster
    umap_neighbors_range = [3, 5, 10, 15] 
    hdbscan_min_samples_range = [2, 3, 5]
    
    best_score = -1
    best_params = {}
    best_labels = []
    best_umap_projection = []
    
    # We iterate through combinations to find the best separation
    for n_neighbors in umap_neighbors_range:
        for min_samples in hdbscan_min_samples_range:
            
            # Step 1: UMAP Dimensionality Reduction
            # We reduce to 5 dimensions for clustering (dense enough for HDBSCAN)
            umap_reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=5, 
                min_dist=0.0,
                metric='cosine',
                random_state=42
            )
            umap_embeddings = umap_reducer.fit_transform(embeddings)
            
            # Step 2: HDBSCAN Clustering
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_samples,
                min_samples=1, # Keep small for small datasets
                metric='euclidean',
                cluster_selection_method='eom'
            )
            labels = clusterer.fit_predict(umap_embeddings)
            
            # Step 3: Validation
            # If all points are noise (-1) or all one cluster, skip score
            unique_labels = set(labels)
            if len(unique_labels) < 2:
                continue
                
            # Calculate Silhouette Score
            # Note: We calculate score on the ORIGINAL embeddings to ensure semantic validity,
            # not the reduced UMAP embeddings.
            # We filter out noise (-1) for a fair scoring
            core_samples_mask = labels != -1
            if np.sum(core_samples_mask) > 1: # Need at least 2 points to score
                score = silhouette_score(
                    embeddings[core_samples_mask], 
                    labels[core_samples_mask], 
                    metric='cosine'
                )
                
                if score > best_score:
                    best_score = score
                    best_params = {'n_neighbors': n_neighbors, 'min_cluster_size': min_samples}
                    best_labels = labels
                    # Save a 2D projection for plotting later
                    best_umap_projection = umap.UMAP(
                        n_neighbors=n_neighbors, n_components=2, min_dist=0.0, metric='cosine', random_state=42
                    ).fit_transform(embeddings)

    print(f"Optimization Complete.")
    print(f"Best Silhouette Score: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")
    return best_labels, best_umap_projection

# --- Execution ---

# 1. Get Embeddings
embeddings = generate_embeddings(sentences)

# 2. Find Optimal Clusters
labels, reduced_data = optimize_clustering(embeddings)

# 3. Analyze Results
df = pd.DataFrame({'Sentence': sentences, 'Cluster': labels})

# Count the number of clusters (excluding noise -1)
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"\n--- OPTIMAL RESULT FOUND ---")
print(f"Optimal Number of Clusters Detected: {num_clusters}")
print(f"Number of Outliers (Noise): {list(labels).count(-1)}")

# 4. Visualization
plt.figure(figsize=(10, 8))
# Plot clusters
clustered = (labels >= 0)
plt.scatter(reduced_data[clustered, 0], reduced_data[clustered, 1], 
            c=labels[clustered], cmap='Spectral', s=100, label='Clusters')

# Plot noise (outliers) in grey
if -1 in labels:
    noise = (labels == -1)
    plt.scatter(reduced_data[noise, 0], reduced_data[noise, 1], 
                c='grey', s=50, alpha=0.5, label='Noise/Outliers')

plt.title(f'Optimal Clustering: {num_clusters} Topics Detected', fontsize=16)
plt.legend()
plt.show()

# Print detailed view
print("\n--- Detailed Cluster Breakdown ---")
for cluster_id in sorted(list(set(labels))):
    if cluster_id == -1:
        print("\n[NOISE / OUTLIERS]:")
    else:
        print(f"\n[Cluster {cluster_id}]:")
    
    cluster_sentences = df[df['Cluster'] == cluster_id]['Sentence'].values
    for s in cluster_sentences:
        print(f"  - {s}")