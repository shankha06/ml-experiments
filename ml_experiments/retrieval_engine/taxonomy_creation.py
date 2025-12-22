import logging
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass

# Manifold & Clustering
import umap.umap_ as umap
import hdbscan

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TaxonomyEngine")

@dataclass
class TaxonomyNode:
    name: str
    description: str
    child_tags: List[str]
    vector: np.ndarray = None  # Embedding of the description

class AdvancedTaxonomyBuilder:
    def __init__(self, embedding_model, llm_wrapper_func, min_cluster_size: int = 5):
        """
        Args:
            embedding_model: SBERT compatible model.
            llm_wrapper_func: Function signature: func(prompt, role) -> str (JSON string)
            min_cluster_size: Minimum number of tags to form a valid cluster in HDBSCAN.
        """
        self.encoder = embedding_model
        self.call_llm = llm_wrapper_func
        self.min_cluster_size = min_cluster_size
        
        # UMAP for dimensionality reduction (Crucial for HDBSCAN performance)
        self.reducer = umap.UMAP(
            n_neighbors=15, 
            n_components=10, # Reduce to 10 dimensions for clustering
            metric='cosine',
            random_state=42
        )
        
        # HDBSCAN for density-based clustering
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=2, # Conservative noise handling
            metric='euclidean', # UMAP output is Euclidean space
            cluster_selection_method='eom' # Excess of Mass usually best for text
        )

    def _generate_cluster_metadata(self, tags: List[str]) -> Dict[str, str]:
        """
        Calls LLM to generate Name and Description for a cluster of tags.
        Expects structured JSON output.
        """
        prompt = (
            "You are a Senior Data Catalog Specialist for a Retail & Dining platform. "
            "I will provide a list of tags belonging to a specific product cluster. "
            "Analyze them to define the category.\n\n"
            "Input Tags:\n" + ", ".join(tags[:25]) + "\n\n" # Limit context
            "Requirements:\n"
            "1. 'category_name': A concise, standard industry term (e.g., 'Appetizers', 'Men's Footwear').\n"
            "2. 'description': A clear definition of what belongs here, used for semantic matching.\n"
            "3. Output MUST be valid JSON.\n\n"
            "Example JSON Output:\n"
            "{\n"
            "  \"category_name\": \"Vegan Entrees\",\n"
            "  \"description\": \"Plant-based main course dishes excluding meat, dairy, and animal by-products.\"\n"
            "}"
        )
        
        try:
            response_str = self.call_llm(
                prompt=prompt, 
                role="Taxonomy Architect"
            )
            # Clean generic markdown fences if present
            clean_json = response_str.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON for cluster: {tags[:3]}")
            return {"category_name": "Uncategorized Cluster", "description": "Auto-generation failed."}
        except Exception as e:
            logger.error(f"LLM Failure: {e}")
            return {"category_name": "Error", "description": "Error in processing."}

    def fit_transform(self, canonical_tags: List[str]) -> List[TaxonomyNode]:
        logger.info(f"Starting Taxonomy Build for {len(canonical_tags)} tags.")
        
        if len(canonical_tags) < self.min_cluster_size:
            logger.warning("Not enough tags to cluster. Returning single root node.")
            return []

        # 1. Vectorize (High-Dim)
        logger.info("Generating SBERT embeddings...")
        high_dim_embeddings = self.encoder.encode(canonical_tags, show_progress_bar=False)

        # 2. Dimensionality Reduction (UMAP)
        # HDBSCAN struggles with high dimensionality (curse of dimensionality). 
        # Reducing to ~10-15 dims preserves local structure while making density apparent.
        logger.info("Reducing dimensions with UMAP...")
        low_dim_embeddings = self.reducer.fit_transform(high_dim_embeddings)

        # 3. Density Clustering (HDBSCAN)
        logger.info("Clustering with HDBSCAN...")
        labels = self.clusterer.fit_predict(low_dim_embeddings)
        
        # Analyze distribution
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        logger.info(f"Found {n_clusters} clusters. {n_noise} tags identified as noise (-1).")

        # 4. Grouping & LLM Enrichment
        # We group tags by their HDBSCAN label
        df = pd.DataFrame({'tag': canonical_tags, 'label': labels})
        
        taxonomy_nodes = []

        for label_id in unique_labels:
            if label_id == -1:
                continue # Skip noise for now, or handle separately

            cluster_tags = df[df['label'] == label_id]['tag'].tolist()
            
            # Ask LLM for Name/Description
            logger.info(f"Processing Cluster {label_id} ({len(cluster_tags)} tags)...")
            metadata = self._generate_cluster_metadata(cluster_tags)
            
            # 5. Create 'Upper Layer' Embedding
            # We embed the DESCRIPTION of the category. This is powerful because
            # it captures the *intent* of the category, not just the keywords.
            cat_vector = self.encoder.encode(metadata['description'])
            
            node = TaxonomyNode(
                name=metadata['category_name'],
                description=metadata['description'],
                child_tags=cluster_tags,
                vector=cat_vector
            )
            taxonomy_nodes.append(node)

        return taxonomy_nodes

# --- MOCKING INFRASTRUCTURE ---

class MockSBERT:
    def encode(self, texts, show_progress_bar=False):
        # Return structured noise to simulate clusters for UMAP/HDBSCAN
        # If input is a single string (description), return 1 vector
        if isinstance(texts, str):
            return np.random.rand(384)
        
        count = len(texts)
        # Create 3 distinct blobs of data to ensure HDBSCAN finds something
        if count > 10:
            blob1 = np.random.normal(0, 0.1, (count // 3, 384))
            blob2 = np.random.normal(5, 0.1, (count // 3, 384))
            blob3 = np.random.normal(10, 0.1, (count - 2 * (count // 3), 384))
            return np.vstack([blob1, blob2, blob3])
        return np.random.rand(count, 384)

def mock_structured_llm_wrapper(prompt, role):
    """
    Simulates a smart LLM returning JSON based on context clues.
    """
    if "sushi" in prompt.lower() or "tempura" in prompt.lower():
        return json.dumps({
            "category_name": "Japanese Cuisine",
            "description": "Traditional dishes from Japan including raw fish, rice, and battered frying."
        })
    elif "shirt" in prompt.lower() or "denim" in prompt.lower():
        return json.dumps({
            "category_name": "Men's Casual Wear",
            "description": "Informal clothing items for men including tops and bottoms suitable for daily wear."
        })
    else:
        return json.dumps({
            "category_name": "General Retail",
            "description": "Miscellaneous retail products."
        })

# --- MAIN EXECUTION FLOW ---

if __name__ == "__main__":
    # 1. Dataset: Retail/Dining Tags
    tags = [
        # Cluster 1: Japanese Food
        "sushi roll", "nigiri", "sashimi", "tempura shrimp", "miso soup", "wasabi", "soy sauce",
        # Cluster 2: Clothing
        "denim jeans", "cotton t-shirt", "flannel shirt", "cargo pants", "hoodie", "polo shirt",
        # Noise
        "random database error", "server timeout"
    ]
    
    # 2. Init
    model = MockSBERT()
    builder = AdvancedTaxonomyBuilder(
        embedding_model=model, 
        llm_wrapper_func=mock_structured_llm_wrapper,
        min_cluster_size=3 # Low number for this tiny demo
    )

    # 3. Execute
    nodes = builder.fit_transform(tags)

    # 4. Inspect Results
    print(f"\nSuccessfully built {len(nodes)} Taxonomy Nodes.\n")
    
    for i, node in enumerate(nodes):
        print(f"--- Node {i+1}: {node.name} ---")
        print(f"Description: {node.description}")
        print(f"Vector Shape: {node.vector.shape}")
        print(f"Child Tags: {node.child_tags[:5]}...")
        print("")

# # --- MAIN EXECUTION ---
# if __name__ == "__main__":
#     # 1. Setup Data
#     raw_unbounded_tags = [
#         "neural networks", "Neural Networks", "deep learning", "Deep Learning", # Synonyms
#         "invoice processing", "invoicing", "taxation", "Tax Deducted",          # Finance
#         "transformers", "LLMs", "BERT", "GPT-4"                                # Specific Models
#     ]

#     # 2. Initialize Models
#     embedding_model = MockSBERT() # Replace with SentenceTransformer('all-MiniLM-L6-v2')

#     # 3. Step A: Clean the Tags
#     cleaner = TagCanonicalizer(embedding_model)
#     clean_result = cleaner.process(raw_unbounded_tags)
    
#     print("\n--- Canonicalization Result ---")
#     print(f"Original Count: {len(raw_unbounded_tags)}")
#     print(f"Unique Count: {clean_result.stats['output_unique']}")
#     print(f"Sample Map: {dict(list(clean_result.canonical_map.items())[:3])}")

#     # 4. Step B: Build Taxonomy from Clean Tags
#     builder = TaxonomyBuilder(embedding_model, mock_llm_wrapper)
#     taxonomy_tree = builder.build_hierarchy(clean_result.unique_tags)

#     print("\n--- Generated Taxonomy ---")
#     print(json.dumps(taxonomy_tree, indent=2))