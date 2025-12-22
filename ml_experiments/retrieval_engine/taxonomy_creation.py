import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import hdbscan
import numpy as np
import pandas as pd
import umap.umap_ as umap

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeepTaxonomy")

@dataclass
class ClusterNode:
    id: str
    level: int
    name: str
    description: str
    child_ids: List[str]  # IDs of nodes in the layer below
    embedding: np.ndarray = None

class RecursiveTaxonomyBuilder:
    def __init__(self, embedding_model, llm_wrapper_func):
        self.encoder = embedding_model
        self.call_llm = llm_wrapper_func
        
        # Hyperparameters for 25K scale
        self.umap_params = {
            'n_neighbors': 10,
            'n_components': 10,
            'metric': 'cosine'
        }
        self.hdbscan_params = {
            'min_cluster_size': 10, # Minimum size to be considered a cluster
            'min_samples': 5,       # Controls how conservative the clustering is
            'metric': 'cosine',
            'cluster_selection_method': 'eom' # Excess of Mass (often better for variable densities)
        }

    def _generate_cluster_metadata(self, items: List[str], level: int) -> Dict[str, str]:
        """
        Calls LLM to get structured Name and Description for a list of items.
        """
        # Context varies: Level 0 are raw tags, Level 1+ are descriptions of previous clusters
        context_items = items[:30] # Limit context
        
        prompt = (
            f"You are a Retail and Dining Taxonomy Architect. I have a cluster of {len(items)} items. "
            "Analyze them and define the specific category they belong to.\n"
            "ITEMS:\n" + ", ".join(context_items) + "\n\n"
            "REQUIREMENTS:\n"
            "1. Output valid JSON only.\n"
            "2. 'category_name': A short, precise business label (e.g., 'Gluten-Free Pastries').\n"
            "3. 'description': A detailed definition explaining what this category covers (used for semantic matching).\n"
            "4. 'examples': 3 representative examples from the list.\n"
            f"Context Level: {level} (Lower is more specific, Higher is broader)."
        )

        try:
            # Assuming wrapper returns a dict or JSON string
            response = self.call_llm(prompt=prompt, role="Taxonomy Architect")
            if isinstance(response, str):
                return json.loads(response)
            return response
        except Exception as e:
            logger.error(f"LLM Error on Level {level}: {e}")
            return {
                "category_name": "Uncategorized Group", 
                "description": "Automatic grouping of diverse items.",
                "examples": items[:3]
            }

    def _cluster_vectors(self, embeddings: np.ndarray):
        """
        Performs UMAP reduction followed by HDBSCAN clustering.
        """
        # 1. Dimensionality Reduction (Critical for HDBSCAN performance)
        # If data is small (<100), skip UMAP or use lower neighbors
        if embeddings.shape[0] < 50:
             # Fallback for top layers where N is small
            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
            labels = clusterer.fit_predict(embeddings)
            return labels

        reducer = umap.UMAP(**self.umap_params)
        reduced_data = reducer.fit_transform(embeddings)

        # 2. Density Clustering
        clusterer = hdbscan.HDBSCAN(**self.hdbscan_params)
        labels = clusterer.fit_predict(reduced_data)
        
        # Note: HDBSCAN returns -1 for noise. 
        # Strategy: We will treat -1 as a "Misc" bucket for this iteration.
        return labels

    def build_layer(self, nodes: List[ClusterNode], current_level: int) -> List[ClusterNode]:
        """
        Recursive function to build one layer of the taxonomy.
        nodes: The nodes from the previous layer (or raw tags wrapper as nodes).
        """
        node_count = len(nodes)
        logger.info(f"--- Processing Layer {current_level} with {node_count} nodes ---")

        # STOPPING CONDITION: If we have too few nodes to cluster effectively
        if node_count < 20: 
            logger.info("Stopping recursion: Too few nodes to cluster further.")
            return []

        # 1. Prepare Embeddings for this layer
        # If Level 0, we embed the tag names.
        # If Level > 0, we embed the DESCRIPTION of the child node.
        texts_to_embed = [n.description if n.description else n.name for n in nodes]
        embeddings = self.encoder.encode(texts_to_embed, show_progress_bar=False)

        # 2. Cluster
        labels = self._cluster_vectors(embeddings)
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        logger.info(f"Layer {current_level}: Found {n_clusters} clusters (plus noise).")

        # 3. Process Clusters into New Parent Nodes
        new_layer_nodes = []
        
        # Group indices by label
        df = pd.DataFrame({'node_obj': nodes, 'label': labels})
        
        for label, group in df.groupby('label'):
            # Get the child nodes belonging to this cluster
            child_nodes = group['node_obj'].tolist()
            child_ids = [n.id for n in child_nodes]
            
            # Extract text representation for LLM
            # For the prompt, we use the NAMES of the children
            child_names = [n.name for n in child_nodes]

            if label == -1:
                # Handling Noise: For now, we create a "Misc" node, 
                # or you could choose to pass these directly to the next layer (orphan adoption).
                meta = {
                    "category_name": f"Miscellaneous Level {current_level}",
                    "description": "Items that did not fit well into dense clusters at this level.",
                    "examples": child_names[:3]
                }
            else:
                meta = self._generate_cluster_metadata(child_names, current_level)

            # Create the Parent Node
            new_node = ClusterNode(
                id=f"L{current_level+1}_C{label}", # Unique ID
                level=current_level + 1,
                name=meta.get('category_name'),
                description=meta.get('description'),
                child_ids=child_ids
            )
            new_layer_nodes.append(new_node)

        # 4. Recursion
        # The new parent nodes become the input for the next layer up
        parent_layer = self.build_layer(new_layer_nodes, current_level + 1)
        
        # Return current layer nodes + any nodes created above them
        return new_layer_nodes + parent_layer

    def execute(self, raw_tags: List[str]):
        # Bootstrapping Level 0 Nodes
        logger.info("Initializing Level 0 Nodes...")
        l0_nodes = []
        for i, tag in enumerate(raw_tags):
            l0_nodes.append(ClusterNode(
                id=f"L0_{i}",
                level=0,
                name=tag,
                description=tag, # Raw tags description is themselves
                child_ids=[]     # Leaf nodes have no children
            ))

        # Start Recursion
        hierarchy_nodes = self.build_layer(l0_nodes, current_level=0)
        
        # Combine L0 and upper layers for full flat list (or organize into tree)
        all_nodes = l0_nodes + hierarchy_nodes
        return all_nodes

# --- MOCKING & USAGE ---

class MockSBERT:
    def encode(self, texts, show_progress_bar=False):
        # Return random 384-dim vectors
        return np.random.rand(len(texts), 384)

def mock_structured_llm(prompt, role):
    """
    Simulates a structured JSON response.
    """
    # Logic to simulate different responses based on input keywords
    if "sushi" in prompt.lower() or "burger" in prompt.lower():
        cat = "Fast Food & Dining"
        desc = "Establishments and products related to ready-to-eat meals and dining experiences."
    elif "shirt" in prompt.lower() or "denim" in prompt.lower():
        cat = "Apparel & Fashion"
        desc = "Clothing items including tops, bottoms, and outerwear for men and women."
    else:
        cat = "General Retail"
        desc = "Various consumer goods available for purchase."

    return {
        "category_name": cat,
        "description": desc,
        "examples": ["Example A", "Example B", "Example C"]
    }

if __name__ == "__main__":
    # 1. Generate Fake Data (200 tags to test flow, usually you'd have 25k)
    # Using repetition to simulate clusters
    tags = ["Sushi Roll"] * 50 + ["Cheese Burger"] * 50 + ["Denim Jacket"] * 50 + ["Cotton Shirt"] * 50
    
    # 2. Initialize
    # Replace MockSBERT with SentenceTransformer('all-MiniLM-L6-v2')
    # Replace mock_structured_llm with your actual LLM call
    builder = RecursiveTaxonomyBuilder(MockSBERT(), mock_structured_llm)
    
    # 3. Build
    logger.info("Starting Taxonomy Build...")
    full_taxonomy = builder.execute(tags)
    
    # 4. Inspect Output
    print(f"\nTotal Nodes Created: {len(full_taxonomy)}")
    
    # Filter for Level 1 Nodes (First Clustering Layer)
    l1_nodes = [n for n in full_taxonomy if n.level == 1]
    print(f"\n--- Level 1 Clusters (Expected ~1k-2k in prod) ---")
    for n in l1_nodes[:3]:
        print(f"ID: {n.id} | Name: {n.name}")
        print(f"Desc: {n.description}")
        print(f"Children Count: {len(n.child_ids)}\n")

    # Filter for Level 2 Nodes (Higher Level)
    l2_nodes = [n for n in full_taxonomy if n.level == 2]
    print(f"\n--- Level 2 Clusters (Expected ~500 in prod) ---")
    for n in l2_nodes[:3]:
        print(f"ID: {n.id} | Name: {n.name}")
        print(f"Desc: {n.description} (Used for embedding)\n")

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