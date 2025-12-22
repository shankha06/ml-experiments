import json
import logging
import numpy as np
from typing import List, Dict, Union
from dataclasses import asdict

from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("TaxonomyIO")

class TaxonomyIO:
    @staticmethod
    def save(nodes: List['ClusterNode'], filepath: str):
        """
        Serializes the list of ClusterNode objects to a JSON file.
        """
        logger.info(f"Saving {len(nodes)} taxonomy nodes to {filepath}...")
        
        # Convert dataclasses to dicts
        data = [asdict(node) for node in nodes]
        
        # Remove numpy arrays (embeddings) if they exist in the object, 
        # as they are not JSON serializable. We re-compute or save separately if needed.
        for item in data:
            if 'embedding' in item:
                del item['embedding']

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info("Save complete.")

    @staticmethod
    def load(filepath: str) -> List['ClusterNode']:
        """
        Loads nodes from JSON and reconstructs ClusterNode objects.
        """
        logger.info(f"Loading taxonomy from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        nodes = []
        for item in data:
            # Reconstruct the ClusterNode (assuming the class is available)
            # We explicitly handle the missing 'embedding' field
            node = ClusterNode(
                id=item['id'],
                level=item['level'],
                name=item['name'],
                description=item['description'],
                child_ids=item['child_ids'],
                embedding=None 
            )
            nodes.append(node)
            
        logger.info(f"Loaded {len(nodes)} nodes.")
        return nodes


class TaxonomyTagger:
    def __init__(self, taxonomy_nodes: List['ClusterNode'], embedding_model):
        """
        Args:
            taxonomy_nodes: The full list of nodes (L0 tags + L1/L2 categories).
            embedding_model: Your SBERT model.
        """
        self.nodes = taxonomy_nodes
        self.encoder = embedding_model
        self.node_map = {n.id: n for n in nodes}
        self.embeddings = None
        self.id_list = []
        
        # Build Index on Init
        self._build_index()

    def _build_index(self):
        """
        Creates embeddings for all nodes to enable vector search.
        We index 'Name' + 'Description' for maximum semantic coverage.
        """
        logger.info("Building inference index for taxonomy...")
        texts = []
        self.id_list = []
        
        for node in self.nodes:
            # Text representation: "Category Name: Category Definition"
            # This helps the embedding model understand the context.
            text = f"{node.name}: {node.description}"
            texts.append(text)
            self.id_list.append(node.id)
            
        self.embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        # Normalize for Cosine Similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / norms
        logger.info(f"Index built. Shape: {self.embeddings.shape}")

    def _get_lineage(self, node_id: str) -> List[str]:
        """
        Optional: Backtrack to find the full path (Root -> Leaf).
        Note: Since our structure stores children, getting parents requires a reverse map.
        For simple tagging, we might just return the node and its immediate level.
        """
        # (This requires a parent_id pointer in the Node, or a pre-computed map.
        #  For this snippet, we'll skip complex backtracking to keep it simple.)
        return []

    def predict(self, query: str, top_k: int = 3, filter_level: int = None) -> List[Dict]:
        """
        Predicts the most relevant tags/categories for a given query.
        
        Args:
            query: User input string.
            top_k: Number of matches to return.
            filter_level: If set (e.g., 1), only returns nodes from that specific hierarchy level.
                          Useful if you only want 'Categories' (Level 1) and not specific tags.
        """
        # 1. Embed Query
        query_vec = self.encoder.encode([query])
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
        
        # 2. Vector Search (Dot product since normalized)
        scores = np.dot(self.embeddings, query_vec.T).flatten()
        
        # 3. Sort and Filter
        # Get indices of top scores
        top_indices = np.argsort(scores)[::-1]
        
        results = []
        count = 0
        
        for idx in top_indices:
            if count >= top_k:
                break
                
            node = self.nodes[idx]
            
            # Apply Level Filter if requested
            if filter_level is not None and node.level != filter_level:
                continue
                
            # Formatting the result
            results.append({
                "tag_name": node.name,
                "tag_id": node.id,
                "level": node.level,
                "score": float(scores[idx]),
                "description": node.description
            })
            count += 1
            
        return results
    
if __name__ == "__main__":
    # --- 1. SETUP (Assume we have 'full_taxonomy' from the previous step) ---
    # For demo, creating a dummy list if you run this standalone
    if 'full_taxonomy' not in locals():
        full_taxonomy = [
            ClusterNode("L1_C1", 1, "Payment Disputes", "Issues regarding unauthorized charges.", ["L0_1"], None),
            ClusterNode("L0_1", 0, "chargeback", "process of disputing a charge", [], None)
        ]
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # --- 2. SAVE TAXONOMY ---
    io = TaxonomyIO()
    io.save(full_taxonomy, "chase_taxonomy_v1.json")

    # --- 3. LOAD & INITIALIZE INFERENCE ---
    loaded_nodes = io.load("chase_taxonomy_v1.json")
    tagger = TaxonomyTagger(loaded_nodes, embedding_model)

    # --- 4. PREDICT FOR NEW QUERY ---
    user_query = "I see a transaction on my statement that I didn't make."
    
    print(f"\nQuery: '{user_query}'")
    
    # Scenario A: Get High-Level Categories (Level 1)
    categories = tagger.predict(user_query, top_k=2, filter_level=1)
    print("\n--- Recommended Categories ---")
    for cat in categories:
        print(f"[{cat['score']:.4f}] {cat['tag_name']}: {cat['description']}")

    # Scenario B: Get Specific Tags (Level 0)
    tags = tagger.predict(user_query, top_k=2, filter_level=0)
    print("\n--- Recommended Specific Tags ---")
    for tag in tags:
        print(f"[{tag['score']:.4f}] {tag['tag_name']}")