import json
import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Setup Production Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("TagEngine")

class TaxonomyBuilder:
    def __init__(self, embedding_model, llm_wrapper_func, cluster_threshold: float = 1.5):
        """
        Args:
            embedding_model: SBERT compatible model.
            llm_wrapper_func: Function signature: func(prompt, role) -> str
            cluster_threshold: Distance threshold for AgglomerativeClustering. 
                               Lower = more granular clusters (more specific categories).
        """
        self.encoder = embedding_model
        self.call_llm = llm_wrapper_func
        self.distance_threshold = cluster_threshold
        logger.info("Initialized TaxonomyBuilder.")

    def _generate_category_name(self, tags: List[str]) -> str:
        """Calls LLM to generalize a list of tags into a Category Name."""
        prompt = (
            "You are a taxonomy expert. I will provide a list of related tags. "
            "Your task is to provide a SINGLE, short, high-level Category Name that encompasses these tags.\n"
            "Rules:\n"
            "1. Output ONLY the category name. No explanations.\n"
            "2. The name should be a noun phrase (e.g., 'Financial Instruments', not 'Using money').\n"
            "3. Be specific but inclusive.\n\n"
            f"Tags: {', '.join(tags[:20])}" # Limit context window just in case
        )
        
        try:
            response = self.call_llm(
                prompt=prompt, 
                role="Senior Information Architect"
            )
            return response.strip().replace('"', '').replace('.', '')
        except Exception as e:
            logger.error(f"LLM call failed for tags {tags[:3]}...: {e}")
            return "Uncategorized"

    def build_hierarchy(self, canonical_tags: List[str]) -> Dict:
        logger.info(f"Building taxonomy for {len(canonical_tags)} tags.")
        
        if len(canonical_tags) < 2:
            return {"Root": canonical_tags}

        # 1. Embeddings
        embeddings = self.encoder.encode(canonical_tags, show_progress_bar=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # 2. Perform Hierarchical Clustering
        # We use distance_threshold=None and n_clusters=None to explore the tree if needed,
        # but here we set distance_threshold to cut the tree at a semantic depth.
        # Affinity='cosine' is ideal, but sklearn generic supports euclidean. 
        # Since vectors are normalized, Euclidean distance is proportional to Cosine distance.
        logger.info("Clustering tags...")
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric='euclidean', # For normalized vectors, this works as proxy for cosine
            linkage='ward'
        )
        clustering.fit(embeddings)

        # 3. Group Tags by Cluster Label
        cluster_map = defaultdict(list)
        for tag, label in zip(canonical_tags, clustering.labels_):
            cluster_map[label].append(tag)
        
        logger.info(f"Derived {len(cluster_map)} initial tag clusters.")

        # 4. LLM Labeling Step
        taxonomy = {}
        for label_id, tags_in_cluster in cluster_map.items():
            # If cluster is too small (noise), might just keep tags as is or generalize
            if len(tags_in_cluster) == 1:
                category_name = "Misc" 
            else:
                category_name = self._generate_category_name(tags_in_cluster)
            
            logger.info(f"Cluster {label_id} ({len(tags_in_cluster)} tags) -> Category: '{category_name}'")
            
            # Structure: Category -> List of Tags
            # In a recursive version, we would cluster 'tags_in_cluster' again if too large.
            if category_name in taxonomy:
                taxonomy[category_name].extend(tags_in_cluster)
            else:
                taxonomy[category_name] = tags_in_cluster

        return taxonomy
    
# --- MOCK DEPENDENCIES ---
class MockSBERT:
    def encode(self, texts, show_progress_bar=False):
        # Return random vectors for demo purposes
        return np.random.rand(len(texts), 384)

def mock_llm_wrapper(prompt, role):
    # Simulate an LLM response based on keywords in prompt
    if "neural" in prompt or "transformer" in prompt:
        return "Deep Learning Architectures"
    if "invoice" in prompt or "tax" in prompt:
        return "Financial Operations"
    return "General Concepts"

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Setup Data
    raw_unbounded_tags = [
        "neural networks", "Neural Networks", "deep learning", "Deep Learning", # Synonyms
        "invoice processing", "invoicing", "taxation", "Tax Deducted",          # Finance
        "transformers", "LLMs", "BERT", "GPT-4"                                # Specific Models
    ]

    # 2. Initialize Models
    embedding_model = MockSBERT() # Replace with SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Step A: Clean the Tags
    cleaner = TagCanonicalizer(embedding_model)
    clean_result = cleaner.process(raw_unbounded_tags)
    
    print("\n--- Canonicalization Result ---")
    print(f"Original Count: {len(raw_unbounded_tags)}")
    print(f"Unique Count: {clean_result.stats['output_unique']}")
    print(f"Sample Map: {dict(list(clean_result.canonical_map.items())[:3])}")

    # 4. Step B: Build Taxonomy from Clean Tags
    builder = TaxonomyBuilder(embedding_model, mock_llm_wrapper)
    taxonomy_tree = builder.build_hierarchy(clean_result.unique_tags)

    print("\n--- Generated Taxonomy ---")
    print(json.dumps(taxonomy_tree, indent=2))