import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Setup Production Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("TagEngine")

@dataclass
class CleaningResult:
    canonical_map: Dict[str, str]  # Map raw -> canonical
    unique_tags: List[str]         # Final list of clean tags
    stats: Dict[str, int]          # Metadata about the run

class TagCanonicalizer:
    def __init__(self, embedding_model, similarity_threshold: float = 0.88):
        """
        Args:
            embedding_model: Object with a .encode(texts) method (e.g., SentenceTransformer).
            similarity_threshold: Cosine similarity score above which tags are merged.
        """
        self.encoder = embedding_model
        self.threshold = similarity_threshold
        logger.info(f"Initialized Canonicalizer with threshold {self.threshold}")

    def _get_canonical_label(self, cluster_tags: List[str]) -> str:
        """
        Heuristic to pick the 'best' representative from a cluster of synonyms.
        Strategy: Prefer Title Case, then shortest length (less verbose), then alphabetical.
        """
        # Sort by: 1. Is Title Case (priority), 2. Length (shorter is often better for tags), 3. Alphabetical
        sorted_tags = sorted(
            cluster_tags, 
            key=lambda x: (not x.istitle(), len(x), x)
        )
        return sorted_tags[0]

    def process(self, raw_tags: List[str]) -> CleaningResult:
        logger.info(f"Starting canonicalization for {len(raw_tags)} tags.")
        
        # 1. Preprocessing
        unique_raw = list(set([t.strip() for t in raw_tags if t.strip()]))
        if not unique_raw:
            logger.warning("No valid tags provided.")
            return CleaningResult({}, [], {"input": 0, "output": 0})

        # 2. Vectorization
        try:
            logger.info("Generating embeddings...")
            embeddings = self.encoder.encode(unique_raw, show_progress_bar=False)
            # Ensure normalized vectors for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise e

        # 3. Graph Construction
        # Calculate cosine similarity matrix
        sim_matrix = np.dot(embeddings, embeddings.T)
        
        # Build graph where edges exist if similarity > threshold
        # Using numpy operations to find indices avoids slow loops
        logger.info("Building similarity graph...")
        graph = nx.Graph()
        graph.add_nodes_from(unique_raw)
        
        # Get upper triangle indices where similarity > threshold (excluding diagonal)
        rows, cols = np.where(np.triu(sim_matrix, k=1) > self.threshold)
        
        edges = []
        for r, c in zip(rows, cols):
            edges.append((unique_raw[r], unique_raw[c]))
        
        graph.add_edges_from(edges)
        logger.debug(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")

        # 4. Community Detection (Connected Components)
        # Each connected component is a group of synonyms
        canonical_map = {}
        final_tags = []
        
        components = list(nx.connected_components(graph))
        logger.info(f"Identified {len(components)} unique semantic clusters.")

        for cluster in components:
            cluster_list = list(cluster)
            if len(cluster_list) == 1:
                canonical = cluster_list[0]
            else:
                canonical = self._get_canonical_label(cluster_list)
                logger.debug(f"Merged {cluster_list} -> {canonical}")
            
            final_tags.append(canonical)
            for tag in cluster_list:
                canonical_map[tag] = canonical

        reduction_pct = (1 - (len(final_tags) / len(unique_raw))) * 100
        logger.info(f"Canonicalization complete. Reduced tags by {reduction_pct:.2f}%.")

        return CleaningResult(
            canonical_map=canonical_map,
            unique_tags=final_tags,
            stats={"input_unique": len(unique_raw), "output_unique": len(final_tags)}
        )