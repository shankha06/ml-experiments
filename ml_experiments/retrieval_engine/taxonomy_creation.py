import logging
import json
import numpy as np
import pandas as pd
import umap.umap_ as umap
import hdbscan
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeepTaxonomy")

@dataclass
class ClusterNode:
    id: str
    level: int
    name: str
    description: str
    child_ids: List[str]
    embedding: np.ndarray = None

class RecursiveTaxonomyBuilder:
    def __init__(self, embedding_model, llm_wrapper_func, max_workers: int = 5):
        self.encoder = embedding_model
        self.call_llm = llm_wrapper_func
        self.max_workers = max_workers  # Control concurrency to respect Rate Limits
        
        self.umap_params = {
            'n_neighbors': 15, 
            'n_components': 5, 
            'metric': 'cosine'
        }
        self.hdbscan_params = {
            'min_cluster_size': 10,
            'min_samples': 5,
            'metric': 'euclidean',
            'cluster_selection_method': 'eom'
        }

    def _generate_cluster_metadata(self, items: List[str], level: int) -> Dict[str, Any]:
        """
        Worker function: Generates metadata for a single cluster.
        Includes retry logic and JSON parsing.
        """
        # ... (Same prompt logic as before) ...
        import random
        sample_items = items[:]
        random.shuffle(sample_items)
        context_items = sample_items[:40]

        prompt = f"""
You are the Principal Information Architect for Chase Bank. Synthesize these tags into a Category Name and Definition.

INPUT CONTEXT (Level {level}):
- Level 0 = Raw specific tags.
- Level 1+ = Descriptions of child clusters.

STRICT GUARDRAILS:
1. Terminology: Standard American Banking.
2. Conciseness: Name is 2-4 words.
3. No Fluff: Start description with definition immediately.
4. Handling Noise: If unrelated, label "Miscellaneous".

EXAMPLES:
Input: ["routing number", "swift code"] -> {{"category_name": "Account Identification", "description": "Identifiers for transaction routing.", "examples": ["routing number"]}}

CURRENT TAGS:
{", ".join(context_items)}

Output valid JSON:
"""
        try:
            response_text = self.call_llm(prompt=prompt, role="Principal Information Architect")
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            logger.error(f"LLM Error on Level {level}: {e}")
            return {
                "category_name": "General Banking Group", 
                "description": "Group created during error fallback.",
                "examples": context_items[:3]
            }

    def _process_single_cluster(self, label: int, child_nodes: List[ClusterNode], current_level: int) -> ClusterNode:
        """
        Helper wrapper to be submitted to the Executor. 
        Calculates inputs, calls LLM, and returns the finished Node.
        """
        child_ids = [n.id for n in child_nodes]
        child_names = [n.name for n in child_nodes]

        # Handling Noise (Label -1) without LLM call to save tokens/time
        if label == -1:
            meta = {
                "category_name": f"Miscellaneous Level {current_level}",
                "description": "Items that did not fit well into dense clusters at this level.",
                "examples": child_names[:3]
            }
        else:
            # The expensive call
            meta = self._generate_cluster_metadata(child_names, current_level)

        return ClusterNode(
            id=f"L{current_level+1}_C{label}",
            level=current_level + 1,
            name=meta.get('category_name', f"Cluster {label}"),
            description=meta.get('description', ""),
            child_ids=child_ids
        )

    def _cluster_vectors(self, embeddings: np.ndarray):
        """UMAP + HDBSCAN Pipeline"""
        if embeddings.shape[0] < 50:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
            return clusterer.fit_predict(embeddings)

        reducer = umap.UMAP(**self.umap_params)
        reduced_data = reducer.fit_transform(embeddings)
        
        clusterer = hdbscan.HDBSCAN(**self.hdbscan_params)
        return clusterer.fit_predict(reduced_data)

    def build_layer(self, nodes: List[ClusterNode], current_level: int) -> List[ClusterNode]:
        node_count = len(nodes)
        logger.info(f"--- Processing Layer {current_level} with {node_count} nodes ---")

        if node_count < 20: 
            logger.info("Stopping recursion: Too few nodes.")
            return []

        # 1. Embeddings (Vectorization is fast on GPU/CPU, keep on main thread)
        texts_to_embed = [n.description if n.description else n.name for n in nodes]
        embeddings = self.encoder.encode(texts_to_embed, show_progress_bar=False)

        # 2. Clustering
        labels = self._cluster_vectors(embeddings)
        
        # 3. Parallel Processing Preparation
        df = pd.DataFrame({'node_obj': nodes, 'label': labels})
        new_layer_nodes = []
        
        # We use a ThreadPoolExecutor to parallelize the LLM calls per cluster
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_label = {}
            
            # Submit tasks
            for label, group in df.groupby('label'):
                child_nodes = group['node_obj'].tolist()
                
                # We submit the task to the pool
                future = executor.submit(
                    self._process_single_cluster, 
                    label=label, 
                    child_nodes=child_nodes, 
                    current_level=current_level
                )
                future_to_label[future] = label

            # Process results as they complete
            logger.info(f"Waiting for {len(future_to_label)} cluster tasks to complete...")
            
            for future in as_completed(future_to_label):
                label = future_to_label[future]
                try:
                    node = future.result()
                    new_layer_nodes.append(node)
                except Exception as exc:
                    logger.error(f"Cluster {label} generated an exception: {exc}")

        logger.info(f"Layer {current_level} complete. Generated {len(new_layer_nodes)} new nodes.")

        # 4. Recursion
        parent_layer = self.build_layer(new_layer_nodes, current_level + 1)
        
        return new_layer_nodes + parent_layer

    def execute(self, raw_tags: List[str]):
        # Bootstrapping
        logger.info("Initializing Level 0 Nodes...")
        l0_nodes = []
        for i, tag in enumerate(raw_tags):
            l0_nodes.append(ClusterNode(
                id=f"L0_{i}",
                level=0,
                name=tag,
                description=tag,
                child_ids=[]
            ))

        hierarchy_nodes = self.build_layer(l0_nodes, current_level=0)
        return l0_nodes + hierarchy_nodes