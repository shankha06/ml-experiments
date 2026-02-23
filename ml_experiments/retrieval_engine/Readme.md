Here is the technical project documentation for the Chase Smart Offer Search system.

# Project: Hybrid Semantic Search for Merchant Offers

## 1. Technical Overview

The primary objective of this project is to engineer a high-precision Hybrid Retrieval System for Chase credit card offers. The legacy system relies on exact lexical matching (searching "Starbucks" matches "Starbucks"), which suffers from low recall when user intent varies (e.g., searching "morning coffee" returns zero results for "Starbucks").

The proposed solution implements a **Hybrid Retrieval System** combining Dense Vector Search (Semantic), Sparse Neural Search (SPLADE), and Traditional Lexical Search (BM25/Whoosh), unified via Reciprocal Rank Fusion (RRF). The system operates within the strict governance constraints of the JP Morgan ecosystem, necessitating the use of approved models (`all-mpnet-base-v2` for embeddings and `Mistral-7B-v0.2` for SLM tasks). This1 moves beyond simple keyword matching by implementing a "Recall-Precision Cascade." This structure retrieves a broad set of candidates using multiple distinct algorithms (Recall Layer) and then progressively refines them using computationally expensive but highly intelligent models (Precision Layer).

### System Architecture & Inference Pipeline

The inference flow is designed as a four-layer cascade to balance latency and precision.

**Layer 1: Query Understanding (NER)**
The raw user query is processed by a Named Entity Recognition (NER) module. This extracts critical entities such as `Merchant_Name` (e.g., "Home Depot"), `Product_Category` (e.g., "Gardening"), `Location`, and `Date`. This structured intent aids downstream filtering.

**Layer 2: Ensemble Candidate Generation**
To maximize recall, candidates are retrieved in parallel from five distinct sources:

1. **Semantic Search (ANN):** Uses an FAISS index built on `mpnet` embeddings of offer descriptions. Captures conceptual similarity.
2. **Semantic Search (Exact L2):** A brute-force Cosine Similarity search using Torch on a smaller subset, ensuring no approximation errors for high-value queries.
3. **SPLADE Search:** Uses the `SPLADE-v2` sparse index. This captures specific keywords that standard dense embeddings might "smooth over" (e.g., specific model numbers or rare brand names).
4. **Lexical Search (BM25):** Built on the `Whoosh` library. Weighted heavily towards the `merchant_name` field to ensure exact name matches always appear (e.g., searching "Uber" must return "Uber").
5. **Tag-Based Caching:** A hierarchical cache lookup:
* *Level 1:* Exact query match.
* *Level 2:* Fuzzy match (>95% similarity).
* *Level 3 (Fallback):* If no cache hit, the query is sent to `Mistral-7B` to generate potential tags. These tags are mapped to the "Inverse Tag Index" to retrieve popular offers associated with those keywords.



**Layer 3: Fusion & Filtering**
The candidates from Layer 2 are segregated into two pools:

* **Overlapping Candidates:** Offers that appear in the high-confidence Tag Search are prioritized.
* **Non-Overlapping Candidates:** Offers found only by vector/lexical search.
Both pools undergo **Reciprocal Rank Fusion (RRF)** to assign a unified ranking score without requiring complex score normalization (see Background Study).

**Layer 4: SLM Re-Ranking**
The top 5 fused candidates are sent to the `Mistral-7B` Small Language Model (SLM). The SLM is prompted to strictly rank these offers based on relevance to the user's query. The final output is a hand-crafted list starting with the exact lexical match (if any), followed by the SLM-ranked offers, and finally the remaining RRF candidates.

---

Here is the detailed technical breakdown of the **Offline Data Enrichment and Indexing Process**. This phase constitutes the "back-end" data engineering pipeline, ensuring that the raw, sparse merchant catalog is transformed into a rich, searchable vector space before any user query is received.

## 3. Background Study

### A. Sentence Transformer Fine-Tuning (MNRL with Triplets)

Standard BERT models produce embeddings that are not suitable for cosine similarity comparison out-of-the-box. To fix this, we fine-tune `all-mpnet-base-v2` using a **Siamese Network** architecture with **Multiple Negatives Ranking Loss (MNRL)**.

#### How it works

In a Siamese network, the Anchor (), Positive (), and Negative () sentences are passed through the *same* BERT model to produce embeddings , , and .

The **Multiple Negatives Ranking Loss** is a variation of Cross-Entropy Loss. For a given batch of  triplets, the model assumes that for an anchor , only  is the positive match. Critically, it treats **all other positives** ( where ) and **all explicit negatives** () in the batch as "negatives" for .

The objective is to maximize the similarity between  while minimizing the similarity between  and all other candidates.

Where  is a temperature hyperparameter scaling the logits.

#### Example Scenario

* **Anchor ():** "discount on running shoes"
* **Positive ():** "Nike Store Offer: Save 10% on athletic footwear and apparel."
* **Hard Negative ():** "Tire Rack Offer: Best deals on all-season tires and wheels." (Selected because "shoes" and "tires" can both appear in "rubber" or "tread" contexts, or simply as a random hard negative from a different category).

**Training Step:**

1. The model computes vector  (Anchor),  (Positive), and  (Negative).
2. It calculates  (aiming for 1.0) and  (aiming for 0.0).
3. The loss function penalizes the model heavily if  is closer to  than to .
4. By using "Hard Negatives" (triplets) rather than just pairs, the model learns fine-grained distinctions (e.g., distinguishing "running shoes" from "car tires" or "dress shoes") rather than just broad topic clustering.

### B. SPLADE (Sparse Lexical and Expansion Model)

While dense models (like mpnet) excel at semantic understanding, they often suffer from the "lexical gap"—failing to match exact model numbers or rare entities.

SPLADE solves this by performing **Learned Sparse Retrieval**. Instead of a dense vector (e.g., 768 float dimensions), SPLADE projects the text into the full BERT vocabulary space (approx 30,000 dimensions) but ensures the result is sparse (mostly zeros).

* **Expansion:** It activates terms *not present* in the text but semantically relevant. For a document containing "Apple", SPLADE might activate the dimension for "iPhone" or "Mac", allowing a user searching for "iPhone" to match the document even if the word is missing.
* **Sparsity:** A regularization term during training forces most weights to zero, making the resulting index compatible with inverted index structures (like Lucene/Whoosh) for extremely fast retrieval.

### C. Reciprocal Rank Fusion (RRF)

We use RRF to merge results from the ensemble (Vector, SPLADE, BM25) because these systems output scores on fundamentally different scales (e.g., Cosine is 0-1, BM25 can be 0-100+).

RRF ignores the absolute scores and relies solely on the **rank** of the item in each list. The score for a document  is calculated as:

Where  is a smoothing constant (typically 60). This method ensures that a document appearing at Rank 1 in *one* system doesn't dominate a document appearing at Rank 2 in *all three* systems, providing a robust consensus ranking.

---
# Dataset and Indexing

This process is divided into three distinct stages: **Data Enrichment**, **Synthetic Data Generation/Model Training**, and **Index Construction**.

## 1. Data Enrichment Phase

**Objective:** The raw merchant data (`pre_msg_tx`) is marketing-centric and sparse, lacking the semantic richness required for deep retrieval. We employ Generative AI (GPT-4 via Azure API) to hallucinate grounded metadata that bridges the gap between merchant offerings and natural language user queries.

**Code Location:** `chasesmartoffersearch/data_enrichment/description_enrichment.py`

### A. Description Generation

We iterate through the raw `chase_offers_data.csv`. For each `mrch_nm` (Merchant Name) and `offr_cat_nm` (Category), we prompt GPT-4 to generate a detailed profile.

* **Input:**
* Merchant: "Blue Apron"
* Category: "Food & Drink"


* **Prompt Strategy:** "Generate a detailed 50-word description of the products and services offered by this merchant. Include specific items they sell and the customer intent they satisfy."
* **Output (`offr_description`):** *"Meal kit delivery service providing fresh ingredients and chef-designed recipes for home cooking. Offers weekly subscription boxes containing meat, vegetables, and pasta. Ideal for busy professionals seeking convenient, healthy dinner options without grocery shopping."*
* **Technical Value:** This introduces terms like "subscription," "vegetables," "dinner," and "healthy" into the searchable text, which were absent in the raw data.

### B. Tag Extraction

Simultaneously, we ask the LLM to generate a list of "search tags."

* **Output (`tags`):** `["meal kit", "grocery delivery", "cooking subscription", "dinner", "fresh food"]`
* **Technical Value:** These tags serve as "semantic anchors." Even if the dense vector drifts, these discrete tokens allow for high-precision tag matching in Layer 2 of the inference pipeline.

---

## 2. Synthetic Data & Model Fine-Tuning

**Objective:** To train the `all-mpnet-base-v2` model to understand the specific domain of Chase offers. Since we lack historical clickstream data (user queries mapped to clicked offers), we generate a **Synthetic Ground Truth**.

### A. Synthetic Query Generation

**Location:** `data/input/chase_synthetic_queries.csv`
We reverse-engineer the search problem. Instead of waiting for users to search, we ask the LLM: *"What would a user type into a search bar to find this specific offer description?"*

* **Input:** The enriched `offr_description` from Step 1.
* **Output:** 5-10 variations of queries per offer.
* Short-tail: "meal kits"
* Long-tail: "best food delivery for cooking at home"
* Natural Language: "I want to cook dinner but don't have time to shop"



### B. Triplet Construction (Synthetic Hard Negative Mining)

**Location:** `data/training/triplet_training.csv`

To train the Sentence Transformer using **Multiple Negatives Ranking Loss (MNRL)**, we construct triplets . Unlike standard approaches that rely on existing catalog descriptions, this methodology utilizes **purpose-built synthetic texts** for both positive and negative samples to maximize the model's ability to discern subtle intent.

1. **Anchor ():**
* **Source:** The **Synthetic Query** generated in Step 2A (e.g., *"discount on premium coffee beans"*).
* **Role:** Represents the user's raw input and intent.


2. **Positive ():**
* **Source:** A **Synthetic Merchant Text** generated by the LLM to be semantically identical to the anchor.
* **Logic:** We ask the LLM: *"Write a short merchant offer description that perfectly satisfies this query."*
* **Example:** *"Peet's Coffee: Exclusive online offer for 20% off single-origin whole bean bags. Free shipping on all premium roasts."*
* **Technical Value:** This ensures the positive pair has high semantic overlap with the specific *phrasing* and *intent* of the query, creating a strong attractor in the vector space.


3. **Negative ():**
* **Source:** A **Synthetic Merchant Text** generated to be *lexically similar* but possessing a **conflicting intent or purpose**.
* **Logic:** We ask the LLM: *"Write a merchant offer description that contains similar keywords to the query (like 'coffee', 'beans') but addresses a completely different user need (e.g., in-store prepared drinks or equipment instead of beans)."*
* **Example:** *"Starbucks Cafe: Stop by for a Buy-One-Get-One deal on all Grande handcrafted lattes and iced coffees. Valid for in-store pickup only."*
* **Key Distinction:** Both texts contain "coffee," but the Anchor is about *buying beans (product)* while the Negative is about *buying a drink (service/experience)*.
* **Technical Value:** This constitutes a "Hard Negative." By forcing the model to push these two vectors apart, the embedding learns to distinguish between the intent of "grocery/product" vs. "dining/service," preventing the search engine from returning a cafe coupon when a user wants to buy a bag of beans.



### C. Model Training

The `all-mpnet-base-v2` is fine-tuned on these triplets for 5-10 epochs. The resulting model is saved as the **Fine-Tuned MPNet**, which is now specialized for the Chase Merchant ecosystem.

---

## 3. Indexing Strategy (The "Seven Indices")

**Code Location:** `chasesmartoffersearch/core/indexing/generate_index.py`

Once the data is enriched and the model is trained, we build seven specialized indices to support the hybrid inference engine.

### 1. Lexical Index (Whoosh)

* **Structure:** Inverted Index (Map of Words  Document IDs).
* **Fields:** `mrch_nm` (boosted 2.0x), `offr_description` (1.0x).
* **Tokenizer:** Standard English analyzer (lowercasing, stemming, stop-word removal).
* **Purpose:** Powers the BM25 search for exact keyword matching.

### 2. Semantic Dense Index (FAISS)

* **Input:** `offr_description` encoded by **Fine-Tuned MPNet**.
* **Structure:** `IndexFlatIP` (Inner Product) or `IndexIVFFlat` (Inverted File with quantizer) depending on dataset size.
* **Dimension:** 768.
* **Purpose:** Powers the core Approximate Nearest Neighbor (ANN) search for high recall.

### 3. Semantic Exact Index (Torch)

* **Input:** Same embeddings as FAISS.
* **Structure:** A raw PyTorch tensor loaded into GPU memory.
* **Purpose:** Allows for exact Matrix Multiplication (Dot Product) without the approximation errors of FAISS. Used for small, high-priority subsets where precision is paramount.

### 4. SPLADE Index (Sparse Neural)

* **Input:** `offr_description` encoded by **SPLADE-v2**.
* **Structure:** An Inverted Index similar to Whoosh, but the "words" are tokens from the BERT vocabulary, and the "counts" are learned importance weights (floats), not integer term frequencies.
* **Purpose:** Captures specific entities (like model numbers or brand names) that dense vectors might smooth over.

### 5. Tag Embedding Index

* **Input:** The list of unique tags generated in the Enrichment Phase.
* **Process:** Each tag string (e.g., "mexican cuisine") is embedded using MPNet.
* **Structure:** FAISS Index of Tag Vectors.
* **Purpose:** Allows the system to map a user's query to the nearest *concept tag* before looking for offers.

### 6. Inverse Tag Index

* **Structure:** Dictionary/Hash Map. `Dict[Tag_String, List[Offer_IDs]]`.
* **Logic:** Since multiple merchants can share the tag "mexican cuisine", this index aggregates them. The list is sorted by offer popularity (if available) or confidence score.
* **Purpose:** Once the Tag Embedding Index identifies the tag "mexican cuisine", this index immediately retrieves all relevant Offer IDs.

### 7. Semantic Cache Index

* **Input:** The **Synthetic Queries** generated in Step 2A.
* **Process:** We pre-compute embeddings for these synthetic queries.
* **Structure:** FAISS Index.
* **Purpose:** This serves as a "Warm Start." If a real user types a query that is semantically identical to a synthetic query we already anticipated, we can bypass the expensive inference steps and serve the pre-calculated results associated with that synthetic query.

---

## 4. References & Further Reading

* **Sentence Transformers & MNRL:**
Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP*.
*Overview of the bi-encoder architecture and training methodology.*
* **SPLADE:**
Formal, T., Piwowarski, B., & Clinchant, S. (2021). "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking." *SIGIR*.
*Technical details on log-saturation and sparse regularization.*
* **Reciprocal Rank Fusion:**
Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). "Reciprocal rank fusion outperforms condorcet and individual rank learning methods." *SIGIR*.
*Foundational paper for the rank aggregation logic used in Layer 3.*
* **Approximate Nearest Neighbors (Faiss):**
Johnson, J., Douze, M., & Jégou, H. (2019). "Billion-scale similarity search with GPUs." *arXiv preprint arXiv:1702.08734*.

* **Reimers, N., & Gurevych, I. (2019).** "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP*. (The foundational paper for the architecture used).
* **Formal, T., et al. (2021).** "SPLADE: Sparse Lexical and Expansion Model." *SIGIR*. (Basis for the sparse retrieval layer).
* **Karpukhin, V., et al. (2020).** "Dense Passage Retrieval for Open-Domain Question Answering." *EMNLP*. (Concept of using in-batch negatives for training efficiency).