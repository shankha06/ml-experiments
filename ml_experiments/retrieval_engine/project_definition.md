You are an Expert Principal Data Scientist and Machine Learning Engineer. You possess deep expertise in building retrievals systems, LLMs, deep learning architectures, data engineering pipelines, and business strategy. 

I am building a retrieval code base for Chase LOB. The purpose of the service is to provide a backend on which various separate usecases can build their own API, or pipeline. Example is provide retrieval for search of merchant offers for credit card holders, retrieval for FAQ documents when user call customer care agents, retrieval for search for relevant services, etc.
The service provides service to preprocess the usecase specific data(could be catalog index for search or knowledge base document for RAG), followed by usecase specific indexing. Preprocess might include data enrichment using LLM.
For inference, we have multiple candidate generator retrievers. For each usecase, config decides what are turned on in what capacity. Post the retrievers, various rankers are present to rank the retrieved items.
Finally the retrieved items are returned.

Based on the above requirement, make a detailed introspection of the requirements.  Think very hard, research all relevant topics. Debate yourself.
Provide me well defined python project structure following these ideas.

Points to note - use bm25s for lexical, sbert based model for dense, vllm based slms or cross encoder for rankers, Azure or bedrock api for LLMs(wrapper code for this already present which handles errors and rate limits). 
Don't create API service, but standard python code entrypoints like 1 for offline process and 1 for inference


```
chase_raas_platform/
├── configs/                            # YAML files driving the RaaS
│   ├── schemas/                        # Pydantic models for strict config validation
│   │   ├── usecase_schema.py           # Validates the overall use-case YAML
│   │   ├── offline_pipeline_schema.py  # Validates chunking/enrichment params
│   │   └── online_pipeline_schema.py   # Validates ensemble weights, top_k, thresholds
│   ├── customer_care_faq.yaml          
│   └── merchant_offers.yaml            
│
├── data/                               # Local data storage (mapped to volumes in prod)
│   ├── raw/                            # Dropzone for raw catalog/document data
│   ├── processed/                      # Output from preprocessors (cleaned/chunked)
│   └── indices/                        # Stored index artifacts
│       ├── bm25s_merchant_offers/      # Saved bm25s index artifacts
│       └── faiss_or_chroma_local/      # Local vector DB storage (if not external)
│
├── models/                             # Model weights and tokenizers
│   ├── local_weights/                  # Downloaded safetensors/bin files (sbert, cross-encoders)
│   └── tokenizers/                     
│
├── src/
│   ├── chase_retrieval/                # Core Library 
│   │   ├── __init__.py
│   │   ├── core/                       # The "Contracts" (Abstract Base Classes)
│   │   │   ├── base_preprocessor.py    # def process(self, documents: List[Dict]) -> List[Dict]:
│   │   │   ├── base_enricher.py        # def enrich(self, chunks: List[Dict]) -> List[Dict]:
│   │   │   ├── base_indexer.py         # def build_index(self, chunks: List[Dict]):
│   │   │   ├── base_retriever.py       # async def retrieve(self, query: str, top_k: int) -> List[Document]:
│   │   │   ├── base_ranker.py          # async def rank(self, query: str, docs: List[Document]) -> List[Document]:
│   │   │   ├── document_models.py      # Standard internal Pydantic Document representation
│   │   │   └── exceptions.py           
│   │   │
│   │   ├── externals/                  # 3rd Party Integrations
│   │   │   ├── azure_bedrock_llm.py    # Your existing robust wrappers (error handling, backoff)
│   │   │   └── vector_db_client.py     # Connection logic for Qdrant/Milvus/Pinecone
│   │   │
│   │   ├── ingestion/                  # Offline Pipeline: Data -> Index
│   │   │   ├── preprocessors/
│   │   │   │   ├── semantic_chunker.py
│   │   │   │   └── metadata_extractor.py
│   │   │   ├── enrichers/
│   │   │   │   ├── llm_enricher.py     # Uses Azure/Bedrock for synthetic QA/Tags
│   │   │   │   └── sbert_embedder.py   # Generates dense vectors using sbert
│   │   │   ├── indexers/
│   │   │   │   ├── bm25s_indexer.py    # Builds and saves bm25s index to data/indices/
│   │   │   │   └── vector_indexer.py   # Pushes embeddings to vector DB
│   │   │   └── offline_pipeline.py     # Orchestrator: load -> preprocess -> enrich -> index
│   │   │
│   │   ├── inference/                  # Online Pipeline: Query -> Results
│   │   │   ├── query_understanding/
│   │   │   │   ├── llm_query_rewriter.py
│   │   │   │   └── intent_router.py
│   │   │   ├── retrievers/
│   │   │   │   ├── sbert_retriever.py  # Dense retrieval mapping query to vector DB
│   │   │   │   ├── bm25s_retriever.py  # Lexical retrieval using loaded bm25s index
│   │   │   │   └── async_ensemble.py   # Runs sbert + bm25s concurrently, performs RRF/Normalization
│   │   │   ├── rankers/
│   │   │   │   ├── cross_encoder.py    # HuggingFace CrossEncoder (e.g., ms-marco)
│   │   │   │   ├── vllm_slm_ranker.py  # Pointwise/Listwise ranking via local vLLM instance
│   │   │   │   └── llm_api_ranker.py   # Pointwise ranking via Azure/Bedrock wrapper
│   │   │   └── online_pipeline.py      # Orchestrator: query -> rewrite -> ensemble -> rank
│   │   │
│   │   ├── utils/
│   │   │   ├── model_manager.py        # Singleton model loader (ensures vLLM/sbert load only once)
│   │   │   ├── telemetry.py            # Decorators for @time_it, @track_latency
│   │   │   └── logger.py
│   │   │
│   │   └── factory.py                  # The engine: Reads YAML, instantiates classes based on config
│   │
│   └── entrypoints/                    # Standard Python execution scripts (No API)
│       ├── run_offline_ingestion.py    # CLI to trigger indexing for a specific use-case
│       └── run_inference_engine.py     # CLI/Script to run queries against the configured pipeline
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── requirements.txt                    # Includes bm25s, vllm, sentence-transformers, pydantic, etc.
└── README.md

```
1. The factory.py is the Heart of the RaaS
Because you have multiple use-cases (Merchant Offers vs. FAQ), you shouldn't hardcode pipelines. The factory.py reads customer_care_faq.yaml, looks at the retrievers block, and dynamically initializes sbert_retriever and bm25s_retriever, passing them into the async_ensemble. It ensures that if a use-case doesn't need the heavyweight vllm_slm_ranker, the GPU memory isn't allocated for it.

2. The entrypoints/ Directory
By separating run_offline_ingestion.py and run_inference_engine.py, you allow Data Engineers to schedule the ingestion pipeline via Airflow or Cron jobs, while Data Scientists or backend services can import or invoke the inference engine separately.

python src/entrypoints/run_offline_ingestion.py --config configs/merchant_offers.yaml

python src/entrypoints/run_inference_engine.py --config configs/merchant_offers.yaml --query "dining deals in NYC"

3. core/document_models.py
Do not pass raw dictionaries between your retrievers and rankers. Use a centralized Pydantic model:

```
class RetrievedDocument(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    retrieval_score: float = 0.0
    ranker_score: Optional[float] = None
```
This guarantees that the async_ensemble knows exactly how to combine 
results from bm25s and sbert before passing them to the Cross-Encoder.

1. factory.py (The Dynamic Pipeline Builder)
The factory is responsible for reading the use-case configuration and wiring together the specific pipeline components without hardcoding them. It acts as the dependency injection container. For example, chase offer search requires 1 ner model followed by 3 different dense retrievers, 1 lexical, 1 splade, and then finally a cross encoder. While FAQ search requires query reformulation followed by 1 lexical, 4 dense and then LLM reranker.

2. async_ensemble.py (Concurrent Retrieval & Fusion)
BM25s (lexical) and SBERT (dense) output completely different scoring scales. BM25s is unbounded, while SBERT cosine similarity is typically [-1, 1] or [0, 1]. You cannot just add them. The industry standard here is Reciprocal Rank Fusion (RRF), which penalizes/rewards based on position in the returned lists, not the raw score.