from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-300m")

# We recommend enabling flash_attention_2 for better acceleration and memory saving,
# together with setting `padding_side` to "left":
# model = SentenceTransformer(
#     "Qwen/Qwen3-Embedding-0.6B",
#     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
#     tokenizer_kwargs={"padding_side": "left"},
# )

# The queries and documents to embed
queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

# Encode the queries and documents. Note that queries benefit from using a prompt
# Here we use the prompt called "query" stored under `model.prompts`, but you can
# also pass your own prompt via the `prompt` argument
query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents)

# Compute the (cosine) similarity between the query and document embeddings
similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)
# tensor([[11.3768, 10.8296,  4.3457]])

model.save_pretrained("models/gemma-300m")

# tar -cvf - gemma-300m | split -b 95M - "gemma.tar.part-"

# cat qwen3.tar.part-* > qwen3.tar
# tar -xvf splade.tar

# tar -cvf - gliner_medium-v2.1 | split -b 95M - "gliner.tar.part-"
