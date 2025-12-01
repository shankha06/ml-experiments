import bm25s
# import Stemmer  # Optional: For better results (requires PyStemmer)

def index_corpus(corpus):
    """
    Indexes the corpus using the bm25s library.
    
    Args:
        corpus (list of str): A list of documents.
        
    Returns:
        tuple: (retriever, corpus_tokens) - The initialized BM25 object and tokenized corpus.
    """
    print("--- Starting Indexing Process ---")
    
    stemmer = None
    print("PyStemmer not found. Proceeding without stemming.")

    # 2. Tokenize the corpus
    #    bm25s.tokenize handles lowercasing, stopwords, and stemming automatically
    #    stopwords="en" removes common English words like "the", "and", etc.
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
    
    # 3. Initialize the BM25 Retriever
    #    method="lucene" is a common default, but "bm25+" or "robertson" are also available
    retriever = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
    
    # 4. Index the tokens
    #    This calculates and stores the sparse matrices needed for fast retrieval
    retriever.index(corpus_tokens)
    
    print(f"Indexed {len(corpus)} documents.")
    print("--- Indexing Complete ---")
    
    return retriever, stemmer

def search(query, retriever, corpus, stemmer=None, top_k=5):
    """
    Searches the indexed corpus for the query.
    
    Args:
        query (str): The search query.
        retriever (bm25s.BM25): The indexed retriever object.
        corpus (list of str): The original list of documents (for display).
        stemmer: The stemmer used during indexing (must be consistent).
        top_k (int): Number of results to return.
    """
    print(f"\n--- Searching for: '{query}' ---")
    
    # 1. Tokenize the query
    #    Must use the same stemmer and stopword logic as indexing
    query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer)
    
    # 2. Retrieve top-k results
    #    returns (documents, scores) if corpus is passed, or (indices, scores) if not
    results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=top_k)
    
    # 3. Display Results
    #    bm25s returns shape (n_queries, k), so we access [0] for the first (only) query
    for i in range(len(results[0])):
        doc_content = results[0][i]
        score = scores[0][i]
        print(f"Rank {i+1} | Score: {score:.4f} | Document: '{doc_content}'")
        
    return results[0], scores[0]

if __name__ == '__main__':
    # Sample corpus
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "A lazy dog is lying in the sun.",
        "The fox is quick, but the dog is lazy and brown.",
        "Programming with Python is fun and efficient.",
        "Learning data structures and algorithms is essential for programmers.",
        "Quick search algorithms improve retrieval speed significantly."
    ]
    
    # 1. Indexing
    retriever, stemmer = index_corpus(corpus)
    
    # 2. Searching
    # Query 1: Focuses on early documents
    search("quick dog retrieval", retriever, corpus, stemmer, top_k=3)

    # Query 2: Focuses on later documents
    search("data structure python", retriever, corpus, stemmer, top_k=3)