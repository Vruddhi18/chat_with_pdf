import faiss
import numpy as np

def create_embeddings(text, embed_model, chunk_size=512):
    """Split text into chunks and store in FAISS index."""
    words = text.split()
    chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]

    index = faiss.IndexFlatL2(384)  # 384-dimensional embeddings
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    index.add(embeddings)
    
    return chunks, index

def search_similar_chunks(query, embed_model, index, chunks, top_k=3):
    """Find similar text chunks using FAISS."""
    query_embedding = embed_model.encode([query])
    _, indices = index.search(query_embedding, k=top_k)
    return [chunks[i] for i in indices[0]]
