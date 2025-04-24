import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.preprocessing import normalize

def generate_embeddings(model_name="all-mpnet-base-v2"):
    """Generate embeddings for algorithm chunks using SentenceTransformer."""
    processed_dir = "./data/processed"
    embeddings_dir = "./data/embeddings"
    os.makedirs(embeddings_dir, exist_ok=True)
    

    chunks_file = os.path.join(processed_dir, "algorithm_chunks.json")
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks for embedding generation")
    
    model = SentenceTransformer(model_name)
    model.max_seq_length = 512 
    

    texts = [chunk["content"] for chunk in chunks]
    

    for i, text in enumerate(texts):
        if len(text) < 10:
            print(f"Warning: Very short text at index {i}: '{text}'")
    
    embeddings = []
    
   
    batch_size = 2
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc=f"Generating embeddings with {model_name}"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]  
    
        batch_embeddings = model.encode(batch_texts, normalize_embeddings=True)
        embeddings.extend(batch_embeddings.tolist())
    
    expected_dim = model.get_sentence_embedding_dimension()
    for i, emb in enumerate(embeddings):
        if len(emb) != expected_dim:
            print(f"Warning: Embedding {i} has unexpected dimension {len(emb)} (expected {expected_dim})")
    
    embeddings_file = os.path.join(embeddings_dir, f"algorithm_embeddings_{model_name.replace('-', '_')}.npy")
    np.save(embeddings_file, np.array(embeddings))
    
    embeddings_mapping = []
    for i, chunk in enumerate(chunks):
        content = chunk["content"]
        metadata = chunk["metadata"].copy()
        
        algorithm_name = metadata.get("name", "")
        if algorithm_name and algorithm_name not in content[:100]:
            content = f"Algorithm: {algorithm_name}\n{content}"
            
        if "sliding window" in algorithm_name.lower() or any("sliding window" in tag.lower() for tag in metadata.get("tags", [])):
            if "longest substring" not in content.lower() and "without repeat" not in content.lower():
                content += "\nExample Problem: Finding the longest substring without repeating characters."
        
        embedding_entry = {
            "embedding_index": i,
            "content": content,
            "metadata": metadata
        }
        embeddings_mapping.append(embedding_entry)
    
    mapping_file = os.path.join(embeddings_dir, "embeddings_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(embeddings_mapping, f, indent=2)
    
    print(f"Saved {len(embeddings)} embeddings and mapping")
    return embeddings, embeddings_mapping

if __name__ == "__main__":
    generate_embeddings(model_name="all-mpnet-base-v2")