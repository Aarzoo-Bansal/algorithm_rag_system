import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def generate_embeddings(model_name="all-MiniLM-L6-v2"):
    """Generate embeddings for algorithm chunks using SentenceTransformer."""
    processed_dir = "./data/processed"
    embeddings_dir = "./data/embeddings"
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Load algorithm chunks
    chunks_file = os.path.join(processed_dir, "algorithm_chunks.json")
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks for embedding generation")
    
    # Initialize embedding model
    model = SentenceTransformer(model_name)
    
    # Generate embeddings
    texts = [chunk["content"] for chunk in chunks]
    embeddings = []
    
    # Process in batches to avoid memory issues
    batch_size = 32
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc=f"Generating embeddings with {model_name}"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        
        batch_embeddings = model.encode(batch_texts)
        embeddings.extend(batch_embeddings.tolist())
    
    # Save embeddings
    embeddings_file = os.path.join(embeddings_dir, f"algorithm_embeddings_{model_name.replace('-', '_')}.npy")
    np.save(embeddings_file, np.array(embeddings))
    
    # Create mapping from embeddings to chunks
    embeddings_mapping = []
    for i, chunk in enumerate(chunks):
        embedding_entry = {
            "embedding_index": i,
            "content": chunk["content"],
            "metadata": chunk["metadata"]
        }
        embeddings_mapping.append(embedding_entry)
    
    mapping_file = os.path.join(embeddings_dir, "embeddings_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(embeddings_mapping, f, indent=2)
    
    print(f"Saved {len(embeddings)} embeddings and mapping")
    return embeddings, embeddings_mapping

if __name__ == "__main__":
    generate_embeddings()