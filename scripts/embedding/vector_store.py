import os
import json
import numpy as np
from tqdm import tqdm
import sys
import faiss

class AlgorithmVectorStore:
    """Simple vector store for algorithm embeddings using FAISS."""
    
    def __init__(self, embeddings_dir="./data/embeddings"):
        self.embeddings_dir = embeddings_dir
        self.embeddings = None
        self.mapping = None
        self.index = None
        self.dimension = None
    
    def load_embeddings(self, model_name="all-MiniLM-L6-v2"):
        """Load embeddings and mapping from files."""
        embeddings_file = os.path.join(self.embeddings_dir, f"algorithm_embeddings_{model_name.replace('-', '_')}.npy")
        mapping_file = os.path.join(self.embeddings_dir, "embeddings_mapping.json")
        
        # Load embeddings
        self.embeddings = np.load(embeddings_file)
        self.dimension = self.embeddings.shape[1]
        
        # Load mapping
        with open(mapping_file, 'r') as f:
            self.mapping = json.load(f)
        
        print(f"Loaded {len(self.embeddings)} embeddings with dimension {self.dimension}")
        return self
    
    def build_index(self):
        """Build FAISS index for fast similarity search."""
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded. Call load_embeddings() first.")
        
    
        self.index = faiss.IndexFlatL2(self.dimension)
        
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Built FAISS index with {self.index.ntotal} vectors")
        return self
    
    def save_index(self, index_name="algorithm_index"):
        """Save the FAISS index to disk."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        index_file = os.path.join(self.embeddings_dir, f"{index_name}.faiss")
        faiss.write_index(self.index, index_file)
        
        print(f"Saved FAISS index to {index_file}")
        return self
    
    def load_index(self, index_name="algorithm_index"):
        """Load a FAISS index from disk."""
        index_file = os.path.join(self.embeddings_dir, f"{index_name}.faiss")
        self.index = faiss.read_index(index_file)
        
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        return self
    
    def search(self, query_embedding, k=5):
        """Search for similar embeddings."""
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() or build_index() first.")
        
        # Convert to correct shape and type
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.mapping):  # Valid index
                result = self.mapping[idx].copy()
                result["distance"] = float(distances[0][i])
                results.append(result)
        
        return results

def setup_vector_store():
    """Set up the vector store with embeddings and index."""
    vector_store = AlgorithmVectorStore()
    
    # Check if index exists
    index_file = os.path.join(vector_store.embeddings_dir, "algorithm_index.faiss")
    
    if os.path.exists(index_file):
        # Load existing index
        vector_store.load_embeddings().load_index()
    else:
        # Create new index
        vector_store.load_embeddings().build_index().save_index()
    
    return vector_store

if __name__ == "__main__":
    setup_vector_store()