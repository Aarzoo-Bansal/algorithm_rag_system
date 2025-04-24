# import os
# import json
# import numpy as np
# from tqdm import tqdm
# import sys
# import faiss

# class AlgorithmVectorStore:
#     """Simple vector store for algorithm embeddings using FAISS."""
    
#     def __init__(self, embeddings_dir="./data/embeddings"):
#         self.embeddings_dir = embeddings_dir
#         self.embeddings = None
#         self.mapping = None
#         self.index = None
#         self.dimension = None
    
#     def load_embeddings(self, model_name="all-MiniLM-L6-v2"):
#         """Load embeddings and mapping from files."""
#         embeddings_file = os.path.join(self.embeddings_dir, f"algorithm_embeddings_{model_name.replace('-', '_')}.npy")
#         mapping_file = os.path.join(self.embeddings_dir, "embeddings_mapping.json")
        
#         # Load embeddings
#         self.embeddings = np.load(embeddings_file)
#         self.dimension = self.embeddings.shape[1]
        
#         # Load mapping
#         with open(mapping_file, 'r') as f:
#             self.mapping = json.load(f)
        
#         print(f"Loaded {len(self.embeddings)} embeddings with dimension {self.dimension}")
#         return self
    
#     def build_index(self):
#         """Build FAISS index for fast similarity search."""
#         if self.embeddings is None:
#             raise ValueError("Embeddings not loaded. Call load_embeddings() first.")
        
    
#         self.index = faiss.IndexFlatL2(self.dimension)
        
#         self.index.add(self.embeddings.astype('float32'))
        
#         print(f"Built FAISS index with {self.index.ntotal} vectors")
#         return self
    
#     def save_index(self, index_name="algorithm_index"):
#         """Save the FAISS index to disk."""
#         if self.index is None:
#             raise ValueError("Index not built. Call build_index() first.")
        
#         index_file = os.path.join(self.embeddings_dir, f"{index_name}.faiss")
#         faiss.write_index(self.index, index_file)
        
#         print(f"Saved FAISS index to {index_file}")
#         return self
    
#     def load_index(self, index_name="algorithm_index"):
#         """Load a FAISS index from disk."""
#         index_file = os.path.join(self.embeddings_dir, f"{index_name}.faiss")
#         self.index = faiss.read_index(index_file)
        
#         print(f"Loaded FAISS index with {self.index.ntotal} vectors")
#         return self
    
#     def search(self, query_embedding, k=5):
#         """Search for similar embeddings."""
#         if self.index is None:
#             raise ValueError("Index not loaded. Call load_index() or build_index() first.")
        
#         # Convert to correct shape and type
#         query_embedding = np.array([query_embedding]).astype('float32')
        
#         # Search
#         distances, indices = self.index.search(query_embedding, k)
        
#         # Get results
#         results = []
#         for i, idx in enumerate(indices[0]):
#             if idx >= 0 and idx < len(self.mapping):  # Valid index
#                 result = self.mapping[idx].copy()
#                 result["distance"] = float(distances[0][i])
#                 results.append(result)
        
#         return results

# def setup_vector_store():
#     """Set up the vector store with embeddings and index."""
#     vector_store = AlgorithmVectorStore()
    
#     # Check if index exists
#     index_file = os.path.join(vector_store.embeddings_dir, "algorithm_index.faiss")
    
#     if os.path.exists(index_file):
#         # Load existing index
#         vector_store.load_embeddings().load_index()
#     else:
#         # Create new index
#         vector_store.load_embeddings().build_index().save_index()
    
#     return vector_store

# if __name__ == "__main__":
#     setup_vector_store()

import os
import json
import numpy as np
from tqdm import tqdm
import sys
import faiss

class AlgorithmVectorStore:
    """Enhanced vector store for algorithm embeddings using FAISS."""
    
    def __init__(self, embeddings_dir="./data/embeddings"):
        self.embeddings_dir = embeddings_dir
        self.embeddings = None
        self.mapping = None
        self.index = None
        self.dimension = None
        self.model_name = None
    
    def load_embeddings(self, model_name="all-mpnet-base-v2"):
        """Load embeddings and mapping from files."""
        self.model_name = model_name
        embeddings_file = os.path.join(self.embeddings_dir, f"algorithm_embeddings_{model_name.replace('-', '_')}.npy")
        mapping_file = os.path.join(self.embeddings_dir, "embeddings_mapping.json")
        
        # Load embeddings
        self.embeddings = np.load(embeddings_file)
        self.dimension = self.embeddings.shape[1]
        
        # Load mapping
        with open(mapping_file, 'r') as f:
            self.mapping = json.load(f)
        
        print(f"Loaded {len(self.embeddings)} embeddings with dimension {self.dimension} from model {model_name}")
        return self
    
    def build_index(self, index_type="cosine"):
        """Build FAISS index for fast similarity search.
        
        Args:
            index_type: Type of index to build - 'cosine' or 'l2'
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded. Call load_embeddings() first.")
        
        # Create the appropriate index type
        if index_type == "cosine":
            # For cosine similarity, we need to normalize vectors and use InnerProductDistance
            # Manual L2-normalization to avoid FAISS type issues
            normalized_embeddings = self.embeddings.astype(np.float32)
            norms = np.linalg.norm(normalized_embeddings, axis=1, keepdims=True)
            normalized_embeddings = normalized_embeddings / norms
            
            # Create inner product index for cosine similarity
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(normalized_embeddings)
            print(f"Built FAISS cosine similarity index with {self.index.ntotal} vectors")
        else:
            # Standard L2 distance index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(self.embeddings.astype(np.float32))
            print(f"Built FAISS L2 distance index with {self.index.ntotal} vectors")
        
        return self
    
    def save_index(self, index_name=None):
        """Save the FAISS index to disk."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        if index_name is None:
            # Create name based on model name
            index_name = f"algorithm_index_{self.model_name.replace('-', '_')}"
        
        index_file = os.path.join(self.embeddings_dir, f"{index_name}.faiss")
        faiss.write_index(self.index, index_file)
        
        # Save index metadata
        metadata = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "vector_count": self.index.ntotal,
        }
        
        metadata_file = os.path.join(self.embeddings_dir, f"{index_name}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved FAISS index to {index_file}")
        return self
    
    def load_index(self, index_name=None):
        """Load a FAISS index from disk."""
        if index_name is None:
            # Create name based on model name if available
            if self.model_name:
                index_name = f"algorithm_index_{self.model_name.replace('-', '_')}"
            else:
                index_name = "algorithm_index"
        
        index_file = os.path.join(self.embeddings_dir, f"{index_name}.faiss")
        
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file {index_file} not found")
            
        self.index = faiss.read_index(index_file)
        
        # Try to load metadata if available
        metadata_file = os.path.join(self.embeddings_dir, f"{index_name}_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.model_name = metadata.get("model_name", self.model_name)
                print(f"Loaded FAISS index created with model {self.model_name}")
        
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        return self
    
    def search(self, query_embedding, k=5, threshold=None):
        """Search for similar embeddings with optional distance threshold.
        
        Args:
            query_embedding: The embedding vector to search for
            k: Number of results to return
            threshold: Optional distance threshold (lower is better for L2, higher is better for IP)
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() or build_index() first.")
        
        # Ensure query embedding is the right type
        query_embedding = query_embedding.astype(np.float32)
        
        # Normalize the query if using cosine similarity (IndexFlatIP)
        if isinstance(self.index, faiss.IndexFlat) and self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
            # Manual L2-normalization
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
        
        # Convert to correct shape
        if len(query_embedding.shape) == 1:
            query_embedding = np.array([query_embedding])
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            distance = float(distances[0][i])
            
            # Apply threshold if specified
            if threshold is not None:
                # For L2 distance, lower is better
                if self.index.metric_type == faiss.METRIC_L2 and distance > threshold:
                    continue
                # For inner product (cosine), higher is better
                elif self.index.metric_type == faiss.METRIC_INNER_PRODUCT and distance < threshold:
                    continue
            
            if idx >= 0 and idx < len(self.mapping):  # Valid index
                result = self.mapping[idx].copy()
                result["distance"] = distance
                
                # For inner product (cosine), convert distance to similarity score (0-1)
                if self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
                    result["similarity_score"] = float(distance)  # Already between -1 and 1 for normalized vectors
                else:
                    # For L2 distance, convert to similarity (higher is better)
                    max_distance = 20.0  # Arbitrary high value for normalization
                    result["similarity_score"] = 1.0 - min(distance / max_distance, 1.0)
                
                results.append(result)
        
        return results

    def hybrid_search(self, query_embedding, query_text, k=5, alpha=0.5):
        """
        Perform hybrid search combining vector similarity with keyword matching.
        
        Args:
            query_embedding: The embedding vector for semantic search
            query_text: The original query text for keyword matching
            k: Number of results to return
            alpha: Weight between vector (alpha) and keyword (1-alpha) search
        """
        import re
        from collections import Counter
        
        # Get more candidates from vector search for reranking
        candidates = self.search(query_embedding, k=k*3)
        
        # Simple keyword matching using term frequency
        query_terms = set(re.findall(r'\b\w+\b', query_text.lower()))
        
        # Score each candidate
        for candidate in candidates:
            content = candidate["content"].lower()
            
            # Count keyword matches
            keyword_matches = sum(1 for term in query_terms if term in content)
            keyword_score = keyword_matches / max(1, len(query_terms))
            
            # Combine scores (vector similarity and keyword matching)
            vector_score = candidate["similarity_score"]
            combined_score = (alpha * vector_score) + ((1 - alpha) * keyword_score)
            
            candidate["hybrid_score"] = combined_score
        
        # Sort by hybrid score
        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # Return top k
        return candidates[:k]


def setup_vector_store(model_name="all-mpnet-base-v2"):
    """Set up the vector store with embeddings and index."""
    vector_store = AlgorithmVectorStore()
    
    # Use model name in index file name
    index_file = os.path.join(vector_store.embeddings_dir, f"algorithm_index_{model_name.replace('-', '_')}.faiss")
    
    if os.path.exists(index_file):
        # Load existing index
        vector_store.load_embeddings(model_name).load_index(f"algorithm_index_{model_name.replace('-', '_')}")
    else:
        # Create new index - use cosine similarity which works better for semantic search
        vector_store.load_embeddings(model_name).build_index(index_type="cosine").save_index()
    
    return vector_store

if __name__ == "__main__":
    # Use the better model by default
    setup_vector_store(model_name="all-mpnet-base-v2")