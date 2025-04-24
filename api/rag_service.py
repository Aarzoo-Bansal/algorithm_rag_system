# import os
# import sys
# import json
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List, Dict, Any, Optional

# # Print current directory to debug
# print("Current working directory:", os.getcwd())

# # Script's directory
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# print("Script directory:", SCRIPT_DIR)

# # Add parent directory to path for imports
# BASE_DIR = os.path.dirname(SCRIPT_DIR)
# sys.path.append(BASE_DIR)
# print("Base directory added to path:", BASE_DIR)
# print("Python path:", sys.path)

# # Try to import vector store
# try:
#     from scripts.embedding.vector_store import AlgorithmVectorStore
#     print("Successfully imported AlgorithmVectorStore")
# except ImportError as e:
#     print(f"Failed to import AlgorithmVectorStore: {e}")
#     # Try a different import path
#     try:
#         sys.path.append(os.path.join(BASE_DIR, "scripts"))
#         from embedding.vector_store import AlgorithmVectorStore
#         print("Successfully imported AlgorithmVectorStore from alternate path")
#     except ImportError as e:
#         print(f"Failed to import from alternate path: {e}")
#         raise

# app = FastAPI(title="Algorithm RAG Service")

# # Initialize embedding model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# print("Initialized embedding model")

# # Initialize vector store with explicit debugging
# try:
#     vector_store = AlgorithmVectorStore().load_embeddings().load_index()
#     print("Initialized vector store successfully")
# except Exception as e:
#     print(f"Error initializing vector store: {e}")
#     raise

# # Define processed data directory - try multiple potential paths
# potential_paths = [
#     os.path.join(BASE_DIR, "data", "processed"),
#     os.path.join(SCRIPT_DIR, "data", "processed"),
#     os.path.join(os.getcwd(), "data", "processed"),
#     os.path.join(BASE_DIR, "..", "data", "processed")
# ]

# # Find the first path that contains the enhanced_algorithms.json file
# PROCESSED_DATA_DIR = None
# for path in potential_paths:
#     print(f"Checking for enhanced_algorithms.json in: {path}")
#     test_file = os.path.join(path, "enhanced_algorithms.json")
#     if os.path.exists(test_file):
#         PROCESSED_DATA_DIR = path
#         print(f"Found enhanced_algorithms.json at: {test_file}")
#         break
#     else:
#         print(f"Tried but couldn't find file at: {test_file}")

# # If we still haven't found the file, try to search for it
# if PROCESSED_DATA_DIR is None:
#     print("Searching for enhanced_algorithms.json in project directories...")
#     for root, dirs, files in os.walk(BASE_DIR):
#         if "enhanced_algorithms.json" in files:
#             PROCESSED_DATA_DIR = os.path.dirname(os.path.join(root, "enhanced_algorithms.json"))
#             print(f"Found file via search at: {os.path.join(root, 'enhanced_algorithms.json')}")
#             break

# if PROCESSED_DATA_DIR is None:
#     print("WARNING: Could not find enhanced_algorithms.json anywhere in the project!")

# class QueryRequest(BaseModel):
#     query: str
#     top_k: int = 3

# class AlgorithmResult(BaseModel):
#     id: str
#     name: str
#     title: Optional[str] = None
#     description: Optional[str] = None
#     tags: List[str]
#     difficulty: Optional[str] = None
#     complexity: Optional[Dict[str, str]] = None
#     use_cases: Optional[List[str]] = None
#     similarity_score: float

# class QueryResponse(BaseModel):
#     query: str
#     results: List[AlgorithmResult]

# @app.post("/api/query", response_model=QueryResponse)
# def query_algorithms(request: QueryRequest):
#     """Query for relevant algorithms based on a problem description."""
#     try:
#         # Generate embedding for the query
#         query_embedding = embedding_model.encode(request.query)
        
#         # Search for similar algorithm chunks
#         raw_results = vector_store.search(query_embedding, k=request.top_k * 2)
        
#         # Process results to return unique algorithms
#         seen_ids = set()
#         results = []
        
#         # Determine the path to use for enhanced_algorithms.json
#         if PROCESSED_DATA_DIR:
#             algorithms_file = os.path.join(PROCESSED_DATA_DIR, "enhanced_algorithms.json")
#         else:
#             # Try all potential paths again
#             for path in potential_paths:
#                 test_file = os.path.join(path, "enhanced_algorithms.json")
#                 if os.path.exists(test_file):
#                     algorithms_file = test_file
#                     break
#             else:
#                 # Last resort - check the current directory and parent directories
#                 algorithms_file = None
#                 current_dir = os.getcwd()
#                 for _ in range(3):  # Check up to 3 levels up
#                     test_file = os.path.join(current_dir, "data", "processed", "enhanced_algorithms.json")
#                     if os.path.exists(test_file):
#                         algorithms_file = test_file
#                         break
#                     current_dir = os.path.dirname(current_dir)
        
#         if not algorithms_file or not os.path.exists(algorithms_file):
#             raise FileNotFoundError(f"Could not find enhanced_algorithms.json in any expected location")
        
#         print(f"Loading algorithms from: {algorithms_file}")
#         with open(algorithms_file, 'r') as f:
#             all_algorithms = json.load(f)
        
#         for result in raw_results:
#             algorithm_id = result["metadata"]["id"]
            
#             if algorithm_id not in seen_ids and len(results) < request.top_k:
#                 seen_ids.add(algorithm_id)
                
#                 algorithm = next((a for a in all_algorithms if a.get("id") == algorithm_id), None)
                
#                 if algorithm:
#                     # Create response entry
#                     algorithm_result = AlgorithmResult(
#                         id=algorithm_id,
#                         name=algorithm.get("name", ""),
#                         title=algorithm.get("title", ""),
#                         description=algorithm.get("description", ""),
#                         tags=algorithm.get("tags", []),
#                         difficulty=algorithm.get("difficulty", ""),
#                         complexity=algorithm.get("complexity", {}),
#                         use_cases=algorithm.get("use_cases", []),
#                         similarity_score=1.0 - result["distance"] / 2  # Convert distance to similarity
#                     )
                    
#                     results.append(algorithm_result)
        
#         return QueryResponse(query=request.query, results=results)
        
#     except Exception as e:
#         import traceback
#         error_details = traceback.format_exc()
#         print(f"Error in query_algorithms: {str(e)}\n{error_details}")
#         raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import re

# Print current directory to debug
print("Current working directory:", os.getcwd())

# Script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print("Script directory:", SCRIPT_DIR)

# Add parent directory to path for imports
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(BASE_DIR)
print("Base directory added to path:", BASE_DIR)
print("Python path:", sys.path)

# Try to import vector store
try:
    from scripts.embedding.vector_store import AlgorithmVectorStore
    print("Successfully imported AlgorithmVectorStore")
except ImportError as e:
    print(f"Failed to import AlgorithmVectorStore: {e}")
    # Try a different import path
    try:
        sys.path.append(os.path.join(BASE_DIR, "scripts"))
        from embedding.vector_store import AlgorithmVectorStore
        print("Successfully imported AlgorithmVectorStore from alternate path")
    except ImportError as e:
        print(f"Failed to import from alternate path: {e}")
        raise

app = FastAPI(title="Algorithm RAG Service")

# Initialize embedding model with the better model
MODEL_NAME = "all-mpnet-base-v2"  # Upgrade from all-MiniLM-L6-v2
embedding_model = SentenceTransformer(MODEL_NAME)
embedding_model.max_seq_length = 512  # Increase from default to handle longer queries
print(f"Initialized embedding model: {MODEL_NAME}")

# Initialize vector store with explicit debugging
try:
    # Use the improved model name that matches our generated embeddings
    vector_store = AlgorithmVectorStore().load_embeddings(MODEL_NAME).load_index(f"algorithm_index_{MODEL_NAME.replace('-', '_')}")
    print("Initialized vector store successfully")
except Exception as e:
    print(f"Error initializing vector store: {e}")
    # Fallback to creating a new index if loading fails
    try:
        print("Attempting to create new index...")
        vector_store = AlgorithmVectorStore().load_embeddings(MODEL_NAME).build_index(index_type="cosine").save_index()
        print("Successfully created and saved new index")
    except Exception as e2:
        print(f"Failed to create new index: {e2}")
        raise

# Define processed data directory - try multiple potential paths
potential_paths = [
    os.path.join(BASE_DIR, "data", "processed"),
    os.path.join(SCRIPT_DIR, "data", "processed"),
    os.path.join(os.getcwd(), "data", "processed"),
    os.path.join(BASE_DIR, "..", "data", "processed")
]

# Find the first path that contains the enhanced_algorithms.json file
PROCESSED_DATA_DIR = None
for path in potential_paths:
    print(f"Checking for enhanced_algorithms.json in: {path}")
    test_file = os.path.join(path, "enhanced_algorithms.json")
    if os.path.exists(test_file):
        PROCESSED_DATA_DIR = path
        print(f"Found enhanced_algorithms.json at: {test_file}")
        break
    else:
        print(f"Tried but couldn't find file at: {test_file}")

# If we still haven't found the file, try to search for it
if PROCESSED_DATA_DIR is None:
    print("Searching for enhanced_algorithms.json in project directories...")
    for root, dirs, files in os.walk(BASE_DIR):
        if "enhanced_algorithms.json" in files:
            PROCESSED_DATA_DIR = os.path.dirname(os.path.join(root, "enhanced_algorithms.json"))
            print(f"Found file via search at: {os.path.join(root, 'enhanced_algorithms.json')}")
            break

if PROCESSED_DATA_DIR is None:
    print("WARNING: Could not find enhanced_algorithms.json anywhere in the project!")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    hybrid_search: bool = True
    alpha: float = 0.6  # Balance between vector (alpha) and keyword (1-alpha) search

class AlgorithmResult(BaseModel):
    id: str
    name: str
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str]
    difficulty: Optional[str] = None
    complexity: Optional[Dict[str, str]] = None
    use_cases: Optional[List[str]] = None
    similarity_score: float
    match_details: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    results: List[AlgorithmResult]
    enhanced_query: Optional[str] = None

def expand_algorithm_query(query: str) -> str:
    """
    Expand algorithm-related queries to include relevant algorithm terms and synonyms.
    This improves the likelihood of matching with appropriate algorithms.
    """
    # Common algorithm problem indicators
    substring_indicators = ["substring", "contiguous", "consecutive", "characters", "chars"]
    sliding_window_indicators = ["without repeating", "without duplicate", "longest", "maximum", "minimum", "subarray", "sum"]
    
    # Lowercased query for matching
    query_lower = query.lower()
    
    # Store potential algorithm categories to expand
    expansions = []
    
    # Check for substring/string problems
    if any(indicator in query_lower for indicator in substring_indicators):
        expansions.append("sliding window string substring character")
    
    # Check for sliding window problems
    if any(indicator in query_lower for indicator in sliding_window_indicators):
        expansions.append("sliding window two pointers contiguous subarray")
    
    # Special case for longest substring without repeats
    if "longest" in query_lower and "substring" in query_lower and ("without" in query_lower or "no" in query_lower) and ("duplicate" in query_lower or "repeat" in query_lower):
        expansions.append("sliding window technique hash table two pointers")
    
    # Combine original query with expansions
    if expansions:
        expanded_query = f"{query} {' '.join(expansions)}"
        return expanded_query
    
    return query

@app.post("/api/query", response_model=QueryResponse)
def query_algorithms(request: QueryRequest):
    """Query for relevant algorithms based on a problem description."""
    try:
        # Expand the query for better algorithm matching
        original_query = request.query
        expanded_query = expand_algorithm_query(original_query)
        print(f"Original query: {original_query}")
        print(f"Expanded query: {expanded_query}")
        
        # Generate embedding for the expanded query
        query_embedding = embedding_model.encode(expanded_query, normalize_embeddings=True)
        
        # Search for similar algorithm chunks
        if request.hybrid_search and hasattr(vector_store, 'hybrid_search'):
            # Use hybrid search if available
            print("Using hybrid search")
            raw_results = vector_store.hybrid_search(
                query_embedding=query_embedding, 
                query_text=expanded_query, 
                k=request.top_k * 2,
                alpha=request.alpha
            )
            # Sort by hybrid score
            raw_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        else:
            # Fallback to regular vector search
            print("Using vector search")
            raw_results = vector_store.search(query_embedding, k=request.top_k * 2)
        
        # Process results to return unique algorithms
        seen_ids = set()
        results = []
        
        # Determine the path to use for enhanced_algorithms.json
        if PROCESSED_DATA_DIR:
            algorithms_file = os.path.join(PROCESSED_DATA_DIR, "enhanced_algorithms.json")
        else:
            # Try all potential paths again
            for path in potential_paths:
                test_file = os.path.join(path, "enhanced_algorithms.json")
                if os.path.exists(test_file):
                    algorithms_file = test_file
                    break
            else:
                # Last resort - check the current directory and parent directories
                algorithms_file = None
                current_dir = os.getcwd()
                for _ in range(3):  # Check up to 3 levels up
                    test_file = os.path.join(current_dir, "data", "processed", "enhanced_algorithms.json")
                    if os.path.exists(test_file):
                        algorithms_file = test_file
                        break
                    current_dir = os.path.dirname(current_dir)
        
        if not algorithms_file or not os.path.exists(algorithms_file):
            raise FileNotFoundError(f"Could not find enhanced_algorithms.json in any expected location")
        
        print(f"Loading algorithms from: {algorithms_file}")
        with open(algorithms_file, 'r') as f:
            all_algorithms = json.load(f)
        
        for result in raw_results:
            algorithm_id = result["metadata"]["id"]
            
            if algorithm_id not in seen_ids and len(results) < request.top_k:
                seen_ids.add(algorithm_id)
                
                algorithm = next((a for a in all_algorithms if a.get("id") == algorithm_id), None)
                
                if algorithm:
                    # Determine match details for transparency
                    content = result.get("content", "")
                    match_details = None
                    
                    # Extract the most relevant snippet that matches the query
                    if content:
                        query_terms = re.findall(r'\b\w+\b', original_query.lower())
                        relevant_snippets = []
                        
                        # Find snippets containing query terms
                        for term in query_terms:
                            if term in content.lower():
                                # Find a snippet around this term
                                start = max(0, content.lower().find(term) - 100)
                                end = min(len(content), start + 200)
                                snippet = content[start:end]
                                relevant_snippets.append(snippet)
                        
                        if relevant_snippets:
                            match_details = "Matched on: " + relevant_snippets[0]
                    
                    # Choose the appropriate score field
                    if "hybrid_score" in result:
                        similarity_score = result["hybrid_score"]
                    elif "similarity_score" in result:
                        similarity_score = result["similarity_score"]
                    else:
                        # Convert distance to similarity if needed
                        similarity_score = 1.0 - result.get("distance", 0) / 2  
                    
                    # Create response entry
                    algorithm_result = AlgorithmResult(
                        id=algorithm_id,
                        name=algorithm.get("name", ""),
                        title=algorithm.get("title", ""),
                        description=algorithm.get("description", ""),
                        tags=algorithm.get("tags", []),
                        difficulty=algorithm.get("difficulty", ""),
                        complexity=algorithm.get("complexity", {}),
                        use_cases=algorithm.get("use_cases", []),
                        similarity_score=similarity_score,
                        match_details=match_details
                    )
                    
                    results.append(algorithm_result)
        
        return QueryResponse(
            query=request.query, 
            results=results,
            enhanced_query=expanded_query if expanded_query != original_query else None
        )
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in query_algorithms: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)