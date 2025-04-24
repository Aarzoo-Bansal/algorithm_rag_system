import os
import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

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

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Initialized embedding model")

# Initialize vector store with explicit debugging
try:
    vector_store = AlgorithmVectorStore().load_embeddings().load_index()
    print("Initialized vector store successfully")
except Exception as e:
    print(f"Error initializing vector store: {e}")
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

class QueryResponse(BaseModel):
    query: str
    results: List[AlgorithmResult]

@app.post("/api/query", response_model=QueryResponse)
def query_algorithms(request: QueryRequest):
    """Query for relevant algorithms based on a problem description."""
    try:
        # Generate embedding for the query
        query_embedding = embedding_model.encode(request.query)
        
        # Search for similar algorithm chunks
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
                        similarity_score=1.0 - result["distance"] / 2  # Convert distance to similarity
                    )
                    
                    results.append(algorithm_result)
        
        return QueryResponse(query=request.query, results=results)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in query_algorithms: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)