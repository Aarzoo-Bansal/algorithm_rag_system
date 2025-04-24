import os
import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import re

print("Current working directory:", os.getcwd())

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print("Script directory:", SCRIPT_DIR)

BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(BASE_DIR)
print("Base directory added to path:", BASE_DIR)
print("Python path:", sys.path)


try:
    from scripts.embedding.vector_store import AlgorithmVectorStore
    print("Successfully imported AlgorithmVectorStore")
except ImportError as e:
    print(f"Failed to import AlgorithmVectorStore: {e}")
    try:
        sys.path.append(os.path.join(BASE_DIR, "scripts"))
        from scripts.embedding.vector_store import AlgorithmVectorStore
        print("Successfully imported AlgorithmVectorStore from alternate path")
    except ImportError as e:
        print(f"Failed to import from alternate path: {e}")
        raise

app = FastAPI(title="Algorithm RAG Service")

MODEL_NAME = "all-mpnet-base-v2" 
embedding_model = SentenceTransformer(MODEL_NAME)
embedding_model.max_seq_length = 512 
print(f"Initialized embedding model: {MODEL_NAME}")

try:
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

potential_paths = [
    os.path.join(BASE_DIR, "data", "processed"),
    os.path.join(SCRIPT_DIR, "data", "processed"),
    os.path.join(os.getcwd(), "data", "processed"),
    os.path.join(BASE_DIR, "..", "data", "processed")
]

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
    alpha: float = 0.6 

class AlgorithmResult(BaseModel):
    id: str
    name: str
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str]
    difficulty: Optional[str] = None
    complexity: Optional[Dict[str, str]] = None
    use_cases: Optional[List[str]] = None
    problem_patterns: Optional[List[str]] = None
    leetcode_indicators: Optional[List[str]] = None
    examples: Optional[List[str]] = None
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
    substring_indicators = ["substring", "contiguous", "consecutive", "characters", "chars"]
    sliding_window_indicators = ["without repeating", "without duplicate", "longest", "maximum", "minimum", "subarray", "sum"]
    
    query_lower = query.lower()
    
    expansions = []
  
    if any(indicator in query_lower for indicator in substring_indicators):
        expansions.append("sliding window string substring character")
    
    if any(indicator in query_lower for indicator in sliding_window_indicators):
        expansions.append("sliding window two pointers contiguous subarray")
    
    if "longest" in query_lower and "substring" in query_lower and ("without" in query_lower or "no" in query_lower) and ("duplicate" in query_lower or "repeat" in query_lower):
        expansions.append("sliding window technique hash table two pointers")
        
    if "trie" in query_lower or "prefix" in query_lower or "dictionary" in query_lower:
        expansions.append("trie prefix tree dictionary autocomplete")

    if "range" in query_lower and ("query" in query_lower or "sum" in query_lower):
        expansions.append("segment tree range query interval")

    if "connected components" in query_lower or "union" in query_lower or "disjoint" in query_lower:
        expansions.append("union find disjoint set connectivity")
    
    if expansions:
        expanded_query = f"{query} {' '.join(expansions)}"
        return expanded_query
    
    return query

@app.post("/api/query", response_model=QueryResponse)
def query_algorithms(request: QueryRequest):
    """Query for relevant algorithms based on a problem description."""
    try:
        original_query = request.query
        expanded_query = expand_algorithm_query(original_query)
        print(f"Original query: {original_query}")
        print(f"Expanded query: {expanded_query}")
        
        query_embedding = embedding_model.encode(expanded_query, normalize_embeddings=True)
        
        if request.hybrid_search and hasattr(vector_store, 'hybrid_search'):
            print("Using hybrid search")
            raw_results = vector_store.hybrid_search(
                query_embedding=query_embedding, 
                query_text=expanded_query, 
                k=request.top_k * 2,
                alpha=request.alpha
            )
            raw_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        else:
            print("Using vector search")
            raw_results = vector_store.search(query_embedding, k=request.top_k * 2)
        
        seen_ids = set()
        results = []
        

        if PROCESSED_DATA_DIR:
            algorithms_file = os.path.join(PROCESSED_DATA_DIR, "enhanced_algorithms.json")
        else:
            for path in potential_paths:
                test_file = os.path.join(path, "enhanced_algorithms.json")
                if os.path.exists(test_file):
                    algorithms_file = test_file
                    break
            else:
                algorithms_file = None
                current_dir = os.getcwd()
                for _ in range(3):
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
                    content = result.get("content", "")
                    match_details = None
                    
                    if content:
                        query_terms = re.findall(r'\b\w+\b', original_query.lower())
                        relevant_snippets = []
                        
                        for term in query_terms:
                            if term in content.lower():
                                start = max(0, content.lower().find(term) - 100)
                                end = min(len(content), start + 200)
                                snippet = content[start:end]
                                relevant_snippets.append(snippet)
                        
                        if relevant_snippets:
                            match_details = "Matched on: " + relevant_snippets[0]
                    
                    if "hybrid_score" in result:
                        similarity_score = result["hybrid_score"]
                    elif "similarity_score" in result:
                        similarity_score = result["similarity_score"]
                    else:
                        similarity_score = 1.0 - result.get("distance", 0) / 2  
                    
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

@app.get("/api/query", response_model=QueryResponse)
def query_algorithms_get(
    query: str = Query(..., description="The problem description"),
    top_k: int = Query(3, description="Number of results to return"),
    hybrid_search: bool = Query(True, description="Use hybrid search"),
    alpha: float = Query(0.6, description="Hybrid search alpha parameter")
):
    """
    GET endpoint for querying algorithms. Accepts query parameters.
    """
    try:
        # Reuse your POST logic by creating a QueryRequest object
        request = QueryRequest(
            query=query,
            top_k=top_k,
            hybrid_search=hybrid_search,
            alpha=alpha
        )
        return query_algorithms(request)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in query_algorithms_get: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    