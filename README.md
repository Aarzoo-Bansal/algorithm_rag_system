# Algorithm RAG System

A Retrieval Augmented Generation (RAG) system designed to recommend appropriate algorithms for programming problems. This system combines vector similarity search with language model reasoning to provide tailored algorithm recommendations.

## Project Overview

The Algorithm RAG System analyzes a user's problem description and recommends the most suitable algorithm(s) for solving it. The system:

1. Processes and indexes a comprehensive database of algorithms
2. Uses semantic search to find relevant algorithms for a specific problem
3. Leverages LLMs to generate detailed recommendations with explanations

## Key Features

- Semantic matching between problem descriptions and algorithm characteristics
- Hybrid search combining vector similarity and keyword matching
- Algorithm composition for problems requiring multiple techniques
- Web interface for interactive algorithm recommendations

## System Architecture

The system consists of the following components:

### 1. Data Processing Pipeline

- `generate_algorithm_database.py`: Creates a comprehensive database of algorithms with metadata
- `data_processor.py`: Processes raw algorithm data into chunks suitable for embedding
- `enhance_with_nlp.py`: Adds NLP-based metadata to algorithms

### 2. Embedding and Vector Storage

- `generate_embeddings.py`: Creates vector embeddings for algorithm chunks
- `vector_store.py`: Implements FAISS-based vector storage and retrieval

### 3. RAG Service

- `rag_service.py`: FastAPI service that handles retrieval and query processing
- `llm_integration.py`: Integrates with LLMs for response generation

### 4. Web Interface

- `app.py`: Flask-based web interface for interacting with the system

## Algorithm Coverage

The system covers a wide range of algorithm types:

- **Sorting Algorithms**: Quick Sort, Merge Sort, Heap Sort, etc.
- **Searching Algorithms**: Binary Search, DFS, BFS
- **Graph Algorithms**: Dijkstra's, Kruskal's, Topological Sort
- **Dynamic Programming**: 0/1 Knapsack, LCS, Coin Change
- **Data Structures**: Linked Lists, Stacks, Queues, Hash Tables
- **String Algorithms**: KMP, Rabin-Karp, Longest Palindromic Substring
- **Tree Algorithms**: Tree Traversal, BST Operations, Lowest Common Ancestor
- **Advanced Data Structures**: Trie, Segment Tree, Union Find
- **Specialized Techniques**: Two Pointers, Sliding Window, Monotonic Stack
- **Algorithm Paradigms**: Backtracking, Bit Manipulation, Line Sweep

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Aarzoo-Bansal/algorithm_rag_system.git
   cd algorithm-rag-system
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate 
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (I have used Gemini API KEY)
   ```

## Usage

### Generating the Algorithm Database

1. Generate the initial algorithm database:
   ```bash
   python scripts/generate_algorithm_database.py
   ```

2. Process the raw data:
   ```bash
   python scripts/processors/data_processor.py
   ```

3. Enhance with NLP:
   ```bash
   python scripts/processors/enhance_with_nlp.py
   ```

### Creating Embeddings

Generate embeddings for the processed algorithm chunks:
```bash
python scripts/embedding/generate_embeddings.py
```

### Running the RAG Service

Start the RAG service:
```bash
python -m uvicorn api.rag_service:app --host 0.0.0.0 --port 8000
```

### Starting the Web Interface

Run the web application:
```bash
python app.py
```

Access the web interface at http://localhost:5001

## Using the API

You can interact with the RAG service directly via its API:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/query",
    json={
        "query": "Find the longest substring without repeating characters",
        "top_k": 3,
        "hybrid_search": True
    }
)

results = response.json()
print(results)
```

## Example Queries

The system can handle various types of algorithm questions, such as:

- "How do I find the shortest path in a weighted graph?"
- "What algorithm should I use to detect cycles in a linked list?"
- "Given an array of integers, find four elements that sum to a target value"
- "Find the longest substring without repeating characters"
- "How to efficiently find if a string is a palindrome?"

## Project Structure

```
algorithm-rag-system/
├── api/
│   ├── llm_integration.py
│   ├── rag_service.py
├── data/
│   ├── raw/
│   ├── extra_data/
│   ├── processed/
│   ├── embeddings/
├── scripts/
│   ├── scrapers/
│   │   ├── custom_algorithms.py
│   │   ├── geeksforgeeks_scraper.py
│   ├── processors/
│   │   ├── data_processor.py
│   │   ├── nlp_enhancer.py
│   ├── embedding/
│   │   ├── generate_embeddings.py
│   │   ├── vector_store.py
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

## Enhancement Ideas

- Add more specialized algorithms for specific domains (e.g., computational geometry)
- Improve retrieval by using more advanced reranking techniques
- Implement cross-algorithm recommendations for hybrid problems
- Add code generation for algorithm implementation
- Create a visualization component to illustrate algorithm execution
