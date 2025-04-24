import os
import sys
import json
import requests
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

def is_algorithm_query(query):
    """Check if the query is related to algorithms or programming problems."""
    
    classification_prompt = f"""You are a query classifier specializing in algorithm and data structure problems


    CAREFULLY distinguish between these DIFFERENT TYPES of queries:
    A. PROBLEM-SOLVING QUERIES: Questions asking for an algorithm to solve a specific computational problem
    B. CONCEPTUAL QUERIES: General questions about what algorithms are or how they work
    C. APPLICATION QUERIES: Questions about applying algorithms in real-life contexts
    D. NON-ALGORITHM QUERIES: Questions unrelated to algorithms or data structures

    A query is considered a PROBLEM-SOLVING query (Type A) if it meets ANY of these criteria:
    1. Asks for an algorithm to solve a specific computational task
    2. Describes a programming problem that needs a solution
    3. Contains LeetCode-style problem descriptions
    4. Asks about the complexity or efficiency of solving a specific problem
    5. Requests code or pseudocode to solve a computational challenge

    EXAMPLES:
    - "How do I find the shortest path in a graph?" → A (Problem-solving query)
    - "What is dynamic programming?" → B (Conceptual query)
    - "How can I apply dynamic programming in real life?" → C (Application query)
    - "What's the weather like today?" → D (Non-algorithm query)
    - "Given an array of integers, find two numbers that add up to a target" → A (Problem-solving query)
    - "How does the merge sort algorithm work?" → B (Conceptual query)
    - "When should I use BFS vs DFS in robotics?" → C (Application query)

Query: {query}

Based on the criteria above, classify this query as either:
A (Problem-solving query)
B (Conceptual query)
C (Application query) 
D (Non-algorithm query)

Return ONLY the single letter classification without explanation."""

    try:
        model = genai.GenerativeModel('models/gemini-2.0-flash-lite-001')

        response = model.generate_content(classification_prompt,
                                          generation_config=genai.types.GenerationConfig(
                                            temperature=0.1,
                                            top_p=0.95,
                                            top_k=40,
                                            max_output_tokens=5
                                        ))
        
        result = response.text.strip().upper()

       # print(f"CLASSIFICATION RESULT: Query '{query[:50]}...' classified as: {result}")
        
        # For problem-solving or conceptual queries, use the RAG system
        return result in ["A", "B"]
    
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        # If classification fails, default to treating it as an algorithm query
        return True
    

def generate_conversation_response(query):
    """Generate a conversational response for non-algorithm queries."""

    conversation_prompt = f"""You are a helpful assistant named Algorithm Assistant. 
    The user is asking a general question, not related to algorithms.
    
    User: {query}
    
    Provide a helpful, conversational response. Be brief and friendly."""

    try:
        model = genai.GenerativeModel('models/gemini-2.0-flash-001')

        response = model.generate_content(conversation_prompt,
                                          generation_config=genai.types.GenerationConfig(
                                            temperature=0.7,
                                            top_p=0.95,
                                            top_k=40,
                                            max_output_tokens=200
                                        ))
        return response.text.strip()
    
    except Exception as e:
        print(f"Error generating conversation response: {str(e)}")
        return "I'm sorry, I encountered an error trying to respond to your message. How can I help you with algorithm-related questions instead?"


def ensure_core_algorithms_exist(algorithms, query):
    """
    Ensure core algorithms are included in results, but prioritize versions from the RAG system.
    Only adds default versions for algorithms that are needed but not already present.
    """
    import uuid
    
    # Define standard algorithm patterns to look for in the query
    query_patterns = {
        "two_pointers": ["two pointer", "pair", "palindrome", "start and end", "left and right"],
        "sliding_window": ["sliding window", "substring", "subarray", "contiguous", "consecutive"],
        "hash_table": ["hash", "lookup", "map", "dictionary", "frequency", "count", "duplicate"],
        "dynamic_programming": ["dp", "subproblem", "optimize", "maximize", "minimize", "memoize"],
        "binary_search": ["binary search", "sorted array", "log n", "find element", "search sorted"],
        "sorting": ["sort", "order", "arrange", "rank", "sequence"],
        "graph_traversal": ["graph", "network", "traversal", "dfs", "bfs", "path", "visit"]
    }
    
    # Default algorithm templates to use if not found in RAG
    core_algorithms = {
        "two_pointers": {
            "id": str(uuid.uuid4()),
            "name": "Two Pointers Technique",
            "description": "A technique that uses two pointers to iterate through a data structure. Often used for problems involving arrays, linked lists, or strings to find pairs that satisfy certain conditions.",
            "tags": ["array", "linked list", "string", "optimization"],
            "complexity": {"time": "O(n)", "space": "O(1)"},
            "similarity_score": 0.85
        },
        "sliding_window": {
            "id": str(uuid.uuid4()),
            "name": "Sliding Window Technique",
            "description": "An efficient technique for processing contiguous sequences in arrays or strings. It maintains a window of elements and slides it through the data to find optimal solutions.",
            "tags": ["array", "string", "optimization"],
            "complexity": {"time": "O(n)", "space": "O(1)"},
            "similarity_score": 0.85
        },
        "hash_table": {
            "id": str(uuid.uuid4()),
            "name": "Hash Table",
            "description": "A data structure that provides efficient insertion, deletion, and lookup operations. Used for quick access to elements based on keys.",
            "tags": ["lookup", "optimization", "data structure"],
            "complexity": {"time": "O(1) average", "space": "O(n)"},
            "similarity_score": 0.85
        },
        "dynamic_programming": {
            "id": str(uuid.uuid4()),
            "name": "Dynamic Programming",
            "description": "A method for solving complex problems by breaking them down into simpler subproblems. It stores the results of subproblems to avoid redundant calculations.",
            "tags": ["optimization", "recursion", "memoization"],
            "complexity": {"time": "Problem dependent", "space": "Usually O(n) to O(n²)"},
            "similarity_score": 0.85
        },
        "binary_search": {
            "id": str(uuid.uuid4()),
            "name": "Binary Search",
            "description": "A divide-and-conquer algorithm for finding a target value in a sorted array by repeatedly dividing the search interval in half.",
            "tags": ["search", "divide and conquer", "sorted array"],
            "complexity": {"time": "O(log n)", "space": "O(1)"},
            "similarity_score": 0.85
        },
        "sorting": {
            "id": str(uuid.uuid4()),
            "name": "Sorting Algorithms",
            "description": "Algorithms for arranging elements in a specific order, such as numerical or lexicographical. Common examples include Quick Sort, Merge Sort, and Heap Sort.",
            "tags": ["array", "comparison", "ordering"],
            "complexity": {"time": "O(n log n) for comparison-based sorts", "space": "O(1) to O(n)"},
            "similarity_score": 0.85
        },
        "graph_traversal": {
            "id": str(uuid.uuid4()),
            "name": "Graph Traversal Algorithms",
            "description": "Algorithms for visiting all nodes in a graph, such as Depth-First Search (DFS) and Breadth-First Search (BFS).",
            "tags": ["graph", "tree", "search"],
            "complexity": {"time": "O(V + E)", "space": "O(V)"},
            "similarity_score": 0.85
        }
    }
    
   
    found_algorithm_types = set()
    
   
    def algorithm_matches_type(algorithm, algo_type):
        name = algorithm.get("name", "").lower()
        description = algorithm.get("description", "").lower() 
        tags = [tag.lower() for tag in algorithm.get("tags", [])]
        
        # Check for each algorithm type with specific conditions
        if algo_type == "two_pointers":
            return ("two pointer" in name or 
                   "two pointers" in name or
                   "two pointer" in description or
                   any(tag in ["two pointers", "two pointer"] for tag in tags))
                   
        elif algo_type == "sliding_window":
            return ("sliding window" in name or 
                   "sliding window" in description or
                   "sliding" in tags)
                   
        elif algo_type == "hash_table":
            return ("hash" in name or 
                   "hash table" in name or 
                   "dictionary" in name or
                   any(tag in ["hash", "hash table", "dictionary"] for tag in tags))
        
        elif algo_type == "dynamic_programming":
            return ("dynamic programming" in name or 
                   "dp" in name or
                   any(tag in ["dynamic programming", "dp"] for tag in tags))
        
        elif algo_type == "binary_search":
            return ("binary search" in name or
                   "binary search" in description or
                   any(tag in ["binary search", "search", "logarithmic"] for tag in tags))
        
        elif algo_type == "sorting":
            return ("sort" in name or
                   any(tag in ["sort", "sorting"] for tag in tags))
        
        elif algo_type == "graph_traversal":
            return (any(term in name for term in ["graph", "dfs", "bfs", "traversal"]) or
                   any(tag in ["graph", "dfs", "bfs", "traversal"] for tag in tags))
        
        return False
    
  
    for algorithm in algorithms:
        for algo_type in core_algorithms.keys():
            if algorithm_matches_type(algorithm, algo_type):
                found_algorithm_types.add(algo_type)
    
    
    needed_algorithm_types = set()
    query_lower = query.lower()
    
    for algo_type, patterns in query_patterns.items():
        if any(pattern in query_lower for pattern in patterns):
            needed_algorithm_types.add(algo_type)
    
    
    if is_problem_solving_query(query) and not needed_algorithm_types:
        needed_algorithm_types = {"hash_table", "two_pointers", "dynamic_programming"}
    
   
    added_algorithms = []
    for algo_type in needed_algorithm_types:
        if algo_type not in found_algorithm_types:
            algo = core_algorithms[algo_type].copy()
            algo["id"] = str(uuid.uuid4()) 
            added_algorithms.append(algo)

    return algorithms + added_algorithms


def is_problem_solving_query(query):
    """Simplified logic to determine if a query is algorithm problem-solving related."""
    query_lower = query.lower()
    problem_indicators = [
        "find", "solve", "calculate", "compute", "determine", 
        "algorithm for", "how to", "implement", "optimize",
        "given an array", "leetcode", "problem", "challenge",
        "complexity", "efficient", "time complexity"
    ]
    
    return any(indicator in query_lower for indicator in problem_indicators)

def handle_sum_problem(query):
    """Create a specific algorithm entry for sum problems."""
    query_lower = query.lower()
    
    if "4 sum" in query_lower or "four sum" in query_lower:
        return {
            "id": "four-sum-algorithm-composed",
            "name": "Four Sum Algorithm",
            "description": "A technique to find all unique quadruplets in an array that sum up to a target value. Combines sorting with the two pointers technique.",
            "tags": ["array", "two pointers", "sorting", "hash table"],
            "complexity": {"time": "O(n³) or O(n²) with hash table", "space": "O(n) to O(n²)"},
            "similarity_score": 0.95 
        }
    elif "3 sum" in query_lower or "three sum" in query_lower:
        return {
            "id": "three-sum-algorithm-composed",
            "name": "Three Sum Algorithm",
            "description": "A technique to find all unique triplets in an array that sum up to a target value. Typically uses sorting with two pointers approach.",
            "tags": ["array", "two pointers", "sorting"],
            "complexity": {"time": "O(n²)", "space": "O(1) to O(n)"},
            "similarity_score": 0.95
        }
    elif "2 sum" in query_lower or "two sum" in query_lower:
        return {
            "id": "two-sum-algorithm-composed",
            "name": "Two Sum Algorithm",
            "description": "A technique to find pairs in an array that sum up to a target value. Most efficiently done using a hash table.",
            "tags": ["array", "hash table"],
            "complexity": {"time": "O(n)", "space": "O(n)"},
            "similarity_score": 0.95
        }
    elif "k sum" in query_lower:
        return {
            "id": "k-sum-algorithm-composed",
            "name": "K Sum Algorithm",
            "description": "A general technique to find k elements in an array that sum up to a target value. Typically uses recursion to reduce to simpler cases.",
            "tags": ["array", "two pointers", "recursion", "hash table"],
            "complexity": {"time": "O(nᵏ⁻¹) or better with optimizations", "space": "O(n) to O(nᵏ⁻¹)"},
            "similarity_score": 0.95
        }
    return None

def retrieve_from_rag(query, rag_endpoint="http://localhost:8000/api/query", top_k=5, params=None):
    """Basic retrieval from RAG service without enhancements."""
    try:
        payload = {
            "query": query, 
            "top_k": top_k,
            "hybrid_search": True,
            "alpha": 0.6 
        }
        
        if params:
            payload.update(params)
        
        response = requests.post(
            rag_endpoint,
            json=payload
        )
        
        if response.status_code == 200:
            response_data = response.json()
            
            if response_data.get("enhanced_query"):
                print(f"Enhanced query used: {response_data['enhanced_query']}")
                
            return response_data["results"]
        else:
            print(f"Error from RAG service: {response.text}")
            return []
            
    except Exception as e:
        print(f"Error retrieving algorithms: {str(e)}")
        return []

def retrieve_algorithms(query, rag_endpoint="http://localhost:8000/api/query", top_k=5, params=None):
    special_case = handle_sum_problem(query)
    
    results = retrieve_from_rag(query, rag_endpoint, top_k, params)
    
    if special_case:
        if not any(result.get("id") == special_case["id"] for result in results):
            results.insert(0, special_case)
    
    enhanced_results = ensure_core_algorithms_exist(results, query)
    
    return enhanced_results

def extract_general_concepts(query):
    """Extract general algorithmic concepts from a specific problem query."""
    query_lower = query.lower()
    
    general_concepts = []
    
    concept_mapping = {
        "sum": ["two pointers", "hash table", "sorting"],
        "substring": ["sliding window", "two pointers", "string algorithms"],
        "path": ["graph algorithms", "dfs", "bfs", "shortest path"],
        "tree": ["tree traversal", "binary tree", "dfs", "bfs"],
        "array": ["array manipulation", "sorting", "two pointers"],
        "parenthesis": ["stack", "backtracking", "dynamic programming"],
        "palindrome": ["two pointers", "string algorithms", "dynamic programming"],
        "subsequence": ["dynamic programming", "two pointers", "greedy"],
        "permutation": ["backtracking", "recursion"],
        "combination": ["backtracking", "recursion", "dynamic programming"],
        "sort": ["sorting algorithms", "quicksort", "mergesort"],
        "grid": ["matrix algorithms", "dfs", "bfs", "dynamic programming"],
        "interval": ["greedy", "sorting", "interval algorithms"],
        "binary search": ["binary search", "divide and conquer"],
        "maximum": ["greedy", "dynamic programming"],
        "minimum": ["greedy", "dynamic programming"],
    }
    
    for pattern, techniques in concept_mapping.items():
        if pattern in query_lower:
            general_concepts.extend(techniques)
    
    common_primitives = ["sorting", "hash table", "two pointers", "dynamic programming"]
    general_concepts.extend(common_primitives)
    
    general_concepts = list(set(general_concepts))
    return f"general algorithm techniques: {', '.join(general_concepts)}"


def retrieve_techniques_and_primitives(query, rag_endpoint="http://localhost:8000/api/query"):
    """
    Retrieve both specific algorithms and general techniques to support
    algorithm composition for problems without exact matches.
    """
 
    specific_results = retrieve_algorithms(
        query, 
        rag_endpoint=rag_endpoint, 
        top_k=3,
        params={"hybrid_search": True, "alpha": 0.7}
    )
    
    general_query = extract_general_concepts(query)
    general_results = retrieve_algorithms(
        general_query,
        rag_endpoint=rag_endpoint,
        top_k=5,
        params={"hybrid_search": True, "alpha": 0.5}  
    )
    
    all_results = []
    seen_ids = set()
    
    
    for result in specific_results:
        result_id = result.get('id')
        if result_id and result_id not in seen_ids:
            seen_ids.add(result_id)
            all_results.append(result)
    
    for result in general_results:
        result_id = result.get('id')
        if result_id and result_id not in seen_ids:
            seen_ids.add(result_id)
            all_results.append(result)
    
    all_results = ensure_core_algorithms_exist(all_results, query)
    
    return {
        "specific_results": specific_results,
        "general_results": general_results,
        "all_results": all_results
    }

def format_algorithm_context(algorithms):
    """Format algorithm information for LLM context."""
    context = "Based on the problem description, here are relevant algorithms:\n\n"
    
    for i, alg in enumerate(algorithms):
        alg_id = alg.get('id', f"unknown-{i}")
        context += f"Algorithm {i+1}: [ID: {alg_id}]: {alg['name']}\n"
        
        if alg.get('description'):
            context += f"Description: {alg['description']}\n"
            
        if alg.get('tags'):
            context += f"Tags: {', '.join(alg['tags'])}\n"
            
        if alg.get('complexity'):
            complexity = alg['complexity']
            time_complexity = complexity.get('time', 'Not specified')
            space_complexity = complexity.get('space', 'Not specified')
            context += f"Complexity: Time - {time_complexity}, Space - {space_complexity}\n"
            
        if alg.get('use_cases'):
            context += f"Use Cases: {', '.join(alg['use_cases'])}\n"
            
        context += f"Match Score: {alg.get('similarity_score', 0):.2f}\n\n"

    context += "IMPORTANT: When recommending an algorithm, you MUST cite it using its exact ID in this format: [ID: algorithm_id]\n\n"
    
    return context

def format_algorithm_composition_prompt(query, results_data):
    """
    Format prompt for LLM to compose algorithms from retrieved techniques.
    """
    prompt = f"""You are an algorithm composition specialist. Your task is to recommend and COMPOSE a solution 
for a specific algorithmic problem that might not have an exact match in our database.

USER PROBLEM: {query}

AVAILABLE ALGORITHMS AND TECHNIQUES:
"""
    
    if results_data["specific_results"]:
        prompt += "\n## SPECIFIC ALGORITHMS THAT MIGHT BE RELEVANT:\n"
        for i, alg in enumerate(results_data["specific_results"]):
            alg_id = alg.get('id', f"unknown-{i}")
            prompt += f"Algorithm {i+1}: [ID: {alg_id}]: {alg.get('name', '')}\n"
            
            if alg.get('description'):
                prompt += f"Description: {alg.get('description', '')}\n"
                
            if alg.get('tags'):
                prompt += f"Tags: {', '.join(alg.get('tags', []))}\n"
                
            if alg.get('complexity') and isinstance(alg.get('complexity'), dict):
                complexity = alg.get('complexity')
                time_complexity = complexity.get('time', 'Not specified')
                space_complexity = complexity.get('space', 'Not specified')
                prompt += f"Complexity: Time - {time_complexity}, Space - {space_complexity}\n"
            
            prompt += f"Match Score: {alg.get('similarity_score', 0):.2f}\n\n"
    
    prompt += "\n## GENERAL TECHNIQUES THAT CAN BE COMBINED:\n"
    for i, alg in enumerate(results_data["general_results"]):
        alg_id = alg.get('id', f"unknown-{i}")
        prompt += f"Technique {i+1}: [ID: {alg_id}]: {alg.get('name', '')}\n"
        
        if alg.get('description'):
            description = alg.get('description', '')
            if len(description) > 400:
                description = description[:400] + "..."
            prompt += f"Description: {description}\n"
            
        if alg.get('tags'):
            prompt += f"Tags: {', '.join(alg.get('tags', []))}\n"
        
        prompt += "\n"
    
    prompt += """
            ## YOUR TASK:
            1. Analyze the user's problem carefully.
            2. Draw from both specific algorithms and general techniques listed above.
            3. COMPOSE a comprehensive algorithm solution tailored to this specific problem.
            4. Explain the approach step by step, including time and space complexity analysis.
            5. IMPORTANT: Cite ALL algorithms/techniques you use with their IDs in this format: [ID: algorithm_id]

            Your solution should be DETAILED and SPECIFIC to this problem, even if you need to combine multiple techniques.
            If the specific algorithms don't perfectly match, explain how to adapt or combine them.

            ## RESPONSE FORMAT:
            - Recommended Approach: [Brief overview of your composed solution]
            - Algorithm Composition: [Detailed explanation of how to combine/adapt techniques]
            - Implementation Steps: [Step-by-step walkthrough]
            - Time & Space Complexity: [Analysis of your composed solution]
            - Key Techniques Used: [List techniques with their IDs]
            """
    
    return prompt

def assess_match_quality(algorithms, query):
    """
    Assess whether the retrieved algorithms match the query well enough
    or if we need to compose algorithms.
    """
    if not algorithms:
        return "compose"
    
    top_score = algorithms[0].get('similarity_score', 0) if algorithms else 0
    
    query_lower = query.lower()
    specific_problems = [
        "4 sum", "k sum", "four sum", "3 sum", "three sum", "two sum", "2 sum",
        "longest palindromic subsequence",
        "k nearest neighbors",
        "serialize and deserialize binary tree",
        "word ladder",
        "sudoku solver",
        "longest increasing subsequence",
        "lru cache",
        "implement trie"
    ]
    
    for problem in specific_problems:
        if problem in query_lower and top_score < 0.85:
            return "compose"
    
    if top_score < 0.6:
        return "compose"
    
    return "standard"



def generate_algorithm_recommendation(query):
    """Generate algorithm recommendation using LLM with RAG and composition if needed."""
    
    algorithms = retrieve_algorithms(query)
    match_quality = assess_match_quality(algorithms, query)
    
    for algorithm in algorithms:
        print(f"algoL ' : {algorithm}")
    
    print(f"Retrieved {len(algorithms)} algorithms")
    print(f"Top algorithm: {algorithms[0]['name'] if algorithms else 'None'}")
    print(f"Top score: {algorithms[0].get('similarity_score', 0) if algorithms else 0}")
    print(f"Match quality assessment: {match_quality}") 
    
    
    
    if match_quality == "compose":
        print("No exact match found, using algorithm composition")
        return generate_algorithm_composition(query)
    else:
        print("Found good matches, using standard recommendation")
        
        if not algorithms:
            return "Could not retrieve relevant algorithms. Please try a different query."
        
        context = format_algorithm_context(algorithms)
    
        recommendation_prompt = f"""You are an algorithm specialist. Your top priority is to help guide users 
        into selecting an appropriate algorithm for their problem.
    
        {context}
    
        User Problem: {query}
    
        Based on the information above, which algorithm would be most suitable for solving this problem, and why? 
        Explain your reasoning in detail, including time and space complexity analysis.
        
        REMEMBER: You MUST cite the algorithm you recommend using its exact ID in the format [ID: algorithm_id]. 
        This ID was provided in the context for each algorithm."""
    
        try:
            model = genai.GenerativeModel('models/gemini-1.5-pro')
    
            response = model.generate_content(recommendation_prompt,
                                            generation_config=genai.types.GenerationConfig(
                                                temperature=0.3,
                                                top_p=0.95,
                                                top_k=40,
                                                max_output_tokens=1000
                                            ))
            
            return response.text.strip()
        
        except Exception as e:
            print(f"Error calling Gemini: {str(e)}")
            if algorithms:
                alg = algorithms[0]
                return f"Based on the retrieved algorithms, {alg['name']} [ID: {alg.get('id', 'unknown')}] appears to be most suitable. {alg.get('description', '')}"
            else:
                return "Could not generate a recommendation. Please try again with a different query."

def generate_algorithm_composition(query):
    """Generate composed algorithm recommendation using LLM and RAG."""
    
    results_data = retrieve_techniques_and_primitives(query)
    
    if not results_data["all_results"]:
        return "Could not retrieve relevant algorithms or techniques. Please try a different query."
    
    prompt = format_algorithm_composition_prompt(query, results_data)

    try:
        model = genai.GenerativeModel('models/gemini-1.5-pro')

        response = model.generate_content(prompt,
                                        generation_config=genai.types.GenerationConfig(
                                            temperature=0.3,
                                            top_p=0.95,
                                            top_k=40,
                                            max_output_tokens=1500
                                        ))
        
        return response.text.strip()
    
    except Exception as e:
        print(f"Error calling Gemini: {str(e)}")
        if results_data["specific_results"]:
            alg = results_data["specific_results"][0]
            return f"Based on the available algorithms, you can adapt {alg['name']} [ID: {alg.get('id', 'unknown')}] for this problem. {alg.get('description', '')}"
        else:
            return "Could not generate a recommendation. Please try again with a different query."

def generate_response(query):
    """Main function to process user queries and generate appropriate responses."""
    
    if is_algorithm_query(query):
        print("Algorithm-related query detected.")
        return generate_algorithm_recommendation(query)
    else:
        print("General conversation query detected.")
        return generate_conversation_response(query)


if __name__ == "__main__":
    test_query = "Given an array of integers, find four elements that sum to a given target value."
    result = generate_response(test_query)
    print("\nFINAL RESPONSE:")
    print(result)