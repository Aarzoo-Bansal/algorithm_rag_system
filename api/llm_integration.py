import os
import sys
import json
import requests
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai as genai

# Adding parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
# for model in genai.list_models():
#     print(f"Name: {model.name}")
#     print(f"Display name: {model.display_name}")
#     print(f"Description: {model.description}")
#     print(f"Generation methods: {model.supported_generation_methods}")
#     print("---")

# # Filter to only show models that support text generation
# print("\nModels supporting text generation:")
# for model in genai.list_models():
#     if 'generateContent' in model.supported_generation_methods:
#         print(f"- {model.name}")
# function to check if the user prompt is related to algorithms or a general conversation
def is_algorithm_query(query):
    """Check if the query is related to algorithms or programming problems."""
    
    # system_prompt = """You are a query classifier specializing in algorithm and data structure problems.
    
    # A query is considered algorithm-related if it meets ANY of these criteria:
    # 1. Mentions specific algorithms (e.g., dynamic programming, BFS, DFS, etc.)
    # 2. Describes a computational problem (e.g., finding shortest path, sorting, searching)
    # 3. Mentions data structures (arrays, trees, graphs, matrices, etc.)
    # 4. Contains LeetCode-style problem descriptions
    # 5. Asks for efficient ways to solve a computational task
    # 6. Mentions complexity (time/space complexity, Big O notation)
    # 7. Involves optimization problems (maximizing/minimizing values)
    
    # Respond with ONLY 'YES' if it's algorithm-related or 'NO' if it's general conversation."""
    
    # user_prompt = f"""Query: {query}
    
    # Based on the criteria above, is this query related to algorithms, data structures, or programming problems?
    # Return ONLY 'YES' or 'NO' without explanation."""

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
        #model = genai.GenerativeModel('models/gemini-1.5-pro')

        response = model.generate_content(classification_prompt,
                                          generation_config=genai.types.GenerationConfig(
                                            temperature=0.1,
                                            top_p=0.95,
                                            top_k=40,
                                            max_output_tokens=5
                                        ))
        
        result = response.text.strip().upper()

        print(f"CLASSIFICATION RESULT: Query '{query[:50]}...' classified as: {result}")

        # Fall back to keyword-based detection if result is not YES
        # if result != "YES":
        #     # List of algorithm-related keywords
        #     # Check if any keyword is in the query
        #     query_lower = query.lower()

        #     algo_specific_words = [
        #         "algorithm", "dynamic programming", "graph algorithm", "tree traversal", 
        #         "array sorting", "binary search", "time complexity", "space complexity",
        #         "leetcode", "data structure", "bfs", "dfs", "djikstra", "knapsack",
        #         "computational", "hash table", "heap sort"]

        #     # Check for specific phrases indicating algorithm-related queries
        #     for keyword in algo_specific_words:
        #         if keyword in query_lower:
        #             print(f"Keyword match: '{keyword}' found in query, classifying as algorithm question")
        #             return True
                
        #     life_advice_indicators = [
        #         "life advice", "career advice", "what should i do with my life",
        #         "personal growth", "my future", "what career", "life path",
        #         "relationship advice", "self-improvement", "meaning of life"
        #     ]
            
        #     for phrase in life_advice_indicators:
        #         if phrase in query_lower:
        #             print(f"Life advice indicator: '{phrase}' found in query")
        #             return False
        
        return result == "A"
    

        # commenting the code for using Local LLM Server
        # client = OpenAI(
        #     base_url="http://127.0.0.1:8080/v1",  # local LLM server
        #     api_key="sk-no-key-required"
        # )
        # response = client.chat.completions.create(
        #     model="local-model",
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt}
        #     ],
        #     temperature=0.1  # Low temperature for more consistent responses
        # )
        
        # result = response.choices[0].message.content.strip().upper()
        # result = result.replace("<|eot_id|>", "").strip()

        # # Debug print
        # print(f"CLASSIFICATION RESULT: Query '{query[:50]}...' classified as: {result}")
        
        # # Fall back to keyword-based detection if result is not YES
        # if result != "YES":
        #     # List of algorithm-related keywords
        #     algo_keywords = [
        #         "algorithm", "dynamic programming", "graph", "tree", "array", "matrix", 
        #         "sort", "search", "path", "complexity", "data structure", "leetcode", 
        #         "problem", "optimization", "minimizes", "maximizes", "shortest", "longest",
        #         "minimum", "maximum", "efficient", "optimal", "grid", "dp", "bfs", "dfs",
        #         "binary search", "two pointers", "sliding window", "heap", "stack", "queue",
        #         "linked list", "hash", "trie", "backtracking", "greedy"
        #     ]
            
        #     # Check if any keyword is in the query
        #     query_lower = query.lower()
        #     for keyword in algo_keywords:
        #         if keyword in query_lower:
        #             print(f"Keyword match: '{keyword}' found in query, classifying as algorithm question")
        #             return True
        
        # return result == "YES"
        
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

    # system_prompt = """You are a helpful assistant named Algorithm Assistant. 
    # The user is asking a general question, not related to algorithms.
    # Provide a helpful, conversational response. Be brief and friendly."""
    
    # user_prompt = f"""User: {query}"""
    
    # try:
    #     client = OpenAI(
    #         base_url="http://127.0.0.1:8080/v1",  # Your local LLM server
    #         api_key="sk-no-key-required"
    #     )
        
    #     response = client.chat.completions.create(
    #         model="local-model",
    #         messages=[
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": user_prompt}
    #         ],
    #         temperature=0.7
    #     )
        
    #     return response.choices[0].message.content
        
    # except Exception as e:
    #     print(f"Error generating conversation response: {str(e)}")
    #     return "I'm sorry, I encountered an error trying to respond to your message. How can I help you with algorithm-related questions instead?"

    

def retrieve_algorithms(query, rag_endpoint="http://localhost:8000/api/query", top_k=5):
    """Retrieve relevant algorithms from RAG service."""
    try:
        response = requests.post(
            rag_endpoint,
            json={"query": query, "top_k": top_k}
        )
        
        if response.status_code == 200:
            return response.json()["results"]
        else:
            print(f"Error from RAG service: {response.text}")
            return []
            
    except Exception as e:
        print(f"Error retrieving algorithms: {str(e)}")
        return []


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


def generate_response(query):
    """Main function to process user queries and generate appropriate responses."""
    # Step 1: Determine if this is an algorithm-related query
    if is_algorithm_query(query):
        # Step 2a: It's algorithm-related, use RAG system
        print("Algorithm-related query detected.")
        return generate_algorithm_recommendation(query)
    else:
        # Step 2b: It's a general question, use conversational response
        print("General conversation query detected.")
        return generate_conversation_response(query)


def generate_algorithm_recommendation(query):
    """Generate algorithm recommendation using LLM with RAG."""
   
    # Retrieve relevant algorithms
    algorithms = retrieve_algorithms(query)
    
    if not algorithms:
        return "Could not retrieve relevant algorithms. Please try a different query."
    
    # Format context for LLM
    context = format_algorithm_context(algorithms)

    recommendation_prompt = f"""You are an algorithm specialist. Your top priority is to help guide users 
    into selecting an appropriate algorithm for their problem.

    {context}

    User Problem: {query}

    Based on the information above, which algorithm would be most suitable for solving this problem, and why? 
    Explain your reasoning in detail, including time and space complexity analysis.
    
    REMEMBER: You MUST cite the algorithm you recommend using its exact ID in the format [RAG_ID: algorithm_id]. 
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
        # Fallback response
        if algorithms:
            alg = algorithms[0]
            return f"Based on the retrieved algorithms, {alg['name']} [RAG_ID: {alg.get('id', 'unknown')}] appears to be most suitable. {alg.get('description', '')}"
        else:
            return "Could not generate a recommendation. Please try again with a different query."
        
#     # Generate system prompt
#     system_prompt = """You are an algorithm specialist. Your top priority is to help guide users 
#     into selecting an appropriate algorithm for their problem. 
    
#     IMPROTANT INSTRUCTION: When recommending an answering, you MUST cite the exact algorithm ID provided in the context.
#     Use the format [RAG_ID: algorithm_id] to cite the algorithm ID. Each algorithm in the context has a unique ID that starts with 'ID:'.

#     Example citation: "I recommend using the Binary Search algorithm [RAG_ID: binary-search-12345] for this problem."
# """
    
#     # Generate user prompt
#     user_prompt = f"""{context}
    
# User Problem: {query}

# Based on the information above, which algorithm would be most suitable for solving this problem, and why? 
# Explain your reasoning in detail, including time and space complexity analysis.

# REMEMBER: You MUST cite the algorithm you recommend using its exact ID in the format [RAG_ID: algorithm_id]. 
# This ID was provided in the context for each algorithm.
# """
    
    # Call LLM
    # client = OpenAI(
    #     base_url="http://127.0.0.1:8080/v1",  # Your local LLM server
    #     api_key="sk-no-key-required"
    # )
    
    # try:
    #     response = client.chat.completions.create(
    #         model="local-model",
    #         messages=[
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": user_prompt}
    #         ],
    #         temperature=0.3
    #     )
        
    #     return response.choices[0].message.content
        
    # except Exception as e:
    #     print(f"Error calling LLM: {str(e)}")
    #     # Fallback response
    #     return f"Based on the retrieved algorithms, {algorithms[0]['name']} appears to be most suitable. {algorithms[0].get('description', '')}"

# if __name__ == "__main__":
#     # Test the integration
#     test_query = """You are given a 0-indexed array of n integers differences, which describes the differences between each pair of consecutive integers of a hidden sequence of length (n + 1). More formally, call the hidden sequence hidden, then we have that differences[i] = hidden[i + 1] - hidden[i].

# You are further given two integers lower and upper that describe the inclusive range of values [lower, upper] that the hidden sequence can contain.

#     For example, given differences = [1, -3, 4], lower = 1, upper = 6, the hidden sequence is a sequence of length 4 whose elements are in between 1 and 6 (inclusive).
#         [3, 4, 1, 5] and [4, 5, 2, 6] are possible hidden sequences.
#         [5, 6, 3, 7] is not possible since it contains an element greater than 6.
#         [1, 2, 3, 4] is not possible since the differences are not correct. What algo can I use to test this?"""
#     recommendation = generate_algorithm_recommendation(test_query)
#     print(recommendation)