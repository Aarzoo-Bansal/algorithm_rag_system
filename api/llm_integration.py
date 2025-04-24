# # # import os
# # # import sys
# # # import json
# # # import requests
# # # from typing import List, Dict, Any
# # # from openai import OpenAI
# # # from dotenv import load_dotenv
# # # import google.generativeai as genai

# # # # Adding parent directory to path for imports
# # # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # # load_dotenv()

# # # # Configure Gemini API
# # # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # # genai.configure(api_key=GEMINI_API_KEY)
# # # # for model in genai.list_models():
# # # #     print(f"Name: {model.name}")
# # # #     print(f"Display name: {model.display_name}")
# # # #     print(f"Description: {model.description}")
# # # #     print(f"Generation methods: {model.supported_generation_methods}")
# # # #     print("---")

# # # # # Filter to only show models that support text generation
# # # # print("\nModels supporting text generation:")
# # # # for model in genai.list_models():
# # # #     if 'generateContent' in model.supported_generation_methods:
# # # #         print(f"- {model.name}")
# # # # function to check if the user prompt is related to algorithms or a general conversation
# # # def is_algorithm_query(query):
# # #     """Check if the query is related to algorithms or programming problems."""
    
# # #     # system_prompt = """You are a query classifier specializing in algorithm and data structure problems.
    
# # #     # A query is considered algorithm-related if it meets ANY of these criteria:
# # #     # 1. Mentions specific algorithms (e.g., dynamic programming, BFS, DFS, etc.)
# # #     # 2. Describes a computational problem (e.g., finding shortest path, sorting, searching)
# # #     # 3. Mentions data structures (arrays, trees, graphs, matrices, etc.)
# # #     # 4. Contains LeetCode-style problem descriptions
# # #     # 5. Asks for efficient ways to solve a computational task
# # #     # 6. Mentions complexity (time/space complexity, Big O notation)
# # #     # 7. Involves optimization problems (maximizing/minimizing values)
    
# # #     # Respond with ONLY 'YES' if it's algorithm-related or 'NO' if it's general conversation."""
    
# # #     # user_prompt = f"""Query: {query}
    
# # #     # Based on the criteria above, is this query related to algorithms, data structures, or programming problems?
# # #     # Return ONLY 'YES' or 'NO' without explanation."""

# # #     classification_prompt = f"""You are a query classifier specializing in algorithm and data structure problems


# # #     CAREFULLY distinguish between these DIFFERENT TYPES of queries:
# # #     A. PROBLEM-SOLVING QUERIES: Questions asking for an algorithm to solve a specific computational problem
# # #     B. CONCEPTUAL QUERIES: General questions about what algorithms are or how they work
# # #     C. APPLICATION QUERIES: Questions about applying algorithms in real-life contexts
# # #     D. NON-ALGORITHM QUERIES: Questions unrelated to algorithms or data structures

# # #     A query is considered a PROBLEM-SOLVING query (Type A) if it meets ANY of these criteria:
# # #     1. Asks for an algorithm to solve a specific computational task
# # #     2. Describes a programming problem that needs a solution
# # #     3. Contains LeetCode-style problem descriptions
# # #     4. Asks about the complexity or efficiency of solving a specific problem
# # #     5. Requests code or pseudocode to solve a computational challenge

# # #     EXAMPLES:
# # #     - "How do I find the shortest path in a graph?" → A (Problem-solving query)
# # #     - "What is dynamic programming?" → B (Conceptual query)
# # #     - "How can I apply dynamic programming in real life?" → C (Application query)
# # #     - "What's the weather like today?" → D (Non-algorithm query)
# # #     - "Given an array of integers, find two numbers that add up to a target" → A (Problem-solving query)
# # #     - "How does the merge sort algorithm work?" → B (Conceptual query)
# # #     - "When should I use BFS vs DFS in robotics?" → C (Application query)

# # # Query: {query}

# # # Based on the criteria above, classify this query as either:
# # # A (Problem-solving query)
# # # B (Conceptual query)
# # # C (Application query) 
# # # D (Non-algorithm query)

# # # Return ONLY the single letter classification without explanation."""

# # #     try:
# # #         model = genai.GenerativeModel('models/gemini-2.0-flash-lite-001')
# # #         #model = genai.GenerativeModel('models/gemini-1.5-pro')

# # #         response = model.generate_content(classification_prompt,
# # #                                           generation_config=genai.types.GenerationConfig(
# # #                                             temperature=0.1,
# # #                                             top_p=0.95,
# # #                                             top_k=40,
# # #                                             max_output_tokens=5
# # #                                         ))
        
# # #         result = response.text.strip().upper()

# # #         print(f"CLASSIFICATION RESULT: Query '{query[:50]}...' classified as: {result}")

# # #         # Fall back to keyword-based detection if result is not YES
# # #         # if result != "YES":
# # #         #     # List of algorithm-related keywords
# # #         #     # Check if any keyword is in the query
# # #         #     query_lower = query.lower()

# # #         #     algo_specific_words = [
# # #         #         "algorithm", "dynamic programming", "graph algorithm", "tree traversal", 
# # #         #         "array sorting", "binary search", "time complexity", "space complexity",
# # #         #         "leetcode", "data structure", "bfs", "dfs", "djikstra", "knapsack",
# # #         #         "computational", "hash table", "heap sort"]

# # #         #     # Check for specific phrases indicating algorithm-related queries
# # #         #     for keyword in algo_specific_words:
# # #         #         if keyword in query_lower:
# # #         #             print(f"Keyword match: '{keyword}' found in query, classifying as algorithm question")
# # #         #             return True
                
# # #         #     life_advice_indicators = [
# # #         #         "life advice", "career advice", "what should i do with my life",
# # #         #         "personal growth", "my future", "what career", "life path",
# # #         #         "relationship advice", "self-improvement", "meaning of life"
# # #         #     ]
            
# # #         #     for phrase in life_advice_indicators:
# # #         #         if phrase in query_lower:
# # #         #             print(f"Life advice indicator: '{phrase}' found in query")
# # #         #             return False
        
# # #         return result == "A"
    

# # #         # commenting the code for using Local LLM Server
# # #         # client = OpenAI(
# # #         #     base_url="http://127.0.0.1:8080/v1",  # local LLM server
# # #         #     api_key="sk-no-key-required"
# # #         # )
# # #         # response = client.chat.completions.create(
# # #         #     model="local-model",
# # #         #     messages=[
# # #         #         {"role": "system", "content": system_prompt},
# # #         #         {"role": "user", "content": user_prompt}
# # #         #     ],
# # #         #     temperature=0.1  # Low temperature for more consistent responses
# # #         # )
        
# # #         # result = response.choices[0].message.content.strip().upper()
# # #         # result = result.replace("<|eot_id|>", "").strip()

# # #         # # Debug print
# # #         # print(f"CLASSIFICATION RESULT: Query '{query[:50]}...' classified as: {result}")
        
# # #         # # Fall back to keyword-based detection if result is not YES
# # #         # if result != "YES":
# # #         #     # List of algorithm-related keywords
# # #         #     algo_keywords = [
# # #         #         "algorithm", "dynamic programming", "graph", "tree", "array", "matrix", 
# # #         #         "sort", "search", "path", "complexity", "data structure", "leetcode", 
# # #         #         "problem", "optimization", "minimizes", "maximizes", "shortest", "longest",
# # #         #         "minimum", "maximum", "efficient", "optimal", "grid", "dp", "bfs", "dfs",
# # #         #         "binary search", "two pointers", "sliding window", "heap", "stack", "queue",
# # #         #         "linked list", "hash", "trie", "backtracking", "greedy"
# # #         #     ]
            
# # #         #     # Check if any keyword is in the query
# # #         #     query_lower = query.lower()
# # #         #     for keyword in algo_keywords:
# # #         #         if keyword in query_lower:
# # #         #             print(f"Keyword match: '{keyword}' found in query, classifying as algorithm question")
# # #         #             return True
        
# # #         # return result == "YES"
        
# # #     except Exception as e:
# # #         print(f"Error in classification: {str(e)}")
# # #         # If classification fails, default to treating it as an algorithm query
# # #         return True
    

# # # def generate_conversation_response(query):
# # #     """Generate a conversational response for non-algorithm queries."""

# # #     conversation_prompt = f"""You are a helpful assistant named Algorithm Assistant. 
# # #     The user is asking a general question, not related to algorithms.
    
# # #     User: {query}
    
# # #     Provide a helpful, conversational response. Be brief and friendly."""

# # #     try:
# # #         model = genai.GenerativeModel('models/gemini-2.0-flash-001')

# # #         response = model.generate_content(conversation_prompt,
# # #                                           generation_config=genai.types.GenerationConfig(
# # #                                             temperature=0.7,
# # #                                             top_p=0.95,
# # #                                             top_k=40,
# # #                                             max_output_tokens=200
# # #                                         ))
# # #         return response.text.strip()
    
# # #     except Exception as e:
# # #         print(f"Error generating conversation response: {str(e)}")
# # #         return "I'm sorry, I encountered an error trying to respond to your message. How can I help you with algorithm-related questions instead?"

# # #     # system_prompt = """You are a helpful assistant named Algorithm Assistant. 
# # #     # The user is asking a general question, not related to algorithms.
# # #     # Provide a helpful, conversational response. Be brief and friendly."""
    
# # #     # user_prompt = f"""User: {query}"""
    
# # #     # try:
# # #     #     client = OpenAI(
# # #     #         base_url="http://127.0.0.1:8080/v1",  # Your local LLM server
# # #     #         api_key="sk-no-key-required"
# # #     #     )
        
# # #     #     response = client.chat.completions.create(
# # #     #         model="local-model",
# # #     #         messages=[
# # #     #             {"role": "system", "content": system_prompt},
# # #     #             {"role": "user", "content": user_prompt}
# # #     #         ],
# # #     #         temperature=0.7
# # #     #     )
        
# # #     #     return response.choices[0].message.content
        
# # #     # except Exception as e:
# # #     #     print(f"Error generating conversation response: {str(e)}")
# # #     #     return "I'm sorry, I encountered an error trying to respond to your message. How can I help you with algorithm-related questions instead?"

    

# # # def retrieve_algorithms(query, rag_endpoint="http://localhost:8000/api/query", top_k=5):
# # #     """Retrieve relevant algorithms from RAG service."""
# # #     try:
# # #         response = requests.post(
# # #             rag_endpoint,
# # #             json={"query": query, "top_k": top_k}
# # #         )
        
# # #         if response.status_code == 200:
# # #             return response.json()["results"]
# # #         else:
# # #             print(f"Error from RAG service: {response.text}")
# # #             return []
            
# # #     except Exception as e:
# # #         print(f"Error retrieving algorithms: {str(e)}")
# # #         return []


# # # def format_algorithm_context(algorithms):
# # #     """Format algorithm information for LLM context."""
# # #     context = "Based on the problem description, here are relevant algorithms:\n\n"
    
# # #     for i, alg in enumerate(algorithms):

# # #         alg_id = alg.get('id', f"unknown-{i}")
# # #         context += f"Algorithm {i+1}: [ID: {alg_id}]: {alg['name']}\n"
        
# # #         if alg.get('description'):
# # #             context += f"Description: {alg['description']}\n"
            
# # #         if alg.get('tags'):
# # #             context += f"Tags: {', '.join(alg['tags'])}\n"
            
# # #         if alg.get('complexity'):
# # #             complexity = alg['complexity']
# # #             time_complexity = complexity.get('time', 'Not specified')
# # #             space_complexity = complexity.get('space', 'Not specified')
# # #             context += f"Complexity: Time - {time_complexity}, Space - {space_complexity}\n"
            
# # #         if alg.get('use_cases'):
# # #             context += f"Use Cases: {', '.join(alg['use_cases'])}\n"
            
# # #         context += f"Match Score: {alg.get('similarity_score', 0):.2f}\n\n"

# # #     context += "IMPORTANT: When recommending an algorithm, you MUST cite it using its exact ID in this format: [ID: algorithm_id]\n\n"
    
# # #     return context


# # # def generate_response(query):
# # #     """Main function to process user queries and generate appropriate responses."""
# # #     # Step 1: Determine if this is an algorithm-related query
# # #     if is_algorithm_query(query):
# # #         # Step 2a: It's algorithm-related, use RAG system
# # #         print("Algorithm-related query detected.")
# # #         return generate_algorithm_recommendation(query)
# # #     else:
# # #         # Step 2b: It's a general question, use conversational response
# # #         print("General conversation query detected.")
# # #         return generate_conversation_response(query)


# # # def generate_algorithm_recommendation(query):
# # #     """Generate algorithm recommendation using LLM with RAG."""
   
# # #     # Retrieve relevant algorithms
# # #     algorithms = retrieve_algorithms(query)
    
# # #     if not algorithms:
# # #         return "Could not retrieve relevant algorithms. Please try a different query."
    
# # #     # Format context for LLM
# # #     context = format_algorithm_context(algorithms)

# # #     recommendation_prompt = f"""You are an algorithm specialist. Your top priority is to help guide users 
# # #     into selecting an appropriate algorithm for their problem.

# # #     {context}

# # #     User Problem: {query}

# # #     Based on the information above, which algorithm would be most suitable for solving this problem, and why? 
# # #     Explain your reasoning in detail, including time and space complexity analysis.
    
# # #     REMEMBER: You MUST cite the algorithm you recommend using its exact ID in the format [RAG_ID: algorithm_id]. 
# # #     This ID was provided in the context for each algorithm."""
    

# # #     try:
# # #         model = genai.GenerativeModel('models/gemini-1.5-pro')

# # #         response = model.generate_content(recommendation_prompt,
# # #                                           generation_config=genai.types.GenerationConfig(
# # #                                             temperature=0.3,
# # #                                             top_p=0.95,
# # #                                             top_k=40,
# # #                                             max_output_tokens=1000
# # #                                         ))
        
# # #         return response.text.strip()
    
# # #     except Exception as e:
# # #         print(f"Error calling Gemini: {str(e)}")
# # #         # Fallback response
# # #         if algorithms:
# # #             alg = algorithms[0]
# # #             return f"Based on the retrieved algorithms, {alg['name']} [RAG_ID: {alg.get('id', 'unknown')}] appears to be most suitable. {alg.get('description', '')}"
# # #         else:
# # #             return "Could not generate a recommendation. Please try again with a different query."
        
# # # #     # Generate system prompt
# # # #     system_prompt = """You are an algorithm specialist. Your top priority is to help guide users 
# # # #     into selecting an appropriate algorithm for their problem. 
    
# # # #     IMPROTANT INSTRUCTION: When recommending an answering, you MUST cite the exact algorithm ID provided in the context.
# # # #     Use the format [RAG_ID: algorithm_id] to cite the algorithm ID. Each algorithm in the context has a unique ID that starts with 'ID:'.

# # # #     Example citation: "I recommend using the Binary Search algorithm [RAG_ID: binary-search-12345] for this problem."
# # # # """
    
# # # #     # Generate user prompt
# # # #     user_prompt = f"""{context}
    
# # # # User Problem: {query}

# # # # Based on the information above, which algorithm would be most suitable for solving this problem, and why? 
# # # # Explain your reasoning in detail, including time and space complexity analysis.

# # # # REMEMBER: You MUST cite the algorithm you recommend using its exact ID in the format [RAG_ID: algorithm_id]. 
# # # # This ID was provided in the context for each algorithm.
# # # # """
    
# # #     # Call LLM
# # #     # client = OpenAI(
# # #     #     base_url="http://127.0.0.1:8080/v1",  # Your local LLM server
# # #     #     api_key="sk-no-key-required"
# # #     # )
    
# # #     # try:
# # #     #     response = client.chat.completions.create(
# # #     #         model="local-model",
# # #     #         messages=[
# # #     #             {"role": "system", "content": system_prompt},
# # #     #             {"role": "user", "content": user_prompt}
# # #     #         ],
# # #     #         temperature=0.3
# # #     #     )
        
# # #     #     return response.choices[0].message.content
        
# # #     # except Exception as e:
# # #     #     print(f"Error calling LLM: {str(e)}")
# # #     #     # Fallback response
# # #     #     return f"Based on the retrieved algorithms, {algorithms[0]['name']} appears to be most suitable. {algorithms[0].get('description', '')}"

# # # # if __name__ == "__main__":
# # # #     # Test the integration
# # # #     test_query = """You are given a 0-indexed array of n integers differences, which describes the differences between each pair of consecutive integers of a hidden sequence of length (n + 1). More formally, call the hidden sequence hidden, then we have that differences[i] = hidden[i + 1] - hidden[i].

# # # # You are further given two integers lower and upper that describe the inclusive range of values [lower, upper] that the hidden sequence can contain.

# # # #     For example, given differences = [1, -3, 4], lower = 1, upper = 6, the hidden sequence is a sequence of length 4 whose elements are in between 1 and 6 (inclusive).
# # # #         [3, 4, 1, 5] and [4, 5, 2, 6] are possible hidden sequences.
# # # #         [5, 6, 3, 7] is not possible since it contains an element greater than 6.
# # # #         [1, 2, 3, 4] is not possible since the differences are not correct. What algo can I use to test this?"""
# # # #     recommendation = generate_algorithm_recommendation(test_query)
# # # #     print(recommendation)

# # import os
# # import sys
# # import json
# # import requests
# # from typing import List, Dict, Any
# # from openai import OpenAI
# # from dotenv import load_dotenv
# # import google.generativeai as genai

# # # Adding parent directory to path for imports
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # load_dotenv()

# # # Configure Gemini API
# # GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# # genai.configure(api_key=GEMINI_API_KEY)

# # def is_algorithm_query(query):
# #     """Check if the query is related to algorithms or programming problems."""
# #     classification_prompt = f"""You are a query classifier specializing in algorithm and data structure problems

# #     CAREFULLY distinguish between these DIFFERENT TYPES of queries:
# #     A. PROBLEM-SOLVING QUERIES: Questions asking for an algorithm to solve a specific computational problem
# #     B. CONCEPTUAL QUERIES: General questions about what algorithms are or how they work
# #     C. APPLICATION QUERIES: Questions about applying algorithms in real-life contexts
# #     D. NON-ALGORITHM QUERIES: Questions unrelated to algorithms or data structures

# #     A query is considered a PROBLEM-SOLVING query (Type A) if it meets ANY of these criteria:
# #     1. Asks for an algorithm to solve a specific computational task
# #     2. Describes a programming problem that needs a solution
# #     3. Contains LeetCode-style problem descriptions
# #     4. Asks about the complexity or efficiency of solving a specific problem
# #     5. Requests code or pseudocode to solve a computational challenge

# #     EXAMPLES:
# #     - "How do I find the shortest path in a graph?" → A (Problem-solving query)
# #     - "What is dynamic programming?" → B (Conceptual query)
# #     - "How can I apply dynamic programming in real life?" → C (Application query)
# #     - "What's the weather like today?" → D (Non-algorithm query)
# #     - "Given an array of integers, find two numbers that add up to a target" → A (Problem-solving query)
# #     - "How does the merge sort algorithm work?" → B (Conceptual query)
# #     - "When should I use BFS vs DFS in robotics?" → C (Application query)

# # Query: {query}

# # Based on the criteria above, classify this query as either:
# # A (Problem-solving query)
# # B (Conceptual query)
# # C (Application query) 
# # D (Non-algorithm query)

# # Return ONLY the single letter classification without explanation."""

# #     try:
# #         model = genai.GenerativeModel('models/gemini-2.0-flash-lite-001')

# #         response = model.generate_content(classification_prompt,
# #                                         generation_config=genai.types.GenerationConfig(
# #                                             temperature=0.1,
# #                                             top_p=0.95,
# #                                             top_k=40,
# #                                             max_output_tokens=5
# #                                         ))
        
# #         result = response.text.strip().upper()

# #         print(f"CLASSIFICATION RESULT: Query '{query[:50]}...' classified as: {result}")
        
# #         # For problem-solving or conceptual queries, use the RAG system
# #         return result in ["A", "B"]
    
# #     except Exception as e:
# #         print(f"Error in classification: {str(e)}")
# #         # If classification fails, default to treating it as an algorithm query
# #         return True
    
# # def generate_conversation_response(query):
# #     """Generate a conversational response for non-algorithm queries."""
# #     conversation_prompt = f"""You are a helpful assistant named Algorithm Assistant. 
# #     The user is asking a general question, not related to algorithms.
    
# #     User: {query}
    
# #     Provide a helpful, conversational response. Be brief and friendly."""

# #     try:
# #         model = genai.GenerativeModel('models/gemini-2.0-flash-001')

# #         response = model.generate_content(conversation_prompt,
# #                                         generation_config=genai.types.GenerationConfig(
# #                                             temperature=0.7,
# #                                             top_p=0.95,
# #                                             top_k=40,
# #                                             max_output_tokens=200
# #                                         ))
# #         return response.text.strip()
    
# #     except Exception as e:
# #         print(f"Error generating conversation response: {str(e)}")
# #         return "I'm sorry, I encountered an error trying to respond to your message. How can I help you with algorithm-related questions instead?"

# # def retrieve_algorithms(query, rag_endpoint="http://localhost:8000/api/query", top_k=5):
# #     """Retrieve relevant algorithms from RAG service with enhanced parameters."""
# #     try:
# #         # Use enhanced API with hybrid search and configurable alpha
# #         response = requests.post(
# #             rag_endpoint,
# #             json={
# #                 "query": query, 
# #                 "top_k": top_k,
# #                 "hybrid_search": True,
# #                 "alpha": 0.6  # Balance between semantic and keyword matching
# #             }
# #         )
        
# #         if response.status_code == 200:
# #             response_data = response.json()
            
# #             # Check if an enhanced query was used and log it
# #             if response_data.get("enhanced_query"):
# #                 print(f"Enhanced query used: {response_data['enhanced_query']}")
                
# #             return response_data["results"]
# #         else:
# #             print(f"Error from RAG service: {response.text}")
# #             return []
            
# #     except Exception as e:
# #         print(f"Error retrieving algorithms: {str(e)}")
# #         return []

# # def format_algorithm_context(algorithms):
# #     """Format algorithm information for LLM context with enhanced details."""
# #     context = "Based on the problem description, here are relevant algorithms:\n\n"
    
# #     for i, alg in enumerate(algorithms):
# #         alg_id = alg.get('id', f"unknown-{i}")
# #         context += f"Algorithm {i+1}: [ID: {alg_id}]: {alg['name']}\n"
        
# #         if alg.get('description'):
# #             context += f"Description: {alg['description']}\n"
            
# #         if alg.get('tags'):
# #             context += f"Tags: {', '.join(alg['tags'])}\n"
            
# #         if alg.get('complexity'):
# #             complexity = alg['complexity']
# #             time_complexity = complexity.get('time', 'Not specified')
# #             space_complexity = complexity.get('space', 'Not specified')
# #             context += f"Complexity: Time - {time_complexity}, Space - {space_complexity}\n"
            
# #         if alg.get('use_cases'):
# #             context += f"Use Cases: {', '.join(alg['use_cases'])}\n"
        
# #         # Include match details if available
# #         if alg.get('match_details'):
# #             context += f"Match Context: {alg['match_details']}\n"
            
# #         # Use similarity_score directly if available
# #         score = alg.get('similarity_score', 0)
# #         context += f"Match Score: {score:.2f}\n\n"

# #     context += "IMPORTANT: When recommending an algorithm, you MUST cite it using its exact ID in this format: [ID: algorithm_id]\n\n"
    
# #     return context

# # def generate_response(query):
# #     """Main function to process user queries and generate appropriate responses."""
# #     # Step 1: Determine if this is an algorithm-related query
# #     if is_algorithm_query(query):
# #         # Step 2a: It's algorithm-related, use RAG system
# #         print("Algorithm-related query detected.")
# #         return generate_algorithm_recommendation(query)
# #     else:
# #         # Step 2b: It's a general question, use conversational response
# #         print("General conversation query detected.")
# #         return generate_conversation_response(query)

# # def generate_algorithm_recommendation(query):
# #     """Generate algorithm recommendation using LLM with enhanced RAG."""
    
# #     # Check for specific algorithm-related keywords in the query
# #     query_lower = query.lower()
# #     rag_techniques = []
    
# #     # Special checks for common algorithm problems
# #     if "substring" in query_lower and any(term in query_lower for term in ["longest", "without duplicate", "without repeating"]):
# #         print("Detected substring without duplicates problem - likely sliding window")
# #         rag_techniques.append("This query appears to involve finding patterns in a substring which often uses sliding window technique.")
    
# #     if "array" in query_lower and "sum" in query_lower and any(term in query_lower for term in ["maximum", "minimum", "subarray"]):
# #         print("Detected subarray sum problem - likely dynamic programming or sliding window")
# #         rag_techniques.append("This query appears to involve subarray sums which often use dynamic programming or sliding window.")
    
# #     # Retrieve relevant algorithms
# #     algorithms = retrieve_algorithms(query)
    
# #     if not algorithms:
# #         return "Could not retrieve relevant algorithms. Please try a different query."
    
# #     # Format context for LLM
# #     context = format_algorithm_context(algorithms)
    
# #     # Enhance prompt with detected techniques if any
# #     technique_hints = ""
# #     if rag_techniques:
# #         technique_hints = "Based on analysis of your query:\n" + "\n".join(rag_techniques) + "\n\n"

# #     recommendation_prompt = f"""You are an algorithm specialist. Your top priority is to help guide users 
# #     into selecting an appropriate algorithm for their problem.

# #     {technique_hints}{context}

# #     User Problem: {query}

# #     Based on the information above, which algorithm would be most suitable for solving this problem, and why? 
# #     Explain your reasoning in detail, including time and space complexity analysis.
    
# #     REMEMBER: You MUST cite the algorithm you recommend using its exact ID in the format [ID: algorithm_id]. 
# #     This ID was provided in the context for each algorithm."""

# #     try:
# #         model = genai.GenerativeModel('models/gemini-1.5-pro')

# #         response = model.generate_content(recommendation_prompt,
# #                                         generation_config=genai.types.GenerationConfig(
# #                                             temperature=0.3,
# #                                             top_p=0.95,
# #                                             top_k=40,
# #                                             max_output_tokens=1000
# #                                         ))
        
# #         return response.text.strip()
    
# #     except Exception as e:
# #         print(f"Error calling Gemini: {str(e)}")
# #         # Fallback response
# #         if algorithms:
# #             alg = algorithms[0]
# #             return f"Based on the retrieved algorithms, {alg['name']} [ID: {alg.get('id', 'unknown')}] appears to be most suitable. {alg.get('description', '')}"
# #         else:
# #             return "Could not generate a recommendation. Please try again with a different query."

# # # For testing
# # if __name__ == "__main__":
# #     test_query = "Given a string, find the length of the longest substring without repeating characters."
# #     result = generate_response(test_query)
# #     print("\nFINAL RESPONSE:")
# #     print(result)

# import os
# import sys
# import json
# import requests
# from typing import List, Dict, Any
# from openai import OpenAI
# from dotenv import load_dotenv
# import google.generativeai as genai

# # Add parent directory to path for imports
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# load_dotenv()

# # Configure Gemini API
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# genai.configure(api_key=GEMINI_API_KEY)

# def is_algorithm_query(query):
#     """Check if the query is related to algorithms or programming problems."""
#     classification_prompt = f"""You are a query classifier specializing in algorithm and data structure problems

#     CAREFULLY distinguish between these DIFFERENT TYPES of queries:
#     A. PROBLEM-SOLVING QUERIES: Questions asking for an algorithm to solve a specific computational problem
#     B. CONCEPTUAL QUERIES: General questions about what algorithms are or how they work
#     C. APPLICATION QUERIES: Questions about applying algorithms in real-life contexts
#     D. NON-ALGORITHM QUERIES: Questions unrelated to algorithms or data structures

#     A query is considered a PROBLEM-SOLVING query (Type A) if it meets ANY of these criteria:
#     1. Asks for an algorithm to solve a specific computational task
#     2. Describes a programming problem that needs a solution
#     3. Contains LeetCode-style problem descriptions
#     4. Asks about the complexity or efficiency of solving a specific problem
#     5. Requests code or pseudocode to solve a computational challenge

#     EXAMPLES:
#     - "How do I find the shortest path in a graph?" → A (Problem-solving query)
#     - "What is dynamic programming?" → B (Conceptual query)
#     - "How can I apply dynamic programming in real life?" → C (Application query)
#     - "What's the weather like today?" → D (Non-algorithm query)
#     - "Given an array of integers, find two numbers that add up to a target" → A (Problem-solving query)
#     - "How does the merge sort algorithm work?" → B (Conceptual query)
#     - "When should I use BFS vs DFS in robotics?" → C (Application query)

# Query: {query}

# Based on the criteria above, classify this query as either:
# A (Problem-solving query)
# B (Conceptual query)
# C (Application query) 
# D (Non-algorithm query)

# Return ONLY the single letter classification without explanation."""

#     try:
#         model = genai.GenerativeModel('models/gemini-2.0-flash-lite-001')

#         response = model.generate_content(classification_prompt,
#                                         generation_config=genai.types.GenerationConfig(
#                                             temperature=0.1,
#                                             top_p=0.95,
#                                             top_k=40,
#                                             max_output_tokens=5
#                                         ))
        
#         result = response.text.strip().upper()

#         print(f"CLASSIFICATION RESULT: Query '{query[:50]}...' classified as: {result}")
        
#         # For problem-solving or conceptual queries, use the RAG system
#         return result in ["A", "B"]
    
#     except Exception as e:
#         print(f"Error in classification: {str(e)}")
#         # If classification fails, default to treating it as an algorithm query
#         return True

# def generate_conversation_response(query):
#     """Generate a conversational response for non-algorithm queries."""
#     conversation_prompt = f"""You are a helpful assistant named Algorithm Assistant. 
#     The user is asking a general question, not related to algorithms.
    
#     User: {query}
    
#     Provide a helpful, conversational response. Be brief and friendly."""

#     try:
#         model = genai.GenerativeModel('models/gemini-2.0-flash-001')

#         response = model.generate_content(conversation_prompt,
#                                         generation_config=genai.types.GenerationConfig(
#                                             temperature=0.7,
#                                             top_p=0.95,
#                                             top_k=40,
#                                             max_output_tokens=200
#                                         ))
#         return response.text.strip()
    
#     except Exception as e:
#         print(f"Error generating conversation response: {str(e)}")
#         return "I'm sorry, I encountered an error trying to respond to your message. How can I help you with algorithm-related questions instead?"

# def retrieve_algorithms(query, rag_endpoint="http://localhost:8000/api/query", top_k=5, params=None):
#     """Retrieve relevant algorithms from RAG service with enhanced parameters."""
#     try:
#         # Build request payload with defaults + any additional params
#         payload = {
#             "query": query, 
#             "top_k": top_k,
#             "hybrid_search": True,
#             "alpha": 0.6  # Balance between semantic and keyword matching
#         }
        
#         # Add any additional parameters
#         if params:
#             payload.update(params)
        
#         # Make the request
#         response = requests.post(rag_endpoint, json=payload)
        
#         if response.status_code == 200:
#             response_data = response.json()
            
#             # Check if an enhanced query was used and log it
#             if response_data.get("enhanced_query"):
#                 print(f"Enhanced query used: {response_data['enhanced_query']}")
                
#             return response_data["results"]
#         else:
#             print(f"Error from RAG service: {response.text}")
#             return []
            
#     except Exception as e:
#         print(f"Error retrieving algorithms: {str(e)}")
#         return []

# def extract_general_concepts(query):
#     """Extract general algorithmic concepts from a specific problem query."""
#     query_lower = query.lower()
    
#     general_concepts = []
    
#     # Map specific problem patterns to general techniques
#     concept_mapping = {
#         "sum": ["two pointers", "hash table", "sorting"],
#         "substring": ["sliding window", "two pointers", "string algorithms"],
#         "path": ["graph algorithms", "dfs", "bfs", "shortest path"],
#         "tree": ["tree traversal", "binary tree", "dfs", "bfs"],
#         "array": ["array manipulation", "sorting", "two pointers"],
#         "parenthesis": ["stack", "backtracking", "dynamic programming"],
#         "palindrome": ["two pointers", "string algorithms", "dynamic programming"],
#         "subsequence": ["dynamic programming", "two pointers", "greedy"],
#         "permutation": ["backtracking", "recursion"],
#         "combination": ["backtracking", "recursion", "dynamic programming"],
#         "sort": ["sorting algorithms", "quicksort", "mergesort"],
#         "grid": ["matrix algorithms", "dfs", "bfs", "dynamic programming"],
#         "interval": ["greedy", "sorting", "interval algorithms"],
#         "binary search": ["binary search", "divide and conquer"],
#         "maximum": ["greedy", "dynamic programming"],
#         "minimum": ["greedy", "dynamic programming"],
#     }
    
#     # Add pattern-specific techniques
#     for pattern, techniques in concept_mapping.items():
#         if pattern in query_lower:
#             general_concepts.extend(techniques)
    
#     # Add common algorithm primitives that are often composed together
#     common_primitives = ["sorting", "hash table", "two pointers", "dynamic programming"]
#     general_concepts.extend(common_primitives)
    
#     # Remove duplicates and join concepts
#     general_concepts = list(set(general_concepts))
#     return f"general algorithm techniques: {', '.join(general_concepts)}"

# def retrieve_techniques_and_primitives(query, rag_endpoint="http://localhost:8000/api/query"):
#     """
#     Retrieve both specific algorithms and general techniques to support
#     algorithm composition for problems without exact matches.
#     """
#     # First, retrieve specific algorithm matches (if any exist)
#     specific_results = retrieve_algorithms(
#         query, 
#         rag_endpoint=rag_endpoint, 
#         top_k=3,
#         params={"hybrid_search": True, "alpha": 0.7}
#     )
    
#     # Then retrieve general algorithmic techniques and primitives
#     # Use more general query derived from the original
#     general_query = extract_general_concepts(query)
#     general_results = retrieve_algorithms(
#         general_query,
#         rag_endpoint=rag_endpoint,
#         top_k=5,
#         params={"hybrid_search": True, "alpha": 0.5}  # More weight on keywords for techniques
#     )
    
#     # Combine results, ensuring we have both specific and general techniques
#     all_results = []
#     seen_ids = set()
    
#     # Add specific results first
#     for result in specific_results:
#         result_id = result.get('id')
#         if result_id not in seen_ids:
#             seen_ids.add(result_id)
#             all_results.append(result)
    
#     # Then add general techniques that aren't duplicates
#     for result in general_results:
#         result_id = result.get('id')
#         if result_id not in seen_ids:
#             seen_ids.add(result_id)
#             all_results.append(result)
    
#     return {
#         "specific_results": specific_results,
#         "general_results": general_results,
#         "all_results": all_results
#     }

# def format_algorithm_context(algorithms):
#     """Format algorithm information for LLM context."""
#     context = "Based on the problem description, here are relevant algorithms:\n\n"
    
#     for i, alg in enumerate(algorithms):
#         alg_id = alg.get('id', f"unknown-{i}")
#         context += f"Algorithm {i+1}: [ID: {alg_id}]: {alg['name']}\n"
        
#         if alg.get('description'):
#             context += f"Description: {alg['description']}\n"
            
#         if alg.get('tags'):
#             context += f"Tags: {', '.join(alg['tags'])}\n"
            
#         if alg.get('complexity'):
#             complexity = alg['complexity']
#             time_complexity = complexity.get('time', 'Not specified')
#             space_complexity = complexity.get('space', 'Not specified')
#             context += f"Complexity: Time - {time_complexity}, Space - {space_complexity}\n"
            
#         if alg.get('use_cases'):
#             context += f"Use Cases: {', '.join(alg['use_cases'])}\n"
            
#         context += f"Match Score: {alg.get('similarity_score', 0):.2f}\n\n"

#     context += "IMPORTANT: When recommending an algorithm, you MUST cite it using its exact ID in this format: [ID: algorithm_id]\n\n"
    
#     return context

# def format_algorithm_composition_prompt(query, results_data):
#     """
#     Format prompt for LLM to compose algorithms from retrieved techniques.
#     """
#     prompt = f"""You are an algorithm composition specialist. Your task is to recommend and COMPOSE a solution 
# for a specific algorithmic problem that might not have an exact match in our database.

# USER PROBLEM: {query}

# AVAILABLE ALGORITHMS AND TECHNIQUES:
# """
    
#     # Add specific algorithms if available
#     if results_data["specific_results"]:
#         prompt += "\n## SPECIFIC ALGORITHMS THAT MIGHT BE RELEVANT:\n"
#         for i, alg in enumerate(results_data["specific_results"]):
#             alg_id = alg.get('id', f"unknown-{i}")
#             prompt += f"Algorithm {i+1}: [ID: {alg_id}]: {alg.get('name', '')}\n"
            
#             if alg.get('description'):
#                 prompt += f"Description: {alg.get('description', '')}\n"
                
#             if alg.get('tags'):
#                 prompt += f"Tags: {', '.join(alg.get('tags', []))}\n"
                
#             if alg.get('complexity') and isinstance(alg.get('complexity'), dict):
#                 complexity = alg.get('complexity')
#                 time_complexity = complexity.get('time', 'Not specified')
#                 space_complexity = complexity.get('space', 'Not specified')
#                 prompt += f"Complexity: Time - {time_complexity}, Space - {space_complexity}\n"
            
#             prompt += f"Match Score: {alg.get('similarity_score', 0):.2f}\n\n"
    
#     # Add general techniques
#     prompt += "\n## GENERAL TECHNIQUES THAT CAN BE COMBINED:\n"
#     for i, alg in enumerate(results_data["general_results"]):
#         alg_id = alg.get('id', f"unknown-{i}")
#         prompt += f"Technique {i+1}: [ID: {alg_id}]: {alg.get('name', '')}\n"
        
#         if alg.get('description'):
#             description = alg.get('description', '')
#             # Truncate long descriptions
#             if len(description) > 200:
#                 description = description[:200] + "..."
#             prompt += f"Description: {description}\n"
            
#         if alg.get('tags'):
#             prompt += f"Tags: {', '.join(alg.get('tags', []))}\n"
        
#         prompt += "\n"
    
#     prompt += """
# ## YOUR TASK:
# 1. Analyze the user's problem carefully.
# 2. Draw from both specific algorithms and general techniques listed above.
# 3. COMPOSE a comprehensive algorithm solution tailored to this specific problem.
# 4. Explain the approach step by step, including time and space complexity analysis.
# 5. IMPORTANT: Cite ALL algorithms/techniques you use with their IDs in this format: [ID: algorithm_id]

# Your solution should be DETAILED and SPECIFIC to this problem, even if you need to combine multiple techniques.
# If the specific algorithms don't perfectly match, explain how to adapt or combine them.

# ## RESPONSE FORMAT:
# - Recommended Approach: [Brief overview of your composed solution]
# - Algorithm Composition: [Detailed explanation of how to combine/adapt techniques]
# - Implementation Steps: [Step-by-step walkthrough]
# - Time & Space Complexity: [Analysis of your composed solution]
# - Key Techniques Used: [List techniques with their IDs]
# """
    
#     return prompt

# def assess_match_quality(algorithms, query):
#     """
#     Assess whether the retrieved algorithms match the query well enough
#     or if we need to compose algorithms.
#     """
#     # If no algorithms found, we need to compose
#     if not algorithms:
#         return "compose"
    
#     # Check if we have a strong match (high similarity score) for specific problems
#     top_score = algorithms[0].get('similarity_score', 0)
    
#     # Look for specific problem indicators that might need composition
#     query_lower = query.lower()
#     specific_problems = [
#         "4 sum", "k sum", "four sum", 
#         "longest palindromic subsequence",
#         "k nearest neighbors",
#         "serialize and deserialize binary tree",
#         "word ladder",
#         "sudoku solver",
#         "longest increasing subsequence",
#         "lru cache",
#         "implement trie"
#     ]
    
#     # If there's a specific problem indicator and the match isn't strong, compose
#     for problem in specific_problems:
#         if problem in query_lower and top_score < 0.85:
#             return "compose"
    
#     # If the top match score is below threshold, compose
#     if top_score < 0.7:
#         return "compose"
    
#     # Otherwise, use standard recommendation
#     return "standard"

# def generate_algorithm_recommendation(query):
#     """Generate algorithm recommendation using LLM with RAG and composition if needed."""
    
#     # Retrieve algorithms
#     algorithms = retrieve_algorithms(query)
    
#     # Assess match quality
#     match_quality = assess_match_quality(algorithms, query)
    
#     if match_quality == "compose":
#         print("No exact match found, using algorithm composition")
#         return generate_algorithm_composition(query)
#     else:
#         print("Found good matches, using standard recommendation")
        
#         if not algorithms:
#             return "Could not retrieve relevant algorithms. Please try a different query."
        
#         # Format context for LLM
#         context = format_algorithm_context(algorithms)
    
#         recommendation_prompt = f"""You are an algorithm specialist. Your top priority is to help guide users 
#         into selecting an appropriate algorithm for their problem.
    
#         {context}
    
#         User Problem: {query}
    
#         Based on the information above, which algorithm would be most suitable for solving this problem, and why? 
#         Explain your reasoning in detail, including time and space complexity analysis.
        
#         REMEMBER: You MUST cite the algorithm you recommend using its exact ID in the format [ID: algorithm_id]. 
#         This ID was provided in the context for each algorithm."""
    
#         try:
#             model = genai.GenerativeModel('models/gemini-1.5-pro')
    
#             response = model.generate_content(recommendation_prompt,
#                                             generation_config=genai.types.GenerationConfig(
#                                                 temperature=0.3,
#                                                 top_p=0.95,
#                                                 top_k=40,
#                                                 max_output_tokens=1000
#                                             ))
            
#             return response.text.strip()
        
#         except Exception as e:
#             print(f"Error calling Gemini: {str(e)}")
#             # Fallback response
#             if algorithms:
#                 alg = algorithms[0]
#                 return f"Based on the retrieved algorithms, {alg['name']} [ID: {alg.get('id', 'unknown')}] appears to be most suitable. {alg.get('description', '')}"
#             else:
#                 return "Could not generate a recommendation. Please try again with a different query."

# def generate_algorithm_composition(query):
#     """Generate composed algorithm recommendation using LLM and RAG."""
    
#     # Retrieve both specific algorithms and general techniques
#     results_data = retrieve_techniques_and_primitives(query)
    
#     if not results_data["all_results"]:
#         return "Could not retrieve relevant algorithms or techniques. Please try a different query."
    
#     # Format prompt for LLM to compose algorithms
#     prompt = format_algorithm_composition_prompt(query, results_data)

#     try:
#         model = genai.GenerativeModel('models/gemini-1.5-pro')

#         response = model.generate_content(prompt,
#                                         generation_config=genai.types.GenerationConfig(
#                                             temperature=0.3,
#                                             top_p=0.95,
#                                             top_k=40,
#                                             max_output_tokens=1500
#                                         ))
        
#         return response.text.strip()
    
#     except Exception as e:
#         print(f"Error calling Gemini: {str(e)}")
#         # Fallback response
#         if results_data["specific_results"]:
#             alg = results_data["specific_results"][0]
#             return f"Based on the available algorithms, you can adapt {alg['name']} [ID: {alg.get('id', 'unknown')}] for this problem. {alg.get('description', '')}"
#         else:
#             return "Could not generate a recommendation. Please try again with a different query."

# def generate_response(query):
#     """Main function to process user queries and generate appropriate responses."""
#     # Step 1: Determine if this is an algorithm-related query
#     if is_algorithm_query(query):
#         # Step 2a: It's algorithm-related, use RAG system
#         print("Algorithm-related query detected.")
#         return generate_algorithm_recommendation(query)
#     else:
#         # Step 2b: It's a general question, use conversational response
#         print("General conversation query detected.")
#         return generate_conversation_response(query)

# # Example usage
# if __name__ == "__main__":
#     test_query = "Given an array of integers, find four elements that sum to a given target value."
#     result = generate_response(test_query)
#     print("\nFINAL RESPONSE:")
#     print(result)

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

        print(f"CLASSIFICATION RESULT: Query '{query[:50]}...' classified as: {result}")
        
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
    """Ensure that core algorithms like two pointers have entries even if not in database."""
    # List of core algorithms that should always be available
    core_algorithms = {
        "two_pointers": {
            "id": "two-pointers-technique-default",
            "name": "Two Pointers Technique",
            "description": "A technique that uses two pointers to iterate through a data structure. Often used for problems involving arrays, linked lists, or strings to find pairs that satisfy certain conditions.",
            "tags": ["array", "linked list", "string", "optimization"],
            "complexity": {"time": "O(n)", "space": "O(1)"}
        },
        "sliding_window": {
            "id": "sliding-window-technique-default",
            "name": "Sliding Window Technique",
            "description": "An efficient technique for processing contiguous sequences in arrays or strings. It maintains a window of elements and slides it through the data to find optimal solutions.",
            "tags": ["array", "string", "optimization"],
            "complexity": {"time": "O(n)", "space": "O(1)"}
        },
        "hash_table": {
            "id": "hash-table-default",
            "name": "Hash Table",
            "description": "A data structure that provides efficient insertion, deletion, and lookup operations. Used for quick access to elements based on keys.",
            "tags": ["lookup", "optimization", "data structure"],
            "complexity": {"time": "O(1) average", "space": "O(n)"}
        },
        "dynamic_programming": {
            "id": "dynamic-programming-default",
            "name": "Dynamic Programming",
            "description": "A method for solving complex problems by breaking them down into simpler subproblems. It stores the results of subproblems to avoid redundant calculations.",
            "tags": ["optimization", "recursion", "memoization"],
            "complexity": {"time": "Problem dependent", "space": "Usually O(n) to O(n²)"}
        },
        "binary_search": {
            "id": "binary-search-default",
            "name": "Binary Search",
            "description": "A divide-and-conquer algorithm for finding a target value in a sorted array by repeatedly dividing the search interval in half.",
            "tags": ["search", "divide and conquer", "sorted array"],
            "complexity": {"time": "O(log n)", "space": "O(1)"}
        },
        "sorting": {
            "id": "sorting-algorithms-default",
            "name": "Sorting Algorithms",
            "description": "Algorithms for arranging elements in a specific order, such as numerical or lexicographical. Common examples include Quick Sort, Merge Sort, and Heap Sort.",
            "tags": ["array", "comparison", "ordering"],
            "complexity": {"time": "O(n log n) for comparison-based sorts", "space": "O(1) to O(n)"}
        },
        "graph_traversal": {
            "id": "graph-traversal-default",
            "name": "Graph Traversal Algorithms",
            "description": "Algorithms for visiting all nodes in a graph, such as Depth-First Search (DFS) and Breadth-First Search (BFS).",
            "tags": ["graph", "tree", "search"],
            "complexity": {"time": "O(V + E)", "space": "O(V)"}
        }
    }
    
    # Check which core algorithms we need based on the query
    query_lower = query.lower()
    needed_algorithms = []
    
    # Check for specific algorithm needs
    if "two pointers" in query_lower or any(term in query_lower for term in ["pair", "sum", "palindrome"]):
        needed_algorithms.append("two_pointers")
    
    if "sliding window" in query_lower or any(term in query_lower for term in ["substring", "subarray", "contiguous"]):
        needed_algorithms.append("sliding_window")
    
    if "hash" in query_lower or any(term in query_lower for term in ["lookup", "map", "dictionary", "frequency", "count"]):
        needed_algorithms.append("hash_table")

    if "dynamic programming" in query_lower or any(term in query_lower for term in ["dp", "subproblem", "optimization", "maximize", "minimize"]):
        needed_algorithms.append("dynamic_programming")
    
    if "binary search" in query_lower or any(term in query_lower for term in ["sorted array", "log n", "find element"]):
        needed_algorithms.append("binary_search")
    
    if "sort" in query_lower:
        needed_algorithms.append("sorting")
    
    if any(term in query_lower for term in ["graph", "tree", "network", "traversal", "dfs", "bfs"]):
        needed_algorithms.append("graph_traversal")
    
    # Add general techniques for all problem-solving queries
    if is_problem_solving_query(query):
        # Add some core primitives as defaults for problem-solving
        if len(needed_algorithms) == 0:
            needed_algorithms.extend(["hash_table", "two_pointers", "dynamic_programming"])
    
    # Check if we already have these algorithms in our results
    existing_ids = [alg.get("id", "").lower() for alg in algorithms]
    existing_names = [alg.get("name", "").lower() for alg in algorithms]
    
    # Add missing core algorithms to results
    added_algorithms = []
    for algo_key in needed_algorithms:
        # Check if we already have this algorithm by ID or name
        if (core_algorithms[algo_key]["id"].lower() not in existing_ids and 
            not any(core_algorithms[algo_key]["name"].lower() in name for name in existing_names)):
            # Add synthetic match score based on relevance
            algo_copy = core_algorithms[algo_key].copy()
            algo_copy["similarity_score"] = 0.85  # High enough to be considered
            added_algorithms.append(algo_copy)
    
    # Return combined list
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
            "similarity_score": 0.95  # Very high score to prioritize
        }
    elif "3 sum" in query_lower or "three sum" in query_lower:
        return {
            "id": "three-sum-algorithm-composed",
            "name": "Three Sum Algorithm",
            "description": "A technique to find all unique triplets in an array that sum up to a target value. Typically uses sorting with two pointers approach.",
            "tags": ["array", "two pointers", "sorting"],
            "complexity": {"time": "O(n²)", "space": "O(1) to O(n)"},
            "similarity_score": 0.95  # Very high score to prioritize
        }
    elif "2 sum" in query_lower or "two sum" in query_lower:
        return {
            "id": "two-sum-algorithm-composed",
            "name": "Two Sum Algorithm",
            "description": "A technique to find pairs in an array that sum up to a target value. Most efficiently done using a hash table.",
            "tags": ["array", "hash table"],
            "complexity": {"time": "O(n)", "space": "O(n)"},
            "similarity_score": 0.95  # Very high score to prioritize
        }
    elif "k sum" in query_lower:
        return {
            "id": "k-sum-algorithm-composed",
            "name": "K Sum Algorithm",
            "description": "A general technique to find k elements in an array that sum up to a target value. Typically uses recursion to reduce to simpler cases.",
            "tags": ["array", "two pointers", "recursion", "hash table"],
            "complexity": {"time": "O(nᵏ⁻¹) or better with optimizations", "space": "O(n) to O(nᵏ⁻¹)"},
            "similarity_score": 0.95  # Very high score to prioritize
        }
    return None

def retrieve_from_rag(query, rag_endpoint="http://localhost:8000/api/query", top_k=5, params=None):
    """Basic retrieval from RAG service without enhancements."""
    try:
        # Build request payload with defaults + any additional params
        payload = {
            "query": query, 
            "top_k": top_k,
            "hybrid_search": True,
            "alpha": 0.6  # Balance between semantic and keyword matching
        }
        
        # Add any additional parameters
        if params:
            payload.update(params)
        
        # Make the request
        response = requests.post(
            rag_endpoint,
            json=payload
        )
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Check if an enhanced query was used and log it
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
    """Retrieve relevant algorithms with enhanced handling for core algorithms."""
    
    # Check for special cases first
    special_case = handle_sum_problem(query)
    
    # Standard retrieval from RAG
    results = retrieve_from_rag(query, rag_endpoint, top_k, params)
    
    # Add special case if applicable
    if special_case:
        # Ensure the special case is included and at the top
        if not any(result.get("id") == special_case["id"] for result in results):
            results.insert(0, special_case)
    
    # Ensure core algorithms are included
    enhanced_results = ensure_core_algorithms_exist(results, query)
    
    return enhanced_results

def extract_general_concepts(query):
    """Extract general algorithmic concepts from a specific problem query."""
    query_lower = query.lower()
    
    general_concepts = []
    
    # Map specific problem patterns to general techniques
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
    
    # Add pattern-specific techniques
    for pattern, techniques in concept_mapping.items():
        if pattern in query_lower:
            general_concepts.extend(techniques)
    
    # Add common algorithm primitives that are often composed together
    common_primitives = ["sorting", "hash table", "two pointers", "dynamic programming"]
    general_concepts.extend(common_primitives)
    
    # Remove duplicates and join concepts
    general_concepts = list(set(general_concepts))
    return f"general algorithm techniques: {', '.join(general_concepts)}"

def retrieve_techniques_and_primitives(query, rag_endpoint="http://localhost:8000/api/query"):
    """
    Retrieve both specific algorithms and general techniques to support
    algorithm composition for problems without exact matches.
    """
    # First, retrieve specific algorithm matches (if any exist)
    specific_results = retrieve_algorithms(
        query, 
        rag_endpoint=rag_endpoint, 
        top_k=3,
        params={"hybrid_search": True, "alpha": 0.7}
    )
    
    # Then retrieve general algorithmic techniques and primitives
    # Use more general query derived from the original
    general_query = extract_general_concepts(query)
    general_results = retrieve_algorithms(
        general_query,
        rag_endpoint=rag_endpoint,
        top_k=5,
        params={"hybrid_search": True, "alpha": 0.5}  # More weight on keywords for techniques
    )
    
    # Combine results, ensuring we have both specific and general techniques
    all_results = []
    seen_ids = set()
    
    # Add specific results first
    for result in specific_results:
        result_id = result.get('id')
        if result_id and result_id not in seen_ids:
            seen_ids.add(result_id)
            all_results.append(result)
    
    # Then add general techniques that aren't duplicates
    for result in general_results:
        result_id = result.get('id')
        if result_id and result_id not in seen_ids:
            seen_ids.add(result_id)
            all_results.append(result)
    
    # Ensure we have any needed core algorithms
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
    
    # Add specific algorithms if available
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
    
    # Add general techniques
    prompt += "\n## GENERAL TECHNIQUES THAT CAN BE COMBINED:\n"
    for i, alg in enumerate(results_data["general_results"]):
        alg_id = alg.get('id', f"unknown-{i}")
        prompt += f"Technique {i+1}: [ID: {alg_id}]: {alg.get('name', '')}\n"
        
        if alg.get('description'):
            description = alg.get('description', '')
            # Truncate long descriptions
            if len(description) > 200:
                description = description[:200] + "..."
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
    # If no algorithms found, we need to compose
    if not algorithms:
        return "compose"
    
    # Check if we have a strong match (high similarity score) for specific problems
    top_score = algorithms[0].get('similarity_score', 0) if algorithms else 0
    
    # Look for specific problem indicators that might need composition
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
    
    # If there's a specific problem indicator and the match isn't strong, compose
    for problem in specific_problems:
        if problem in query_lower and top_score < 0.85:
            return "compose"
    
    # If the top match score is below threshold, compose
    if top_score < 0.7:
        return "compose"
    
    # Otherwise, use standard recommendation
    return "standard"

def generate_algorithm_recommendation(query):
    """Generate algorithm recommendation using LLM with RAG and composition if needed."""
    
    # Retrieve algorithms
    algorithms = retrieve_algorithms(query)
    
    # Assess match quality
    match_quality = assess_match_quality(algorithms, query)
    
    if match_quality == "compose":
        print("No exact match found, using algorithm composition")
        return generate_algorithm_composition(query)
    else:
        print("Found good matches, using standard recommendation")
        
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
            # Fallback response
            if algorithms:
                alg = algorithms[0]
                return f"Based on the retrieved algorithms, {alg['name']} [ID: {alg.get('id', 'unknown')}] appears to be most suitable. {alg.get('description', '')}"
            else:
                return "Could not generate a recommendation. Please try again with a different query."

def generate_algorithm_composition(query):
    """Generate composed algorithm recommendation using LLM and RAG."""
    
    # Retrieve both specific algorithms and general techniques
    results_data = retrieve_techniques_and_primitives(query)
    
    if not results_data["all_results"]:
        return "Could not retrieve relevant algorithms or techniques. Please try a different query."
    
    # Format prompt for LLM to compose algorithms
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
        # Fallback response
        if results_data["specific_results"]:
            alg = results_data["specific_results"][0]
            return f"Based on the available algorithms, you can adapt {alg['name']} [ID: {alg.get('id', 'unknown')}] for this problem. {alg.get('description', '')}"
        else:
            return "Could not generate a recommendation. Please try again with a different query."

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

# For testing
if __name__ == "__main__":
    test_query = "Given an array of integers, find four elements that sum to a given target value."
    result = generate_response(test_query)
    print("\nFINAL RESPONSE:")
    print(result)