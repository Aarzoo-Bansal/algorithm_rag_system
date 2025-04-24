# import os
# import json
# import pandas as pd
# import uuid
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import glob
# from tqdm import tqdm
# from algorithm_hierarchy import get_algorithm_hierarchy, get_category_for_algorithm

# def generate_uuid():
#     """Generate a unique identifier."""
#     return str(uuid.uuid4())

# def extract_use_cases(description):
#     """Extract use cases from a description."""
#     # Simple implementation - in a real system, you might use NLP
#     use_cases = []
    
#     # Look for applications/uses/problems sections
#     sentences = description.split('.')
#     for sentence in sentences:
#         sentence = sentence.strip().lower()
#         if any(phrase in sentence for phrase in ['used for', 'used to', 'applies to', 'solves', 'application']):
#             use_cases.append(sentence.capitalize())
    
#     # If none found, return empty list
#     if not use_cases and len(sentences) > 2:
#         use_cases.append(sentences[1].strip().capitalize())
        
#     return use_cases[:3]  # Return at most 3 use cases

# def determine_difficulty(complexity_text):
#     """Determine algorithm difficulty based on complexity."""
#     # Simple implementation - in production, use more sophisticated logic
#     if 'O(n)' in complexity_text or 'O(log n)' in complexity_text:
#         return "Easy"
#     elif 'O(n log n)' in complexity_text:
#         return "Medium"
#     elif 'O(n^2)' in complexity_text or 'O(2^n)' in complexity_text:
#         return "Hard"
#     else:
#         # Default if we can't determine
#         return "Medium"

# def determine_secondary_tag(description):
#     """Determine a secondary tag based on description."""
#     # Simple implementation
#     tags = ["sorting", "dynamic programming", "graph", "tree", "array", "string", "math"]
    
#     description_lower = description.lower()
#     matched_tags = []
    
#     for tag in tags:
#         if tag in description_lower:
#             matched_tags.append(tag)
    
#     if not matched_tags:
#         return "general"
    
#     return matched_tags[0]

# def load_all_raw_data():
#     """Load all raw algorithm data from the data/raw directory."""
#     file_path = os.path.join("./data/raw", "custom_algorithm_database.json")
#     try:
#         with open(file_path, 'r') as f:
#             data = json.load(f)

#         algorithms = data.get('algorithms', [])

#         # Adding id tags to algorithms if not present
#         for algo in algorithms:
#             if 'id' not in algo or not algo['id']:
#                 algo['id'] = generate_uuid()

#         print(f"Loaded {len(algorithms)} algorithms from {file_path}")
#         return algorithms
#     except Exception as e:
#         print(f"Error loading {file_path}: {e}")
#         return []


# def process_algorithm_data(data):
#     """Convert raw data into structured format for embedding."""
#     if isinstance(data, pd.DataFrame):
#         df = data
#     else:
#         df = pd.DataFrame(data)
    
   
#     documents = []
#     for _, row in df.iterrows():
#         try:
        
#             if isinstance(row, dict):
                
#                 title = row.get('title', '')
#                 description = row.get('description', '')
#                 name = row.get('name', '')
#                 difficulty = row.get('difficulty', '')
#                 tags = row.get('tags', [])

#                 item_id = row.get('id')
#                 if not item_id:
#                     item_id = generate_uuid()
#                     row['id'] = item_id
            
           
#                 if 'complexity' in row and isinstance(row['complexity'], dict):
#                     complexity_time = row['complexity'].get('time', '')
#                     complexity_space = row['complexity'].get('space', '')
#                 else:
#                     complexity_time = 'O(n)'
#                     complexity_space = 'O(n)'
                    
#                 use_cases = row.get('use_cases', [])
#                 solution = row.get('solution', '')
              
#             else:
#                 # DataFrame row access
#                 title = row['title'] if 'title' in row else ''
#                 description = row['description'] if 'description' in row else ''
#                 name = row['name'] if 'name' in row else ''
#                 difficulty = row['difficulty'] if 'difficulty' in row else ''
#                 tags = row['tags'] if 'tags' in row else []
                
#                 if 'id' in row and row['id']:
#                     item_id = row['id']
#                 else:
#                     item_id = generate_uuid()

#                 # Handle nested complexity
#                 if 'complexity' in row and isinstance(row['complexity'], dict):
#                     complexity_time = row['complexity'].get('time', '')
#                     complexity_space = row['complexity'].get('space', '')
#                 else:
#                     complexity_time = 'O(n)'
#                     complexity_space = 'O(n)'
                    
#                 use_cases = row['use_cases'] if 'use_cases' in row else []
#                 solution = row['solution'] if 'solution' in row else ''
               
#             # Convert lists to strings
#             tags_str = ', '.join(tags) if isinstance(tags, list) else str(tags)
#             use_cases_str = ', '.join(use_cases) if isinstance(use_cases, list) else str(use_cases)
            
#             # Add category based on hierarchy if not present
#             if not any(tag in ['sorting', 'searching', 'graph', 'dynamic_programming', 'data_structures', 
#                             'string', 'mathematical', 'greedy', 'backtracking'] for tag in tags):
#                 category, subcategory = get_category_for_algorithm(name)
#                 if category:
#                     if isinstance(tags, list):
#                         tags.append(category)
#                         if subcategory:
#                             tags.append(subcategory)
#                     else:
#                         tags = [category]
#                         if subcategory:
#                             tags.append(subcategory)
#                     tags_str = ', '.join(tags) if isinstance(tags, list) else str(tags)
            
#             # Create a comprehensive document
#             content = f"""
#             Title: {title}
#             Description: {description}
#             Name: {name}
#             Difficulty: {difficulty}
#             Tags: {tags_str}
#             Complexity (Time): {complexity_time}
#             Complexity (Space): {complexity_space}
#             Use Cases: {use_cases_str}
#             Solution Approach: {solution}
#             """
            
#             # Store document with metadata
#             documents.append({
#                 "content": content.strip(),
#                 "metadata": {
#                     "id": item_id,
#                     "title": title,
#                     "name": name,
#                     "difficulty": difficulty,
#                     "tags": tags if isinstance(tags, list) else [tags],
#                 }
#             })
#         except Exception as e:
#             print(f"Error processing row: {e}")
#             continue
    
#     return documents

# def split_documents(documents, chunk_size=400, chunk_overlap=50):
#     """Split documents into smaller chunks for embedding."""
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap
#     )
    
#     chunks = []
#     for doc in tqdm(documents, desc="Splitting documents"):
#         splits = text_splitter.split_text(doc["content"])
#         for split in splits:
#             chunks.append({
#                 "content": split,
#                 "metadata": doc["metadata"]
#             })
    
#     return chunks

# def combine_and_deduplicate(all_data):
#     """Combine and deduplicate algorithm data."""
  
#     unique_algorithms = {}
    
#     for alg in all_data:
#         name = alg.get('name', '').lower()

#         if 'id' not in alg or not alg['id']:
#             alg['id'] = generate_uuid()

#         if name in unique_algorithms:
#             existing = unique_algorithms[name]
#             alg_id = existing.get('id')

#             for key, value in alg.items():
#                 if key not in existing or not existing[key]:
#                     if key == 'id' and alg_id:
#                         continue
#                     existing[key] = value
#                 elif key == 'tags' and isinstance(value, list) and isinstance(existing[key], list):
#                     existing[key] = list(set(existing[key] + value))
#                 elif key == 'problem_examples' and isinstance(value, list):
#                     if 'problem_examples' not in existing:
#                         existing['problem_examples'] = []
#                     existing['problem_examples'].extend(value)
#         else:
#             unique_algorithms[name] = alg
    
#     return list(unique_algorithms.values())

# def process_all_data():
#     """Process all algorithm data and prepare for embedding."""
#     # Create output directories
#     processed_dir = "./data/processed"
#     os.makedirs(processed_dir, exist_ok=True)
    
#     # Load all raw data
#     all_data = load_all_raw_data()
#     print(f"Loaded {len(all_data)} total algorithm entries")
    
#     # Combine and deduplicate
#     combined_data = combine_and_deduplicate(all_data)
#     print(f"Combined into {len(combined_data)} unique algorithms")
    
#     # Process into document format
#     documents = process_algorithm_data(combined_data)
#     print(f"Created {len(documents)} algorithm documents")
    
#     # Split into chunks for embedding
#     chunks = split_documents(documents)
#     print(f"Split into {len(chunks)} text chunks for embedding")
    
#     # Save processed data
#     with open(os.path.join(processed_dir, "combined_algorithms.json"), 'w') as f:
#         json.dump(combined_data, f, indent=2)
        
#     with open(os.path.join(processed_dir, "algorithm_documents.json"), 'w') as f:
#         json.dump(documents, f, indent=2)
        
#     with open(os.path.join(processed_dir, "algorithm_chunks.json"), 'w') as f:
#         json.dump(chunks, f, indent=2)
    
#     return combined_data, documents, chunks

# if __name__ == "__main__":
#     process_all_data()

import os
import json
import pandas as pd
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
from tqdm import tqdm
from algorithm_hierarchy import get_algorithm_hierarchy, get_category_for_algorithm

def generate_uuid():
    """Generate a unique identifier."""
    return str(uuid.uuid4())

def extract_use_cases(description):
    """Extract use cases from a description."""
    # Simple implementation - in a real system, you might use NLP
    use_cases = []
    
    # Look for applications/uses/problems sections
    sentences = description.split('.')
    for sentence in sentences:
        sentence = sentence.strip().lower()
        if any(phrase in sentence for phrase in ['used for', 'used to', 'applies to', 'solves', 'application']):
            use_cases.append(sentence.capitalize())
    
    # If none found, return empty list
    if not use_cases and len(sentences) > 2:
        use_cases.append(sentences[1].strip().capitalize())
        
    return use_cases[:3]  # Return at most 3 use cases

def determine_difficulty(complexity_text):
    """Determine algorithm difficulty based on complexity."""
    # Simple implementation - in production, use more sophisticated logic
    if 'O(n)' in complexity_text or 'O(log n)' in complexity_text:
        return "Easy"
    elif 'O(n log n)' in complexity_text:
        return "Medium"
    elif 'O(n^2)' in complexity_text or 'O(2^n)' in complexity_text:
        return "Hard"
    else:
        # Default if we can't determine
        return "Medium"

def determine_secondary_tag(description):
    """Determine a secondary tag based on description."""
    # Simple implementation
    tags = ["sorting", "dynamic programming", "graph", "tree", "array", "string", "math"]
    
    description_lower = description.lower()
    matched_tags = []
    
    for tag in tags:
        if tag in description_lower:
            matched_tags.append(tag)
    
    if not matched_tags:
        return "general"
    
    return matched_tags[0]

def load_all_raw_data():
    """Load all raw algorithm data from the data/raw directory."""
    file_path = os.path.join("./data/raw", "custom_algorithm_database.json")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        algorithms = data.get('algorithms', [])

        # Adding id tags to algorithms if not present
        for algo in algorithms:
            if 'id' not in algo or not algo['id']:
                algo['id'] = generate_uuid()

        print(f"Loaded {len(algorithms)} algorithms from {file_path}")
        return algorithms
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def add_common_examples(documents):
    """
    Add common example problems for algorithms to improve retrieval.
    This is especially important for frequently asked algorithm questions.
    """
    examples_map = {
        "sliding window": [
            "Finding the longest substring without repeating characters",
            "Finding maximum/minimum size subarray with a sum that equals a given value",
            "Finding all anagrams in a string",
            "Maximum sum subarray of size k"
        ],
        "dynamic programming": [
            "Longest common subsequence",
            "Knapsack problem",
            "Coin change problem",
            "Edit distance"
        ],
        "graph": [
            "Shortest path problem",
            "Minimum spanning tree",
            "Detect cycle in a graph",
            "Topological sorting"
        ],
        "binary search": [
            "Search in a sorted array",
            "Find first and last occurrence of an element",
            "Search in a rotated sorted array",
            "Find peak element"
        ],
        "two pointers": [
            "Two sum problem",
            "Container with most water",
            "Remove duplicates from sorted array",
            "Finding triplets that sum to zero"
        ],
        "tree traversal": [
            "Inorder, preorder, postorder traversal",
            "Level order traversal",
            "Find height/depth of a tree",
            "Check if a binary tree is balanced"
        ]
    }
    
    enhanced_documents = []
    
    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]
        tags = metadata.get("tags", [])
        name_lower = metadata.get("name", "").lower()
        
        # Check if this algorithm matches any of our example categories
        added_examples = []
        for key, examples in examples_map.items():
            if key in name_lower or any(key in tag.lower() for tag in tags):
                # Add examples to the content
                if "Example Problems:" not in content:
                    example_text = "\nExample Problems:\n"
                    for i, example in enumerate(examples):
                        example_text += f"{i+1}. {example}\n"
                    content += example_text
                    added_examples.extend(examples)
                break  # Only add examples from one category
        
        # For sliding window specifically, ensure longest substring problem is included
        if "sliding window" in name_lower or any("sliding window" in tag.lower() for tag in tags):
            longest_substring_example = "Finding the longest substring without repeating characters"
            if longest_substring_example not in content:
                if "Example Problems:" not in content:
                    content += f"\nExample Problems:\n1. {longest_substring_example}\n"
                else:
                    # Add to existing examples
                    content += f"- {longest_substring_example}\n"
        
        enhanced_documents.append({
            "content": content,
            "metadata": metadata
        })
    
    return enhanced_documents

def process_algorithm_data(data):
    """Convert raw data into structured format for embedding."""
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)
    
    documents = []
    for _, row in df.iterrows():
        try:
            if isinstance(row, dict):
                title = row.get('title', '')
                description = row.get('description', '')
                name = row.get('name', '')
                difficulty = row.get('difficulty', '')
                tags = row.get('tags', [])

                item_id = row.get('id')
                if not item_id:
                    item_id = generate_uuid()
                    row['id'] = item_id
            
                if 'complexity' in row and isinstance(row['complexity'], dict):
                    complexity_time = row['complexity'].get('time', '')
                    complexity_space = row['complexity'].get('space', '')
                else:
                    complexity_time = 'O(n)'
                    complexity_space = 'O(n)'
                    
                use_cases = row.get('use_cases', [])
                solution = row.get('solution', '')
              
            else:
                # DataFrame row access
                title = row['title'] if 'title' in row else ''
                description = row['description'] if 'description' in row else ''
                name = row['name'] if 'name' in row else ''
                difficulty = row['difficulty'] if 'difficulty' in row else ''
                tags = row['tags'] if 'tags' in row else []
                
                if 'id' in row and row['id']:
                    item_id = row['id']
                else:
                    item_id = generate_uuid()

                # Handle nested complexity
                if 'complexity' in row and isinstance(row['complexity'], dict):
                    complexity_time = row['complexity'].get('time', '')
                    complexity_space = row['complexity'].get('space', '')
                else:
                    complexity_time = 'O(n)'
                    complexity_space = 'O(n)'
                    
                use_cases = row['use_cases'] if 'use_cases' in row else []
                solution = row['solution'] if 'solution' in row else ''
               
            # Convert lists to strings
            tags_str = ', '.join(tags) if isinstance(tags, list) else str(tags)
            use_cases_str = ', '.join(use_cases) if isinstance(use_cases, list) else str(use_cases)
            
            # Add category based on hierarchy if not present
            if not any(tag in ['sorting', 'searching', 'graph', 'dynamic_programming', 'data_structures', 
                            'string', 'mathematical', 'greedy', 'backtracking'] for tag in tags):
                category, subcategory = get_category_for_algorithm(name)
                if category:
                    if isinstance(tags, list):
                        tags.append(category)
                        if subcategory:
                            tags.append(subcategory)
                    else:
                        tags = [category]
                        if subcategory:
                            tags.append(subcategory)
                    tags_str = ', '.join(tags) if isinstance(tags, list) else str(tags)
            
            # Enhance the description for better retrieval
            if description and name and "sliding window" in name.lower():
                if "longest substring without repeating characters" not in description.lower():
                    description += " This technique is particularly useful for problems like finding the longest substring without repeating characters."
            
            # Create a comprehensive document with a more structured format
            content = f"""Algorithm: {name}
Description: {description}
Tags: {tags_str}
Complexity (Time): {complexity_time}
Complexity (Space): {complexity_space}
Use Cases: {use_cases_str}
"""
            
            if solution:
                content += f"Solution Approach: {solution}\n"
                
            # Store document with metadata
            documents.append({
                "content": content.strip(),
                "metadata": {
                    "id": item_id,
                    "title": title,
                    "name": name,
                    "difficulty": difficulty,
                    "tags": tags if isinstance(tags, list) else [tags],
                }
            })
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    # Add common example problems to appropriate algorithms
    enhanced_documents = add_common_examples(documents)
    
    return enhanced_documents

def alternative_chunking_strategy(documents):
    """
    OPTION 1: Keep each algorithm as a single chunk (no splitting)
    This ensures all information about an algorithm stays together.
    """
    return documents

def semantic_chunking_strategy(documents, chunk_size=1000, chunk_overlap=200):
    """
    OPTION 2: Split by semantic boundaries instead of character count.
    Uses larger chunks with more overlap to maintain context.
    """
    chunks = []
    
    # Use a more careful text splitter with larger size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    for doc in tqdm(documents, desc="Creating semantic chunks"):
        content = doc["content"]
        metadata = doc["metadata"]
        
        # For shorter algorithms, keep as a single chunk
        if len(content) < chunk_size:
            chunks.append({
                "content": content,
                "metadata": metadata
            })
            continue
            
        # For longer ones, split with attention to semantic boundaries
        content_parts = []
        
        # First attempt to split by sections
        sections = {}
        
        # Extract key sections
        if "Algorithm:" in content:
            sections["header"] = content.split("Description:")[0] if "Description:" in content else content
        
        if "Description:" in content:
            desc_part = content.split("Description:")[1]
            if "Tags:" in desc_part:
                sections["description"] = "Description:" + desc_part.split("Tags:")[0]
            else:
                sections["description"] = "Description:" + desc_part
        
        if "Tags:" in content:
            tags_part = content.split("Tags:")[1]
            if "Complexity" in tags_part:
                sections["tags"] = "Tags:" + tags_part.split("Complexity")[0]
            else:
                sections["tags"] = "Tags:" + tags_part
        
        if "Complexity" in content:
            complexity_part = content.split("Complexity")[1]
            if "Use Cases:" in complexity_part:
                sections["complexity"] = "Complexity" + complexity_part.split("Use Cases:")[0]
            else:
                sections["complexity"] = "Complexity" + complexity_part
        
        if "Use Cases:" in content:
            cases_part = content.split("Use Cases:")[1]
            if "Solution Approach:" in cases_part:
                sections["use_cases"] = "Use Cases:" + cases_part.split("Solution Approach:")[0]
            else:
                sections["use_cases"] = "Use Cases:" + cases_part
        
        if "Solution Approach:" in content:
            sections["solution"] = "Solution Approach:" + content.split("Solution Approach:")[1]
        
        if "Example Problems:" in content:
            sections["examples"] = "Example Problems:" + content.split("Example Problems:")[1]
        
        # If we successfully extracted sections, create chunks with meaningful section combinations
        if sections:
            # First chunk: Essential algorithm info (header, description, tags, complexity)
            first_chunk = ""
            for section in ["header", "description", "tags", "complexity"]:
                if section in sections:
                    first_chunk += sections[section] + "\n"
            
            if first_chunk:
                chunks.append({
                    "content": first_chunk.strip(),
                    "metadata": metadata
                })
            
            # Second chunk: Examples and use cases
            second_chunk = ""
            for section in ["examples", "use_cases"]:
                if section in sections:
                    second_chunk += sections[section] + "\n"
            
            if second_chunk:
                chunks.append({
                    "content": second_chunk.strip(),
                    "metadata": metadata
                })
            
            # Third chunk: Solution approach (if it exists and is long)
            if "solution" in sections and len(sections["solution"]) > 200:
                chunks.append({
                    "content": sections["solution"].strip(),
                    "metadata": metadata
                })
        else:
            # Fallback to standard chunking if section extraction fails
            splits = text_splitter.split_text(content)
            for split in splits:
                chunks.append({
                    "content": split,
                    "metadata": metadata
                })
    
    return chunks

def combine_and_deduplicate(all_data):
    """Combine and deduplicate algorithm data."""
  
    unique_algorithms = {}
    
    for alg in all_data:
        name = alg.get('name', '').lower()

        if 'id' not in alg or not alg['id']:
            alg['id'] = generate_uuid()

        if name in unique_algorithms:
            existing = unique_algorithms[name]
            alg_id = existing.get('id')

            for key, value in alg.items():
                if key not in existing or not existing[key]:
                    if key == 'id' and alg_id:
                        continue
                    existing[key] = value
                elif key == 'tags' and isinstance(value, list) and isinstance(existing[key], list):
                    existing[key] = list(set(existing[key] + value))
                elif key == 'problem_examples' and isinstance(value, list):
                    if 'problem_examples' not in existing:
                        existing['problem_examples'] = []
                    existing['problem_examples'].extend(value)
        else:
            unique_algorithms[name] = alg
    
    return list(unique_algorithms.values())

def process_all_data(chunking_strategy="semantic", chunk_size=1000, chunk_overlap=200):
    """Process all algorithm data and prepare for embedding."""
    # Create output directories
    processed_dir = "./data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load all raw data
    all_data = load_all_raw_data()
    print(f"Loaded {len(all_data)} total algorithm entries")
    
    # Combine and deduplicate
    combined_data = combine_and_deduplicate(all_data)
    print(f"Combined into {len(combined_data)} unique algorithms")
    
    # Process into document format
    documents = process_algorithm_data(combined_data)
    print(f"Created {len(documents)} algorithm documents")
    
    # Choose chunking strategy
    if chunking_strategy == "none":
        # Don't split - keep one document per algorithm
        chunks = alternative_chunking_strategy(documents)
        print(f"Using whole-algorithm chunking: {len(chunks)} chunks created")
    elif chunking_strategy == "semantic":
        # Split using semantic boundaries
        chunks = semantic_chunking_strategy(documents, chunk_size, chunk_overlap)
        print(f"Using semantic chunking: {len(chunks)} chunks created")
    else:
        # Use original character-based splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = []
        for doc in tqdm(documents, desc="Splitting documents"):
            splits = text_splitter.split_text(doc["content"])
            for split in splits:
                chunks.append({
                    "content": split,
                    "metadata": doc["metadata"]
                })
        print(f"Using character-based chunking: {len(chunks)} chunks created")
    
    # Save processed data
    with open(os.path.join(processed_dir, "combined_algorithms.json"), 'w') as f:
        json.dump(combined_data, f, indent=2)
        
    with open(os.path.join(processed_dir, "algorithm_documents.json"), 'w') as f:
        json.dump(documents, f, indent=2)
        
    with open(os.path.join(processed_dir, "algorithm_chunks.json"), 'w') as f:
        json.dump(chunks, f, indent=2)
    
    return combined_data, documents, chunks

if __name__ == "__main__":
    # Use semantic chunking with larger chunks and overlap by default
    process_all_data(chunking_strategy="semantic", chunk_size=1000, chunk_overlap=200)