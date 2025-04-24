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
            
            # Create a comprehensive document
            content = f"""
            Title: {title}
            Description: {description}
            Name: {name}
            Difficulty: {difficulty}
            Tags: {tags_str}
            Complexity (Time): {complexity_time}
            Complexity (Space): {complexity_space}
            Use Cases: {use_cases_str}
            Solution Approach: {solution}
            """
            
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
    
    return documents

def split_documents(documents, chunk_size=400, chunk_overlap=50):
    """Split documents into smaller chunks for embedding."""
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

def process_all_data():
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
    
    # Split into chunks for embedding
    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} text chunks for embedding")
    
    # Save processed data
    with open(os.path.join(processed_dir, "combined_algorithms.json"), 'w') as f:
        json.dump(combined_data, f, indent=2)
        
    with open(os.path.join(processed_dir, "algorithm_documents.json"), 'w') as f:
        json.dump(documents, f, indent=2)
        
    with open(os.path.join(processed_dir, "algorithm_chunks.json"), 'w') as f:
        json.dump(chunks, f, indent=2)
    
    return combined_data, documents, chunks

if __name__ == "__main__":
    process_all_data()