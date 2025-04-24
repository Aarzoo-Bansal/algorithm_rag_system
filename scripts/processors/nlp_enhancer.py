import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

def enhance_algorithm_database_with_nlp(algorithm_data):
    """Enhance algorithm database with NLP for better tagging and categorization."""
    # Download NLTK resources if needed
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    
    # Extract descriptions for processing
    descriptions = [alg.get('description', '') for alg in algorithm_data]
    
    # Preprocess descriptions
    stop_words = set(stopwords.words('english'))
    processed_descriptions = []
    
    for desc in descriptions:
        # Tokenize and filter
        tokens = nltk.word_tokenize(desc.lower())
        filtered = [word for word in tokens if word not in stop_words]
        processed_descriptions.append(' '.join(filtered))

    if any(not desc for desc in processed_descriptions):
        print("Warning: Some descriptions are empty after processing")
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=1000, min_df=1, stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(processed_descriptions)
    
    # Extract key terms for each algorithm
    feature_names = vectorizer.get_feature_names_out()
    
    # For each algorithm, extract key terms
    for i, alg in enumerate(tqdm(algorithm_data, desc="Enhancing algorithms with NLP")):
        if i >= len(processed_descriptions):
            continue
            
        # Get TF-IDF scores for this document
        tfidf_scores = tfidf_matrix[i].toarray()[0]
        
        # Get top terms
        top_indices = np.argsort(tfidf_scores)[-10:]  # Top 10 terms
        top_terms = [feature_names[idx] for idx in top_indices if tfidf_scores[idx] > 0]
        
        # Add to tags if not already present
        if 'tags' not in alg:
            alg['tags'] = []
        
        for term in top_terms:
            if term not in alg['tags'] and len(term) > 3:  # Avoid short terms
                alg['tags'].append(term)
    
    # Cluster algorithms based on descriptions
    num_clusters = min(20, len(algorithm_data))  # Cap at 20 clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    
    # Add cluster information to algorithms
    for i, cluster_id in enumerate(clusters):
        if i < len(algorithm_data):
            algorithm_data[i]['cluster_id'] = int(cluster_id)
    
    return algorithm_data

def enhance_processed_data():
    """Enhance the processed algorithm data with NLP techniques."""
    processed_dir = "./data/processed"
    
    # Load combined algorithms
    combined_file = os.path.join(processed_dir, "combined_algorithms.json")
    with open(combined_file, 'r') as f:
        algorithms = json.load(f)
    
    print(f"Loaded {len(algorithms)} algorithms for NLP enhancement")
    
    # Apply NLP enhancement
    enhanced_algorithms = enhance_algorithm_database_with_nlp(algorithms)
    
    # Save enhanced data
    enhanced_file = os.path.join(processed_dir, "enhanced_algorithms.json")
    with open(enhanced_file, 'w') as f:
        json.dump(enhanced_algorithms, f, indent=2)
    
    print(f"Saved {len(enhanced_algorithms)} enhanced algorithms")
    return enhanced_algorithms

if __name__ == "__main__":
    enhance_processed_data()