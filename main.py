import os
import sys
import argparse

def setup_folders():
    """Set up the folder structure for the project."""
    folders = [
        "data/raw",
        "data/processed",
        "data/embeddings",
        "scripts/scrapers",
        "scripts/processors",
        "scripts/embedding",
        "api",
        "models"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

def run_data_collection():
    """Run all data collection scripts."""
    from scripts.scrapers.geeksforgeeks_scraper import scrape_geeksforgeeks_algorithms
    from scripts.scrapers.leetcode_scraper import scrape_leetcode_algorithms
    from scripts.scrapers.custom_algorithms import create_custom_algorithms
    
    print("Running GeeksforGeeks scraper...")
    scrape_geeksforgeeks_algorithms()
    
    print("Running LeetCode scraper...")
    scrape_leetcode_algorithms()
    
    print("Creating custom algorithm entries...")
    create_custom_algorithms()

def run_data_processing():
    """Run all data processing scripts."""
    from scripts.processors.algorithm_hierarchy import process_all_data
    from scripts.processors.nlp_enhancer import enhance_processed_data
    
    print("Processing all algorithm data...")
    process_all_data()
    
    print("Enhancing data with NLP techniques...")
    enhance_processed_data()

def run_embedding_generation():
    """Run embedding generation and vector store setup."""
    from scripts.embedding.generate_embeddings import generate_embeddings
    from scripts.embedding.vector_store import setup_vector_store
    
    print("Generating embeddings...")
    generate_embeddings()
    
    print("Setting up vector store...")
    setup_vector_store()

def run_api_server():
    """Run the API server for RAG integration."""
    import uvicorn
    
    print("Starting RAG API server...")
    uvicorn.run("api.rag_service:app", host="0.0.0.0", port=8000, reload=True)

def main():
    parser = argparse.ArgumentParser(description="Algorithm RAG System")
    parser.add_argument("--setup", action="store_true", help="Set up folder structure")
    parser.add_argument("--collect", action="store_true", help="Run data collection")
    parser.add_argument("--process", action="store_true", help="Run data processing")
    parser.add_argument("--embed", action="store_true", help="Generate embeddings")
    parser.add_argument("--serve", action="store_true", help="Run RAG API server")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    
    args = parser.parse_args()
    
    if args.setup or args.all:
        setup_folders()
    
    if args.collect or args.all:
        run_data_collection()
    
    if args.process or args.all:
        run_data_processing()
    
    if args.embed or args.all:
        run_embedding_generation()
    
    if args.serve or args.all:
        run_api_server()
    
    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()
