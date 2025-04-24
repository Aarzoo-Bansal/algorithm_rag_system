"""
Defines a hierarchical structure for organizing algorithms.
This is used to categorize algorithms during processing.
"""

ALGORITHM_HIERARCHY = {
    "sorting": {
        "comparison_based": ["quick sort", "merge sort", "heap sort", "insertion sort", "bubble sort"],
        "non_comparison": ["counting sort", "radix sort", "bucket sort"]
    },
    "searching": {
        "array_search": ["binary search", "linear search", "jump search", "interpolation search"],
        "graph_search": ["depth-first search", "breadth-first search", "a* search", "best-first search"]
    },
    "graph": {
        "traversal": ["depth-first search", "breadth-first search"],
        "shortest_path": ["dijkstra", "bellman-ford", "floyd-warshall", "a*"],
        "minimum_spanning_tree": ["prim", "kruskal"],
        "connectivity": ["tarjan", "kosaraju"],
        "maximum_flow": ["ford-fulkerson", "edmonds-karp", "dinic"]
    },
    "dynamic_programming": {
        "optimization": ["knapsack", "longest common subsequence", "matrix chain multiplication"],
        "counting": ["ways to climb stairs", "combination sum", "coin change"],
        "path_finding": ["grid paths", "edit distance"]
    },
    "data_structures": {
        "linear": ["array", "linked list", "stack", "queue", "deque"],
        "tree": ["binary tree", "binary search tree", "avl tree", "red-black tree", "b-tree"],
        "graph": ["adjacency list", "adjacency matrix", "edge list"],
        "hash": ["hash table", "hash map", "hash set"],
        "heap": ["binary heap", "fibonacci heap", "min heap", "max heap"]
    },
    "string": {
        "pattern_matching": ["naive", "kmp", "rabin-karp", "boyer-moore"],
        "string_distance": ["edit distance", "hamming distance", "levenshtein"],
        "tries": ["trie", "suffix tree", "suffix array"]
    },
    "mathematical": {
        "number_theory": ["gcd", "lcm", "sieve of eratosthenes", "modular exponentiation"],
        "numerical": ["newton raphson", "gaussian elimination"],
        "matrix": ["matrix multiplication", "matrix exponentiation"]
    },
    "greedy": {
        "scheduling": ["interval scheduling", "job sequencing"],
        "selection": ["activity selection", "coin change"]
    },
    "backtracking": {
        "permutation": ["n-queens", "permutations"],
        "combination": ["subset sum", "combination sum"]
    },
    "advanced": {
        "computational_geometry": ["convex hull", "line intersection"],
        "randomized": ["randomized quicksort", "karger min cut"],
        "approximation": ["traveling salesman approximation", "set cover"]
    }
}

def get_algorithm_hierarchy():
    """Return the algorithm hierarchy structure."""
    return ALGORITHM_HIERARCHY

def get_all_algorithm_names():
    """Extract all algorithm names from the hierarchy."""
    all_names = []
    
    for category, subcategories in ALGORITHM_HIERARCHY.items():
        for subcategory, algorithms in subcategories.items():
            all_names.extend(algorithms)
    
    return list(set(all_names))

def get_category_for_algorithm(algorithm_name):
    """Find the category and subcategory for a given algorithm name."""
    algorithm_name = algorithm_name.lower()
    
    for category, subcategories in ALGORITHM_HIERARCHY.items():
        for subcategory, algorithms in subcategories.items():
            if any(algorithm in algorithm_name for algorithm in algorithms):
                return category, subcategory
    
    return None, None

if __name__ == "__main__":
    # Print all algorithms in the hierarchy
    all_algorithms = get_all_algorithm_names()
    print(f"Total algorithms in hierarchy: {len(all_algorithms)}")
    print(all_algorithms)  # Print first 10 as example