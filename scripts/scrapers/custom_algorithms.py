import json
import uuid
import os
from tqdm import tqdm

os.makedirs("data", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

def generate_uuid():
    """Generate a unique identifier."""
    return str(uuid.uuid4())

def generate_algorithm_dataset():
    """Generate a comprehensive dataset of algorithms for a RAG system."""
    print("Generating algorithm dataset...")
    
    categories = [
        "Sorting Algorithms",
        "Searching Algorithms",
        "Graph Algorithms",
        "Dynamic Programming",
        "String Algorithms",
        "Tree Algorithms",
        "Greedy Algorithms",
        "Divide and Conquer",
        "Mathematical Algorithms",
        "Bit Manipulation",
        "Linked List Algorithms",
        "Stack Algorithms",
        "Queue Algorithms",
        "Hash Table Algorithms",
        "Advanced Data Structures",
        "Advanced Algorithms",
        "Specialized Techniques"
    ]
    
    algorithms = []
    
    sorting_algorithms = [
        {
            "name": "Quick Sort",
            "tags": ["sorting", "divide and conquer", "comparison sort"],
            "description": "Quick sort is a highly efficient sorting algorithm that uses a divide-and-conquer strategy. It works by selecting a 'pivot' element from the array and partitioning the other elements into two sub-arrays according to whether they are less than or greater than the pivot. The sub-arrays are then recursively sorted.",
            "complexity": {
                "time": "O(n log n)",
                "worst_time": "O(n²)",
                "space": "O(log n)"
            },
            "problem_patterns": [
                "Need to sort an array or list efficiently",
                "When average-case performance is more important than worst-case",
                "When in-place sorting is desired"
            ],
            "leetcode_indicators": [
                "Sorting array or list",
                "Problems requiring efficient ordering",
                "Problems where elements need to be partitioned"
            ],
            "implementation": """
                def quick_sort(arr):
                    if len(arr) <= 1:
                        return arr
                    
                    pivot = arr[len(arr) // 2]
                    left = [x for x in arr if x < pivot]
                    middle = [x for x in arr if x == pivot]
                    right = [x for x in arr if x > pivot]
                    
                    return quick_sort(left) + middle + quick_sort(right)
                """
        },
        {
            "name": "Merge Sort",
            "tags": ["sorting", "divide and conquer", "stable sort"],
            "description": "Merge sort is an efficient, stable, comparison-based, divide and conquer sorting algorithm. It divides the input array into two halves, recursively sorts them, and then merges the sorted halves. The merge step is the key operation, where the two sorted sub-arrays are combined to form a single sorted array.",
            "complexity": {
                "time": "O(n log n)",
                "space": "O(n)"
            },
            "problem_patterns": [
                "Need for stable sorting (preserving relative order of equal elements)",
                "When guaranteed worst-case performance is important",
                "Sorting linked lists"
            ],
            "leetcode_indicators": [
                "Stable sorting required",
                "Linked list sorting",
                "Problems involving counting inversions"
            ],
            "implementation": """
                def merge_sort(arr):
                    if len(arr) <= 1:
                        return arr
                    
                    mid = len(arr) // 2
                    left = merge_sort(arr[:mid])
                    right = merge_sort(arr[mid:])
                    
                    return merge(left, right)

                def merge(left, right):
                    result = []
                    i = j = 0
                    
                    while i < len(left) and j < len(right):
                        if left[i] <= right[j]:
                            result.append(left[i])
                            i += 1
                        else:
                            result.append(right[j])
                            j += 1
                    
                    result.extend(left[i:])
                    result.extend(right[j:])
                    return result
                """
        },
        {
            "name": "Heap Sort",
            "tags": ["sorting", "comparison sort", "in-place"],
            "description": "Heap sort is a comparison-based sorting algorithm that uses a binary heap data structure. It divides its input into a sorted region and an unsorted region, and iteratively shrinks the unsorted region by extracting the largest element and inserting it into the sorted region.",
            "complexity": {
                "time": "O(n log n)",
                "space": "O(1)"
            },
            "problem_patterns": [
                "When space complexity is a concern",
                "Finding the k largest/smallest elements",
                "When in-place sorting is required"
            ],
            "leetcode_indicators": [
                "K largest/smallest elements",
                "Priority queue problems",
                "Sorting with minimal extra space"
            ],
            "implementation": """
        def heapify(arr, n, i):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n and arr[left] > arr[largest]:
                largest = left
            
            if right < n and arr[right] > arr[largest]:
                largest = right
            
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                heapify(arr, n, largest)

        def heap_sort(arr):
            n = len(arr)
            
            # Build max heap
            for i in range(n // 2 - 1, -1, -1):
                heapify(arr, n, i)
            
            # Extract elements one by one
            for i in range(n - 1, 0, -1):
                arr[i], arr[0] = arr[0], arr[i]
                heapify(arr, i, 0)
            
            return arr
        """
        }
    ]
    
    algorithms.extend(sorting_algorithms)
    
    searching_algorithms = [
        {
            "name": "Binary Search",
            "tags": ["searching", "divide and conquer", "sorted array"],
            "description": "Binary search is an efficient algorithm for finding an item from a sorted list of items. It works by repeatedly dividing in half the portion of the list that could contain the item, until you've narrowed down the possible locations to just one.",
            "complexity": {
                "time": "O(log n)",
                "space": "O(1)"
            },
            "problem_patterns": [
                "Searching in a sorted array or list",
                "Finding the position to insert an element in a sorted array",
                "Problems requiring efficient search in monotonic functions"
            ],
            "leetcode_indicators": [
                "Search in sorted array",
                "Find first/last position of element",
                "Problems with O(log n) time complexity requirement",
                "Problems involving rotated sorted arrays"
            ],
            "implementation": """
                def binary_search(arr, target):
                    left, right = 0, len(arr) - 1
                    
                    while left <= right:
                        mid = (left + right) // 2
                        
                        if arr[mid] == target:
                            return mid
                        elif arr[mid] < target:
                            left = mid + 1
                        else:
                            right = mid - 1
                    
                    return -1  # Target not found
                """
        },
        {
            "name": "Depth-First Search (DFS)",
            "tags": ["searching", "graph algorithm", "tree traversal"],
            "description": "Depth-First Search is an algorithm for traversing or searching tree or graph data structures. The algorithm starts at the root node and explores as far as possible along each branch before backtracking.",
            "complexity": {
                "time": "O(V + E)",
                "space": "O(V)"
            },
            "problem_patterns": [
                "Traversing trees or graphs",
                "Finding connected components",
                "Path finding problems",
                "Cycle detection"
            ],
            "leetcode_indicators": [
                "Graph or tree traversal",
                "Path finding",
                "Connected components",
                "Cycle detection",
                "Problems requiring backtracking"
            ],
            "implementation": """
                def dfs(graph, start, visited=None):
                    if visited is None:
                        visited = set()
                    
                    visited.add(start)
                    print(start, end=' ')
                    
                    for neighbor in graph[start]:
                        if neighbor not in visited:
                            dfs(graph, neighbor, visited)
                    
                    return visited
                """
        },
        {
            "name": "Breadth-First Search (BFS)",
            "tags": ["searching", "graph algorithm", "tree traversal"],
            "description": "Breadth-First Search is an algorithm for traversing or searching tree or graph data structures. It starts at the tree root (or some arbitrary node in a graph) and explores all of the neighbor nodes at the present depth prior to moving on to the nodes at the next depth level.",
            "complexity": {
                "time": "O(V + E)",
                "space": "O(V)"
            },
            "problem_patterns": [
                "Finding shortest path in unweighted graphs",
                "Level order traversal of trees",
                "Finding all nodes within a distance k",
                "Problems requiring level-by-level processing"
            ],
            "leetcode_indicators": [
                "Shortest path in unweighted graph",
                "Level order traversal",
                "Minimum steps to reach target",
                "Problems involving word ladder or transformation"
            ],
            "implementation": """
                from collections import deque

                def bfs(graph, start):
                    visited = set([start])
                    queue = deque([start])
                    
                    while queue:
                        vertex = queue.popleft()
                        print(vertex, end=' ')
                        
                        for neighbor in graph[vertex]:
                            if neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)
                    
                    return visited
                """
        }
    ]
    
    algorithms.extend(searching_algorithms)
    
    graph_algorithms = [
        {
            "name": "Dijkstra's Algorithm",
            "tags": ["graph algorithm", "shortest path", "weighted graph"],
            "description": "Dijkstra's algorithm is used to find the shortest paths between nodes in a graph with non-negative edge weights. It uses a priority queue to greedily select the closest vertex that has not yet been processed and updates the distances to all its neighbors.",
            "complexity": {
                "time": "O((V + E) log V)",
                "space": "O(V)"
            },
            "problem_patterns": [
                "Finding shortest path in weighted graphs",
                "Network routing problems",
                "Problems involving path optimization"
            ],
            "leetcode_indicators": [
                "Shortest path in weighted graph",
                "Path with minimum cost/time/distance",
                "Network routing problems"
            ],
            "implementation": """
                import heapq

                def dijkstra(graph, start):
                    # Initialize distances with infinity for all nodes except start
                    distances = {node: float('infinity') for node in graph}
                    distances[start] = 0
                    
                    # Priority queue to store vertices to be processed
                    priority_queue = [(0, start)]
                    
                    while priority_queue:
                        current_distance, current_node = heapq.heappop(priority_queue)
                        
                        # If current distance is greater than the known distance, skip
                        if current_distance > distances[current_node]:
                            continue
                        
                        # Process neighbors
                        for neighbor, weight in graph[current_node].items():
                            distance = current_distance + weight
                            
                            # If we found a shorter path, update and add to queue
                            if distance < distances[neighbor]:
                                distances[neighbor] = distance
                                heapq.heappush(priority_queue, (distance, neighbor))
                    
                    return distances
                """
        },
        {
            "name": "Kruskal's Algorithm",
            "tags": ["graph algorithm", "minimum spanning tree", "greedy"],
            "description": "Kruskal's algorithm is a greedy algorithm that finds a minimum spanning tree for a connected weighted graph. It adds the edges in order of their weight (smallest to largest) as long as adding an edge doesn't create a cycle.",
            "complexity": {
                "time": "O(E log E)",
                "space": "O(V + E)"
            },
            "problem_patterns": [
                "Finding minimum spanning tree",
                "Network design problems",
                "Clustering problems"
            ],
            "leetcode_indicators": [
                "Minimum spanning tree",
                "Problems involving connecting all nodes at minimum cost",
                "Network design optimization"
            ],
            "implementation": """
                def find(parent, i):
                    if parent[i] != i:
                        parent[i] = find(parent, parent[i])
                    return parent[i]

                def union(parent, rank, x, y):
                    root_x = find(parent, x)
                    root_y = find(parent, y)
                    
                    if root_x == root_y:
                        return
                    
                    if rank[root_x] < rank[root_y]:
                        parent[root_x] = root_y
                    elif rank[root_x] > rank[root_y]:
                        parent[root_y] = root_x
                    else:
                        parent[root_y] = root_x
                        rank[root_x] += 1

                def kruskal(graph, vertices):
                    result = []
                    i, e = 0, 0
                    
                    # Sort edges by weight
                    graph = sorted(graph, key=lambda item: item[2])
                    
                    parent = []
                    rank = []
                    
                    # Initialize parent and rank arrays
                    for node in range(vertices):
                        parent.append(node)
                        rank.append(0)
                    
                    # Process edges
                    while e < vertices - 1 and i < len(graph):
                        u, v, w = graph[i]
                        i += 1
                        
                        x = find(parent, u)
                        y = find(parent, v)
                        
                        if x != y:
                            e += 1
                            result.append([u, v, w])
                            union(parent, rank, x, y)
                    
                    return result
                """
        },
        {
            "name": "Topological Sort",
            "tags": ["graph algorithm", "directed acyclic graph", "ordering"],
            "description": "Topological Sort is an algorithm for ordering the vertices of a directed acyclic graph (DAG) such that for every directed edge (u, v), vertex u comes before vertex v in the ordering.",
            "complexity": {
                "time": "O(V + E)",
                "space": "O(V)"
            },
            "problem_patterns": [
                "Task scheduling with dependencies",
                "Course prerequisites ordering",
                "Any problem requiring ordering based on dependencies"
            ],
            "leetcode_indicators": [
                "Course schedule problems",
                "Task scheduling with prerequisites",
                "Problems involving dependency ordering"
            ],
            "implementation": """
                from collections import defaultdict, deque

                def topological_sort(graph):
                    # Count in-degrees of all vertices
                    in_degree = {node: 0 for node in graph}
                    for node in graph:
                        for neighbor in graph[node]:
                            in_degree[neighbor] += 1
                    
                    # Queue with all nodes that have no incoming edges
                    queue = deque([node for node, degree in in_degree.items() if degree == 0])
                    result = []
                    
                    # Process nodes
                    while queue:
                        node = queue.popleft()
                        result.append(node)
                        
                        # Decrease in-degree of neighbors
                        for neighbor in graph[node]:
                            in_degree[neighbor] -= 1
                            if in_degree[neighbor] == 0:
                                queue.append(neighbor)
                    
                    # Check if there's a cycle
                    if len(result) != len(graph):
                        return []  # Graph has at least one cycle
                    
                    return result
                """
        }
    ]
    
    algorithms.extend(graph_algorithms)

    linked_list_algorithms = [
    {
        "name": "Linked List Reversal",
        "tags": ["linked list", "pointer manipulation", "in-place"],
        "description": "The Linked List Reversal algorithm takes a singly linked list and reverses the order of its nodes in-place by manipulating the pointers. This is done by iterating through the list and changing each node's next pointer to point to the previous node instead of the next one.",
        "complexity": {
            "time": "O(n)",
            "space": "O(1)"
        },
        "problem_patterns": [
            "Reversing a linked list or parts of a linked list",
            "Problems requiring modification of link directions",
            "In-place list restructuring"
        ],
        "leetcode_indicators": [
            "Reverse a linked list",
            "Reverse nodes in k-group",
            "Problems involving list direction manipulation"
        ],
        "implementation": """
            def reverse_linked_list(head):
                prev = None
                current = head
                
                while current:
                    # Store next node
                    next_node = current.next
                    
                    # Reverse the pointer
                    current.next = prev
                    
                    # Move to next iteration
                    prev = current
                    current = next_node
                
                # Return new head (which is the previous tail)
                return prev
            """
    },
    {
        "name": "Linked List Cycle Detection",
        "tags": ["linked list", "two pointers", "cycle detection"],
        "description": "The Linked List Cycle Detection algorithm (also known as Floyd's Tortoise and Hare algorithm) determines if a linked list has a cycle by using two pointers that move at different speeds. If there is a cycle, the fast pointer will eventually catch up to the slow pointer.",
        "complexity": {
            "time": "O(n)",
            "space": "O(1)"
        },
        "problem_patterns": [
            "Detecting cycles in linked lists",
            "Finding the start of a cycle",
            "Problems involving loop detection"
        ],
        "leetcode_indicators": [
            "Linked list cycle detection",
            "Find the start of cycle",
            "Check if a linked list contains a loop"
        ],
        "implementation": """
            def has_cycle(head):
                if not head or not head.next:
                    return False
                
                # Initialize slow and fast pointers
                slow = head
                fast = head
                
                # Move slow by 1 and fast by 2
                while fast and fast.next:
                    slow = slow.next
                    fast = fast.next.next
                    
                    # If they meet, there's a cycle
                    if slow == fast:
                        return True
                
                # If fast reaches the end, there's no cycle
                return False

            def find_cycle_start(head):
                if not head or not head.next:
                    return None
                
                # First, detect if there's a cycle
                slow = fast = head
                has_cycle = False
                
                while fast and fast.next:
                    slow = slow.next
                    fast = fast.next.next
                    
                    if slow == fast:
                        has_cycle = True
                        break
                
                # If no cycle, return None
                if not has_cycle:
                    return None
                
                # Reset slow to head and keep fast at meeting point
                slow = head
                
                # Move both at same pace until they meet
                while slow != fast:
                    slow = slow.next
                    fast = fast.next
                
                # Return the start of the cycle
                return slow
            """
    },
    {
        "name": "Linked List Rotation",
        "tags": ["linked list", "pointer manipulation", "two pointers"],
        "description": "The Linked List Rotation algorithm rotates a linked list to the right or left by k positions by manipulating pointers. The operation is performed by connecting the tail of the list to the head to form a circle, then breaking the circle at the appropriate point.",
        "complexity": {
            "time": "O(n)",
            "space": "O(1)"
        },
        "problem_patterns": [
            "Rotating elements in a linked list",
            "Problems involving circular rearrangement",
            "Shifting node positions without creating new nodes"
        ],
        "leetcode_indicators": [
            "Rotate list to the right/left by k places",
            "Rotate elements in a linked list",
            "Shift node positions in a linked list"
        ],
        "implementation": """
            def rotate_right(head, k):
                # Handle edge cases
                if not head or not head.next or k == 0:
                    return head
                
                # Find the length of the list and the tail
                current = head
                length = 1
                
                while current.next:
                    current = current.next
                    length += 1
                
                # Connect tail to head to make it circular
                tail = current
                tail.next = head
                
                # Calculate the number of effective rotations
                k = k % length
                
                # Find the new tail: (length - k - 1)th node
                current = head
                for _ in range(length - k - 1):
                    current = current.next
                
                # The new head is the next node
                new_head = current.next
                
                # Break the circle
                current.next = None
                
                return new_head
            """
    },
    {
        "name": "Linked List Merge",
        "tags": ["linked list", "two pointers", "sorting"],
        "description": "The Linked List Merge algorithm combines two sorted linked lists into a single sorted linked list by comparing nodes from both lists and linking them in the correct order.",
        "complexity": {
            "time": "O(n + m)",
            "space": "O(1)"
        },
        "problem_patterns": [
            "Merging sorted linked lists",
            "Combining multiple sorted structures",
            "In-place list integration"
        ],
        "leetcode_indicators": [
            "Merge two sorted linked lists",
            "Merge k sorted linked lists",
            "Sort a linked list using merge sort"
        ],
        "implementation": """
            def merge_two_lists(l1, l2):
                # Create a dummy head
                dummy = ListNode(0)
                current = dummy
                
                # Compare nodes and link them in order
                while l1 and l2:
                    if l1.val <= l2.val:
                        current.next = l1
                        l1 = l1.next
                    else:
                        current.next = l2
                        l2 = l2.next
                    current = current.next
                
                # Link remaining nodes
                current.next = l1 if l1 else l2
                
                return dummy.next
            """
    }
    ]

    algorithms.extend(linked_list_algorithms)

    # Stack Algorithms  
    stack_algorithms = [
    {
        "name": "Stack Implementation",
        "tags": ["stack", "data structure", "LIFO"],
        "description": "The Stack data structure follows Last-In-First-Out (LIFO) principle. It supports two primary operations: push (adding an element to the top) and pop (removing the top element). Stacks can be implemented using arrays or linked lists.",
        "complexity": {
            "time": "O(1) for push/pop operations",
            "space": "O(n)"
        },
        "problem_patterns": [
            "Problems requiring last-in-first-out processing",
            "Function call management",
            "Expression evaluation and parsing"
        ],
        "leetcode_indicators": [
            "Valid parentheses",
            "Evaluate expressions",
            "History tracking",
            "Undo operations"
        ],
        "implementation": """
            # Array-based stack implementation
            class Stack:
                def __init__(self):
                    self.items = []
                
                def is_empty(self):
                    return len(self.items) == 0
                
                def push(self, item):
                    self.items.append(item)
                
                def pop(self):
                    if self.is_empty():
                        raise IndexError("Pop from an empty stack")
                    return self.items.pop()
                
                def peek(self):
                    if self.is_empty():
                        raise IndexError("Peek from an empty stack")
                    return self.items[-1]
                
                def size(self):
                    return len(self.items)

            # Linked list-based stack implementation
            class Node:
                def __init__(self, value):
                    self.value = value
                    self.next = None

            class LinkedStack:
                def __init__(self):
                    self.top = None
                    self.size = 0
                
                def is_empty(self):
                    return self.top is None
                
                def push(self, value):
                    new_node = Node(value)
                    new_node.next = self.top
                    self.top = new_node
                    self.size += 1
                
                def pop(self):
                    if self.is_empty():
                        raise IndexError("Pop from an empty stack")
                    value = self.top.value
                    self.top = self.top.next
                    self.size -= 1
                    return value
                
                def peek(self):
                    if self.is_empty():
                        raise IndexError("Peek from an empty stack")
                    return self.top.value
            """
    },
    {
        "name": "Balanced Parentheses Check",
        "tags": ["stack", "string", "validation"],
        "description": "The Balanced Parentheses Check algorithm uses a stack to verify if an expression has balanced parentheses, brackets, and braces. It scans the expression from left to right, pushing opening delimiters onto a stack and popping when matching closing delimiters are encountered.",
        "complexity": {
            "time": "O(n)",
            "space": "O(n)"
        },
        "problem_patterns": [
            "Validating proper nesting of parentheses, brackets, and braces",
            "Checking syntax in expressions",
            "Problems requiring matching of opening and closing characters"
        ],
        "leetcode_indicators": [
            "Valid parentheses",
            "Check balanced brackets",
            "Expression validation"
        ],
        "implementation": """
            def is_balanced(expression):
                stack = []
                
                # Dictionary to map closing brackets to their opening counterparts
                brackets_map = {')': '(', '}': '{', ']': '['}
                
                # Scan the expression
                for char in expression:
                    # If it's an opening bracket, push to stack
                    if char in '({[':
                        stack.append(char)
                    # If it's a closing bracket
                    elif char in ')}]':
                        # If stack is empty or brackets don't match, it's not balanced
                        if not stack or stack.pop() != brackets_map[char]:
                            return False
                
                # If stack is empty, all brackets were matched
                return len(stack) == 0
            """
    },
    {
        "name": "Infix to Postfix Conversion",
        "tags": ["stack", "expression", "conversion"],
        "description": "The Infix to Postfix Conversion algorithm transforms an infix expression (standard mathematical notation with operators between operands) to postfix notation (operators follow their operands) using a stack to handle operator precedence and parentheses.",
        "complexity": {
            "time": "O(n)",
            "space": "O(n)"
        },
        "problem_patterns": [
            "Expression parsing and evaluation",
            "Compiler design problems",
            "Problems involving operator precedence"
        ],
        "leetcode_indicators": [
            "Expression evaluation",
            "Convert expression notation",
            "Calculator implementation"
        ],
        "implementation": """
            def infix_to_postfix(expression):
                # Define operator precedence
                precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
                
                # Initialize result and stack
                result = []
                stack = []
                
                # Process each character
                for char in expression:
                    # If character is an operand, add to result
                    if char.isalnum():
                        result.append(char)
                    # If character is an opening bracket, push to stack
                    elif char == '(':
                        stack.append(char)
                    # If character is a closing bracket, pop from stack until opening bracket
                    elif char == ')':
                        while stack and stack[-1] != '(':
                            result.append(stack.pop())
                        stack.pop()  # Remove the opening bracket
                    # If character is an operator
                    else:
                        # Pop operators with higher or equal precedence
                        while stack and stack[-1] != '(' and (stack[-1] in precedence) and (precedence.get(char, 0) <= precedence.get(stack[-1], 0)):
                            result.append(stack.pop())
                        stack.append(char)
                
                # Pop any remaining operators
                while stack:
                    result.append(stack.pop())
                
                # Join the result
                return ''.join(result)
            """
    }
    ]
    
    algorithms.extend(stack_algorithms)

    dp_algorithms = [
        {
            "name": "0/1 Knapsack",
            "tags": ["dynamic programming", "optimization", "combinatorial"],
            "description": "The 0/1 Knapsack problem is a problem in combinatorial optimization: given a set of items, each with a weight and a value, determine which items to include in a collection so that the total weight is less than or equal to a given limit and the total value is as large as possible.",
            "complexity": {
                "time": "O(n*W)",
                "space": "O(n*W)"
            },
            "problem_patterns": [
                "Resource allocation with constraints",
                "Item selection to maximize value with weight constraint",
                "Problems involving yes/no decisions for each item"
            ],
            "leetcode_indicators": [
                "Maximize value with weight constraint",
                "Problems involving subset selection with constraints",
                "Target sum with specific items"
            ],
            "implementation": """
                def knapsack_01(values, weights, capacity):
                    n = len(values)
                    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
                    
                    for i in range(1, n + 1):
                        for w in range(capacity + 1):
                            if weights[i-1] <= w:
                                dp[i][w] = max(
                                    values[i-1] + dp[i-1][w-weights[i-1]],  # Include item
                                    dp[i-1][w]  # Exclude item
                                )
                            else:
                                dp[i][w] = dp[i-1][w]  # Can't include, so exclude
                    
                    return dp[n][capacity]
                """
        },
        {
            "name": "Longest Common Subsequence",
            "tags": ["dynamic programming", "string algorithm", "sequence comparison"],
            "description": "The Longest Common Subsequence (LCS) algorithm finds the longest sequence that is present in both given sequences in the same order (not necessarily consecutive).",
            "complexity": {
                "time": "O(m*n)",
                "space": "O(m*n)"
            },
            "problem_patterns": [
                "String comparison and similarity",
                "Sequence alignment problems",
                "Edit distance variations"
            ],
            "leetcode_indicators": [
                "Find common subsequence between strings",
                "String similarity problems",
                "Problems involving sequence comparison",
                "Edit distance variations"
            ],
            "implementation": """
                def lcs(text1, text2):
                    m, n = len(text1), len(text2)
                    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
                    
                    for i in range(1, m + 1):
                        for j in range(1, n + 1):
                            if text1[i-1] == text2[j-1]:
                                dp[i][j] = dp[i-1][j-1] + 1
                            else:
                                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                    
                    return dp[m][n]
                """
        },
        {
            "name": "Coin Change",
            "tags": ["dynamic programming", "greedy", "optimization"],
            "description": "The Coin Change problem asks for the minimum number of coins needed to make a certain amount of change, given a set of coin denominations.",
            "complexity": {
                "time": "O(amount * n)",
                "space": "O(amount)"
            },
            "problem_patterns": [
                "Making change with minimum number of coins",
                "Problems involving combinations that sum to target",
                "Minimum resource allocation problems"
            ],
            "leetcode_indicators": [
                "Minimum coins to make change",
                "Ways to make sum with given numbers",
                "Problems involving counting combinations"
            ],
            "implementation": """
                def coin_change(coins, amount):
                    # Initialize dp array with amount+1 (representing infinity)
                    dp = [amount + 1] * (amount + 1)
                    dp[0] = 0
                    
                    for coin in coins:
                        for i in range(coin, amount + 1):
                            dp[i] = min(dp[i], dp[i - coin] + 1)
                    
                    return dp[amount] if dp[amount] <= amount else -1
                """
        }
    ]
    
    algorithms.extend(dp_algorithms)
    
    # String Algorithms
    string_algorithms = [
        {
            "name": "Knuth-Morris-Pratt (KMP)",
            "tags": ["string algorithm", "pattern matching", "substring search"],
            "description": "The Knuth-Morris-Pratt algorithm searches for occurrences of a 'pattern' within a main 'text' by employing the observation that when a mismatch occurs, the pattern itself contains sufficient information to determine where the next match could begin, thus bypassing re-examination of previously matched characters.",
            "complexity": {
                "time": "O(n + m)",
                "space": "O(m)"
            },
            "problem_patterns": [
                "Efficient substring search",
                "Pattern matching in strings",
                "Text processing problems"
            ],
            "leetcode_indicators": [
                "Find all occurrences of pattern in text",
                "String matching problems",
                "Problems requiring efficient substring search"
            ],
            "implementation": """
                def kmp_search(text, pattern):
                    if not pattern:
                        return 0  # Empty pattern matches at position 0
                    
                    # Preprocess: Compute the longest proper prefix which is also suffix array
                    lps = [0] * len(pattern)
                    compute_lps_array(pattern, lps)
                    
                    i, j = 0, 0  # i for text, j for pattern
                    results = []
                    
                    while i < len(text):
                        if pattern[j] == text[i]:
                            i += 1
                            j += 1
                        
                        if j == len(pattern):
                            results.append(i - j)  # Found a match
                            j = lps[j - 1]
                        elif i < len(text) and pattern[j] != text[i]:
                            if j != 0:
                                j = lps[j - 1]
                            else:
                                i += 1
                    
                    return results

                    def compute_lps_array(pattern, lps):
                        length = 0
                        i = 1
                        
                        while i < len(pattern):
                            if pattern[i] == pattern[length]:
                                length += 1
                                lps[i] = length
                                i += 1
                            else:
                                if length != 0:
                                    length = lps[length - 1]
                                else:
                                    lps[i] = 0
                                    i += 1
                    """
        },
        {
            "name": "Rabin-Karp",
            "tags": ["string algorithm", "pattern matching", "hashing"],
            "description": "The Rabin-Karp algorithm is a string-searching algorithm that uses hashing to find patterns in strings. It calculates a hash value for the pattern and for each possible substring of the text, then compares the hash values instead of comparing the strings character by character.",
            "complexity": {
                "time": "O(n + m)",
                "worst_time": "O(n*m)",
                "space": "O(1)"
            },
            "problem_patterns": [
                "Multiple pattern search",
                "Substring matching",
                "Plagiarism detection"
            ],
            "leetcode_indicators": [
                "String matching with hash function",
                "Multiple pattern search in text",
                "Substring search problems"
            ],
            "implementation": """
                def rabin_karp(text, pattern):
                    if not pattern:
                        return 0
                    
                    # Prime number for hash calculation
                    q = 101
                    
                    # Radix for the number system (ASCII)
                    d = 256
                    
                    m, n = len(pattern), len(text)
                    p = 0  # Hash value for pattern
                    t = 0  # Hash value for text
                    h = 1
                    results = []
                    
                    # Calculate h = d^(m-1) % q
                    for i in range(m - 1):
                        h = (h * d) % q
                    
                    # Calculate initial hash values
                    for i in range(m):
                        p = (d * p + ord(pattern[i])) % q
                        t = (d * t + ord(text[i])) % q
                    
                    # Slide pattern over text
                    for i in range(n - m + 1):
                        # Check hash values
                        if p == t:
                            # Check characters one by one
                            match = True
                            for j in range(m):
                                if text[i + j] != pattern[j]:
                                    match = False
                                    break
                            
                            if match:
                                results.append(i)
                        
                        # Calculate hash for next window
                        if i < n - m:
                            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % q
                            if t < 0:
                                t += q
                    
                    return results
                """
        },
        {
            "name": "Longest Palindromic Substring",
            "tags": ["string algorithm", "dynamic programming"],
            "description": "The Longest Palindromic Substring algorithm finds the longest substring within a string that is a palindrome (reads the same backward as forward).",
            "complexity": {
                "time": "O(n²)",
                "space": "O(1)"
            },
            "problem_patterns": [
                "Finding palindromes in strings",
                "String processing with symmetry",
                "Text analysis problems"
            ],
            "leetcode_indicators": [
                "Find longest palindrome in string",
                "Problems involving substring palindromes",
                "String symmetry problems"
            ],
            "implementation": """
                def longest_palindromic_substring(s):
                    if not s:
                        return ""
                    
                    start = 0
                    max_length = 1
                    
                    # Helper function to expand around center
                    def expand_around_center(left, right):
                        while left >= 0 and right < len(s) and s[left] == s[right]:
                            left -= 1
                            right += 1
                        return right - left - 1
                    
                    for i in range(len(s)):
                        # Expand for odd length palindromes
                        odd_length = expand_around_center(i, i)
                        
                        # Expand for even length palindromes
                        even_length = expand_around_center(i, i + 1)
                        
                        # Update if longer palindrome found
                        length = max(odd_length, even_length)
                        if length > max_length:
                            max_length = length
                            start = i - (length - 1) // 2
                    
                    return s[start:start + max_length]
                """
        }
    ]
    
    algorithms.extend(string_algorithms)
    
    tree_algorithms = [
        {
            "name": "Binary Tree Traversal",
            "tags": ["tree algorithm", "data structure", "traversal"],
            "description": "Binary Tree Traversal algorithms systematically visit each node in a binary tree. The three most common traversal methods are in-order (left-root-right), pre-order (root-left-right), and post-order (left-right-root).",
            "complexity": {
                "time": "O(n)",
                "space": "O(h)"
            },
            "problem_patterns": [
                "Tree processing in specific orders",
                "Converting tree to array representations",
                "Tree validation problems"
            ],
            "leetcode_indicators": [
                "Tree traversal problems",
                "Convert tree to array",
                "Problems requiring specific node visit order"
            ],
            "implementation": """
                class TreeNode:
                    def __init__(self, val=0, left=None, right=None):
                        self.val = val
                        self.left = left
                        self.right = right

                # In-order traversal
                def inorder_traversal(root):
                    result = []
                    
                    def dfs(node):
                        if not node:
                            return
                        dfs(node.left)
                        result.append(node.val)
                        dfs(node.right)
                    
                    dfs(root)
                    return result

                # Pre-order traversal
                def preorder_traversal(root):
                    result = []
                    
                    def dfs(node):
                        if not node:
                            return
                        result.append(node.val)
                        dfs(node.left)
                        dfs(node.right)
                    
                    dfs(root)
                    return result

                # Post-order traversal
                def postorder_traversal(root):
                    result = []
                    
                    def dfs(node):
                        if not node:
                            return
                        dfs(node.left)
                        dfs(node.right)
                        result.append(node.val)
                    
                    dfs(root)
                    return result
                """
        },
        {
            "name": "Binary Search Tree Operations",
            "tags": ["tree algorithm", "binary search tree", "data structure"],
            "description": "Binary Search Tree (BST) operations include insertion, deletion, and searching in a tree where for each node, all elements in the left subtree are less than the node's value, and all elements in the right subtree are greater.",
            "complexity": {
                "time": "O(h)",
                "space": "O(h)"
            },
            "problem_patterns": [
                "Efficient data structures for sorted data",
                "Problems requiring ordered data operations",
                "Tree construction and modification"
            ],
            "leetcode_indicators": [
                "Binary search tree problems",
                "Tree with ordered property",
                "Problems involving tree insertion/deletion"
            ],
            "implementation": """
                class TreeNode:
                        def __init__(self, val=0, left=None, right=None):
                            self.val = val
                            self.left = left
                            self.right = right

                        # Insert value into BST
                        def insert(root, val):
                            if not root:
                                return TreeNode(val)
                                        
                            if val < root.val:
                                root.left = insert(root.left, val)
                            else:
                                root.right = insert(root.right, val)
                                        
                            return root

                        # Search for value in BST
                        def search(root, val):
                            if not root or root.val == val:
                                return root
                                        
                            if val < root.val:
                                return search(root.left, val)
                            else:
                                return search(root.right, val)

                        # Delete value from BST
                        def delete(root, val):
                            if not root:
                                return None
                
                            if val < root.val:
                                root.left = delete(root.left, val)
                            elif val > root.val:
                                root.right = delete(root.right, val)
                            else:
                                # Node with only one child or no child
                            if not root.left:
                                return root.right
                            elif not root.right:
                                return root.left
                    
                            # Node with two children
                            # Get inorder successor (smallest in right subtree)
                            temp = find_min(root.right)
                            root.val = temp.val
                            root.right = delete(root.right, temp.val)
                
                            return root

                        def find_min(node):
                            current = node
                            while current.left:
                                current = current.left
                            return current
                        """
        },
        {
            "name": "Lowest Common Ancestor",
            "tags": ["tree algorithm", "binary tree", "ancestor finding"],
            "description": "The Lowest Common Ancestor (LCA) algorithm finds the lowest node in a tree that has both given nodes as descendants. A node can be a descendant of itself.",
            "complexity": {
                "time": "O(n)",
                "space": "O(h)"
            },
            "problem_patterns": [
                "Finding common ancestors in trees",
                "Relationship problems in hierarchical structures",
                "Tree navigation problems"
            ],
            "leetcode_indicators": [
                "Lowest common ancestor problems",
                "Tree node relationship questions",
                "Problems involving finding a common parent"
            ],
            "implementation": """
                class TreeNode:
                    def __init__(self, val=0, left=None, right=None):
                        self.val = val
                        self.left = left
                        self.right = right

                def lowest_common_ancestor(root, p, q):
                    # Base case
                    if not root or root == p or root == q:
                        return root
                    
                    # Look for p and q in left and right subtrees
                    left = lowest_common_ancestor(root.left, p, q)
                    right = lowest_common_ancestor(root.right, p, q)
                    
                    # If both p and q are found, this node is the LCA
                    if left and right:
                        return root
                    
                    # Otherwise, return the non-null value
                    return left if left else right
                """
        }
    ]
    
    algorithms.extend(tree_algorithms)

    queue_algorithms = [
    {
        "name": "Queue Implementation",
        "tags": ["queue", "data structure", "FIFO"],
        "description": "The Queue data structure follows First-In-First-Out (FIFO) principle. It supports two primary operations: enqueue (adding an element to the rear) and dequeue (removing the front element). Queues can be implemented using arrays, linked lists, or a combination of stacks.",
        "complexity": {
            "time": "O(1) for enqueue/dequeue operations",
            "space": "O(n)"
        },
        "problem_patterns": [
            "Problems requiring first-in-first-out processing",
            "Breadth-first search",
            "Task scheduling",
            "Buffer management"
        ],
        "leetcode_indicators": [
            "Level order traversal",
            "BFS problems",
            "First-come-first-serve processing"
        ],
        "implementation": """
            # Array-based queue implementation (using a Python list)
            class Queue:
                def __init__(self):
                    self.items = []
                
                def is_empty(self):
                    return len(self.items) == 0
                
                def enqueue(self, item):
                    self.items.append(item)
                
                def dequeue(self):
                    if self.is_empty():
                        raise IndexError("Dequeue from an empty queue")
                    return self.items.pop(0)
                
                def peek(self):
                    if self.is_empty():
                        raise IndexError("Peek from an empty queue")
                    return self.items[0]
                
                def size(self):
                    return len(self.items)

            # Linked list-based queue implementation
            class Node:
                def __init__(self, value):
                    self.value = value
                    self.next = None

            class LinkedQueue:
                def __init__(self):
                    self.front = None
                    self.rear = None
                    self.size = 0
                
                def is_empty(self):
                    return self.front is None
                
                def enqueue(self, value):
                    new_node = Node(value)
                    
                    if self.is_empty():
                        self.front = new_node
                    else:
                        self.rear.next = new_node
                    
                    self.rear = new_node
                    self.size += 1
                
                def dequeue(self):
                    if self.is_empty():
                        raise IndexError("Dequeue from an empty queue")
                    
                    value = self.front.value
                    self.front = self.front.next
                    
                    # If queue becomes empty, update rear
                    if self.front is None:
                        self.rear = None
                    
                    self.size -= 1
                    return value
                
                def peek(self):
                    if self.is_empty():
                        raise IndexError("Peek from an empty queue")
                    return self.front.value
            """
    },
    {
        "name": "Circular Queue Implementation",
        "tags": ["queue", "circular", "data structure"],
        "description": "A Circular Queue (also called Ring Buffer) is an enhancement of the regular queue that efficiently uses space by wrapping around to the beginning when it reaches the end of the allocated space. It maintains two pointers: front and rear, and uses modulo arithmetic to handle the wrap-around.",
        "complexity": {
            "time": "O(1) for enqueue/dequeue operations",
            "space": "O(n)"
        },
        "problem_patterns": [
            "Fixed-size buffer management",
            "Stream processing",
            "Problems requiring circular data structures",
            "Round-robin scheduling"
        ],
        "leetcode_indicators": [
            "Design circular queue",
            "Circular buffer",
            "Problems involving wraparound indexing"
        ],
        "implementation": """
            class CircularQueue:
                def __init__(self, capacity):
                    self.capacity = capacity
                    self.queue = [None] * capacity
                    self.front = self.size = 0
                    self.rear = capacity - 1
                
                def is_full(self):
                    return self.size == self.capacity
                
                def is_empty(self):
                    return self.size == 0
                
                def enqueue(self, item):
                    if self.is_full():
                        raise IndexError("Queue is full")
                    
                    self.rear = (self.rear + 1) % self.capacity
                    self.queue[self.rear] = item
                    self.size += 1
                
                def dequeue(self):
                    if self.is_empty():
                        raise IndexError("Dequeue from an empty queue")
                    
                    item = self.queue[self.front]
                    self.front = (self.front + 1) % self.capacity
                    self.size -= 1
                    return item
                
                def peek(self):
                    if self.is_empty():
                        raise IndexError("Peek from an empty queue")
                    return self.queue[self.front]
            """
    },
    {
        "name": "Priority Queue Implementation",
        "tags": ["queue", "priority", "heap", "data structure"],
        "description": "A Priority Queue is an abstract data type similar to a regular queue but where each element has a priority. Elements with higher priority are dequeued before elements with lower priority. It can be implemented using a heap, a binary search tree, or an ordered array.",
        "complexity": {
            "time": "O(log n) for insertion/deletion with heap implementation",
            "space": "O(n)"
        },
        "problem_patterns": [
            "Problems requiring elements to be processed based on priority",
            "Scheduling algorithms",
            "Graph algorithms like Dijkstra's",
            "Huffman coding"
        ],
        "leetcode_indicators": [
            "Top-k elements",
            "Minimum cost problems",
            "Scheduling problems",
            "Merge k sorted lists"
        ],
        "implementation": """
            import heapq

            # Priority Queue using Python's heapq (min-heap)
            class PriorityQueue:
                def __init__(self):
                    self.elements = []
                
                def is_empty(self):
                    return len(self.elements) == 0
                
                def put(self, item, priority):
                    # For min-heap, use priority as the first element
                    heapq.heappush(self.elements, (priority, item))
                
                def get(self):
                    if self.is_empty():
                        raise IndexError("Dequeue from an empty priority queue")
                    return heapq.heappop(self.elements)[1]
                
                def peek(self):
                    if self.is_empty():
                        raise IndexError("Peek from an empty priority queue")
                    return self.elements[0][1]
                
                def size(self):
                    return len(self.elements)

            # Custom implementation of a Priority Queue using a binary heap
            class CustomPriorityQueue:
                def __init__(self, is_min_heap=True):
                    self.heap = []
                    self.is_min_heap = is_min_heap
                
                def size(self):
                    return len(self.heap)
                
                def is_empty(self):
                    return self.size() == 0
                
                def get_parent(self, i):
                    return (i - 1) // 2
                
                def get_left_child(self, i):
                    return 2 * i + 1
                
                def get_right_child(self, i):
                    return 2 * i + 2
                
                def has_parent(self, i):
                    return self.get_parent(i) >= 0
                
                def has_left_child(self, i):
                    return self.get_left_child(i) < self.size()
                
                def has_right_child(self, i):
                    return self.get_right_child(i) < self.size()
                
                def compare(self, a, b):
                    if self.is_min_heap:
                        return a < b
                    return a > b
                
                def swap(self, i, j):
                    self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
                
                def peek(self):
                    if self.is_empty():
                        raise IndexError("Peek from an empty priority queue")
                    return self.heap[0][1]
                
                def push(self, priority, item):
                    self.heap.append((priority, item))
                    self.heapify_up(self.size() - 1)
                
                def pop(self):
                    if self.is_empty():
                        raise IndexError("Pop from an empty priority queue")
                    
                    item = self.heap[0][1]
                    self.heap[0] = self.heap[-1]
                    self.heap.pop()
                    if self.size() > 0:
                        self.heapify_down(0)
                    return item
                
                def heapify_up(self, index):
                    while (self.has_parent(index) and 
                        self.compare(self.heap[index][0], self.heap[self.get_parent(index)][0])):
                        self.swap(index, self.get_parent(index))
                        index = self.get_parent(index)
                
                def heapify_down(self, index):
                    smallest = index
                    
                    if (self.has_left_child(index) and 
                        self.compare(self.heap[self.get_left_child(index)][0], self.heap[smallest][0])):
                        smallest = self.get_left_child(index)
                    
                    if (self.has_right_child(index) and 
                        self.compare(self.heap[self.get_right_child(index)][0], self.heap[smallest][0])):
                        smallest = self.get_right_child(index)
                    
                    if smallest != index:
                        self.swap(index, smallest)
                        self.heapify_down(smallest)
            """
    }
]

    algorithms.extend(queue_algorithms)

    # Hash Table Algorithms
    hash_table_algorithms = [
    {
        "name": "Hash Table Implementation",
        "tags": ["hash table", "data structure", "key-value"],
        "description": "A Hash Table is a data structure that implements an associative array abstract data type, a structure that can map keys to values. It uses a hash function to compute an index into an array of buckets or slots, from which the desired value can be found.",
        "complexity": {
            "time": "O(1) average for insert/search/delete, O(n) worst case",
            "space": "O(n)"
        },
        "problem_patterns": [
            "Fast lookup, insertion, and deletion",
            "Caching and memoization",
            "Counting occurrences",
            "Two-sum type problems"
        ],
        "leetcode_indicators": [
            "Two sum",
            "Problems requiring fast lookup",
            "Frequency counting",
            "Symbol tables"
        ],
        "implementation": """
            class HashNode:
                def __init__(self, key, value):
                    self.key = key
                    self.value = value
                    self.next = None

            class HashTable:
                def __init__(self, size=10):
                    # Initialize the hash table with empty buckets
                    self.size = size
                    self.buckets = [None] * size
                    self.count = 0
                
                def _hash(self, key):
                    # Simple hash function
                    if isinstance(key, int):
                        return key % self.size
                    # If key is a string, sum the ASCII values
                    if isinstance(key, str):
                        total = 0
                        for char in key:
                            total += ord(char)
                        return total % self.size
                    # For other types, use their hash
                    return hash(key) % self.size
                
                def set(self, key, value):
                    # Find the bucket
                    index = self._hash(key)
                    node = self.buckets[index]
                    
                    # Check if key already exists
                    while node:
                        if node.key == key:
                            node.value = value  # Update value
                            return
                        node = node.next
                    
                    # Key not found, create new node
                    new_node = HashNode(key, value)
                    new_node.next = self.buckets[index]
                    self.buckets[index] = new_node
                    self.count += 1
                    
                    # Resize if load factor exceeds threshold
                    if self.count > self.size * 0.7:
                        self._resize(self.size * 2)
                
                def get(self, key):
                    # Find the bucket
                    index = self._hash(key)
                    node = self.buckets[index]
                    
                    # Look for the key
                    while node:
                        if node.key == key:
                            return node.value
                        node = node.next
                    
                    # Key not found
                    return None
                
                def delete(self, key):
                    # Find the bucket
                    index = self._hash(key)
                    node = self.buckets[index]
                    prev = None
                    
                    # Look for the key
                    while node and node.key != key:
                        prev = node
                        node = node.next
                    
                    # If key found
                    if node:
                        if prev:
                            prev.next = node.next
                        else:
                            self.buckets[index] = node.next
                        self.count -= 1
                        return True
                    
                    # Key not found
                    return False
                
                def contains(self, key):
                    return self.get(key) is not None
                
                def _resize(self, new_size):
                    old_buckets = self.buckets
                    self.size = new_size
                    self.buckets = [None] * new_size
                    self.count = 0
                    
                    # Rehash all entries
                    for head in old_buckets:
                        node = head
                        while node:
                            self.set(node.key, node.value)
                            node = node.next
            """
    },
    {
        "name": "Collision Resolution with Chaining",
        "tags": ["hash table", "collision resolution", "linked list"],
        "description": "Collision Resolution with Chaining is a technique used in hash tables to handle multiple keys that hash to the same index. In chaining, each bucket (array index) contains a linked list of all key-value pairs whose keys hash to that index, allowing multiple entries to exist at the same location.",
        "complexity": {
            "time": "O(1 + α) average for operations, where α is the load factor",
            "space": "O(n + m) where n is the number of entries and m is the number of buckets"
        },
        "problem_patterns": [
            "Implementing hash tables with predictable performance",
            "Handling hash collisions",
            "Problems requiring separate chaining"
        ],
        "leetcode_indicators": [
            "Design hash map",
            "Design hash set",
            "Problems involving custom hash table implementation"
        ],
        "implementation": """
            class HashTableWithChaining:
                def __init__(self, size=10):
                    self.size = size
                    self.table = [[] for _ in range(size)]  # List of lists for chaining
                
                def _hash(self, key):
                    # Simple hash function
                    if isinstance(key, int):
                        return key % self.size
                    # For strings, sum the ASCII values
                    if isinstance(key, str):
                        return sum(ord(char) for char in key) % self.size
                    # For other types, use their hash
                    return hash(key) % self.size
                
                def insert(self, key, value):
                    # Find the bucket
                    index = self._hash(key)
                    bucket = self.table[index]
                    
                    # Check if key already exists
                    for i, (k, v) in enumerate(bucket):
                        if k == key:
                            bucket[i] = (key, value)  # Update value
                            return
                    
                    # Key not found, add new entry
                    bucket.append((key, value))
                
                def get(self, key):
                    # Find the bucket
                    index = self._hash(key)
                    bucket = self.table[index]
                    
                    # Look for the key
                    for k, v in bucket:
                        if k == key:
                            return v
                    
                    # Key not found
                    return None
                
                def remove(self, key):
                    # Find the bucket
                    index = self._hash(key)
                    bucket = self.table[index]
                    
                    # Look for the key and remove it
                    for i, (k, v) in enumerate(bucket):
                        if k == key:
                            del bucket[i]
                            return True
                    
                    # Key not found
                    return False
                
                def display(self):
                    for i, bucket in enumerate(self.table):
                        if bucket:  # Only show non-empty buckets
                            print(f"Bucket {i}: {bucket}")
            """
    },
    {
        "name": "Open Addressing (Linear Probing)",
        "tags": ["hash table", "collision resolution", "open addressing"],
        "description": "Open Addressing is a collision resolution technique where all elements are stored in the hash table itself (no external data structures). Linear Probing is one method of open addressing where, if a collision occurs, we sequentially search for the next available slot.",
        "complexity": {
            "time": "O(1) average for operations with low load factor, O(n) worst case",
            "space": "O(n)"
        },
        "problem_patterns": [
            "Implementing memory-efficient hash tables",
            "Problems requiring cache efficiency",
            "Situations where chaining is impractical"
        ],
        "leetcode_indicators": [
            "Design hash map with space constraints",
            "Problems involving linear probing",
            "Cache-friendly hash table design"
        ],
        "implementation": """
            class HashTableWithLinearProbing:
                def __init__(self, size=10):
                    self.size = size
                    self.keys = [None] * size
                    self.values = [None] * size
                    self.tombstone = object()  # Special marker for deleted entries
                    self.count = 0
                
                def _hash(self, key):
                    # Simple hash function
                    if isinstance(key, int):
                        return key % self.size
                    # For strings, sum the ASCII values
                    if isinstance(key, str):
                        return sum(ord(char) for char in key) % self.size
                    # For other types, use their hash
                    return hash(key) % self.size
                
                def _get_index(self, key):
                    # Find the position for a key using linear probing
                    start_index = self._hash(key)
                    
                    # Linear probe until we find the key, an empty slot, or visit all positions
                    for i in range(self.size):
                        index = (start_index + i
                        ) % self.size
                        
                        # Found the key
                        if self.keys[index] == key:
                            return index
                        
                        # Found an empty slot
                        if self.keys[index] is None:
                            return -1
                    
                    # Hash table is full and key not found
                    return -1
                
                def _find_slot(self, key):
                    # Find the position to insert a key using linear probing
                    start_index = self._hash(key)
                    
                    # Linear probe until we find the key, an empty slot, or a tombstone
                    for i in range(self.size):
                        index = (start_index + i) % self.size
                        
                        # Found the key
                        if self.keys[index] == key:
                            return index
                        
                        # Found an empty slot or tombstone
                        if self.keys[index] is None or self.keys[index] is self.tombstone:
                            return index
                    
                    # Hash table is full
                    return -1
                
                def put(self, key, value):
                    # Don't allow None as a key
                    if key is None:
                        raise ValueError("None is not allowed as a key")
                    
                    # If load factor is too high, resize
                    if self.count >= self.size * 0.7:
                        self._resize(self.size * 2)
                    
                    # Find slot for insertion
                    index = self._find_slot(key)
                    
                    # If hash table is full
                    if index == -1:
                        self._resize(self.size * 2)
                        index = self._find_slot(key)
                    
                    # Check if this is a new entry
                    is_new = self.keys[index] is None or self.keys[index] is self.tombstone
                    
                    # Insert key-value pair
                    self.keys[index] = key
                    self.values[index] = value
                    
                    # Increment count for new entries
                    if is_new:
                        self.count += 1
                
                def get(self, key):
                    index = self._get_index(key)
                    
                    # Key not found
                    if index == -1:
                        return None
                    
                    # Return value
                    return self.values[index]
                
                def remove(self, key):
                    index = self._get_index(key)
                    
                    # Key not found
                    if index == -1:
                        return False
                    
                    # Mark as deleted with tombstone
                    self.keys[index] = self.tombstone
                    self.values[index] = None
                    self.count -= 1
                    return True
                
                def _resize(self, new_size):
                    old_keys = self.keys
                    old_values = self.values
                    
                    # Create new arrays
                    self.size = new_size
                    self.keys = [None] * new_size
                    self.values = [None] * new_size
                    self.count = 0
                    
                    # Rehash all entries
                    for i in range(len(old_keys)):
                        if old_keys[i] is not None and old_keys[i] is not self.tombstone:
                            self.put(old_keys[i], old_values[i])
            """
    }
]
    
    algorithms.extend(hash_table_algorithms)


    trie_algorithm = {
    "name": "Trie (Prefix Tree)",
    "tags": ["data structure", "tree", "string", "prefix", "search"],
    "description": "A Trie is a tree-like data structure used to store a dynamic set of strings. Tries are efficient for prefix-based operations and are commonly used for fast retrieval of keys in a dataset of strings. Unlike a binary search tree, no node in the trie stores the key associated with that node; instead, its position in the tree defines the key with which it is associated.",
    "complexity": {
        "time": "O(m) for insert/search/delete where m is the length of the key",
        "space": "O(n * m) where n is the number of keys and m is the key length"
    },
    "problem_patterns": [
        "Dictionary implementations",
        "Prefix searching",
        "Autocomplete systems",
        "Spell checkers",
        "IP routing (longest prefix matching)"
    ],
    "leetcode_indicators": [
        "Word dictionary implementation",
        "Problems involving prefix matching",
        "Add and Search Word",
        "Implement Trie (Prefix Tree)",
        "Word Search II"
    ],
    "implementation": """
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.is_end_of_word = False

        class Trie:
            def __init__(self):
                self.root = TrieNode()
            
            def insert(self, word):
                node = self.root
                for char in word:
                    if char not in node.children:
                        node.children[char] = TrieNode()
                    node = node.children[char]
                node.is_end_of_word = True
            
            def search(self, word):
                node = self.root
                for char in word:
                    if char not in node.children:
                        return False
                    node = node.children[char]
                return node.is_end_of_word
            
            def starts_with(self, prefix):
                node = self.root
                for char in prefix:
                    if char not in node.children:
                        return False
                    node = node.children[char]
                return True
            
            def delete(self, word):
                def _delete_helper(node, word, index):
                    # If we've reached the end of the word
                    if index == len(word):
                        # This node is no longer the end of a word
                        if node.is_end_of_word:
                            node.is_end_of_word = False
                            
                        # Return True if this node can be deleted (has no children and is not end of another word)
                        return len(node.children) == 0 and not node.is_end_of_word
                    
                    char = word[index]
                    if char not in node.children:
                        return False  # Word not in trie
                    
                    should_delete_child = _delete_helper(node.children[char], word, index + 1)
                    
                    # If child can be deleted, remove it
                    if should_delete_child:
                        del node.children[char]
                        
                    # Current node can be deleted if it has no children and is not end of another word
                    return len(node.children) == 0 and not node.is_end_of_word
                
                _delete_helper(self.root, word, 0)
        """
    }

    segment_tree_algorithm = {
    "name": "Segment Tree",
    "tags": ["data structure", "tree", "range queries", "interval tree"],
    "description": "A Segment Tree is a tree data structure for storing intervals or segments. It allows querying which of the stored segments contain a given point. It is, in principle, a static structure; that is, it's a structure that cannot be modified once it's built. A segment tree for a set of n intervals uses O(n log n) storage and can be built in O(n log n) time. Segment trees support range queries and updates in O(log n) time.",
    "complexity": {
        "time": "O(n log n) to build, O(log n) for range query and update",
        "space": "O(n)"
    },
    "problem_patterns": [
        "Range sum/min/max queries",
        "Range update operations",
        "Problems requiring efficient interval operations",
        "Finding the minimum/maximum in a range",
        "Counting number of elements in a range"
    ],
    "leetcode_indicators": [
        "Range sum query",
        "Finding minimum/maximum in a range",
        "Problems involving interval modifications and queries",
        "Problems requiring efficient range operations"
    ],
    "implementation": """
        class SegmentTree:
            def __init__(self, arr):
                self.n = len(arr)
                # The size of the segment tree array
                self.tree = [0] * (4 * self.n)
                if self.n > 0:
                    self._build_tree(arr, 0, 0, self.n - 1)
            
            def _build_tree(self, arr, node_idx, start, end):
                # Leaf node
                if start == end:
                    self.tree[node_idx] = arr[start]
                    return
                
                mid = (start + end) // 2
                # Build left subtree
                self._build_tree(arr, 2 * node_idx + 1, start, mid)
                # Build right subtree
                self._build_tree(arr, 2 * node_idx + 2, mid + 1, end)
                # Internal node will have the sum of both its children
                self.tree[node_idx] = self.tree[2 * node_idx + 1] + self.tree[2 * node_idx + 2]
            
            def query(self, start, end):
                if start < 0 or end >= self.n or start > end:
                    raise ValueError("Invalid range")
                return self._query(0, 0, self.n - 1, start, end)
            
            def _query(self, node_idx, node_start, node_end, query_start, query_end):
                # If segment of this node is completely outside the query range
                if query_end < node_start or query_start > node_end:
                    return 0
                
                # If segment of this node is completely inside the query range
                if node_start >= query_start and node_end <= query_end:
                    return self.tree[node_idx]
                
                # If segment of this node is partially inside and partially outside the query range
                mid = (node_start + node_end) // 2
                left_sum = self._query(2 * node_idx + 1, node_start, mid, query_start, query_end)
                right_sum = self._query(2 * node_idx + 2, mid + 1, node_end, query_start, query_end)
                return left_sum + right_sum
            
            def update(self, index, value):
                if index < 0 or index >= self.n:
                    raise ValueError("Invalid index")
                self._update(0, 0, self.n - 1, index, value)
            
            def _update(self, node_idx, node_start, node_end, index, value):
                # Leaf node: update the value
                if node_start == node_end:
                    self.tree[node_idx] = value
                    return
                
                mid = (node_start + node_end) // 2
                if index <= mid:
                    # Update left subtree
                    self._update(2 * node_idx + 1, node_start, mid, index, value)
                else:
                    # Update right subtree
                    self._update(2 * node_idx + 2, mid + 1, node_end, index, value)
                
                # Update the current node based on its children
                self.tree[node_idx] = self.tree[2 * node_idx + 1] + self.tree[2 * node_idx + 2]
        """
    }

    union_find_algorithm = {
    "name": "Union Find (Disjoint Set)",
    "tags": ["data structure", "graph", "disjoint set", "connectivity"],
    "description": "Union Find is a data structure that tracks a set of elements partitioned into a number of disjoint (non-overlapping) subsets. It provides near-constant-time operations to add new sets, merge existing sets, and determine whether elements are in the same set. Union Find is particularly useful for Kruskal's algorithm and for tracking connected components in graphs.",
    "complexity": {
        "time": "O(α(n)) for find and union operations, where α(n) is the inverse Ackermann function",
        "space": "O(n)"
    },
    "problem_patterns": [
        "Finding connected components in a graph",
        "Detecting cycles in an undirected graph",
        "Minimum spanning tree algorithms (Kruskal's)",
        "Network connectivity",
        "Least common ancestor in trees"
    ],
    "leetcode_indicators": [
        "Problems involving graph connectivity",
        "Friend circles",
        "Number of connected components",
        "Redundant connection",
        "Account merging"
    ],
    "implementation": """
        class UnionFind:
            def __init__(self, n):
                # Initially, each element is its own parent/representative
                self.parent = list(range(n))
                # Rank is used for union by rank optimization
                self.rank = [0] * n
                # Number of disjoint sets
                self.count = n
            
            def find(self, x):
                # Path compression: make every examined node point directly to the root
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                # Find the roots of x and y
                root_x = self.find(x)
                root_y = self.find(y)
                
                # If x and y are already in the same set
                if root_x == root_y:
                    return False
                
                # Union by rank: attach smaller rank tree under root of higher rank tree
                if self.rank[root_x] < self.rank[root_y]:
                    self.parent[root_x] = root_y
                elif self.rank[root_x] > self.rank[root_y]:
                    self.parent[root_y] = root_x
                else:
                    # If ranks are same, make one as root and increment its rank
                    self.parent[root_y] = root_x
                    self.rank[root_x] += 1
                
                # Decrease the count of disjoint sets
                self.count -= 1
                return True
            
            def is_connected(self, x, y):
                return self.find(x) == self.find(y)
            
            def get_count(self):
                return self.count
        """
    }

    advanced_data_structures = [
        trie_algorithm,
        segment_tree_algorithm,
        union_find_algorithm
    ]
    
    algorithms.extend(advanced_data_structures)
    
    a_star_algorithm = {
    "name": "A* Search Algorithm",
    "tags": ["graph algorithm", "pathfinding", "heuristic", "search"],
    "description": "A* (pronounced 'A star') is a pathfinding algorithm that finds the shortest path between two nodes. It uses a heuristic function to guide the search, making it more efficient than Dijkstra's algorithm in many cases. A* evaluates nodes by combining the cost to reach the node and the estimated cost to reach the goal. It's widely used in games, robotics, and navigation systems.",
    "complexity": {
        "time": "O(E) in the worst case, where E is the number of edges, but typically much better with a good heuristic",
        "space": "O(V), where V is the number of vertices"
    },
    "problem_patterns": [
        "Shortest path finding with heuristics",
        "Maze solving",
        "Navigation in games and robotics",
        "Path planning with obstacles",
        "Routing algorithms with additional constraints"
    ],
    "leetcode_indicators": [
        "Shortest path problems with specific constraints",
        "Problems requiring path optimization with heuristics",
        "Grid-based pathfinding"
    ],
    "implementation": """
        import heapq

        def a_star(graph, start, goal, heuristic):
            # Open set is a priority queue of nodes to explore
            # Each element is (f_score, node) where f_score = g_score + heuristic
            # g_score is the cost from start to node
            # f_score is the estimated cost from start to goal through node
            open_set = [(0 + heuristic(start, goal), start)]
            
            # g_score[n] is the cost of the cheapest path from start to n
            g_score = {start: 0}
            
            # came_from[n] is the node immediately preceding n on the cheapest path
            came_from = {}
            
            while open_set:
                # Get the node with lowest f_score
                current_f, current = heapq.heappop(open_set)
                
                # If we've reached the goal, reconstruct the path
                if current == goal:
                    path = []
                    while current in came_from:
                        path.append(current)
                        current = came_from[current]
                    path.append(start)
                    path.reverse()
                    return path
                
                # Explore neighbors
                for neighbor, weight in graph[current].items():
                    # Tentative g_score is the cost from start to neighbor through current
                    tentative_g_score = g_score[current] + weight
                    
                    # If this path to neighbor is better than any previous one, record it
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, neighbor))
            
            # If we get here, there's no path from start to goal
            return None

        # Example heuristic for a grid-based graph (Manhattan distance)
        def manhattan_distance(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        """
    }

    bellman_ford_algorithm = {
    "name": "Bellman-Ford Algorithm",
    "tags": ["graph algorithm", "shortest path", "negative weight", "dynamic programming"],
    "description": "The Bellman-Ford algorithm finds the shortest paths from a single source vertex to all other vertices in a weighted graph. Unlike Dijkstra's algorithm, Bellman-Ford can handle graphs with negative weight edges. It also detects negative weight cycles, which are cycles whose edges sum to a negative value. The algorithm relaxes all edges V-1 times, where V is the number of vertices.",
    "complexity": {
        "time": "O(V * E) where V is the number of vertices and E is the number of edges",
        "space": "O(V)"
    },
    "problem_patterns": [
        "Finding shortest paths with negative weights",
        "Detecting negative cycles in graphs",
        "Network routing",
        "Currency exchange and arbitrage detection",
        "Dynamic programming on graphs"
    ],
    "leetcode_indicators": [
        "Shortest path problems with negative weights",
        "Problems requiring detection of negative cycles",
        "Network delay with possible negative costs"
    ],
    "implementation": """
        def bellman_ford(graph, source):
            # Initialize distances
            distances = {vertex: float('infinity') for vertex in graph}
            distances[source] = 0
            
            # Store predecessors for path reconstruction
            predecessors = {vertex: None for vertex in graph}
            
            # Get all vertices
            vertices = list(graph.keys())
            
            # Relax all edges |V| - 1 times
            for _ in range(len(vertices) - 1):
                for u in graph:
                    for v, weight in graph[u].items():
                        if distances[u] != float('infinity') and distances[u] + weight < distances[v]:
                            distances[v] = distances[u] + weight
                            predecessors[v] = u
            
            # Check for negative weight cycles
            # If we can still relax edges, then we have a negative cycle
            negative_cycle = False
            for u in graph:
                for v, weight in graph[u].items():
                    if distances[u] != float('infinity') and distances[u] + weight < distances[v]:
                        negative_cycle = True
                        break
            
            return distances, predecessors, negative_cycle

        def reconstruct_path(predecessors, source, destination):
            path = []
            current = destination
            
            while current != source:
                if current is None:
                    return None  # No path exists
                path.append(current)
                current = predecessors[current]
            
            path.append(source)
            path.reverse()
            return path
        """ 
    }

    floyd_warshall_algorithm = {
    "name": "Floyd-Warshall Algorithm",
    "tags": ["graph algorithm", "all-pairs shortest path", "dynamic programming"],
    "description": "The Floyd-Warshall algorithm finds the shortest paths between all pairs of vertices in a weighted graph. It works with positive or negative edge weights and can detect negative cycles. The algorithm uses dynamic programming to gradually improve an estimate on the shortest path between two vertices by including intermediate vertices.",
    "complexity": {
        "time": "O(V³) where V is the number of vertices",
        "space": "O(V²)"
    },
    "problem_patterns": [
        "Finding shortest paths between all pairs of vertices",
        "Transitive closure of directed graphs",
        "Detecting negative cycles",
        "Problems requiring the shortest distance between any two points"
    ],
    "leetcode_indicators": [
        "All-pairs shortest path problems",
        "Problems requiring shortest paths between multiple sources and destinations",
        "Network connectivity with shortest path requirement",
        "Graph diameter calculation"
    ],
    "implementation": """
        def floyd_warshall(graph):
            # Number of vertices
            n = len(graph)
            
            # Initialize distance matrix
            # dist[i][j] will be the shortest distance from vertex i to j
            dist = [row[:] for row in graph]  # Create a copy of the graph
            
            # Initialize path reconstruction matrix
            # next_vertex[i][j] will be the next vertex on the path from i to j
            next_vertex = [[j if graph[i][j] != float('inf') and i != j else None for j in range(n)] for i in range(n)]
            
            # Main algorithm: consider each vertex as an intermediate
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        if dist[i][k] != float('inf') and dist[k][j] != float('inf') and dist[i][k] + dist[k][j] < dist[i][j]:
                            dist[i][j] = dist[i][k] + dist[k][j]
                            next_vertex[i][j] = next_vertex[i][k]
            
            # Check for negative cycles
            has_negative_cycle = False
            for i in range(n):
                if dist[i][i] < 0:
                    has_negative_cycle = True
                    break
            
            return dist, next_vertex, has_negative_cycle

        def reconstruct_path(next_vertex, u, v):
            # If there's no path
            if next_vertex[u][v] is None:
                return []
            
            path = [u]
            while u != v:
                u = next_vertex[u][v]
                path.append(u)
            
            return path
        """
    }

    advanced_algorithms = [
        a_star_algorithm,
        bellman_ford_algorithm, 
        floyd_warshall_algorithm
    ]
    
    algorithms.extend(advanced_algorithms)
   
    two_pointers_technique = {
    "name": "Two Pointers Technique",
    "tags": ["algorithm technique", "array", "string", "optimization"],
    "description": "The Two Pointers technique is an algorithmic approach that uses two pointers to iterate through a data structure (usually an array or linked list). The pointers can move toward each other, in the same direction at different speeds, or start at different positions. This technique is often used to search for pairs, triplets, or subarrays with certain properties, and it typically reduces the time complexity from O(n²) to O(n).",
    "complexity": {
        "time": "O(n) in most cases",
        "space": "O(1)"
    },
    "problem_patterns": [
        "Finding pairs with a target sum in a sorted array",
        "Detecting palindromes",
        "Removing duplicates from sorted arrays",
        "Finding the longest substring without repeating characters",
        "Container with most water problem",
        "Three sum problem"
    ],
    "leetcode_indicators": [
        "Two pointers",
        "Problems involving sorted arrays",
        "Finding pairs/triplets with specific sum",
        "Problems with phrases like 'find a pair'",
        "Container with most water"
    ],
    "implementation": """
        # Example 1: Two sum in sorted array
        def two_sum_sorted(nums, target):
            left, right = 0, len(nums) - 1
            
            while left < right:
                current_sum = nums[left] + nums[right]
                
                if current_sum == target:
                    return [left, right]
                elif current_sum < target:
                    left += 1
                else:
                    right -= 1
            
            return []  # No solution

        # Example 2: Remove duplicates from sorted array
        def remove_duplicates(nums):
            if not nums:
                return 0
            
            # Position to place the next non-duplicate
            write_pointer = 1
            
            # Iterate through the array
            for read_pointer in range(1, len(nums)):
                # If current element is different from the previous one
                if nums[read_pointer] != nums[read_pointer - 1]:
                    # Place it at the write_pointer position
                    nums[write_pointer] = nums[read_pointer]
                    write_pointer += 1
            
            return write_pointer  # Length of the array without duplicates

        # Example 3: Check if string is palindrome (ignoring non-alphanumeric)
        def is_palindrome(s):
            # Convert to lowercase and filter out non-alphanumeric characters
            s = ''.join(c.lower() for c in s if c.isalnum())
            
            left, right = 0, len(s) - 1
            while left < right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            
            return True
        """
    }

    monotonic_stack_technique = {
    "name": "Monotonic Stack/Queue",
    "tags": ["data structure", "stack", "queue", "optimization"],
    "description": "A Monotonic Stack is a stack that maintains its elements in either strictly increasing or strictly decreasing order. Similarly, a Monotonic Queue maintains the same property. These structures are particularly useful for problems involving finding the next greater/smaller element or maximum/minimum in a sliding window. By maintaining the monotonic property, these structures allow for efficient solving of these problems in linear time.",
    "complexity": {
        "time": "O(n) for most operations (each element is pushed and popped at most once)",
        "space": "O(n)"
    },
    "problem_patterns": [
        "Next greater/smaller element problems",
        "Largest rectangle in histogram",
        "Maximum value in sliding window",
        "Problems involving finding the closest boundary",
        "Stock span problem",
        "Problems involving 'finding the next' pattern"
    ],
    "leetcode_indicators": [
        "Next greater element",
        "Next smaller element",
        "Largest rectangle in histogram",
        "Sliding window maximum",
        "Problems involving finding closest larger/smaller elements"
    ],
    "implementation": """
        # Example 1: Next Greater Element
        def next_greater_element(nums):
            n = len(nums)
            result = [-1] * n  # Initialize with -1 (no greater element)
            stack = []  # Monotonic decreasing stack
            
            for i in range(n):
                # While stack is not empty and current element is greater than stack top
                while stack and nums[i] > nums[stack[-1]]:
                    # Pop and update result
                    result[stack.pop()] = nums[i]
                
                # Push current index
                stack.append(i)
            
            return result

        # Example 2: Largest Rectangle in Histogram
        def largest_rectangle_area(heights):
            n = len(heights)
            stack = []  # Monotonic increasing stack of indices
            max_area = 0
            
            i = 0
            while i < n:
                # If stack is empty or current height is >= height at stack top
                if not stack or heights[i] >= heights[stack[-1]]:
                    stack.append(i)
                    i += 1
                else:
                    # Pop and calculate area
                    top_index = stack.pop()
                    
                    # Calculate width
                    width = i if not stack else i - stack[-1] - 1
                    
                    # Update max area
                    max_area = max(max_area, heights[top_index] * width)
            
            # Process remaining elements in stack
            while stack:
                top_index = stack.pop()
                width = n if not stack else n - stack[-1] - 1
                max_area = max(max_area, heights[top_index] * width)
            
            return max_area

        # Example 3: Sliding Window Maximum using Monotonic Queue
        from collections import deque

        def max_sliding_window(nums, k):
            n = len(nums)
            if n == 0 or k == 0:
                return []
            
            result = []
            queue = deque()  # Monotonic decreasing queue of indices
            
            for i in range(n):
                # Remove elements outside the window
                while queue and queue[0] < i - k + 1:
                    queue.popleft()
                
                # Remove smaller elements (they will never be the maximum)
                while queue and nums[i] > nums[queue[-1]]:
                    queue.pop()
                
                # Add current element
                queue.append(i)
                
                # Add to result if we have a full window
                if i >= k - 1:
                    result.append(nums[queue[0]])
            
            return result
        """
    }

    backtracking_technique = {
    "name": "Backtracking",
    "tags": ["algorithm technique", "recursion", "combinatorial", "search"],
    "description": "Backtracking is an algorithmic technique for solving problems recursively by trying to build a solution incrementally, one step at a time, removing solutions that fail to satisfy the constraints of the problem. It's like a systematic trial and error approach. The name comes from the fact that when you reach a state where you can't proceed further, you 'backtrack' to the previous state and try a different path.",
    "complexity": {
        "time": "O(b^d) where b is the branching factor and d is the maximum depth",
        "space": "O(d) for the recursion stack"
    },
    "problem_patterns": [
        "Permutations and combinations",
        "Subset problems",
        "Constraint satisfaction problems",
        "N-Queens problem",
        "Sudoku solving",
        "Path finding in a maze",
        "Parsing and grammar-related problems"
    ],
    "leetcode_indicators": [
        "Generate all possible",
        "All permutations/combinations/subsets",
        "N-Queens",
        "Problems involving trying all possibilities",
        "Sudoku solver"
    ],
    "implementation": """
        # Example 1: Generate all permutations
        def permute(nums):
            result = []
            
            def backtrack(current, remaining):
                # If no more elements to add, add current permutation to result
                if not remaining:
                    result.append(current[:])
                    return
                
                # Try each remaining element
                for i, num in enumerate(remaining):
                    # Add the current element
                    current.append(num)
                    
                    # Recursively generate permutations with the remaining elements
                    backtrack(current, remaining[:i] + remaining[i+1:])
                    
                    # Backtrack by removing the current element
                    current.pop()
            
            backtrack([], nums)
            return result

        # Example 2: Subset Generation
        def subsets(nums):
            result = []
            
            def backtrack(start, current):
                # Add the current subset to the result
                result.append(current[:])
                
                # Try each element as the next to add
                for i in range(start, len(nums)):
                    # Add the element
                    current.append(nums[i])
                    
                    # Recursively generate subsets with next elements
                    backtrack(i + 1, current)
                    
                    # Backtrack by removing the element
                    current.pop()
            
            backtrack(0, [])
            return result

        # Example 3: N-Queens
        def solve_n_queens(n):
            result = []
            
            def is_safe(board, row, col):
                # Check column
                for i in range(row):
                    if board[i][col] == 'Q':
                        return False
                
                # Check upper-left diagonal
                for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
                    if board[i][j] == 'Q':
                        return False
                
                # Check upper-right diagonal
                for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
                    if board[i][j] == 'Q':
                        return False
                
                return True
            
            def backtrack(board, row):
                # If all queens are placed, add the solution
                if row == n:
                    result.append([''.join(row) for row in board])
                    return
                
                # Try placing a queen in each column of the current row
                for col in range(n):
                    if is_safe(board, row, col):
                        # Place the queen
                        board[row][col] = 'Q'
                        
                        # Recursively place queens in the next row
                        backtrack(board, row + 1)
                        
                        # Backtrack by removing the queen
                        board[row][col] = '.'
            
            # Initialize board with empty cells
            board = [['.' for _ in range(n)] for _ in range(n)]
            backtrack(board, 0)
            
            return result
        """
    }

    specialized_techniques = [
    two_pointers_technique,
    monotonic_stack_technique,
    backtracking_technique
    ]
    
    algorithms.extend(specialized_techniques)
    
    dp_knapsack_01 = {
    "name": "0/1 Knapsack Pattern",
    "tags": ["dynamic programming", "optimization", "knapsack", "decision"],
    "description": "The 0/1 Knapsack pattern is a dynamic programming approach for problems where you have a set of items with values and weights, and you need to determine which items to include to maximize the value while staying within a weight constraint. Each item can either be included (1) or excluded (0), hence the name 0/1 Knapsack. This pattern extends to many problems involving decision-making with constraints.",
    "complexity": {
        "time": "O(n*W) where n is the number of items and W is the weight constraint",
        "space": "O(n*W) for the standard approach, can be optimized to O(W)"
    },
    "problem_patterns": [
        "Subset sum problems",
        "Partition equal subset sum",
        "Minimum difference subset sum",
        "Count of subsets with given sum",
        "Target sum problems",
        "Problems involving picking items with constraints"
    ],
    "leetcode_indicators": [
        "Given array of numbers, find subset with target sum",
        "Partition array into two subsets with equal/minimum difference sum",
        "Problems involving either including or excluding items",
        "Problems with phrases like 'pick items to maximize value with weight constraint'"
    ],
    "implementation": """
        # Example 1: Classic 0/1 Knapsack
        def knapsack_01(values, weights, capacity):
            n = len(values)
            
            # Initialize DP table
            dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
            
            # Fill DP table
            for i in range(1, n + 1):
                for w in range(capacity + 1):
                    # If current item's weight is less than or equal to the capacity
                    if weights[i-1] <= w:
                        # Max of including or excluding the current item
                        dp[i][w] = max(
                            values[i-1] + dp[i-1][w-weights[i-1]],  # Include item
                            dp[i-1][w]  # Exclude item
                        )
                    else:
                        # Can't include this item
                        dp[i][w] = dp[i-1][w]
            
            # Reconstruct the solution
            selected_items = []
            i, j = n, capacity
            while i > 0 and j > 0:
                if dp[i][j] != dp[i-1][j]:
                    # Item was included
                    selected_items.append(i-1)
                    j -= weights[i-1]
                i -= 1
            
            return dp[n][capacity], selected_items[::-1]

        # Example 2: Subset Sum Problem
        def subset_sum(nums, target_sum):
            n = len(nums)
            
            # Initialize DP table
            dp = [[False for _ in range(target_sum + 1)] for _ in range(n + 1)]
            
            # Empty subset has sum 0
            for i in range(n + 1):
                dp[i][0] = True
            
            # Fill DP table
            for i in range(1, n + 1):
                for j in range(1, target_sum + 1):
                    # If current element is greater than sum j
                    if nums[i-1] > j:
                        dp[i][j] = dp[i-1][j]
                    else:
                        # Include or exclude current element
                        dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i-1]]
            
            return dp[n][target_sum]

        # Example 3: Partition Equal Subset Sum
        def can_partition(nums):
            total_sum = sum(nums)
            
            # If sum is odd, can't partition into equal subsets
            if total_sum % 2 != 0:
                return False
            
            target_sum = total_sum // 2
            return subset_sum(nums, target_sum)
        """
    }

    dp_lcs_pattern = {
    "name": "Longest Common Subsequence Pattern",
    "tags": ["dynamic programming", "string", "sequence", "comparison"],
    "description": "The Longest Common Subsequence (LCS) pattern is a dynamic programming approach for finding the longest subsequence common to two sequences. A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements. The LCS pattern extends to many string and sequence comparison problems.",
    "complexity": {
        "time": "O(m*n) where m and n are the lengths of the sequences",
        "space": "O(m*n)"
    },
    "problem_patterns": [
        "Longest common subsequence/substring",
        "Edit distance (insert, delete, replace)",
        "Shortest common supersequence",
        "Longest palindromic subsequence",
        "Minimum deletions/insertions to transform one string to another",
        "Sequence alignment problems"
    ],
    "leetcode_indicators": [
        "Find longest common subsequence/substring",
        "Minimum edits to transform one string to another",
        "Problems involving comparison of two strings/sequences",
        "Problems involving finding similarities between sequences"
    ],
    "implementation": """
        # Example 1: Longest Common Subsequence
        def longest_common_subsequence(text1, text2):
            m, n = len(text1), len(text2)
            
            # Initialize DP table
            dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
            
            # Fill DP table
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if text1[i-1] == text2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            # Reconstruct the LCS
            i, j = m, n
            lcs = []
            
            while i > 0 and j > 0:
                if text1[i-1] == text2[j-1]:
                    lcs.append(text1[i-1])
                    i -= 1
                    j -= 1
                elif dp[i-1][j] > dp[i][j-1]:
                    i -= 1
                else:
                    j -= 1
            
            return ''.join(reversed(lcs)), dp[m][n]

        # Example 2: Edit Distance
        def edit_distance(word1, word2):
            m, n = len(word1), len(word2)
            
            # Initialize DP table
            dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
            
            # Fill the first row and column
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
            
            # Fill DP table
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if word1[i-1] == word2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(
                            dp[i-1][j],      # Delete
                            dp[i][j-1],      # Insert
                            dp[i-1][j-1]     # Replace
                        )
            
            return dp[m][n]

        # Example 3: Longest Palindromic Subsequence
        def longest_palindromic_subsequence(s):
            # The LPS of a string is the LCS of the string and its reverse
            return longest_common_subsequence(s, s[::-1])[1]
        """
    }

    dp_patterns = [
        dp_knapsack_01,
        dp_lcs_pattern
    ]
    
    algorithms.extend(dp_patterns)
    
    sliding_window_algorithm = {
    "name": "Sliding Window Algorithm",
    "tags": ["algorithm technique", "array", "string", "optimization"],
    "description": "The Sliding Window algorithm is a technique used to process arrays or strings by maintaining a 'window' of elements. This window can grow or shrink as needed while sliding through the data structure. This approach is particularly useful for problems involving subarrays or substrings with specific properties, and it typically reduces time complexity from O(n²) to O(n).",
    "complexity": {
        "time": "O(n) where n is the size of the array/string",
        "space": "O(1) to O(k) where k is the window size or alphabet size"
    },
    "problem_patterns": [
        "Finding the longest substring with k distinct characters",
        "Finding the longest substring without repeating characters",
        "Finding the minimum window substring",
        "Maximum sum subarray of size k",
        "Finding the longest subarray with ones after replacement",
        "Maximum number of fruits in two baskets"
    ],
    "leetcode_indicators": [
        "Sliding window",
        "Substring with specific properties",
        "Problems involving consecutive elements",
        "Maximum/minimum subarray/substring",
        "Problems with phrases like 'all subarrays of length k'"
    ],
    "implementation": """
        # Example 1: Fixed-size sliding window (Maximum sum subarray of size k)
        def max_sum_subarray(arr, k):
            n = len(arr)
            if k > n:
                return None
            
            # Initialize window sum and result
            window_sum = sum(arr[:k])
            max_sum = window_sum
            
            # Slide the window
            for i in range(k, n):
                # Remove the element going out of the window
                window_sum -= arr[i - k]
                # Add the element coming into the window
                window_sum += arr[i]
                # Update max_sum
                max_sum = max(max_sum, window_sum)
            
            return max_sum

        # Example 2: Variable-size sliding window (Longest substring without repeating characters)
        def length_of_longest_substring(s):
            n = len(s)
            if n == 0:
                return 0
            
            char_index = {}  # To store the index of each character
            max_length = 0
            window_start = 0
            
            for window_end in range(n):
                # If character is in the current window, update window_start
                if s[window_end] in char_index and char_index[s[window_end]] >= window_start:
                    window_start = char_index[s[window_end]] + 1
                else:
                    # Update max_length
                    max_length = max(max_length, window_end - window_start + 1)
                
                # Update the character's index
                char_index[s[window_end]] = window_end
            
            return max_length

        # Example 3: Minimum Window Substring
        def min_window(s, t):
            if not s or not t:
                return ""
            
            # Dictionary to keep count of characters in t
            target_counts = {}
            for char in t:
                target_counts[char] = target_counts.get(char, 0) + 1
            
            # Variables to track the window
            window_counts = {}
            required = len(target_counts)
            formed = 0
            window_start, window_end = 0, 0
            
            # Variables to track the minimum window
            min_len = float('inf')
            result_start = 0
            
            while window_end < len(s):
                # Add current character to window
                char = s[window_end]
                window_counts[char] = window_counts.get(char, 0) + 1
                
                # Check if this character's count in window meets requirement
                if char in target_counts and window_counts[char] == target_counts[char]:
                    formed += 1
                
                # Try to minimize the window
                while window_start <= window_end and formed == required:
                    char = s[window_start]
                    
                    # Update minimum window
                    if window_end - window_start + 1 < min_len:
                        min_len = window_end - window_start + 1
                        result_start = window_start
                    
                    # Remove character from window
                    window_counts[char] -= 1
                    if char in target_counts and window_counts[char] < target_counts[char]:
                        formed -= 1
                    
                    window_start += 1
                
                window_end += 1
            
            return "" if min_len == float('inf') else s[result_start:result_start + min_len]
        """
    }
    
    algorithms.append(sliding_window_algorithm)
    
    
    # Generate unique IDs for each algorithm
    for algorithm in algorithms:
        algorithm["id"] = generate_uuid()
    
    # Generate problems with updated templates
    problem_templates = {
        "sorting": [
            {
                "title": "Sort an Array",
                "difficulty": "Medium",
                "content": "Given an array of integers nums, sort the array in ascending order.",
                "tags": ["Array", "Sorting", "Divide and Conquer"]
            },
            {
                "title": "Merge Sorted Array",
                "difficulty": "Easy",
                "content": "You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively. Merge nums1 and nums2 into a single array sorted in non-decreasing order.",
                "tags": ["Array", "Two Pointers", "Sorting"]
            }
        ],
        "searching": [
            {
                "title": "Search in Rotated Sorted Array",
                "difficulty": "Medium",
                "content": "There is an integer array nums sorted in ascending order (with distinct values). Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length). Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.",
                "tags": ["Array", "Binary Search"]
            },
            {
                "title": "Find First and Last Position of Element in Sorted Array",
                "difficulty": "Medium",
                "content": "Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value. If target is not found in the array, return [-1, -1].",
                "tags": ["Array", "Binary Search"]
            }
        ],
        "graph": [
            {
                "title": "Course Schedule",
                "difficulty": "Medium",
                "content": "There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai. Return true if you can finish all courses. Otherwise, return false.",
                "tags": ["Depth-First Search", "Breadth-First Search", "Graph", "Topological Sort"]
            },
            {
                "title": "Network Delay Time",
                "difficulty": "Medium",
                "content": "You are given a network of n nodes, labeled from 1 to n. You are also given times, a list of travel times as directed edges times[i] = (ui, vi, wi), where ui is the source node, vi is the target node, and wi is the time it takes for a signal to travel from source to target. We will send a signal from a given node k. Return the time it takes for all the n nodes to receive the signal. If it is impossible for all the n nodes to receive the signal, return -1.",
                "tags": ["Depth-First Search", "Breadth-First Search", "Graph", "Heap (Priority Queue)", "Shortest Path"]
            }
        ],
        "dynamic programming": [
            {
                "title": "Coin Change",
                "difficulty": "Medium",
                "content": "You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money. Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.",
                "tags": ["Array", "Dynamic Programming", "Breadth-First Search"]
            },
            {
                "title": "Longest Increasing Subsequence",
                "difficulty": "Medium",
                "content": "Given an integer array nums, return the length of the longest strictly increasing subsequence.",
                "tags": ["Array", "Binary Search", "Dynamic Programming"]
            }
        ],
        "string": [
            {
                "title": "Longest Palindromic Substring",
                "difficulty": "Medium",
                "content": "Given a string s, return the longest palindromic substring in s.",
                "tags": ["String", "Dynamic Programming"]
            },
            {
                "title": "Implement strStr()",
                "difficulty": "Easy",
                "content": "Given two strings needle and haystack, return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.",
                "tags": ["Two Pointers", "String", "String Matching"]
            }
        ],
        "tree": [
            {
                "title": "Binary Tree Inorder Traversal",
                "difficulty": "Easy",
                "content": "Given the root of a binary tree, return the inorder traversal of its nodes' values.",
                "tags": ["Stack", "Tree", "Depth-First Search", "Binary Tree"]
            },
            {
                "title": "Lowest Common Ancestor of a Binary Tree",
                "difficulty": "Medium",
                "content": "Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree. According to the definition of LCA on Wikipedia: 'The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).'",
                "tags": ["Tree", "Depth-First Search", "Binary Tree"]
            }
        ],
        "linked list": [
            {
                "title": "Reverse Linked List",
                "difficulty": "Easy",
                "content": "Given the head of a singly linked list, reverse the list, and return the reversed list.",
                "tags": ["Linked List", "Recursion"],
                "suitable_algorithms": ["Linked List Reversal"]
            },
            {
                "title": "Linked List Cycle",
                "difficulty": "Easy",
                "content": "Given head, the head of a linked list, determine if the linked list has a cycle in it.",
                "tags": ["Linked List", "Two Pointers", "Hash Table"],
                "suitable_algorithms": ["Linked List Cycle Detection"]
            },
            {
                "title": "Rotate List",
                "difficulty": "Medium",
                "content": "Given the head of a linked list, rotate the list to the right by k places.",
                "tags": ["Linked List", "Two Pointers"],
                "suitable_algorithms": ["Linked List Rotation"]
            },
            {
                "title": "Merge Two Sorted Lists",
                "difficulty": "Easy",
                "content": "You are given the heads of two sorted linked lists list1 and list2. Merge the two lists into one sorted list.",
                "tags": ["Linked List", "Recursion"],
                "suitable_algorithms": ["Linked List Merge"]
            }
        ],
        "stack": [
            {
                "title": "Valid Parentheses",
                "difficulty": "Easy",
                "content": "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.",
                "tags": ["String", "Stack"],
                "suitable_algorithms": ["Balanced Parentheses Check", "Stack Implementation"]
            },
            {
                "title": "Min Stack",
                "difficulty": "Medium",
                "content": "Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.",
                "tags": ["Stack", "Design"],
                "suitable_algorithms": ["Stack Implementation"]
            }
        ],
        "queue": [
            {
                "title": "Design Circular Queue",
                "difficulty": "Medium",
                "content": "Design your implementation of the circular queue. The circular queue is a linear data structure in which the operations are performed based on FIFO principle.",
                "tags": ["Array", "Linked List", "Design", "Queue"],
                "suitable_algorithms": ["Circular Queue Implementation"]
            },
            {
                "title": "Sliding Window Maximum",
                "difficulty": "Hard",
                "content": "You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position. Return the max sliding window.",
                "tags": ["Array", "Queue", "Sliding Window", "Heap (Priority Queue)"],
                "suitable_algorithms": ["Queue Implementation", "Priority Queue Implementation"]
            }
        ],
        "hash table": [
            {
                "title": "Two Sum",
                "difficulty": "Easy",
                "content": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
                "tags": ["Array", "Hash Table"],
                "suitable_algorithms": ["Hash Table Implementation"]
            },
            {
                "title": "Design HashMap",
                "difficulty": "Easy",
                "content": "Design a HashMap without using any built-in hash table libraries.",
                "tags": ["Array", "Hash Table", "Linked List", "Design"],
                "suitable_algorithms": ["Hash Table Implementation", "Collision Resolution with Chaining", "Open Addressing (Linear Probing)"]
            }
        ],
        "trie": [
            {
                "title": "Implement Trie (Prefix Tree)",
                "difficulty": "Medium",
                "content": "A trie (pronounced as \"try\") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. Implement the Trie class with insert, search, and startsWith methods.",
                "tags": ["Hash Table", "String", "Design", "Trie"],
                "suitable_algorithms": ["Trie (Prefix Tree)"]
            },
            {
                "title": "Word Search II",
                "difficulty": "Hard", 
                "content": "Given an m x n board of characters and a list of strings words, return all words on the board. Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring.",
                "tags": ["Array", "String", "Backtracking", "Trie"],
                "suitable_algorithms": ["Trie (Prefix Tree)", "Backtracking"]
            }
        ],
        "segment_tree": [
            {
                "title": "Range Sum Query - Mutable",
                "difficulty": "Medium",
                "content": "Given an array nums and two types of queries: update an element to a new value, and calculate the sum of the elements of nums between indices left and right inclusive where left <= right.",
                "tags": ["Array", "Design", "Binary Indexed Tree", "Segment Tree"],
                "suitable_algorithms": ["Segment Tree"]
            }
        ],
        "union_find": [
            {
                "title": "Number of Connected Components in an Undirected Graph",
                "difficulty": "Medium",
                "content": "You have a graph of n nodes. You are given an integer n and an array edges where edges[i] = [ai, bi] indicates that there is an edge between ai and bi in the graph. Return the number of connected components in the graph.",
                "tags": ["Depth-First Search", "Breadth-First Search", "Union Find", "Graph"],
                "suitable_algorithms": ["Union Find (Disjoint Set)", "Depth-First Search (DFS)", "Breadth-First Search (BFS)"]
            }
        ],
        "advanced_algorithms": [
            {
                "title": "Path With Minimum Effort",
                "difficulty": "Medium",
                "content": "You are a hiker preparing for an upcoming hike. You are given heights, a 2D array of size rows x columns, where heights[row][col] represents the height of cell (row, col). You are situated in the top-left cell, (0, 0), and you hope to travel to the bottom-right cell, (rows-1, columns-1).",
                "tags": ["Array", "Binary Search", "Depth-First Search", "Breadth-First Search", "Union Find", "Heap (Priority Queue)", "Matrix"],
                "suitable_algorithms": ["A* Search Algorithm", "Dijkstra's Algorithm"]
            }
        ],
        "two_pointers": [
            {
                "title": "Container With Most Water",
                "difficulty": "Medium",
                "content": "You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]). Find two lines that together with the x-axis form a container, such that the container contains the most water.",
                "tags": ["Array", "Two Pointers", "Greedy"],
                "suitable_algorithms": ["Two Pointers Technique"]
            }
        ],
        "monotonic_stack": [
            {
                "title": "Next Greater Element I",
                "difficulty": "Easy",
                "content": "The next greater element of some element x in an array is the first greater element that is to the right of x in the same array. You are given two distinct 0-indexed integer arrays nums1 and nums2, where nums1 is a subset of nums2. Find all the next greater elements of nums1 in nums2 and return them in an array.",
                "tags": ["Array", "Hash Table", "Stack", "Monotonic Stack"],
                "suitable_algorithms": ["Monotonic Stack/Queue"]
            }
        ],
        "backtracking": [
            {
                "title": "Permutations", 
                "difficulty": "Medium",
                "content": "Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.",
                "tags": ["Array", "Backtracking"],
                "suitable_algorithms": ["Backtracking"]
            }
        ]
    }
    
    # Generate problems for each algorithm category
    problems = []
    problem_id = 1
    for category, templates in problem_templates.items():
        category_algorithms = [a for a in algorithms if any(tag.lower() in category.lower() for tag in a["tags"])]
        
        for template in templates:
            problem = {
                "id": str(problem_id),
                "title": template["title"],
                "difficulty": template["difficulty"],
                "content": template["content"],
                "tags": template["tags"],
                "url": f"https://leetcode.com/problems/{template['title'].lower().replace(' ', '-')}/",
                "similar_questions": [],
                "hints": ["Try to think about the problem systematically.", "Can you break it down into smaller steps?"],
                "suitable_algorithms": template.get("suitable_algorithms", [a["name"] for a in category_algorithms][:3])  
            }
            
            problems.append(problem)
            problem_id += 1
    
    additional_problems = [
        {
            "id": str(problem_id),
            "title": "Two Sum",
            "difficulty": "Easy",
            "content": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice.",
            "tags": ["Array", "Hash Table"],
            "url": "https://leetcode.com/problems/two-sum/",
            "similar_questions": ["Three Sum", "Four Sum", "Two Sum II"],
            "hints": ["Think about using a hash table to store numbers you've seen."],
            "suitable_algorithms": ["Hash Table Implementation", "Two Pointers Technique"]
        },
        {
            "id": str(problem_id + 1),
            "title": "Add Two Numbers",
            "difficulty": "Medium",
            "content": "You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.",
            "tags": ["Linked List", "Math", "Recursion"],
            "url": "https://leetcode.com/problems/add-two-numbers/",
            "similar_questions": ["Multiply Strings", "Add Binary"],
            "hints": ["Keep track of the carry using a variable."],
            "suitable_algorithms": ["Linked List Traversal", "Math"]
        },
        {
            "id": str(problem_id + 2),
            "title": "Median of Two Sorted Arrays",
            "difficulty": "Hard",
            "content": "Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.",
            "tags": ["Array", "Binary Search", "Divide and Conquer"],
            "url": "https://leetcode.com/problems/median-of-two-sorted-arrays/",
            "similar_questions": ["Find K-th Smallest Pair Distance"],
            "hints": ["Think about a binary search approach."],
            "suitable_algorithms": ["Binary Search", "Divide and Conquer"]
        },
        {
            "id": str(problem_id + 3),
            "title": "LRU Cache",
            "difficulty": "Medium",
            "content": "Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.",
            "tags": ["Hash Table", "Linked List", "Design", "Doubly-Linked List"],
            "url": "https://leetcode.com/problems/lru-cache/",
            "similar_questions": ["LFU Cache", "Design In-Memory File System"],
            "hints": ["Use a combination of a hash table and a doubly linked list."],
            "suitable_algorithms": ["Hash Table Implementation", "Linked List Implementation"]
        }
    ]
    
    problems.extend(additional_problems)
    
    algorithm_to_problems, problem_to_algorithms = enhanced_map_algorithms_to_problems(algorithms, problems)
    
    dataset = {
        "algorithms": algorithms,
        "problems": problems,
        "mappings": {
            "algorithm_to_problems": algorithm_to_problems,
            "problem_to_algorithms": problem_to_algorithms
        }
    }
    
    return dataset

def generate_leetcode_problems(algorithms):
    """Generate mock LeetCode problems related to the algorithms."""
    print("Generating mock LeetCode problems...")
    
    return [] 

def enhanced_map_algorithms_to_problems(algorithms, problems):
    """Enhanced mapping between algorithms and problems with better matching."""
    algorithm_to_problems = {}
    problem_to_algorithms = {}
    
    for algorithm in algorithms:
        algorithm_to_problems[algorithm["name"]] = []
    

    def match_algorithm_to_problem(problem, algorithm):
        if algorithm["name"] in problem.get("suitable_algorithms", []):
            return True
      
        algo_tags = [tag.lower() for tag in algorithm["tags"]]
        problem_tags = [tag.lower() for tag in problem.get("tags", [])]
        
        if any(tag in problem_tags for tag in algo_tags):
            return True
            
        problem_content = problem.get("content", "").lower()
        
        if algorithm["name"] == "Trie (Prefix Tree)" and any(term in problem_content for term in ["prefix", "word dictionary", "autocomplete"]):
            return True
            
        if algorithm["name"] == "Segment Tree" and any(term in problem_content for term in ["range", "interval query", "sum query"]):
            return True
            
        if algorithm["name"] == "Union Find (Disjoint Set)" and any(term in problem_content for term in ["connected components", "groups", "clusters"]):
            return True
            
        if algorithm["name"] == "Backtracking" and any(term in problem_content for term in ["all possible", "combinations", "permutations"]):
            return True
            
        if algorithm["name"] == "Two Pointers Technique" and any(term in problem_content for term in ["pair", "two numbers", "container", "palindrome"]):
            return True
            
        if algorithm["name"] == "Sliding Window Algorithm" and any(term in problem_content for term in ["substring", "subarray", "consecutive", "window"]):
            return True
            
        if algorithm["name"] == "0/1 Knapsack Pattern" and any(term in problem_content for term in ["subset", "partition", "maximize value"]):
            return True
        
        if algorithm["name"].lower() in problem_content:
            return True
        
        return False
    
    for problem in problems:
        problem_id = problem["id"]
        matched_algorithms = []
        
        if "suitable_algorithms" in problem and problem["suitable_algorithms"]:
            matched_algorithms = problem["suitable_algorithms"]
        else:
            for algorithm in algorithms:
                if match_algorithm_to_problem(problem, algorithm):
                    matched_algorithms.append(algorithm["name"])
        
        problem_to_algorithms[problem_id] = matched_algorithms
        
        for algorithm_name in matched_algorithms:
            if algorithm_name in algorithm_to_problems:
                algorithm_to_problems[algorithm_name].append({
                    "id": problem["id"],
                    "title": problem["title"],
                    "difficulty": problem["difficulty"],
                    "url": problem.get("url", "")
                })
    
    return algorithm_to_problems, problem_to_algorithms

def save_algorithm_database(dataset, file_path="./data/raw/custom_algorithm_database.json"):
    """Save the algorithm database to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved database with {len(dataset['algorithms'])} algorithms and {len(dataset['problems'])} problems to {file_path}")

if __name__ == "__main__":
    print("Starting algorithm dataset generation...")

    dataset = generate_algorithm_dataset()
    save_algorithm_database(dataset)
    
    print("Algorithm dataset generation completed successfully!")