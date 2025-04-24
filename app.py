import os
import sys
import json
import threading
import subprocess
from flask import Flask, request, jsonify, render_template, session
import requests
from datetime import datetime
import uuid

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing integration code (modify path if needed)
from api.llm_integration import generate_algorithm_recommendation, retrieve_algorithms, format_algorithm_context, generate_response

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Function to start the RAG service in a separate process
def start_rag_service():
    subprocess.Popen(["python", "-m", "uvicorn", "api.rag_service:app", "--host", "0.0.0.0", "--port", "8000"])

class ChatSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.messages = []
        self.created_at = datetime.now()
    
    def add_message(self, role, content):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def get_messages(self):
        return self.messages
    
    def to_dict(self):
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }

# Store chat sessions
chat_sessions = {}

@app.route('/')
def index():
    """Render the main chatbot interface."""
    # Generate a session ID if not already in the session
    if 'session_id' not in session:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        chat_sessions[session_id] = ChatSession(session_id)
        
        # Add welcome message
        chat_sessions[session_id].add_message(
            "assistant", 
            "Hello! I'm your Algorithm Assistant. Describe a programming problem, and I'll recommend the most suitable algorithm for your needs."
        )
    else:
        session_id = session['session_id']
        # Create a new session if session_id not in chat_sessions
        if session_id not in chat_sessions:
            chat_sessions[session_id] = ChatSession(session_id)
            chat_sessions[session_id].add_message(
                "assistant", 
                "Hello! I'm your Algorithm Assistant. Describe a programming problem, and I'll recommend the most suitable algorithm for your needs."
            )
    
    # Get messages for the current session
    messages = chat_sessions[session_id].get_messages()
    
    return render_template('chat.html', messages=messages)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat messages from the user."""
    data = request.json
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    # Get or create session
    session_id = session.get('session_id')
    if not session_id or session_id not in chat_sessions:
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        chat_sessions[session_id] = ChatSession(session_id)
    
    # Add user message to chat history
    chat_sessions[session_id].add_message("user", user_message)
    
    try:
        # First, check for special commands
        if user_message.lower() in ['/help', 'help']:
            response = """
                I can help you find the right algorithm for your coding problems. Just describe what you're trying to accomplish, and I'll recommend the most suitable algorithm.
                
                You can ask things like:
                - How can I find the shortest path in a graph?
                - What algorithm should I use to sort a large dataset?
                - I need to find all permutations of a string
                - Find the longest substring without repeating characters
                
                You can also use these commands:
                - /help - Show this help message
                - /clear - Start a new conversation
                - /explain [algorithm] - Get more information about a specific algorithm
            """
        elif user_message.lower() == '/clear':
            # Clear the chat history and start a new session
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
            chat_sessions[session_id] = ChatSession(session_id)
            response = "Conversation cleared. Let's start fresh! What algorithm problem can I help you with?"
        elif user_message.lower().startswith('/explain '):
            # Extract algorithm name from command
            algorithm_name = user_message[9:].strip()
            # Use the enhanced retrieval with hybrid search
            algorithms = retrieve_algorithms(
                f"Tell me about {algorithm_name}", 
                top_k=1,
                rag_endpoint="http://localhost:8000/api/query",
                params={"hybrid_search": True, "alpha": 0.6}  # Add hybrid search parameters
            )
            if algorithms:
                alg = algorithms[0]
                response = f"Here's what I know about {alg['name']}:\n\n"
                if alg.get('description'):
                    response += f"{alg['description']}\n\n"
                if alg.get('complexity'):
                    complexity = alg['complexity']
                    time_complexity = complexity.get('time', 'Not specified')
                    space_complexity = complexity.get('space', 'Not specified')
                    response += f"Complexity: Time - {time_complexity}, Space - {space_complexity}\n\n"
                if alg.get('use_cases'):
                    response += f"Use Cases: {', '.join(alg['use_cases'])}\n\n"
                if alg.get('tags'):
                    response += f"Tags: {', '.join(alg['tags'])}"
                # Add match details if available
                if alg.get('match_details'):
                    response += f"\n\nRelevant context: {alg['match_details']}"
            else:
                response = f"I couldn't find specific information about {algorithm_name}. Could you try another algorithm or describe a problem instead?"
        else:
            # Special handling for algorithm queries about longest substring
            if "longest substring" in user_message.lower() and ("without repeat" in user_message.lower() or "without duplicate" in user_message.lower()):
                print("Detected longest substring without repeating characters query - prioritizing sliding window technique")
                # Call generate_response with a hint that this is a sliding window problem
                response = generate_response(user_message)
            else:
                # Normal query - generate algorithm recommendation
                response = generate_response(user_message)
        
        # Add assistant response to chat history
        chat_sessions[session_id].add_message("assistant", response)
        
        return jsonify({
            "response": response,
            "session_id": session_id
        })
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        chat_sessions[session_id].add_message("assistant", error_msg)
        return jsonify({
            "response": error_msg,
            "session_id": session_id
        })

@app.route('/new_chat')
def new_chat():
    """Start a new chat session."""
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    chat_sessions[session_id] = ChatSession(session_id)
    
    # Add welcome message
    chat_sessions[session_id].add_message(
        "assistant", 
        "Hello! I'm your Algorithm Assistant. Describe a programming problem, and I'll recommend the most suitable algorithm for your needs."
    )
    
    return jsonify({"redirect": "/"})

@app.route('/sessions')
def list_sessions():
    """View all chat sessions (admin feature)."""
    # Would typically require authentication in a real app
    sessions_list = []
    for session_id, chat_session in chat_sessions.items():
        sessions_list.append({
            "session_id": session_id,
            "created_at": chat_session.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "messages_count": len(chat_session.messages)
        })
    
    return render_template('sessions.html', sessions=sessions_list)

@app.route('/about')
def about():
    """About page with information on the system."""
    return render_template('about.html')

if __name__ == '__main__':
    # Ensure templates directory exists
    os.makedirs('templates', exist_ok=True)
    
    # Create templates if they don't exist
    if not os.path.exists('templates/chat.html'):
        with open('templates/chat.html', 'w') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Algorithm RAG Chatbot</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body {
            background-color: #f5f5f5;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .chat-container {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            max-width: 1200px;
            margin: 0 auto;
            width: 100%;
        }
        .chat-header {
            background-color: #4a69bd;
            color: white;
            padding: 15px 20px;
            border-radius: 10px 10px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .chat-header h1 {
            margin: 0;
            font-size: 1.5rem;
        }
        .chat-messages {
            flex-grow: 1;
            padding: 20px;
            overflow-y: auto;
            background-color: white;
            border-left: 1px solid #e0e0e0;
            border-right: 1px solid #e0e0e0;
            max-height: calc(100vh - 170px);
        }
        .message {
            margin-bottom: 15px;
            display: flex;
            flex-direction: column;
        }
        .message-content {
            padding: 10px 15px;
            border-radius: 10px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .chat-input textarea {
            flex-grow: 1;
            padding: 10px 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            outline: none;
            resize: none; /* Prevents manual resizing */
            height: auto;
            min-height: 50px;
            max-height: 150px; /* Limits how tall it can get */
            font-family: inherit;
            font-size: inherit;
        }
        .user-message {
            align-items: flex-end;
        }
        .user-message .message-content {
            background-color: #4a69bd;
            color: white;
        }
        .assistant-message {
            align-items: flex-start;
        }
        .assistant-message .message-content {
            background-color: #f0f2f5;
            color: #333;
        }
        .message-time {
            font-size: 0.7rem;
            color: #777;
            margin-top: 5px;
            margin-left: 5px;
            margin-right: 5px;
        }
        .chat-input {
            padding: 15px;
            background-color: #f8f9fa;
            border-top: 1px solid #e0e0e0;
            border-radius: 0 0 10px 10px;
            display: flex;
        }
        .chat-input input {
            flex-grow: 1;
            padding: 10px 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            outline: none;
        }
        .chat-input button {
            background-color: #4a69bd;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            margin-left: 10px;
            cursor: pointer;
        }
        .chat-input button:hover {
            background-color: #3a539b;
        }
        .commands {
            position: absolute;
            bottom: 80px;
            right: 20px;
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: none;
            z-index: 100;
        }
        .commands ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .commands li {
            padding: 5px 10px;
            cursor: pointer;
        }
        .commands li:hover {
            background-color: #f5f5f5;
        }
        .show-commands {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: #4a69bd;
            color: white;
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .navbar {
            background-color: #4a69bd;
        }
        .navbar-brand {
            color: white;
            font-weight: bold;
        }
        .navbar-nav .nav-link {
            color: rgba(255,255,255,0.8);
        }
        .navbar-nav .nav-link:hover {
            color: white;
        }
        code {
            background-color: #282c34;
            color: #abb2bf;
            padding: 2px 4px;
            border-radius: 4px;
        }
        pre {
            background-color: #282c34;
            color: #abb2bf;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 10px 0;
        }
        .typing-indicator {
            display: flex;
            padding: 10px 15px;
            background-color: #f0f2f5;
            border-radius: 10px;
            width: fit-content;
        }
        .typing-indicator span {
            height: 10px;
            width: 10px;
            margin: 0 2px;
            background-color: #9E9EA1;
            display: block;
            border-radius: 50%;
            opacity: 0.4;
        }
        .typing-indicator span:nth-of-type(1) {
            animation: typing 1s infinite;
        }
        .typing-indicator span:nth-of-type(2) {
            animation: typing 1s 0.25s infinite;
        }
        .typing-indicator span:nth-of-type(3) {
            animation: typing 1s 0.5s infinite;
        }
        @keyframes typing {
            0% {
                opacity: 0.4;
                transform: translateY(0);
            }
            50% {
                opacity: 1;
                transform: translateY(-5px);
            }
            100% {
                opacity: 0.4;
                transform: translateY(0);
            }
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg">
        <div class="container">
            <a class="navbar-brand" href="/">Algorithm RAG Chatbot</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="#" id="new-chat-btn">New Chat</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/about">About</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container chat-container">
        <div class="chat-header">
            <h1>Algorithm RAG System</h1>
        </div>
        
        <div class="chat-messages" id="chat-messages">
            {% for message in messages %}
                <div class="message {{ 'user-message' if message.role == 'user' else 'assistant-message' }}">
                    <div class="message-content">{{ message.content | safe }}</div>
                    <div class="message-time">{{ message.timestamp }}</div>
                </div>
            {% endfor %}
        </div>
        
        <div class="chat-input">
            <textarea id="user-input" placeholder="Ask about an algorithm problem...(Shift+Enter to submit)" rows="2"></textarea>
            <button id="send-button"><i class="fas fa-paper-plane"></i></button>
        </div>
    </div>
    
    <button class="show-commands" id="show-commands">
        <i class="fas fa-question"></i>
    </button>
    
    <div class="commands" id="commands">
        <ul>
            <li data-command="/help">Help</li>
            <li data-command="/clear">Clear Chat</li>
            <li data-command="/explain">Explain Algorithm</li>
        </ul>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const chatMessages = document.getElementById('chat-messages');
            const userInput = document.getElementById('user-input');
            const sendButton = document.getElementById('send-button');
            const showCommandsBtn = document.getElementById('show-commands');
            const commandsPanel = document.getElementById('commands');
            const newChatBtn = document.getElementById('new-chat-btn');
            
            // Scroll to bottom of chat
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            // Send message function
            function sendMessage() {
                const message = userInput.value.trim();
                if (message === '') return;
                
                // Add user message to chat
                addMessage('user', message);
                userInput.value = '';
                
                // Show typing indicator
                showTypingIndicator();
                
                // Send message to server
                fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ message: message })
                })
                .then(response => response.json())
                .then(data => {
                    // Remove typing indicator
                    removeTypingIndicator();
                    
                    // Add response to chat
                    addMessage('assistant', data.response);
                    
                    // Process any special responses
                    if (data.redirect) {
                        window.location.href = data.redirect;
                    }
                })
                .catch(error => {
                    // Remove typing indicator
                    removeTypingIndicator();
                    
                    // Add error message
                    addMessage('assistant', 'Sorry, there was an error processing your request.');
                    console.error('Error:', error);
                });
            }
            
            // Add message to chat
            function addMessage(role, content) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${role === 'user' ? 'user-message' : 'assistant-message'}`;
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                
                // Format code blocks
                if (role === 'assistant') {
                    content = formatCodeBlocks(content);
                }
                
                contentDiv.innerHTML = content;
                
                const timeDiv = document.createElement('div');
                timeDiv.className = 'message-time';
                
                const now = new Date();
                timeDiv.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                
                messageDiv.appendChild(contentDiv);
                messageDiv.appendChild(timeDiv);
                
                chatMessages.appendChild(messageDiv);
                
                // Scroll to bottom
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            // Format code blocks
            function formatCodeBlocks(content) {
                // Replace ```code``` with <pre><code>code</code></pre>
                return content.replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>')
                              .replace(/`([^`]+)`/g, '<code>$1</code>')
                              .replace(/\\n/g, '<br>');
            }
            
            // Show typing indicator
            function showTypingIndicator() {
                const typingDiv = document.createElement('div');
                typingDiv.className = 'message assistant-message';
                typingDiv.id = 'typing-indicator';
                
                const typingContent = document.createElement('div');
                typingContent.className = 'typing-indicator';
                typingContent.innerHTML = '<span></span><span></span><span></span>';
                
                typingDiv.appendChild(typingContent);
                chatMessages.appendChild(typingDiv);
                
                // Scroll to bottom
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            // Remove typing indicator
            function removeTypingIndicator() {
                const typingIndicator = document.getElementById('typing-indicator');
                if (typingIndicator) {
                    typingIndicator.remove();
                }
            }
            
            // Event listeners
            sendButton.addEventListener('click', sendMessage);
            
            userInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }else if (e.key === 'Enter') {}
            });
            
            // Commands panel
            showCommandsBtn.addEventListener('click', function() {
                commandsPanel.style.display = commandsPanel.style.display === 'block' ? 'none' : 'block';
            });
            
            // Command selection
            document.querySelectorAll('.commands li').forEach(item => {
                item.addEventListener('click', function() {
                    let command = this.getAttribute('data-command');
                    
                    if (command === '/explain') {
                        // Prompt for algorithm name
                        const algorithm = prompt('Which algorithm would you like to learn about?');
                        if (algorithm) {
                            command += ' ' + algorithm;
                        } else {
                            return;
                        }
                    }
                    
                    userInput.value = command;
                    commandsPanel.style.display = 'none';
                    // Focus on input
                    userInput.focus();
                });
            });
            
            // New chat button
            newChatBtn.addEventListener('click', function(e) {
                e.preventDefault();
                
                // Confirm before clearing
                if (confirm('Start a new chat? This will clear your current conversation.')) {
                    fetch('/new_chat')
                        .then(response => response.json())
                        .then(data => {
                            if (data.redirect) {
                                window.location.href = data.redirect;
                            }
                        });
                }
            });
            
            // Close commands panel when clicking outside
            document.addEventListener('click', function(e) {
                if (!commandsPanel.contains(e.target) && e.target !== showCommandsBtn) {
                    commandsPanel.style.display = 'none';
                }
            });
        });
    </script>
</body>
</html>""")
    
    if not os.path.exists('templates/about.html'):
        with open('templates/about.html', 'w') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About - Algorithm RAG Chatbot</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f5f5f5;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .about-container {
            max-width: 800px;
            margin: 50px auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        }
        .navbar {
            background-color: #4a69bd;
        }
        .navbar-brand {
            color: white;
            font-weight: bold;
        }
        .navbar-nav .nav-link {
            color: rgba(255,255,255,0.8);
        }
        .navbar-nav .nav-link:hover {
            color: white;
        }
        .tech-stack {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
        }
        .tech-badge {
            background-color: #f0f2f5;
            color: #333;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.9rem;
        }
        .section {
            margin-bottom: 30px;
        }
        .section-title {
            color: #4a69bd;
            border-bottom: 2px solid #4a69bd;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg">
        <div class="container">
            <a class="navbar-brand" href="/">Algorithm RAG Chatbot</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="/">Back to Chat</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container about-container">
        <h1 class="text-center mb-4">About This System</h1>
        
        <div class="section">
            <h2 class="section-title">What is it?</h2>
            <p>
                The Algorithm RAG Chatbot is a Retrieval Augmented Generation (RAG) system designed to help programmers 
                find the most suitable algorithms for their specific problems. The system combines a vector database of 
                algorithm information with a language model to provide tailored recommendations.
            </p>
        </div>
        
        <div class="section">
            <h2 class="section-title">How It Works</h2>
            <p>
                When you describe a programming problem, the system:
            </p>
            <ol>
                <li>Converts your query into a vector embedding using Sentence Transformers</li>
                <li>Searches the vector database using FAISS to find the most similar algorithm descriptions</li>
                <li>Retrieves detailed information about the best matching algorithms</li>
                <li>Uses a language model to analyze the matches and generate a recommendation tailored to your specific problem</li>
                <li>Formats and presents the recommendation with explanations of why the algorithm is suitable</li>
            </ol>
        </div>
        
        <div class="section">
            <h2 class="section-title">Technology Stack</h2>
            <p>This project was built using the following technologies:</p>
            <div class="tech-stack">
                <span class="tech-badge">Python</span>
                <span class="tech-badge">FastAPI</span>
                <span class="tech-badge">Flask</span>
                <span class="tech-badge">Sentence Transformers</span>
                <span class="tech-badge">FAISS</span>
                <span class="tech-badge">LLM Integration</span>
                <span class="tech-badge">HTML/CSS</span>
                <span class="tech-badge">JavaScript</span>
                <span class="tech-badge">Bootstrap</span>
            </div>
        </div>
        
        <div class="section">
            <h2 class="section-title">Features</h2>
            <ul>
                <li>Natural language understanding of programming problems</li>
                <li>Semantic search for finding relevant algorithms</li>
                <li>Detailed explanations of why an algorithm is recommended</li>
                <li>Information about algorithm complexity and use cases</li>
                <li>Conversation memory to reference previous queries</li>
                <li>Special commands for system interaction</li>
            </ul>
        </div>
        
        <div class="section">
            <h2 class="section-title">Team</h2>
            <p>
                This project was developed as part of a course project on RAG systems and LLM applications.
            </p>
            <p>
                <strong>Team Members:</strong>
            </p>
            <ul>
                <li>Aarzoo Bansal - RAG System Development</li>
                <li>Mandar Ambulkar - Data Collection and Processing</li>
            </ul>
        </div>
    </div>
</body>
</html>""")
    
    if not os.path.exists('templates/sessions.html'):
        with open('templates/sessions.html', 'w') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sessions - Algorithm RAG Chatbot</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f5f5f5;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .sessions-container {
            max-width: 800px;
            margin: 50px auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        }
        .navbar {
            background-color: #4a69bd;
        }
        .navbar-brand {
            color: white;
            font-weight: bold;
        }
        .navbar-nav .nav-link {
            color: rgba(255,255,255,0.8);
        }
        .navbar-nav .nav-link:hover {
            color: white;
        }
        .session-item {
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 10px;
            margin-bottom: 15px;
            transition: all 0.2s ease;
        }
        .session-item:hover {
            background-color: #f8f9fa;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg">
        <div class="container">
            <a class="navbar-brand" href="/">Algorithm RAG Chatbot</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    <li class="nav-item">
                        <a class="nav-link" href="/">Back to Chat</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container sessions-container">
        <h1 class="text-center mb-4">Chat Sessions</h1>
        
        <div class="mb-4">
            <a href="/" class="btn btn-outline-primary">Back to Chat</a>
        </div>
        
        {% if sessions %}
            <div class="session-list">
                {% for session in sessions %}
                    <div class="session-item">
                        <h5>Session ID: {{ session.session_id[:8] }}...</h5>
                        <p>Created: {{ session.created_at }}</p>
                        <p>Messages: {{ session.messages_count }}</p>
                    </div>
                {% endfor %}
            </div>
        {% else %}
            <p class="text-center">No chat sessions found.</p>
        {% endif %}
    </div>
</body>
</html>""")
    
    # If running the RAG service separately is needed, uncomment this
    # Start the RAG service in a separate thread
    # rag_thread = threading.Thread(target=start_rag_service)
    # rag_thread.daemon = True  # This ensures the thread will exit when the main program exits
    # rag_thread.start()
    
    # Wait a moment for the RAG service to start
    # import time
    # time.sleep(2)
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)