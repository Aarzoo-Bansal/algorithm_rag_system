#!/bin/bash

echo "Starting rag_service.py..."
python api/rag_service.py &

# Get PID to optionally kill later
RAG_PID=$!

# Wait for FastAPI (port 8000) to be ready
echo "Waiting for rag_service.py to be ready..."
while ! nc -z localhost 8000; do   
  sleep 1
done

echo "rag_service.py is up. Starting app.py..."
python app.py

# Optionally wait or cleanup
wait $RAG_PID
