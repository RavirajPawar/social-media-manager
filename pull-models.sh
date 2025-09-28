#!/bin/bash
set -e  # Exit on any error

# Start Ollama in the background
echo "Starting Ollama server..."
ollama serve & 
OLLAMA_PID=$!

# Wait for Ollama to start (adjust sleep time if needed)
echo "Waiting for Ollama to initialize..."
sleep 10

# Debug: Print the contents of the script to verify it's correct
echo "Contents of pull-models.sh:"
cat /scripts/pull-models.sh

# Check if Ollama is running by listing models (this will also validate installation)
echo "Validating Ollama installation..."
if ollama list; then
    echo "✅ Ollama is installed and running correctly."
    ollama list
else
    echo "❌ Ollama validation failed. Please check the installation."
    exit 1
fi

# Pull the desired models (customize this list as needed)
echo "Pulling models..."
ollama pull gemma3:1b

echo "✅ All specified models pulled successfully."

# Keep the container running and wait for Ollama processes
wait $OLLAMA_PID