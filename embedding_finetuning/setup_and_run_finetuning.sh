#!/bin/bash

# Setup and Run Fine-tuning for Vietnamese Food Embedding Model

echo "🍜 VIETNAMESE FOOD EMBEDDING FINE-TUNING SETUP"
echo "=============================================="

# Change to project root directory
cd "$(dirname "$0")/.."

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📚 Installing fine-tuning requirements..."
pip install -r embedding_finetuning/requirements_finetuning.txt

# Check GPU availability
echo "🖥️ Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Create models directory
echo "📁 Creating models directory..."
mkdir -p models

# Run fine-tuning pipeline
echo "🚀 Starting fine-tuning pipeline..."
cd embedding_finetuning
python3 run_fine_tuning.py

echo "✅ Fine-tuning setup and execution completed!"