#!/bin/bash
# PRISM Environment Setup Script
# Uses pyenv for virtual environment management

set -e

echo "=========================================="
echo "PRISM Environment Setup"
echo "=========================================="

# Check if pyenv is available
if ! command -v pyenv &> /dev/null; then
    echo "Error: pyenv is not installed or not in PATH"
    exit 1
fi

# Check current pyenv version
current_env=$(pyenv version-name 2>/dev/null || echo "none")
echo "Current pyenv environment: $current_env"

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "Python version: $python_version"

# Verify we're in the prism environment
if [ "$current_env" != "prism" ]; then
    echo ""
    echo "Warning: Not in 'prism' pyenv environment."
    echo "Run these commands first:"
    echo "  pyenv virtualenv 3.13.7 prism"
    echo "  pyenv local prism"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Download NLP models
echo ""
echo "Downloading NLP models..."
python -m nltk.downloader punkt vader_lexicon stopwords 2>/dev/null || true
python -m spacy download en_core_web_sm 2>/dev/null || true

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp env_template.txt .env
    echo "Please edit .env and add your OPENAI_API_KEY"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p logs
mkdir -p .cache/llm_responses

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OPENAI_API_KEY"
echo "2. Run the dashboard: make run"
echo ""
