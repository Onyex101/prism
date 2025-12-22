# PRISM Makefile
# Common commands for development workflow
# Uses pyenv virtualenv: pyenv virtualenv 3.13.7 prism

.PHONY: setup install test lint format run clean help

# Default target
help:
	@echo "PRISM - Predictive Risk Intelligence for Software Management"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup      - Install dependencies (pyenv prism env must be active)"
	@echo "  make install    - Install dependencies only"
	@echo "  make test       - Run all tests"
	@echo "  make test-cov   - Run tests with coverage report"
	@echo "  make lint       - Run linting checks"
	@echo "  make format     - Format code with black and isort"
	@echo "  make run        - Launch Streamlit dashboard"
	@echo "  make clean      - Remove cache and build files"
	@echo "  make nlp-setup  - Download NLP models (NLTK, spaCy)"
	@echo ""
	@echo "Prerequisites:"
	@echo "  pyenv virtualenv 3.13.7 prism"
	@echo "  pyenv local prism"
	@echo ""

# Setup - install dependencies (pyenv prism env should already be active)
setup:
	@echo "Using pyenv environment: $$(pyenv version-name)"
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo ""
	@echo "Setup complete! Run 'make nlp-setup' to download NLP models."

# Install dependencies only
install:
	pip install --upgrade pip
	pip install -r requirements.txt

# Download NLP models
nlp-setup:
	python -m nltk.downloader punkt vader_lexicon stopwords
	python -m spacy download en_core_web_sm

# Run all tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

# Run linting
lint:
	flake8 src/ app/ tests/ --max-line-length=100 --ignore=E501,W503
	mypy src/ --ignore-missing-imports

# Format code
format:
	black src/ app/ tests/ --line-length=100
	isort src/ app/ tests/ --profile=black

# Launch Streamlit dashboard
run:
	streamlit run app/app.py

# Run in development mode with auto-reload
run-dev:
	streamlit run app/app.py --server.runOnSave=true

# Clean cache and build files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "Cleaned cache and build files"

# Generate documentation
docs:
	mkdocs build

# Serve documentation locally
docs-serve:
	mkdocs serve

# Train ML models
train:
	python scripts/train_models.py

# Run full evaluation
evaluate:
	python scripts/evaluate_system.py

# Export results
export:
	python scripts/export_results.py
