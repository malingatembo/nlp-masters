.PHONY: setup install clean query evaluate

# Configuration
CONFIG_FILE = configs/paths.yaml

# Default target
all: install setup

# Install dependencies
install:
	pip install -r requirements.txt

# Set up the pipeline (generate embeddings and populate vector DB)
setup:
	python scripts/main.py --mode setup --config $(CONFIG_FILE)

# Run a question answering query
query-qa:
	python scripts/main.py --mode query --task question_answering --model gpt4_rag --query "$(QUERY)" --config $(CONFIG_FILE)

# Run a code generation query
query-code:
	python scripts/main.py --mode query --task code_generation --model gpt4_rag --query "$(QUERY)" --config $(CONFIG_FILE)

# Run a summarization query
query-summary:
	python scripts/main.py --mode query --task summarization --model gpt4_rag --query "$(QUERY)" --config $(CONFIG_FILE)

# Run baseline model (Tiny LLaMA)
query-baseline:
	python scripts/main.py --mode query --task $(TASK) --model tiny_llama --query "$(QUERY)" --config $(CONFIG_FILE)

# Run evaluation across all tasks and models
evaluate:
	python scripts/main.py --mode evaluate --config $(CONFIG_FILE)

# Clean generated files
clean:
	rm -rf output/embeddings/*
	rm -rf output/chroma_db/*
	rm -rf output/results/*
