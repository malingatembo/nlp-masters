# tests/test_rag_pipeline.py
import pytest
import os
import json
from unittest.mock import patch, MagicMock
from scripts.rag_pipeline import RAGPipeline

# Mock ChromaDB collection
@pytest.fixture
def mock_collection():
    mock_col = MagicMock()
    
    # Mock query response
    mock_col.query.return_value = {
        "documents": [["Sample document 1", "Sample document 2"]],
        "metadatas": [[
            {"id": "doc_001", "type": "text", "title": "Sample Text 1"},
            {"id": "doc_002", "type": "code", "title": "Sample Code"}
        ]]
    }
    
    return mock_col

# Test retrieving context
def test_retrieve_context(mock_collection):
    pipeline = RAGPipeline(vector_db=mock_collection)
    
    documents, metadatas = pipeline.retrieve_context(
        query="How to configure interfaces?",
        n_results=2
    )
    
    # Verify that the collection's query method was called
    mock_collection.query.assert_called_once()
    
    # Verify returned data
    assert len(documents) == 2
    assert len(metadatas) == 2
    assert documents[0] == "Sample document 1"
    assert metadatas[0]["id"] == "doc_001"

# Test formatting context
def test_format_context(mock_collection):
    pipeline = RAGPipeline(vector_db=mock_collection)
    
    documents = [
        "This is a text document",
        "```\nshow interface status\n```",
        "| Column1 | Column2 |\n|---------|---------|"
    ]
    
    metadatas = [
        {"type": "text", "title": "Text Document"},
        {"type": "code", "title": "Code Example"},
        {"type": "table", "title": "Table Example"}
    ]
    
    formatted_context = pipeline.format_context(documents, metadatas)
    
    # Verify formatting
    assert "### Section: Text Document" in formatted_context
    assert "### Code Example: Code Example" in formatted_context
    assert "### Table: Table Example" in formatted_context
    assert "This is a text document" in formatted_context
    assert "```\nshow interface status\n```" in formatted_context
    assert "| Column1 | Column2 |" in formatted_context

# Test processing query with mocked generation functions
@patch.object(RAGPipeline, 'generate_with_gpt4')
@patch.object(RAGPipeline, 'retrieve_context')
def test_process_query_gpt4_rag(mock_retrieve, mock_generate, mock_collection):
    # Set up mocks
    mock_retrieve.return_value = (
        ["Test document 1", "Test document 2"], 
        [{"id": "doc_001", "title": "Doc 1"}, {"id": "doc_002", "title": "Doc 2"}]
    )
    mock_generate.return_value = "Generated response using GPT-4 with RAG"
    
    # Initialize pipeline
    pipeline = RAGPipeline(vector_db=mock_collection)
    
    # Process query with GPT-4 RAG
    result = pipeline.process_query(
        query="How to configure BGP?",
        task_type="question_answering",
        model="gpt4_rag"
    )
    
    # Verify result
    assert result["query"] == "How to configure BGP?"
    assert result["task_type"] == "question_answering"
    assert result["model"] == "gpt4_rag"
    assert result["response"] == "Generated response using GPT-4 with RAG"
    assert len(result["context_used"]) == 2
    assert result["context_used"][0] == "doc_001"

# Test Tiny LLaMA baseline (with mocked generation function)
@patch.object(RAGPipeline, 'generate_with_tiny_llama')
def test_process_query_tiny_llama(mock_generate, mock_collection):
    # Set up mock
    mock_generate.return_value = "Generated response using Tiny LLaMA"
    
    # Initialize pipeline
    pipeline = RAGPipeline(vector_db=mock_collection)
    
    # Process query with Tiny LLaMA
    result = pipeline.process_query(
        query="Explain OSPF routing",
        task_type="summarization",
        model="tiny_llama"
    )
    
    # Verify result
    assert result["query"] == "Explain OSPF routing"
    assert result["task_type"] == "summarization"
    assert result["model"] == "tiny_llama"
    assert result["response"] == "Generated response using Tiny LLaMA"
    assert len(result["context_used"]) == 0  # No context for baseline model
