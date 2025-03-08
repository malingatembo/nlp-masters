# tests/test_vector_database.py
import os
import json
import numpy as np
import pytest
import tempfile
from scripts.vector_database import ChromaVectorStore

@pytest.fixture
def sample_embeddings_and_metadata(tmp_path):
    # Create sample embeddings
    embeddings = np.random.rand(3, 384).astype(np.float32)  # 3 documents, 384 dimensions
    
    # Save embeddings to file
    embeddings_file = tmp_path / "embeddings.npy"
    np.save(embeddings_file, embeddings)
    
    # Create metadata
    metadata = [
        {
            "id": "doc_001",
            "type": "text",
            "title": "Sample Text 1",
            "text": "This is sample text one for testing vector database.",
            "embedding_index": 0
        },
        {
            "id": "doc_002",
            "type": "text",
            "title": "Sample Text 2",
            "text": "This is sample text two for testing vector database.",
            "embedding_index": 1
        },
        {
            "id": "doc_003",
            "type": "code",
            "title": "Sample Code",
            "text": "show interface ethernet1/1 status",
            "embedding_index": 2
        }
    ]
    
    # Save metadata to file
    metadata_file = tmp_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)
    
    return str(embeddings_file), str(metadata_file)

# Test creating a new collection
def test_create_collection():
    # Use in-memory database for testing
    vector_store = ChromaVectorStore()
    collection = vector_store.create_collection("test_collection")
    
    assert collection.name == "test_collection"
    
    # Test overwriting existing collection
    new_collection = vector_store.create_collection("test_collection", overwrite=True)
    assert new_collection.name == "test_collection"

# Test adding documents from embeddings
def test_add_documents_from_embeddings(sample_embeddings_and_metadata, tmp_path):
    embeddings_file, metadata_file = sample_embeddings_and_metadata
    
    # Use persistent database for this test
    persist_dir = str(tmp_path / "chroma_db")
    vector_store = ChromaVectorStore(persist_directory=persist_dir)
    
    # Create collection
    collection = vector_store.create_collection("test_documents")
    
    # Add documents
    vector_store.add_documents_from_embeddings(
        collection=collection,
        embeddings_file=embeddings_file,
        metadata_file=metadata_file,
        batch_size=2  # Small batch size to test batching
    )
    
    # Verify documents were added
    collection_info = collection.count()
    assert collection_info == 3  # Should have 3 documents

# Test querying the collection
def test_query(sample_embeddings_and_metadata, tmp_path):
    embeddings_file, metadata_file = sample_embeddings_and_metadata
    
    # Use persistent database for this test
    persist_dir = str(tmp_path / "chroma_db")
    vector_store = ChromaVectorStore(persist_directory=persist_dir)
    
    # Create collection and add documents
    collection = vector_store.create_collection("test_query")
    vector_store.add_documents_from_embeddings(
        collection=collection,
        embeddings_file=embeddings_file,
        metadata_file=metadata_file
    )
    
    # Test querying
    results = vector_store.query(
        collection=collection,
        query_text="Show me information about interfaces",
        n_results=2
    )
    
    # Verify results structure
    assert "documents" in results
    assert "metadatas" in results
    assert len(results["documents"][0]) <= 2  # Should have at most 2 results
    
    # Test filtering
    filtered_results = vector_store.query(
        collection=collection,
        query_text="Show me information",
        n_results=3,
        filter_criteria={"type": "code"}
    )
    
    # If there are results, they should all be code type
    if filtered_results["documents"][0]:
        for metadata in filtered_results["metadatas"][0]:
            assert metadata["type"] == "code"
