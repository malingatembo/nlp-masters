# tests/test_embedding_generator.py
import json
import os
import numpy as np
import pytest
from scripts.embedding_generator import EmbeddingGenerator

@pytest.fixture
def sample_chunks():
    return [
        {"text": "This is a sample text.", "metadata": {"id": "text1", "type": "text"}},
        {"text": "print('Hello, World!')", "metadata": {"id": "code1", "type": "code"}}
    ]

def test_embedding_generation(sample_chunks):
    generator = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')
    chunks_with_embeddings = generator.generate_embeddings(sample_chunks)

    # Verify embeddings are added
    assert 'embedding' in chunks_with_embeddings[0]
    assert len(chunks_with_embeddings[0]['embedding']) == 384  # Embedding dimension for all-MiniLM-L6-v2

def test_embedding_saving(sample_chunks, tmpdir):
    generator = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')
    chunks_with_embeddings = generator.generate_embeddings(sample_chunks)

    # Save embeddings
    embeddings_file, metadata_file = generator.save_embeddings(chunks_with_embeddings, str(tmpdir))

    # Verify files are created
    assert os.path.exists(embeddings_file)
    assert os.path.exists(metadata_file)

    # Verify embeddings file content
    embeddings = np.load(embeddings_file)
    assert embeddings.shape == (2, 384)

    # Verify metadata file content
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        assert len(metadata) == 2
        assert metadata[0]['id'] == 'text1'
