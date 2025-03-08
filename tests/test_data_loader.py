# tests/test_data_loader.py
import numpy as np
import os
import json
import pytest
from scripts.data_loader import ArtelDataLoader

@pytest.fixture
def sample_data_dir(tmpdir):
    # Create sample JSON files
    text_dir = tmpdir.mkdir("text")
    code_dir = tmpdir.mkdir("code_snippets")
    table_dir = tmpdir.mkdir("tables")

    # Sample text data
    text_file = text_dir.join("text1.json")
    text_file.write(json.dumps({
        "id": "text1",
        "content": "This is a sample text.",
        "parent": "",
        "title": "Sample Text"
    }))

    # Sample code data
    code_file = code_dir.join("code1.json")
    code_file.write(json.dumps({
        "id": "code1",
        "content": "print('Hello, World!')",
        "parent": "",
        "title": "Sample Code"
    }))

    # Sample table data
    table_file = table_dir.join("table1.json")
    table_file.write(json.dumps({
        "id": "table1",
        "content": "| Header 1 | Header 2 |\n|----------|----------|\n| Data 1   | Data 2   |",
        "parent": "",
        "title": "Sample Table"
    }))

    return str(tmpdir)

def test_data_loader(sample_data_dir):
    loader = ArtelDataLoader(base_dir=sample_data_dir)
    all_content = loader.load_all_content()

    # Verify loaded content
    assert len(all_content['text']) == 1
    assert len(all_content['code']) == 1
    assert len(all_content['tables']) == 1

    # Verify chunk preparation
    chunks = loader.prepare_chunks_for_embedding(all_content)
    assert len(chunks) == 3
    assert chunks[0]['metadata']['type'] == 'text'
    assert chunks[1]['metadata']['type'] == 'code'
    assert chunks[2]['metadata']['type'] == 'table'
