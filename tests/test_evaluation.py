# tests/test_evaluation.py
import pytest
import json
import os
import numpy as np
from unittest.mock import patch, MagicMock
from scripts.evaluation import EvaluationMetrics, ResultsEvaluator

# Fixture for evaluation metrics
@pytest.fixture
def evaluation_metrics():
    return EvaluationMetrics(embedding_model="all-MiniLM-L6-v2")

# Test ROUGE calculation
def test_calculate_rouge(evaluation_metrics):
    generated = ["This is a generated summary about Arista networking."]
    reference = ["This is a reference summary about Arista networking devices."]
    
    rouge_scores = evaluation_metrics.calculate_rouge(generated, reference)
    
    # Verify scores exist
    assert "rouge-1" in rouge_scores
    assert "rouge-2" in rouge_scores
    assert "rouge-l" in rouge_scores
    
    # Verify score structure
    assert "f" in rouge_scores["rouge-1"]
    assert "p" in rouge_scores["rouge-1"]
    assert "r" in rouge_scores["rouge-1"]

# Test semantic similarity calculation
def test_calculate_semantic_similarity(evaluation_metrics):
    generated = ["This is a test about Arista networks."]
    reference = ["This is a test about Arista networking equipment."]
    
    similarities = evaluation_metrics.calculate_semantic_similarity(generated, reference)
    
    assert len(similarities) == 1
    assert 0 <= similarities[0] <= 1  # Cosine similarity should be between 0 and 1

# Test BLEU score calculation
def test_calculate_bleu(evaluation_metrics):
    generated = ["configure terminal\ninterface ethernet1/1\nno shutdown"]
    reference = [["configure terminal\ninterface ethernet1/1\nno shutdown\nexit"]]
    
    bleu_score = evaluation_metrics.calculate_bleu(generated, reference)
    
    assert isinstance(bleu_score, float)
    assert 0 <= bleu_score <= 100  # BLEU score is typically 0-100

# Fixture for sample results
@pytest.fixture
def sample_results_dir(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Create sample result files
    result1 = {
        "query": "How to configure BGP?",
        "task_type": "question_answering",
        "model": "gpt4_rag",
        "response": "To configure BGP, you need to...",
        "context_used": ["doc_001", "doc_002"]
    }
    
    result2 = {
        "query": "How to configure BGP?",
        "task_type": "question_answering",
        "model": "tiny_llama",
        "response": "BGP configuration requires...",
        "context_used": []
    }
    
    with open(results_dir / "gpt4_rag_qa_1.json", "w") as f:
        json.dump(result1, f)
    
    with open(results_dir / "tiny_llama_qa_1.json", "w") as f:
        json.dump(result2, f)
    
    return results_dir

# Test loading results
def test_load_results(evaluation_metrics, sample_results_dir):
    evaluator = ResultsEvaluator(metrics=evaluation_metrics)
    
    results = evaluator.load_results(str(sample_results_dir))
    
    # Verify loaded results
    assert "question_answering" in results
    assert "gpt4_rag" in results["question_answering"]
    assert "tiny_llama" in results["question_answering"]
    
    assert len(results["question_answering"]["gpt4_rag"]) == 1
    assert len(results["question_answering"]["tiny_llama"]) == 1
    
    assert results["question_answering"]["gpt4_rag"][0]["query"] == "How to configure BGP?"

# Test question answering evaluation
@patch.object(EvaluationMetrics, 'calculate_semantic_similarity')
def test_evaluate_question_answering(mock_similarity, evaluation_metrics, sample_results_dir):
    # Set up mock
    mock_similarity.return_value = [0.85]
    
    evaluator = ResultsEvaluator(metrics=evaluation_metrics)
    results = evaluator.load_results(str(sample_results_dir))
    
    reference_answers = [
        {
            "query": "How to configure BGP?",
            "answer": "To configure BGP, first enable routing..."
        }
    ]
    
    qa_eval = evaluator.evaluate_question_answering(
        results["question_answering"],
        reference_answers
    )
    
    # Verify evaluation results
    assert "gpt4_rag" in qa_eval
    assert "tiny_llama" in qa_eval
    
    assert "semantic_similarity" in qa_eval["gpt4_rag"]
    assert qa_eval["gpt4_rag"]["semantic_similarity"] == 0.85
    assert qa_eval["gpt4_rag"]["sample_count"] == 1
