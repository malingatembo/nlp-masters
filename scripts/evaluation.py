# scripts/evaluation.py
import os
import logging
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from rouge import Rouge
from sacrebleu import corpus_bleu
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """Evaluation metrics for NLP tasks"""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        """Initialize evaluation metrics"""
        self.rouge = Rouge()
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info(f"Initialized evaluation metrics with embedding model: {embedding_model}")
    
    def calculate_rouge(self, generated: List[str], reference: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate ROUGE scores for summarization evaluation"""
        try:
            scores = self.rouge.get_scores(generated, reference, avg=True)
            logger.info(f"Calculated ROUGE scores: {scores}")
            return scores
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            return {}
    
    def calculate_bleu(self, generated: List[str], reference: List[List[str]]) -> float:
        """Calculate BLEU score for text generation evaluation"""
        try:
            score = corpus_bleu(generated, reference).score
            logger.info(f"Calculated BLEU score: {score}")
            return score
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def calculate_semantic_similarity(self, generated: List[str], reference: List[str]) -> List[float]:
        """Calculate semantic similarity between generated and reference texts"""
        try:
            generated_embeddings = self.embedding_model.encode(generated)
            reference_embeddings = self.embedding_model.encode(reference)
            
            similarities = []
            for gen_emb, ref_emb in zip(generated_embeddings, reference_embeddings):
                similarity = cosine_similarity([gen_emb], [ref_emb])[0][0]
                similarities.append(similarity)
            
            logger.info(f"Calculated semantic similarities: mean={np.mean(similarities)}")
            return similarities
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return []

class ResultsEvaluator:
    """Evaluator for comparing model performance across NLP tasks"""
    
    def __init__(self, metrics: EvaluationMetrics):
        """Initialize with evaluation metrics"""
        self.metrics = metrics
    
    def load_results(self, results_dir: str, model_filter: Optional[List[str]] = None) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Load results from JSON files in the specified directory"""
        results = {
            "summarization": {},
            "question_answering": {},
            "code_generation": {}
        }
        
        try:
            for filename in os.listdir(results_dir):
                if not filename.endswith('.json'):
                    continue
                
                filepath = os.path.join(results_dir, filename)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                model = data.get('model', '')
                task_type = data.get('task_type', '')
                
                if model_filter and model not in model_filter:
                    continue
                
                if task_type in results:
                    if model not in results[task_type]:
                        results[task_type][model] = []
                    
                    results[task_type][model].append(data)
            
            logger.info(f"Loaded results: {sum(len(models) for task, models in results.items() for model, _ in models.items())} files")
            return results
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            return results
    
    def evaluate_summarization(self, results: Dict[str, List[Dict[str, Any]]], reference_summaries: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
        """Evaluate summarization results against reference summaries"""
        evaluation = {}
        
        for model, model_results in results.items():
            generated_summaries = [result['response'] for result in model_results]
            references = []
            
            # Match reference summaries with queries
            for result in model_results:
                query = result['query']
                found = False
                
                for reference in reference_summaries:
                    if reference['query'] == query:
                        references.append(reference['summary'])
                        found = True
                        break
                
                if not found:
                    logger.warning(f"No reference found for query: {query}")
                    references.append("")
            
            # Calculate metrics
            rouge_scores = self.metrics.calculate_rouge(generated_summaries, references)
            semantic_similarities = self.metrics.calculate_semantic_similarity(generated_summaries, references)
            
            evaluation[model] = {
                "rouge-1": rouge_scores.get('rouge-1', {}).get('f', 0.0),
                "rouge-2": rouge_scores.get('rouge-2', {}).get('f', 0.0),
                "rouge-l": rouge_scores.get('rouge-l', {}).get('f', 0.0),
                "semantic_similarity": np.mean(semantic_similarities) if semantic_similarities else 0.0,
                "sample_count": len(generated_summaries)
            }
        
        return evaluation
    
    def evaluate_question_answering(self, results: Dict[str, List[Dict[str, Any]]], reference_answers: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
        """Evaluate question answering results against reference answers"""
        evaluation = {}
        
        for model, model_results in results.items():
            generated_answers = [result['response'] for result in model_results]
            references = []
            
            # Match reference answers with queries
            for result in model_results:
                query = result['query']
                found = False
                
                for reference in reference_answers:
                    if reference['query'] == query:
                        references.append(reference['answer'])
                        found = True
                        break
                
                if not found:
                    logger.warning(f"No reference found for query: {query}")
                    references.append("")
            
            # Calculate metrics
            semantic_similarities = self.metrics.calculate_semantic_similarity(generated_answers, references)
            
            evaluation[model] = {
                "semantic_similarity": np.mean(semantic_similarities) if semantic_similarities else 0.0,
                "sample_count": len(generated_answers)
            }
        
        return evaluation
    
    def evaluate_code_generation(self, results: Dict[str, List[Dict[str, Any]]], reference_code: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
        """Evaluate code generation results against reference code"""
        evaluation = {}
        
        for model, model_results in results.items():
            generated_code = [result['response'] for result in model_results]
            references = []
            reference_lists = []
            
            # Match reference code with queries
            for result in model_results:
                query = result['query']
                found = False
                
                for reference in reference_code:
                    if reference['query'] == query:
                        references.append(reference['code'])
                        reference_lists.append([reference['code']])
                        found = True
                        break
                
                if not found:
                    logger.warning(f"No reference found for query: {query}")
                    references.append("")
                    reference_lists.append([""])
            
            # Calculate metrics
            bleu_score = self.metrics.calculate_bleu(generated_code, reference_lists)
            semantic_similarities = self.metrics.calculate_semantic_similarity(generated_code, references)
            
            evaluation[model] = {
                "bleu": bleu_score,
                "semantic_similarity": np.mean(semantic_similarities) if semantic_similarities else 0.0,
                "sample_count": len(generated_code)
            }
        
        return evaluation
    
    def generate_comparison_report(self, 
                                 summarization_eval: Dict[str, Dict[str, float]], 
                                 qa_eval: Dict[str, Dict[str, float]], 
                                 code_eval: Dict[str, Dict[str, float]],
                                 output_file: str) -> Dict[str, Any]:
        """Generate a comparison report across all tasks and models"""
        report = {
            "summarization": summarization_eval,
            "question_answering": qa_eval,
            "code_generation": code_eval
        }
        
        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Generated comparison report and saved to {output_file}")
        return report
