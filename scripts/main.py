# scripts/main.py
import os
import argparse
import logging
import json
import yaml
from datetime import datetime
from dotenv import load_dotenv  # Added for .env file support

# Import components
from data_loader import ArtelDataLoader
from embedding_generator import EmbeddingGenerator
from vector_database import ChromaVectorStore
from rag_pipeline import RAGPipeline
from evaluation import EvaluationMetrics, ResultsEvaluator

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_pipeline(config):
    """Set up the embedding and vector database pipeline"""
    # Step 1: Load and prepare data
    logger.info("Loading data from processed JSON files...")
    data_loader = ArtelDataLoader(base_dir=config['directories']['processed_data'])
    all_content = data_loader.load_all_content()
    chunks = data_loader.prepare_chunks_for_embedding(all_content)
    
    # Step 2: Generate embeddings
    logger.info("Generating embeddings...")
    embedding_model = config.get('models', {}).get('embedding', 'all-MiniLM-L6-v2')
    embedding_generator = EmbeddingGenerator(model_name=embedding_model)
    chunks_with_embeddings = embedding_generator.generate_embeddings(chunks)
    
    # Step 3: Save embeddings
    logger.info("Saving embeddings...")
    embeddings_dir = os.path.join(config['directories']['output'], "embeddings")
    embeddings_file, metadata_file = embedding_generator.save_embeddings(
        chunks_with_embeddings, embeddings_dir
    )
    
    # Step 4: Set up vector database
    logger.info("Setting up vector database...")
    vector_db = ChromaVectorStore(
        persist_directory=os.path.join(config['directories']['output'], "chroma_db")
    )
    collection = vector_db.create_collection(
        name="arista_telecom_docs",
        overwrite=True
    )
    
    # Step 5: Add documents to vector database
    logger.info("Adding documents to vector database...")
    vector_db.add_documents_from_embeddings(
        collection=collection,
        embeddings_file=embeddings_file,
        metadata_file=metadata_file
    )
    
    logger.info("Setup completed successfully!")
    return collection, vector_db

def run_query(config, query, task_type, model):
    """Run a single query through the RAG pipeline"""
    # Initialize vector database
    vector_db = ChromaVectorStore(
        persist_directory=os.path.join(config['directories']['output'], "chroma_db")
    )
    collection = vector_db.create_collection(
        name="arista_telecom_docs",
        overwrite=False
    )
    
    # Get API key with priority: environment variable > config file
    openai_api_key = os.environ.get('OPENAI_API_KEY', config.get('api_keys', {}).get('openai', ''))
    
    # Check if API key is available when using GPT models
    if 'gpt' in model.lower() and not openai_api_key:
        logger.error("OpenAI API key not found. Set OPENAI_API_KEY in .env file or config file.")
        raise ValueError("OpenAI API key is required for GPT models but was not provided")
    
    # Initialize RAG pipeline with API key
    pipeline = RAGPipeline(
        vector_db=collection,
        config_path=config['config_files']['hyperparams'],
        openai_api_key=openai_api_key,
        tiny_llama_path=config.get('models', {}).get('tiny_llama')
    )
    
    # Process query
    result = pipeline.process_query(
        query=query,
        task_type=task_type,
        model=model,
        n_results=config.get('retrieval', {}).get('n_results', 5)
    )
    
    # Save result
    results_dir = os.path.join(config['directories']['output'], "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(
        results_dir, 
        f"{model}_{task_type}_{timestamp}.json"
    )
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return result, result_file

def run_evaluation(config):
    """Run evaluation across all tasks and models"""
    # Initialize metrics
    embedding_model = config.get('models', {}).get('embedding', 'all-MiniLM-L6-v2')
    metrics = EvaluationMetrics(embedding_model=embedding_model)
    evaluator = ResultsEvaluator(metrics=metrics)
    
    # Load results
    results_dir = os.path.join(config['directories']['output'], "results")
    results = evaluator.load_results(
        results_dir=results_dir,
        model_filter=["gpt4_rag", "gpt4_no_rag", "tiny_llama"]
    )
    
    # Load reference data
    reference_dir = config.get('directories', {}).get('reference_data', 'data/reference')
    
    reference_summaries = []
    reference_qa = []
    reference_code = []
    
    try:
        with open(os.path.join(reference_dir, 'reference_summaries.json'), 'r') as f:
            reference_summaries = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load reference summaries: {e}")
    
    try:
        with open(os.path.join(reference_dir, 'reference_qa.json'), 'r') as f:
            reference_qa = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load reference QA: {e}")
    
    try:
        with open(os.path.join(reference_dir, 'reference_code.json'), 'r') as f:
            reference_code = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load reference code: {e}")
    
    # Evaluate results
    summarization_eval = evaluator.evaluate_summarization(
        results.get('summarization', {}),
        reference_summaries
    )
    
    qa_eval = evaluator.evaluate_question_answering(
        results.get('question_answering', {}),
        reference_qa
    )
    
    code_eval = evaluator.evaluate_code_generation(
        results.get('code_generation', {}),
        reference_code
    )
    
    # Generate report
    output_file = os.path.join(config['directories']['output'], "evaluation_report.json")
    report = evaluator.generate_comparison_report(
        summarization_eval=summarization_eval,
        qa_eval=qa_eval,
        code_eval=code_eval,
        output_file=output_file
    )
    
    return report, output_file

def main():
    parser = argparse.ArgumentParser(description="Telecommunications RAG Pipeline")
    parser.add_argument("--config", default="configs/paths.yaml", help="Path to config")
    parser.add_argument("--mode", choices=["setup", "query", "evaluate"], default="setup", 
                        help="Mode: 'setup' for initial setup, 'query' to run queries, 'evaluate' to run evaluation")
    parser.add_argument("--query", type=str, help="Query text")
    parser.add_argument("--task", choices=["summarization", "question_answering", "code_generation"],
                        default="question_answering", help="Task type")
    parser.add_argument("--model", choices=["gpt4_rag", "gpt4_no_rag", "tiny_llama"],
                        default="gpt4_rag", help="Model to use")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Check if API key is available when using GPT models
    if 'gpt' in args.model.lower():
        api_key = os.environ.get('OPENAI_API_KEY', config.get('api_keys', {}).get('openai', ''))
        if not api_key:
            logger.warning("No OpenAI API key found in .env file or config file")
            if args.mode != "setup":
                logger.error("OpenAI API key is required for GPT models")
                print("Error: OpenAI API key not found. Please add it to your .env file in the project root.")
                return
    
    if args.mode == "setup":
        setup_pipeline(config)
    
    elif args.mode == "query":
        if not args.query:
            logger.error("Query must be provided in query mode")
            return
        
        result, result_file = run_query(
            config=config,
            query=args.query,
            task_type=args.task,
            model=args.model
        )

        # Print result
        print("\n" + "="*80)
        print(f"Query: {result['query']}")
        print(f"Task: {result['task_type']}")
        print(f"Model: {result['model']}")
        print("="*80)
        print(result['response'])
        print("="*80)
        print(f"Result saved to: {result_file}")

    elif args.mode == "evaluate":
        report, output_file = run_evaluation(config)

        # Print summary of results
        print("\n" + "="*80)
        print("Evaluation Results Summary")
        print("="*80)

        for task, task_results in report.items():
            print(f"\n{task.upper()}")
            for model, metrics in task_results.items():
                print(f"  {model}:")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value}")

        print("\n" + "="*80)
        print(f"Full report saved to: {output_file}")

if __name__ == "__main__":
    main()
