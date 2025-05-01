# scripts/main.py
import os
import argparse
import logging
import json
import yaml
from datetime import datetime

# Import components
from data_loader import ArtelDataLoader
from embedding_generator import EmbeddingGenerator
from vector_database import ChromaVectorStore
from rag_pipeline import RAGPipeline
from evaluation import EvaluationMetrics, ResultsEvaluator

# Handle dotenv import gracefully
try:
    from dotenv import load_dotenv
    load_dotenv()
    logging.info("Successfully loaded environment variables from .env file")
except ImportError:
    logging.warning("python-dotenv package not installed. Environment variables will only be loaded from system.")
    logging.warning("To enable .env file support, run: pip install python-dotenv")

# Set up detailed logging for debugging retrieval issues
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_pipeline_debug.log"),
        logging.StreamHandler()
    ]
)
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
    
    if not all_content:
        logger.error("No content loaded! Check your processed data directory.")
        return None, None
    
    logger.info(f"Successfully loaded {len(all_content)} documents")
    
    chunks = data_loader.prepare_chunks_for_embedding(all_content)
    logger.info(f"Created {len(chunks)} chunks for embedding")
    
    # Step 2: Generate embeddings
    logger.info("Generating embeddings...")
    embedding_model = config.get('models', {}).get('embedding', 'all-MiniLM-L6-v2')
    embedding_generator = EmbeddingGenerator(model_name=embedding_model)
    chunks_with_embeddings = embedding_generator.generate_embeddings(chunks)
    
    logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
    
    # Step 3: Save embeddings
    logger.info("Saving embeddings...")
    embeddings_dir = os.path.join(config['directories']['output'], "embeddings")
    embeddings_file, metadata_file = embedding_generator.save_embeddings(
        chunks_with_embeddings, embeddings_dir
    )
    
    logger.info(f"Saved embeddings to {embeddings_file} and metadata to {metadata_file}")
    
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

def verify_vector_db(config):
    """Verify that the vector database contains documents without using generate_query_embedding"""
    logger.info("Verifying vector database...")
    
    try:
        # Just check if we can access the vector database
        vector_db_path = os.path.join(config['directories']['output'], "chroma_db")
        if not os.path.exists(vector_db_path):
            logger.error(f"Vector database directory does not exist: {vector_db_path}")
            return False
            
        vector_db = ChromaVectorStore(
            persist_directory=vector_db_path
        )
        
        collection = vector_db.create_collection(
            name="arista_telecom_docs",
            overwrite=False
        )
        
        logger.info("Successfully connected to the vector database")
        print("✅ Vector database connection successful")
        return True
            
    except Exception as e:
        logger.error(f"Vector database verification failed: {str(e)}")
        return False

def run_test_query(config, query_text):
    """Run a simple query through the RAG pipeline to test if it's working"""
    logger.info(f"Testing retrieval with query: '{query_text}'")
    
    # Initialize RAG pipeline
    vector_db = ChromaVectorStore(
        persist_directory=os.path.join(config['directories']['output'], "chroma_db")
    )
    
    collection = vector_db.create_collection(
        name="arista_telecom_docs",
        overwrite=False
    )
    
    # Get API key with priority: environment variable > config file
    openai_api_key = os.environ.get('OPENAI_API_KEY', config.get('api_keys', {}).get('openai', ''))
    
    # Initialize RAG pipeline with lower threshold to increase results
    pipeline = RAGPipeline(
        vector_db=collection,
        config_path=config['config_files']['hyperparams'],
        openai_api_key=openai_api_key,
        tiny_llama_path=config.get('models', {}).get('tiny_llama')
    )
    
    # Try to run a test query to see if context is retrieved
    try:
        # Use the most flexible task type for testing
        result = pipeline.process_query(
            query=query_text,
            task_type="question_answering",
            model="gpt4_rag",
            n_results=10  # Try to get more results
        )
        
        context_count = len(result.get('context_used', []))
        logger.info(f"Retrieved {context_count} context passages for test query")
        
        print(f"\nTest query: '{query_text}'")
        print(f"Retrieved {context_count} context passages")
        
        if context_count > 0:
            print("\nFirst few context passages:")
            for i, ctx in enumerate(result['context_used'][:3]):  # Show first 3 contexts
                print(f"\n--- Context {i+1} " + "-"*30)
                print(ctx[:200] + "..." if len(ctx) > 200 else ctx)
            return True
        else:
            print("⚠️  No context retrieved for test query")
            return False
            
    except Exception as e:
        logger.error(f"Test query failed: {str(e)}")
        print(f"❌ Test query error: {str(e)}")
        return False

def run_query(config, query, task_type, model):
    """Run a single query through the RAG pipeline with debugging"""
    # Initialize vector database
    vector_db = ChromaVectorStore(
        persist_directory=os.path.join(config['directories']['output'], "chroma_db")
    )
    
    try:
        collection = vector_db.create_collection(
            name="arista_telecom_docs",
            overwrite=False
        )
    except Exception as e:
        logger.error(f"Failed to access vector database: {str(e)}")
        raise
    
    # Get API key with priority: environment variable > config file
    openai_api_key = os.environ.get('OPENAI_API_KEY', config.get('api_keys', {}).get('openai', ''))
    
    # Check if API key is available when using GPT models
    if 'gpt' in model.lower() and not openai_api_key:
        logger.error("OpenAI API key not found. Set OPENAI_API_KEY in .env file or config file.")
        raise ValueError("OpenAI API key is required for GPT models but was not provided")
    
    # Initialize RAG pipeline - Note: We're depending on your RAG pipeline's implementation here
    pipeline = RAGPipeline(
        vector_db=collection,
        config_path=config['config_files']['hyperparams'],
        openai_api_key=openai_api_key,
        tiny_llama_path=config.get('models', {}).get('tiny_llama')
    )
    
    # Force a higher number of results to increase chances of relevant context
    n_results = config.get('retrieval', {}).get('n_results', 5)
    
    # Process query with debug info
    logger.info(f"Processing query: '{query}' using model: {model}")
    result = pipeline.process_query(
        query=query,
        task_type=task_type,
        model=model,
        n_results=n_results
    )
    
    # Log context length to help debug
    context_length = len(result.get('context_used', []))
    logger.info(f"Query completed. Retrieved {context_length} context passages.")
    
    if context_length == 0:
        logger.warning("No context was retrieved for this query!")
    
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
    parser.add_argument("--mode", choices=["setup", "query", "evaluate", "verify", "test"], default="setup", 
                        help="Mode: 'setup' for initial setup, 'query' to run queries, 'evaluate' to run evaluation, 'verify' to check vector db, 'test' to test retrieval")
    parser.add_argument("--query", type=str, help="Query text")
    parser.add_argument("--task", choices=["summarization", "question_answering", "code_generation"],
                        default="question_answering", help="Task type")
    parser.add_argument("--model", choices=["gpt4_rag", "gpt4_no_rag", "tiny_llama"],
                        default="gpt4_rag", help="Model to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Setup debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Load configuration
    config = load_config(args.config)
    
    # Check if API key is available when using GPT models
    if 'gpt' in args.model.lower():
        api_key = os.environ.get('OPENAI_API_KEY', config.get('api_keys', {}).get('openai', ''))
        if not api_key:
            logger.warning("No OpenAI API key found in environment variables or config file")
            if args.mode != "setup" and args.mode != "verify":
                logger.error("OpenAI API key is required for GPT models")
                print("Error: OpenAI API key not found. Please add it to your .env file in the project root.")
                return
    
    if args.mode == "setup":
        logger.info("Running setup mode...")
        setup_pipeline(config)
    
    elif args.mode == "verify":
        logger.info("Running vector database verification...")
        if verify_vector_db(config):
            print("✅ Vector database verification successful!")
        else:
            print("❌ Vector database verification failed!")
    
    elif args.mode == "test":
        test_queries = [
            "What information is displayed in the 'show interfaces status' output table?",
            "Write a configuration to enable forced speed settings on Ethernet interfaces.",
            "What is the purpose of configuring the management interface on a switch?"
        ]
        
        if args.query:
            test_queries = [args.query]
            
        for query in test_queries:
            print("\n" + "="*80)
            run_test_query(config, query)
    
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
        
        # Print context information
        context_count = len(result.get('context_used', []))
        print(f"Retrieved {context_count} context passages")
        if context_count > 0:
            print("\nContext Used:")
            for i, ctx in enumerate(result['context_used']):
                print(f"\n--- Context {i+1} " + "-"*50)
                print(ctx[:300] + "..." if len(ctx) > 300 else ctx)
        
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
