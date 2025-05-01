# scripts/rag_pipeline.py
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import chromadb
import yaml
from openai import OpenAI
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for telecommunications domain"""
    
    def __init__(
        self,
        vector_db: chromadb.Collection,
        config_path: str = 'configs/hyperparams.yaml',
        openai_api_key: Optional[str] = None,
        tiny_llama_path: Optional[str] = None
    ):
        """Initialize RAG pipeline with vector database and models"""
        self.vector_db = vector_db
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.use_gpt4 = bool(openai_api_key)
        self.use_tiny_llama = False
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup Tiny LLaMA if model path is provided
        if tiny_llama_path:
            try:
                self.tiny_llama_tokenizer = AutoTokenizer.from_pretrained(tiny_llama_path)
                self.tiny_llama_model = AutoModelForCausalLM.from_pretrained(
                    tiny_llama_path, 
                    device_map="auto"
                )
                self.use_tiny_llama = True
                logger.info(f"Tiny LLaMA model loaded from {tiny_llama_path}")
            except Exception as e:
                logger.error(f"Error loading Tiny LLaMA model: {e}")
                logger.info("Falling back to HuggingFace model API")
                self.tiny_llama = pipeline(
                    "text-generation",
                    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    max_length=512
                )
                self.use_tiny_llama = True
    
    def retrieve_context(
        self,
        query: str,
        n_results: int = 5,
        filter_criteria: Optional[Dict] = None
    ) -> Tuple[List[str], List[Dict]]:
        """Retrieve relevant context from vector database"""
        # Ensure we only get maximum 5 results to avoid token limit issues
        n_results = min(n_results, 5)
        
        results = self.vector_db.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_criteria
        )
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        logger.info(f"Retrieved {len(documents)} relevant documents for context")
        return documents, metadatas
    
    def format_context(self, documents: List[str], metadatas: List[Dict]) -> str:
        """Format retrieved documents into a structured context string"""
        context_parts = []
        
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            doc_type = meta.get('type', 'text')
            title = meta.get('title', f"Document {i+1}")
            
            if doc_type == 'code':
                # Code is already formatted with markdown in the document
                context_parts.append(f"### Code Example: {title}\n{doc}")
            elif doc_type == 'table':
                context_parts.append(f"### Table: {title}\n{doc}")
            else:
                context_parts.append(f"### Section: {title}\n{doc}")
        
        return "\n\n".join(context_parts)
    
    def generate_with_gpt4(
        self, 
        query: str, 
        context: str, 
        task_type: str
    ) -> str:
        """Generate a response using GPT-4 with retrieved context"""
        if not self.use_gpt4:
            return "GPT-4 API not configured"
        
        # Task-specific instructions
        if task_type == "summarization":
            instruction = "Generate a comprehensive and concise summary of the telecommunications documentation in the context. Focus on technical details, command syntax, and configuration steps."
        elif task_type == "question_answering":
            instruction = "Answer the query based solely on the information provided in the context. If the context doesn't contain relevant information, acknowledge the limitations."
        elif task_type == "code_generation":
            instruction = "Generate configuration code or commands to address the query, based on the examples and documentation in the context. Ensure the code follows the conventions shown in the context."
        else:
            instruction = "Generate a comprehensive answer addressing the query using only the information provided in the context."
        
        # Create prompt with context
        prompt = f"""
        [CONTEXT]
        {context}
        [/CONTEXT]
        
        [QUERY]
        {query}
        [/QUERY]
        
        [INSTRUCTION]
        {instruction}
        [/INSTRUCTION]
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.get('generation', {}).get('gpt4', {}).get('temperature', 0.3),
                max_tokens=self.config.get('generation', {}).get('gpt4', {}).get('max_tokens', 1000)
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error with GPT-4 API: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_with_tiny_llama(self, query: str, task_type: str) -> str:
        """Generate a response using Tiny LLaMA model (baseline without RAG)"""
        if not self.use_tiny_llama:
            return "Tiny LLaMA not configured"
        
        # Task-specific prompts
        if task_type == "summarization":
            prompt = f"Please summarize the following telecommunications documentation:\n\n{query}"
        elif task_type == "question_answering":
            prompt = f"Please answer the following telecommunications question:\n\n{query}"
        elif task_type == "code_generation":
            prompt = f"Please generate network configuration code for:\n\n{query}"
        else:
            prompt = f"Please respond to the following query about telecommunications:\n\n{query}"
        
        try:
            # Use the loaded model directly if available
            if hasattr(self, 'tiny_llama_model') and hasattr(self, 'tiny_llama_tokenizer'):
                inputs = self.tiny_llama_tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.tiny_llama_model.device) for k, v in inputs.items()}
                
                generation_config = {
                    "max_length": self.config.get('generation', {}).get('tiny_llama', {}).get('max_length', 512),
                    "temperature": self.config.get('generation', {}).get('tiny_llama', {}).get('temperature', 0.7),
                    "top_p": self.config.get('generation', {}).get('tiny_llama', {}).get('top_p', 0.9),
                    "do_sample": True
                }
                
                outputs = self.tiny_llama_model.generate(**inputs, **generation_config)
                response = self.tiny_llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            # Otherwise, use the pipeline API
            else:
                outputs = self.tiny_llama(
                    prompt,
                    temperature=self.config.get('generation', {}).get('tiny_llama', {}).get('temperature', 0.7),
                    max_length=self.config.get('generation', {}).get('tiny_llama', {}).get('max_length', 512),
                    top_p=self.config.get('generation', {}).get('tiny_llama', {}).get('top_p', 0.9),
                    do_sample=True
                )
                response = outputs[0]['generated_text']
            
            # Extract the actual response by removing the prompt
            response = response.replace(prompt, "").strip()
            
            return response
        except Exception as e:
            logger.error(f"Error with Tiny LLaMA: {e}")
            return f"Error generating response: {str(e)}"
    
    def process_query(
        self,
        query: str,
        task_type: str = "question_answering",
        model: str = "gpt4_rag",
        n_results: int = 5
    ) -> Dict[str, Any]:
        """Process a query through the RAG pipeline"""
        result = {
            "query": query,
            "task_type": task_type,
            "model": model,
            "response": "",
            "context_used": [],
            "metadata": {}
        }
        
        try:
            # RAG approach with GPT-4
            if model == "gpt4_rag":
                # Limit n_results to 5 to prevent token limit issues
                safe_n_results = min(n_results, 5)
                
                # Retrieve context
                documents, metadatas = self.retrieve_context(
                    query, 
                    n_results=safe_n_results
                )
                context = self.format_context(documents, metadatas)
                
                # Generate response with context
                response = self.generate_with_gpt4(query, context, task_type)
                
                result["response"] = response
                result["context_used"] = [m.get('id', '') for m in metadatas]
                result["metadata"]["context_length"] = len(context)
            
            # GPT-4 without RAG (for comparison)
            elif model == "gpt4_no_rag":
                response = self.generate_with_gpt4(query, "", task_type)
                result["response"] = response
            
            # Tiny LLaMA baseline
            elif model == "tiny_llama":
                response = self.generate_with_tiny_llama(query, task_type)
                result["response"] = response
            
            else:
                result["response"] = f"Unknown model: {model}"
                result["metadata"]["error"] = "Invalid model specified"
        
        except Exception as e:
            logger.error(f"Error in process_query: {e}")
            result["response"] = f"Error processing query: {str(e)}"
            result["metadata"]["error"] = str(e)
        
        return result
