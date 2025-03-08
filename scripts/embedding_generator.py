# scripts/embedding_generator.py
import os
import logging
import numpy as np
import json
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for text chunks using Sentence Transformers"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with the specified model"""
        logger.info(f"Initializing embedding generator with model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Generate embeddings for a list of text chunks"""
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(texts)} chunks with batch size {batch_size}")
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
            all_embeddings.extend(batch_embeddings)
        
        # Add embeddings to chunks
        for i, embedding in enumerate(all_embeddings):
            chunks[i]['embedding'] = embedding
        
        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return chunks
    
    def save_embeddings(self, chunks: List[Dict[str, Any]], output_dir: str) -> Tuple[str, str]:
        """Save embeddings and metadata to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract embeddings as array
        embeddings = np.array([chunk['embedding'] for chunk in chunks])
        embeddings_file = os.path.join(output_dir, 'embeddings.npy')
        np.save(embeddings_file, embeddings)
        
        # Save metadata separately
        metadata = []
        for i, chunk in enumerate(chunks):
            meta = chunk['metadata'].copy()
            meta['text'] = chunk['text']
            meta['embedding_index'] = i
            metadata.append(meta)
        
        metadata_file = os.path.join(output_dir, 'metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved embeddings to {embeddings_file} and metadata to {metadata_file}")
        return embeddings_file, metadata_file
