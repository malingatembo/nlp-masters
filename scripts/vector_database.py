# scripts/vector_database.py
import os
import logging
import chromadb
import numpy as np
import json
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """Vector database implementation using Chroma"""
    
    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize with optional persistence directory"""
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
            logger.info(f"Initialized persistent Chroma client at {persist_directory}")
        else:
            self.client = chromadb.Client()
            logger.info("Initialized in-memory Chroma client")
    
    def create_collection(self, name: str, overwrite: bool = False) -> chromadb.Collection:
        """Create a new collection or get existing one"""
        # Check if collection exists
        try:
            existing_collections = self.client.list_collections()
            exists = any(c.name == name for c in existing_collections)
            
            if exists and overwrite:
                self.client.delete_collection(name)
                logger.info(f"Deleted existing collection: {name}")
                exists = False
            
            if not exists:
                collection = self.client.create_collection(name=name)
                logger.info(f"Created new collection: {name}")
            else:
                collection = self.client.get_collection(name=name)
                logger.info(f"Using existing collection: {name}")
            
            return collection
        except Exception as e:
            logger.error(f"Error with collection {name}: {e}")
            raise
    
    def add_documents_from_embeddings(
        self, 
        collection: chromadb.Collection,
        embeddings_file: str,
        metadata_file: str,
        batch_size: int = 64
    ):
        """Add documents to collection from pre-computed embeddings"""
        # Load embeddings and metadata
        logger.info(f"Loading embeddings from {embeddings_file}")
        embeddings = np.load(embeddings_file)
        
        logger.info(f"Loading metadata from {metadata_file}")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        if len(embeddings) != len(metadata):
            raise ValueError(f"Mismatch between embeddings ({len(embeddings)}) and metadata ({len(metadata)})")
        
        # Add documents in batches
        total_docs = len(metadata)
        logger.info(f"Adding {total_docs} documents to collection {collection.name} in batches of {batch_size}")
        
        for i in range(0, total_docs, batch_size):
            end_idx = min(i + batch_size, total_docs)
            batch_metadata = metadata[i:end_idx]
            batch_embeddings = embeddings[i:end_idx].tolist()
            batch_documents = [item['text'] for item in batch_metadata]
            batch_ids = [f"doc_{i+j}" for j in range(len(batch_metadata))]
            
            # Clean metadata for Chroma (remove text which is already in documents)
            batch_metadata_clean = []
            for meta in batch_metadata:
                meta_clean = meta.copy()
                if 'text' in meta_clean:
                    del meta_clean['text']
            
                # sanitize empty metadata valuees
                for key, value in list(meta_clean.items()):
                    if value is None:
                        meta_clean[key] = ""

                batch_metadata_clean.append(meta_clean)

            # Add to collection
            collection.add(
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadata_clean,
                ids=batch_ids
            )
            
            logger.info(f"Added batch {i+1} to {end_idx} of {total_docs}")
        
        logger.info(f"Successfully added {total_docs} documents to collection {collection.name}")
    
    def query(
        self, 
        collection: chromadb.Collection,
        query_text: str,
        n_results: int = 5,
        filter_criteria: Optional[Dict] = None
    ) -> Dict:
        """Query the collection for similar documents"""
        # Convert query to embedding
        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=filter_criteria
            )
            
            logger.info(f"Query returned {len(results['documents'][0])} results")
            return results
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            raise
