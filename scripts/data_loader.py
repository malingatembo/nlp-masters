# scripts/data_loader.py
import json
import os
import glob
import logging
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArtelDataLoader:
    """Data loader for processed Arista documentation content"""
    
    def __init__(self, base_dir: str = 'data/processed'):
        """Initialize with the base directory for processed content"""
        self.base_dir = base_dir
        self.text_dir = os.path.join(base_dir, 'text')
        self.code_dir = os.path.join(base_dir, 'code_snippets')
        self.table_dir = os.path.join(base_dir, 'tables')
        
        # Validation check
        self._validate_dirs()
        
    def _validate_dirs(self):
        """Validate that required directories exist"""
        for dir_path in [self.text_dir, self.code_dir, self.table_dir]:
            if not os.path.exists(dir_path):
                logger.warning(f"Directory {dir_path} does not exist")
    
    def load_json_file(self, file_path: str) -> Dict[str, Any]:
        """Load a single JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Add source file info to metadata
                data['source_file'] = os.path.basename(file_path)
                return data
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return {}
    
    def load_directory_content(self, directory: str) -> List[Dict[str, Any]]:
        """Load all JSON files from a directory"""
        content = []
        json_files = glob.glob(os.path.join(directory, "*.json"))
        
        for file_path in json_files:
            data = self.load_json_file(file_path)
            if data:  # Only add non-empty data
                content.append(data)
        
        logger.info(f"Loaded {len(content)} files from {directory}")
        return content
    
    def load_all_content(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load all content from processed directories"""
        all_content = {
            'text': self.load_directory_content(self.text_dir),
            'code': self.load_directory_content(self.code_dir),
            'tables': self.load_directory_content(self.table_dir)
        }
        
        logger.info(f"Total loaded content: {sum(len(items) for items in all_content.values())} files")
        return all_content
    
    def build_relationship_map(self, all_content: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, List[str]]]:
        """Build a map of parent-child relationships between content items"""
        relationship_map = {}
        
        # Process each content type
        for content_type, items in all_content.items():
            # Initialize the map for this content type
            if content_type not in relationship_map:
                relationship_map[content_type] = {'parent_to_children': {}, 'id_to_item': {}}
            
            # Process each item
            for item in items:
                item_id = item.get('id', '')
                if not item_id:
                    continue
                
                # Store the item by ID for easy lookup
                relationship_map[content_type]['id_to_item'][item_id] = item
                
                # Add parent-child relationship
                parent_id = item.get('parent', '')
                if parent_id:
                    if parent_id not in relationship_map[content_type]['parent_to_children']:
                        relationship_map[content_type]['parent_to_children'][parent_id] = []
                    relationship_map[content_type]['parent_to_children'][parent_id].append(item_id)
        
        return relationship_map
    
    def prepare_chunks_for_embedding(self, all_content: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Prepare content chunks for embedding generation"""
        chunks = []
        
        # Process text content
        for item in all_content['text']:
            content = item.get('content', '')
            if not content:
                continue
            
            chunk = {
                'text': content,
                'metadata': {
                    'id': item.get('id', ''),
                    'type': 'text',
                    'parent': item.get('parent', ''),
                    'title': item.get('title', ''),
                    'source_file': item.get('source_file', '')
                }
            }
            chunks.append(chunk)
        
        # Process code snippets
        for item in all_content['code']:
            content = item.get('content', '')
            if not content:
                continue
            
            chunk = {
                'text': f"```\n{content}\n```",  # Format code with markdown
                'metadata': {
                    'id': item.get('id', ''),
                    'type': 'code',
                    'parent': item.get('parent', ''),
                    'title': item.get('title', ''),
                    'source_file': item.get('source_file', '')
                }
            }
            chunks.append(chunk)
        
        # Process tables
        for item in all_content['tables']:
            content = item.get('content', '')
            if not content:
                continue
            
            chunk = {
                'text': content,
                'metadata': {
                    'id': item.get('id', ''),
                    'type': 'table',
                    'parent': item.get('parent', ''),
                    'title': item.get('title', ''),
                    'source_file': item.get('source_file', '')
                }
            }
            chunks.append(chunk)
        
        logger.info(f"Prepared {len(chunks)} chunks for embedding")
        return chunks
