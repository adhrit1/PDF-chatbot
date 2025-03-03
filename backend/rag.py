# rag.py
import os
from typing import Dict, Any, List
import json
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage

class DocumentProcessor:
    def __init__(self):
        # Create storage directory if it doesn't exist
        os.makedirs("storage", exist_ok=True)
    
    def process_document(self, file_path: str) -> str:
        """
        Process a document and return the index ID
        """
        try:
            # Generate a unique ID for this document
            import uuid
            index_id = str(uuid.uuid4())
            
            # Load document
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
            
            # Create index
            index = VectorStoreIndex.from_documents(documents)
            
            # Save index to disk
            index_dir = f"storage/{index_id}"
            os.makedirs(index_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=index_dir)
            
            return index_id
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")

class RAGService:
    def __init__(self):
        self.indices = {}  # Cache for loaded indices
    
    def _load_index(self, index_id: str) -> VectorStoreIndex:
        """
        Load an index from storage or cache
        """
        if index_id in self.indices:
            return self.indices[index_id]
        
        # Load index from disk
        index_dir = f"storage/{index_id}"
        if not os.path.exists(index_dir):
            raise Exception(f"Index {index_id} not found")
        
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
        
        # Cache the index
        self.indices[index_id] = index
        
        return index
    
    def query(self, query: str, index_id: str) -> str:
        """
        Query an index and return the response
        """
        try:
            # Load index
            index = self._load_index(index_id)
            
            # Create query engine
            query_engine = index.as_query_engine()
            
            # Get response
            response = query_engine.query(query)
            
            return str(response)
        except Exception as e:
            raise Exception(f"Error querying index: {str(e)}")