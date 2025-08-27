import chromadb
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import os
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class SimpleVectorStore:
    def __init__(self, collection_name: str = "documents", persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB client with persistent storage"""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Use persistent client for actual database storage
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Try to get existing collection or create new one
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception as e:
            # Collection doesn't exist, create it
            logger.info(f"Collection '{collection_name}' not found ({str(e)}), creating new one...")
            self.collection = self.client.create_collection(name=collection_name)
            logger.info(f"Created new collection: {collection_name}")
        
    def add_documents(self, texts: List[str], metadata: List[Dict], ids: Optional[List[str]] = None):
        """Add documents to the collection"""
        if ids is None:
            ids = [str(i) for i in range(len(texts))]
            
        # Add documents to collection
        self.collection.add(
            documents=texts,
            metadatas=metadata,
            ids=ids
        )
        
        logger.info(f"Added {len(texts)} documents to collection {self.collection_name}")
        
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Format results
        documents = []
        for i in range(len(results['ids'][0])):
            documents.append({
                'id': results['ids'][0][i],
                'content': results['metadatas'][0][i],
                'text': results['documents'][0][i],
                'similarity': results['distances'][0][i] if 'distances' in results else 0.0
            })
            
        return documents
        
    def reset(self):
        """Reset the collection"""
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        self.collection = self.client.create_collection(name=self.collection_name)
