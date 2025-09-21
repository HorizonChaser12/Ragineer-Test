"""
Simple Vector Store implementation using ChromaDB.
This version ensures compatibility with ChromaDB v1.0.20+
"""

import chromadb
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
import logging
import os
from datetime import datetime
import json
import uuid

logger = logging.getLogger(__name__)

class SimpleVectorStore:
    def __init__(self, 
                 collection_name: str = "documents", 
                 persist_directory: str = "./chroma_db",
                 embedding_function: Optional[Callable] = None):
        """Initialize ChromaDB client with persistent storage
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            embedding_function: Optional function to use for embedding texts (stored but not directly used with ChromaDB)
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = embedding_function  # Store for our own use
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Use persistent client for actual database storage
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
            
            # Try to get existing collection or create new one
            try:
                self.collection = self.client.get_collection(name=collection_name)
                logger.info(f"Loaded existing collection: {collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                logger.info(f"Creating new collection: {collection_name}")
                self.collection = self.client.create_collection(name=collection_name)
                logger.info(f"Created new collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {e}")
            raise
            
    def add_documents(self, 
                     texts: List[str], 
                     metadata: List[Dict], 
                     embeddings: Optional[List[List[float]]] = None,
                     ids: Optional[List[str]] = None):
        """Add documents to the collection
        
        Args:
            texts: List of document texts to add
            metadata: List of metadata dictionaries for each document
            embeddings: Optional pre-computed embeddings
            ids: Optional list of IDs for documents
        """
        if not texts:
            logger.warning("No texts provided to add_documents")
            return
            
        if len(texts) != len(metadata):
            raise ValueError("Length of texts and metadata must match")
            
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Add documents to collection
        try:
            if embeddings is not None:
                # If embeddings are provided, use them
                if len(embeddings) != len(texts):
                    raise ValueError("Length of embeddings must match texts")
                    
                self.collection.add(
                    documents=texts,
                    metadatas=metadata,
                    embeddings=embeddings,
                    ids=ids
                )
            elif self.embedding_function:
                # If we have a custom embedding function, compute embeddings here
                # and use them with the ChromaDB API
                computed_embeddings = [self.embedding_function(text) for text in texts]
                
                self.collection.add(
                    documents=texts,
                    metadatas=metadata,
                    embeddings=computed_embeddings,
                    ids=ids
                )
            else:
                # Otherwise let ChromaDB handle the embedding
                self.collection.add(
                    documents=texts,
                    metadatas=metadata,
                    ids=ids
                )
                
            logger.info(f"Added {len(texts)} documents to collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
        
        logger.info(f"Added {len(texts)} documents to collection {self.collection_name}")
        
    def search(self, 
              query: str, 
              k: int = 5,
              query_embedding: Optional[List[float]] = None,
              filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar documents
        
        Args:
            query: The text query
            k: Number of results to return
            query_embedding: Optional pre-computed embedding for the query
            filter: Optional metadata filter for search
            
        Returns:
            List of document dictionaries with id, metadata, text and similarity
        """
        try:
            # Perform the search
            if query_embedding is not None:
                # Use the pre-computed embedding
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    where=filter
                )
            elif self.embedding_function:
                # Compute embedding with our function
                computed_embedding = self.embedding_function(query)
                results = self.collection.query(
                    query_embeddings=[computed_embedding],
                    n_results=k,
                    where=filter
                )
            else:
                # Let ChromaDB handle the embedding
                results = self.collection.query(
                    query_texts=[query],
                    n_results=k,
                    where=filter
                )
            
            # Check if we got any results
            if not results['ids'] or not results['ids'][0]:
                return []
            
            # Format results
            documents = []
            for i in range(len(results['ids'][0])):
                doc = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i]
                }
                
                # Add metadata if available
                if 'metadatas' in results and results['metadatas'] and results['metadatas'][0]:
                    doc['metadata'] = results['metadatas'][0][i]
                else:
                    doc['metadata'] = {}
                
                # Add similarity score if available
                if 'distances' in results and results['distances'] and results['distances'][0]:
                    # Convert distance to similarity score (1.0 is most similar)
                    doc['similarity'] = 1.0 - results['distances'][0][i]
                else:
                    doc['similarity'] = 1.0  # Default to perfect match if no score
                
                documents.append(doc)
                
            return documents
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
            
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "name": self.collection_name,
                "error": str(e)
            }
            
    def delete_document(self, document_id: str):
        """Delete a document by ID"""
        try:
            self.collection.delete(ids=[document_id])
            logger.info(f"Deleted document {document_id} from collection {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
        
    def reset(self):
        """Reset the collection by deleting and recreating it"""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as e:
            logger.warning(f"Error deleting collection: {e}")
            
        # Recreate the collection
        self.collection = self.client.create_collection(name=self.collection_name)
        logger.info(f"Reset collection: {self.collection_name}")
        
    def update_embedding_function(self, embedding_function: Callable):
        """Update the embedding function and recreate the collection"""
        # Store the new embedding function
        self.embedding_function = embedding_function
        
        # Get the current documents
        all_docs = self.get_all_documents()
        
        # Reset the collection
        self.reset()
        
        # Re-add all documents if we had any
        if all_docs:
            texts = [doc["text"] for doc in all_docs]
            metadatas = [doc["metadata"] for doc in all_docs]
            ids = [doc["id"] for doc in all_docs]
            
            # Re-add with new embeddings
            self.add_documents(texts=texts, metadata=metadatas, ids=ids)
            
    def get_all_documents(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all documents from the collection"""
        try:
            results = self.collection.get(limit=limit)
            
            documents = []
            if not results["ids"]:
                return documents
                
            for i in range(len(results["ids"])):
                doc = {
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                }
                
                # Add metadata if available
                if "metadatas" in results and results["metadatas"]:
                    doc["metadata"] = results["metadatas"][i]
                else:
                    doc["metadata"] = {}
                documents.append(doc)
                
            return documents
        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return []
