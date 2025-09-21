# ChromaDB Vector Store Integration Guide

This guide explains how ChromaDB is integrated into the RAG system for efficient vector storage and retrieval.

## Overview

ChromaDB is used as the primary vector database for storing document embeddings and enabling semantic search. The implementation provides:

1. Persistent storage of document embeddings
2. Efficient similarity search
3. Flexible filtering by metadata
4. Multiple embedding function support

## Key Components

### SimpleVectorStore Class

The `SimpleVectorStore` class in `db/simple_store.py` provides a clean interface to ChromaDB with the following features:

- **Persistent storage**: Embeddings are stored on disk and persist between sessions
- **Custom embedding functions**: Supports using any embedding model
- **Metadata filtering**: Filter search results by document metadata
- **Pre-computed embeddings**: Can use pre-computed embeddings for efficiency

### Integration with RAG System

The RAG system (`EnhancedAdaptiveRAGSystem` class) uses the vector store through:

1. Initialization with custom embedding function
2. Document addition with metadata
3. Semantic search for relevant documents
4. System status reporting with vector store statistics

## Usage Examples

### Basic Initialization

```python
from db.simple_store import SimpleVectorStore

# Create a vector store with default settings
vector_store = SimpleVectorStore(
    collection_name="documents",
    persist_directory="./db/chroma_db"
)
```

### Using Custom Embedding Function

```python
from db.simple_store import SimpleVectorStore

# Define embedding function
def my_embedding_function(text):
    # Your embedding logic here
    return embedding_vector  # list of floats

# Create store with custom embedding function
vector_store = SimpleVectorStore(
    collection_name="documents",
    persist_directory="./db/chroma_db",
    embedding_function=my_embedding_function
)
```

### Adding Documents

```python
# Add documents with metadata
texts = ["Document 1 content", "Document 2 content"]
metadata = [
    {"source": "web", "date": "2023-01-01"},
    {"source": "file", "date": "2023-02-01"}
]

# Option 1: Let ChromaDB handle embeddings
vector_store.add_documents(texts=texts, metadata=metadata)

# Option 2: Provide pre-computed embeddings
embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...]]  # Your pre-computed embeddings
vector_store.add_documents(texts=texts, metadata=metadata, embeddings=embeddings)
```

### Searching Documents

```python
# Basic search
results = vector_store.search(query="my search query", k=5)

# Search with pre-computed query embedding
query_embedding = my_embedding_function("my search query")
results = vector_store.search(
    query="my search query",
    query_embedding=query_embedding,
    k=5
)

# Search with metadata filter
results = vector_store.search(
    query="my search query",
    k=5,
    filter={"source": "web"}
)
```

### Managing the Vector Store

```python
# Get collection statistics
stats = vector_store.get_collection_stats()
print(f"Document count: {stats['document_count']}")

# Reset the collection (remove all documents)
vector_store.reset()

# Update embedding function
vector_store.update_embedding_function(new_embedding_function)

# Delete specific document
vector_store.delete_document(document_id="doc123")

# Get all documents
all_docs = vector_store.get_all_documents()
```

## API Endpoints

The system provides the following API endpoints for managing the vector store:

- `/reset-vector-store`: Reset the vector store, removing all documents
- `/status`: Get system status, including vector store statistics
- `/initialize`: Initialize the system with a new vector store

## Performance Considerations

1. **Batch Processing**: When adding many documents, batch them in groups of 500-1000 for better performance.

2. **Embedding Computation**: Pre-computing embeddings can improve performance when adding large sets of documents.

3. **Storage Size**: Monitor the size of the persist directory as it grows with the number of documents.

4. **Memory Usage**: ChromaDB loads embeddings into memory for fast search, so monitor RAM usage with large collections.

## Troubleshooting

If you encounter issues with the vector store:

1. Check the logs for any error messages
2. Verify that the persist directory exists and is writable
3. Ensure the embedding dimensions match consistently
4. Try resetting the vector store if it becomes corrupted

For more details, refer to the [ChromaDB documentation](https://docs.trychroma.com/).
