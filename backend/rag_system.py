import logging
import numpy as np
import pandas as pd
from pathlib import Path
from langchain.prompts import PromptTemplate
import sys
import os

# Add the parent directory to the path to allow imports from config and models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig
from models import ModelFactory
from typing import Union, Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import Document
import datetime
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import torch
from db.simple_store import SimpleVectorStore

# Configure logging to file only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_system_api.log"),
    filemode="a",
)

logger = logging.getLogger(__name__)

# Initialize model configuration
model_config = ModelConfig()

if not model_config.huggingface_token:
    logger.warning(
        "Hugging Face token not found in .env file. Sentence transformers and reranker may not work properly."
    )

# Validate credentials
if not model_config.validate_credentials():
    logger.error("API credentials validation failed.")


# --- Utility function for JSON serialization ---
def make_serializable(obj: Any) -> Any:
    """ "
    Recursively converts non-serializable objects (like datetime, numpy types)
    in a data structure to JSON-serializable types.
    """

    if isinstance(obj, (datetime.date, datetime.datetime, pd.Timestamp)):

        return obj.isoformat()

    if isinstance(obj, dict):

        return {make_serializable(k): make_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):

        return [make_serializable(i) for i in obj]

    if isinstance(obj, (np.ndarray, np.generic)):

        return obj.tolist()

    if isinstance(obj, torch.Tensor):

        return obj.detach().cpu().numpy().tolist()

    return obj


class EnhancedAdaptiveRAGSystem:

    def __init__(
        self,
        data_source: str = None,  # Can be file path or directory path
        temperature: float = 0.7,
        concise_prompt: bool = False,
        use_sentence_transformers: bool = True,
        use_reranker: bool = True,
        auto_initialize: bool = True,  # Auto-load and process data
    ):
        self.data_source = data_source or "data"  # Default to 'data' directory
        self.concise_prompt = concise_prompt
        self.use_sentence_transformers = use_sentence_transformers
        self.use_reranker = use_reranker
        self.temperature = temperature
        self.auto_initialize = auto_initialize
        
        # Initialize vector store with persistent storage
        self.vector_store = SimpleVectorStore(persist_directory="./db/chroma_db")
        self.chunk_to_original_doc_mapping: List[int] = []
        
        # Initialize chat memory
        from .db.chat_memory import ChatMemoryStore
        self.chat_memory = ChatMemoryStore(memory_directory="./db/chat_memory")
        
        # Store loaded data info
        self.loaded_files = []
        self.total_documents = 0

        # Initialize sentence transformer and reranker models

        if self.use_sentence_transformers:

            logger.info(
                "Loading sentence transformer model: all-MiniLM-L6-v2"
            )

            try:
                if not model_config.huggingface_token:
                    logger.warning(
                        "No Hugging Face token available. Model download may fail."
                    )

                self.sentence_transformer = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2", token=model_config.huggingface_token
                )

                logger.info("Sentence transformer model loaded successfully")

            except Exception as e:

                logger.error(f"Failed to load sentence transformer: {e}")

                self.sentence_transformer = None

                self.use_sentence_transformers = False

        else:

            self.sentence_transformer = None

        if self.use_reranker:

            logger.info("Loading reranker model: cross-encoder/ms-marco-MiniLM-L6-v2")

            try:
                if not model_config.huggingface_token:
                    logger.warning(
                        "No Hugging Face token available. Model download may fail."
                    )

                self.reranker = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L6-v2", 
                    token=model_config.huggingface_token
                )

                logger.info("Reranker model loaded successfully")

            except Exception as e:

                logger.error(f"Failed to load reranker: {e}")

                self.reranker = None

                self.use_reranker = False

        else:

            self.reranker = None

        # Initialize models based on configuration
        logger.info("Initializing AI models")

        try:
            current_config = model_config.get_current_config()
            
            # Initialize embedding model
            self.embedding_model = ModelFactory.create_embedding_model(
                current_config, 
                model_config.api_provider
            )
            
            if not self.embedding_model:
                raise ValueError("Failed to initialize embedding model")
            
            # Set expected dimension for Google's embedding model
            self.dimension = 768  # Gemini embedding dimension
            
            # Initialize LLM
            self.llm = ModelFactory.create_llm_model(
                current_config,
                model_config.api_provider,
                temperature=temperature
            )
            
            if not self.llm:
                raise ValueError("Failed to initialize LLM model")
                
            logger.info(f"{model_config.api_provider.title()} models initialized successfully.")
            logger.info("Google Gemini Models initialized successfully.")

        except Exception as e:

            logger.error(f"Fatal: Failed to initialize Google Gemini models: {e}.")

            self.embedding_model = None

            self.llm = None

        self.data = None

        self.metadata: Optional[List[Dict[Any, Any]]] = None

        self.index = None

        self.st_embeddings = None  # Store sentence transformer embeddings

        # Set expected dimension for Google's embedding model
        self.dimension = 768  # Gemini embedding dimension

        self.column_info: Dict[str, Dict[str, Any]] = {}

        if (self.embedding_model and self.llm) or (self.sentence_transformer):
            if self.auto_initialize:
                logger.info(f"Auto-initializing system with data source: {self.data_source}")
                self._auto_load_and_process_data()
            else:
                logger.info("Auto-initialization disabled. Call initialize_system() manually.")
        else:
            logger.error(
                "Skipping data loading and index building due to model initialization failure."
            )

        if self.llm:

            self.response_template = """
You are a helpful technical support assistant. Your goal is to provide accurate and contextual answers based on the available data and conversation history.

Previous Conversation Context:
{conversation_context}

Dataset Overview:
{dataset_overview}

Retrieved Documents:
{retrieved_documents_context}

Pattern Analysis:
{pattern_analysis_summary}

Current User Query: {query}

RESPONSE GUIDELINES:
1. If the user is asking about previous questions or context (like "what did I ask?", "tell me again", "repeat that"), refer to the conversation context above and provide a clear, direct answer.

2. If the query is about technical issues, defects, or data analysis, provide a structured response with relevant document references with a good explanation of the documents and important info that needs to be addressed.

3. If the query is completely unrelated to the available technical data (like food, weather, entertainment), simply respond: "Sorry, I can only help with technical issues and data analysis based on the available documents. Please ask questions related to defects, bugs, system issues, or data analysis."

4. Keep responses concise and focused. Only use detailed sections (Summary, Specific Details, Recommendations) when the query specifically requests comprehensive analysis and please be able to answer all the things based on the internet data too.

5. For follow-up questions or clarifications, provide direct answers without unnecessary structure.

Format your response appropriately:
- For memory/context questions: Direct, conversational answer
- For simple technical questions: Brief, focused response with document references
- For complex analysis requests: Structured response with sections
- For unrelated questions: Polite redirect message

Response:
"""

            self.prompt = PromptTemplate(
                input_variables=[
                    "conversation_context",
                    "dataset_overview",
                    "retrieved_documents_context", 
                    "pattern_analysis_summary",
                    "query",
                ],
                template=self.response_template,
                validate_template=True,
            )

            # Use modern LangChain syntax: prompt | llm
            self.chain = self.prompt | self.llm

            logger.info("LLM Chain initialized with comprehensive prompt using modern syntax.")

        else:

            self.chain = None

            logger.error(
                "LLMChain could not be initialized because LLM is not available."
            )

    def _generate_dataset_overview_summary(self) -> str:
        """Generates a textual summary of the dataset's structure."""

        if self.data is None or self.metadata is None:

            return "Dataset information is currently unavailable."

        num_records = len(self.metadata)

        summary_parts = [
            f"The dataset contains {num_records} records (e.g., rows or entries)."
        ]

        if not self.column_info:

            summary_parts.append("Column details are not analyzed.")

            return "\n".join(summary_parts)

        summary_parts.append("It has the following columns:")

        for col_name, info in self.column_info.items():

            col_desc = f"- '{col_name}': Type: {info.get('data_type', 'N/A')}"

            if "semantic_type" in info:

                col_desc += f", Semantic Role: {info.get('semantic_type')}"

            if (
                "categories" in info
                and isinstance(info["categories"], list)
                and info["categories"]
            ):

                preview_cats = info["categories"][:3]

                etc_cats = "..." if len(info["categories"]) > 3 else ""

                col_desc += f" (e.g., {', '.join(map(str, preview_cats))}{etc_cats})"

            summary_parts.append(col_desc)

        return "\n".join(summary_parts)

    def _load_data(self):
        """Legacy method - redirects to auto-loading"""
        logger.info("Using legacy _load_data - redirecting to auto-loading")
        # For backward compatibility, treat data_source as excel file path if it exists
        if hasattr(self, 'excel_file_path'):
            self.data_source = self.excel_file_path
        self._auto_load_and_process_data()

    def _prepare_data(self):

        if self.data is None:

            logger.error("Cannot prepare data: self.data is None.")

            return

        self.data = self.data.fillna("")

        self.data = self.data.dropna(how="all").dropna(axis=1, how="all")

        self._analyze_columns()

        self.data["combined_text"] = self.data.apply(
            lambda row: " ".join(
                f"{col}: {val}"
                for col, val in row.items()
                if str(val).strip() != "" and col != "combined_text"
            ),
            axis=1,
        )

        logger.info("Data preparation complete. 'combined_text' created.")

    def _analyze_columns(self):
        """Simplified column analysis without complex pattern detection."""
        if self.data is None:
            return

        logger.info("Analyzing data columns (simplified)...")
        self.column_info = {}

        for col in self.data.columns:
            if col == "combined_text":
                continue

            # Simple data type detection without complex analysis
            original_col_data = self.data[col]
            data_type = "text"  # Default to text

            # Simple numeric detection
            if pd.api.types.is_numeric_dtype(original_col_data.infer_objects()):
                data_type = "numeric"
            # Simple date detection (without sampling to avoid warnings)
            elif original_col_data.dtype == 'datetime64[ns]':
                data_type = "date"

            # Basic info without problematic operations
            empty_count = original_col_data.isna().sum()
            sparsity = empty_count / max(1, len(original_col_data)) if len(original_col_data) > 0 else 0

            self.column_info[col] = {
                "data_type": data_type,
                "sparsity": sparsity,
                "semantic_type": data_type  # Simplified
            }

        logger.info("Column analysis complete (simplified).")

    def _populate_chunk_mapping_from_data(self):
        """Recreate the chunk mapping from data if index exists but mapping was lost"""

        if self.data is None or "combined_text" not in self.data.columns:

            logger.warning(
                "Cannot populate chunk mapping: data or combined_text column missing"
            )

            return

        self.chunk_to_original_doc_mapping = list(range(len(self.data)))

        logger.info(
            f"Populated chunk mapping with {len(self.chunk_to_original_doc_mapping)} entries"
        )

    def _auto_load_and_process_data(self):
        """Automatically detect and load data from file or directory"""
        logger.info(f"Auto-loading data from: {self.data_source}")
        
        data_path = Path(self.data_source)
        
        if data_path.is_file():
            # Single file
            self._load_single_file(data_path)
        elif data_path.is_dir():
            # Directory - scan for supported files
            self._load_from_directory(data_path)
        else:
            logger.error(f"Data source not found: {self.data_source}")
            self._create_dummy_data()
            return
            
        if self.data is not None and "combined_text" in self.data.columns:
            logger.info("Building index from loaded data...")
            self._build_index()
            logger.info(f"System ready! Loaded {self.total_documents} documents from {len(self.loaded_files)} files.")
        else:
            logger.error("No valid data available for indexing.")

    def _load_from_directory(self, directory_path: Path):
        """Load all supported files from a directory"""
        supported_extensions = {'.xlsx', '.xls', '.csv', '.txt', '.json'}
        all_data = []
        loaded_files_count = 0
        
        logger.info(f"Scanning directory: {directory_path}")
        
        for file_path in directory_path.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                logger.info(f"Processing file: {file_path.name}")
                file_data = self._load_single_file(file_path, append_to_main=False)
                if file_data is not None and not file_data.empty:
                    # Add source file info
                    file_data['source_file'] = file_path.name
                    all_data.append(file_data)
                    self.loaded_files.append(str(file_path))
                    loaded_files_count += 1
                    logger.info(f"Successfully loaded {len(file_data)} records from {file_path.name}")
        
        logger.info(f"Found and processed {loaded_files_count} files")
        
        if all_data:
            # Combine all data
            self.data = pd.concat(all_data, ignore_index=True)
            self._prepare_data()
            self.metadata = self.data.to_dict(orient="records")
            self.total_documents = len(self.data)
            logger.info(f"Combined data from {len(all_data)} files into {len(self.data)} records")
            logger.info(f"Loaded files: {self.loaded_files}")
        else:
            logger.warning("No supported files found in directory")
            self._create_dummy_data()

    def _load_single_file(self, file_path: Path, append_to_main: bool = True):
        """Load data from a single file based on its extension"""
        try:
            file_extension = file_path.suffix.lower()
            
            if file_extension in ['.xlsx', '.xls']:
                data = self._load_excel_file(file_path)
            elif file_extension == '.csv':
                data = pd.read_csv(file_path)
            elif file_extension == '.txt':
                data = self._load_text_file(file_path)
            elif file_extension == '.json':
                data = self._load_json_file(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return None
                
            if append_to_main:
                self.data = data
                self._prepare_data()
                self.metadata = self.data.to_dict(orient="records")
                self.loaded_files.append(str(file_path))
                self.total_documents = len(self.data)
                
            return data
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return None

    def _load_excel_file(self, file_path: Path):
        """Load Excel file (supports multiple sheets)"""
        excel_data = pd.read_excel(file_path, sheet_name=None)
        
        # Find the first non-empty sheet
        for sheet_name, df in excel_data.items():
            if not df.empty:
                logger.info(f"Using sheet '{sheet_name}' from {file_path.name}")
                return df
                
        # If all sheets are empty, return empty DataFrame
        logger.warning(f"All sheets empty in {file_path.name}")
        return pd.DataFrame()

    def _load_text_file(self, file_path: Path):
        """Load plain text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into paragraphs or sections
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        return pd.DataFrame({
            'content': paragraphs,
            'document_type': 'text'
        })

    def _load_json_file(self, file_path: Path):
        """Load JSON file and flatten nested structures"""
        import json
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Helper function to safely convert values to strings, handling nested structures
            def safe_convert_value(value):
                if isinstance(value, (list, dict)):
                    return str(value)  # Convert lists and dicts to strings
                elif value is None:
                    return ""
                else:
                    return str(value)
            
            # Handle different JSON structures
            if isinstance(json_data, list):
                # Array of objects - flatten each object
                rows = []
                for item in json_data:
                    if isinstance(item, dict):
                        # Flatten the dictionary and convert lists/dicts to strings
                        flattened_item = {}
                        for key, value in item.items():
                            flattened_item[key] = safe_convert_value(value)
                        rows.append(flattened_item)
                    else:
                        rows.append({'content': safe_convert_value(item)})
                return pd.DataFrame(rows)
                
            elif isinstance(json_data, dict):
                # Check if it contains arrays of objects
                rows = []
                for key, value in json_data.items():
                    if isinstance(value, list) and len(value) > 0:
                        # Add category/type info to each item
                        for item in value:
                            if isinstance(item, dict):
                                # Flatten the item and add category
                                flattened_item = {'category': key}
                                for item_key, item_value in item.items():
                                    flattened_item[item_key] = safe_convert_value(item_value)
                                rows.append(flattened_item)
                            else:
                                rows.append({'content': safe_convert_value(item), 'category': key})
                    else:
                        # Single value
                        rows.append({'content': safe_convert_value(value), 'category': key})
                
                if rows:
                    return pd.DataFrame(rows)
                else:
                    # Simple key-value pairs
                    data_items = []
                    for key, value in json_data.items():
                        data_items.append({'key': key, 'value': safe_convert_value(value)})
                    return pd.DataFrame(data_items)
            else:
                # Simple value
                return pd.DataFrame({'content': [safe_convert_value(json_data)], 'document_type': 'json'})
                
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")
            return pd.DataFrame()

    def _create_dummy_data(self):
        """Create dummy data when no valid files are found"""
        logger.warning("Creating dummy data due to no valid files found")
        self.data = pd.DataFrame({
            'content': ['No data files found. Please add Excel, CSV, TXT, or JSON files to the data directory.'],
            'document_type': 'error'
        })
        self._prepare_data()
        self.metadata = self.data.to_dict(orient="records")
        self.total_documents = 0

    def initialize_system(self, data_source: str = None):
        """Manually initialize the system with optional new data source"""
        if data_source:
            self.data_source = data_source
        
        logger.info(f"Manually initializing system with: {self.data_source}")
        self._auto_load_and_process_data()

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and loaded data information"""
        return {
            'data_source': self.data_source,
            'loaded_files': self.loaded_files,
            'files_loaded_count': len(self.loaded_files) if self.loaded_files else 0,
            'total_documents': self.total_documents,
            'vector_store_ready': hasattr(self, 'vector_store') and self.vector_store is not None and hasattr(self.vector_store, 'collection') and self.vector_store.collection is not None,
            'models_ready': {
                'embedding_model': self.embedding_model is not None,
                'llm': self.llm is not None,
                'sentence_transformer': getattr(self, 'sentence_transformer', None) is not None,
                'reranker': getattr(self, 'reranker', None) is not None if self.use_reranker else False
            }
        }

    def reload_data(self):
        """Reload data from the current data source"""
        logger.info("Reloading data...")
        self.vector_store.reset()
        self.loaded_files = []
        self.total_documents = 0
        self._auto_load_and_process_data()

    def _build_index(self):
        """Build vector store with documents"""
        if self.data is None or "combined_text" not in self.data.columns:
            logger.error("Cannot build index: data or combined_text column missing")
            return

        logger.info("Building index from Excel data...")
        
        # Reset the vector store
        self.vector_store.reset()

        # Prepare documents and metadata
        documents = self.data["combined_text"].tolist()
        chunk_metadata = []
        
        # Convert DataFrame to list of dictionaries for metadata
        for idx, row in self.data.iterrows():
            metadata = {}
            for key, value in row.items():
                if key == 'combined_text':
                    continue
                    
                # Handle different data types including lists and nested structures
                if isinstance(value, (list, tuple, np.ndarray, dict)):
                    # Convert lists/arrays/dicts to strings to avoid unhashable type errors
                    metadata[key] = str(value)
                elif pd.isna(value):
                    metadata[key] = ""  # ChromaDB doesn't accept None, use empty string instead
                elif isinstance(value, (pd.Timestamp, datetime.datetime)):
                    metadata[key] = value.isoformat()
                elif isinstance(value, (int, float, bool)):
                    metadata[key] = value
                else:
                    metadata[key] = str(value)
                    
            chunk_metadata.append(metadata)

        # Add documents to vector store
        self.vector_store.add_documents(
            texts=documents,
            metadata=chunk_metadata
        )
        
        logger.info(f"Added {len(documents)} documents to vector store")

        # Build sentence transformer embeddings if enabled
        if self.use_sentence_transformers and self.sentence_transformer:
            try:
                logger.info("Generating sentence transformer embeddings...")
                self.st_embeddings = self.sentence_transformer.encode(
                    documents, convert_to_tensor=False
                )
                logger.info(
                    f"Generated {len(self.st_embeddings)} sentence transformer embeddings"
                )
            except Exception as e:
                logger.error(f"Failed to generate sentence transformer embeddings: {e}")
                self.st_embeddings = None

        # Note: ChromaDB handles embeddings internally, so we don't need to generate them manually
        # The vector store will handle embedding generation when documents are added
        
        logger.info(f"Index built successfully with {len(documents)} documents")

    def _sentence_transformer_retrieve(
        self, query: str, k: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve using sentence transformers"""

        if (
            not self.use_sentence_transformers
            or self.sentence_transformer is None
            or self.st_embeddings is None
        ):

            return []

        logger.info(f"Performing sentence transformer retrieval for: {query[:100]}...")

        try:

            # Encode query

            query_embedding = self.sentence_transformer.encode([query])

            # Calculate cosine similarities

            similarities = cosine_similarity(query_embedding, self.st_embeddings)[0]

            # Get top k results

            top_indices = np.argsort(similarities)[::-1][:k]

            retrieved_docs = []

            for i, idx in enumerate(top_indices):

                if idx >= len(self.chunk_to_original_doc_mapping):

                    continue

                doc_idx = self.chunk_to_original_doc_mapping[idx]

                if doc_idx >= len(self.metadata):

                    continue

                similarity_score = float(similarities[idx])

                doc_content = self.metadata[doc_idx].copy()

                if "combined_text" in doc_content:

                    del doc_content["combined_text"]

                retrieved_docs.append(
                    {
                        "id": doc_idx,
                        "content": doc_content,
                        "similarity": similarity_score,
                        "retrieval_method": "sentence_transformer",
                    }
                )

            logger.info(
                f"Sentence transformer retrieved {len(retrieved_docs)} documents."
            )

            return retrieved_docs

        except Exception as e:

            logger.error(
                f"Error during sentence transformer retrieval: {e}", exc_info=True
            )

            return []

    def _embeddings_retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve using vector store"""
        logger.info(f"Performing vector store retrieval for: {query[:100]}...")

        try:
            # Search in vector store
            results = self.vector_store.search(query=query, k=k)
            
            retrieved_docs = []
            for hit in results:
                retrieved_docs.append({
                    "id": hit['id'],
                    "content": hit['content'],
                    "similarity": 1.0 - hit['similarity'],  # Convert distance to similarity
                    "retrieval_method": "vector_store",
                    "text": hit['text']
                })

            logger.info(f"Vector store retrieved {len(retrieved_docs)} documents.")
            return retrieved_docs

        except Exception as e:
            logger.error(f"Error during vector store retrieval: {e}", exc_info=True)
            return []

    def _rerank_documents(
        self, query: str, documents: List[Dict[str, Any]], top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Re-rank documents using cross-encoder"""

        if not self.use_reranker or self.reranker is None or not documents:

            return documents

        logger.info(f"Re-ranking {len(documents)} documents...")

        try:

            # Prepare query-document pairs for reranking

            query_doc_pairs = []

            for doc in documents:

                # Get document text for reranking
                doc_text = ""
                
                # Try to get text from the document itself first
                if "text" in doc:
                    doc_text = doc["text"]
                elif "content" in doc and isinstance(doc["content"], dict):
                    # Concatenate content fields
                    doc_text = " ".join(
                        f"{k}: {v}" for k, v in doc["content"].items() if str(v).strip()
                    )
                elif "content" in doc and isinstance(doc["content"], str):
                    doc_text = doc["content"]
                else:
                    # Fallback: try to access metadata if available
                    try:
                        doc_id = doc.get("id")
                        if doc_id is not None and hasattr(self, 'metadata') and self.metadata:
                            if isinstance(self.metadata, list) and isinstance(doc_id, int) and 0 <= doc_id < len(self.metadata):
                                metadata_item = self.metadata[doc_id]
                                if "combined_text" in metadata_item:
                                    doc_text = metadata_item["combined_text"]
                                else:
                                    doc_text = " ".join(
                                        f"{k}: {v}" for k, v in metadata_item.items() 
                                        if str(v).strip() and k != "combined_text"
                                    )
                    except (TypeError, IndexError, KeyError) as e:
                        logger.warning(f"Could not access metadata for doc {doc.get('id')}: {e}")
                        doc_text = str(doc)  # Last resort

                if not doc_text.strip():
                    doc_text = str(doc)  # Ultimate fallback

                query_doc_pairs.append([query, doc_text])

            # Get reranking scores

            rerank_scores = self.reranker.predict(query_doc_pairs)

            # Add rerank scores to documents and sort

            for i, doc in enumerate(documents):

                doc["rerank_score"] = float(rerank_scores[i])

            # Sort by rerank score (higher is better for cross-encoder)

            reranked_docs = sorted(
                documents, key=lambda x: x["rerank_score"], reverse=True
            )

            # Limit to top_k if specified

            if top_k:

                reranked_docs = reranked_docs[:top_k]

            logger.info(
                f"Re-ranking complete. Top document rerank score: {reranked_docs[0]['rerank_score']:.4f}"
            )

            return reranked_docs

        except Exception as e:

            logger.error(f"Error during re-ranking: {e}", exc_info=True)

            return documents

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Enhanced retrieval combining multiple methods"""

        logger.info(f"Starting enhanced retrieval for query: {query[:100]}...")

        all_retrieved_docs = []

        # Method 1: Sentence Transformer retrieval

        if self.use_sentence_transformers:

            st_docs = self._sentence_transformer_retrieve(
                query, k * 2
            )  # Get more for diversity

            all_retrieved_docs.extend(st_docs)

        # Method 2: Embeddings-based retrieval

        embedding_docs = self._embeddings_retrieve(query, k * 2)

        all_retrieved_docs.extend(embedding_docs)

        # Remove duplicates based on document ID

        seen_ids = set()

        unique_docs = []

        for doc in all_retrieved_docs:

            if doc["id"] not in seen_ids:

                unique_docs.append(doc)

                seen_ids.add(doc["id"])

        logger.info(f"Combined retrieval found {len(unique_docs)} unique documents")

        # Re-rank if enabled

        if self.use_reranker:

            final_docs = self._rerank_documents(query, unique_docs, k)

        else:

            # Sort by similarity and take top k

            final_docs = sorted(
                unique_docs, key=lambda x: x.get("similarity", 0), reverse=True
            )[:k]

        logger.info(f"Final retrieval returned {len(final_docs)} documents")

        return final_docs

    def format_retrieved_document_for_llm(self, doc: Dict) -> str:
        """Format a retrieved document for inclusion in the LLM context"""

        formatted_content = [f"DOCUMENT ID: {doc['id']}"]

        # Add retrieval method and scores

        if "retrieval_method" in doc:

            formatted_content.append(f"Retrieval Method: {doc['retrieval_method']}")

        if "similarity" in doc:

            formatted_content.append(f"Similarity Score: {doc['similarity']:.4f}")

        if "rerank_score" in doc:

            formatted_content.append(f"Rerank Score: {doc['rerank_score']:.4f}")

        if "content" not in doc or not doc["content"]:

            return (
                "\n".join(formatted_content)
                + "\nNo content available for this document."
            )

        for key, value in doc["content"].items():

            value_str = str(value) if value is not None else ""

            if value_str.strip():

                formatted_key = key.replace("_", " ").title()

                formatted_content.append(f"{formatted_key}: {value_str}")

        return "\n".join(formatted_content)

    def analyze_patterns(self, retrieved_docs: List[Dict]) -> Dict[str, Any]:

        if not retrieved_docs:
            return {"count": 0, "patterns": {}, "date_range": None}

        analysis = {"count": len(retrieved_docs), "patterns": {}, "date_range": None}

        for col_name, info in self.column_info.items():

            if info.get("semantic_type") == "category" or (
                info.get("data_type") == "text"
                and info.get("value_diversity", 1.0) < 0.5
            ):

                value_counts = {}

                for doc in retrieved_docs:

                    value = doc["content"].get(col_name)

                    if value is not None and str(value).strip():

                        value_str = str(value)

                        value_counts[value_str] = value_counts.get(value_str, 0) + 1

                if value_counts:
                    analysis["patterns"][col_name] = value_counts

            if info.get("data_type") == "date":

                dates = []

                for doc in retrieved_docs:

                    date_val = doc["content"].get(col_name)

                    if date_val:

                        try:
                            dt = pd.to_datetime(date_val, errors="coerce")

                        except:
                            dt = None

                        if pd.notna(dt):
                            dates.append(dt)

                if dates:

                    min_date, max_date = min(dates), max(dates)

                    if analysis["date_range"] is None:

                        analysis["date_range"] = {
                            "column": col_name,
                            "min_date": min_date.strftime("%Y-%m-%d"),
                            "max_date": max_date.strftime("%Y-%m-%d"),
                            "span_days": (max_date - min_date).days,
                        }

        logger.info(f"Pattern analysis complete: {analysis}")

        return analysis

    def generate_response(self, query: str, k: int = 3) -> Dict[str, Any]:
        if not self.chain:

            logger.error("Cannot generate response: LLM chain not initialized.")

            return {
                "response": "System error: Unable to process request.",
                "retrieved_docs": [],
                "pattern_analysis": {"count": 0},
            }

        logger.info(f"Generating enhanced response for query: {query[:100]}..., k={k}")
        
        # Add user message to chat memory
        self.chat_memory.add_user_message(query)

        dataset_overview_summary = self._generate_dataset_overview_summary()

        retrieved_docs = self.retrieve(query, k)

        if retrieved_docs:

            retrieved_documents_llm_context = "\n\n===\n\n".join(
                [self.format_retrieved_document_for_llm(doc) for doc in retrieved_docs]
            )

        else:

            retrieved_documents_llm_context = (
                "No specific documents were found to be highly relevant to this query."
            )

            logger.warning("No relevant documents found for the query to pass to LLM.")

        pattern_analysis = self.analyze_patterns(retrieved_docs)

        pattern_analysis_llm_summary_parts = [
            "Summary of Patterns Found in Retrieved Documents:"
        ]

        if pattern_analysis["count"] > 0:

            pattern_analysis_llm_summary_parts.append(
                f"- Number of similar records found: {pattern_analysis['count']}"
            )

            if pattern_analysis.get("date_range"):

                dr = pattern_analysis["date_range"]

                pattern_analysis_llm_summary_parts.append(
                    f"- These records span from {dr['min_date']} to {dr['max_date']} ({dr['span_days']} days) in the '{dr['column']}' field"
                )

            if pattern_analysis.get("patterns"):

                pattern_analysis_llm_summary_parts.append(
                    "- Common patterns identified:"
                )

                for field, value_counts in pattern_analysis["patterns"].items():

                    top_values = sorted(
                        value_counts.items(), key=lambda x: x[1], reverse=True
                    )[:3]

                    field_display = field.replace("_", " ").title()

                    values_display = ", ".join(
                        [f"{val} ({count}x)" for val, count in top_values]
                    )

                    pattern_analysis_llm_summary_parts.append(
                        f" * {field_display}: {values_display}"
                    )

        else:

            pattern_analysis_llm_summary_parts.append(
                "- No specific patterns identified in the retrieved documents."
            )

        pattern_analysis_llm_summary = "\n".join(pattern_analysis_llm_summary_parts)

        try:

            logger.info("Calling LLM chain to generate response...")

            # Get conversation context for the prompt
            conversation_context = self.chat_memory.get_recent_context(num_turns=3)

            response = self.chain.invoke({
                "conversation_context": conversation_context,
                "dataset_overview": dataset_overview_summary,
                "retrieved_documents_context": retrieved_documents_llm_context,
                "pattern_analysis_summary": pattern_analysis_llm_summary,
                "query": query,
            })
            
            response_text = response['text'] if isinstance(response, dict) else response

            logger.info(
                f"LLM response generated successfully. Length: {len(response_text)} characters"
            )

        except Exception as e:

            logger.error(f"Error generating LLM response: {e}", exc_info=True)

            response_text = f"I apologize, but I encountered an error while processing your query: {str(e)}"

        # Add assistant response to chat memory
        self.chat_memory.add_assistant_message(response_text)

        # Prepare response with serializable data

        serializable_retrieved_docs = make_serializable(retrieved_docs)

        serializable_pattern_analysis = make_serializable(pattern_analysis)

        return {
            "response": response_text,
            "retrieved_docs": serializable_retrieved_docs,
            "pattern_analysis": serializable_pattern_analysis,
            "dataset_overview": dataset_overview_summary,
            "query": query,
        }

    # Chit-chat detection removed temporarily - will be re-implemented later
    # when the PoC is fully ready
    
    def _format_response_for_streaming(self, response_text: str) -> str:
        """Format response text for streaming by ensuring proper structure."""
        # Clean up any extra whitespace and ensure proper formatting
        lines = response_text.strip().split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

    def _html_format_response(self, response_text: str) -> str:
        """Convert response text to HTML format for streaming."""
        import re
        
        # Split into paragraphs
        paragraphs = response_text.split('\n\n')
        html_parts = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # Check if it's a header (ends with colon and is relatively short)
            if paragraph.endswith(':') and len(paragraph) < 100 and '\n' not in paragraph:
                html_parts.append(f'<strong class="section-header">{paragraph}</strong>')
            # Check if it contains bullet points
            elif paragraph.startswith('*') or paragraph.startswith('-') or '\n*' in paragraph or '\n-' in paragraph:
                lines = paragraph.split('\n')
                bullet_group = []
                regular_lines = []
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('*') or line.startswith('-'):
                        # If we have regular lines before this bullet, add them first
                        if regular_lines:
                            html_parts.append(f'<p class="response-paragraph">{"<br>".join(regular_lines)}</p>')
                            regular_lines = []
                        # Remove the bullet character and wrap in bullet div
                        content = line[1:].strip()
                        if content:  # Only add non-empty bullet items
                            bullet_group.append(f'<div class="bullet-item">{content}</div>')
                    elif line:
                        regular_lines.append(line)
                
                # Add any collected bullet items
                if bullet_group:
                    html_parts.extend(bullet_group)
                # Add any remaining regular lines
                if regular_lines:
                    html_parts.append(f'<p class="response-paragraph">{"<br>".join(regular_lines)}</p>')
            else:
                # Regular paragraph
                html_parts.append(f'<p class="response-paragraph">{paragraph}</p>')
        
        return ''.join(html_parts)

    async def generate_response_stream(self, query: str, k: int = 3):
        if not self.chain:
            logger.error("Cannot generate response: LLM chain not initialized.")
            yield {
                "type": "error",
                "content": "System error: Unable to process request.",
                "done": True
            }
            return

        logger.info(f"Generating streaming response for query: {query[:100]}..., k={k}")
        
        # Add user message to chat memory
        self.chat_memory.add_user_message(query)

        # Generate dataset overview and retrieve documents
        try:
            dataset_overview_summary = self._generate_dataset_overview_summary()
            retrieved_docs = self.retrieve(query, k)

            # Prepare context
            if retrieved_docs:
                retrieved_documents_llm_context = "\n\n===\n\n".join(
                    [self.format_retrieved_document_for_llm(doc) for doc in retrieved_docs]
                )
            else:
                retrieved_documents_llm_context = (
                    "No specific documents were found to be highly relevant to this query."
                )

            pattern_analysis = self.analyze_patterns(retrieved_docs)

            # Build pattern analysis summary
            pattern_analysis_llm_summary_parts = [
                "Summary of Patterns Found in Retrieved Documents:"
            ]

            if pattern_analysis["count"] > 0:
                pattern_analysis_llm_summary_parts.append(
                    f"- Number of similar records found: {pattern_analysis['count']}"
                )

                if pattern_analysis.get("date_range"):
                    dr = pattern_analysis["date_range"]
                    pattern_analysis_llm_summary_parts.append(
                        f"- These records span from {dr['min_date']} to {dr['max_date']} ({dr['span_days']} days) in the '{dr['column']}' field"
                    )

                if pattern_analysis.get("patterns"):
                    pattern_analysis_llm_summary_parts.append(
                        "- Common patterns identified:"
                    )

                    for field, value_counts in pattern_analysis["patterns"].items():
                        top_values = sorted(
                            value_counts.items(), key=lambda x: x[1], reverse=True
                        )[:3]

                        field_display = field.replace("_", " ").title()
                        values_display = ", ".join(
                            [f"{val} ({count}x)" for val, count in top_values]
                        )

                        pattern_analysis_llm_summary_parts.append(
                            f" * {field_display}: {values_display}"
                        )
            else:
                pattern_analysis_llm_summary_parts.append(
                    "- No specific patterns identified in the retrieved documents."
                )

            pattern_analysis_llm_summary = "\n".join(pattern_analysis_llm_summary_parts)

            # Generate the complete response first
            conversation_context = self.chat_memory.get_recent_context(num_turns=3)
            
            response = self.chain.invoke({
                "conversation_context": conversation_context,
                "dataset_overview": dataset_overview_summary,
                "retrieved_documents_context": retrieved_documents_llm_context,
                "pattern_analysis_summary": pattern_analysis_llm_summary,
                "query": query,
            })
            
            response_text = response['text'] if isinstance(response, dict) else response

            # Add assistant response to chat memory
            self.chat_memory.add_assistant_message(response_text)

            # Format the response properly before streaming
            formatted_response = self._format_response_for_streaming(response_text)

            # First format the entire response as HTML
            import re
            formatted_html = self._html_format_response(response_text)
            
            # Split the formatted HTML into meaningful chunks for streaming
            # This approach preserves complete HTML tags and word boundaries
            chunks = []
            current_chunk = ""
            in_tag = False
            
            i = 0
            while i < len(formatted_html):
                char = formatted_html[i]
                
                if char == '<':
                    # If we have content in current_chunk, add it
                    if current_chunk.strip():
                        chunks.append(current_chunk)
                        current_chunk = ""
                    in_tag = True
                    current_chunk += char
                elif char == '>' and in_tag:
                    in_tag = False
                    current_chunk += char
                    # Add complete tag as a chunk
                    chunks.append(current_chunk)
                    current_chunk = ""
                elif in_tag:
                    current_chunk += char
                elif char == ' ':
                    if current_chunk.strip():
                        current_chunk += char
                        chunks.append(current_chunk)
                        current_chunk = ""
                else:
                    current_chunk += char
                
                i += 1
            
            # Add any remaining content
            if current_chunk.strip():
                chunks.append(current_chunk)
            
            # Stream the formatted chunks
            for chunk in chunks:
                if chunk.strip():  # Only send non-empty chunks
                    yield {
                        "type": "formatted_token",
                        "content": chunk,
                        "done": False
                    }
                    # Add a small delay for realistic streaming effect
                    import asyncio
                    await asyncio.sleep(0.05)  # 50ms delay between chunks

            # Send final completion signal
            yield {
                "type": "done",
                "content": "",
                "done": True,
                "metadata": {
                    "retrieved_docs": make_serializable(retrieved_docs),
                    "pattern_analysis": make_serializable(pattern_analysis),
                    "dataset_overview": dataset_overview_summary,
                    "query": query,
                }
            }

        except Exception as e:
            logger.error(f"Error generating streaming response: {e}", exc_info=True)
            yield {
                "type": "error",
                "content": f"I apologize, but I encountered an error while processing your query: {str(e)}",
                "done": True
            }


# Convenience functions for easy system initialization

def create_rag_system(data_source: str = "data", **kwargs) -> EnhancedAdaptiveRAGSystem:
    """
    Create a RAG system with automatic data loading.
    
    Args:
        data_source: Path to data file or directory (default: "data")
        **kwargs: Additional parameters for EnhancedAdaptiveRAGSystem
    
    Returns:
        Initialized RAG system ready to use
    """
    return EnhancedAdaptiveRAGSystem(data_source=data_source, **kwargs)

def quick_start(data_directory: str = "data") -> EnhancedAdaptiveRAGSystem:
    """
    Quick start function - creates a ready-to-use RAG system.
    
    Args:
        data_directory: Directory containing your data files
        
    Returns:
        Fully initialized RAG system
    """
    logger.info(f"🚀 Quick starting RAG system with data from: {data_directory}")
    system = create_rag_system(
        data_source=data_directory,
        auto_initialize=True,
        use_sentence_transformers=True,
        use_reranker=True
    )
    
    status = system.get_system_status()
    loaded_files = status.get('loaded_files', []) or []  # Handle None case
    total_docs = status.get('total_documents', 0) or 0
    logger.info(f"✅ System ready! Loaded {total_docs} documents from {len(loaded_files)} files")
    
    return system
