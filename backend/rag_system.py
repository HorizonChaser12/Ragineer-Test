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
from typing import Union, Optional, List, Dict, Any, Callable
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import Document
import datetime
import os
import uuid
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import torch
# Import will happen inside __init__ for better error handling

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
        
        # Initialize vector store with persistent storage and custom embedding function
        # We'll set up the actual embedding function after model initialization
        self.vector_store = None
        self.chunk_to_original_doc_mapping: List[int] = []
        
        # Initialize chat memory
        from db.chat_memory import ChatMemoryStore
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
            
            # Create embedding function wrapper for the vector store
            def embedding_function_wrapper(text):
                """Wrapper for embedding model to be used with ChromaDB"""
                if isinstance(text, list):
                    # Handle batch embedding
                    return [self.embedding_model.embed_query(t) for t in text]
                # Handle single text embedding
                return self.embedding_model.embed_query(text)
                
            # We'll initialize the vector store when building the index
            self.vector_store = None
            logger.info("Vector store will be initialized during index building")
            
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
You are a friendly and knowledgeable technical support assistant who helps people understand defect information in a conversational, human way. Your goal is to provide helpful insights while being approachable and easy to understand.

CRITICAL HTML FORMATTING REQUIREMENTS:
1. ALWAYS write complete paragraphs: Each <p></p> must contain a full sentence or thought
2. NEVER put <strong></strong> tags on separate lines - they must be INLINE within sentences
3. CORRECT: "<p>I found several <strong>accessibility issues</strong> that need attention.</p>"
4. WRONG: "<p>I found several</p><p><strong>accessibility issues</strong></p><p>that need attention.</p>"
5. Keep all HTML tags on the SAME LINE as the text they contain
6. Write flowing, natural sentences with emphasis INSIDE them

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

1. BE CONVERSATIONAL AND HUMAN
- Write like you're having a friendly conversation with a colleague
- Use natural language, not robotic bullet points
- Show personality and be engaging
- Use transitional phrases like "I found some interesting things..." or "Here's what stood out to me..."

2. PROVIDE CONTEXT AND INSIGHTS
- Don't just list data - explain what it means
- Highlight patterns and trends you notice
- Compare severity levels and explain their significance
- Mention any concerning trends or positive findings

3. STRUCTURE NATURALLY
- Start with a friendly opening that summarizes what you found
- Group related issues together with explanatory text
- Use paragraphs with clear topic sentences
- End with actionable insights or key takeaways

4. BE HELPFUL AND INFORMATIVE
- Include defect IDs naturally in sentences (not as separate bullet points)
- Explain technical terms when needed
- Provide context about severity levels and what they mean
- Suggest patterns or next steps when appropriate

WRITING STYLE EXAMPLES:

For Simple Greetings:
User: "Hello" → "<p>Hi there! I'm your defect analysis assistant. How can I help you today?</p>"
User: "Thanks" → "<p>You're welcome! Let me know if you need help with anything else.</p>"
User: "How are you?" → "<p>I'm doing well and ready to help! What would you like to know about the system defects?</p>"

For Defect Queries - CORRECT FORMATTING WITH EMPHASIS:
"<p>Looking at the mobile app defects, I found several <strong>accessibility issues</strong> that need attention. The most serious is a <strong>critical startup crash</strong> (LIFE-1095) marked as <strong>urgent priority</strong> - users literally can't open the app.</p>
<p>I also noticed some <strong>accessibility problems</strong> with low contrast text (LIFE-1013 and LIFE-1041). While these are <strong>low severity</strong>, they're marked <strong>urgent</strong> and <strong>medium priority</strong> respectively, which makes sense for usability.</p>
<p>On the <strong>performance side</strong>, there's excessive battery usage (LIFE-1060) that's a <strong>medium severity issue</strong> but could really impact user experience over time.</p>"

WRONG - DO NOT DO THIS:
"<p>First off, there are a few</p>
<p><strong>accessibility issues</strong></p>
<p>that need attention.</p>"

CRITICAL FORMATTING RULES:
1. ALWAYS keep <strong></strong> tags INSIDE the paragraph where they belong
2. NEVER put emphasis tags on separate lines or in separate paragraphs
3. Write complete sentences with emphasis inline: "<p>I found a <strong>critical issue</strong> that affects users.</p>"
4. DO NOT break sentences across paragraphs - each <p></p> should contain complete thoughts

FORMAT GUIDELINES:
- Use <p></p> for complete paragraphs - NEVER break a thought mid-sentence
- Each paragraph should contain one complete idea or topic
- Use <strong></strong> for emphasis WITHIN paragraphs, not as separate elements
- Include defect IDs naturally within sentences
- Write complete, flowing sentences without artificial breaks
- Example: "<p>I found a <strong>critical issue</strong> where the app crashes on startup (LIFE-1095). This is marked as urgent priority.</p>"
- NEVER write: "<p>First off, there's a</p><p><strong>critical issue</strong></p>"

RESPONSE APPROACH:

FIRST - ANALYZE THE QUERY TYPE:
1. **Simple Greetings/Chitchat** (hello, hi, how are you, thanks, etc.):
   - Respond with a brief, friendly greeting in 1-2 sentences
   - Do NOT include defect information unless specifically asked
   - Example: "<p>Hello! I'm here to help you with any questions about defects or technical issues. What would you like to know?</p>"

2. **Defect/Technical Queries** (anything about bugs, issues, defects, problems):
   - Use the full conversational analysis approach below
   - Include relevant defect information and insights

3. **Off-topic Questions** (weather, politics, etc.):
   - Politely redirect: "<p>I can only help with questions about documented defects and technical issues. What would you like to know about the system?</p>"

FOR DEFECT/TECHNICAL QUERIES:
1. Start with a friendly, contextual opening
2. Organize findings into logical themes/groups
3. Explain what the data means, not just what it says
4. Include severity context and business impact
5. End with insights, patterns, or recommendations

TONE: Friendly, knowledgeable, conversational, helpful
AVOID: Robotic bullet points, dry data dumps, overly formal language
INCLUDE: Natural explanations, context, insights, personality

FORMATTING FOR EMPHASIS:
- Use <strong></strong> to emphasize:
  * Severity levels: <strong>Critical</strong>, <strong>High</strong>, <strong>Medium</strong>, <strong>Low</strong>
  * Priority levels: <strong>Urgent</strong>, <strong>High Priority</strong>, <strong>Medium Priority</strong>
  * Issue categories: <strong>accessibility problems</strong>, <strong>functionality issues</strong>, <strong>performance issues</strong>
  * Key defect statuses: <strong>resolved</strong>, <strong>open</strong>, <strong>in progress</strong>
  * Important timeframes: <strong>within a day</strong>, <strong>recently found</strong>
  * Critical phrases: <strong>major issue</strong>, <strong>prevents users from</strong>, <strong>impacts usability</strong>

EXAMPLES WITH PROPER EMPHASIS:
"I found a <strong>critical issue</strong> where the app crashes on startup (LIFE-1095). This is marked as <strong>urgent priority</strong> since users can't even access the app."

"There are several <strong>accessibility problems</strong> including low contrast text (LIFE-1013, LIFE-1041). While marked as <strong>low severity</strong>, they have <strong>urgent</strong> and <strong>medium priority</strong> respectively."

"On the <strong>performance side</strong>, there's excessive battery usage (LIFE-1060) which is a <strong>medium severity issue</strong> that could impact user experience significantly."

Remember: You're not just reporting data - you're helping someone understand what's happening with their system in a friendly, human way.

Response (start directly with HTML):
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
        try:
            logger.info(f"Auto-loading data from: {self.data_source}")
            
            data_path = Path(self.data_source)
            
            if not data_path.exists():
                raise FileNotFoundError(f"Data source not found: {self.data_source}")
                
            if data_path.is_file():
                logger.info(f"Loading single file: {data_path}")
                self._load_single_file(data_path)
            elif data_path.is_dir():
                logger.info(f"Loading from directory: {data_path}")
                self._load_from_directory(data_path)
            else:
                raise ValueError(f"Data source is neither file nor directory: {self.data_source}")
                
            # Verify data was loaded
            if self.data is None or self.data.empty:
                raise RuntimeError("No data was loaded")
                
            # Verify we have the necessary columns
            if "combined_text" not in self.data.columns:
                raise ValueError("Required column 'combined_text' missing from loaded data")
                
            logger.info(f"Successfully loaded {len(self.data)} records with required columns")
            
        except Exception as e:
            logger.error(f"Error during data loading: {e}", exc_info=True)
            self._create_dummy_data()
            raise RuntimeError(f"Failed to load data: {str(e)}")
            
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
        try:
            if data_source:
                logger.info(f"Updating data source from {self.data_source} to {data_source}")
                self.data_source = data_source
            
            logger.info(f"Manually initializing system with data source: {self.data_source}")
            
            # Verify data source exists
            if not os.path.exists(self.data_source):
                raise FileNotFoundError(f"Data source not found at: {self.data_source}")
                
            self._auto_load_and_process_data()
            
            # Verify initialization was successful
            if not hasattr(self, '_documents') or not self._documents:
                raise RuntimeError("No documents were loaded during initialization")
                
            if not hasattr(self, 'vector_store') or not self.vector_store:
                raise RuntimeError("Vector store failed to initialize")
                
            logger.info(f"System initialized successfully with {len(self._documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}", exc_info=True)
            raise

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and loaded data information"""
        try:
            # Check vector store status
            vector_store_stats = {}
            vector_store_ready = False
            if hasattr(self, 'vector_store') and self.vector_store is not None:
                try:
                    vector_store_stats = self.vector_store.get_collection_stats()
                    vector_store_ready = (hasattr(self.vector_store, 'collection') 
                                        and self.vector_store.collection is not None)
                except Exception as e:
                    logger.error(f"Error getting vector store stats: {e}")
                    vector_store_stats = {"error": str(e)}
            
            # Check model status
            models_ready = {
                'embedding_model': self.embedding_model is not None,
                'llm': self.llm is not None,
                'sentence_transformer': getattr(self, 'sentence_transformer', None) is not None,
                'reranker': getattr(self, 'reranker', None) is not None if self.use_reranker else True
            }
            
            # Prepare basic status
            status = {
                'data_source': self.data_source,
                'loaded_files': self.loaded_files or [],
                'files_loaded_count': len(self.loaded_files) if self.loaded_files else 0,
                'total_documents': self.total_documents,
                'vector_store_ready': vector_store_ready,
                'vector_store_stats': vector_store_stats,
                'models_ready': models_ready
            }
            
            # Add computed status
            status['system_ready'] = (
                vector_store_ready 
                and status['files_loaded_count'] > 0 
                and all(models_ready.values())
            )
            
            # Add detailed status message
            if not status['system_ready']:
                if not vector_store_ready:
                    status['status_message'] = "Vector store not ready"
                elif status['files_loaded_count'] == 0:
                    status['status_message'] = "No files loaded"
                elif not all(models_ready.values()):
                    unready_models = [name for name, ready in models_ready.items() if not ready]
                    status['status_message'] = f"Models not ready: {', '.join(unready_models)}"
            else:
                status['status_message'] = "System fully operational"
                
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}", exc_info=True)
            return {
                'error': str(e),
                'system_ready': False,
                'status_message': f"Error getting system status: {str(e)}"
            }

    def reload_data(self):
        """Reload data from the current data source"""
        try:
            logger.info("Reloading data...")
            
            # Reset state
            self.loaded_files = []
            self.total_documents = 0
            
            # Reset vector store if it exists
            if self.vector_store and hasattr(self.vector_store, 'reset'):
                logger.info("Resetting vector store")
                self.vector_store.reset()
            else:
                logger.info("No existing vector store to reset")
                
            # Load and process new data
            self._auto_load_and_process_data()
            
        except Exception as e:
            logger.error(f"Error reloading data: {e}", exc_info=True)
            raise RuntimeError(f"Failed to reload data: {str(e)}")

    def _build_index(self):
        """Build vector store with documents"""
        try:
            if self.data is None or "combined_text" not in self.data.columns:
                logger.error("Cannot build index: data or combined_text column missing")
                return

            logger.info("Building index from Excel data...")
            
            # Check if embedding model is available
            if not self.embedding_model:
                raise RuntimeError("Embedding model not initialized")
            
            # Ensure vector store is initialized
            if not self.vector_store:
                # Create embedding function wrapper
                def embedding_function_wrapper(text):
                    if isinstance(text, list):
                        return [self.embedding_model.embed_query(t) for t in text]
                    return self.embedding_model.embed_query(text)
                
                # Import and initialize vector store - handle different import paths
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if root_dir not in sys.path:
                    sys.path.insert(0, root_dir)  # Use insert(0, ...) to prioritize this path
                
                # Debug: Print the paths being checked
                logger.info(f"Root directory: {root_dir}")
                logger.info(f"Current sys.path entries: {[p for p in sys.path if 'Ragineer' in p]}")
                
                # Check if the db module exists
                db_path = os.path.join(root_dir, 'db')
                simple_store_path = os.path.join(db_path, 'simple_store.py')
                logger.info(f"Checking for db module at: {db_path}")
                logger.info(f"Checking for simple_store.py at: {simple_store_path}")
                logger.info(f"DB path exists: {os.path.exists(db_path)}")
                logger.info(f"simple_store.py exists: {os.path.exists(simple_store_path)}")
                
                try:
                    from db.simple_store import SimpleVectorStore
                    logger.info("Successfully imported SimpleVectorStore")
                except ImportError as e:
                    logger.error(f"Failed to import SimpleVectorStore: {e}")
                    # Try alternative import path
                    try:
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("simple_store", simple_store_path)
                        simple_store_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(simple_store_module)
                        SimpleVectorStore = simple_store_module.SimpleVectorStore
                        logger.info("Successfully imported SimpleVectorStore using direct file import")
                    except Exception as e2:
                        logger.error(f"Failed direct import as well: {e2}")
                        raise RuntimeError("Vector store module not found. Please check installation.")
                
                # Create the vector store with absolute path
                persist_dir = os.path.join(root_dir, "db", "chroma_db")
                os.makedirs(persist_dir, exist_ok=True)
                
                logger.info(f"Initializing vector store with persist_directory: {persist_dir}")
                self.vector_store = SimpleVectorStore(
                    collection_name="documents",
                    persist_directory=persist_dir,
                    embedding_function=embedding_function_wrapper
                )
                logger.info("Vector store initialized successfully")
                logger.info("Vector store initialized")
            
            # Reset the vector store
            if hasattr(self.vector_store, 'reset'):
                logger.info("Resetting vector store")
                self.vector_store.reset()
            else:
                logger.error("Vector store missing reset method")
                raise RuntimeError("Invalid vector store configuration")
                
        except Exception as e:
            logger.error(f"Error building index: {e}", exc_info=True)
            raise RuntimeError(f"Failed to build index: {str(e)}")

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

        # Generate embeddings for documents if using sentence transformers
        embeddings = None
        if self.use_sentence_transformers and self.sentence_transformer:
            try:
                logger.info("Generating sentence transformer embeddings...")
                self.st_embeddings = self.sentence_transformer.encode(
                    documents, convert_to_tensor=False
                )
                logger.info(
                    f"Generated {len(self.st_embeddings)} sentence transformer embeddings"
                )
                # Use these embeddings for the vector store
                embeddings = self.st_embeddings
            except Exception as e:
                logger.error(f"Failed to generate sentence transformer embeddings: {e}")
                self.st_embeddings = None
                embeddings = None

        # Add documents to vector store
        try:
            # Add documents to vector store (it will handle embeddings appropriately)
            self.vector_store.add_documents(
                texts=documents,
                metadata=chunk_metadata,
                embeddings=embeddings  # May be None, and that's okay
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}", exc_info=True)
        
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
        """Retrieve using vector store with optional pre-computed embeddings"""
        logger.info(f"Performing vector store retrieval for: {query[:100]}...")

        try:
            # Generate query embedding if using sentence transformers
            query_embedding = None
            if self.use_sentence_transformers and self.sentence_transformer:
                try:
                    query_embedding = self.sentence_transformer.encode(query, convert_to_tensor=False)
                    if isinstance(query_embedding, np.ndarray):
                        query_embedding = query_embedding.tolist()
                    logger.info("Using sentence transformer for query embedding")
                except Exception as e:
                    logger.error(f"Error generating query embedding with sentence transformer: {e}")
                    query_embedding = None
            
            # Search in vector store
            results = self.vector_store.search(
                query=query, 
                k=k,
                query_embedding=query_embedding
            )
            
            if not results:
                logger.warning("No results returned from vector store search")
                return []
            
            retrieved_docs = []
            for hit in results:
                # Handle the updated format from our enhanced SimpleVectorStore
                retrieved_docs.append({
                    "id": hit['id'],
                    "content": hit.get('metadata', {}),
                    "similarity": hit.get('similarity', 0.0),  # Already converted to similarity in SimpleVectorStore
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

        formatted_content = []

        # Skip document IDs, retrieval methods, and scores for cleaner user experience
        # Only include the actual content data

        if "content" not in doc or not doc["content"]:
            return "No content available for this document."

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

    def generate_response(self, query: str, k: int = 5) -> Dict[str, Any]:
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

    async def generate_response_stream(self, query: str, k: int = 5):
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

        # Check if this is a simple greeting/chitchat before doing expensive retrieval
        query_lower = query.lower().strip()
        simple_greetings = ['hello', 'hi', 'hey', 'thanks', 'thank you', 'how are you', 'good morning', 'good afternoon', 'good evening']
        
        if any(greeting in query_lower for greeting in simple_greetings) and len(query.split()) <= 4:
            # Handle simple greetings without retrieval
            logger.info(f"Detected simple greeting, responding without document retrieval")
            
            if 'hello' in query_lower or 'hi' in query_lower or 'hey' in query_lower:
                response_text = "<p>Hi there! I'm your defect analysis assistant. How can I help you today?</p>"
            elif 'thanks' in query_lower or 'thank you' in query_lower:
                response_text = "<p>You're welcome! Let me know if you need help with anything else.</p>"
            elif 'how are you' in query_lower:
                response_text = "<p>I'm doing well and ready to help! What would you like to know about the system defects?</p>"
            else:
                response_text = "<p>Hello! I'm here to help you with any questions about defects or technical issues. What would you like to know?</p>"
            
            # Add assistant response to chat memory
            self.chat_memory.add_assistant_message(response_text)
            
            # Stream the simple response
            yield {
                "type": "formatted_token",
                "content": response_text,
                "done": False
            }
            
            # Send completion signal
            yield {
                "type": "done",
                "content": "",
                "done": True,
                "metadata": {
                    "query": query,
                    "response_type": "simple_greeting"
                }
            }
            return

        # For technical queries, proceed with full retrieval and analysis
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

            # Clean up the response to remove any unwanted prefixes/suffixes the LLM might add
            response_text = response_text.strip()
            
            # Remove common unwanted prefixes
            unwanted_prefixes = ['"html', '```html', 'html', '```', '"']
            for prefix in unwanted_prefixes:
                if response_text.startswith(prefix):
                    response_text = response_text[len(prefix):].strip()
            
            # Remove common unwanted suffixes
            unwanted_suffixes = ['```', '"']
            for suffix in unwanted_suffixes:
                if response_text.endswith(suffix):
                    response_text = response_text[:-len(suffix)].strip()
            
            # CRITICAL FIX: Normalize HTML to prevent standalone bold lines and punctuation fragments
            # Strategy: parse paragraph blocks, then merge single-strong or punctuation-only paragraphs into previous paragraph
            import re

            # Normalize tag syntax without stripping surrounding text spaces
            # Keep spaces outside tags intact to avoid words sticking together
            response_text = re.sub(r'<\s*(\w+)\s*>', r'<\1>', response_text)   # opening tags
            response_text = re.sub(r'</\s*(\w+)\s*>', r'</\1>', response_text)  # closing tags

            # Ensure a space before an opening emphasis tag if it follows a word character
            response_text = re.sub(r'(?<=\w)<(strong|em)>', r' <\1>', response_text)
            # Ensure a space after a closing emphasis tag if followed by a word character
            response_text = re.sub(r'</(strong|em)>(?=\w)', r'</\1> ', response_text)
            # Avoid spaces before punctuation like , . ; :
            response_text = re.sub(r'\s+([,.;:])', r'\1', response_text)

            # Wrap any strong/em segments that appear between paragraphs into their own paragraph
            # e.g., </p>\n<strong>text</strong>\n<p> => </p><p><strong>text</strong></p><p>
            response_text = re.sub(r'(</p>)\s*(<(?:strong|em)[^>]*>.*?</(?:strong|em)>)\s*(?=<p|$)', r'\1<p>\2</p>', response_text, flags=re.DOTALL)
            # Also handle when a strong/em block appears at the start before any <p>
            response_text = re.sub(r'^\s*(<(?:strong|em)[^>]*>.*?</(?:strong|em)>)\s*(?=<p|$)', r'<p>\1</p>', response_text, flags=re.DOTALL)

            # Extract paragraphs preserving order
            paras = re.findall(r'<p>(.*?)</p>', response_text, flags=re.DOTALL)

            def strip_tags(s: str) -> str:
                return re.sub(r'<[^>]+>', '', s)

            def is_strong_only(s: str) -> bool:
                return re.fullmatch(r'\s*<strong>.*?</strong>\s*', s, flags=re.DOTALL) is not None

            def is_small_fragment(plain: str) -> bool:
                t = plain.strip()
                if not t:
                    return True
                # punctuation or conjunction fragments
                small_tokens = {',', '.', ';', ':', 'and', 'but', 'or'}
                if t in small_tokens:
                    return True
                if t[0] in ',.;:' or t.lower() in small_tokens:
                    return True
                # one-to-three word short fragments commonly split
                return len(t.split()) <= 3

            normalized_paras: list[str] = []
            for i, p in enumerate(paras):
                plain = strip_tags(p)
                # Merge strong-only or tiny fragments into previous paragraph
                if (is_strong_only(p) or is_small_fragment(plain)) and normalized_paras:
                    prev_inner = re.findall(r'<p>(.*?)</p>', normalized_paras[-1], flags=re.DOTALL)
                    prev_inner = prev_inner[0] if prev_inner else ''

                    # Prepare join text (keep strong tags if present)
                    join_text = p.strip()
                    join_plain = strip_tags(join_text)

                    # Decide spacing: no space before punctuation like , . ; :
                    if join_plain and join_plain[0] in ',.;:':
                        new_prev = f"{prev_inner}{join_plain}"
                    else:
                        # default with a space
                        new_prev = f"{prev_inner} {join_text}"

                    normalized_paras[-1] = f"<p>{new_prev}</p>"
                else:
                    # Start a new proper paragraph
                    normalized_paras.append(f"<p>{p.strip()}</p>")

            # Remove empty paragraphs and tidy spaces inside paragraphs
            cleaned = []
            for para in normalized_paras:
                inner = re.findall(r'<p>(.*?)</p>', para, flags=re.DOTALL)
                inner_text = inner[0] if inner else ''
                inner_text = re.sub(r'\s+', ' ', inner_text).strip()
                if inner_text:
                    cleaned.append(f"<p>{inner_text}</p>")

            response_text = ''.join(cleaned)
            
            logger.info(f"Cleaned and normalized response_text preview: {response_text[:200]}...")

            # Add assistant response to chat memory
            self.chat_memory.add_assistant_message(response_text)

            # Stream the response token by token for better UX
            import asyncio
            
            # Stream the response in natural chunks that preserve formatting
            # Split by complete paragraph tags to maintain structure
            import re
            
            # Find complete paragraphs with opening and closing tags
            paragraph_pattern = r'(<p>.*?</p>)'
            paragraphs = re.findall(paragraph_pattern, response_text, re.DOTALL)
            
            # If no paragraphs found, send the entire response as one chunk
            if not paragraphs:
                yield {
                    "type": "formatted_token",
                    "content": response_text,
                    "done": False
                }
            else:
                # Send each complete paragraph
                for paragraph in paragraphs:
                    if paragraph.strip():
                        yield {
                            "type": "formatted_token",
                            "content": paragraph,
                            "done": False
                        }
                        # Small delay between paragraphs for smooth streaming
                        await asyncio.sleep(0.1)
            
            # Small final delay
            await asyncio.sleep(0.05)

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

def reset_vector_store(rag_system: EnhancedAdaptiveRAGSystem) -> bool:
    """
    Reset the vector store to clear all indexed documents.
    
    Args:
        rag_system: The RAG system instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Resetting vector store...")
        if rag_system.vector_store:
            rag_system.vector_store.reset()
            logger.info("Vector store has been reset successfully")
            
            # Reset other related variables
            rag_system.st_embeddings = None
            rag_system.loaded_files = []
            rag_system.total_documents = 0
            rag_system.chunk_to_original_doc_mapping = []
            
            return True
        else:
            logger.error("Vector store not initialized")
            return False
    except Exception as e:
        logger.error(f"Error resetting vector store: {e}", exc_info=True)
        return False
