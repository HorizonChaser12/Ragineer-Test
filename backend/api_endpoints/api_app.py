# FastAPI Imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.staticfiles import StaticFiles

# for uvicorn web server
import uvicorn

# For logging
import logging
import datetime
import json
from typing import Optional

# Pydantic Model Imports
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.dirname(parent_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Local imports
from schema.pydantic_models import InitializeRequest, RebuildIndexRequest, QueryRequest
from rag_system import EnhancedAdaptiveRAGSystem, make_serializable


# Configure logging to file only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=os.path.join(root_dir, "rag_system_api.log"),  # Use absolute path for log file
    filemode="a",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced RAG System API",
    description="Advanced RAG system with multiple retrieval methods and re-ranking",
    version="2.0.0",
)

# Track initialization status
initialization_complete = False

# This is for passing all the hosts where requests are being sent
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os

# Mount static files (for frontend) - only if directory exists
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# Global variable to hold the RAG system instance
rag_system: Optional[EnhancedAdaptiveRAGSystem] = None

# Auto-initialize RAG system on startup
def initialize_on_startup():
    global rag_system, initialization_complete
    try:
        logger.info("Auto-initializing RAG system on startup...")
        
        # Ensure data directory exists
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
        logger.info(f"Looking for data directory at: {data_dir}")
        if not os.path.exists(data_dir):
            logger.error(f"Data directory not found at: {data_dir}")
            raise FileNotFoundError(f"Data directory not found at: {data_dir}")
        
        # Check for Excel files
        excel_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
        if not excel_files:
            logger.error(f"No Excel files found in data directory: {data_dir}")
            raise FileNotFoundError(f"No Excel files found in data directory: {data_dir}")
            
        logger.info(f"Found Excel files: {excel_files}")
        
        # Debug: Check for db module
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_dir = os.path.join(root_dir, "db")
        logger.info(f"Root directory: {root_dir}")
        logger.info(f"DB directory: {db_dir}, exists: {os.path.exists(db_dir)}")
        if os.path.exists(db_dir):
            db_files = os.listdir(db_dir)
            logger.info(f"Files in db directory: {db_files}")
        
        # Initialize RAG system with absolute path
        rag_system = EnhancedAdaptiveRAGSystem(
            data_source=data_dir,
            auto_initialize=True,
            use_sentence_transformers=True,
            use_reranker=True,
            temperature=0.7
        )
        
        # Verify initialization
        status = rag_system.get_system_status()
        logger.info(f"System initialization status: {status}")
        
        if not status['vector_store_ready']:
            raise RuntimeError("Vector store failed to initialize")
            
        if status['files_loaded_count'] == 0:
            raise RuntimeError("No files were loaded during initialization")
            
        logger.info("RAG system initialized successfully on startup")
        initialization_complete = True
        
    except Exception as e:
        logger.error(f"Failed to auto-initialize RAG system: {e}", exc_info=True)
        raise

# Initialize on module load with error handling
try:
    initialize_on_startup()
    logger.info("Module initialization complete")
    # Ensure initialization_complete is set if we reach here
    if rag_system is not None:
        initialization_complete = True
except Exception as e:
    logger.error(f"Module initialization failed: {e}", exc_info=True)
    initialization_complete = False
    # Don't suppress the error - let it propagate
    raise

# Rate limiting for status calls
import time
last_status_calls = {}  # Track last status call per client IP


# Exception handler for better error responses
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):

    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500, content={"detail": "Internal server error", "error": str(exc)}
    )


@app.post("/initialize")
async def initialize_system(request: InitializeRequest):
    """Initialize the RAG system with specified configuration"""
    global rag_system

    try:
        # Validate data source path
        data_source = request.excel_file_path
        if not os.path.exists(data_source):
            raise FileNotFoundError(f"Data source not found: {data_source}")
            
        logger.info(f"Initializing RAG system with data source: {data_source}")
        
        # Initialize RAG system
        rag_system = EnhancedAdaptiveRAGSystem(
            data_source=data_source,
            temperature=request.temperature,
            concise_prompt=request.concise_prompt,
            use_sentence_transformers=request.use_sentence_transformers,
            use_reranker=request.use_reranker,
            auto_initialize=True
        )

        status = rag_system.get_system_status()

        logger.info("RAG system initialized successfully")

        return {"message": "RAG system initialized successfully", "status": status}

    except Exception as e:

        logger.error(f"Failed to initialize RAG system: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@app.post("/quick-start")
async def quick_start_system(data_directory: str = "data"):
    """Quick start the RAG system with automatic data loading from a directory"""
    
    global rag_system
    
    try:
        logger.info(f"🚀 Quick starting RAG system with data from: {data_directory}")
        
        # Use the convenience function
        from backend.rag_system import quick_start
        rag_system = quick_start(data_directory)
        
        status = rag_system.get_system_status()
        
        logger.info("✅ RAG system quick started successfully")
        
        # Safely handle loaded_files and total_documents
        loaded_files = status.get('loaded_files', []) or []
        total_documents = status.get('total_documents', 0) or 0
        
        return {
            "message": "RAG system quick started successfully", 
            "status": status,
            "loaded_files": loaded_files,
            "total_documents": total_documents
        }
        
    except Exception as e:
        logger.error(f"Failed to quick start RAG system: {e}", exc_info=True)
        # More detailed error information
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": str(e.__traceback__) if hasattr(e, '__traceback__') else "No traceback"
        }
        logger.error(f"Quick start error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"Quick start failed: {str(e)}")


@app.post("/auto-initialize")
async def auto_initialize_system(data_source: str = "data"):
    """Auto-initialize the RAG system with data from file or directory"""
    
    global rag_system
    
    try:
        logger.info(f"Auto-initializing RAG system with: {data_source}")
        
        rag_system = EnhancedAdaptiveRAGSystem(
            data_source=data_source,
            auto_initialize=True,
            use_sentence_transformers=True,
            use_reranker=True,
            temperature=0.7
        )
        
        status = rag_system.get_system_status()
        
        logger.info("RAG system auto-initialized successfully")
        
        # Safely handle loaded_files and total_documents
        loaded_files = status.get('loaded_files', []) or []
        total_documents = status.get('total_documents', 0) or 0
        
        return {
            "message": "RAG system auto-initialized successfully", 
            "status": status,
            "data_source": data_source,
            "loaded_files": loaded_files,
            "total_documents": total_documents
        }
        
    except Exception as e:
        logger.error(f"Failed to auto-initialize RAG system: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Auto-initialization failed: {str(e)}")


@app.get("/status")
async def get_system_status(request: Request):
    """Get current system status and information"""
    
    global rag_system
    
    # Simple logging without rate limiting
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Status check from {client_ip}")
    
    if rag_system is None:
        return JSONResponse(content={
            "initialized": False,
            "system_ready": False,
            "status_message": "RAG system not initialized",
            "vector_store_ready": False,
            "files_loaded_count": 0,
            "total_documents": 0,
            "models_ready": {
                "embedding_model": False,
                "llm": False,
                "sentence_transformer": False,
                "reranker": False
            }
        })
    
    try:
        status = rag_system.get_system_status()
        status["initialized"] = True
        
        # Ensure all required fields are present
        default_status = {
            "initialized": True,
            "system_ready": False,
            "status_message": "",
            "vector_store_ready": False,
            "files_loaded_count": 0,
            "total_documents": 0,
            "models_ready": {
                "embedding_model": False,
                "llm": False,
                "sentence_transformer": False,
                "reranker": False
            }
        }
        
        # Update default status with actual values
        default_status.update(status)
        
        return JSONResponse(content=make_serializable(default_status))
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return JSONResponse(
            content={
                "initialized": False,
                "system_ready": False,
                "status_message": f"Error getting system status: {str(e)}",
                "error": str(e)
            },
            status_code=500
        )


@app.post("/reload")
async def reload_data():
    """Reload data from the current data source"""
    
    global rag_system
    
    if rag_system is None:
        raise HTTPException(
            status_code=400,
            detail="RAG system not initialized. Please initialize first."
        )
    
    try:
        rag_system.reload_data()
        status = rag_system.get_system_status()
        
        return {
            "message": "Data reloaded successfully",
            "status": status,
            "loaded_files": status.get('loaded_files', []),
            "total_documents": status.get('total_documents', 0)
        }
        
    except Exception as e:
        logger.error(f"Failed to reload data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Data reload failed: {str(e)}")


@app.post("/query")
async def query_system(request: QueryRequest):
    """Query the RAG system"""

    global rag_system

    if rag_system is None:
        raise HTTPException(
            status_code=400,
            detail="RAG system not initialized. Please call /initialize first.",
        )

    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        logger.info(f"Request k value: {request.k}")  # Debug log
        
        # Ensure k is set to a reasonable value if None
        k_value = request.k if request.k is not None else 10
        logger.info(f"Using k value: {k_value}")  # Debug log

        # Update temperature if provided
        if request.temperature is not None and rag_system.llm:
            rag_system.llm.temperature = request.temperature

        # Use RAG mode
        response = rag_system.generate_response(request.query, k_value)
        logger.info("Query processed successfully")

        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Query processing failed: {str(e)}"
        )


@app.post("/query/stream")
async def query_system_stream(request: QueryRequest):
    """Query the RAG system with streaming response"""
    
    global rag_system

    if rag_system is None:
        raise HTTPException(
            status_code=400,
            detail="RAG system not initialized. Please call /initialize first.",
        )

    async def generate_stream():
        try:
            logger.info(f"Processing streaming query: {request.query[:100]}...")
            logger.info(f"Streaming request k value: {request.k}")  # Debug log
            
            # Ensure k is set to a reasonable value if None
            k_value = request.k if request.k is not None else 10
            logger.info(f"Using streaming k value: {k_value}")  # Debug log
            
            # Update temperature if provided
            if request.temperature is not None and rag_system.llm:
                rag_system.llm.temperature = request.temperature

            # Stream RAG response
            async for chunk in rag_system.generate_response_stream(request.query, k_value):
                yield f"data: {json.dumps(chunk)}\n\n"
                    
        except Exception as e:
            logger.error(f"Error in streaming query: {e}", exc_info=True)
            error_data = {
                "type": "error",
                "content": f"Error processing query: {str(e)}",
                "done": True
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )


@app.post("/rebuild-index")
async def rebuild_index(request: RebuildIndexRequest):
    """Rebuild the FAISS index and sentence transformer embeddings"""

    global rag_system

    if rag_system is None:

        raise HTTPException(
            status_code=400,
            detail="RAG system not initialized. Please call /initialize first.",
        )

    try:

        logger.info("Rebuilding indices...")

        rag_system._build_index()

        status = rag_system.get_system_status()

        logger.info("Indices rebuilt successfully")

        return {"message": "Indices rebuilt successfully", "status": status}

    except Exception as e:

        logger.error(f"Failed to rebuild indices: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {str(e)}")


@app.post("/retrieve")
async def retrieve_documents(request: QueryRequest):
    """Retrieve relevant documents without generating a response"""

    global rag_system

    if rag_system is None:
        raise HTTPException(
            status_code=400,
            detail="RAG system not initialized. Please call /initialize first.",
        )

    try:
        logger.info(f"Retrieving documents for query: {request.query[:100]}...")

        retrieved_docs = rag_system.retrieve(request.query, request.k)
        pattern_analysis = rag_system.analyze_patterns(retrieved_docs)

        return {
            "query": request.query,
            "retrieved_docs": make_serializable(retrieved_docs),
            "pattern_analysis": make_serializable(pattern_analysis),
            "count": len(retrieved_docs),
        }

    except Exception as e:
        logger.error(f"Error retrieving documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Document retrieval failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    
    global rag_system, initialization_complete
    
    # Check if system is properly initialized
    is_ready = False
    system_status = None
    
    if rag_system is not None:
        try:
            # Try to get system status to verify it's working
            system_status = rag_system.get_system_status()
            is_ready = system_status.get('system_ready', False)
            
            # If system is ready but flag isn't set, update the flag
            if is_ready and not initialization_complete:
                initialization_complete = True
                logger.info("Updated initialization_complete flag based on system status")
                
            logger.info("Health check: System is ready" if is_ready else "Health check: System not ready")
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            is_ready = False
    
    response = {
        "status": "healthy" if is_ready else "initializing",
        "timestamp": datetime.datetime.now().isoformat(),
        "system_initialized": rag_system is not None,
        "ready": is_ready,
        "initialization_complete": initialization_complete
    }
    
    # Add system details if available
    if system_status:
        response["system_details"] = {
            "total_documents": system_status.get('total_documents', 0),
            "files_loaded_count": system_status.get('files_loaded_count', 0),
            "vector_store_ready": system_status.get('vector_store_ready', False),
            "models_ready": system_status.get('models_ready', {})
        }
    
    logger.info(f"Health check response: {response}")
    return response


@app.get("/chat/history")
async def get_chat_history(limit: int = 20):
    """Get chat conversation history"""
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=400, detail="RAG system not initialized")
    
    try:
        history = rag_system.chat_memory.get_conversation_history(limit=limit)
        return {
            "history": history,
            "session_stats": rag_system.chat_memory.get_session_stats()
        }
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/clear")
async def clear_chat_history():
    """Clear current chat session"""
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=400, detail="RAG system not initialized")
    
    try:
        rag_system.chat_memory.clear_session()
        return {"message": "Chat session cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/sessions")
async def list_chat_sessions():
    """List all available chat sessions"""
    global rag_system
    
    if rag_system is None:
        raise HTTPException(status_code=400, detail="RAG system not initialized")
    
    try:
        sessions = rag_system.chat_memory.list_sessions()
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error listing chat sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Serve the chat interface"""
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    else:
        return HTMLResponse("""
        <html>
            <body>
                <h1>Advanced Fix Finder API</h1>
                <p>Frontend not found. Please ensure index.html is in the frontend directory.</p>
                <p>API documentation: <a href="/docs">/docs</a></p>
                <p>API Status: <a href="/api-info">/api-info</a></p>
            </body>
        </html>
        """)

@app.get("/api-info")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Enhanced RAG System API",
        "version": "2.0.0",
        "features": [
            "Google Gemini Integration",
            "Sentence Transformers",
            "Cross-Encoder Re-ranking",
            "ChromaDB Vector Search",
            "Auto-loading Data Processing",
        ],
        "endpoints": {
            "POST /quick-start": "Quick start with auto-loading",
            "POST /auto-initialize": "Initialize with custom data source",
            "POST /query": "Query the system for responses",
            "POST /reload": "Reload data from source",
            "GET /status": "Get system status",
            "GET /health": "Health check",
        },
    }


@app.post("/reset-vector-store")
async def reset_vector_store_endpoint():
    """Reset the vector store to clear all indexed documents"""
    
    global rag_system
    
    if not rag_system:
        raise HTTPException(
            status_code=400, 
            detail="RAG system is not initialized. Please initialize the system first."
        )
    
    try:
        logger.info("Resetting vector store...")
        
        # Import the function
        from backend.rag_system import reset_vector_store
        
        # Reset the vector store
        success = reset_vector_store(rag_system)
        
        if success:
            status = rag_system.get_system_status()
            logger.info("Vector store reset successfully")
            
            return {
                "message": "Vector store reset successfully",
                "status": status
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail="Failed to reset vector store. Check server logs for details."
            )
            
    except Exception as e:
        logger.error(f"Error resetting vector store: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to reset vector store: {str(e)}"
        )

# Global variables for rate limiting
last_status_calls = {}  # Track last status call time per IP


if __name__ == "__main__":

    logger.info("Starting Enhanced RAG System API server...")

    uvicorn.run(
        "api_app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        reload=True,
        reload_excludes=["*.log", "rag_system_api.log"]  # Exclude log files from reload watching
    )
