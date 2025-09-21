# Ragineer-Test / Fix Finder Coding Guide

## Architecture Overview

Fix Finder is a RAG (Retrieval-Augmented Generation) system for testing teams that:
- Processes defect data from Excel/CSV files
- Uses vector embeddings (via ChromaDB) for semantic search
- Provides a chat interface for querying defect information
- Formats responses with color-coded severity indicators

### Key Components

1. **Backend (`backend/`)**: 
   - `rag_system.py`: Core RAG engine with document processing and retrieval logic
   - `api_endpoints/api_app.py`: FastAPI server with streaming endpoints
   - `schema/pydantic_models.py`: Data models for API requests/responses

2. **Frontend (`frontend/`)**: 
   - Single-page application with custom formatting for defect information
   - Special handling for document ID removal and severity color-coding

3. **Data (`data/`)**: 
   - Excel files with defect information (e.g., `Defects.xlsx`)

4. **Vector Store (`db/`)**: 
   - ChromaDB persistent storage for embeddings

## Development Workflow

### Running the Application

Always use the run script to start both servers simultaneously:
```bash
python scripts/run.py
```

This script:
- Launches FastAPI backend on port 8000
- Starts frontend HTTP server on port 3000
- Opens browser with cache-busting parameter

### Environment Setup

1. Configure with `.env` file (copy from `.env.example`)
2. Use virtual environment (`myvenv/`) for dependencies
3. Install requirements with `pip install -r requirements.txt`

## Code Conventions

### Formatting

- Defect entries use the format: `**LIFE-XXXX:** "Description" (Severity, Priority)`
- Frontend applies custom styling to these defects with:
  - Color-coded severity levels
  - Structured display with defect-item containers
  - Document ID removal for cleaner UI

### Query Parameters

- Default search depth: `k=10` (in `schema/pydantic_models.py`)
- For defect searches, results are formatted with severity highlighting

## Key Integrations

1. **ChromaDB**: Vector database for efficient semantic search
   - Configuration in `backend/rag_system.py`
   - Persistent storage in `db/chroma_db/`

2. **Frontend-Backend Communication**:
   - Backend streams responses with FastAPI's `StreamingResponse`
   - Frontend formats responses with custom regex patterns

## Debugging Tips

- Clear Python cache with `scripts/cleanup.py` when experiencing stale behavior
- Force browser cache refresh with Ctrl+F5 when testing frontend changes
- Backend logs are stored in `backend/rag_system_api.log`
- For formatting issues in responses, check the `formatAssistantMessage` function in `frontend/index.html`