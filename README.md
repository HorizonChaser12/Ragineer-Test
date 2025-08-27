# Fix Finder - Intelligent Testing Assistant

**Advanced RAG System for IT Testing Teams and Quality Assurance**

## Latest Updates (v2.0)

- ✅ **Fixed Page Refresh Issues**: Eliminated all reload dialogs and navigation interruptions
- ✅ **Streaming Responses**: Real-time token generation with live status updates
- ✅ **Enhanced Formatting**: Rich HTML with proper headers, bullet points, and structure
- ✅ **Smart Loading**: Dynamic typing animations that transform into status updates
- ✅ **Production Stability**: Disabled file watching, comprehensive error handling
- ✅ **Universal Support**: Works with web, mobile, API, database, and enterprise testing


## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Testing Teams & QA Engineers                 │
│        (Web Testing, Mobile Testing, API Testing, etc.)         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                Fix Finder Web Interface                         │
│           (Intelligent Testing Assistant)                       │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Real-time Queries & Responses
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                              │
│        (RESTful APIs, Streaming, CORS Enabled)                  │
└────────────┬─────────────────────────────────┬──────────────────┘
             │                                 │
             ▼                                 ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│    RAG Intelligence     │          │   Knowledge Management  │
│                         │          │                         │
│ • Semantic Search       │          │ • Multi-Format Import   │
│ • Pattern Recognition   │          │ • Data Preprocessing    │
│ • Context Generation    │          │ • Vector Embeddings     │
│ • Response Synthesis    │          │ • Index Management      │
└────────────┬────────────┘          └────────────┬────────────┘
             │                                    │
             ▼                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Unified Knowledge Base                       │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │   Web App   │  │   Mobile    │  │     API     │  │  Other  │ │
│  │   Defects   │  │   Defects   │  │   Testing   │  │ Testing │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
│                                                                 │
│  Data Sources: JIRA, Azure DevOps, TestRail, Excel, CSV, JSON   │
└─────────────────────────────────────────────────────────────────┘

Information Flow:
1. Tester asks question → Web Interface
2. Query processed → RAG Intelligence Engine  
3. Relevant defects retrieved → Knowledge Base Search
4. Context-aware response → LLM Generation
5. Formatted answer → Real-time Streaming to User
6. Insights & patterns → Testing Team Knowledge
```

## Quick Start

1. **Clone Repository**
```bash
git clone <repo-url>
cd fix-finder
```

2. **Setup Virtual Environment**
```bash
python -m venv myvenv
myvenv\Scripts\activate  # Windows
# source myvenv/bin/activate  # Linux/Mac
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Add Your Data**
Put testing data files in `data/` folder:
- Excel (.xlsx): JIRA exports, defect reports
- CSV (.csv): TestRail results, bug tracking exports
- JSON (.json): API responses, structured data

5. **Start Servers**
```bash
# Terminal 1: Start API server
python -m uvicorn api_endpoints.api_app:app --host 0.0.0.0 --port 8000

# Terminal 2: Start frontend
cd frontend
python -m http.server 3000
```

6. **Access Application**
Open: http://localhost:3000/chat_interface.html

## Environment Setup (.env)

Create `.env` file in root directory (optional):

```env
# Optional: OpenAI API (if not using default model)
OPENAI_API_KEY=your-openai-api-key

# Optional: Custom configurations  
MODEL_NAME=gpt-3.5-turbo
MAX_DOCUMENTS=10
TEMPERATURE=0.7
```

## Changelog

### v2.0.0 (Current)
- Universal testing support across all domains
- Complete UX overhaul with streaming responses
- Production stability and error recovery
- Multi-format data processing
- Enhanced pattern recognition

### v1.0.0 (Initial)
- Basic RAG functionality
- Simple defect search
- ChromaDB integration
- [Configuration](#configuration)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)


## API Endpoints

### Core Endpoints

| Endpoint | Method | Description | Example |
|----------|--------|-------------|---------|
| `/status` | GET | Get system status and document count | `GET /status` |
| `/health` | GET | Simple health check | `GET /health` |
| `/quick-start` | POST | Initialize system with default data directory | `POST /quick-start?data_directory=data` |
| `/query/stream` | POST | Stream real-time responses | `POST /query/stream` |
| `/query` | POST | Get complete response (non-streaming) | `POST /query` |

### Request Examples

#### Streaming Query
```bash
curl -X POST "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are common API testing defects in our system?"}'
```

#### System Status  
```bash
curl -X GET "http://localhost:8000/status"
```

#### Quick Start with Custom Data
```bash
curl -X POST "http://localhost:8000/quick-start?data_directory=testing_data"
```

### Response Formats

#### Status Response
```json
{
  "initialized": true,
  "total_documents": 247,
  "document_sources": [
    {"source": "web_defects.xlsx", "count": 89},
    {"source": "mobile_bugs.csv", "count": 67},
    {"source": "api_issues.json", "count": 45},
    {"source": "db_testing.xlsx", "count": 46}
  ],
  "system_ready": true,
  "vector_store_ready": true,
  "llm_ready": true
}
```

#### Streaming Response (SSE)
```
data: {"type": "status", "content": "Retrieving relevant testing documents...", "done": false}
data: {"type": "status", "content": "Found 8 relevant defects across web and API testing. Analyzing patterns...", "done": false}
data: {"type": "formatted_token", "content": "<p><strong>", "done": false}
data: {"type": "formatted_token", "content": "Common Testing Defects Analysis", "done": false}
data: {"type": "formatted_token", "content": "</strong></p>", "done": false}
data: {"type": "done", "content": "", "done": true}
```

### Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `500`: Internal Server Error
- `503`: Service Unavailable (system not initialized)

## Configuration

### Environment Setup

The system uses environment variables for configuration. Create a `.env` file in the root directory:

```env
# Optional: If using OpenAI API instead of default model
OPENAI_API_KEY=your-openai-api-key

# Optional: Custom model configurations
MODEL_NAME=gpt-3.5-turbo
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Optional: System configurations
MAX_DOCUMENTS=10
TEMPERATURE=0.7
```

### Data Configuration

The system supports various testing data sources and formats:

**Testing Data Types:**
- **Web Application Defects**: Browser compatibility, UI/UX issues, functionality bugs
- **Mobile Application Issues**: Device-specific problems, performance issues, crashes  
- **API Testing Results**: Endpoint failures, response validation, performance bottlenecks
- **Database Testing Findings**: Data integrity issues, query performance, schema problems
- **Performance Test Results**: Load testing failures, memory leaks, timeout issues
- **Security Testing Reports**: Vulnerabilities, authentication issues, data exposure
- **Integration Testing Issues**: System interface problems, data flow issues

**Supported File Formats:**
```python
# Excel files with structured defect data
SUPPORTED_EXCEL = ['.xlsx', '.xls']

# CSV files from various testing tools  
SUPPORTED_CSV = ['.csv', '.tsv']

# JSON exports from APIs and modern tools
SUPPORTED_JSON = ['.json', '.jsonl']

# XML from legacy systems
SUPPORTED_XML = ['.xml']
```

### System Settings

Default configurations (can be modified in `rag_system.py`):
```python
# Document retrieval settings
DEFAULT_K = 5  # Number of documents to retrieve
MAX_CHUNK_SIZE = 1000  # Maximum chunk size for processing

# Streaming settings
STREAM_DELAY = 0.05  # Delay between streaming tokens (seconds)
STATUS_UPDATE_INTERVAL = 1.0  # Status update frequency

# Vector database settings
COLLECTION_NAME = "testing_defects"  # Universal collection name
SIMILARITY_THRESHOLD = 0.7
CROSS_DOMAIN_SEARCH = True  # Enable search across different testing domains
```


## Tech Stack

### Backend
- **Python 3.11+**: Core programming language
- **FastAPI**: High-performance web framework with automatic API documentation
- **Uvicorn**: ASGI server for production-grade performance
- **ChromaDB**: Vector database for document embeddings and similarity search
- **Sentence Transformers**: State-of-the-art text embeddings
- **LangChain**: LLM framework for retrieval-augmented generation
- **Pandas**: Data manipulation and analysis
- **Pydantic**: Data validation and serialization

### Frontend
- **HTML5/CSS3**: Modern web standards
- **JavaScript (ES6+)**: Interactive functionality and streaming
- **Material Design Icons**: Consistent iconography
- **Server-Sent Events (SSE)**: Real-time streaming communication
- **CSS Grid/Flexbox**: Responsive layout system

### Database & Storage
- **ChromaDB**: Vector embeddings storage and retrieval
- **File System**: Document storage and caching
- **JSON**: Configuration and metadata storage

### Development & Deployment
- **Git**: Version control
- **Python Virtual Environments**: Dependency isolation
- **HTTP Server**: Frontend serving (development)
- **CORS**: Cross-origin resource sharing
- **Logging**: Comprehensive application logging


## Contributing

We welcome contributions to Fix Finder! Here's how you can help:

