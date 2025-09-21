# Ragineer-Test Scripts

This directory contains utility scripts for managing the Ragineer-Test application.

## Available Scripts

### `run.py`

Starts both the backend and frontend servers concurrently.

```bash
python scripts/run.py
```

Features:
- Launches FastAPI backend on port 8000
- Starts frontend HTTP server on port 3000
- Opens browser with cache-busting parameter
- Provides clear logging and shutdown instructions

### `cleanup.py`

Cleans up unnecessary files and data to maintain system health.

```bash
# Basic usage (cleans Python cache)
python scripts/cleanup.py

# Clean log files (truncate to empty)
python scripts/cleanup.py --logs

# Delete log files completely
python scripts/cleanup.py --delete-logs

# Clean all caches and logs
python scripts/cleanup.py --all

# WARNING: Remove ChromaDB files (requires explicit flag)
python scripts/cleanup.py --chroma
```

Features:
- Removes Python cache directories (`__pycache__`)
- Cleans or deletes log files
- Removes temporary files (`.tmp`, `.bak`, `.pyc`, etc.)
- Can reset ChromaDB vector store (with `--chroma` flag)

**NOTE:** After running the cleanup script, always restart the application with `run.py`.