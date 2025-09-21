"""
Enhanced Cleanup Script for Ragineer-Test

This script:
1. Removes Python cache directories (__pycache__)
2. Cleans log files (option to truncate or delete)
3. Removes temporary files (.tmp, .bak, .pyc)
4. Removes unused test files
5. Offers ChromaDB cleanup option
"""
import os
import shutil
import sys
import logging
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def cleanup_system(clean_logs=True, clean_chroma=False, delete_logs=False):
    """
    Clean up unnecessary files and Python cache
    
    Args:
        clean_logs (bool): Whether to clean log files (truncate to 0 bytes)
        clean_chroma (bool): Whether to clean ChromaDB files (warning: removes vector store)
        delete_logs (bool): Whether to delete log files completely (rather than truncate)
    """
    
    # Get the project root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    logger.info(f"Cleaning up project at: {root_dir}")
    
    # 1. Remove Python cache directories
    dirs_to_clean = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip virtual environment directory
        if 'myvenv' in dirpath:
            continue
            
        # Add __pycache__ directories to the cleanup list
        for dirname in dirnames:
            if dirname == '__pycache__':
                dirs_to_clean.append(os.path.join(dirpath, dirname))
    
    # Delete the cache directories
    for directory in dirs_to_clean:
        logger.info(f"Removing Python cache: {directory}")
        shutil.rmtree(directory, ignore_errors=True)
    
    # 2. Remove test files we don't need anymore
    test_files = [
        os.path.join(root_dir, 'scripts', 'test_chroma_store.py'),
        os.path.join(root_dir, 'scripts', 'test_chroma_direct.py'),
        os.path.join(root_dir, 'scripts', 'test_basic_store.py'),
        os.path.join(root_dir, 'scripts', 'compare_stores.py')
    ]
    
    for file in test_files:
        if os.path.exists(file):
            logger.info(f"Removing test file: {file}")
            os.remove(file)
    
    # 3. Find and clean all log files
    log_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip virtual environment directory
        if 'myvenv' in dirpath:
            continue
            
        # Add all .log files to the list
        for filename in filenames:
            if filename.endswith('.log'):
                log_files.append(os.path.join(dirpath, filename))
    
    # Process log files
    for file in log_files:
        if os.path.exists(file):
            if delete_logs:
                logger.info(f"Deleting log file: {file}")
                os.remove(file)
            else:
                logger.info(f"Clearing log file: {file}")
                with open(file, 'w') as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"# Log cleared by cleanup script on {timestamp}\n")
    
    # 4. Clean temporary files (*.tmp, *.bak, *.pyc)
    temp_extensions = ['.tmp', '.bak', '.pyc', '.pyo', '.pyd', '.~']
    temp_files = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip virtual environment directory
        if 'myvenv' in dirpath:
            continue
            
        # Find temporary files
        for filename in filenames:
            if any(filename.endswith(ext) for ext in temp_extensions):
                temp_files.append(os.path.join(dirpath, filename))
    
    # Remove temporary files
    for file in temp_files:
        logger.info(f"Removing temporary file: {file}")
        os.remove(file)
    
    # 5. Clean ChromaDB files if requested (warning: this removes embeddings)
    if clean_chroma:
        chroma_db_path = os.path.join(root_dir, 'db', 'chroma_db')
        if os.path.exists(chroma_db_path):
            logger.warning(f"Removing ChromaDB files: {chroma_db_path}")
            shutil.rmtree(chroma_db_path, ignore_errors=True)
            # Create empty directory to maintain structure
            os.makedirs(chroma_db_path, exist_ok=True)
    
    logger.info("Cleanup completed successfully")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Ragineer-Test Cleanup Utility')
    parser.add_argument('--logs', action='store_true', help='Clean log files (truncate)')
    parser.add_argument('--delete-logs', action='store_true', help='Delete log files completely')
    parser.add_argument('--chroma', action='store_true', 
                        help='WARNING: Clean ChromaDB files (removes vector store)')
    parser.add_argument('--all', action='store_true', 
                        help='Clean everything (except ChromaDB which requires explicit flag)')
    return parser.parse_args()

if __name__ == "__main__":
    print("🧹 Ragineer-Test Cleanup Utility 🧹")
    print("-----------------------------------")
    
    args = parse_arguments()
    
    # Default behavior without args: clean cache only
    clean_logs = args.logs or args.all or args.delete_logs
    clean_chroma = args.chroma  # ChromaDB cleanup requires explicit flag
    delete_logs = args.delete_logs
    
    # Execute cleanup
    cleanup_system(clean_logs=clean_logs, clean_chroma=clean_chroma, delete_logs=delete_logs)
    
    print("\n✅ System cleaned up successfully!")
    
    if clean_chroma:
        print("\n⚠️  WARNING: ChromaDB files were removed. You will need to rebuild your vector database!")
        
    print("\n▶️  Recommended next step: restart the application with 'python scripts/run.py'")
