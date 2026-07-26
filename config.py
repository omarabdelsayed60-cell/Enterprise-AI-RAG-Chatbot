"""
config.py
---------
Central configuration module for the RAG Assistant project.
Loads environment variables from .env and defines global paths & settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from logger import logger

# 1. Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# 2. Define Base Directories
BASE_DIR = Path(__file__).parent.resolve()
RESOURCE_DIR = BASE_DIR / "Resource"
FAISS_INDEX_DIR = BASE_DIR / "faiss_index"
EXCEL_PATH = RESOURCE_DIR / "company_policies.xlsx"

# Ensure required directories exist on startup
RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# 3. Environment Variables & Credentials
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# 4. Gemini Model & Embedding Configuration
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "models/gemini-embedding-001")

# 5. Configurable Retrieval & Chunking Settings
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
SIMILARITY_SCORE_THRESHOLD = float(os.getenv("SIMILARITY_SCORE_THRESHOLD", "0.0"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# 6. Retry & Backoff Configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", "2.0"))

# 7. SQL Server Database Configuration
DB_SERVER = os.getenv("DB_SERVER", "localhost")
DB_NAME = os.getenv("DB_NAME", "KB_RAG")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
DB_TRUSTED_CONNECTION = os.getenv("DB_TRUSTED_CONNECTION", "yes")
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

def get_sql_connection_string(include_db: bool = True) -> str:
    """
    Constructs a pyodbc connection string for Microsoft SQL Server.
    
    Args:
        include_db (bool): If True, connects directly to DB_NAME.
                           If False, connects to master DB (useful when creating the database).
    
    Returns:
        str: Formatted ODBC connection string
    """
    conn_str = f"DRIVER={{{DB_DRIVER}}};SERVER={DB_SERVER};"
    
    if include_db:
        conn_str += f"DATABASE={DB_NAME};"
        
    if DB_TRUSTED_CONNECTION.lower() in ["yes", "true", "1"]:
        conn_str += "Trusted_Connection=yes;"
    else:
        conn_str += f"UID={DB_USER};PWD={DB_PASSWORD};"
        
    return conn_str

logger.info(f"[Config Loaded] Base Directory: {BASE_DIR}")
logger.info(f"[Config Loaded] Model: {GEMINI_MODEL_NAME} | Embedding Model: {EMBEDDING_MODEL_NAME}")
logger.info(f"[Config Loaded] SQL Server: {DB_SERVER} | Database: {DB_NAME}")
