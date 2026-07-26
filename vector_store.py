"""
vector_store.py
---------------
FAISS Vector Store Manager & SQL Embedding Sync Engine.

Key Architectural Highlights:
1. SQL Server is the single source of truth for all policy content and metadata. FAISS is a derived search index generated from SQL.
2. Persists the derived FAISS vector index to disk (faiss_index/) with metadata validation (metadata.json).
3. Validates embedding model consistency, chunk size, and chunk overlap on startup; rebuilds automatically if settings change.
4. Supports item-level error isolation and exponential backoff retry for Gemini embedding API calls.
5. Thread-safe execution: no shared global model/FAISS objects across requests or threads.
"""

import json
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_RETRIES,
    BACKOFF_FACTOR,
    SIMILARITY_SCORE_THRESHOLD
)
from logger import logger
import database

METADATA_FILE_PATH = FAISS_INDEX_DIR / "metadata.json"
INDEX_FAISS_PATH = FAISS_INDEX_DIR / "index.faiss"
INDEX_PKL_PATH = FAISS_INDEX_DIR / "index.pkl"

def get_embeddings_model(api_key: Optional[str] = None) -> GoogleGenerativeAIEmbeddings:
    """
    Initializes and returns a fresh Google Gemini Embeddings model instance per request.
    """
    key = api_key or GEMINI_API_KEY
    if not key:
        logger.error("GEMINI_API_KEY is missing in environment settings!")
        raise ValueError("Google Gemini API Key is missing. Please set GEMINI_API_KEY in your .env file.")
        
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        google_api_key=key
    )

def save_index_metadata(doc_count: int) -> None:
    """
    Saves index configuration metadata to metadata.json in FAISS_INDEX_DIR.
    """
    metadata = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "document_count": doc_count,
        "created_at": datetime.now().isoformat()
    }
    try:
        with open(METADATA_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved FAISS index metadata to {METADATA_FILE_PATH}")
    except Exception as err:
        logger.warning(f"Could not save FAISS index metadata: {err}")

def validate_index_metadata() -> bool:
    """
    Verifies that the disk index exists and matches current embedding model, chunk size, and chunk overlap settings.
    """
    if not (INDEX_FAISS_PATH.exists() and INDEX_PKL_PATH.exists() and METADATA_FILE_PATH.exists()):
        logger.info("FAISS disk index or metadata files missing.")
        return False

    try:
        with open(METADATA_FILE_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model_match = meta.get("embedding_model") == EMBEDDING_MODEL_NAME
        size_match = meta.get("chunk_size") == CHUNK_SIZE
        overlap_match = meta.get("chunk_overlap") == CHUNK_OVERLAP

        if model_match and size_match and overlap_match:
            logger.info(f"Disk FAISS metadata validated cleanly (Model: {EMBEDDING_MODEL_NAME}, Chunks: {meta.get('document_count')}).")
            return True
        else:
            logger.warning(f"FAISS index configuration mismatch! Disk: {meta} vs Config: model={EMBEDDING_MODEL_NAME}, size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
            return False

    except Exception as err:
        logger.warning(f"Failed to read or validate metadata.json: {err}")
        return False

def embed_query_with_retry(embeddings_model: GoogleGenerativeAIEmbeddings, text: str) -> List[float]:
    """
    Computes text vector embeddings using exponential backoff retry logic.
    """
    last_exception = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return embeddings_model.embed_query(text)
        except Exception as err:
            last_exception = err
            logger.warning(f"Embedding API attempt {attempt}/{MAX_RETRIES} failed: {err}")
            if attempt < MAX_RETRIES:
                sleep_time = BACKOFF_FACTOR ** attempt
                logger.info(f"Retrying embedding API in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

    logger.error(f"All {MAX_RETRIES} embedding API attempts failed for text snippet.")
    raise last_exception

def load_faiss_vector_store(api_key: Optional[str] = None, force_rebuild: bool = False) -> Optional[FAISS]:
    """
    Loads FAISS vector store from disk if valid. Rebuilds from SQL Server if missing, invalid, or force_rebuild=True.
    """
    start_time = time.perf_counter()
    embeddings_model = get_embeddings_model(api_key)

    # 1. Try loading from disk if valid and not force_rebuild
    if not force_rebuild and validate_index_metadata():
        try:
            logger.info(f"Loading FAISS vector store index from disk ({FAISS_INDEX_DIR})...")
            vector_store = FAISS.load_local(
                folder_path=str(FAISS_INDEX_DIR),
                embeddings=embeddings_model,
                allow_dangerous_deserialization=True
            )
            elapsed = time.perf_counter() - start_time
            logger.info(f"[PERF] FAISS vector store loaded successfully from disk in {elapsed:.3f}s")
            return vector_store
        except Exception as err:
            logger.warning(f"Failed to load FAISS index from disk despite validation: {err}. Rebuilding from SQL...")

    # 2. Rebuild index from SQL Server
    logger.info("Rebuilding FAISS index from SQL Server vector records...")
    embeddings_records = database.get_all_stored_embeddings()

    if not embeddings_records:
        logger.info("No stored vector embeddings found in SQL Server database.")
        return None

    text_embeddings_pairs = []
    metadatas = []

    for rec in embeddings_records:
        text_embeddings_pairs.append((rec["chunk_text"], rec["vector"]))
        metadatas.append({
            "topic_id": rec["topic_id"],
            "title": rec["title"],
            "category": rec["category"],
            "last_updated": rec["last_updated"],
            "chunk_index": rec["chunk_index"]
        })

    logger.info(f"Constructing in-memory FAISS index from {len(text_embeddings_pairs)} SQL Server vector records...")
    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embeddings_pairs,
        embedding=embeddings_model,
        metadatas=metadatas
    )

    # Persist to disk and update metadata.json
    try:
        vector_store.save_local(folder_path=str(FAISS_INDEX_DIR))
        save_index_metadata(len(text_embeddings_pairs))
        logger.info(f"Persisted rebuilt FAISS index to disk at {FAISS_INDEX_DIR}")
    except Exception as err:
        logger.error(f"Failed to save rebuilt FAISS index to disk: {err}")

    elapsed = time.perf_counter() - start_time
    logger.info(f"[PERF] FAISS vector store rebuilt from SQL Server and persisted in {elapsed:.3f}s")
    return vector_store

def sync_vector_store(api_key: Optional[str] = None) -> Tuple[int, List[int]]:
    """
    Synchronizes Knowledge Base:
    1. Cleans up orphan vector embeddings for deleted SQL policies.
    2. Fetches records where Processed = 0 from SQL Server.
    3. Chunks policy content using RecursiveCharacterTextSplitter.
    4. Generates Gemini vector embeddings with retry logic & item-level error handling.
    5. Saves chunks & vectors into SQL Server and sets Processed = 1.
    6. Rebuilds and persists the updated FAISS index to disk.
    """
    start_time = time.perf_counter()
    logger.info("Starting Knowledge Base synchronization...")

    # Step 1: Clean up orphan embeddings for deleted policies
    database.cleanup_deleted_topic_embeddings()

    # Step 2: Fetch unprocessed policy topics
    unprocessed_policies = database.get_unprocessed_policies()

    if not unprocessed_policies:
        logger.info("All policy topics are already synced in SQL Server.")
        # Ensure disk index is populated and valid
        load_faiss_vector_store(api_key=api_key)
        elapsed = time.perf_counter() - start_time
        logger.info(f"[PERF] Knowledge Base sync check completed in {elapsed:.3f}s (Already up to date)")
        return 0, []

    logger.info(f"Processing vector embedding generation for {len(unprocessed_policies)} pending policy topics...")
    embeddings_model = get_embeddings_model(api_key)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    records_to_save_in_sql = []
    synced_topic_ids = []

    for policy in unprocessed_policies:
        topic_id = policy["TopicID"]
        category = policy["Category"]
        title = policy["Title"]
        content = policy["Content"]
        last_updated = policy["LastUpdated"]

        logger.info(f"[SYNC PROCESSING] TopicID {topic_id}: '{title}'...")

        try:
            full_text = f"Policy Topic: {title}\nCategory: {category}\nLast Updated: {last_updated}\n\nContent:\n{content}"
            chunks = text_splitter.split_text(full_text)

            policy_chunks_data = []
            for idx, chunk_text in enumerate(chunks):
                vector_floats = embed_query_with_retry(embeddings_model, chunk_text)
                policy_chunks_data.append({
                    "topic_id": topic_id,
                    "chunk_index": idx,
                    "chunk_text": chunk_text,
                    "vector": vector_floats,
                    "title": title,
                    "category": category,
                    "last_updated": last_updated
                })

            records_to_save_in_sql.extend(policy_chunks_data)
            synced_topic_ids.append(topic_id)
            logger.info(f"[SYNC GENERATED] TopicID {topic_id}: Generated {len(chunks)} Gemini vector embeddings.")

        except Exception as err:
            logger.error(f"[SYNC ITEM ERROR] Failed generating embeddings for TopicID {topic_id} ('{title}'): {err}. Leaving Processed=0.")
            continue

    if records_to_save_in_sql:
        # Save chunks and vector floats directly into SQL Server
        database.save_chunk_embeddings(records_to_save_in_sql)
        database.mark_policies_as_processed(synced_topic_ids)

    # Rebuild FAISS index from updated SQL database and save to disk
    load_faiss_vector_store(api_key=api_key, force_rebuild=True)

    elapsed = time.perf_counter() - start_time
    logger.info(f"[PERF] Knowledge Base sync finished: {len(synced_topic_ids)} topics / {len(records_to_save_in_sql)} chunks synced in {elapsed:.3f}s")
    return len(records_to_save_in_sql), synced_topic_ids

def search_vector_store(
    query: str,
    k: int = 3,
    vector_store_instance: Optional[FAISS] = None,
    api_key: Optional[str] = None
) -> List[Tuple[Document, float]]:
    """
    Performs similarity search using the provided or disk-loaded FAISS index.
    """
    start_time = time.perf_counter()
    logger.info(f"Searching knowledge base for query: '{query}'...")

    vector_store = vector_store_instance or load_faiss_vector_store(api_key=api_key)

    if vector_store is None:
        logger.warning("Search failed: Vector store is empty.")
        return []

    raw_results = vector_store.similarity_search_with_score(query, k=k)

    # Apply score threshold filtering if configured
    if SIMILARITY_SCORE_THRESHOLD > 0.0:
        filtered_results = [res for res in raw_results if res[1] >= SIMILARITY_SCORE_THRESHOLD]
    else:
        filtered_results = raw_results

    elapsed = time.perf_counter() - start_time
    logger.info(f"[PERF] FAISS similarity search returned {len(filtered_results)} matching chunks in {elapsed:.3f}s")
    return filtered_results
