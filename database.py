"""
database.py
-----------
SQL Server database helper module for RAG Assistant.
Handles querying policies, saving vector embeddings directly into SQL Server,
and updating synchronization statuses.

SQL Server is the single source of truth for all policy documents AND vector embeddings.
"""

import pyodbc
import json
import time
from typing import List, Dict, Any
from config import get_sql_connection_string
from logger import logger

def get_connection() -> pyodbc.Connection:
    """
    Establishes and returns a pyodbc connection to the KB_RAG database.
    """
    conn_str = get_sql_connection_string(include_db=True)
    return pyodbc.connect(conn_str, autocommit=True)

def get_all_policies() -> List[Dict[str, Any]]:
    """
    Fetches all policy records from SQL Server.
    """
    start_time = time.perf_counter()
    logger.info("Fetching all policy records from SQL Server database...")
    query = "SELECT TopicID, Category, Title, Content, LastUpdated, Processed FROM dbo.company_policies ORDER BY TopicID"
    
    policies = []
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        for row in rows:
            policies.append({
                "TopicID": row.TopicID,
                "Category": row.Category,
                "Title": row.Title,
                "Content": row.Content,
                "LastUpdated": row.LastUpdated,
                "Processed": row.Processed
            })
            
    elapsed = time.perf_counter() - start_time
    logger.info(f"[PERF] SQL get_all_policies fetched {len(policies)} records in {elapsed:.3f}s")
    return policies

def get_unprocessed_policies() -> List[Dict[str, Any]]:
    """
    Fetches records where Processed = 0 (pending sync).
    """
    start_time = time.perf_counter()
    logger.info("Querying SQL Server for unprocessed policies (Processed = 0)...")
    query = "SELECT TopicID, Category, Title, Content, LastUpdated, Processed FROM dbo.company_policies WHERE Processed = 0 ORDER BY TopicID"
    
    unprocessed = []
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        for row in rows:
            unprocessed.append({
                "TopicID": row.TopicID,
                "Category": row.Category,
                "Title": row.Title,
                "Content": row.Content,
                "LastUpdated": row.LastUpdated,
                "Processed": row.Processed
            })
            
    elapsed = time.perf_counter() - start_time
    logger.info(f"[PERF] SQL get_unprocessed_policies fetched {len(unprocessed)} records in {elapsed:.3f}s")
    return unprocessed

def mark_policies_as_processed(topic_ids: List[int]) -> int:
    """
    Updates SQL Server records setting Processed = 1 for the given topic_ids list.
    """
    if not topic_ids:
        return 0
        
    start_time = time.perf_counter()
    logger.info(f"Updating SQL Server Processed=1 for TopicIDs: {topic_ids}")
    placeholders = ",".join(["?"] * len(topic_ids))
    query = f"UPDATE dbo.company_policies SET Processed = 1 WHERE TopicID IN ({placeholders})"
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, topic_ids)
        rowcount = cursor.rowcount
        
    elapsed = time.perf_counter() - start_time
    logger.info(f"[PERF] SQL mark_policies_as_processed updated {rowcount} rows in {elapsed:.3f}s")
    return rowcount

def save_chunk_embeddings(embeddings_records: List[Dict[str, Any]]) -> int:
    """
    Saves text chunks and their Gemini vector embeddings directly into SQL Server
    (dbo.policy_embeddings table).
    
    Args:
        embeddings_records (List[Dict]): List of dicts containing topic_id, chunk_index,
                                         chunk_text, vector (List[float]), title, category, last_updated.
    """
    if not embeddings_records:
        return 0
        
    start_time = time.perf_counter()
    logger.info(f"Saving {len(embeddings_records)} vector embeddings into SQL Server dbo.policy_embeddings table...")
    insert_query = """
    INSERT INTO dbo.policy_embeddings (TopicID, ChunkIndex, ChunkText, VectorData, Title, Category, LastUpdated)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    
    with get_connection() as conn:
        cursor = conn.cursor()
        for rec in embeddings_records:
            # Delete any previous embeddings for this topic before inserting replacement
            cursor.execute("DELETE FROM dbo.policy_embeddings WHERE TopicID = ? AND ChunkIndex = ?", (rec["topic_id"], rec["chunk_index"]))
            
            # Serialize float list vector to JSON string
            vector_json = json.dumps(rec["vector"])
            
            cursor.execute(insert_query, (
                rec["topic_id"],
                rec["chunk_index"],
                rec["chunk_text"],
                vector_json,
                rec["title"],
                rec["category"],
                rec["last_updated"]
            ))
            
    elapsed = time.perf_counter() - start_time
    logger.info(f"[PERF] SQL save_chunk_embeddings stored {len(embeddings_records)} records in {elapsed:.3f}s")
    return len(embeddings_records)

def get_all_stored_embeddings() -> List[Dict[str, Any]]:
    """
    Retrieves all stored vector embeddings directly from SQL Server table dbo.policy_embeddings.
    
    Returns:
        List[Dict[str, Any]]: List of chunk texts, metadata, and float vector arrays.
    """
    start_time = time.perf_counter()
    logger.info("Reading vector embeddings from SQL Server dbo.policy_embeddings...")
    query = "SELECT EmbeddingID, TopicID, ChunkIndex, ChunkText, VectorData, Title, Category, LastUpdated FROM dbo.policy_embeddings"
    
    results = []
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        for row in rows:
            vector_floats = json.loads(row.VectorData)
            results.append({
                "embedding_id": row.EmbeddingID,
                "topic_id": row.TopicID,
                "chunk_index": row.ChunkIndex,
                "chunk_text": row.ChunkText,
                "vector": vector_floats,
                "title": row.Title,
                "category": row.Category,
                "last_updated": row.LastUpdated
            })
            
    elapsed = time.perf_counter() - start_time
    logger.info(f"[PERF] SQL get_all_stored_embeddings fetched {len(results)} vector rows in {elapsed:.3f}s")
    return results

def delete_topic_embeddings(topic_id: int) -> int:
    """
    Deletes all vector embedding records for a specific TopicID from SQL Server.
    """
    logger.info(f"Deleting embeddings for TopicID {topic_id} from SQL Server...")
    query = "DELETE FROM dbo.policy_embeddings WHERE TopicID = ?"
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (topic_id,))
        return cursor.rowcount

def cleanup_deleted_topic_embeddings() -> int:
    """
    Deletes orphan vector embeddings in dbo.policy_embeddings for TopicIDs that no longer exist in dbo.company_policies.
    """
    start_time = time.perf_counter()
    query = """
    DELETE FROM dbo.policy_embeddings
    WHERE TopicID NOT IN (SELECT TopicID FROM dbo.company_policies)
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        deleted_count = cursor.rowcount
    elapsed = time.perf_counter() - start_time
    logger.info(f"[PERF] Cleaned up {deleted_count} orphan embeddings for deleted policies in {elapsed:.3f}s")
    return deleted_count
