"""
db_setup.py
-----------
Database Initialization & Seed Script for RAG Assistant.

What this script does:
1. Connects to local Microsoft SQL Server.
2. Creates the database 'KB_RAG' if it does not exist.
3. Creates the 'company_policies' table.
4. Creates the 'policy_embeddings' table to store vector embeddings directly in SQL Server!
5. Inserts sample company policy data for testing.
6. Exports the exact SQL table data to 'Resource/company_policies.xlsx'.
"""

import pyodbc
import pandas as pd
from config import get_sql_connection_string, DB_NAME, EXCEL_PATH
from logger import logger

def initialize_database():
    """
    Creates the KB_RAG database, company_policies table, policy_embeddings table,
    populates seed data, and exports Excel file in Resource/.
    """
    print("\n==================================================")
    print("      STEP 1: INITIALIZING SQL SERVER DATABASE    ")
    print("==================================================\n")
    
    # ---------------------------------------------------------
    # 1. Connect to master database to check/create KB_RAG DB
    # ---------------------------------------------------------
    master_conn_str = get_sql_connection_string(include_db=False)
    logger.info("Connecting to SQL Server master database...")
    
    try:
        conn = pyodbc.connect(master_conn_str, autocommit=True)
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT database_id FROM sys.databases WHERE name = '{DB_NAME}'")
        db_exists = cursor.fetchone()
        
        if not db_exists:
            logger.info(f"Database '{DB_NAME}' not found. Creating database...")
            cursor.execute(f"CREATE DATABASE [{DB_NAME}]")
            print(f"[SUCCESS] Database '{DB_NAME}' created successfully.")
        else:
            logger.info(f"Database '{DB_NAME}' already exists.")
            
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Failed to connect or create database '{DB_NAME}': {e}")
        raise e

    # ---------------------------------------------------------
    # 2. Connect directly to KB_RAG database & create tables
    # ---------------------------------------------------------
    db_conn_str = get_sql_connection_string(include_db=True)
    logger.info(f"Connecting to database '{DB_NAME}'...")
    
    conn = pyodbc.connect(db_conn_str, autocommit=True)
    cursor = conn.cursor()
    
    # Create company_policies table
    cursor.execute("""
        IF OBJECT_ID('dbo.company_policies', 'U') IS NOT NULL
            DROP TABLE dbo.company_policies;
    """)
    
    create_policies_table_query = """
    CREATE TABLE dbo.company_policies (
        TopicID INT PRIMARY KEY,
        Category NVARCHAR(100) NOT NULL,
        Title NVARCHAR(200) NOT NULL,
        Content NVARCHAR(MAX) NOT NULL,
        LastUpdated NVARCHAR(50) NOT NULL,
        Processed INT NOT NULL DEFAULT 0  -- 0 = Pending Sync, 1 = Synced
    );
    """
    logger.info("Creating table 'company_policies'...")
    cursor.execute(create_policies_table_query)
    print("[SUCCESS] Table 'company_policies' created successfully.")

    # Create policy_embeddings table to store vector embeddings directly in SQL Server
    cursor.execute("""
        IF OBJECT_ID('dbo.policy_embeddings', 'U') IS NOT NULL
            DROP TABLE dbo.policy_embeddings;
    """)
    
    create_embeddings_table_query = """
    CREATE TABLE dbo.policy_embeddings (
        EmbeddingID INT IDENTITY(1,1) PRIMARY KEY,
        TopicID INT NOT NULL,
        ChunkIndex INT NOT NULL,
        ChunkText NVARCHAR(MAX) NOT NULL,
        VectorData NVARCHAR(MAX) NOT NULL, -- JSON array string of float vector numbers
        Title NVARCHAR(200) NOT NULL,
        Category NVARCHAR(100) NOT NULL,
        LastUpdated NVARCHAR(50) NOT NULL,
        CreatedAt DATETIME DEFAULT GETDATE()
    );
    """
    logger.info("Creating table 'policy_embeddings' in SQL Server...")
    cursor.execute(create_embeddings_table_query)
    print("[SUCCESS] Table 'policy_embeddings' created successfully in SQL Server.")

    # ---------------------------------------------------------
    # 3. Seed Example Company Policies Data
    # ---------------------------------------------------------
    sample_policies = [
        (
            101,
            "Workplace & HR",
            "Remote Work & Hybrid Work Policy",
            "Employees are eligible for up to 3 days of remote work per week with supervisor approval. "
            "Core business hours for remote availability are 9:00 AM to 5:00 PM EST. "
            "The company provides a one-time $300 stipend for home office ergonomic equipment setups.",
            "2026-01-15",
            0  # Initial status: Pending Sync
        ),
        (
            102,
            "Benefits & Time Off",
            "Paid Time Off (PTO) & Vacation Policy",
            "Full-time employees receive 20 days of paid time off per annual calendar year. "
            "A maximum of 5 unused PTO days can be carried over into the next calendar year. "
            "Sick leave is separate from PTO and provides 10 fully paid sick days annually.",
            "2026-02-01",
            0  # Initial status: Pending Sync
        ),
        (
            103,
            "Finance & Travel",
            "Expense Reimbursement & Travel Daily Meal Policy",
            "Business travel expenses must be submitted via the finance portal within 14 business days. "
            "The daily meal allowance for business trips is capped at $50 per day (itemized receipts required). "
            "Hotel stays are reimbursed up to $200 per night for domestic travel and $350 for international travel.",
            "2026-02-10",
            0  # Initial status: Pending Sync
        ),
        (
            104,
            "IT & Security",
            "Cybersecurity & Password Governance Policy",
            "Passwords must be at least 12 characters long and contain numbers, symbols, and uppercase letters. "
            "All employee corporate accounts require mandatory multi-factor authentication (MFA). "
            "Passwords must be changed every 90 days, and reusing the last 5 passwords is restricted.",
            "2026-03-01",
            0  # Initial status: Pending Sync
        ),
        (
            105,
            "Ethics & Governance",
            "Workplace Code of Conduct & Anti-Harassment Policy",
            "The company maintains a zero-tolerance policy regarding harassment, discrimination, and bullying. "
            "Employees must report compliance violations to HR or via the anonymous ethics hotline at extension 4040. "
            "Gifts from clients or vendors exceeding $100 in value must be declared to the Compliance Officer.",
            "2026-03-15",
            0  # Initial status: Pending Sync
        )
    ]

    insert_query = """
    INSERT INTO dbo.company_policies (TopicID, Category, Title, Content, LastUpdated, Processed)
    VALUES (?, ?, ?, ?, ?, ?);
    """
    
    logger.info("Inserting seed company policy records into SQL Server...")
    for policy in sample_policies:
        cursor.execute(insert_query, policy)
        print(f"[INSERTED] TopicID {policy[0]}: '{policy[2]}'")
        
    print(f"[SUCCESS] Total {len(sample_policies)} policy records seeded into SQL Server.")

    # ---------------------------------------------------------
    # 4. Export exact SQL data to Excel file in Resource/
    # ---------------------------------------------------------
    logger.info("Exporting database table to Excel spreadsheet...")
    try:
        df = pd.read_sql("SELECT * FROM dbo.company_policies", conn)
        df.to_excel(EXCEL_PATH, index=False)
        print(f"[EXCEL EXPORT] Successfully created Excel backup at: {EXCEL_PATH}")
    except Exception as excel_err:
        logger.warning(f"Could not update Excel file (it may be open in Excel): {excel_err}")

    cursor.close()
    conn.close()
    print("\n==================================================")
    print("   SQL DATABASE INITIALIZATION COMPLETED PERFECTLY ")
    print("==================================================\n")

if __name__ == "__main__":
    initialize_database()
