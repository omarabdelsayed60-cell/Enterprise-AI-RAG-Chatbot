"""
rag_pipeline.py
---------------
LangChain RAG QA Pipeline using Google Gemini.

Flow:
1. Receives user question.
2. Performs vector search over FAISS vector index (session-isolated / disk-loaded).
3. Constructs strict prompt enforcing answers ONLY from retrieved policy context.
4. Invokes ChatGoogleGenerativeAI using the configured single model (GEMINI_MODEL_NAME) with exponential backoff retries.
"""

import time
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL_NAME,
    TOP_K_RESULTS,
    MAX_RETRIES,
    BACKOFF_FACTOR
)
from logger import logger
import vector_store

# ---------------------------------------------------------
# Strict System Prompt Template
# Guarantees the LLM answers ONLY from retrieved SQL data
# ---------------------------------------------------------
STRICT_RAG_PROMPT_TEMPLATE = """
You are a helpful corporate policy assistant. Answer the user's question using ONLY the provided company policy context below.

STRICT RULES:
1. Base your answer STRICTLY on the retrieved policy context provided below.
2. Do NOT use outside knowledge, external facts, or make unverified assumptions.
3. If the answer cannot be found directly within the provided policy context, respond EXACTLY with:
   "I cannot answer this question based on the provided company policy data."
4. Be concise, direct, and professional.

RETRIEVED POLICY CONTEXT:
--------------------------------------------------
{context}
--------------------------------------------------

USER QUESTION: {question}

YOUR STRICT ANSWER:
"""

def get_llm_model(api_key: Optional[str] = None) -> ChatGoogleGenerativeAI:
    """
    Initializes and returns a fresh ChatGoogleGenerativeAI model instance per request.
    """
    key = api_key or GEMINI_API_KEY
    if not key:
        logger.error("GEMINI_API_KEY is missing!")
        raise ValueError("Google Gemini API Key is missing. Please set GEMINI_API_KEY in your .env file.")
        
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        google_api_key=key
    )

def ask_rag_assistant(
    question: str,
    k: Optional[int] = None,
    vector_store_instance: Optional[Any] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main RAG function:
    1. Searches vector store for top-k relevant chunks from SQL Server FAISS index.
    2. Formats retrieved chunks as context.
    3. Invokes configured Gemini LLM with strict prompt and exponential backoff retries.
    """
    total_start_time = time.perf_counter()
    k_val = k or TOP_K_RESULTS
    logger.info(f"Querying RAG Pipeline for question: '{question}' (top_k={k_val})")
    
    # 1. Retrieve top-k chunks from FAISS vector store
    search_start = time.perf_counter()
    search_results = vector_store.search_vector_store(
        query=question,
        k=k_val,
        vector_store_instance=vector_store_instance,
        api_key=api_key
    )
    search_elapsed = time.perf_counter() - search_start
    logger.info(f"[PERF] RAG similarity search completed in {search_elapsed:.3f}s")
    
    if not search_results:
        logger.warning("[RAG WARNING] No relevant policy chunks retrieved.")
        total_elapsed = time.perf_counter() - total_start_time
        logger.info(f"[PERF] RAG total question response time: {total_elapsed:.3f}s")
        return {
            "question": question,
            "answer": "I cannot answer this question based on the provided company policy data.",
            "retrieved_chunks": []
        }
        
    # 2. Build Context String from retrieved document chunks
    formatted_context_blocks = []
    retrieved_sources = []
    
    for idx, (doc, score) in enumerate(search_results):
        block = f"[Source #{idx+1} | Topic: {doc.metadata.get('title')}]\n{doc.page_content}"
        formatted_context_blocks.append(block)
        
        retrieved_sources.append({
            "topic_id": doc.metadata.get("topic_id"),
            "title": doc.metadata.get("title"),
            "category": doc.metadata.get("category"),
            "last_updated": doc.metadata.get("last_updated"),
            "content": doc.page_content,
            "similarity_score": round(float(score), 4)
        })
        
    combined_context = "\n\n".join(formatted_context_blocks)
    
    # 3. Format Prompt & Call LLM with Exponential Backoff Retry (No Model Switching)
    prompt = PromptTemplate(
        template=STRICT_RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    formatted_prompt = prompt.format(context=combined_context, question=question)
    
    answer_text = ""
    last_exception = None
    llm_start = time.perf_counter()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Attempting response generation with model '{GEMINI_MODEL_NAME}' (attempt {attempt}/{MAX_RETRIES})...")
            llm = get_llm_model(api_key=api_key)
            response = llm.invoke(formatted_prompt)
            
            if isinstance(response.content, list):
                answer_text = "".join([part.get("text", str(part)) if isinstance(part, dict) else str(part) for part in response.content]).strip()
            else:
                answer_text = str(response.content).strip()
                
            if answer_text:
                break
        except Exception as err:
            last_exception = err
            logger.warning(f"Model '{GEMINI_MODEL_NAME}' attempt {attempt} failed: {err}")
            if attempt < MAX_RETRIES:
                sleep_time = BACKOFF_FACTOR ** attempt
                logger.info(f"Retrying Gemini LLM call in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

    llm_elapsed = time.perf_counter() - llm_start
    logger.info(f"[PERF] Gemini LLM generation completed in {llm_elapsed:.3f}s")

    if not answer_text:
        logger.error(f"[LLM GENERATION ERROR] Model '{GEMINI_MODEL_NAME}' failed: {last_exception}")
        answer_text = "I cannot answer this question right now based on the company policy data. Please try again in a few moments or contact HR."

    total_elapsed = time.perf_counter() - total_start_time
    logger.info(f"[PERF] RAG total question response time: {total_elapsed:.3f}s")

    return {
        "question": question,
        "answer": answer_text,
        "retrieved_chunks": retrieved_sources
    }
