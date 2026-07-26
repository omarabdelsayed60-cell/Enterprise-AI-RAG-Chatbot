"""
logger.py
---------
Configures logging for the RAG Assistant project.
Provides a clean, readable logging format so you can trace every operation in console and log files.
"""

import logging
import sys

def setup_logger(name: str = "RAG_Assistant") -> logging.Logger:
    """
    Creates and returns a configured logger instance.
    
    Args:
        name (str): Name of the logger (defaults to RAG_Assistant)
        
    Returns:
        logging.Logger: Configured logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if logger is initialized multiple times
    if not logger.handlers:
        # Formatter for log messages: [Timestamp] [Level] Message
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Stream Handler (console output)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

# Export a default logger for simple importing
logger = setup_logger()
