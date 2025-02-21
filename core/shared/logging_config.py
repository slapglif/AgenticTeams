"""
Centralized logging configuration for the application.
"""
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback

# Initialize Rich console
console = Console()

def setup_logging(log_dir="logs/e2e_tests"):
    """Configure logging with both console and file outputs."""
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add Rich console handler for INFO and above
    logger.add(
        RichHandler(
            console=console,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True
        ),
        level="INFO",
        format="{message}"
    )
    
    # Add debug file handler with rotation
    logger.add(
        os.path.join(log_dir, "debug_{time}.log"),
        rotation="500 MB",
        retention="10 days",
        compression="zip",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        backtrace=True,
        diagnose=True
    )
    
    # Add error file handler
    logger.add(
        os.path.join(log_dir, "error_{time}.log"),
        rotation="100 MB",
        retention="10 days",
        compression="zip",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
        backtrace=True,
        diagnose=True
    )
    
    # Install rich traceback handler
    install_rich_traceback(show_locals=True)
    
    # Log initial setup
    logger.info("Logging configured with directory: {}", log_dir)

def log_operation(operation_name, **kwargs):
    """Log an operation with rich formatting and detailed debug info."""
    # Always log timestamp for debugging
    logger.debug("Operation: {}", operation_name)
    
    # Log each kwarg as a separate debug entry
    for key, value in kwargs.items():
        logger.debug("{}: {}", key, value)
    
    # For console, format a nice summary
    if kwargs:
        summary = [f"{k}={v}" for k, v in kwargs.items()]
        logger.info("[bold]{}: [/bold]{}", operation_name, ", ".join(summary))
    else:
        logger.info("[bold]{}[/bold]", operation_name)

def get_logger(name):
    """Get a logger instance with the specified name."""
    return logger.bind(name=name) 