"""Custom exceptions for research operations."""

class ResearchError(Exception):
    """Base class for research-related errors."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message) 