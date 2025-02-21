"""
Document processor for RAG pipeline that handles HTML cleaning, text chunking, and metadata management.
"""

import re
import html
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from datetime import datetime, UTC
from langchain_core.documents import Document
from loguru import logger
import json
import hashlib

class DocumentProcessor:
    """Handles document processing for RAG pipeline including cleaning, chunking, and metadata management."""
    
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        """Initialize document processor.
        
        Args:
            chunk_size: Size of text chunks (default: 250 chars)
            chunk_overlap: Overlap between chunks (default: 50 chars)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"Initialized DocumentProcessor with chunk_size={chunk_size}, overlap={chunk_overlap}")

    def clean_html(self, html_content: str) -> str:
        """Clean HTML content and extract meaningful text.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Cleaned text content
        """
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "meta", "noscript", "header", "footer", "nav"]):
                script.decompose()
            
            # Get text and clean
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text)
            text = html.unescape(text)
            
            # Remove any remaining HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning HTML: {e}")
            # Return original text if cleaning fails
            return html_content

    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """Split text into overlapping chunks and create Document objects.
        
        Args:
            text: Text to split into chunks
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of Document objects with chunks and metadata
        """
        try:
            chunks = []
            start = 0
            text_length = len(text)

            while start < text_length:
                # Get chunk of text
                end = start + self.chunk_size
                chunk_text = text[start:end]
                
                # Generate unique ID for chunk
                chunk_id = hashlib.sha256(
                    f"{metadata.get('url', '')}-{start}-{end}".encode()
                ).hexdigest()[:16]
                
                # Create chunk metadata
                chunk_metadata = {
                    **metadata,
                    "chunk_id": chunk_id,
                    "chunk_start": start,
                    "chunk_end": end,
                    "chunk_size": len(chunk_text),
                    "total_chunks": (text_length + self.chunk_size - 1) // self.chunk_size,
                    "processed_timestamp": datetime.now(UTC).isoformat()
                }
                
                # Create Document object
                doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                )
                chunks.append(doc)
                
                # Move to next chunk with overlap
                start = end - self.chunk_overlap
            
            logger.debug(f"Created {len(chunks)} chunks from text of length {text_length}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            # Return single chunk with error metadata if chunking fails
            error_metadata = {
                **metadata,
                "error": str(e),
                "chunk_error": True,
                "processed_timestamp": datetime.now(UTC).isoformat()
            }
            return [Document(page_content=text, metadata=error_metadata)]

    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Process plain text content into chunks.
        
        Args:
            text: Plain text content to process
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Document objects with chunks and metadata
        """
        if metadata is None:
            metadata = {}
        
        # Add text-specific metadata
        text_metadata = {
            **metadata,
            "content_type": "text",
            "text_length": len(text)
        }
        
        # Process as raw document with is_html=False
        return self.process_raw_document(text, text_metadata, is_html=False)

    def process_raw_document(self, content: str, metadata: Dict[str, Any], is_html: bool = True) -> List[Document]:
        """Process a raw document including cleaning and chunking.
        
        Args:
            content: Raw document content
            metadata: Document metadata
            is_html: Whether content is HTML (default: True)
            
        Returns:
            List of processed Document objects
        """
        try:
            # Clean content if it's HTML
            cleaned_text = self.clean_html(content) if is_html else content
            
            # Add processing metadata
            processing_metadata = {
                **metadata,
                "is_html": is_html,
                "original_length": len(content),
                "cleaned_length": len(cleaned_text),
                "processing_timestamp": datetime.now(UTC).isoformat()
            }
            
            # Create chunks with metadata
            chunks = self.create_chunks(cleaned_text, processing_metadata)
            
            # Store reference to original text in metadata
            doc_id = hashlib.sha256(
                f"{metadata.get('url', '')}-{content[:100]}".encode()
            ).hexdigest()[:16]
            
            # Update all chunks with document ID
            for chunk in chunks:
                chunk.metadata["document_id"] = doc_id
            
            logger.info(f"Processed document into {len(chunks)} chunks with ID {doc_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            # Return single error document if processing fails
            error_metadata = {
                **metadata,
                "error": str(e),
                "processing_error": True,
                "processing_timestamp": datetime.now(UTC).isoformat()
            }
            return [Document(page_content=content, metadata=error_metadata)]

    def batch_process_documents(self, documents: List[Dict[str, Any]]) -> List[Document]:
        """Process multiple documents in batch.
        
        Args:
            documents: List of documents with content and metadata
            
        Returns:
            List of processed Document objects
        """
        try:
            processed_docs = []
            for doc in documents:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                is_html = doc.get("is_html", True)
                
                chunks = self.process_raw_document(content, metadata, is_html)
                processed_docs.extend(chunks)
            
            logger.info(f"Batch processed {len(documents)} documents into {len(processed_docs)} chunks")
            return processed_docs
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return [] 