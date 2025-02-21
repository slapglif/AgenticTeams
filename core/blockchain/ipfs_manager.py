"""
IPFS Manager - Handles IPFS storage and retrieval with real IPFS node support.
"""
import json
from typing import Dict, Any, Optional
from loguru import logger
from datetime import datetime, UTC
import ipfshttpclient
import aiohttp
import asyncio

class IPFSManager:
    """Manages IPFS storage operations."""
    def __init__(self, api_endpoint: str = "/ip4/127.0.0.1/tcp/5001"):
        """Initialize IPFS manager.
        
        Args:
            api_endpoint: IPFS API endpoint
        """
        self.api_endpoint = api_endpoint
        self.client = ipfshttpclient.connect(api_endpoint)
        self.last_task_id = None
        
    async def store_data(self, data: Dict[str, Any]) -> str:
        """Store data in IPFS.
        
        Args:
            data: Data to store including task data, memories, and metadata
            
        Returns:
            IPFS hash
        """
        try:
            # Store task ID for reference
            self.last_task_id = data.get('task_id')
            logger.debug(f"Storing task data in IPFS: {self.last_task_id}")
            
            # Add metadata and timestamps
            enriched_data = {
                **data,
                "metadata": {
                    "stored_at": datetime.now(UTC).isoformat(),
                    "version": "1.0",
                    "type": "task_execution",
                    **data.get("metadata", {})
                }
            }
            
            # Convert to JSON and store
            content = json.dumps(enriched_data)
            result = await asyncio.to_thread(self.client.add_json, enriched_data)
            
            # Pin the content
            await asyncio.to_thread(self.client.pin.add, result)
            
            logger.info(f"Stored data in IPFS with hash: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error storing data in IPFS: {e}")
            raise
            
    async def get_data(self, ipfs_hash: str) -> Dict[str, Any]:
        """Get data from IPFS.
        
        Args:
            ipfs_hash: IPFS hash to retrieve
            
        Returns:
            Retrieved data
        """
        try:
            logger.debug(f"Getting data from IPFS with hash: {ipfs_hash}")
            
            # Retrieve and parse data
            data = await asyncio.to_thread(self.client.get_json, ipfs_hash)
            
            logger.info(f"Retrieved data from IPFS hash: {ipfs_hash}")
            return data
            
        except Exception as e:
            logger.error(f"Error getting data from IPFS: {e}")
            raise
            
    async def store_memory(self, memory_data: Dict[str, Any]) -> str:
        """Store memory data in IPFS.
        
        Args:
            memory_data: Memory data including embeddings and metadata
            
        Returns:
            IPFS hash
        """
        try:
            # Add memory-specific metadata
            enriched_memory = {
                **memory_data,
                "metadata": {
                    "stored_at": datetime.now(UTC).isoformat(),
                    "version": "1.0",
                    "type": "memory",
                    **memory_data.get("metadata", {})
                }
            }
            
            # Store in IPFS
            result = await asyncio.to_thread(self.client.add_json, enriched_memory)
            await asyncio.to_thread(self.client.pin.add, result)
            
            logger.info(f"Stored memory in IPFS with hash: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error storing memory in IPFS: {e}")
            raise
            
    async def store_graph(self, graph_data: Dict[str, Any]) -> str:
        """Store computation graph data in IPFS.
        
        Args:
            graph_data: Graph data including nodes, edges and metadata
            
        Returns:
            IPFS hash
        """
        try:
            # Add graph-specific metadata
            enriched_graph = {
                **graph_data,
                "metadata": {
                    "stored_at": datetime.now(UTC).isoformat(),
                    "version": "1.0", 
                    "type": "computation_graph",
                    **graph_data.get("metadata", {})
                }
            }
            
            # Store in IPFS
            result = await asyncio.to_thread(self.client.add_json, enriched_graph)
            await asyncio.to_thread(self.client.pin.add, result)
            
            logger.info(f"Stored graph in IPFS with hash: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error storing graph in IPFS: {e}")
            raise
            
    def __del__(self):
        """Clean up IPFS client connection."""
        try:
            self.client.close()
        except:
            pass 