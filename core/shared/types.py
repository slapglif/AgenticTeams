"""Shared type definitions to avoid circular imports."""

from typing import Dict, Any, List, Optional, Protocol, runtime_checkable

@runtime_checkable
class MemoryInterface(Protocol):
    """Protocol defining the memory manager interface."""
    
    async def store_task(self, task_id: str, task_data: Dict[str, Any]) -> bool:
        ...
        
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        ...
        
    async def store_memory(
        self,
        user_id: str,
        memory_type: str,
        content: Dict[str, Any]
    ) -> bool:
        ...
        
    async def get_memory(
        self, 
        user_id: str,
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        ...

    async def store_tool_result(self, operation_id: str, tool_name: str, result: Dict[str, Any], error: Optional[str] = None) -> None:
        """Store a tool execution result."""
        ...

@runtime_checkable
class DataProcessorInterface(Protocol):
    """Protocol defining the data processor interface."""
    
    async def process_data(
        self,
        data: Any,
        schema_type: Optional[str] = None,
        schema_class: Optional[Any] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        ... 