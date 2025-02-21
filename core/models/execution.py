from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

class ExecutionResult(BaseModel):
    operations: List[Dict[str, Any]]
    success: bool = True
    error: Optional[str] = None
    metrics: Dict[str, Any] = {}
    synthesis: Dict[str, Any] = {}
    document_path: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def items(self):
        return {
            "operations": self.operations,
            "success": self.success,
            "error": self.error,
            "metrics": self.metrics,
            "synthesis": self.synthesis,
            "document_path": self.document_path,
            "start_time": self.start_time,
            "end_time": self.end_time
        }.items()

    def model_dump(self) -> Dict[str, Any]:
        return {
            "operations": self.operations,
            "success": self.success,
            "error": self.error,
            "metrics": self.metrics,
            "synthesis": self.synthesis,
            "document_path": self.document_path,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None
        } 