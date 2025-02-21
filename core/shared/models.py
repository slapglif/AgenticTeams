"""Data models for task execution and planning."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, ItemsView, Union
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

# Base task execution models
class TaskOperation(BaseModel):
    """Represents a single operation in a task execution plan."""
    id: str = Field(description="Unique identifier for the operation")
    tool_name: str = Field(description="Name of the tool to execute")
    description: str = Field(description="Description of what the operation does")
    dependencies: List[str] = Field(default_factory=list, description="List of operation IDs this operation depends on")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool execution")
    status: str = Field(default="pending", description="Current status of the operation")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Result of the operation execution")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")
    start_time: Optional[str] = Field(default=None, description="ISO format timestamp when operation started")
    end_time: Optional[str] = Field(default=None, description="ISO format timestamp when operation completed")

class ExecutionPlan(BaseModel):
    """Represents a plan for executing a task."""
    task_id: str = Field(description="Unique identifier for the task")
    description: str = Field(description="Description of the task")
    operations: List[TaskOperation] = Field(default_factory=list, description="List of operations to execute")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the plan")
    status: str = Field(default="pending", description="Current status of the plan")
    start_time: Optional[datetime] = Field(default=None, description="When execution started")
    end_time: Optional[datetime] = Field(default=None, description="When execution completed")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")

class ExecutionResult(BaseModel):
    """Result of executing a task."""
    operations: List[Dict[str, Any]] = Field(default_factory=list)
    success: bool = Field(default=False)
    error: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    synthesis: Dict[str, Any] = Field(default_factory=dict)
    document_path: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def __init__(self, **data):
        # Convert TaskOperation objects to dictionaries
        if "operations" in data and isinstance(data["operations"], list):
            data["operations"] = [
                op.dict() if hasattr(op, "dict") else op
                for op in data["operations"]
            ]
        super().__init__(**data)

    def __getitem__(self, key: str) -> Any:
        """Get item from the model."""
        return getattr(self, key)

    def items(self) -> ItemsView[str, Any]:
        """Get items from the model."""
        return self.dict().items()

    def model_dump(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
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

# Research tool schemas
class SearchSchema(BaseModel):
    """Schema for search tool arguments."""
    query: str = Field(..., description="Search query string")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Optional configuration parameters")

class AspectSchema(BaseModel):
    """Schema for aspect-based search arguments."""
    aspect: Dict[str, Any] = Field(..., description="Aspect to generate search query for")
    reasoning: Optional[str] = Field(None, description="Reasoning about why this aspect is important")

class ContextSchema(BaseModel):
    """Schema for context-based analysis arguments."""
    context: Dict[str, Any] = Field(..., description="Context for analysis")

class SearchQuery(BaseModel):
    """Schema for search query results."""
    primary_query: str = Field(..., description="Main search query")
    alternative_queries: List[str] = Field(..., description="Alternative search queries")
    subtopics: List[str] = Field(..., description="Related subtopics to search")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class DataAnalysisResponse(BaseModel):
    """Schema for data analysis results."""
    key_metrics: Dict[str, float] = Field(..., description="Key metrics from the analysis")
    correlations: List[Dict[str, Any]] = Field(..., description="Correlations found in the data")
    significant_findings: List[Dict[str, Any]] = Field(..., description="Significant findings from the analysis")
    analysis: Dict[str, Any] = Field(..., description="Detailed analysis results")
    insights: List[str] = Field(..., description="Key insights from analysis")
    confidence: float = Field(..., description="Confidence score for analysis", ge=0, le=1)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class NetworkAnalysisResponse(BaseModel):
    """Schema for network analysis results."""
    network_topology: Dict[str, Any] = Field(..., description="Network topology information")
    key_nodes: List[Dict[str, Any]] = Field(..., description="Key nodes in the network")
    relationships: List[Dict[str, Any]] = Field(..., description="Relationships between nodes")
    communities: List[Dict[str, Any]] = Field(..., description="Community detection results")
    metrics: Dict[str, float] = Field(..., description="Network metrics")
    findings: List[Dict[str, Any]] = Field(..., description="Analysis findings")

class GraphAnalysisResponse(BaseModel):
    """Schema for graph analysis results."""
    topology: Dict[str, Any] = Field(..., description="Graph topology information")
    nodes: List[Dict[str, Any]] = Field(..., description="Graph nodes")
    edges: List[Dict[str, Any]] = Field(..., description="Graph edges")
    metrics: Dict[str, Union[float, Dict[str, float]]] = Field(..., description="Graph metrics, can include nested metrics like centrality measures")
    communities: List[Dict[str, Any]] = Field(..., description="Community detection results")
    findings: List[Dict[str, Any]] = Field(..., description="Analysis findings")

class CitationAnalysisResponse(BaseModel):
    """Schema for citation analysis results."""
    citations: List[Dict[str, Any]] = Field(..., description="List of citations analyzed")
    patterns: List[Dict[str, Any]] = Field(..., description="Citation patterns identified")
    recommendations: List[Dict[str, Any]] = Field(..., description="Citation-based recommendations")
    synthesis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Synthesized insights including sources and key insights"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class CitationInput(BaseModel):
    """Input schema for citation analysis."""
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="List of citations to analyze")
    query: Optional[str] = Field(default=None, description="Optional search query for context")
    reasoning: Optional[str] = Field(default=None, description="Reasoning about the citation analysis")
    implementation_notes: Optional[str] = Field(default=None, description="Notes about implementation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata") 