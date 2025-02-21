"""
Data models for the LangGraph Compiler.

This module contains:
1. Tool and task models
2. State and execution models
3. Response models
"""
from typing import Dict, Any, List, Optional, Union, Callable, Sequence
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Required, NotRequired
from langchain.schema import BaseMessage
from langchain_core.messages import MessageLikeRepresentation
from datetime import datetime

from core.shared.models import ExecutionPlan, ExecutionResult, TaskOperation

class Tool:
    """Tool definition."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable[..., Any]

    def __init__(self, name: str, description: str, parameters: Dict[str, Any], function: Callable[..., Any]):
        """Initialize tool."""
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool function."""
        return await self.function(*args, **kwargs)

    def __str__(self) -> str:
        """String representation."""
        return f"Tool({self.name})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Tool(name='{self.name}', description='{self.description}', parameters={self.parameters})"

class Task(BaseModel):
    """Task definition."""
    id: str = Field(description="Task ID")
    description: str = Field(description="Task description")
    tool: Tool = Field(description="Tool to execute")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")
    dependencies: List[str] = Field(default_factory=list, description="IDs of tasks this task depends on")

    model_config = {
        "arbitrary_types_allowed": True
    }

    def __str__(self) -> str:
        """String representation."""
        return f"Task({self.id}: {self.description})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Task(id='{self.id}', description='{self.description}', tool={self.tool}, args={self.args}, dependencies={self.dependencies})"

class OperationResult(BaseModel):
    """Result of executing an operation."""
    operation_id: str = Field(description="Operation ID")
    success: bool = Field(description="Whether the operation succeeded")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Operation result data")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")
    start_time: Optional[str] = Field(default=None, description="Operation start time")
    end_time: Optional[str] = Field(default=None, description="Operation end time")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Optional metrics from the operation")

class FinalResponse(BaseModel):
    """Final response for a completed task."""
    task_id: str = Field(description="Task ID")
    success: bool = Field(description="Whether the task succeeded")
    result: Dict[str, Any] = Field(description="Task result data")
    summary: str = Field(description="Summary of the task execution")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Optional metrics from the task")

class Replan(BaseModel):
    """Request to replan a task."""
    feedback: str = Field(description="Feedback on why replanning is needed")
    error_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Analysis of what went wrong")

class JoinOutputs(BaseModel):
    """Output from the joiner analyzing execution results."""
    thought: str = Field(description="Joiner's analysis of the results")
    action: Union[Replan, FinalResponse] = Field(description="Action to take based on analysis")

class State(TypedDict, total=False):
    """State definition for LangGraph.
    
    Required fields are marked with Required[], optional with NotRequired[]
    """
    messages: Required[List[MessageLikeRepresentation]]  # Using List for invariance
    task: Required[Task]
    metrics: Required[Dict[str, Any]]
    execution_plan: NotRequired[ExecutionPlan]
    schedule: NotRequired[Dict[str, Any]]
    execution_result: NotRequired[ExecutionResult]
    analysis: NotRequired[Dict[str, Any]]  # Analysis results from execution
