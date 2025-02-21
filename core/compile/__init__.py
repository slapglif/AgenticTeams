"""
LangGraph Compiler package for executing language model task graphs.

This package provides tools for:
1. Building task graphs from descriptions
2. Scheduling operations across workers
3. Executing operations in dependency order
4. Collecting and analyzing results
"""

from .compiler import LangGraphCompiler, CompilerConfig
from .models import (
    Tool,
    Task,
    State,
    FinalResponse,
    Replan,
    JoinOutputs,
    TaskOperation,
    ExecutionPlan,
    OperationResult,
    ExecutionResult
)
from .exceptions import (
    CompilerError,
    PlanningError,
    ExecutionError,
    ValidationError,
    ToolError,
    DependencyError,
    TimeoutError,
    ResourceError,
    GraphError,
    MetricsError,
    StateError,
    SerializationError,
    ConfigurationError
)
from .graph.builder import build_graph
from .graph.executor import (
    execute_operation,
    execute_task_graph,
    execute_with_retry,
    execute_with_fallback,
    execute_batch
)
from .graph.scheduler import (
    Schedule,
    ScheduleSlot,
    schedule_operations,
    optimize_schedule,
    analyze_schedule,
    estimate_duration
)

__version__ = "0.1.0"
__author__ = "Michael Brown"
__all__ = [
    # Main compiler
    "LangGraphCompiler",
    "CompilerConfig",
    
    # Models
    "Tool",
    "Task",
    "State",
    "FinalResponse",
    "Replan",
    "JoinOutputs",
    "TaskOperation",
    "ExecutionPlan",
    "OperationResult",
    "ExecutionResult",
    
    # Exceptions
    "CompilerError",
    "PlanningError",
    "ExecutionError",
    "ValidationError",
    "ToolError",
    "DependencyError",
    "TimeoutError",
    "ResourceError",
    "GraphError",
    "MetricsError",
    "StateError",
    "SerializationError",
    "ConfigurationError",
    
    # Graph utilities
    "build_graph",
    "execute_operation",
    "execute_task_graph",
    "execute_with_retry",
    "execute_with_fallback",
    "execute_batch",
    
    # Scheduling
    "Schedule",
    "ScheduleSlot",
    "schedule_operations",
    "optimize_schedule",
    "analyze_schedule",
    "estimate_duration"
] 