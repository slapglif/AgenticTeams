"""
Core utilities for LangGraph execution and state management.
"""
from typing import Dict, List, Any, Optional, Union, Callable, Type, cast
from datetime import datetime, timezone
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger
from langchain.prompts import ChatPromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.messages import MessageLikeRepresentation
from langchain_core.runnables import RunnableSerializable

from .models import State, Tool, Task, ExecutionResult, TaskOperation, ExecutionPlan, OperationResult
from core.schemas.pydantic_schemas import (
    BrainstormingResponse, FinalPlan, SpecializedAgentResponse,
    SummarizationResponse, StepCompleteResponse, SupervisorResponse,
    TopicResponse, FinalResponse, MetaReviewResponse, RevisedResponse,
    NetworkAnalysisResponse, ToolDataAnalysis, ToolGraphAnalysis, ToolCitationAnalysis,
    MemoryQuery, MemoryUpdate, MemoryContextWindow,
    ATRWorkUnit, ResponsibilityNFT
)

UTC = timezone.utc
console = Console()

# Map output modes to their corresponding Pydantic models
OUTPUT_MODEL_MAP = {
    "brainstorming": BrainstormingResponse,
    "plan": FinalPlan,
    "specialized": SpecializedAgentResponse,
    "summarization": SummarizationResponse,
    "step_complete": StepCompleteResponse,
    "supervisor": SupervisorResponse,
    "topic": TopicResponse,
    "final": FinalResponse,
    "meta_review": MetaReviewResponse,
    "revised": RevisedResponse,
    "data_analysis": ToolDataAnalysis,
    "graph_analysis": ToolGraphAnalysis,
    "citation_analysis": ToolCitationAnalysis,
    "network_analysis": NetworkAnalysisResponse,
    "memory_query": MemoryQuery,
    "memory_update": MemoryUpdate,
    "memory_context": MemoryContextWindow,
    "work_unit": ATRWorkUnit,
    "nft": ResponsibilityNFT
}


def add_message_to_state(state: State, message: BaseMessage) -> State:
    """Add a message to the state's message history.
    
    Args:
        state: Current state
        message: Message to add
        
    Returns:
        Updated state with new message
    """
    state_dict: Dict[str, Any] = {
        **state,
        "messages": [*state["messages"], message]
    }
    return cast(State, state_dict)


def create_tool_registry(tools: Dict[str, Callable]) -> Dict[str, Tool]:
    """Create a registry of tools from functions.
    
    Args:
        tools: Dictionary mapping tool names to functions
        
    Returns:
        Dictionary mapping tool names to Tool instances
    """
    return {
        name: Tool(
            name=name,
            description=func.__doc__ or "No description available",
            parameters={},
            function=func
        )
        for name, func in tools.items()
    }


def create_initial_state(task: Task) -> State:
    """Create initial state for task execution.
    
    Args:
        task: Task to execute
        
    Returns:
        Initial state dictionary
    """
    state_dict: Dict[str, Any] = {
        "messages": [
            SystemMessage(content="Starting task execution"),
            HumanMessage(content=task.description)
        ],
        "task": task,
        "metrics": {}
    }
    return cast(State, state_dict)


def create_chain(
    llm: Optional[BaseLanguageModel] = None,
    prompt: Optional[ChatPromptTemplate] = None,
    output_mode: str = '',
    output_parser: Any = None,
    schema_class: Any = None,
    **kwargs: Any
) -> Any:
    """Create a chain with configurable output and error handling.
    
    Args:
        llm: Language model to use
        prompt: Chat prompt template
        output_mode: Output mode for structured responses
        output_parser: Custom output parser
        schema_class: Pydantic model class for validation
        **kwargs: Additional arguments
        
    Returns:
        Chain configured with the specified components
    """
    if llm is None:
        raise ValueError("LLM must be provided")

    # Create base chain with prompt if provided
    chain = prompt | llm if prompt else llm

    # Handle output parsing
    if output_mode and output_mode in OUTPUT_MODEL_MAP:
        # Use provided schema class or get from map
        model_class = schema_class or OUTPUT_MODEL_MAP[output_mode]
        parser = PydanticOutputParser(pydantic_object=model_class)
        chain = chain | parser
    elif output_parser:
        chain = chain | output_parser

    return chain


def calculate_final_metrics(state: State) -> Dict[str, Any]:
    """Calculate final metrics from execution state.
    
    Args:
        state: Current execution state
        
    Returns:
        Dictionary of metrics
    """
    metrics = state.get("metrics", {})
    
    if "execution_result" in state:
        result = state["execution_result"]
        metrics.update({
            "success": result.success,
            "operation_count": len(result.operations),
            "failed_operations": sum(1 for op in result.operations if not op.status == "completed"),
            "total_duration": sum(
                (datetime.fromisoformat(op.end_time) - 
                 datetime.fromisoformat(op.start_time)).total_seconds()
                for op in result.operations
                if op.start_time and op.end_time
            )
        })
        
    return metrics


def generate_summary(metrics: Dict[str, Any]) -> str:
    """Generate a summary string from execution metrics.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Summary string
    """
    success = metrics.get("success", False)
    op_count = metrics.get("operation_count", 0)
    failed = metrics.get("failed_operations", 0)
    duration = metrics.get("total_duration", 0)
    
    status = "succeeded" if success else "failed"
    return (
        f"Task {status} after {op_count} operations "
        f"({failed} failed) in {duration:.2f} seconds"
    )


def display_progress(message: str, status: str = "info") -> None:
    """Display progress message with logging.
    
    Args:
        message: Progress message to display
        status: Status level (info, success, error)
    """
    style = {
        "info": "cyan",
        "success": "green",
        "error": "red"
    }.get(status, "white")
    
    logger.info(f"[{style}]{message}")


def display_result(result: ExecutionResult) -> None:
    """Display execution result.
    
    Args:
        result: Execution result to display
    """
    table = Table(title="Operation Results")
    table.add_column("Operation", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Duration", style="magenta")
    table.add_column("Error", style="red")

    for op in result.operations:
        status = "✓" if op.status == "completed" else "✗"
        duration = "N/A"
        if op.start_time and op.end_time:
            try:
                start = datetime.fromisoformat(op.start_time)
                end = datetime.fromisoformat(op.end_time)
                duration = f"{(end - start).total_seconds():.2f}s"
            except ValueError:
                pass
        error = op.error or ""
        table.add_row(op.description, status, duration, error)

    console.print(table)


def display_operation_results(operations: List[OperationResult]) -> None:
    """Display operation results in a table.
    
    Args:
        operations: List of operation results to display
    """
    table = Table(title="Operation Results")
    table.add_column("Operation", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Duration", style="blue")
    table.add_column("Result", style="white")
    
    for op in operations:
        status = "✓" if op.success else "✗"
        duration = "N/A"
        if op.start_time and op.end_time:
            try:
                start = datetime.fromisoformat(op.start_time)
                end = datetime.fromisoformat(op.end_time)
                duration = f"{(end - start).total_seconds():.2f}s"
            except ValueError:
                pass
        result = str(op.result) if op.result else ""
        table.add_row(op.operation_id, status, duration, result)

    console.print(table)


def display_execution_plan(plan: ExecutionPlan) -> None:
    """Display execution plan in a rich panel.
    
    Args:
        plan: Execution plan to display
    """
    table = Table(title=f"Execution Plan: {plan.description}")
    table.add_column("Operation", style="cyan")
    table.add_column("Tool", style="magenta")
    table.add_column("Description", style="white")
    table.add_column("Dependencies", style="blue")
    
    for op in plan.operations:
        deps = ", ".join(op.dependencies) if op.dependencies else "-"
        table.add_row(
            op.id,
            op.tool_name,
            op.description,
            deps
        )
        
    console.print(Panel(table, title="Execution Plan", border_style="green"))


def update_state_with_result(
    state: State,
    execution_result: ExecutionResult,
    messages: Optional[List[MessageLikeRepresentation]] = None
) -> State:
    """Update state with execution result."""
    state_dict: Dict[str, Any] = {
        "messages": messages if messages is not None else state["messages"],
        "task": state["task"],
        "metrics": {
            "success": execution_result.success,
            "operation_count": len(execution_result.operations)
        },
        "execution_plan": state.get("execution_plan"),
        "execution_result": execution_result
    }
    return cast(State, state_dict)


def get_operation_result(operation: TaskOperation) -> Dict[str, Any]:
    """Get operation result as a dictionary.
    
    Args:
        operation: Operation to convert to dictionary
        
    Returns:
        Dictionary containing operation details
    """
    return {
        "id": operation.id,
        "tool_name": operation.tool_name,
        "description": operation.description,
        "status": operation.status,
        "result": operation.result,
        "error": operation.error,
        "start_time": operation.start_time,
        "end_time": operation.end_time
    }
