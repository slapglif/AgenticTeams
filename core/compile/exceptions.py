"""
Custom exceptions for the LangGraph Compiler.
"""

from typing import Optional, Dict, Any

class CompilerError(Exception):
    """Base class for compiler exceptions."""
    pass

class PlanningError(CompilerError):
    """Raised when task planning fails."""
    pass

class ExecutionError(CompilerError):
    """Raised when task execution fails."""
    pass

class ValidationError(CompilerError):
    """Raised when input validation fails."""
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"Validation error for '{field}': {message}")

class ToolError(CompilerError):
    """Raised when a tool execution fails."""
    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"Tool '{tool_name}' failed: {message}")

class DependencyError(CompilerError):
    """Raised when a dependency cannot be satisfied."""
    def __init__(self, operation_id: str, dependency_id: str):
        self.operation_id = operation_id
        self.dependency_id = dependency_id
        super().__init__(f"Operation '{operation_id}' has unsatisfied dependency '{dependency_id}'")

class TimeoutError(CompilerError):
    """Raised when an operation times out."""
    def __init__(self, operation_id: str, timeout: float):
        self.operation_id = operation_id
        self.timeout = timeout
        super().__init__(f"Operation '{operation_id}' timed out after {timeout} seconds")

class ResourceError(CompilerError):
    """Raised when required resources are unavailable."""
    def __init__(self, resource_type: str, message: str):
        self.resource_type = resource_type
        self.message = message
        super().__init__(f"Resource '{resource_type}' error: {message}")

class GraphError(Exception):
    """Error raised during graph operations."""
    
    def __init__(self, operation: str, message: str, state: Optional[Dict[str, Any]] = None) -> None:
        """Initialize GraphError.
        
        Args:
            operation: Operation that failed
            message: Error message
            state: Optional state when error occurred
        """
        self.operation = operation
        self.message = message
        self.state = state
        super().__init__(f"{operation}: {message}")

    def with_state(self, state: Dict[str, Any]) -> 'GraphError':
        """Add state information to error."""
        self.state = state
        return self

class MetricsError(CompilerError):
    """Raised when there is an error calculating metrics."""
    def __init__(self, metric_name: str, message: str):
        self.metric_name = metric_name
        self.message = message
        super().__init__(f"Error calculating metric '{metric_name}': {message}")

class StateError(CompilerError):
    """Raised when there is an error with the graph state."""
    def __init__(self, state_key: str, message: str):
        self.state_key = state_key
        self.message = message
        super().__init__(f"State error for '{state_key}': {message}")

class SerializationError(CompilerError):
    """Raised when serialization or deserialization fails."""
    def __init__(self, data_type: str, message: str):
        self.data_type = data_type
        self.message = message
        super().__init__(f"Serialization error for '{data_type}': {message}")

class ConfigurationError(CompilerError):
    """Raised when there is an error in the configuration."""
    def __init__(self, config_key: str, message: str):
        self.config_key = config_key
        self.message = message
        super().__init__(f"Configuration error for '{config_key}': {message}")

def handle_compiler_error(error: Exception) -> Dict[str, Any]:
    """Handle compiler errors and return a standardized error response.
    
    Args:
        error: The exception to handle
        
    Returns:
        A dictionary containing error details
    """
    if isinstance(error, CompilerError):
        error_type = error.__class__.__name__
        error_details: Dict[str, Any] = {
            "type": error_type,
            "message": str(error)
        }
        
        # Add specific error details based on error type
        if isinstance(error, ToolError):
            error_details["tool_name"] = error.tool_name
        elif isinstance(error, DependencyError):
            error_details.update({
                "operation_id": error.operation_id,
                "dependency_id": error.dependency_id
            })
        elif isinstance(error, TimeoutError):
            error_details.update({
                "operation_id": error.operation_id,
                "timeout": error.timeout
            })
        elif isinstance(error, ResourceError):
            error_details["resource_type"] = error.resource_type
        elif isinstance(error, GraphError):
            error_details["state"] = error.state
        elif isinstance(error, MetricsError):
            error_details["metric_name"] = error.metric_name
        elif isinstance(error, StateError):
            error_details["state_key"] = error.state_key
        elif isinstance(error, SerializationError):
            error_details["data_type"] = error.data_type
        elif isinstance(error, ConfigurationError):
            error_details["config_key"] = error.config_key
            
        return error_details
    else:
        # Handle unexpected errors
        return {
            "type": "UnexpectedError",
            "message": str(error),
            "error_class": error.__class__.__name__
        } 