"""
Graph execution utilities for the LangGraph Compiler.
"""
from typing import Dict, List, Any, Optional, Union, Set, cast
from datetime import datetime, UTC
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_core.runnables import RunnableConfig

from ..models import (
    TaskOperation,
    OperationResult,
    ExecutionResult,
    State
)
from ..exceptions import (
    ExecutionError,
    TimeoutError,
    DependencyError
)
from ..metrics import calculate_metrics

async def execute_operation(
    operation: TaskOperation,
    config: Optional[RunnableConfig] = None,
    timeout: float = 1024.0
) -> OperationResult:
    """Execute a single operation.
    
    Args:
        operation: Operation to execute
        config: Optional configuration
        timeout: Timeout in seconds
        
    Returns:
        Operation result
        
    Raises:
        TimeoutError: If operation times out
        ExecutionError: If operation fails
    """
    try:
        start_time = datetime.now(UTC)
        
        # Create task for operation
        task = asyncio.create_task(
            _execute_operation_with_timeout(operation, config, timeout)
        )
        
        # Wait for completion or timeout
        try:
            result = await task
        except asyncio.TimeoutError:
            task.cancel()
            raise TimeoutError(operation.id, timeout)
            
        # Calculate metrics
        end_time = datetime.now(UTC)
        execution_time = (end_time - start_time).total_seconds()
        
        return OperationResult(
            operation_id=operation.id,
            success=True,
            result=result,
            metrics={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "execution_time": execution_time
            }
        )
        
    except TimeoutError:
        raise
    except Exception as e:
        return OperationResult(
            operation_id=operation.id,
            success=False,
            result={},
            error=str(e),
            metrics={
                "start_time": start_time.isoformat(),
                "end_time": datetime.now(UTC).isoformat(),
                "execution_time": (datetime.now(UTC) - start_time).total_seconds()
            }
        )

async def _execute_operation_with_timeout(
    operation: TaskOperation,
    config: Optional[RunnableConfig],
    timeout: float
) -> Dict[str, Any]:
    """Execute an operation with timeout.
    
    Args:
        operation: Operation to execute
        config: Optional configuration
        timeout: Timeout in seconds
        
    Returns:
        Operation result
        
    Raises:
        TimeoutError: If operation times out
    """
    try:
        if not config:
            raise ExecutionError("Configuration required")
            
        # Get tool function
        tools = config.get("tools", {})
        if not tools:
            raise ExecutionError("No tools configured")
            
        tool_func = tools.get(operation.tool.name)
        if not tool_func:
            raise ExecutionError(f"Tool '{operation.tool.name}' not found")
            
        # Execute with timeout
        async with asyncio.timeout(timeout):
            result = await tool_func(**operation.inputs)
            return result
            
    except asyncio.TimeoutError:
        raise TimeoutError(operation.id, timeout)
    except Exception as e:
        raise ExecutionError(f"Operation execution failed: {str(e)}")

async def execute_task_graph(
    execution_plan: Dict[str, Any],
    config: Optional[RunnableConfig] = None,
    max_workers: int = 5
) -> ExecutionResult:
    """Execute a complete task graph.
    
    Args:
        execution_plan: The execution plan
        config: Optional configuration
        max_workers: Maximum number of parallel workers
        
    Returns:
        Execution result
    """
    try:
        operations: List[TaskOperation] = execution_plan.get("operations", [])
        if not operations:
            raise ExecutionError("No operations in execution plan")
            
        # Create thread pool
        executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Track operation results
        results: List[OperationResult] = []
        completed_ops: Set[str] = set()
        
        # Execute operations in dependency order
        while len(completed_ops) < len(operations):
            # Find ready operations
            ready_ops = [
                op for op in operations
                if op.id not in completed_ops and
                all(dep in completed_ops for dep in op.dependencies)
            ]
            
            if not ready_ops and len(completed_ops) < len(operations):
                raise DependencyError("unknown", "Circular dependency detected")
                
            # Execute ready operations in parallel
            tasks = [
                execute_operation(op, config)
                for op in ready_ops
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for op, result in zip(ready_ops, batch_results):
                if isinstance(result, Exception):
                    results.append(OperationResult(
                        operation_id=op.id,
                        success=False,
                        result={},
                        error=str(result)
                    ))
                else:
                    results.append(cast(OperationResult, result))
                completed_ops.add(op.id)
                
        # Calculate final metrics
        metrics = calculate_metrics(ExecutionResult(
            task_id=execution_plan.get("task_id", "unknown"),
            operation_results=results,
            success=all(r.success for r in results),
            metrics={}
        ))
        
        # Create execution result
        return ExecutionResult(
            task_id=execution_plan.get("task_id", "unknown"),
            operation_results=results,
            success=all(r.success for r in results),
            metrics=metrics
        )
        
    except Exception as e:
        raise ExecutionError(f"Task execution failed: {str(e)}")

def generate_execution_summary(results: List[OperationResult]) -> str:
    """Generate a summary of execution results.
    
    Args:
        results: List of operation results
        
    Returns:
        Summary string
    """
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful
    
    summary = f"Executed {total} operations: {successful} succeeded, {failed} failed"
    
    if failed > 0:
        failed_ops = [r for r in results if not r.success]
        summary += "\nFailed operations:"
        for op in failed_ops:
            summary += f"\n- {op.operation_id}: {op.error}"
            
    return summary

async def execute_with_retry(
    operation: TaskOperation,
    config: Optional[RunnableConfig] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> OperationResult:
    """Execute an operation with retries.
    
    Args:
        operation: Operation to execute
        config: Optional configuration
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Operation result
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            result = await execute_operation(operation, config)
            if result.success:
                return result
                
            last_error = result.error
            
        except Exception as e:
            last_error = str(e)
            
        # Wait before retry
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay * (attempt + 1))
            
    return OperationResult(
        operation_id=operation.id,
        success=False,
        result={},
        error=f"Operation failed after {max_retries} attempts. Last error: {last_error}"
    )

async def execute_with_fallback(
    operation: TaskOperation,
    fallback_operation: TaskOperation,
    config: Optional[RunnableConfig] = None
) -> OperationResult:
    """Execute an operation with fallback.
    
    Args:
        operation: Primary operation to execute
        fallback_operation: Fallback operation if primary fails
        config: Optional configuration
        
    Returns:
        Operation result
    """
    # Try primary operation
    result = await execute_operation(operation, config)
    if result.success:
        return result
        
    # If primary fails, try fallback
    fallback_result = await execute_operation(fallback_operation, config)
    if fallback_result.success:
        return fallback_result
        
    # If both fail, return primary error
    return result

async def execute_batch(
    operations: List[TaskOperation],
    config: Optional[RunnableConfig] = None,
    max_concurrent: int = 5
) -> List[OperationResult]:
    """Execute a batch of operations with concurrency control.
    
    Args:
        operations: List of operations to execute
        config: Optional configuration
        max_concurrent: Maximum number of concurrent operations
        
    Returns:
        List of operation results
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_with_semaphore(op: TaskOperation) -> OperationResult:
        async with semaphore:
            return await execute_operation(op, config)
            
    tasks = [execute_with_semaphore(op) for op in operations]
    return await asyncio.gather(*tasks)

def validate_dependencies(operations: List[TaskOperation]) -> None:
    """Validate operation dependencies.
    
    Args:
        operations: List of operations to validate
        
    Raises:
        DependencyError: If validation fails
    """
    # Get all operation IDs
    op_ids = {op.id for op in operations}
    
    # Check each operation's dependencies
    for op in operations:
        for dep in op.dependencies:
            if dep not in op_ids:
                raise DependencyError(op.id, dep)
                
    # Check for cycles
    visited = set()
    path = set()
    
    def check_cycle(op_id: str) -> None:
        if op_id in path:
            cycle = list(path)[list(path).index(op_id):] + [op_id]
            raise DependencyError(op_id, f"Cycle detected: {' -> '.join(cycle)}")
            
        if op_id in visited:
            return
            
        visited.add(op_id)
        path.add(op_id)
        
        op = next(op for op in operations if op.id == op_id)
        for dep in op.dependencies:
            check_cycle(dep)
            
        path.remove(op_id)
        
    for op in operations:
        check_cycle(op.id) 