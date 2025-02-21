"""
Metrics calculation utilities for the LangGraph Compiler.
"""
from typing import Dict, List, Any, Union, Optional, cast
from datetime import datetime, UTC
from langchain_core.messages import BaseMessage, SystemMessage
import numpy as np
from .models import ExecutionResult, OperationResult, TaskOperation
from .exceptions import MetricsError

def calculate_complexity_score(task_description: str) -> float:
    """Calculate a complexity score for a task based on its description.
    
    Args:
        task_description: Description of the task
        
    Returns:
        Complexity score between 0 and 1
    """
    # Factors that indicate complexity
    factors = {
        "dependencies": ["depends", "requires", "after", "before", "then"],
        "conditions": ["if", "when", "unless", "while", "until"],
        "operations": ["analyze", "process", "compute", "calculate", "evaluate"],
        "data_types": ["graph", "network", "matrix", "dataset", "series"]
    }
    
    score = 0.0
    description = task_description.lower()
    
    # Calculate score based on presence of complexity factors
    for category, terms in factors.items():
        category_score = sum(1 for term in terms if term in description)
        score += category_score * 0.2  # Weight each category
    
    # Normalize score to 0-1 range
    return min(1.0, score)

def calculate_completion_score(messages: List[BaseMessage]) -> float:
    """Calculate a completion score based on message history.
    
    Args:
        messages: List of messages in the conversation
        
    Returns:
        Completion score between 0 and 1
    """
    if not messages:
        return 0.0
        
    # Analyze message patterns
    completion_indicators = {
        "success": ["completed", "finished", "done", "succeeded"],
        "failure": ["failed", "error", "exception", "invalid"],
        "progress": ["progress", "step", "phase", "stage"]
    }
    
    scores = []
    for msg in messages:
        content = str(msg.content).lower()
        
        # Calculate score for each message
        msg_score = 0.0
        for category, terms in completion_indicators.items():
            if category == "failure":
                # Reduce score for failure indicators
                msg_score -= sum(1 for term in terms if term in content) * 0.2
            else:
                msg_score += sum(1 for term in terms if term in content) * 0.2
                
        scores.append(msg_score)
    
    # Weight recent messages more heavily
    weights = np.linspace(0.5, 1.0, len(scores))
    weighted_score = float(np.average(scores, weights=weights))
    
    # Normalize to 0-1 range
    return max(0.0, min(1.0, weighted_score))

def calculate_quality_score(messages: List[BaseMessage]) -> float:
    """Calculate a quality score based on message history.
    
    Args:
        messages: List of messages in the conversation
        
    Returns:
        Quality score between 0 and 1
    """
    if not messages:
        return 0.0
        
    # Quality indicators in messages
    quality_indicators = {
        "high": ["accurate", "precise", "thorough", "comprehensive", "detailed"],
        "medium": ["good", "reasonable", "adequate", "sufficient", "acceptable"],
        "low": ["poor", "incomplete", "insufficient", "inadequate", "limited"]
    }
    
    scores = []
    for msg in messages:
        content = str(msg.content).lower()
        
        # Calculate score for each message
        msg_score = 0.0
        for level, terms in quality_indicators.items():
            matches = sum(1 for term in terms if term in content)
            if level == "high":
                msg_score += matches * 0.3
            elif level == "medium":
                msg_score += matches * 0.2
            else:  # low
                msg_score -= matches * 0.2
                
        scores.append(msg_score)
    
    # Weight recent messages more heavily
    weights = np.linspace(0.5, 1.0, len(scores))
    weighted_score = float(np.average(scores, weights=weights))
    
    # Normalize to 0-1 range
    return max(0.0, min(1.0, weighted_score))

def calculate_execution_metrics(results: List[OperationResult]) -> Dict[str, Any]:
    """Calculate metrics for a set of operation results.
    
    Args:
        results: List of operation results
        
    Returns:
        Dictionary of calculated metrics
    """
    try:
        total_ops = len(results)
        successful_ops = sum(1 for r in results if r.success)
        failed_ops = total_ops - successful_ops
        
        # Calculate timing metrics
        execution_times = [
            float(r.metrics.get("execution_time", 0.0)) 
            for r in results 
            if r.metrics is not None
        ]
        
        avg_execution_time = float(np.mean(execution_times)) if execution_times else 0.0
        max_execution_time = max(execution_times) if execution_times else 0.0
        
        return {
            "total_operations": total_ops,
            "successful_operations": successful_ops,
            "failed_operations": failed_ops,
            "success_rate": successful_ops / total_ops if total_ops > 0 else 0.0,
            "average_execution_time": avg_execution_time,
            "max_execution_time": max_execution_time,
            "total_execution_time": sum(execution_times)
        }
        
    except Exception as e:
        raise MetricsError("execution_metrics", str(e))

def calculate_final_metrics(final_state: Union[ExecutionResult, Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate final metrics for a completed task.
    
    Args:
        final_state: Final execution result or state
        
    Returns:
        Dictionary of calculated metrics
    """
    try:
        if isinstance(final_state, ExecutionResult):
            operation_results = [
                OperationResult(**result) if isinstance(result, dict) else result
                for result in final_state.operation_results
            ]
            success = final_state.success
            error = final_state.error
        else:
            operation_results = [
                OperationResult(**result) if isinstance(result, dict) else result
                for result in final_state.get("operation_results", [])
            ]
            success = bool(final_state.get("success", False))
            error = final_state.get("error")
            
        # Calculate operation metrics
        op_metrics = calculate_execution_metrics(operation_results)
        
        # Calculate timing metrics
        start_times = [
            float(op.metrics.get("start_time", float("inf")))
            for op in operation_results
            if op.metrics is not None
        ]
        end_times = [
            float(op.metrics.get("end_time", 0))
            for op in operation_results
            if op.metrics is not None
        ]
        
        start_time = min(start_times) if start_times else 0
        end_time = max(end_times) if end_times else 0
        total_time = end_time - start_time if start_time != float("inf") else 0
        
        return {
            "success": success,
            "has_error": error is not None,
            "total_time": total_time,
            "operations": op_metrics,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        raise MetricsError("final_metrics", str(e))

def calculate_metrics(execution_result: ExecutionResult) -> Dict[str, Any]:
    """Calculate comprehensive metrics for an execution result.
    
    Args:
        execution_result: The execution result to analyze
        
    Returns:
        Dictionary of calculated metrics
    """
    try:
        # Get basic metrics
        basic_metrics = calculate_final_metrics(execution_result)
        
        # Calculate additional metrics
        operation_results = [
            OperationResult(**result) if isinstance(result, dict) else result
            for result in execution_result.operation_results
        ]
        
        # Get dependencies from operation results
        dependencies = [
            result.get("dependencies", []) if isinstance(result, dict) else []
            for result in execution_result.operation_results
        ]
        
        dependency_metrics = {
            "avg_dependencies": float(np.mean([len(deps) for deps in dependencies])) if dependencies else 0.0,
            "max_dependencies": max(len(deps) for deps in dependencies) if dependencies else 0,
            "total_dependencies": sum(len(deps) for deps in dependencies)
        }
        
        # Calculate parallel execution metrics
        parallel_ops = len([deps for deps in dependencies if not deps])
        sequential_ops = len(dependencies) - parallel_ops
        
        parallelism_metrics = {
            "parallel_operations": parallel_ops,
            "sequential_operations": sequential_ops,
            "parallelism_ratio": parallel_ops / len(dependencies) if dependencies else 0
        }
        
        # Combine all metrics
        return {
            **basic_metrics,
            "dependencies": dependency_metrics,
            "parallelism": parallelism_metrics,
            "complexity": calculate_complexity_score(execution_result.description or "")
        }
        
    except Exception as e:
        raise MetricsError("comprehensive_metrics", str(e)) 