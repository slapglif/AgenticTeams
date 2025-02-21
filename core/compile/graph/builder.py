"""
Graph building utilities for the LangGraph Compiler.
"""
from typing import Dict, List, Any, Union, Optional, Set, TypeVar, cast, Callable
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage

from ..models import State, ExecutionResult, TaskOperation, ExecutionPlan
from ..exceptions import GraphError

T = TypeVar('T')

def execute_operations(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute operations in the execution plan."""
    try:
        plan = state.get("execution_plan")
        if not plan:
            raise ValueError("No execution plan found")
        
        # TODO: Implement actual execution logic
        state["execution_result"] = ExecutionResult(
            task_id=plan.task_id if isinstance(plan, ExecutionPlan) else "unknown",
            operation_results=[],
            success=True,
            metrics={},
            description="Execution completed"
        )
        return state
    except Exception as e:
        raise GraphError("execute_operations", str(e), state)

def join_results(state: Dict[str, Any]) -> Dict[str, Any]:
    """Join execution results."""
    try:
        result = state.get("execution_result")
        if not result:
            raise ValueError("No execution result found")
        
        # TODO: Implement actual join logic
        state["join_output"] = {
            "success": result.success,
            "message": "Results joined successfully"
        }
        return state
    except Exception as e:
        raise GraphError("join_results", str(e), state)

class GraphBuilder:
    """Builder for constructing task execution graphs."""
    
    def __init__(self) -> None:
        self.graph = StateGraph(Dict[str, Any])
        self.nodes: Dict[str, Callable] = {}
        self.edges: Dict[str, Dict[str, Any]] = {}
        self._entry_point: Optional[str] = None
    
    def add_node(self, name: str, func: Callable) -> None:
        """Add a node to the graph."""
        self.nodes[name] = func
        self.edges[name] = {}
    
    def add_edge(self, source: str, target: str) -> None:
        """Add an edge between nodes."""
        if source not in self.nodes or target not in self.nodes:
            raise GraphError("add_edge", f"Invalid edge: {source} -> {target}")
        self.edges[source][target] = {}
    
    def set_entry_point(self, node: str) -> None:
        """Set the entry point of the graph."""
        if node not in self.nodes:
            raise GraphError("set_entry_point", f"Invalid entry point: {node}")
        self._entry_point = node
    
    def build(self) -> StateGraph:
        """Build and return the graph."""
        if not self._entry_point:
            raise GraphError("build", "No entry point set")
        
        graph = StateGraph(Dict[str, Any])
        for name, func in self.nodes.items():
            graph.add_node(name, func)
        
        for source, targets in self.edges.items():
            for target in targets:
                graph.add_edge(source, target)
        
        graph.set_entry_point(self._entry_point)
        return graph

def build_graph() -> StateGraph:
    """Build the task execution graph."""
    try:
        builder = GraphBuilder()
        
        # Add nodes
        builder.add_node("plan", plan_and_schedule)
        builder.add_node("execute", execute_operations)
        builder.add_node("join", join_results)
        
        # Add edges
        builder.add_edge("plan", "execute")
        builder.add_edge("execute", "join")
        
        # Set entry point
        builder.set_entry_point("plan")
        
        return builder.build()
    except Exception as e:
        raise GraphError("build_graph", str(e))

def should_continue(state: Dict[str, Any]) -> str:
    """Determine if execution should continue or end."""
    try:
        messages = state.get("messages", [])
        if not messages:
            return "end"
            
        last_message = messages[-1]
        content = last_message.get("content", "") if isinstance(last_message, dict) else ""
        
        if "final_response" in str(content):
            return "end"
            
        if "replan" in str(content):
            return "continue"
            
        return "end"
    except Exception as e:
        raise GraphError("should_continue", str(e), state)

def plan_and_schedule(state: Dict[str, Any]) -> Dict[str, Any]:
    """Plan and schedule task operations."""
    try:
        task = state.get("task")
        if not task:
            raise GraphError("plan_and_schedule", "No task found", state)
            
        # Create execution plan
        plan = create_execution_plan(task)
        
        # Schedule operations
        scheduled_ops = schedule_operations(plan.operations if isinstance(plan, ExecutionPlan) else [])
        
        # Add plan to state
        messages = state.get("messages", [])
        messages.extend([
            SystemMessage(content=f"Execution plan created with {len(scheduled_ops)} operations"),
            AIMessage(content=str({
                "plan": plan.dict() if hasattr(plan, "dict") else plan,
                "scheduled_operations": scheduled_ops
            }))
        ])
        
        state["messages"] = messages
        return state
    except GraphError:
        raise
    except Exception as e:
        raise GraphError("plan_and_schedule", str(e), state)

def create_execution_plan(task: Any) -> ExecutionPlan:
    """Create an execution plan for a task."""
    try:
        return ExecutionPlan(
            task_id="dummy_task",
            description="Dummy plan",
            operations=[]
        )
    except Exception as e:
        raise GraphError("create_execution_plan", str(e))

def schedule_operations(operations: List[TaskOperation]) -> List[Dict[str, Any]]:
    """Schedule operations based on dependencies."""
    try:
        scheduled: List[Dict[str, Any]] = []
        remaining = operations.copy()
        current_time = 0
        
        while remaining:
            ready = [
                op for op in remaining
                if all(dep in [s["operation"].id for s in scheduled] for dep in op.dependencies)
            ]
            
            if not ready:
                raise GraphError("schedule_operations", "Dependency cycle detected")
                
            for op in ready:
                scheduled.append({
                    "operation": op,
                    "start_time": current_time,
                    "estimated_duration": estimate_duration(op)
                })
                remaining.remove(op)
                
            current_time += 1
            
        return scheduled
    except GraphError:
        raise
    except Exception as e:
        raise GraphError("schedule_operations", str(e))

def estimate_duration(operation: TaskOperation) -> float:
    """Estimate the duration of an operation."""
    return 1.0

def validate_graph(graph: StateGraph) -> None:
    """Validate a task execution graph."""
    try:
        builder = cast(GraphBuilder, graph)
        required_nodes = {"plan", "execute", "join"}
        graph_nodes = set(builder.nodes.keys())
        missing_nodes = required_nodes - graph_nodes
        
        if missing_nodes:
            raise GraphError("validate_graph", f"Missing nodes: {missing_nodes}")
            
        if not builder._entry_point:
            raise GraphError("validate_graph", "No entry point")
            
        for node in graph_nodes:
            if not builder.edges.get(node):
                raise GraphError("validate_graph", f"No edges for node: {node}")
    except GraphError:
        raise
    except Exception as e:
        raise GraphError("validate_graph", str(e))

def get_node_info(graph: StateGraph, node_name: str) -> Dict[str, Any]:
    """Get information about a graph node."""
    try:
        builder = cast(GraphBuilder, graph)
        if node_name not in builder.nodes:
            raise GraphError("get_node_info", f"Node not found: {node_name}")
            
        return {
            "name": node_name,
            "function": builder.nodes[node_name].__name__,
            "incoming_edges": [
                source for source, targets in builder.edges.items()
                if node_name in targets
            ],
            "outgoing_edges": list(builder.edges[node_name].keys()),
            "is_entry": node_name == builder._entry_point
        }
    except GraphError:
        raise
    except Exception as e:
        raise GraphError("get_node_info", str(e))

def analyze_graph(graph: StateGraph) -> Dict[str, Any]:
    """Analyze a task execution graph."""
    try:
        builder = cast(GraphBuilder, graph)
        return {
            "node_count": len(builder.nodes),
            "edge_count": sum(len(targets) for targets in builder.edges.values()),
            "cycles": find_cycles(graph),
            "max_path_length": calculate_max_path_length(graph),
            "entry_point": builder._entry_point,
            "terminal_nodes": [
                node for node in builder.nodes
                if not builder.edges[node]
            ]
        }
    except GraphError:
        raise
    except Exception as e:
        raise GraphError("analyze_graph", str(e))

def find_cycles(graph: StateGraph) -> List[List[str]]:
    """Find cycles in the graph."""
    try:
        builder = cast(GraphBuilder, graph)
        cycles: List[List[str]] = []
        visited: Set[str] = set()
        path: List[str] = []
        
        def dfs(node: str) -> None:
            if node in path:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:])
                return
                
            if node in visited:
                return
                
            visited.add(node)
            path.append(node)
            
            for next_node in builder.edges[node]:
                dfs(next_node)
                
            path.pop()
            
        for node in builder.nodes:
            if node not in visited:
                dfs(node)
                
        return cycles
    except GraphError:
        raise
    except Exception as e:
        raise GraphError("find_cycles", str(e))

def calculate_max_path_length(graph: StateGraph) -> int:
    """Calculate the maximum path length in the graph."""
    try:
        builder = cast(GraphBuilder, graph)
        max_length = 0
        visited: Set[str] = set()
        
        def dfs(node: str, length: int) -> None:
            nonlocal max_length
            max_length = max(max_length, length)
            
            if node in visited:
                return
                
            visited.add(node)
            for next_node in builder.edges[node]:
                dfs(next_node, length + 1)
                
            visited.remove(node)
            
        if builder._entry_point:
            dfs(builder._entry_point, 0)
            
        return max_length
    except GraphError:
        raise
    except Exception as e:
        raise GraphError("calculate_max_path_length", str(e)) 