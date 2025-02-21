"""
Graph scheduling utilities for the LangGraph Compiler.
"""
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..models import TaskOperation
from ..exceptions import DependencyError

@dataclass
class ScheduleSlot:
    """Represents a time slot in the schedule."""
    start_time: datetime
    end_time: datetime
    operation_id: str
    worker_id: int

@dataclass
class Schedule:
    """Represents a complete schedule of operations."""
    slots: List[ScheduleSlot]
    start_time: datetime
    end_time: datetime
    worker_count: int
    
    def get_operation_slot(self, operation_id: str) -> Optional[ScheduleSlot]:
        """Get the slot for a specific operation."""
        for slot in self.slots:
            if slot.operation_id == operation_id:
                return slot
        return None

def schedule_operations(
    operations: List[TaskOperation],
    start_time: datetime,
    max_workers: int = 5,
    default_duration: timedelta = timedelta(minutes=5)
) -> Schedule:
    """Schedule operations across available workers.
    
    Args:
        operations: List of operations to schedule
        start_time: Schedule start time
        max_workers: Maximum number of concurrent workers
        default_duration: Default operation duration
        
    Returns:
        Complete schedule
        
    Raises:
        DependencyError: If dependencies cannot be satisfied
    """
    # Validate dependencies
    validate_dependencies(operations)
    
    # Track scheduled operations and available workers
    scheduled: Set[str] = set()
    worker_end_times: List[datetime] = [start_time] * max_workers
    slots: List[ScheduleSlot] = []
    
    while len(scheduled) < len(operations):
        # Find ready operations
        ready_ops = [
            op for op in operations
            if op.id not in scheduled and
            all(dep in scheduled for dep in op.dependencies)
        ]
        
        if not ready_ops and len(scheduled) < len(operations):
            raise DependencyError("unknown", "Circular dependency detected")
            
        # Schedule each ready operation
        for op in ready_ops:
            # Find earliest available worker
            worker_id = min(range(max_workers), key=lambda i: worker_end_times[i])
            slot_start = worker_end_times[worker_id]
            
            # Ensure all dependencies complete before start
            for dep in op.dependencies:
                dep_slot = next(s for s in slots if s.operation_id == dep)
                slot_start = max(slot_start, dep_slot.end_time)
                
            # Create schedule slot
            slot_end = slot_start + default_duration
            slots.append(ScheduleSlot(
                start_time=slot_start,
                end_time=slot_end,
                operation_id=op.id,
                worker_id=worker_id
            ))
            
            # Update worker end time
            worker_end_times[worker_id] = slot_end
            scheduled.add(op.id)
            
    return Schedule(
        slots=slots,
        start_time=start_time,
        end_time=max(worker_end_times),
        worker_count=max_workers
    )

def estimate_duration(
    operation: TaskOperation,
    historical_data: Optional[Dict[str, List[float]]] = None
) -> timedelta:
    """Estimate duration of an operation.
    
    Args:
        operation: Operation to estimate
        historical_data: Optional historical execution times
        
    Returns:
        Estimated duration
    """
    if not historical_data or operation.tool.name not in historical_data:
        return timedelta(minutes=5)  # Default duration
        
    # Calculate average from historical data
    times = historical_data[operation.tool.name]
    avg_seconds = sum(times) / len(times)
    return timedelta(seconds=avg_seconds)

def optimize_schedule(
    schedule: Schedule,
    operations: List[TaskOperation]
) -> Schedule:
    """Optimize an existing schedule.
    
    Args:
        schedule: Current schedule
        operations: List of operations
        
    Returns:
        Optimized schedule
    """
    # Build dependency graph
    deps: Dict[str, Set[str]] = {op.id: set(op.dependencies) for op in operations}
    
    # Calculate earliest possible start times
    earliest_starts: Dict[str, datetime] = {}
    
    def calculate_earliest_start(op_id: str) -> datetime:
        if op_id in earliest_starts:
            return earliest_starts[op_id]
            
        if not deps[op_id]:
            earliest_starts[op_id] = schedule.start_time
            return schedule.start_time
            
        # Must start after all dependencies
        start = max(
            calculate_earliest_start(dep)
            for dep in deps[op_id]
        )
        earliest_starts[op_id] = start
        return start
        
    for op in operations:
        calculate_earliest_start(op.id)
        
    # Sort slots by earliest start time
    sorted_slots = sorted(
        schedule.slots,
        key=lambda s: earliest_starts[s.operation_id]
    )
    
    # Rebuild schedule with optimized slots
    worker_end_times = [schedule.start_time] * schedule.worker_count
    new_slots: List[ScheduleSlot] = []
    
    for slot in sorted_slots:
        # Find earliest available worker
        worker_id = min(
            range(schedule.worker_count),
            key=lambda i: worker_end_times[i]
        )
        
        # Calculate start time
        start = max(
            worker_end_times[worker_id],
            earliest_starts[slot.operation_id]
        )
        
        # Create new slot
        duration = slot.end_time - slot.start_time
        new_slots.append(ScheduleSlot(
            start_time=start,
            end_time=start + duration,
            operation_id=slot.operation_id,
            worker_id=worker_id
        ))
        
        # Update worker end time
        worker_end_times[worker_id] = start + duration
        
    return Schedule(
        slots=new_slots,
        start_time=schedule.start_time,
        end_time=max(worker_end_times),
        worker_count=schedule.worker_count
    )

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

def analyze_schedule(schedule: Schedule) -> Dict[str, Any]:
    """Analyze a schedule for metrics and insights.
    
    Args:
        schedule: Schedule to analyze
        
    Returns:
        Dictionary of metrics and insights
    """
    total_duration = (schedule.end_time - schedule.start_time).total_seconds()
    
    # Calculate worker utilization
    worker_times = [0.0] * schedule.worker_count
    for slot in schedule.slots:
        duration = (slot.end_time - slot.start_time).total_seconds()
        worker_times[slot.worker_id] += duration
        
    utilization = [t / total_duration for t in worker_times]
    avg_utilization = sum(utilization) / len(utilization)
    
    # Calculate critical path
    critical_path = find_critical_path(schedule)
    
    return {
        "total_duration": total_duration,
        "worker_utilization": utilization,
        "average_utilization": avg_utilization,
        "critical_path": critical_path,
        "operation_count": len(schedule.slots),
        "max_parallel": max(
            sum(1 for s in schedule.slots if s.start_time <= t <= s.end_time)
            for t in {s.start_time for s in schedule.slots}
        )
    }

def find_critical_path(schedule: Schedule) -> List[str]:
    """Find the critical path through the schedule.
    
    Args:
        schedule: Schedule to analyze
        
    Returns:
        List of operation IDs in critical path
    """
    # Build graph of overlapping operations
    overlaps: Dict[str, Set[str]] = {
        slot.operation_id: set()
        for slot in schedule.slots
    }
    
    for s1 in schedule.slots:
        for s2 in schedule.slots:
            if s1.operation_id != s2.operation_id:
                if (s1.start_time < s2.end_time and
                    s2.start_time < s1.end_time):
                    overlaps[s1.operation_id].add(s2.operation_id)
                    
    # Find longest path
    distances: Dict[str, float] = {}
    predecessors: Dict[str, Optional[str]] = {}
    
    def get_duration(op_id: str) -> float:
        slot = schedule.get_operation_slot(op_id)
        if not slot:
            return 0.0
        return (slot.end_time - slot.start_time).total_seconds()
        
    # Initialize distances
    for op_id in overlaps:
        distances[op_id] = get_duration(op_id)
        predecessors[op_id] = None
        
    # Find longest paths
    for op_id in overlaps:
        for dep_id in overlaps[op_id]:
            dist = distances[op_id] + get_duration(dep_id)
            if dist > distances[dep_id]:
                distances[dep_id] = dist
                predecessors[dep_id] = op_id
                
    # Build critical path
    end_op = max(distances.items(), key=lambda x: x[1])[0]
    path = []
    current = end_op
    
    while current:
        path.append(current)
        current = predecessors[current]
        
    return list(reversed(path)) 