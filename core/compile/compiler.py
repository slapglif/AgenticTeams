"""
LangGraph Compiler implementation based on Kim et al.

This module implements a compiler for LangGraph that includes three main components:
1. Planner: Creates execution plans for tasks
2. Task Fetching Unit: Executes operations in dependency order
3. Joiner: Analyzes execution results and determines next steps

The compiler uses LangGraph for state management and execution flow.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, cast, Sequence, TypedDict, Required, NotRequired, Type, TypeVar, Literal, TypeGuard, Tuple, Protocol
import json
import os

from langchain.schema import AIMessage, BaseMessage, SystemMessage
from langchain.schema.language_model import BaseLanguageModel
from langchain_core.messages import HumanMessage, MessageLikeRepresentation
from langchain_core.runnables import RunnableConfig
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt.chat_agent_executor import AgentState
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.engine.research_tools import ResearchTools
from core.memory.memory_manager import MemoryManager
from core.memory.langgraph_memory import KnowledgeTriple
from core.shared.settings import build_llm
from core.shared.logging_config import console, get_logger, log_operation
from core.compile.utils import (
    create_chain,
    display_operation_results,
    display_execution_plan,
    calculate_final_metrics,
    generate_summary
)
from .models import (
    State,
    FinalResponse,
    Replan,
    JoinOutputs,
    TaskOperation,
    ExecutionPlan,
    OperationResult,
    ExecutionResult,
    Tool,
    Task
)
from .prompts import (
    PLANNER_PROMPT,
    JOINER_PROMPT,
    ANALYSIS_PROMPT,
    REPLAN_PROMPT
)
from core.schemas.pydantic_schemas import (
    FinalPlan,
    SpecializedAgentResponse,
    SummarizationResponse,
    StepCompleteResponse,
    SupervisorResponse,
    JoinerResponse
)
from pydantic import ValidationError
from rich.console import Console

# Initialize logger
logger = get_logger(__name__)

# Table columns for operations
OPERATION_TABLE_COLUMNS = [
    ("ID", "cyan"),
    ("Description", "white"),
    ("Tool", "yellow"),
    ("Status", "green")
]

UTC = timezone.utc

# Type alias for operation results
OperationResultsList = List[TaskOperation]

# Type aliases
TaskOperationList = Sequence[TaskOperation]
MessageList = List[MessageLikeRepresentation]
MetricsDict = Dict[str, Any]

T = TypeVar('T')

def safe_get(lst: List[Dict[str, Any]], key: str, default: T = None) -> T:
    """Safely get a value from a list of dicts."""
    return next((item[key] for item in lst if key in item), default)

def safe_dict_get(d: Dict[str, Any], key: str, default: T = None) -> T:
    """Safely get a value from a dict."""
    return d.get(key, default)

@dataclass
class CompilerConfig:
    """Configuration for the LangGraph Compiler.
    
    Args:
        max_workers: Maximum number of concurrent workers
        max_retries: Maximum number of retries for failed operations
        operation_timeout: Timeout in seconds for each operation
        max_replan_attempts: Maximum number of replanning attempts
        chain_config: Configuration for LangChain chains
        display_progress: Whether to display progress during execution
    """
    max_workers: int = 4
    max_retries: int = 3
    operation_timeout: float = 60.0
    max_replan_attempts: int = 2
    chain_config: Optional[RunnableConfig] = None
    display_progress: bool = True


class LangGraphCompiler:
    """LangGraph Compiler for orchestrating agent interactions."""

    def __init__(
            self,
            llm: BaseLanguageModel,
            memory_manager: MemoryManager,
            research_tools: ResearchTools,
            tool_config: Optional[RunnableConfig] = None
    ):
        """Initialize the compiler.
        
        Args:
            llm: Language model for planning and analysis
            memory_manager: Memory manager for storing task data
            research_tools: Research tools for executing operations
            tool_config: Optional configuration for tool execution
        """
        self.llm = llm
        self.memory_manager = memory_manager
        self.research_tools = research_tools
        self.tool_config = tool_config or {}
        self.console = Console()
        self.logger = get_logger(__name__)

        # Initialize tool map with research tools
        self.tool_map = {
            "search": Tool(
                name="search",
                description="Search for information using Searx",
                parameters={},
                function=research_tools.search
            ),
            "analyze_data": Tool(
                name="analyze_data",
                description="Analyze data using LLM",
                parameters={},
                function=research_tools.analyze_data
            ),
            "analyze_graph": Tool(
                name="analyze_graph",
                description="Analyze graph structures",
                parameters={},
                function=research_tools.analyze_graph
            ),
            "analyze_citations": Tool(
                name="analyze_citations",
                description="Analyze citation patterns",
                parameters={},
                function=research_tools.analyze_citations
            ),
            "analyze_network": Tool(
                name="analyze_network",
                description="Analyze network relationships",
                parameters={},
                function=research_tools.analyze_network
            ),
            "generate_domain_search_query": Tool(
                name="generate_domain_search_query",
                description="Generate domain-specific search queries",
                parameters={},
                function=research_tools.generate_domain_search_query
            )
        }

        # Create graph components
        self.graph = self._build_graph()

    def _build_llm(self, **kwargs):
        """Build the LLM component."""
        from core.shared.settings import build_llm
        return build_llm()

    def _build_graph(self) -> Any:
        """Build the LangGraph computation graph."""
        graph = StateGraph(State)

        # Add nodes for the main components
        graph.add_node("plan", self._plan_task)  # Planner
        graph.add_node("validate_plan", self._validate_plan)  # Plan validation
        graph.add_node("execute", self._execute_task)  # Task Fetching Unit
        graph.add_node("analyze", self._analyze_results)  # Result analysis
        graph.add_node("join", self._join_results)  # Joiner

        # Add edges for main flow
        graph.add_edge("plan", "validate_plan")
        graph.add_edge("validate_plan", "execute")
        graph.add_edge("execute", "analyze") 
        graph.add_edge("analyze", "join")

        def should_continue(state: State) -> Union[Literal["plan"], Literal["end"]]:
            """Determine if we should continue planning or end."""
            messages = state["messages"]
            if not messages:
                return "end"

            last_message = messages[-1]
            if isinstance(last_message, AIMessage):
                content = last_message.content
                if isinstance(content, dict) and content.get("type") == "replan":
                    return "plan"
                return "end"
            return "end"

        # Add end node
        graph.add_node("end", lambda x: x)

        # Add conditional edges for replanning
        graph.add_conditional_edges(
            "join",
            should_continue,
            {
                "plan": "plan",
                "end": "end"
            }
        )

        # Set entry point
        graph.set_entry_point("plan")

        # Compile the graph into a runnable
        return graph.compile()

    async def compile_and_execute(
            self,
            task: Task,
            initial_state: Optional[State] = None
    ) -> FinalResponse:
        """Compile and execute a task."""
        try:
            # Create initial state if not provided
            if initial_state is None:
                initial_state = {
                    "messages": [],
                    "task": task,
                    "metrics": {}
                }
            state = cast(State, initial_state)

            # Add initial messages
            new_messages = [
                SystemMessage(content="Starting task execution"),
                HumanMessage(content=task.description)
            ]
            messages = cast(List[MessageLikeRepresentation], add_messages(state["messages"], new_messages))
            state["messages"] = messages

            # Display initial progress
            logger.info(f"Starting task: {task.description}")

            # Execute graph
            logger.info(f"Starting execution for task: {task.id}")
            final_state = await self.graph.ainvoke(state)

            # Get final response
            final_message = final_state["messages"][-1]
            if not isinstance(final_message, (AIMessage, Dict)):
                raise ValueError("Final state does not contain valid message")

            content = final_message.content if isinstance(final_message, AIMessage) else final_message.get("content")
            if not content:
                raise ValueError("Message has no content")

            # Calculate final metrics
            metrics = calculate_final_metrics(final_state)

            # Display completion progress
            logger.info(f"Task completed: {generate_summary(metrics)}")

            return FinalResponse(
                task_id=task.id,
                success=True,
                result={"response": str(content)},
                summary=generate_summary(metrics),
                metrics=metrics
            )

        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            raise

    async def _plan_task(self, state: State) -> State:
        """Plan task operations (Planner component)."""
        try:
            # Get task from state
            task = state.get("task")
            if not isinstance(task, Task):
                raise ValueError("Invalid task type")

            # Check if task already has analysis
            analysis = task.args.get("analysis")
            if not analysis:
                # Generate analysis
                analysis = await self._generate_analysis(task)

            # Create execution plan
            plan = await self._create_execution_plan(task, analysis)

            # Add plan to state
            new_messages = [
                SystemMessage(content=f"Execution plan created with {len(plan.operations)} operations"),
                AIMessage(content=str(plan.dict()))
            ]
            messages = cast(List[MessageLikeRepresentation], add_messages(state["messages"], new_messages))
            
            # Create new state with updated fields
            new_state: State = {
                "messages": messages,
                "task": task,
                "metrics": state["metrics"],
                "execution_plan": plan
            }

            return new_state

        except Exception as e:
            logger.error(f"Planning failed: {str(e)}")
            raise

    async def _execute_task(self, state: State) -> State:
        """Execute task operations (Executor component)."""
        try:
            # Get task from state
            task = state.get("task")
            if not isinstance(task, Task):
                raise ValueError("Invalid task type")

            # Get task data from memory
            task_data = await self.memory_manager.get_task(task.id)
            if not task_data:
                raise ValueError(f"Task {task.id} not found in memory")

            # Get operations from task data
            analysis = task_data.get("analysis", {})
            if not isinstance(analysis, dict):
                raise ValueError("Invalid task analysis: must be a dictionary")

            # Get operations from the revised plan if it exists
            operations = []
            if "revised_plan" in analysis and "steps" in analysis["revised_plan"]:
                # Convert steps to TaskOperation objects
                for step in analysis["revised_plan"]["steps"]:
                    operation = TaskOperation(
                        id=f"op_{step.get('step_number', len(operations) + 1)}",
                        description=step.get("action", ""),
                        tool_name=step.get("tool", ""),
                        args={
                            "reasoning": step.get("expected_outcome", ""),
                            "dependencies": step.get("dependencies", [])
                        }
                    )
                    operations.append(operation)
            else:
                # Convert existing operations to TaskOperation objects if they aren't already
                for op in analysis.get("operations", []):
                    if isinstance(op, TaskOperation):
                        operations.append(op)
                    else:
                        operation = TaskOperation(
                            id=op.get("id", f"op_{len(operations) + 1}"),
                            description=op.get("description", ""),
                            tool_name=op.get("tool_name", ""),
                            args=op.get("args", {})
                        )
                        operations.append(operation)

            if not isinstance(operations, list):
                raise ValueError("Invalid plan: operations must be a list")

            logger.info(f"Found {len(operations)} operations in analysis")

            # Create execution plan
            execution_plan = ExecutionPlan(
                task_id=task.id,
                description=task.description,
                operations=operations,
                status="pending",
                metadata={"created_at": datetime.now(UTC).isoformat()}
            )
            logger.info("Created execution plan")

            # Display plan
            display_execution_plan(execution_plan)

            # Create operations table
            operations_table = Table(
                title="Task Operations",
                show_header=True,
                header_style="bold magenta"
            )
            operations_table.add_column("ID", style="cyan")
            operations_table.add_column("Description", style="green")
            operations_table.add_column("Tool", style="yellow")
            operations_table.add_column("Status", style="blue")

            # Execute operations
            operation_results: List[TaskOperation] = []
            total_ops = len(execution_plan.operations)
            
            for i, op in enumerate(execution_plan.operations, 1):
                try:
                    # Get tool function
                    tool_func = self.tool_map[op.tool_name]
                    if not tool_func:
                        raise ValueError(f"Tool {op.tool_name} not found")

                    # Log progress
                    logger.info(f"Executing operation {i}/{total_ops}: {op.description}")

                    # Execute tool with timing
                    start = datetime.now(UTC)
                    
                    # Convert parameters to match tool interface
                    tool_params = {}
                    if op.tool_name == "search":
                        query = op.args.get("query", "")
                        if isinstance(op.args, dict):
                            query = op.args.get("query", op.args.get("reasoning", ""))
                        tool_params = {"input": query}
                    elif op.tool_name == "generate_domain_search_query":
                        tool_params = {"input": {"aspect": op.args}}
                    elif op.tool_name == "analyze_citations":
                        memories = await self.memory_manager.get_memory(op.args.get("reasoning", ""))
                        tool_params = {"input": {"context": op.args, "memories": memories}}
                    elif op.tool_name in ["analyze_data", "analyze_graph", "analyze_network"]:
                        tool_params = {"input": {"context": op.args}}
                    else:
                        tool_params = op.args
                    
                    result = await tool_func(**tool_params)
                    end = datetime.now(UTC)
                    duration = (end - start).total_seconds()
                    logger.info(f"Tool execution completed in {duration} seconds")

                    # Update operation with result
                    op.status = "completed"
                    op.result = result
                    op.start_time = start.isoformat()
                    op.end_time = end.isoformat()
                    operation_results.append(op)

                    # Update table
                    operations_table.add_row(
                        op.id,
                        op.description,
                        op.tool_name,
                        "[green]Success[/green]"
                    )

                except Exception as e:
                    logger.error(f"Operation failed: {str(e)}")
                    op.status = "failed"
                    op.error = str(e)
                    operation_results.append(op)

                    # Update table
                    operations_table.add_row(
                        op.id,
                        op.description,
                        op.tool_name,
                        f"[red]Failed: {str(e)}[/red]"
                    )

            # Display operations table
            console.print(operations_table)

            # Create execution result
            execution_result = ExecutionResult(
                task_id=task.id,
                description=task.description,
                success=all(op.status == "completed" for op in operation_results),
                operations=operation_results,
                error=None if all(op.status == "completed" for op in operation_results) else "Some operations failed",
                start_time=operation_results[0].start_time if operation_results else None,
                end_time=operation_results[-1].end_time if operation_results else None,
                metrics={"total_operations": len(operation_results)}
            )

            # Create final response
            final_response = FinalResponse(
                task_id=task.id,
                success=execution_result.success,
                result={"operations": [r.dict() for r in operation_results]},
                summary="Task execution completed successfully" if execution_result.success else "Task execution failed",
                metrics={"total_operations": len(operation_results)}
            )

            # Update state with execution result
            new_state = create_state_dict(
                messages=cast(List[MessageLikeRepresentation], add_messages(
                    state["messages"],
                    [SystemMessage(content=f"Executed {len(operation_results)} operations")]
                )),
                task=state["task"],
                metrics={
                    **state["metrics"],
                    "operation_count": len(operation_results),
                    "success": execution_result.success
                },
                execution_plan=state.get("execution_plan"),
                execution_result=execution_result
            )

            return new_state

        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            # Create error state
            return create_state_dict(
                messages=cast(List[MessageLikeRepresentation], add_messages(
                    state["messages"],
                    [SystemMessage(content=f"Task execution failed: {str(e)}")]
                )),
                task=state["task"],
                metrics=state["metrics"],
                execution_result=ExecutionResult(
                    task_id=task.id,
                    description="Task execution failed",
                    success=False,
                    operations=[],
                    error=str(e),
                    metrics={},
                    start_time=None,
                    end_time=None
                )
            )

    async def _join_results(self, state: State) -> State:
        """Join and analyze results (Joiner component)."""
        try:
            logger.info("Analyzing execution results...")

            # Get execution result from state
            execution_result = state.get("execution_result")
            if not isinstance(execution_result, ExecutionResult):
                logger.error("Invalid execution result type: %s", type(execution_result))
                raise ValueError("Invalid execution result format")

            # Convert operations to proper format
            formatted_operations = []
            for op in execution_result.operations:
                if isinstance(op, TaskOperation):
                    formatted_operations.append(op.model_dump())
                elif isinstance(op, dict):
                    formatted_operations.append(op)
                else:
                    logger.warning(f"Skipping operation with invalid type: {type(op)}")

            # Create chain with proper prompt template
            chain = JOINER_PROMPT | self.llm | PydanticOutputParser(pydantic_object=JoinerResponse)

            # Get execution time metrics
            time_metrics = self._get_execution_time(execution_result)

            # Prepare input for analysis
            analysis_input = {
                "execution_result": {
                    "operations": formatted_operations,
                    "metrics": execution_result.metrics,
                    "success": execution_result.success,
                    "error": execution_result.error,
                    "synthesis": execution_result.synthesis,
                    "document_path": execution_result.document_path,
                    "start_time": time_metrics["start_time"],
                    "end_time": time_metrics["end_time"]
                },
                "previous_plans": state.get("previous_plans", [])
            }

            try:
                # Analyze results with proper input dict
                join_result = await chain.ainvoke(analysis_input)
                join_output = JoinerResponse(**join_result)

                # Start with thought message
                messages = cast(List[MessageLikeRepresentation], add_messages(
                    state["messages"],
                    [SystemMessage(content=join_output.thought)]
                ))

                # Create new state with proper typing
                new_state = create_state_dict(
                    messages=messages,
                    task=state["task"],
                    metrics=state["metrics"],
                    execution_plan=state.get("execution_plan"),
                    execution_result=execution_result,
                    analysis=join_result
                )

                # Handle action
                if join_output.action.type == "replan":
                    # Add replan feedback using add_messages
                    if join_output.action.content.suggested_improvements:
                        new_state["messages"] = cast(List[MessageLikeRepresentation], add_messages(
                            new_state["messages"],
                            [HumanMessage(content=join_output.action.content.suggested_improvements)]
                        ))
                    
                    if join_output.action.content.error_analysis:
                        console.print(Panel(
                            Text(join_output.action.content.error_analysis),
                            title="Replanning Required",
                            border_style="yellow"
                        ))
                    logger.info("Replanning required...")
                else:
                    # Add final response using add_messages
                    if join_output.action.content.summary:
                        new_state["messages"] = cast(List[MessageLikeRepresentation], add_messages(
                            new_state["messages"],
                            [AIMessage(content=join_output.action.content.summary)]
                        ))
                        console.print(Panel(
                            Text(join_output.action.content.summary),
                            title="Final Response",
                            border_style="green"
                        ))
                    logger.info("Analysis completed successfully")

                return new_state

            except ValidationError as e:
                logger.error("Validation error in join result: %s", str(e))
                raise
            except Exception as e:
                logger.error("Error in join analysis: %s", str(e))
                raise

        except Exception as e:
            logger.error("Joining failed: %s", str(e))
            raise

    async def compile_task(self, task_description: str) -> Tuple[str, ExecutionPlan]:
        """Compile a task into an execution plan.

        Args:
            task_description: Description of the task to compile

        Returns:
            Tuple of task ID and execution plan
        """
        try:
            # Generate task ID
            task_id = f"task_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Generated task ID: {task_id}")

            # Create Task object
            task = Task(
                id=task_id,
                description=task_description,
                tool=Tool(
                    name="research",
                    description="Research task execution",
                    parameters={},
                    function=lambda: None  # Placeholder function
                ),
                args={}
            )
            logger.info("Created Task object")

            # Initialize state
            state = {
                "task": task,
                "messages": [],
                "tools": [],
                "plan": None,
                "execution_plan": None
            }
            logger.info("Initialized state")

            # Add initial messages to state
            state["messages"].append({
                "role": "system",
                "content": "You are a task planning assistant. Your role is to break down tasks into clear, actionable steps."
            })
            state["messages"].append({
                "role": "user",
                "content": f"Task: {task_description}"
            })
            logger.info("Added initial messages to state")

            # Create tool descriptions
            logger.info("Creating tool descriptions...")
            tool_descriptions = []
            for tool_name, tool in self.tool_map.items():
                description = {
                    "name": tool_name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
                tool_descriptions.append(description)
            logger.info(f"Created descriptions for {len(tool_descriptions)} tools")

            # Build LLM and planning chain
            logger.info("Building LLM and planning chain...")
            planning_chain = create_chain(
                prompt=PLANNER_PROMPT,
                llm=self.llm,
                output_key="plan",
                output_parser=PydanticOutputParser(pydantic_object=FinalPlan)
            )
            logger.info("Planning chain created")

            # Invoke planning chain
            logger.info("Invoking planning chain...")
            logger.info(f"Input task description: {task_description}")
            logger.info(f"Tool descriptions length: {len(str(tool_descriptions))} chars")
            response = await planning_chain.ainvoke({
                "task": task_description,
                "tool_descriptions": tool_descriptions
            })

            # Parse response
            logger.info(f"Parsed plan: {response}")
            plan = response.plan if response and hasattr(response, 'plan') else []

            # Create execution plan
            operations = []
            for step in plan:
                step_number = step.step_number if hasattr(step, 'step_number') else 0
                tool_suggestions = step.tool_suggestions if hasattr(step, 'tool_suggestions') else []
                tool_name = list(self.tool_map.keys())[tool_suggestions[0] - 1] if tool_suggestions else None

                if tool_name:
                    operation = {
                        "id": f"op_{step_number}",
                        "description": step.action if hasattr(step, 'action') else "",
                        "tool_name": tool_name,
                        "args": {
                            "reasoning": step.reasoning if hasattr(step, 'reasoning') else "",
                            "completion_conditions": step.completion_conditions if hasattr(step, 'completion_conditions') else "",
                            "implementation_notes": step.implementation_notes if hasattr(step, 'implementation_notes') else ""
                        },
                        "dependencies": [f"op_{i}" for i in range(1, step_number) if i in [s.step_number for s in plan]]
                    }
                    operations.append(operation)

            execution_plan = ExecutionPlan(
                task_id=task_id,
                description=task_description,
                operations=operations
            )
            logger.info("Created execution plan")

            # Save task to memory
            memories: List[KnowledgeTriple] = [
                {
                    "subject": task_id,  # Don't add extra task_ prefix
                    "predicate": "has_description",
                    "object_": task_description
                },
                {
                    "subject": task_id,  # Don't add extra task_ prefix
                    "predicate": "has_analysis",
                    "object_": json.dumps({"operations": operations})
                }
            ]
            await self.memory_manager.memory.save_recall_memory(memories, task_id)  # Use task_id as user_id
            logger.info("Task saved to memory")

            return task_id, execution_plan

        except Exception as e:
            logger.error(f"Error compiling task: {str(e)}")
            raise

    def _get_tool_names(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tool_map.keys())

    async def _validate_plan(self, state: State) -> State:
        """Validate the execution plan."""
        state_dict = cast(Dict[str, Any], state)
        if not is_valid_state(state_dict):
            raise ValueError("Invalid state provided to validate_plan")
        
        plan = state.get("execution_plan")
        if not plan:
            return state

        # Create validation chain with correct parameter order
        chain = create_chain(
            prompt=ANALYSIS_PROMPT,
            llm=self.llm,
            output_mode='json',
            output_parser=PydanticOutputParser(pydantic_object=FinalPlan)
        )

        # Validate plan
        validation = await chain.ainvoke({
            "plan": plan,
            "tools": self._get_tool_names()
        })

        if not validation.is_valid:
            # Request replan if validation fails
            messages = [*state["messages"]]
            messages.append(AIMessage(content=str({
                "type": "replan",
                "reason": validation.errors if hasattr(validation, "errors") else ["Plan validation failed"]
            })))
            
            new_state: State = {
                "messages": messages,
                "task": state["task"],
                "metrics": state["metrics"]
            }
            if "execution_plan" in state:
                new_state["execution_plan"] = state["execution_plan"]
            if "schedule" in state:
                new_state["schedule"] = state["schedule"]
            if "execution_result" in state:
                new_state["execution_result"] = state["execution_result"]
            
            return new_state

        return state

    async def _analyze_results(self, state: State) -> State:
        """Analyze execution results."""
        result = state.get("execution_result")
        if not result:
            return state

        # Create analysis chain
        chain = create_chain(
            prompt=ANALYSIS_PROMPT,
            output_mode='json',
            output_parser=PydanticOutputParser(pydantic_object=FinalPlan),
            llm=self.llm
        )

        # Analyze results
        analysis = await chain.ainvoke({
            "result": result,
            "metrics": state.get("metrics", {})
        })

        messages = [*state["messages"]]
        messages.append(AIMessage(content=str(analysis)))
        
        new_state: State = {
            "messages": messages,
            "task": state["task"],
            "metrics": state["metrics"],
            "analysis": analysis
        }
        if "execution_plan" in state:
            new_state["execution_plan"] = state["execution_plan"]
        if "schedule" in state:
            new_state["schedule"] = state["schedule"]
        if "execution_result" in state:
            new_state["execution_result"] = state["execution_result"]
        
        return new_state

    def _get_tool_descriptions(self) -> str:
        """Get descriptions of available tools.
        
        Returns:
            String containing tool descriptions
        """
        descriptions = []

        # Get tool methods from ResearchTools
        tool_methods = [
            method for method in dir(self.research_tools)
            if not method.startswith('_') and callable(getattr(self.research_tools, method))
        ]

        for method in tool_methods:
            tool = getattr(self.research_tools, method)
            if hasattr(tool, '__doc__') and tool.__doc__:
                desc = f"{method}: {tool.__doc__.strip()}"
                descriptions.append(desc)

        return "\n".join(descriptions)

    async def execute_task(
            self,
            task_id: str,
            tool_config: Optional[RunnableConfig] = None
    ) -> ExecutionResult:
        """Execute a compiled task."""
        try:
            logger.info("Starting execute_task for task_id: %s", task_id)

            # Get task from memory - memory_manager.get_memory returns Dict[str, Any]
            task_data = await self.memory_manager.get_task(task_id)
            if not task_data or not isinstance(task_data, dict):
                logger.error("Task %s not found in memory", task_id)
                raise ValueError(f"Task {task_id} not found")
            logger.info("Retrieved task data from memory: keys=%s", list(task_data.keys()))

            # Extract analysis from task data
            analysis = task_data.get("analysis", {})
            if not isinstance(analysis, dict) or "operations" not in analysis:
                logger.error("Invalid task analysis format: type=%s, has_operations=%s",
                             type(analysis), "operations" in analysis if isinstance(analysis, dict) else False)
                raise ValueError("Invalid task analysis: missing operations list")
            logger.info("Task analysis contains %d operations", len(analysis.get("operations", [])))

            # Create a proper Task object
            task = Task(
                id=task_id,
                description=str(task_data.get("description", "")),
                tool=Tool(
                    name="research",
                    description="Research task execution",
                    parameters={},
                    function=lambda: None  # Placeholder function
                ),
                args={"tool_config": tool_config, "analysis": analysis} if tool_config else {"analysis": analysis}
            )

            # Create initial state using helper function
            state = create_state_dict(
                messages=[],
                task=task,
                metrics={}
            )

            # Add initial messages
            state["messages"] = cast(List[MessageLikeRepresentation], add_messages(
                state["messages"],
                [
                    SystemMessage(content=f"Starting execution of task {task_id}"),
                    HumanMessage(content=str(task_data.get("description", "")))
                ]
            ))
            logger.info("Added initial messages to state")

            # Execute graph
            logger.info("Starting graph execution for task: %s", task_id)
            final_state = await self.graph.ainvoke(state)  # Use ainvoke for async execution
            logger.info("Graph execution completed, final state keys: %s", 
                list(final_state.keys()))

            # Get execution result from final state
            execution_result = final_state.get("execution_result")
            if not isinstance(execution_result, ExecutionResult):
                logger.error("Invalid execution result type: %s", type(execution_result))
                raise ValueError("Invalid execution result format")
            logger.info("Retrieved execution result: success=%s", execution_result.success)

            return ExecutionResult(
                task_id=task_id,
                description="Task execution completed successfully",
                success=True,
                operations=execution_result.operations,
                error=None,
                metrics=execution_result.metrics,
                summary="Task execution completed successfully",
                start_time=execution_result.start_time,
                end_time=execution_result.end_time
            )

        except Exception as e:
            logger.error("Task execution failed: %s", str(e), exc_info=True)
            return ExecutionResult(
                task_id=task_id,
                description="Task execution failed",
                success=False,
                operations=[],
                error=str(e),
                metrics={},
                summary="Task execution failed",
                start_time=None,
                end_time=None
            )

    async def _generate_analysis(self, task: Task) -> Dict[str, Any]:
        """Generate analysis for a task."""
        try:
            chain = create_chain(
                prompt=ANALYSIS_PROMPT,
                llm=self.llm,
                output_key="analysis"
            )
            
            result = await chain.ainvoke({
                "task": task.description,
                "tools": self._get_tool_descriptions()
            })
            
            return result["analysis"]
            
        except Exception as e:
            logger.error(f"Analysis generation failed: {str(e)}")
            raise

    async def _create_execution_plan(
        self,
        task: Task,
        analysis: Dict[str, Any]
    ) -> ExecutionPlan:
        """Create execution plan from analysis."""
        try:
            operations = []
            for op in analysis.get("operations", []):
                tool_name = op.get("tool_name")
                if not tool_name or tool_name not in self.tool_map:
                    raise ValueError(f"Invalid tool: {tool_name}")
                    
                tool = self.tool_map[tool_name]
                operations.append(
                    TaskOperation(
                        id=str(op["id"]),
                        tool_name=tool_name,
                        description=str(op["description"]),
                        dependencies=list(op.get("dependencies", [])),
                        args=dict(op.get("args", {}))
                    )
                )
                
            return ExecutionPlan(
                task_id=task.id,
                description=task.description,
                operations=operations
            )
            
        except Exception as e:
            logger.error(f"Plan creation failed: {str(e)}")
            raise

    async def _execute_operation(self, operation: TaskOperation) -> OperationResult:
        """Execute a single operation."""
        try:
            # Get tool and inputs
            tool_name = operation.tool_name
            args = operation.args
            
            # Log operation start
            logger.info(f"Executing operation {operation.id} with tool {tool_name}")
            
            # Get tool instance
            tool = self.tool_map.get(tool_name)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")
                
            # Execute tool
            start_time = datetime.now(UTC)
            start_time_str = start_time.isoformat()
            result = await tool(**args)
            end_time = datetime.now(UTC)
            end_time_str = end_time.isoformat()
            
            # Log success
            logger.info(f"Operation {operation.id} completed in {(end_time - start_time).total_seconds():.2f}s")
            
            return OperationResult(
                operation_id=operation.id,
                success=True,
                result=result,
                error=None,
                start_time=start_time_str,
                end_time=end_time_str,
                metrics={"duration": (end_time - start_time).total_seconds()}
            )
            
        except Exception as e:
            logger.error(f"Operation {operation.id} failed: {str(e)}")
            return OperationResult(
                operation_id=operation.id,
                success=False,
                result=None,
                error=str(e),
                start_time=None,
                end_time=None,
                metrics=None
            )

    async def _execute_plan(self, state: State) -> State:
        """Execute operations in the plan (Executor component)."""
        try:
            # Get execution plan
            plan = state.get("execution_plan")
            if not isinstance(plan, ExecutionPlan):
                raise ValueError("No execution plan in state")

            # Execute each operation
            results: List[OperationResult] = []
            for operation in plan.operations:
                result = await self._execute_operation(operation)
                results.append(result)
                
                # Update operation status based on success
                operation.status = "completed" if result.success else "failed"
                operation.result = result.result if result.result is not None else None
                operation.error = result.error if result.error is not None else None
                
            # Create execution result
            execution_result = ExecutionResult(
                task_id=plan.task_id,
                description=plan.description,
                success=all(r.success for r in results),
                operations=cast(List[TaskOperation], results),
                error=None if all(r.success for r in results) else "Some operations failed",
                start_time=results[0].start_time if results else None,
                end_time=results[-1].end_time if results else None,
                metrics={"total_operations": len(results)}
            )

            # Add result to state
            new_messages = [
                SystemMessage(content=f"Execution completed with {len(results)} operations"),
                AIMessage(content=execution_result.model_dump_json())
            ]
            messages = cast(List[MessageLikeRepresentation], add_messages(state["messages"], new_messages))
            
            # Create new state with proper casting
            new_state: Dict[str, Any] = {
                "messages": messages,
                "task": state["task"],
                "metrics": {
                    **state["metrics"],
                    "operation_count": len(results),
                    "success": execution_result.success
                },
                "execution_plan": plan,
                "execution_result": execution_result
            }

            return cast(State, new_state)

        except Exception as e:
            logger.error(f"Execution failed: {str(e)}")
            raise

    async def analyze_results(self, results: List[OperationResult]) -> Dict[str, Any]:
        """Analyze operation results."""
        try:
            # Create analysis chain
            chain = create_chain(
                llm=self.llm,
                prompt=ANALYSIS_PROMPT,
                output_mode='json',
                output_parser=PydanticOutputParser(pydantic_object=FinalPlan)
            )
            
            # Format results for analysis
            formatted_results = "\n".join(
                f"Operation {i+1}:\n{result.result}\n"
                for i, result in enumerate(results)
            )
            
            # Execute chain
            analysis = await chain.ainvoke({
                "results": formatted_results
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Results analysis failed: {str(e)}")
            raise

    async def replan_task(self, task_id: str, feedback: str) -> Dict[str, Any]:
        """Replan a task based on feedback.
        
        Args:
            task_id: ID of the task to replan
            feedback: Feedback to incorporate into replanning
            
        Returns:
            Dictionary containing the replanning results
        """
        try:
            # Get task data from memory
            task_data = await self.memory_manager.get_task(task_id)
            if not task_data:
                raise ValueError(f"Task {task_id} not found")
            
            # Get original plan and execution results
            original_plan = task_data.get("analysis", {})
            execution_results = task_data.get("execution_results", {})
            
            # Create chain for replanning
            chain = create_chain(
                llm=self.llm,
                prompt=REPLAN_PROMPT,
                output_parser=PydanticOutputParser(pydantic_object=Replan)
            )
            
            # Generate new plan with all required variables
            result = await chain.ainvoke({
                "task": task_data,
                "original_plan": original_plan,
                "execution_results": execution_results,
                "tool_descriptions": self._get_tool_descriptions(),
                "feedback": feedback
            })
            
            # Convert result to dictionary format
            replan_result = {
                "feedback": feedback,  # Required string field
                "error_analysis": {  # Optional dictionary field
                    "issues": [],
                    "successful_parts": result.operations if hasattr(result, "operations") else [],
                    "metrics_analysis": {},
                    "revised_plan": {
                        "operations": [op.model_dump() for op in result.operations] if hasattr(result, "operations") else []
                    },
                    "recommendations": []
                }
            }
            
            # Store updated plan
            await self.memory_manager.store_task(task_id, {
                "description": task_data.get("description", ""),
                "analysis": replan_result
            })
            
            return replan_result
            
        except Exception as e:
            logger.error(f"Replanning failed: {str(e)}")
            raise

    def _get_execution_time(self, execution_result: ExecutionResult) -> Dict[str, str]:
        """Get execution time metrics."""
        current_time = datetime.now(UTC).isoformat()
        return {
            "start_time": current_time,
            "end_time": current_time
        }

def display_execution_plan(plan: ExecutionPlan) -> None:
    """Display the execution plan in a formatted table."""
    try:
        # Create table
        table = Table(
            title=f"Execution Plan: {plan.task_id}",
            show_header=True,
            header_style="bold magenta"
        )
        
        # Add columns
        for col, style in OPERATION_TABLE_COLUMNS:
            table.add_column(col, style=style)
            
        # Add rows
        for op in plan.operations:
            table.add_row(
                op.id,
                op.description,
                op.tool_name,
                op.status
            )
            
        # Display table
        console.print(table)
        
    except Exception as e:
        logger.error(f"Failed to display execution plan: {str(e)}")
        raise

def create_initial_state(task: Task, messages: MessageList) -> State:
    """Create initial state with required fields."""
    state_dict = {
        "messages": messages,
        "task": task,
        "metrics": {"success": True},
        "execution_plan": None,
        "execution_result": None
    }
    return cast(State, state_dict)

def create_error_state(task: Task, messages: MessageList, error: str) -> State:
    """Create error state with required fields."""
    state_dict = {
        "messages": messages,
        "task": task,
        "metrics": {
            "success": False,
            "error": error
        },
        "execution_plan": None,
        "execution_result": None
    }
    return cast(State, state_dict)

def create_execution_result(
    task_id: str,
    description: str,
    operations: Sequence[TaskOperation],
    error: Optional[str] = None
) -> ExecutionResult:
    """Create execution result from operations."""
    success = all(getattr(op, "status", "") == "completed" for op in operations)
    
    # Get start and end times from operations
    start_time = None
    end_time = None
    if operations:
        # Try to get datetime objects from operation timestamps
        try:
            start_times = [
                datetime.fromisoformat(str(getattr(op, "start_time", "")))
                for op in operations
                if getattr(op, "start_time", None) is not None
            ]
            end_times = [
                datetime.fromisoformat(str(getattr(op, "end_time", "")))
                for op in operations
                if getattr(op, "end_time", None) is not None
            ]
            if start_times:
                start_time = min(start_times)
            if end_times:
                end_time = max(end_times)
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing operation timestamps: {str(e)}")
    
    return ExecutionResult(
        task_id=task_id,
        description=description,
        success=success,
        operations=list(operations),
        error=error if error else None if success else "Some operations failed",
        start_time=start_time,
        end_time=end_time,
        metrics={"total_operations": len(operations)}
    )

def create_final_response(
    task_id: str,
    success: bool,
    operations: TaskOperationList,
    error: Optional[str] = None
) -> FinalResponse:
    """Create final response from operations."""
    return FinalResponse(
        task_id=task_id,
        success=success,
        result={"operations": [get_operation_result(op) for op in operations]},
        summary="Task execution completed successfully" if success else "Task execution failed",
        metrics={"total_operations": len(operations)}
    )

def update_state_with_result(
    state: State,
    execution_result: ExecutionResult,
    messages: Optional[MessageList] = None
) -> State:
    """Update state with execution result."""
    state_dict = {
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
    """Get operation result as a dictionary."""
    return {
        "id": operation.id,
        "tool_name": operation.tool_name,
        "description": operation.description,
        "status": getattr(operation, "status", "unknown"),
        "result": getattr(operation, "result", None),
        "error": getattr(operation, "error", None),
        "start_time": getattr(operation, "start_time", None),
        "end_time": getattr(operation, "end_time", None)
    }

def is_valid_state(obj: Any) -> bool:
    """Check if an object is a valid state."""
    return (
        isinstance(obj, dict) and
        "messages" in obj and
        "task" in obj and
        "metrics" in obj and
        isinstance(obj["messages"], list) and
        isinstance(obj["task"], Task) and
        isinstance(obj["metrics"], dict)
    )

def create_state_dict(
    messages: List[MessageLikeRepresentation],
    task: Task,
    metrics: Dict[str, Any],
    execution_plan: Optional[ExecutionPlan] = None,
    execution_result: Optional[ExecutionResult] = None,
    analysis: Optional[Dict[str, Any]] = None
) -> State:
    """Create a state dictionary with proper typing."""
    state_dict: Dict[str, Any] = {
        "messages": messages,
        "task": task,
        "metrics": metrics
    }
    if execution_plan is not None:
        state_dict["execution_plan"] = execution_plan
    if execution_result is not None:
        state_dict["execution_result"] = execution_result
    if analysis is not None:
        state_dict["analysis"] = analysis
    return cast(State, state_dict)
