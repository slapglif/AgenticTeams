"""End-to-end test for agentic task execution with memory management and constraints."""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import json
import asyncio
from datetime import datetime, UTC, timedelta
from typing import List, Optional, Dict, Any
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.panel import Panel
from langchain_core.runnables import RunnableConfig
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import aiohttp

from core.memory.memory_manager import MemoryManager
from core.engine.research_tools import ResearchTools
from core.compile.compiler import LangGraphCompiler
from core.shared.settings import build_llm
from core.shared.models import ExecutionResult, TaskOperation
from core.shared.logging_config import console, get_logger, log_operation, setup_logging

# Initialize logger
logger = get_logger(__name__)

# Configure logging with a specific directory for e2e tests
setup_logging(log_dir="logs/e2e_tests")

progress_columns = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeElapsedColumn(),
)

class AgentTaskContract:
    """Contract for tracking API calls and memory usage during tests."""
    
    def __init__(self):
        self.api_calls = 0
        self.memory_usage = 0
        logger.debug("Initialized AgentTaskContract")
    
    def increment_api_calls(self):
        """Increment API call counter."""
        self.api_calls += 1
        logger.debug("API calls incremented to {}", self.api_calls)
    
    def update_memory_usage(self, bytes_used):
        """Update memory usage tracking."""
        self.memory_usage += bytes_used
        logger.debug("Memory usage updated to {} bytes", self.memory_usage)

async def setup_components():
    """Initialize and set up all required components for testing."""
    logger.info("Starting component initialization")
    
    try:
        # Create progress display
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        )
        
        with progress:
            # Set up memory first
            task1 = progress.add_task("Setting up memory...", total=100)
            memory_manager = MemoryManager()
            progress.update(task1, completed=100)
            logger.info("Memory manager initialized")
            
            # Initialize research tools with memory manager
            task2 = progress.add_task("Initializing research tools...", total=100)
            research_tools = ResearchTools(memory_manager=memory_manager)
            progress.update(task2, completed=100)
            logger.info("Research tools initialized")
            
            # Initialize compiler
            task3 = progress.add_task("Setting up compiler...", total=100)
            llm = build_llm()
            compiler = LangGraphCompiler(
                llm=llm,
                memory_manager=memory_manager,
                research_tools=research_tools
            )
            progress.update(task3, completed=100)
            logger.info("Compiler initialized")
        
        return compiler, memory_manager, research_tools
    
    except Exception as e:
        logger.error("Failed to initialize components: {}", str(e))
        raise

async def cleanup_components(compiler, memory_manager, research_tools):
    """Clean up test components."""
    logger.info("Starting component cleanup")
    try:
        # Clean up research tools
        if research_tools:
            if hasattr(research_tools, '_session') and research_tools._session:
                await research_tools._session.close()
            logger.debug("Research tools cleaned up")

        # Clean up compiler
        if compiler:
            if hasattr(compiler, 'cleanup'):
                await compiler.cleanup()
            logger.debug("Compiler cleaned up")

        # Clean up memory manager
        if memory_manager:
            if hasattr(memory_manager, 'memory'):
                memory_manager.memory = None
            if hasattr(memory_manager, 'vector_store'):
                memory_manager.vector_store = None
            logger.debug("Memory manager cleaned up")

    except Exception as e:
        logger.error("Error during cleanup: {}", str(e))
        raise

async def test_research_task():
    """Test end-to-end research task execution."""
    logger.info("Starting e2e research task test")
    
    compiler = None
    memory_manager = None
    research_tools = None
    
    try:
        # Set up components
        compiler, memory_manager, research_tools = await setup_components()
        
        # Create progress display for task execution
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        )
        
        with progress:
            # Test dynamic replanning with a complex task
            task = progress.add_task("Executing research task with replanning...", total=100)
            
            # Initial task compilation
            task_description = "Research quantum computing advancements in error correction, focusing on recent breakthroughs"
            
            # Initial plan compilation
            task_id, execution_plan = await compiler.compile_task(task_description)
            logger.info(f"Task ID generated: {task_id}")
            
            progress.update(task, completed=25)
            logger.info("Initial plan compiled")
            
            # Store initial plan with proper task ID format
            await memory_manager.store_task(task_id, {
                "description": task_description,
                "analysis": {
                    "operations": [op.model_dump() for op in execution_plan.operations],
                    "initial_plan": True,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "feedback": {
                        "analysis": {
                            "issues": [],
                            "successful_parts": ["Initial plan compilation successful"],
                            "metrics_analysis": {
                                "key_findings": [],
                                "areas_for_improvement": []
                            }
                        }
                    }
                }
            })
            
            progress.update(task, completed=50)
            
            # Execute initial phase
            result = await compiler.execute_task(task_id)
            
            # Trigger replanning with new context
            logger.info("Triggering dynamic replanning")
            feedback = "The research plan needs to be revised to focus more on recent experimental demonstrations and practical implementations of quantum error correction techniques. The current results lack emphasis on real-world applications and achieved error rates. We should prioritize papers that showcase actual implementations and their achieved error rates."
            replan_result = await compiler.replan_task(task_id, feedback)
            
            # Verify replan result
            assert replan_result is not None
            assert isinstance(replan_result, dict)
            assert "feedback" in replan_result
            assert isinstance(replan_result["feedback"], str)
            
            progress.update(task, completed=75)
            
            # Execute final phase
            final_result = await compiler.execute_task(task_id)
            
            progress.update(task, completed=100)
        
        # Verify results
        assert final_result is not None
        assert isinstance(final_result, ExecutionResult)
        assert final_result.success is True
        
        # Verify replanning occurred by checking the number of operations
        assert len(final_result.operations) > 0
        assert all(isinstance(op, TaskOperation) for op in final_result.operations)
        
        logger.info("Research task completed successfully with dynamic replanning")
        logger.debug(f"Final operations after replanning: {len(final_result.operations)}")
        
        return final_result
    
    except Exception as e:
        logger.error("Research task failed: {}", str(e))
        raise
    
    finally:
        # Clean up
        if any([compiler, memory_manager, research_tools]):
            await cleanup_components(compiler, memory_manager, research_tools)

def main():
    """Main entry point for running the test directly with Python."""
    try:
        # Run the async test
        result = asyncio.run(test_research_task())
        
        # Print results in a nice format
        console.print("\n")
        console.print(Panel.fit(
            f"[green]Test completed successfully![/green]\n"
            f"Task ID: {result.task_id}\n"
            f"Operations executed: {len(result.operations)}\n"
            f"Start time: {result.start_time}\n"
            f"End time: {result.end_time}\n"
            f"Summary: {result.summary}",
            title="Test Results",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"\n[red]Test failed: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
