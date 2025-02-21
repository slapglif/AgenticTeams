"""Phase 4 diagnostic tests for ATR Framework - Enhanced Agent Selection and Reputation."""

import os
import sys
import asyncio
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

# Add core directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from core.engine.agent_manager import AgentManager
from core.agents.personas import SPECIALIZED_PERSONAS

# Configure rich console
console = Console()

# Configure logging with rich handler
logger.remove()  # Remove default handler
logger.add(
    RichHandler(console=console, show_time=False),
    format="{message}",
    level="INFO"
)
logger.add(
    "logs/phase4_diagnostic_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG"
)

async def test_agent_selection_reputation_preference():
    """Test that agent selection prefers agents with higher reputation when capabilities are equal."""
    console.rule("[bold blue]Testing Agent Selection - Reputation Preference")
    
    agent_manager = AgentManager()
    
    # Create a test topic that multiple agents could handle
    test_topic = {
        "description": "Analyze chemical compounds and their properties",
        "required_capabilities": ["analyze", "chemical analysis"]
    }
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Selecting agent...", total=1)
        
        # Select agent for the task
        result = await agent_manager.select_specialized_agent(test_topic)
        progress.update(task, advance=1)
        
        # Create a table to display results
        table = Table(title="Agent Selection Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Selected Agent ID", str(result["selected_agent_id"]))
        table.add_row("Selected Agent", SPECIALIZED_PERSONAS[result["selected_agent_id"]]["name"])
        table.add_row("Expertise Match", f"{result['expertise_match']:.2f}")
        table.add_row("Required Capabilities", ", ".join(result["capability_requirements"]))
        table.add_row("Selection Rationale", result["rationale"])
        
        console.print(table)
        
        return result

async def test_agent_selection_capability_matching():
    """Test that agent selection matches capabilities to requirements."""
    console.rule("[bold blue]Testing Agent Selection - Capability Matching")
    
    agent_manager = AgentManager()
    
    # Create a test topic with specific capability requirements
    test_topic = {
        "description": "Analyze gene expression patterns in response to natural compounds",
        "required_capabilities": ["genomics", "bioinformatics", "data analysis"]
    }
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Selecting agent...", total=1)
        
        # Select agent for the task
        result = await agent_manager.select_specialized_agent(test_topic)
        progress.update(task, advance=1)
        
        # Create a table to display results
        table = Table(title="Agent Selection Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Selected Agent ID", str(result["selected_agent_id"]))
        table.add_row("Selected Agent", SPECIALIZED_PERSONAS[result["selected_agent_id"]]["name"])
        table.add_row("Expertise Match", f"{result['expertise_match']:.2f}")
        table.add_row("Required Capabilities", ", ".join(result["capability_requirements"]))
        table.add_row("Selection Rationale", result["rationale"])
        
        console.print(table)
        
        return result

async def test_reputation_updates():
    """Test reputation updates based on task outcomes."""
    console.rule("[bold blue]Testing Reputation Updates")
    
    agent_manager = AgentManager()
    
    # Test with Chemoinformatics Agent (ID: 2)
    test_agent_id = 2
    initial_reputation = SPECIALIZED_PERSONAS[test_agent_id]["initial_reputation"]
    
    console.print(f"Initial reputation for Agent {test_agent_id}: {initial_reputation}")
    
    # Simulate successful task completion
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Simulating successful task...", total=1)
        
        # Update reputation for success (quality: 0.9, completion_time: 0.8, complexity: 8)
        await agent_manager.update_agent_reputation(
            agent_id=test_agent_id,
            quality_score=0.9,
            completion_time=0.8,
            complexity=8
        )
        progress.update(task, advance=1)
    
    # Get updated reputation
    reputation_after_success = await agent_manager.get_agent_reputation(test_agent_id)
    console.print(f"Reputation after successful task: {reputation_after_success}")
    
    # Simulate failed task
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Simulating failed task...", total=1)
        
        # Update reputation for failure (quality: 0.4, completion_time: 1.5, complexity: 5)
        await agent_manager.update_agent_reputation(
            agent_id=test_agent_id,
            quality_score=0.4,
            completion_time=1.5,
            complexity=5
        )
        progress.update(task, advance=1)
    
    # Get final reputation
    final_reputation = await agent_manager.get_agent_reputation(test_agent_id)
    console.print(f"Final reputation: {final_reputation}")
    
    # Get reputation history
    history = await agent_manager.get_agent_reputation_history(test_agent_id)
    
    # Create a table to display reputation history
    table = Table(title="Reputation History")
    table.add_column("Timestamp", style="cyan")
    table.add_column("Old Score", style="yellow")
    table.add_column("New Score", style="green")
    table.add_column("Reason", style="blue")
    
    for entry in history:
        table.add_row(
            entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            str(entry["old_score"]),
            str(entry["new_score"]),
            entry["reason"]
        )
    
    console.print(table)

async def main():
    """Run all Phase 4 diagnostic tests."""
    console.print(Panel.fit(
        "[bold blue]Phase 4 Diagnostics - Enhanced Agent Selection and Reputation",
        title="ATR Framework"
    ))
    
    try:
        # Test agent selection based on reputation
        await test_agent_selection_reputation_preference()
        
        # Test agent selection based on capabilities
        await test_agent_selection_capability_matching()
        
        # Test reputation updates
        await test_reputation_updates()
        
        console.print("[bold green]✓ Phase 4 Diagnostic Tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in Phase 4 diagnostics: {e}")
        console.print(f"[bold red]✗ Phase 4 Diagnostic Tests failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 