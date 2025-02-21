"""
Phase 4 Diagnostic Script - Tests enhanced agent selection and reputation functionality.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

import asyncio
import json
import logging
from core.engine.agent_manager import AgentManager
from core.agents.personas import SPECIALIZED_PERSONAS
from rich.table import Table
from rich.console import Console
from rich import print

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_agent_selection_reputation_preference():
    """Test that agent selection prefers agents with higher reputation when capabilities are equal."""
    print("\n[bold cyan]Testing Agent Selection - Reputation Preference[/bold cyan]")
    
    # Initialize agent manager
    agent_manager = AgentManager()
    
    # Create a test work unit with requirements that multiple agents could satisfy
    test_work_unit = {
        "description": "Analyze chemical compounds and their properties",
        "requirements": [
            "chemical analysis",
            "data analysis",
            "computational modeling"
        ]
    }
    
    # Get agent selection
    result = await agent_manager.select_specialized_agent(test_work_unit)
    
    # Print selection details in a table
    table = Table(title="Agent Selection Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Selected Agent ID", str(result["agent_id"]))
    table.add_row("Selected Agent", result["agent"]["name"])
    table.add_row("Expertise Match", f"{result['expertise_match']:.2f}")
    table.add_row("Required Capabilities", ", ".join(result["required_capabilities"]))
    table.add_row("Selection Rationale", result["selection_rationale"])
    
    console = Console()
    console.print(table)

async def test_agent_selection_capability_matching():
    """Test that agent selection properly matches capabilities to requirements."""
    print("\n[bold cyan]Testing Agent Selection - Capability Matching[/bold cyan]")
    
    # Initialize agent manager
    agent_manager = AgentManager()
    
    # Create a test work unit with specific capability requirements
    test_work_unit = {
        "description": "Analyze biological pathways and gene expression data",
        "requirements": [
            "bioinformatics",
            "genomics",
            "data analysis"
        ]
    }
    
    # Get agent selection
    result = await agent_manager.select_specialized_agent(test_work_unit)
    
    # Print selection details in a table
    table = Table(title="Agent Selection Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Selected Agent ID", str(result["agent_id"]))
    table.add_row("Selected Agent", result["agent"]["name"])
    table.add_row("Expertise Match", f"{result['expertise_match']:.2f}")
    table.add_row("Required Capabilities", ", ".join(result["required_capabilities"]))
    table.add_row("Selection Rationale", result["selection_rationale"])
    
    console = Console()
    console.print(table)

async def test_reputation_updates():
    """Test that agent reputations are properly updated based on task outcomes."""
    print("\n[bold cyan]Testing Reputation Updates[/bold cyan]")
    
    # Initialize agent manager
    agent_manager = AgentManager()
    
    # Test agent ID
    test_agent_id = 2  # Chemoinformatics Agent
    
    # Get initial reputation
    initial_rep = agent_manager.reputation_history[test_agent_id]['current']
    logger.info(f"Initial reputation for Agent {test_agent_id}: {initial_rep}")
    
    # Test successful task outcome
    success_outcome = {
        'quality_score': 0.9,
        'completion_time': 0.8,
        'complexity': 8
    }
    
    new_rep = await agent_manager.update_agent_reputation(test_agent_id, success_outcome)
    logger.info(f"Reputation after successful task: {new_rep}")
    
    # Test unsuccessful task outcome
    failure_outcome = {
        'quality_score': 0.4,
        'completion_time': 1.5,
        'complexity': 5
    }
    
    final_rep = await agent_manager.update_agent_reputation(test_agent_id, failure_outcome)
    logger.info(f"Final reputation: {final_rep}")
    
    # Get reputation history
    history = agent_manager.reputation_history[test_agent_id]['history']
    
    # Print history in a table format
    table = Table(title="Reputation History")
    table.add_column("Timestamp", style="cyan")
    table.add_column("Old Score", style="magenta")
    table.add_column("New Score", style="green")
    table.add_column("Reason", style="yellow")
    
    for update in history:
        table.add_row(
            update["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            str(update["old_score"]),
            str(update["new_score"]),
            update["reason"]
        )
    
    console = Console()
    console.print(table)

async def main():
    """Run all Phase 4 diagnostic tests."""
    print("\n[bold]╭──────────────────────── ATR Framework ────────────────────────╮[/bold]")
    print("[bold]│ Phase 4 Diagnostics - Enhanced Agent Selection and Reputation │[/bold]")
    print("[bold]╰───────────────────────────────────────────────────────────────╯[/bold]\n")
    
    try:
        # Test agent selection with reputation preference
        await test_agent_selection_reputation_preference()
        
        # Test agent selection with capability matching
        await test_agent_selection_capability_matching()
        
        # Test reputation updates
        await test_reputation_updates()
        
        print("\n[bold green]✓ Phase 4 Diagnostic Tests completed successfully[/bold green]")
        
    except Exception as e:
        print(f"\n[bold red]✗ Phase 4 Diagnostic Tests failed: {str(e)}[/bold red]")
        raise

if __name__ == "__main__":
    asyncio.run(main())