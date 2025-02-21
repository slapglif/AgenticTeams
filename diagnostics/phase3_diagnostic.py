"""Phase 3 diagnostic tests for ATR Framework."""

import os
import json
import time
import uuid
import sys
from pathlib import Path
from datetime import datetime, timedelta
from web3 import Web3
from eth_tester import EthereumTester, PyEVMBackend
from solcx import compile_source, install_solc
from loguru import logger
from rich.console import Console
from rich.traceback import install
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

# Add core directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from core.blockchain.atr_testnet import ATRTestnet

# Configure rich
console = Console()
install(show_locals=True)

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    RichHandler(
        console=console,
        show_path=False,
        enable_link_path=False,
        markup=True,
        rich_tracebacks=True
    ),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/phase3_diagnostic_{time}.log",
    rotation="500 MB",
    retention="10 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)

def test_responsibility_nft(atr_testnet):
    """Test the minting and management of Responsibility NFTs."""
    console.rule("[bold blue]Starting Responsibility NFT Tests[/bold blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Register test agent
        task = progress.add_task("Registering test agent...", total=None)
        agent_type = "TestAgent"
        capabilities = ["test_capability"]
        agent_uuid = atr_testnet.register_agent_onchain(agent_type, capabilities)
        progress.update(task, completed=True)
        logger.success(f"Registered agent with UUID {agent_uuid}")
        
        # Create work unit
        task = progress.add_task("Creating test work unit...", total=None)
        current_time = int(datetime.now().timestamp())
        test_work_unit = {
            "name": "NFT Test Work Unit",
            "description": "Test work unit for NFT minting",
            "version": "1.0",
            "inputs": json.dumps(["input1"]),
            "outputs": json.dumps(["output1"]),
            "requirements": json.dumps({
                "agent_type": "TestAgent",
                "capabilities": ["test_capability"]
            }),
            "temporal_constraints": json.dumps({
                "start_time": current_time - 1800,
                "deadline": current_time + 1800,
                "max_duration": 3600
            }),
            "data_access_controls": json.dumps({
                "allowed_sources": ["source1"],
                "allowed_queries": ["query1"]
            }),
            "payment_token": "0x0000000000000000000000000000000000000000",
            "payment_amount": Web3.to_wei(1, 'ether')
        }
        
        work_unit_id = atr_testnet.define_work_unit_onchain(**test_work_unit)
        progress.update(task, completed=True)
        
        # Test NFT minting for assignment
        console.print("\n[bold cyan]Test 1: Minting NFT for work unit assignment[/bold cyan]")
        try:
            nft_id = atr_testnet.mint_responsibility_nft(
                agent_uuid,
                work_unit_id,
                "assignment",
                {
                    "assignment_time": current_time,
                    "requirements": json.loads(test_work_unit["requirements"])
                }
            )
            logger.success(f"Successfully minted assignment NFT with ID {nft_id}")
        except Exception as e:
            logger.error(f"Test 1 failed: {str(e)}")
        
        # Test NFT minting for completion
        console.print("\n[bold cyan]Test 2: Minting NFT for work unit completion[/bold cyan]")
        try:
            nft_id = atr_testnet.mint_responsibility_nft(
                agent_uuid,
                work_unit_id,
                "completion",
                {
                    "completion_time": current_time + 1000,
                    "outputs": json.loads(test_work_unit["outputs"]),
                    "quality_metrics": {
                        "accuracy": 0.95,
                        "latency": 850
                    }
                }
            )
            logger.success(f"Successfully minted completion NFT with ID {nft_id}")
        except Exception as e:
            logger.error(f"Test 2 failed: {str(e)}")

def test_work_unit_status_transitions(atr_testnet):
    """Test work unit status transitions and validation."""
    console.rule("[bold blue]Starting Work Unit Status Transition Tests[/bold blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Register test agent
        task = progress.add_task("Registering test agent...", total=None)
        agent_type = "TestAgent"
        capabilities = ["test_capability"]
        agent_uuid = atr_testnet.register_agent_onchain(agent_type, capabilities)
        progress.update(task, completed=True)
        logger.success(f"Registered agent with UUID {agent_uuid}")
        
        # Create work unit
        task = progress.add_task("Creating test work unit...", total=None)
        current_time = int(datetime.now().timestamp())
        test_work_unit = {
            "name": "Status Test Work Unit",
            "description": "Test work unit for status transitions",
            "version": "1.0",
            "inputs": json.dumps(["input1"]),
            "outputs": json.dumps(["output1"]),
            "requirements": json.dumps({
                "agent_type": "TestAgent",
                "capabilities": ["test_capability"]
            }),
            "temporal_constraints": json.dumps({
                "start_time": current_time - 1800,
                "deadline": current_time + 1800,
                "max_duration": 3600
            }),
            "data_access_controls": json.dumps({
                "allowed_sources": ["source1"],
                "allowed_queries": ["query1"]
            }),
            "payment_token": "0x0000000000000000000000000000000000000000",
            "payment_amount": Web3.to_wei(1, 'ether')
        }
        
        work_unit_id = atr_testnet.define_work_unit_onchain(**test_work_unit)
        progress.update(task, completed=True)
        
        # Test valid status transitions
        console.print("\n[bold cyan]Test 1: Valid status transitions[/bold cyan]")
        try:
            # Create status transition table
            table = Table(title="Status Transitions")
            table.add_column("From Status", style="cyan")
            table.add_column("To Status", style="green")
            table.add_column("Result", style="magenta")
            
            # CREATED -> ASSIGNED
            atr_testnet.assign_agent_to_work_unit_onchain(work_unit_id, agent_uuid)
            table.add_row("CREATED", "ASSIGNED", "✅ Success")
            
            # ASSIGNED -> IN_PROGRESS
            atr_testnet.update_work_unit_status_onchain(
                work_unit_id,
                "IN_PROGRESS",
                {"start_time": current_time}
            )
            table.add_row("ASSIGNED", "IN_PROGRESS", "✅ Success")
            
            # IN_PROGRESS -> COMPLETED
            atr_testnet.update_work_unit_status_onchain(
                work_unit_id,
                "COMPLETED",
                {
                    "completion_time": current_time + 1000,
                    "outputs": json.loads(test_work_unit["outputs"])
                }
            )
            table.add_row("IN_PROGRESS", "COMPLETED", "✅ Success")
            
            console.print(table)
            
        except Exception as e:
            logger.error(f"Test 1 failed: {str(e)}")
        
        # Test invalid status transitions
        console.print("\n[bold cyan]Test 2: Invalid status transitions[/bold cyan]")
        try:
            # Create new work unit for invalid transition test
            work_unit_id = atr_testnet.define_work_unit_onchain(**test_work_unit)
            
            # Create invalid transitions table
            table = Table(title="Invalid Status Transitions")
            table.add_column("From Status", style="cyan")
            table.add_column("To Status", style="red")
            table.add_column("Expected Result", style="magenta")
            table.add_column("Actual Result", style="green")
            
            # Try CREATED -> COMPLETED (invalid)
            try:
                atr_testnet.update_work_unit_status_onchain(
                    work_unit_id,
                    "COMPLETED",
                    {"completion_time": current_time}
                )
                table.add_row("CREATED", "COMPLETED", "Should Fail", "❌ Unexpectedly Succeeded")
            except Exception as e:
                if "Invalid status transition" in str(e):
                    table.add_row("CREATED", "COMPLETED", "Should Fail", "✅ Failed as Expected")
                else:
                    table.add_row("CREATED", "COMPLETED", "Should Fail", f"❌ Failed Unexpectedly: {str(e)}")
            
            console.print(table)
            
        except Exception as e:
            logger.error(f"Test 2 failed: {str(e)}")

def main():
    """Run all Phase 3 diagnostic tests."""
    console.print(Panel.fit(
        "[bold blue]Starting Phase 3 Diagnostic Tests[/bold blue]",
        border_style="blue"
    ))
    
    try:
        atr_testnet = ATRTestnet()
        test_responsibility_nft(atr_testnet)
        test_work_unit_status_transitions(atr_testnet)
        
        console.print(Panel.fit(
            "[bold green]✨ Phase 3 Diagnostic Tests Completed Successfully[/bold green]",
            border_style="green"
        ))
        
    except Exception as e:
        logger.exception("Phase 3 diagnostics failed")
        console.print(Panel.fit(
            f"[bold red]❌ Phase 3 Diagnostic Tests Failed: {str(e)}[/bold red]",
            border_style="red"
        ))
        raise

if __name__ == "__main__":
    main() 