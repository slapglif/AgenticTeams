"""Run end-to-end test for agentic task execution."""

import asyncio
from tests.test_e2e import test_research_task
from rich.console import Console
from rich.panel import Panel

console = Console()

async def main():
    """Run the end-to-end test."""
    try:
        console.print(Panel("Starting End-to-End Test", style="cyan"))
        await test_research_task()
        console.print(Panel("Test Completed Successfully", style="green"))
    except Exception as e:
        console.print(Panel(f"Test Failed: {str(e)}", style="red"))
        raise

if __name__ == "__main__":
    asyncio.run(main()) 