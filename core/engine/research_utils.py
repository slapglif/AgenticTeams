"""Research utilities for executing and analyzing research operations."""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime, UTC
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from rich.console import Console
from rich.table import Table
from loguru import logger

from core.shared.types import MemoryInterface
from core.shared.settings import build_llm
from core.engine.research_prompts import (
    DATA_ANALYSIS_PROMPT,
    DOMAIN_SEARCH_QUERY_PROMPT,
    SYSTEM_ROLES
)
from core.schemas.pydantic_schemas import (
    SearchQuery,
    DataAnalysisResponse,
    GraphAnalysisResponse,
    CitationAnalysisResponse,
    NetworkAnalysisResponse
)

async def execute_analysis(
    memory_manager: MemoryInterface,
    prompt: str,
    system_role: str,
    tool_type: str,
    input_data: Dict[str, Any],
    table_configs: List[str],
    empty_result_fn: Any,
    context_key: str
) -> Dict[str, Any]:
    """Execute analysis chain with given context."""
    try:
        # Build chain
        llm = build_llm(temperature=0.2)
        prompt_template = ChatPromptTemplate.from_template(prompt)
        
        # Select appropriate Pydantic model based on tool type
        model_map = {
            "data_analysis": DataAnalysisResponse,
            "graph_analysis": GraphAnalysisResponse,
            "citation_analysis": CitationAnalysisResponse,
            "network_analysis": NetworkAnalysisResponse
        }
        
        parser = PydanticOutputParser(pydantic_object=model_map[tool_type])
        chain = prompt_template | llm | parser
        
        # Get input context
        context = input_data.get(context_key, {})
        
        # Execute chain
        result = await chain.ainvoke({
            "context": context,
            "system_role": system_role,
            "table_configs": table_configs
        })
        
        if not result:
            return empty_result_fn()
            
        # Store result
        await store_analysis_result(
            memory_manager=memory_manager,
            result=result.model_dump(),
            context={"tool_type": tool_type, "input": input_data}
        )
        
        return result.model_dump()
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

async def execute_analysis_chain(
    prompt: str,
    system_role: str,
    input_data: Dict[str, Any],
    memories: List[Any],
    tool_type: str
) -> Dict[str, Any]:
    """Execute analysis chain with given context and configuration."""
    try:
        # Build chain
        llm = build_llm(temperature=0.2)
        prompt_template = ChatPromptTemplate.from_template(prompt)
        
        # Select appropriate Pydantic model based on tool type
        model_map = {
            "data_analysis": DataAnalysisResponse,
            "graph_analysis": GraphAnalysisResponse,
            "citation_analysis": CitationAnalysisResponse,
            "network_analysis": NetworkAnalysisResponse
        }
        
        parser = PydanticOutputParser(pydantic_object=model_map[tool_type])
        chain = prompt_template | llm | parser
        
        # Execute chain
        result = await chain.ainvoke({
            "input": input_data,
            "system_role": system_role,
            "memories": memories
        })
        
        return result.model_dump()
        
    except Exception as e:
        logger.error(f"Analysis chain failed: {str(e)}")
        raise

def create_result_tables(results: List[Dict[str, Any]]) -> List[Table]:
    """Create formatted tables from results."""
    tables = []
    
    for result in results:
        # Create table
        table = Table(
            title=result.get("title", "Analysis Result"),
            show_header=True,
            header_style="bold magenta"
        )
        
        # Add columns
        for col in result.get("columns", []):
            table.add_column(col["name"], style=col.get("style", ""))
            
        # Add rows
        for row in result.get("rows", []):
            table.add_row(*[str(cell) for cell in row])
            
        tables.append(table)
        
    return tables

async def store_analysis_result(
    memory_manager: MemoryInterface,
    result: Dict[str, Any],
    context: Dict[str, Any]
) -> None:
    """Store analysis result in memory."""
    try:
        # Create metadata
        metadata = {
            "type": "analysis_result",
            "timestamp": datetime.now(UTC).isoformat(),
            "context": context
        }
        
        # Store result
        await memory_manager.store_tool_result(
            operation_id=context.get("operation_id", "unknown"),
            tool_name=context.get("tool_name", "analysis"),
            result=result,
            error=None
        )
        
    except Exception as e:
        logger.error(f"Failed to store analysis result: {str(e)}")
        raise 