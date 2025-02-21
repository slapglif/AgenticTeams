"""Research tools module for executing various research-related tools."""

import logging
from datetime import datetime, UTC
from typing import Dict, Any, List, Optional, Union
import json
import discord
from discord import Embed, Color, Thread
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.utilities import SearxSearchWrapper
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
import asyncio
import aiohttp
from langchain.tools import StructuredTool
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from loguru import logger
from pydantic import BaseModel, Field

from core.shared.types import MemoryInterface
from core.shared.settings import build_llm, SEARX_HOST
from core.shared.document_processor import DocumentProcessor
from core.shared.models import (
    SearchSchema,
    AspectSchema,
    ContextSchema,
    SearchQuery,
    DataAnalysisResponse,
    GraphAnalysisResponse,
    CitationAnalysisResponse,
    NetworkAnalysisResponse,
    CitationInput
)
from core.engine.research_prompts import (
    DOMAIN_SEARCH_QUERY_PROMPT,
    DATA_ANALYSIS_PROMPT,
    GRAPH_ANALYSIS_PROMPT,
    CITATION_ANALYSIS_PROMPT,
    NETWORK_ANALYSIS_PROMPT,
    NETWORK_STRUCTURE_PROMPT,
    SYSTEM_ROLES
)
from core.engine.research_fallbacks import (
    create_empty_search_result,
    create_empty_data_analysis_result,
    create_empty_graph_result,
    create_empty_citation_result,
    create_empty_network_result,
    create_empty_network_structure_result,
    create_error_response,
    create_formatted_result,
    DISCORD_LIMITS
)
from core.engine.research_utils import (
    execute_analysis,
    execute_analysis_chain,
    create_result_tables,
    store_analysis_result
)

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)
console = Console()

class AnalysisInput(BaseModel):
    """Input schema for analysis tools."""
    context: str
    implementation_notes: str = "Analyze the provided data."
    memories: List[Dict[str, Any]] = []

class ResearchTools:
    """Handles execution of research tools and search functionality.
    
    Attributes:
        logger: Logger instance for this class
        memory_manager: Memory manager instance for storing and retrieving research data
        session: Optional aiohttp client session for making HTTP requests
        searx_search: SearxSearchWrapper instance for web searches
        doc_processor: DocumentProcessor instance for processing text documents
        discord_interface: Optional Discord interface for sending messages
        TOOLS: Dictionary mapping tool IDs to tool names
        tool_map: Dictionary mapping tool names to StructuredTool instances
    """
    
    def __init__(self, memory_manager: MemoryInterface) -> None:
        """Initialize research tools.
        
        Args:
            memory_manager: Memory manager for storing results
        """
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.memory_manager = memory_manager
        self.document_processor = DocumentProcessor()
        self._session: Optional[aiohttp.ClientSession] = None
        self.discord_interface = None
        
        # Initialize searx search
        try:
            self.searx_search: SearxSearchWrapper = SearxSearchWrapper(searx_host=SEARX_HOST)
            logger.info("Searx Search initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Searx search: {e}")
        
        # Tool definitions
        self.TOOLS: Dict[int, str] = {
            1: "Search Tool",
            2: "Data Analysis Tool", 
            3: "Graph Analysis Tool",
            4: "Citation Analysis Tool",
            5: "Network Analysis Tool"
        }
        
        # Tool map
        self.tool_map: Dict[str, StructuredTool] = {
            "search": StructuredTool(
                name="search",
                description="Execute a search query and return results",
                coroutine=self.search,
                args_schema=SearchSchema
            ),
            "generate_domain_search_query": StructuredTool(
                name="generate_domain_search_query",
                description="Generate domain-specific search queries",
                coroutine=self.generate_domain_search_query,
                args_schema=AspectSchema
            ),
            "analyze_data": StructuredTool(
                name="analyze_data",
                description="Analyze data from the research context",
                coroutine=self.analyze_data,
                args_schema=AnalysisInput
            ),
            "analyze_graph": StructuredTool(
                name="analyze_graph",
                description="Analyze graph structure from the research context",
                coroutine=self.analyze_graph,
                args_schema=AnalysisInput
            ),
            "analyze_citations": StructuredTool(
                name="analyze_citations",
                description="Analyze citations from the research context",
                coroutine=self.analyze_citations,
                args_schema=CitationInput
            ),
            "analyze_network": StructuredTool(
                name="analyze_network",
                description="Analyze network from the research context",
                coroutine=self.analyze_network,
                args_schema=AnalysisInput
            )
        }
        
    async def execute_tool(self, tool_id: int, context: Union[str, Dict[str, Any]], tool_type: Optional[str] = None) -> Dict[str, Any]:
        """Execute a research tool.
        
        Args:
            tool_id: ID of the tool to execute
            context: Input context for the tool (can be string or dictionary)
            tool_type: Optional tool type identifier
            
        Returns:
            Tool execution results
        """
        try:
            tool_type_str = str(tool_type) if tool_type is not None else "unknown"
            
            # Helper function to create AnalysisInput
            def create_analysis_input(ctx: Union[str, Dict[str, Any]]) -> AnalysisInput:
                if isinstance(ctx, dict):
                    return AnalysisInput(
                        context=ctx.get("context", json.dumps(ctx)),
                        implementation_notes=ctx.get("implementation_notes", ""),
                        memories=ctx.get("memories", [])
                    )
                return AnalysisInput(
                    context=ctx,
                    implementation_notes="",
                    memories=[]
                )
            
            # Helper function to create CitationInput
            def create_citation_input(ctx: Union[str, Dict[str, Any]]) -> CitationInput:
                if isinstance(ctx, dict):
                    return CitationInput(
                        citations=ctx.get("citations", []),
                        query=ctx.get("query"),
                        reasoning=ctx.get("reasoning"),
                        implementation_notes=ctx.get("implementation_notes"),
                        metadata=ctx.get("metadata", {})
                    )
                return CitationInput(
                    citations=[],
                    query=None,
                    reasoning=None,
                    implementation_notes=None,
                    metadata={}
                )
            
            if tool_id == 1:  # Search
                return await self.search(context)
            elif tool_id == 2:  # Data Analysis
                return await self.analyze_data(create_analysis_input(context))
            elif tool_id == 3:  # Graph Analysis  
                return await self.analyze_graph(create_analysis_input(context))
            elif tool_id == 4:  # Citation Analysis
                return await self.analyze_citations(create_citation_input(context))
            elif tool_id == 5:  # Network Analysis
                return await self.analyze_network(create_analysis_input(context))
            else:
                return create_error_response(f"Unknown tool ID: {tool_id}", tool_type_str)
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}")
            return create_error_response(str(e), tool_type_str)

    async def generate_domain_search_query(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Generate domain-specific search queries.
        
        Args:
            input: Input data containing aspect to generate queries for
            
        Returns:
            Generated search queries and metadata
        """
        try:
            # Create chain
            chain = ChatPromptTemplate.from_template(DOMAIN_SEARCH_QUERY_PROMPT) | build_llm() | PydanticOutputParser(pydantic_object=SearchQuery)
            
            # Generate query
            result = await chain.ainvoke(input)
            
            # Store result
            await self.memory_manager.store_tool_result(
                operation_id="generate_query",
                tool_name="generate_domain_search_query",
                result=result.model_dump(),
                error=None
            )
            
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Query generation failed: {str(e)}")
            return create_empty_search_result()

    async def search(self, input: Union[str, Dict[str, Any]], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Execute a search query.
        
        Args:
            input: Search query string or dictionary with query and config
            config: Optional configuration parameters
            
        Returns:
            Search results and metadata
        """
        try:
            # Extract query from input
            query = input if isinstance(input, str) else input.get("query", "")
            if isinstance(input, dict) and "reasoning" in input:
                logger.info(f"Executing search: {input['reasoning']}")
            else:
                logger.info(f"Executing search: {query}")
            
            # Execute search
            raw_results = await asyncio.to_thread(self.searx_search.run, query)
            logger.info(f"Got {len(raw_results)} raw results")
            
            # Process results
            processed_results = []
            if isinstance(raw_results, list):
                for result in raw_results:
                    if isinstance(result, dict):
                        processed_result = {
                            "title": result.get("title", ""),
                            "url": result.get("link", ""),
                            "snippet": result.get("snippet", ""),
                            "source": result.get("source", ""),
                            "relevance_score": result.get("score", 0.0)
                        }
                    elif isinstance(result, str):
                        # Handle string results as complete entries
                        processed_result = {
                            "title": result,
                            "url": "",
                            "snippet": result,
                            "source": "",
                            "relevance_score": 0.0
                        }
                    else:
                        logger.warning(f"Skipping invalid result type: {type(result)}")
                        continue
                    processed_results.append(processed_result)
            elif isinstance(raw_results, str):
                # Handle single string result
                processed_results.append({
                    "title": raw_results,
                    "url": "",
                    "snippet": raw_results,
                    "source": "",
                    "relevance_score": 0.0
                })
            
            # Create response
            response = {
                "query": query,
                "results": processed_results,
                "metadata": {
                    "total_results": len(processed_results),
                    "timestamp": datetime.now(UTC).isoformat()
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return create_empty_search_result()

    async def analyze_data(self, input: AnalysisInput) -> Dict[str, Any]:
        """Analyze input data using the data analysis chain."""
        try:
            chain = ChatPromptTemplate.from_template(DATA_ANALYSIS_PROMPT) | build_llm() | PydanticOutputParser(pydantic_object=DataAnalysisResponse)
            
            # Convert input to dict and add defaults if needed
            if isinstance(input, dict):
                input_dict = {
                    "implementation_notes": input.get("implementation_notes", ""),
                    "memories": input.get("memories", [])
                }
            else:
                input_dict = {
                    "implementation_notes": input.implementation_notes if hasattr(input, "implementation_notes") else "",
                    "memories": input.memories if hasattr(input, "memories") else []
                }
            
            # Execute analysis
            result = await chain.ainvoke(input_dict)
            
            # Format result to match DataAnalysisResponse schema
            formatted_result = {
                "key_metrics": result.key_metrics if hasattr(result, 'key_metrics') else {"relevance_score": 0.0},
                "correlations": result.correlations if hasattr(result, 'correlations') else [],
                "significant_findings": result.significant_findings if hasattr(result, 'significant_findings') else [],
                "analysis": result.analysis if hasattr(result, 'analysis') else {},
                "insights": result.insights if hasattr(result, 'insights') else [],
                "confidence": result.confidence if hasattr(result, 'confidence') else 0.5,
                "metadata": result.metadata if hasattr(result, 'metadata') else {}
            }
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"Data analysis failed: {str(e)}")
            return {
                "key_metrics": {"relevance_score": 0.0},
                "correlations": [],
                "significant_findings": [],
                "analysis": {"error": str(e)},
                "insights": [],
                "confidence": 0.0,
                "metadata": {}
            }

    async def analyze_graph(self, input: AnalysisInput) -> Dict[str, Any]:
        """Analyze graph structure using the graph analysis chain."""
        try:
            chain = ChatPromptTemplate.from_template(GRAPH_ANALYSIS_PROMPT) | build_llm() | PydanticOutputParser(pydantic_object=GraphAnalysisResponse)
            
            # Convert input to dict and add defaults if needed
            if isinstance(input, dict):
                input_dict = {
                    "context": input.get("context", ""),
                    "implementation_notes": input.get("implementation_notes", ""),
                    "memories": input.get("memories", []),
                    "topology": {
                        "type": "citation_network",
                        "description": "Network of research paper citations and their relationships",
                        "key_properties": ["citation_count", "publication_date", "impact_factor"]
                    }
                }
            else:
                input_dict = {
                    "context": input.context if hasattr(input, "context") else "",
                    "implementation_notes": input.implementation_notes if hasattr(input, "implementation_notes") else "",
                    "memories": input.memories if hasattr(input, "memories") else [],
                    "topology": {
                        "type": "citation_network",
                        "description": "Network of research paper citations and their relationships",
                        "key_properties": ["citation_count", "publication_date", "impact_factor"]
                    }
                }
            
            # Execute analysis
            result = await chain.ainvoke(input_dict)
            
            # Store result
            await self.memory_manager.store_tool_result(
                operation_id="graph_analysis",
                tool_name="analyze_graph",
                result=result.model_dump() if hasattr(result, "model_dump") else result,
                error=None
            )
            
            return result.model_dump() if hasattr(result, "model_dump") else result
            
        except Exception as e:
            logger.error(f"Graph analysis failed: {str(e)}")
            return create_empty_graph_result()

    async def analyze_citations(self, input: CitationInput) -> Dict[str, Any]:
        """Analyze citations using the citation analysis chain.
        
        Args:
            input: Input data containing research context and citations
            
        Returns:
            Citation analysis results
        """
        try:
            # Create chain
            chain = ChatPromptTemplate.from_template(CITATION_ANALYSIS_PROMPT) | build_llm() | PydanticOutputParser(pydantic_object=CitationAnalysisResponse)
            
            # Convert input to dict and add defaults if needed
            if isinstance(input, dict):
                input_dict = {
                    "research_context": input.get("implementation_notes", ""),
                    "citations": input.get("citations", [])
                }
            else:
                input_dict = {
                    "research_context": input.implementation_notes if hasattr(input, "implementation_notes") else "",
                    "citations": input.citations if hasattr(input, "citations") else []
                }
            
            # Execute analysis
            result = await chain.ainvoke(input_dict)
            
            # Store result
            await self.memory_manager.store_tool_result(
                operation_id="citation_analysis",
                tool_name="analyze_citations",
                result=result.model_dump() if hasattr(result, "model_dump") else result,
                error=None
            )
            
            return result.model_dump() if hasattr(result, "model_dump") else result
            
        except Exception as e:
            logger.error(f"Citation analysis failed: {str(e)}")
            return create_empty_citation_result()

    async def analyze_network(self, input: AnalysisInput) -> Dict[str, Any]:
        """Analyze network structure and relationships.
        
        Args:
            input: Input data containing context for network analysis
            
        Returns:
            Network analysis results
        """
        try:
            # Create chain with explicit JSON output format
            chain = ChatPromptTemplate.from_template(NETWORK_ANALYSIS_PROMPT) | build_llm() | PydanticOutputParser(pydantic_object=NetworkAnalysisResponse)
            
            # Convert input to dict and add defaults if needed
            if isinstance(input, dict):
                input_dict = {
                    "context": input.get("context", ""),
                    "implementation_notes": input.get("implementation_notes", ""),
                    "memories": input.get("memories", [])
                }
            else:
                input_dict = {
                    "context": input.context if hasattr(input, "context") else "",
                    "implementation_notes": input.implementation_notes if hasattr(input, "implementation_notes") else "",
                    "memories": input.memories if hasattr(input, "memories") else []
                }
            
            # Execute analysis
            result = await chain.ainvoke(input_dict)
            
            # Store result
            await self.memory_manager.store_tool_result(
                operation_id="network_analysis",
                tool_name="analyze_network",
                result=result.model_dump(),
                error=None
            )
            
            logger.info(f"Network analysis completed successfully")
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Network analysis failed: {str(e)}")
            return create_empty_network_result()

    async def execute_tools(self, tools: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a list of research tools.
        
        Args:
            tools: List of tool IDs to execute
            context: Context data for the tools
            
        Returns:
            Dictionary mapping tool IDs to their results
        """
        try:
            results = {}
            for tool in tools:
                try:
                    result = await self.execute_tool(int(tool), context, str(tool))
                    await store_analysis_result(
                        memory_manager=self.memory_manager,
                        result=result,
                        context={"tool_type": str(tool), "input": context}
                    )
                    results[tool] = result
                except Exception as e:
                    logger.error(f"Error executing tool {tool}: {e}")
                    results[tool] = create_error_response(str(e), str(tool))
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing tools: {e}")
            return create_error_response(str(e), "tools")

    async def __aenter__(self) -> "ResearchTools":
        """Enter async context manager."""
        self._session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> None:
        """Exit async context manager."""
        if self._session:
            await self._session.close()
            self._session = None
            
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp client session.
        
        Returns:
            An active aiohttp ClientSession instance. If no session exists,
            a new one is created.
        """
        if not self._session:
            self._session = aiohttp.ClientSession()
        return self._session
        
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._session:
            await self._session.close()
            self._session = None

    def get_tool(self, tool_name: str) -> Optional[StructuredTool]:
        """Get a tool function by name.
        
        Args:
            tool_name: Name of the tool to get
            
        Returns:
            The StructuredTool instance if found, None otherwise
        """
        return self.tool_map.get(tool_name)