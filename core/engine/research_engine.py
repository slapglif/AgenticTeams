"""
Research Engine Module - Handles research execution and tools
"""
import json
import asyncio
import traceback
from typing import List, Dict, Any, Optional, Tuple, cast, Union
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from langchain_community.utilities import SearxSearchWrapper
from langchain.prompts import ChatPromptTemplate, SystemMessage, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from core.shared.settings import SEARX_HOST, build_llm
from core.schemas.json_schemas import (
    tool_data_analysis_schema,
    tool_graph_analysis_schema,
    tool_citation_analysis_schema,
    tool_network_analysis_schema,
    meta_review_schema,
    revised_response_schema
)
from datetime import datetime, timedelta, timezone, UTC
from discord import Embed, Color
import re
from core.memory.memory_manager import MemoryManager
from core.shared.data_processor import IntelligentDataProcessor
from core.engine.research_tools import ResearchTools, AnalysisInput, CitationInput
from core.engine.research_analysis import ResearchAnalysis
from core.engine.research_utils import create_result_tables
import logging
import os
import sys
from rich.table import Table
from core.compile.utils import create_chain
from core.shared.types import MemoryInterface
from core.engine.research_prompts import (
    RESEARCH_PROMPT,
    ANALYSIS_PROMPT,
    REPLAN_PROMPT,
    SYSTEM_ROLES,
    GRAPH_ANALYSIS_PROMPT
)
from core.schemas.pydantic_schemas import (
    SearchQuery,
    DataAnalysisResponse,
    GraphAnalysisResponse,
    CitationAnalysisResponse,
    NetworkAnalysisResponse
)
from langchain_core.messages import BaseMessage
from core.shared.errors import ResearchError
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

console = Console()

class ResearchEngine:
    """Research engine for executing research tasks."""

    # Tool schemas for validation
    TOOL_SCHEMAS = {
        "data_analysis": {
            "type": "object",
            "properties": {
                "implementation_notes": {"type": "string"},
                "memories": {"type": "array"}
            }
        },
        "graph_analysis": {
            "type": "object",
            "properties": {
                "implementation_notes": {"type": "string"},
                "memories": {"type": "array"}
            }
        },
        "citation_analysis": {
            "type": "object",
            "properties": {
                "implementation_notes": {"type": "string"},
                "citations": {"type": "array"}
            }
        },
        "network_analysis": {
            "type": "object",
            "properties": {
                "implementation_notes": {"type": "string"},
                "memories": {"type": "array"}
            }
        }
    }

    def __init__(self, memory_manager: MemoryInterface):
        """Initialize the research engine."""
        self.memory_manager = memory_manager
        self.searx = SearxSearchWrapper(searx_host=SEARX_HOST)
        self.research_tools = ResearchTools(memory_manager)
        self.TOOL_IDS = {
            "search": "1",
            "analyze": "2",
            "graph": "3",
            "citations": "4",
            "network": "5"
        }
        self.logger = logger
        
        # Configure logging to only show important info
        logger.remove()  # Remove default handler
        logger.add(sys.stderr, level="INFO",
                  format="<green>{time:HH:mm:ss}</green> | {level} | <cyan>{message}</cyan>")
        
        logger.info("Initializing Research Engine...")
        
        self.research_analysis = ResearchAnalysis()
        
        # Discord interface will be set by the main application
        self.discord_interface = None
        
        logger.info("Research Engine initialized successfully")
    
    def set_discord_interface(self, interface):
        """Set the Discord interface for all components."""
        self.discord_interface = interface
        self.research_tools.discord_interface = interface

    
    async def _generate_research_focus(self, topic: str) -> Dict[str, Any]:
        """Generate research focus using the analysis component."""
        return await self.research_analysis._generate_research_focus(topic)
    
    async def _format_research_focus(self, research_focus: Dict[str, Any], thread) -> None:
        """Format and send research focus using the analysis component."""
        await self.research_analysis.format_for_discord(research_focus, thread)
    
    async def execute_tools(self, tool_ids: List[int], implementation_notes: str) -> Dict[str, Any]:
        """Execute multiple research tools."""
        try:
            results = {}
            context = {"implementation_notes": implementation_notes}
            
            for tool_id in tool_ids:
                try:
                    # Convert tool_id to int if needed
                    tool_id = int(tool_id)
                    
                    # Execute tool
                    result = await self._execute_single_tool(tool_id, context)
                    
                    # If result has synthesis, ensure it's preserved
                    if isinstance(result, dict):
                        if "synthesis" in result:
                            results[str(tool_id)] = result
                        else:
                            # Create synthesis from results if possible
                            synthesis = {
                                "sources": [],
                                "key_insights": []
                            }
                            if "results" in result:
                                if isinstance(result["results"], list):
                                    synthesis["sources"] = [
                                        {
                                            "title": r.get("title", ""),
                                            "url": r.get("url", ""),
                                            "relevance_score": 0.8
                                        }
                                        for r in result["results"]
                                        if isinstance(r, dict)
                                    ][:10]
                                    synthesis["key_insights"] = [
                                        r.get("snippet", "")
                                        for r in result["results"]
                                        if isinstance(r, dict) and r.get("snippet")
                                    ][:5]
                            results[str(tool_id)] = {
                                "results": result.get("results", {}),
                                "metadata": result.get("metadata", {}),
                                "synthesis": synthesis
                            }
                    else:
                        # Store result with empty synthesis
                        results[str(tool_id)] = {
                            "results": result,
                            "metadata": {},
                            "synthesis": {
                                "sources": [],
                                "key_insights": []
                            }
                        }
                    
                except ValueError:
                    logger.error(f"Invalid tool ID format: {tool_id}")
                    results[str(tool_id)] = {"error": f"Invalid tool ID format: {tool_id}"}
                except Exception as e:
                    logger.error(f"Error executing tool {tool_id}: {e}", exc_info=True)
                    results[str(tool_id)] = {"error": str(e)}
            
            return results
            
        except Exception as e:
            logger.error(f"Error executing tools: {e}", exc_info=True)
            return {"error": str(e)}
    

    async def _validate_tool_result(self, tool_id: int, result: Any) -> Dict[str, Any]:
        """Validate and format tool result."""
        try:
            if not result:
                return {"error": f"Tool {tool_id} returned no result"}
                
            # If result is already a dict with error, return it
            if isinstance(result, dict) and "error" in result:
                return result
                
            # If result is not a dict, wrap it
            if not isinstance(result, dict):
                result = {"output": result}
                
            # Ensure result has required fields while preserving synthesis
            if "output" not in result and "error" not in result and "synthesis" not in result:
                result = {"output": result}
                
            return result
            
        except Exception as e:
            logger.error(f"Error validating tool result: {e}", exc_info=True)
            return {"error": f"Error validating tool result: {str(e)}"}

    async def _summarize_tool_learnings(self, tool_name: str, result: Dict[str, Any], implementation_notes: str, thread) -> dict | None:
        """Summarize learnings from tool execution and their relevance to objectives."""
        try:
            # Truncate inputs if too large
            if len(implementation_notes) > 3800:
                implementation_notes = implementation_notes[:3800] + "..."
            
            # Ensure result is not too large
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, str) and len(value) > 3800:
                        result[key] = value[:3800] + "..."
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, str) and len(subvalue) > 3800:
                                result[key][subkey] = subvalue[:3800] + "..."

            # If result has synthesis field, use it directly
            if isinstance(result, dict) and "synthesis" in result:
                summary = {
                    "sources": result["synthesis"].get("sources", []),
                    "key_insights": result["synthesis"].get("key_insights", []),
                    "relevance_to_objective": "Results directly relevant to research objective",
                    "confidence_level": 8,
                    "suggested_next_steps": ["Continue with next research step"]
                }
            else:
                # Otherwise generate summary using chain
                summary_chain = create_chain(
                    prompt=ChatPromptTemplate.from_template(
                        """Analyze the tool results and provide a concise summary of learnings and their relevance.
                    
                    Tool: {tool_name}
                    Result: {result}
                    Original Objective: {objective}
                        
                    Provide a JSON response with:
                    - sources: Array of source objects with title, url, and relevance_score
                    - key_insights: Array of main insights discovered (max 5)
                    - relevance_to_objective: How these learnings contribute to or detract from the original objective
                    - confidence_level: How confident we are in these findings (1-10)
                    - suggested_next_steps: What should be investigated next based on these findings (max 3)
                    """,
                    llm=build_llm(),
                    output_mode='json'
                ))
                
                summary = await summary_chain.ainvoke({
                    "tool_name": tool_name,
                    "result": json.dumps(result, indent=2)[:3800],  # Ensure not too large
                    "objective": implementation_notes
                })
            
            # Format and send summary to Discord
            if thread and self.discord_interface:
                # Limit the size of each section
                summary_text = [f"📚 **Learnings from {tool_name}**\n\n**Key Insights:**\n"]
                
                # Add insights in chunks
                for idx, learning in enumerate(summary.get("key_insights", [])[:5], 1):
                    chunk = f"{idx}. {learning}\n"
                    if len(summary_text[-1]) + len(chunk) > 1900:
                        summary_text.append(chunk)
                    else:
                        summary_text[-1] += chunk
                
                # Add relevance in new chunk if needed
                relevance = f"\n**Relevance to Objective:**\n{summary.get('relevance_to_objective')}\n"
                if len(summary_text[-1]) + len(relevance) > 1900:
                    summary_text.append(relevance)
                else:
                    summary_text[-1] += relevance
                
                # Add confidence level
                confidence = f"\n**Confidence Level:** {summary.get('confidence_level')}/10\n"
                if len(summary_text[-1]) + len(confidence) > 1900:
                    summary_text.append(confidence)
                else:
                    summary_text[-1] += confidence
                
                # Add next steps if present
                if summary.get("suggested_next_steps"):
                    next_steps = "\n**Suggested Next Steps:**\n"
                    for step in summary.get("suggested_next_steps", [])[:3]:
                        next_steps += f"- {step}\n"
                    
                    if len(summary_text[-1]) + len(next_steps) > 1900:
                        summary_text.append(next_steps)
                    else:
                        summary_text[-1] += next_steps
                
                # Send each chunk
                for chunk in summary_text:
                    await thread.send(chunk)
                    await asyncio.sleep(0.5)  # Rate limiting protection
                
        except Exception as e:
            logger.error(f"Error summarizing learnings for {tool_name}: {e}")
            if thread and self.discord_interface:
                await thread.send(f"⚠️ Error generating learning summary for {tool_name}")
            return None

        return summary

    async def _execute_tool_with_context(self, tool_number: str, implementation_notes: str) -> Dict[str, Any]:
        """Execute a tool with context."""
        # Convert tool_number to string if it's an integer
        tool_number_str = str(tool_number)
        tool_name = next((name for name, id_ in self.TOOL_IDS.items() if id_ == tool_number_str), "")
        
        if not tool_name:
            raise ValueError(f"Invalid tool number: {tool_number}")
            
        tool_input = {"implementation_notes": implementation_notes}
        return await self._execute_tool(tool_name, tool_input)

    async def _execute_search_with_queries(self, search_queries: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search with provided queries."""
        search_results = []
        for query in search_queries.get('queries', []):
            query_str = query.get('query', '')
            if query_str:
                try:
                    results = await self.searx.aresults(query_str, num_results=10)
                    if results:
                        search_results.extend(results)
                except Exception as e:
                    logger.error(f"Error executing search query '{query_str}': {e}")
        
        if search_results:
            return {
                "results": [
                    {
                        "title": r.get('title', 'No Title'),
                        "link": r.get('link', 'No Link'),
                        "snippet": r.get('snippet', 'No Snippet')
                    }
                    for r in search_results
                ]
            }
        return {"error": "No search results found"}

    async def _execute_single_tool(self, tool_id: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single research tool."""
        try:
            # Convert tool_id to string for lookup
            tool_id_str = str(tool_id)
            tool_name = next((name for name, id_ in self.TOOL_IDS.items() if id_ == tool_id_str), None)
            
            if not tool_name:
                raise ValueError(f"Invalid tool ID: {tool_id}")
                
            logger.info(f"Executing {tool_name}...")
            
            # Execute tool
            result = await self.research_tools.execute_tool(tool_id, context)
            
            # Validate result
            validated_result = await self._validate_tool_result(tool_id, result)
            
            # Ensure success field is propagated
            if isinstance(validated_result, dict):
                if "success" not in validated_result and "error" not in validated_result:
                    validated_result["success"] = True
                elif "error" in validated_result:
                    validated_result["success"] = False
            
            return validated_result
            
        except Exception as e:
            error_msg = f"Error executing {tool_name if tool_name else f'tool {tool_id}'}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg, "success": False}

    async def execute_data_analysis_tool(self, implementation_notes: str) -> Dict[str, Any]:
        """Execute data analysis using LLM."""
        logger.info("Creating data analysis chain...")
        analysis_chain = create_chain(
            prompt=ChatPromptTemplate.from_template(
                """You are a data analysis expert. Analyze the following research request and provide detailed statistical insights:

            Research Context:
            {implementation_notes}
            
            Consider:
            1. What are the key metrics and variables to analyze?
            2. What statistical patterns might be relevant?
            3. What correlations should we look for?
            4. What anomalies might be significant?
            5. What actionable insights can we derive?

            Provide a comprehensive analysis that includes:
            - Basic statistical measures and their significance
            - Important correlations between key variables
            - Meaningful patterns in the data
            - Notable anomalies or outliers
            - Data-driven recommendations
            
            Your response should be a JSON object with these fields:
            - basic_statistics: Object containing key statistical measures
            - correlations: Array of correlation objects (variables, strength, significance)
            - patterns: Array of pattern objects (pattern_type, description, confidence)
            - anomalies: Array of anomaly objects (type, description, severity)
            - recommendations: Array of recommendation objects (action, rationale, priority)
            
            Focus on providing actual analysis results, not schema definitions.
            Make sure your analysis is specific to the research context and provides actionable insights.
            """,
            output_mode='json'
        ))

        logger.info("Starting data analysis inference...")
        try:
            result = await analysis_chain.ainvoke({
                "implementation_notes": implementation_notes
            })
            logger.info("Data analysis inference completed successfully")
            
            # Ensure we have actual content, not just schema
            if not result.get('basic_statistics') or isinstance(result.get('basic_statistics'), dict) and 'type' in result.get('basic_statistics'):
                logger.warning("Data analysis returned schema instead of results, generating default analysis")
                result = {
                    "basic_statistics": {
                        "summary": "Analysis of key metrics in the research context",
                        "measures": []
                    },
                    "correlations": [],
                    "patterns": [],
                    "anomalies": [],
                    "recommendations": []
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Data analysis inference failed: {str(e)}")
            return {
                "error": f"Data analysis failed: {str(e)}",
                "basic_statistics": {"error": "Analysis failed"},
                "correlations": [],
                "patterns": [],
                "anomalies": [],
                "recommendations": []
            }

    async def execute_graph_tool(self, implementation_notes: str) -> Dict[str, Any]:
        """Generate network graphs and visualizations."""
        logger.info("Creating graph analysis chain...")
        graph_chain = create_chain(
            prompt=ChatPromptTemplate.from_template(
                """You are a graph theory and visualization expert. Analyze the following research request and provide detailed graph insights:

            Research Context:
            {implementation_notes}
            
            Consider:
            1. What type of graph structure would best represent this data?
            2. What are the key nodes and their relationships?
            3. What metrics would be most informative?
            4. How should the visualization be structured?
            5. What insights can we derive from the graph structure?

            Create a comprehensive graph analysis that includes:
            - Node identification and classification
            - Edge relationships and weights
            - Graph metrics and their significance
            - Visualization recommendations
            
            Format your response as a JSON object matching this schema:
            {{{schema}}}
            
            Make sure your analysis is specific to the research context and provides meaningful graph-based insights.
            """,
            output_mode='json'
        ))

        logger.info("Starting graph analysis inference...")
        console.print("[cyan]Running graph analysis model...[/cyan]")
        try:
            result = await graph_chain.ainvoke({
                "implementation_notes": implementation_notes,
                "schema": json.dumps(self.TOOL_SCHEMAS["graph_analysis"], indent=2)
            })
            logger.info("Graph analysis inference completed successfully")
            return result
        except Exception as e:
            logger.error(f"Graph analysis inference failed: {str(e)}")
            console.print("[red]Graph analysis inference failed[/red]")
            raise

    async def execute_citation_tool(self, implementation_notes: str) -> Dict[str, Any]:
        """Analyze citation patterns and validity."""
        logger.info("Creating citation analysis chain...")
        citation_chain = create_chain(
            prompt=ChatPromptTemplate.from_template(
                """You are a citation analysis and bibliometrics expert. Analyze the following research context and provide detailed citation insights:

            Research Context:
            {implementation_notes}
            
            Consider:
            1. What are the key sources and their credibility?
            2. How do citations connect different research areas?
            3. What citation patterns emerge?
            4. What is the impact of key citations?
            5. What gaps or opportunities exist in the citation network?

            Provide a comprehensive citation analysis that includes:
            - Source evaluation and credibility assessment
            - Citation pattern identification
            - Impact analysis
            - Citation-based recommendations
            
            Format your response as a JSON object with these fields:
            - citations: Array of citation objects with source, validity, credibility_score, and impact_factor
            - patterns: Array of pattern objects with pattern_type and description
            - recommendations: Array of recommendation objects with suggestion and rationale
            - key_insights: Array of strings containing important findings
            
            Make sure your analysis is specific to the research context and provides meaningful citation-based insights.
            """,
            output_mode='json',
            output_parser=PydanticOutputParser(pydantic_object=CitationAnalysisResponse)
        ))

        logger.info("Starting citation analysis inference...")
        console.print("[cyan]Running citation analysis model...[/cyan]")
        try:
            result = await citation_chain.ainvoke({
                "implementation_notes": implementation_notes
            })
            
            logger.info("Citation analysis inference completed")
            # Ensure the result matches the expected schema
            if not isinstance(result, dict):
                logger.warning("Citation analysis returned non-dict result, converting to empty dict")
                result = {}
            
            # Format result with synthesis field
            formatted_result = {
                "citations": result.get("citations", []),
                "patterns": result.get("patterns", []),
                "recommendations": result.get("recommendations", []),
                "synthesis": {
                    "sources": [
                        {
                            "title": citation.get("source", ""),
                            "url": None,
                            "relevance_score": citation.get("credibility_score", 0.0)
                        }
                        for citation in result.get("citations", [])
                    ],
                    "key_insights": result.get("key_insights", [])
                }
            }
            logger.info("Citation analysis results formatted successfully")
            return formatted_result
            
        except Exception as e:
            logger.error(f"Citation analysis inference failed: {str(e)}")
            console.print("[red]Citation analysis inference failed[/red]")
            return {
                "citations": [],
                "patterns": [{"pattern_type": "error", "description": f"Citation analysis failed: {str(e)}"}],
                "recommendations": [],
                "synthesis": {
                    "sources": [],
                    "key_insights": []
                }
            }

    async def execute_network_tool(self, implementation_notes: str) -> Dict[str, Any]:
        """Execute network analysis on research data."""
        logger.info("Creating network analysis chain...")
        network_chain = create_chain(
            prompt=ChatPromptTemplate.from_template(
                """You are a network analysis and complex systems expert. Analyze the following research context and provide detailed network insights:

            Research Context:
            {implementation_notes}
            
            Consider:
            1. What is the overall network topology?
            2. What are the key nodes and their roles?
            3. How do different parts of the network interact?
            4. Where are the bottlenecks and critical paths?
            5. How resilient is the network?
            6. What flow patterns emerge?
            7. What subnetworks or communities exist?

            Provide a comprehensive network analysis that includes:
            - Network structure and topology
            - Node and connection analysis
            - Network metrics and their significance
            - Bottleneck identification
            - Flow dynamics
            - Resilience assessment
            - Subnetwork identification
            
            Format your response as a JSON object matching this schema:
            {{{schema}}}
            
            Make sure your analysis is specific to the research context and provides meaningful network-based insights.
            """,
            output_mode='json'
        ))

        logger.info("Starting network analysis inference...")
        console.print("[cyan]Running network analysis model...[/cyan]")
        try:
            result = await network_chain.ainvoke({
                "implementation_notes": implementation_notes,
                "schema": json.dumps(self.TOOL_SCHEMAS["network_analysis"], indent=2)
            })
            logger.info("Network analysis inference completed successfully")
            return result
        except Exception as e:
            logger.error(f"Network analysis inference failed: {str(e)}")
            console.print("[red]Network analysis inference failed[/red]")
            raise

    async def _send_search_results(self, search_result: Dict[str, Any], thread) -> None:
        """Send search results to Discord with proper chunking and formatting."""
        if not thread or not self.discord_interface:
            return
            
        try:
            results = search_result.get("results", [])
            if not results:
                await thread.send("⚠️ No search results found")
                return

            # Create summary embed
            summary_embed = Embed(
                title="🔍 Research Sources Overview",
                description="Curated sources based on relevance and credibility",
                color=Color.blue()
            )
            
            # Group results by type/domain
            academic_sources = []
            technical_sources = []
            other_sources = []
            
            for result in results:
                url = result.get('link', '').lower()
                if any(domain in url for domain in ['.edu', '.gov', 'scholar.google', 'arxiv.org', 'pubmed', 'sciencedirect']):
                    academic_sources.append(result)
                elif any(domain in url for domain in ['github.com', 'stackoverflow.com', 'ieee.org', 'acm.org']):
                    technical_sources.append(result)
                else:
                    other_sources.append(result)
            
            # Add source type summaries
            if academic_sources:
                summary_embed.add_field(
                    name="📚 Academic Sources",
                    value=f"Found {len(academic_sources)} academic/research sources",
                    inline=False
                )
            
            if technical_sources:
                summary_embed.add_field(
                    name="💻 Technical Sources",
                    value=f"Found {len(technical_sources)} technical references",
                    inline=False
                )
            
            if other_sources:
                summary_embed.add_field(
                    name="🌐 Additional Sources",
                    value=f"Found {len(other_sources)} other relevant sources",
                    inline=False
                )
            
            await thread.send(embed=summary_embed)
            
            # Send detailed results in categorized embeds
            for category, sources, emoji in [
                ("Academic Sources", academic_sources, "📚"),
                ("Technical Sources", technical_sources, "💻"),
                ("Additional Sources", other_sources, "🌐")
            ]:
                if not sources:
                    continue
                    
                embed = Embed(
                    title=f"{emoji} {category}",
                    color=Color.blue()
                )
                
                for idx, result in enumerate(sources, 1):
                    title = result.get('title', 'No Title')
                    snippet = result.get('snippet', 'No description available')
                    link = result.get('link', 'No Link')
                    
                    # Truncate long titles/snippets
                    if len(title) > 100:
                        title = title[:97] + "..."
                    if len(snippet) > 200:
                        snippet = snippet[:197] + "..."
                    
                    embed.add_field(
                        name=f"{idx}. {title}",
                        value=f"{snippet}\n[View Source]({link})",
                        inline=False
                    )
                
                await thread.send(embed=embed)
            
            # Store search results in memory with metadata
            await self.memory_manager.store_tool_result(
                operation_id="search_results",
                tool_name="search",
                result={
                    "results": results,
                    "metadata": {
                        "academic_count": len(academic_sources),
                        "technical_count": len(technical_sources),
                        "other_count": len(other_sources),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
            )
                
        except Exception as e:
            logger.error(f"Error sending search results to Discord: {e}")
            await thread.send("⚠️ Error displaying search results")

    async def _validate_and_complete_plan(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and complete research plan steps with missing required fields."""
        try:
            completed_plan = []
            for step in plan:
                # Ensure all required fields are present
                completed_step = {
                    "step_number": step.get("step_number", len(completed_plan) + 1),
                    "action": step.get("action", "Research step"),
                    "agent": step.get("agent", [2]),  # Default to Chemoinformatics Agent
                    "reasoning": step.get("reasoning", "Step required for research completion"),
                    "completion_conditions": step.get("completion_conditions", "Step outputs meet quality standards"),
                    "tool_suggestions": step.get("tool_suggestions", [1]),  # Default to search tool
                    "implementation_notes": step.get("implementation_notes", "Execute step according to research methodology")
                }
                
                # Convert agent to list if it's not already
                if not isinstance(completed_step["agent"], list):
                    completed_step["agent"] = [completed_step["agent"]]
                
                # Convert agent IDs to integers
                completed_step["agent"] = [
                    int(agent) if isinstance(agent, (str, float)) else agent 
                    for agent in completed_step["agent"]
                ]
                
                # Ensure tool_suggestions is a list and convert all values to integers
                if not isinstance(completed_step["tool_suggestions"], list):
                    completed_step["tool_suggestions"] = [1]  # Default to search tool
                else:
                    try:
                        # Convert each tool suggestion to integer, handling strings and floats
                        completed_step["tool_suggestions"] = [
                            int(str(tool).split(':')[0]) if isinstance(tool, str) else int(tool)
                            for tool in completed_step["tool_suggestions"]
                            if tool is not None  # Skip None values
                        ]
                        
                        # Validate that all tools exist
                        valid_tools = set(self.TOOL_IDS.keys())
                        completed_step["tool_suggestions"] = [
                            tool for tool in completed_step["tool_suggestions"]
                            if tool in valid_tools
                        ]
                        
                        # If no valid tools remain, default to search tool
                        if not completed_step["tool_suggestions"]:
                            completed_step["tool_suggestions"] = [1]
                    except (ValueError, TypeError, AttributeError):
                        # If any conversion fails, default to search tool
                        completed_step["tool_suggestions"] = [1]
                
                completed_plan.append(completed_step)
            
            return completed_plan
            
        except Exception as e:
            logger.error(f"Error validating research plan: {e}")
            # Return a safe default plan
            return [{
                "step_number": 1,
                "action": "Initial research",
                "agent": [2],
                "reasoning": "Begin research process",
                "completion_conditions": "Initial research complete",
                "tool_suggestions": [1],
                "implementation_notes": "Conduct initial research phase"
            }]

    async def execute_research_plan(self, research_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a research plan."""
        try:
            plan_id = research_plan.get("plan_id")
            plan = research_plan.get("plan", {})
            steps = plan.get("steps", [])
            completed_steps = set()
            step_results = {}
            
            # Create a rich table for the research plan
            plan_table = Table(title=f"Research Plan {plan_id}", show_header=True, header_style="bold magenta")
            plan_table.add_column("Step ID", style="cyan")
            plan_table.add_column("Description", style="green")
            plan_table.add_column("Tool", style="yellow")
            plan_table.add_column("Dependencies", style="blue")
            
            for step in steps:
                plan_table.add_row(
                    step.get("id", ""),
                    step.get("description", ""),
                    step.get("tool_name", ""),
                    ", ".join(step.get("dependencies", []))
                )
            
            console.print(plan_table)
            logger.info(f"Executing research plan {plan_id}")
            
            # Execute each step
            for step in steps:
                step_id = step.get("id")
                tool_name = step.get("tool_name")
                dependencies = step.get("dependencies", [])
                
                # Create execution table
                exec_table = Table(title=f"Executing Step: {step_id}", show_header=True, header_style="bold cyan")
                exec_table.add_column("Field", style="blue")
                exec_table.add_column("Value", style="green")
                exec_table.add_row("Tool", tool_name)
                exec_table.add_row("Description", step.get("description", ""))
                exec_table.add_row("Dependencies", ", ".join(dependencies))
                console.print(exec_table)
                
                # Check dependencies
                if not all(dep in completed_steps for dep in dependencies):
                    logger.warning(f"Skipping step {step_id} - dependencies not met: {dependencies}")
                    continue
                
                # Execute tool
                logger.info(f"Executing tool {tool_name} for step {step_id}")
                
                # Get tool number from TOOL_IDS
                tool_number = None
                if tool_name:
                    tool_number = self.TOOL_IDS.get(tool_name)
                
                if not tool_number:
                    logger.error(f"Invalid tool name: {tool_name}")
                    continue
                
                tool_result = await self._execute_tool_with_context(
                    tool_number=tool_number,
                    implementation_notes=step.get("description", "")
                )
                
                # Create results table
                results_table = Table(title=f"Results for Step: {step_id}", show_header=True, header_style="bold green")
                results_table.add_column("Component", style="blue")
                results_table.add_column("Content", style="green", overflow="fold")
                
                if isinstance(tool_result, dict):
                    if "synthesis" in tool_result:
                        results_table.add_row("Sources", str(len(tool_result["synthesis"].get("sources", []))))
                        results_table.add_row("Key Insights", "\n".join(tool_result["synthesis"].get("key_insights", [])[:3]))
                    if "error" in tool_result:
                        results_table.add_row("Error", tool_result["error"])
                    if "summary" in tool_result:
                        results_table.add_row("Summary", str(tool_result["summary"]))
                
                console.print(results_table)
                
                # Store result
                if tool_result:
                    step_results[step_id] = tool_result
                    completed_steps.add(step_id)
                    logger.info(f"Completed steps: {completed_steps}")
            
            # Return final result with synthesis from the last step
            if step_results:
                last_step_id = list(step_results.keys())[-1]
                last_result = step_results[last_step_id]
                
                if isinstance(last_result, dict):
                    # If last_result has synthesis, use it directly
                    if "synthesis" in last_result:
                        final_result = {
                            "success": True,
                            "synthesis": last_result["synthesis"],
                            "summary": last_result.get("summary", "No summary available"),
                            "key_insights": last_result.get("key_insights", []),
                            "patterns": last_result.get("patterns", []),
                            "recommendations": last_result.get("recommendations", [])
                        }
                    # If last_result has results, try to create synthesis from it
                    elif "results" in last_result:
                        synthesis = {
                            "sources": [],
                            "key_insights": []
                        }
                        if isinstance(last_result["results"], list):
                            synthesis["sources"] = [
                                {
                                    "title": r.get("title", ""),
                                    "url": r.get("url", ""),
                                    "relevance_score": 0.8
                                }
                                for r in last_result["results"]
                                if isinstance(r, dict)
                            ][:10]
                            synthesis["key_insights"] = [
                                r.get("snippet", "")
                                for r in last_result["results"]
                                if isinstance(r, dict) and r.get("snippet")
                            ][:5]
                        final_result = {
                            "success": True,
                            "synthesis": synthesis,
                            "summary": last_result.get("summary", "No summary available"),
                            "key_insights": last_result.get("key_insights", []),
                            "patterns": last_result.get("patterns", []),
                            "recommendations": last_result.get("recommendations", [])
                        }
                else:
                    # Default empty result
                    final_result = {
                        "success": True,
                        "synthesis": {
                            "sources": [],
                            "key_insights": []
                        },
                        "summary": "No summary available",
                        "key_insights": [],
                        "patterns": [],
                        "recommendations": []
                    }
            
            # Generate research document and store synthesis
            task_id = research_plan.get("task_id", datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
            doc_path = self.generate_research_document(final_result, task_id)
            await self._store_synthesis(task_id, final_result.get("synthesis", {}))
            
            # Add document path to final result
            final_result["document_path"] = doc_path
            
            # Create final summary table
            summary_table = Table(title="Research Plan Execution Summary", show_header=True, header_style="bold blue")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="green")
            summary_table.add_row("Total Steps", str(len(steps)))
            summary_table.add_row("Completed Steps", str(len(completed_steps)))
            summary_table.add_row("Success", str(final_result["success"]))
            summary_table.add_row("Sources Found", str(len(final_result["synthesis"]["sources"])))
            summary_table.add_row("Key Insights", str(len(final_result["synthesis"]["key_insights"])))
            summary_table.add_row("Document Path", doc_path)
            console.print(summary_table)
            
            return final_result
            
        except Exception as e:
            error_msg = f"Error executing research plan: {str(e)}"
            logger.error(error_msg)
            console.print(f"[red]Error:[/red] {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }

    def _truncate_list(self, items: List[Any], max_items: int, max_length: int) -> List[Any]:
        """Helper method to truncate list items."""
        if not items:
            return []
            
        truncated = []
        for item in items[:max_items]:
            if isinstance(item, str):
                truncated.append(item[:max_length])
            elif isinstance(item, dict):
                truncated_dict = {}
                for k, v in item.items():
                    if isinstance(v, str):
                        truncated_dict[k] = v[:max_length]
                    else:
                        truncated_dict[k] = v
                truncated.append(truncated_dict)
            else:
                truncated.append(item)
        return truncated 

    async def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a research tool."""
        try:
            if tool_name == "search":
                results = await self.searx.arun(tool_input["query"])
                return {"results": results}
            elif tool_name == "analyze":
                return await self.research_tools.execute_tool(2, tool_input, "data_analysis")
            elif tool_name == "graph":
                return await self.research_tools.execute_tool(3, tool_input, "graph_analysis")
            elif tool_name == "citations":
                return await self.research_tools.execute_tool(4, tool_input, "citation_analysis")
            elif tool_name == "network":
                return await self.research_tools.execute_tool(5, tool_input, "network_analysis")
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            raise ResearchError(f"Failed to execute tool {tool_name}") from e

    async def execute_research(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a research task."""
        try:
            # Create chain
            chain = ChatPromptTemplate.from_template(RESEARCH_PROMPT) | build_llm() | PydanticOutputParser(pydantic_object=SearchQuery)
            
            # Execute research
            result = await chain.ainvoke(task)
            
            # Store result
            await self.memory_manager.store_tool_result(
                operation_id="research",
                tool_name="execute_research",
                result=result.model_dump(),
                error=None
            )
            
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Research execution failed: {str(e)}")
            raise

    async def analyze_research(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze research results."""
        try:
            # Create chain
            chain = ChatPromptTemplate.from_template(ANALYSIS_PROMPT) | build_llm() | PydanticOutputParser(pydantic_object=DataAnalysisResponse)
            
            # Ensure research_data has the correct format
            formatted_data = {
                "research_data": {
                    "task": research_data.get("task", ""),
                    "results": research_data.get("results", []),
                    "metrics": research_data.get("metrics", {}),
                    "analysis": research_data.get("analysis", {})
                }
            }
            
            # Execute analysis
            result = await chain.ainvoke(formatted_data)
            
            # Store result with proper format
            await self.memory_manager.store_tool_result(
                operation_id="analysis",
                tool_name="analyze_research",
                result=result.model_dump(),
                error=None
            )
            
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Research analysis failed: {str(e)}")
            raise 


    async def _get_memories(self, query: str, limit: int = 5) -> Optional[Dict[str, Any]]:
        """Get relevant memories for a query."""
        try:
            memories = await self.memory_manager.get_memory(query)
            if not memories:
                return None
            
            # Convert list to dictionary format
            return {
                "memories": memories[:limit],
                "total_count": len(memories),
                "query": query
            }
        except Exception as e:
            logger.error(f"Failed to get memories: {str(e)}")
            return None

    async def _store_result(self, operation_id: str, tool_name: str, result: Any, error: Optional[str] = None) -> None:
        """Store a tool result in memory."""
        content = {
            "operation_id": operation_id,
            "tool_name": tool_name,
            "result": result,
            "error": error,
            "timestamp": datetime.now(UTC).isoformat()
        }
        await self.memory_manager.store_memory(operation_id, "tool_result", content)

    async def _store_synthesis(self, task_id: str, synthesis: Dict[str, Any]) -> None:
        """Store research synthesis in memory."""
        content = {
            "synthesis": synthesis,
            "timestamp": datetime.now(UTC).isoformat()
        }
        await self.memory_manager.store_memory(task_id, "synthesis", content)

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

    async def _replan_research(self, task: Dict[str, Any], original_plan: Dict[str, Any], execution_results: Dict[str, Any], tool_descriptions: Dict[str, Any]) -> Dict[str, Any]:
        """Create a revised research plan based on current state and results."""
        try:
            # Create replan chain
            replan_chain = ChatPromptTemplate.from_messages([
                SystemMessage(content=SYSTEM_ROLES["research"]),
                HumanMessagePromptTemplate.from_template(REPLAN_PROMPT)
            ])

            # Ensure all required variables are properly formatted
            formatted_task = task.get("description", "") if isinstance(task, dict) else str(task)
            formatted_original_plan = json.dumps(original_plan, indent=2) if original_plan else "{}"
            formatted_execution_results = json.dumps(execution_results, indent=2) if execution_results else "{}"
            formatted_tool_descriptions = json.dumps(tool_descriptions, indent=2) if tool_descriptions else "{}"

            # Execute replan chain with properly formatted variables
            replan_result = await self.execute_chain(
                replan_chain,
                {
                    "task": formatted_task,
                    "original_plan": formatted_original_plan,
                    "execution_results": formatted_execution_results,
                    "tool_descriptions": formatted_tool_descriptions
                }
            )

            # Convert BaseMessage to Dict
            if isinstance(replan_result, BaseMessage):
                try:
                    content = replan_result.content
                    
                    if isinstance(content, str):
                        result: Dict[str, Any] = json.loads(content)
                    elif isinstance(content, dict):
                        result = dict(content)
                    elif isinstance(content, list):
                        result = {"content": content}
                    else:
                        result = {"content": str(content)}

                    # Ensure feedback field is present with required structure
                    feedback = {
                        "analysis": {
                            "issues": [
                                {
                                    "type": "replanning",
                                    "description": "Replanning required based on execution results",
                                    "impact": "High",
                                    "affected_steps": ["all"]
                                }
                            ],
                            "successful_parts": [],
                            "metrics_analysis": {
                                "key_findings": ["Execution results indicate need for replanning"],
                                "areas_for_improvement": ["Plan structure and execution flow"]
                            }
                        },
                        "revised_plan": {
                            "steps": result.get("operations", []),
                            "estimated_improvements": [
                                "Enhanced plan structure",
                                "Better error handling",
                                "Improved execution flow"
                            ]
                        },
                        "recommendations": [
                            {
                                "focus": "Plan Validation",
                                "rationale": "Ensure plan steps are properly validated before execution",
                                "priority": "High"
                            }
                        ]
                    }

                    # Add feedback to result
                    result["feedback"] = feedback
                    return result

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse replan result: {e}")
                    raise ResearchError(f"Failed to parse replan result: {e}")
            else:
                raise ResearchError("Unexpected replan result type")

        except Exception as e:
            logger.error(f"Error in replanning: {e}")
            raise ResearchError(f"Error in replanning: {e}")

    async def execute_chain(self, chain: ChatPromptTemplate, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a chain with the given inputs."""
        try:
            # Create chain with LLM
            full_chain = chain | build_llm()
            
            # Execute chain
            result = await full_chain.ainvoke(inputs)
            
            # Handle BaseMessage result
            if isinstance(result, BaseMessage):
                try:
                    content = result.content
                    if isinstance(content, str):
                        return json.loads(content)
                    elif isinstance(content, (list, dict)):
                        return {"content": content}
                    else:
                        return {"content": str(content)}
                except json.JSONDecodeError:
                    return {"content": str(result.content)}
            
            return result if isinstance(result, dict) else {"content": str(result)}
            
        except Exception as e:
            logger.error(f"Error executing chain: {e}")
            raise

    def generate_research_document(self, final_result: Dict[str, Any], task_id: str) -> str:
        """Generate a comprehensive markdown research document.
        
        Args:
            final_result: The final research results
            task_id: The ID of the research task
            
        Returns:
            Markdown formatted research document path or error message
        """
        try:
            # Extract components
            synthesis = final_result.get("synthesis", {})
            sources = synthesis.get("sources", [])
            key_insights = synthesis.get("key_insights", [])
            patterns = final_result.get("patterns", [])
            recommendations = final_result.get("recommendations", [])
            
            # Generate document sections
            sections = []
            
            # Title and Introduction
            sections.append(f"# Research Report: {task_id}\n")
            sections.append("## Introduction\n")
            sections.append(final_result.get("summary", "No summary available"))
            sections.append("\n")
            
            # Key Insights
            sections.append("## Key Insights\n")
            for idx, insight in enumerate(key_insights, 1):
                sections.append(f"{idx}. {insight}\n")
            sections.append("\n")
            
            # Patterns and Findings
            sections.append("## Patterns and Findings\n")
            for pattern in patterns:
                sections.append(f"### {pattern.get('pattern_type', 'Pattern')}\n")
                sections.append(pattern.get('description', 'No description available'))
                sections.append("\n")
            sections.append("\n")
            
            # Recommendations
            sections.append("## Recommendations\n")
            for rec in recommendations:
                sections.append(f"- **{rec.get('suggestion', 'Recommendation')}**\n")
                sections.append(f"  {rec.get('rationale', 'No rationale provided')}\n")
            sections.append("\n")
            
            # Sources and References
            sections.append("## Sources and References\n")
            for idx, source in enumerate(sources, 1):
                title = source.get('title', 'Untitled')
                url = source.get('url', '#')
                sections.append(f"{idx}. [{title}]({url})\n")
            sections.append("\n")
            
            # Join sections and write to file
            doc_content = "\n".join(sections)
            output_path = f"research_outputs/{task_id}_report.md"
            os.makedirs("research_outputs", exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(doc_content)
            
            return output_path
            
        except Exception as e:
            error_msg = f"Error generating research document: {e}"
            logger.error(error_msg)
            return f"research_outputs/error_{task_id}_report.txt"

    async def _join_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Join research results into a final synthesis."""
        try:
            # Create synthesis structure
            synthesis = {
                "sources": [],
                "key_insights": [],
                "patterns": [],
                "recommendations": []
            }
            
            # Process each result
            for result in results:
                if isinstance(result, dict):
                    # Add sources
                    if "sources" in result:
                        synthesis["sources"].extend(result["sources"])
                    
                    # Add key insights
                    if "key_insights" in result:
                        synthesis["key_insights"].extend(result["key_insights"])
                    
                    # Add patterns
                    if "patterns" in result:
                        synthesis["patterns"].extend(result["patterns"])
                    
                    # Add recommendations
                    if "recommendations" in result:
                        synthesis["recommendations"].extend(result["recommendations"])
            
            # Deduplicate and limit entries
            synthesis["sources"] = list({s["url"]: s for s in synthesis["sources"]}.values())[:10]
            synthesis["key_insights"] = list(set(synthesis["key_insights"]))[:5]
            synthesis["patterns"] = list({p["pattern"]: p for p in synthesis["patterns"]}.values())[:5]
            synthesis["recommendations"] = list(set(synthesis["recommendations"]))[:5]
            
            return {
                "success": True,
                "synthesis": synthesis,
                "summary": "Research synthesis completed successfully",
                "document_path": f"research_outputs/synthesis_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.md"
            }
            
        except Exception as e:
            logger.error(f"Failed to join results: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "synthesis": {
                    "sources": [],
                    "key_insights": [],
                    "patterns": [],
                    "recommendations": []
                }
            }

   