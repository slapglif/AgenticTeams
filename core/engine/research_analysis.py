"""Research analysis module for generating research focus and meta-review functionality."""

import json
import logging
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, UTC
from discord import Embed, Color
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import discord

from core.memory.memory_manager import MemoryManager
from core.shared.settings import build_llm
from core.shared.data_processor import IntelligentDataProcessor
from core.shared.message_formatter import MessageFormatter
from core.compile.utils import create_chain

logger = logging.getLogger(__name__)

class ResearchAnalysis:
    """Handles research focus generation and meta-review functionality.
    
    Attributes:
        memory_manager: Memory manager for storing and retrieving research data
        data_processor: Data processor for handling research data
        message_formatter: Message formatter for Discord output
        logger: Logger instance for this class
        llm: Language model instance
        focus_schema: JSON schema for research focus output
        analysis_prompts: Dictionary of prompts for different analysis types
    """
    
    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        data_processor: Optional[IntelligentDataProcessor] = None,
        message_formatter: Optional[MessageFormatter] = None
    ) -> None:
        """Initialize the research analysis component.
        
        Args:
            memory_manager: Optional memory manager instance
            data_processor: Optional data processor instance
            message_formatter: Optional message formatter instance
        """
        self.memory_manager = memory_manager or MemoryManager()
        self.data_processor = data_processor or IntelligentDataProcessor()
        self.message_formatter = message_formatter or MessageFormatter()
        self.logger = logging.getLogger(__name__)
        self.llm = build_llm()
        
        # Define schemas
        self.focus_schema = {
            "type": "object",
            "properties": {
                "focus_areas": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "area": {"type": "string"},
                            "description": {"type": "string"},
                            "requirements": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["area", "description", "requirements", "dependencies"]
                    }
                },
                "technical_scope": {
                    "type": "object",
                    "properties": {
                        "complexity": {"type": "integer", "minimum": 1, "maximum": 10},
                        "breadth": {"type": "integer", "minimum": 1, "maximum": 10},
                        "depth": {"type": "integer", "minimum": 1, "maximum": 10},
                        "timeline_estimate": {"type": "string"}
                    },
                    "required": ["complexity", "breadth", "depth", "timeline_estimate"]
                },
                "methodology": {
                    "type": "object",
                    "properties": {
                        "approach": {"type": "string"},
                        "tools": {"type": "array", "items": {"type": "string"}},
                        "validation": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["approach", "tools", "validation"]
                }
            },
            "required": ["focus_areas", "technical_scope", "methodology"]
        }
        
        # Define analysis prompts
        self.analysis_prompts = {
            "focus": """Generate a research focus for this topic.
            
            Topic: {topic}
            Current Time: {current_time}
            Related Memories: {memories}
            Previous Focus Areas: {focus_areas}
            
            Consider:
            1. Key research areas
            2. Technical requirements
            3. Integration opportunities
            
            Return a JSON object with:
            {{
                "focus_areas": [
                    {{
                        "area": str,  // Research area name
                        "description": str,  // Detailed description
                        "requirements": [str],  // Technical requirements
                        "dependencies": [str]  // Related areas or prerequisites
                    }}
                ],
                "technical_scope": {{
                    "complexity": int,  // 1-10 score
                    "breadth": int,  // 1-10 score
                    "depth": int,  // 1-10 score
                    "timeline_estimate": str
                }},
                "methodology": {{
                    "approach": str,
                    "tools": [str],
                    "validation": [str]
                }}
            }}
            """,
            
            "meta_review": """Perform a meta-review of these research findings.
            
            Findings: {findings}
            Context: {context}
            Previous Reviews: {previous_reviews}
            
            Consider:
            1. Quality and completeness
            2. Technical validity
            3. Integration potential
            4. Future directions
            
            Return a JSON object with:
            {{
                "quality_assessment": {{
                    "completeness": float,  // 0-1 score
                    "technical_depth": float,  // 0-1 score
                    "innovation": float  // 0-1 score
                }},
                "key_insights": [
                    {{
                        "insight": str,
                        "supporting_evidence": [str],
                        "implications": [str]
                    }}
                ],
                "integration_opportunities": [
                    {{
                        "area": str,
                        "potential": str,
                        "requirements": [str]
                    }}
                ],
                "recommendations": [
                    {{
                        "action": str,
                        "rationale": str,
                        "priority": int  // 1-3
                    }}
                ]
            }}
            """,
            
            "synthesis": """Synthesize these research components into a cohesive analysis.
            
            Components: {components}
            Context: {context}
            Objectives: {objectives}
            
            Return a JSON synthesis with:
            {{
                "integrated_findings": [
                    {{
                        "finding": str,
                        "supporting_components": [str],
                        "confidence": float,  // 0-1 score
                        "implications": [str]
                    }}
                ],
                "system_patterns": [
                    {{
                        "pattern": str,
                        "evidence": [str],
                        "significance": str,
                        "applications": [str]
                    }}
                ],
                "strategic_insights": [
                    {{
                        "insight": str,
                        "rationale": str,
                        "action_items": [str],
                        "priority": int  // 1-3
                    }}
                ],
                "synthesis_quality": {{
                    "completeness": float,  // 0-1 score
                    "coherence": float,  // 0-1 score
                    "actionability": float  // 0-1 score
                }}
            }}
            """
        }
        
    async def _generate_research_focus(self, topic: str) -> Dict[str, Any]:
        """Generate research focus using LLM."""
        focus_chain = create_chain(
            prompt=ChatPromptTemplate.from_template(
                """Analyze the research topic and generate a focused research plan.
            
            Topic: {topic}
            
            Consider:
            1. What are the key aspects to investigate?
            2. What specific questions need answers?
            3. What methods would be most effective?
            4. What potential challenges exist?
            
            Provide a JSON response with:
            - aspects: Array of key aspects to investigate
            - questions: Array of specific research questions
            - methods: Array of suggested research methods
            - challenges: Array of potential challenges
            """,
            output_mode='json'
        )
        )
        
        return await focus_chain.ainvoke({"topic": topic})

    async def _meta_review_result(self, findings: List[dict], result: Dict[str, Any], tool_name: str, thread=None) -> Dict[str, Any]:
        """Perform meta-review of tool results."""
        review_chain = create_chain(
            prompt=ChatPromptTemplate.from_template(
                """Review the tool results and provide a meta-analysis.
            
            Tool: {tool_name}
            Result: {result}
            
            Provide a JSON response with:
            - key_findings: Array of main findings
            - validity: Assessment of result validity (1-10)
            - limitations: Array of result limitations
            - next_steps: Array of suggested next steps
            """,
            output_mode='json'
         )
    
        )
        return await review_chain.ainvoke({
            "tool_name": tool_name,
            "result": json.dumps(result),
            "findings": json.dumps(findings)
        })

    async def generate_research_focus(self, topic: str) -> Dict[str, Any]:
        """Generate research focus areas and methodology for a given topic.
        
        Args:
            topic: The research topic to analyze
            
        Returns:
            Dictionary containing focus areas, methodology, and impact assessment
            
        Raises:
            Exception: If focus generation fails
        """
        try:
            # Generate research focus using LLM
            result = await self._generate_research_focus(topic)
            
            # Validate result has required fields
            if not all(key in result for key in ["focus_areas", "methodology", "impact_assessment"]):
                raise ValueError("Missing required fields in research focus generation")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating research focus: {str(e)}")
            return {
                "focus_areas": [{
                    "area": "General research",
                    "rationale": "Default focus area due to error",
                    "priority": 1
                }],
                "methodology": {
                    "approach": "Standard research methodology",
                    "key_methods": ["Literature review", "Data analysis"],
                    "validation_strategy": "Peer review"
                },
                "impact_assessment": {
                    "potential_impact": "To be determined",
                    "success_metrics": ["Completion of research objectives"],
                    "risk_factors": ["Technical limitations"]
                }
            }

    async def perform_meta_review(
        self,
        findings: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform a meta-review of research findings.
        
        Args:
            findings: List of research findings to review
            context: Context information for the review
            
        Returns:
            Dictionary containing quality assessment, insights, and recommendations
            
        Raises:
            Exception: If meta-review fails
        """
        try:
            # Perform meta-review of tool results
            result = await self._meta_review_result(findings, context, "Research Analysis")
            
            # Validate result has required fields
            if not all(key in result for key in ["quality_assessment", "key_insights", "recommendations"]):
                raise ValueError("Missing required fields in meta-review")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error performing meta-review: {str(e)}")
            return {
                "quality_assessment": {
                    "methodology_score": 0.5,
                    "data_quality_score": 0.5,
                    "reliability_score": 0.5,
                    "limitations": ["Error in meta-review process"]
                },
                "key_insights": [{
                    "insight": "Unable to generate insights due to error",
                    "confidence": 0.5,
                    "supporting_evidence": ["Error occurred during analysis"]
                }],
                "recommendations": [{
                    "recommendation": "Review and retry analysis",
                    "priority": 1,
                    "rationale": "Error in meta-review process"
                }]
            }

    async def synthesize_components(
        self,
        components: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        objectives: Optional[List[str]] = None
    ):
        """Synthesize research components into a cohesive analysis.
        
        Args:
            components: List of research components to synthesize
            context: Optional context information
            objectives: Optional list of research objectives
            
        Returns:
            Dictionary containing integrated findings, patterns, and insights
            
        Raises:
            Exception: If synthesis fails
        """
  
            # Create synthesis chain
        chain = create_chain(
            prompt=ChatPromptTemplate.from_template(
                """Synthesize these research components into a cohesive analysis.
            
            Components: {components}
            Context: {context}
            Objectives: {objectives}
            
            Return a JSON synthesis with:
            {{
                "integrated_findings": [{{
                    "finding": str,
                    "supporting_components": [str],
                    "confidence": float,  // 0-1 score
                    "implications": [str]
                }}],
                "system_patterns": [{{
                    "pattern": str,
                    "evidence": [str],
                    "significance": str,
                    "applications": [str]
                }}],
                "strategic_insights": [{{
                    "insight": str,
                    "rationale": str,
                    "action_items": [str],
                    "priority": int  // 1-3
                }}],
                "synthesis_quality": {{
                    "completeness": float,  // 0-1 score
                    "coherence": float,  // 0-1 score
                    "actionability": float  // 0-1 score
                }}
            }}
            """,
            output_mode="json"
        ))
        
            # Generate synthesis
        synthesis = await chain.ainvoke({
            "components": json.dumps(components),
            "context": json.dumps(context) if context else "{}",
            "objectives": json.dumps(objectives) if objectives else "[]"
        })
        return synthesis
            
        
    async def format_for_discord(
        self,
        content: Dict[str, Any],
        content_type: str = "analysis"
    ) -> List[discord.Embed]:
        """Format research content for Discord output.
        
        Args:
            content: The content to format
            content_type: The type of content ('analysis', 'review', or 'synthesis')
            
        Returns:
            List of Discord embeds containing the formatted content
            
        Raises:
            Exception: If formatting fails
        """
        try:
            # Convert content to embeds using message formatter
            formatted_content = await self.message_formatter.format_research_content(content, content_type)
            
            # Create embeds from formatted content
            embeds = []
            for chunk in formatted_content:
                embed = discord.Embed(
                    description=chunk,
                    color=discord.Color.blue()
                )
                embeds.append(embed)
            
            return embeds
            
        except Exception as e:
            self.logger.error(f"Error formatting for Discord: {str(e)}")
            # Return error embed
            error_embed = discord.Embed(
                title="Error",
                description=f"Failed to format content: {str(e)}",
                color=discord.Color.red()
            )
            return [error_embed] 