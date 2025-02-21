"""Intelligent error handler using LLMs to process and explain errors."""

import json
import traceback
from typing import Dict, Any, Optional, Union, List, cast, Type, Literal
from loguru import logger
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from datetime import datetime, UTC
import logging

from core.shared.settings import build_llm
from core.shared.data_processor import IntelligentDataProcessor
from core.compile.utils import create_chain

class BaseErrorSchema(BaseModel):
    """Base schema for error responses."""
    error: str = Field(description="Error message")
    message: str = Field(description="Additional message")

class AnalysisSchema(BaseModel):
    """Schema for error analysis."""
    severity: Literal["low", "medium", "high", "critical"] = Field(description="Error severity")
    explanation: str = Field(description="Error explanation")
    potential_causes: List[str] = Field(description="Potential causes")
    impact: str = Field(description="Impact description")
    recommended_actions: List[str] = Field(description="Recommended actions")
    user_message: str = Field(description="User-friendly message")

class ExplanationSchema(BaseModel):
    """Schema for error explanation."""
    title: str = Field(description="Error title")
    summary: str = Field(description="Error summary")
    impact: str = Field(description="Impact description")
    next_steps: List[str] = Field(description="Next steps")
    technical_details: Optional[str] = Field(default=None, description="Technical details")

class RecoverySchema(BaseModel):
    """Schema for recovery plan."""
    immediate_actions: List[str] = Field(description="Immediate actions")
    required_fixes: List[str] = Field(description="Required fixes")
    validation_steps: List[str] = Field(description="Validation steps")
    estimated_effort: str = Field(description="Estimated effort")

class ErrorHandler:
    def __init__(self):
        """Initialize the error handler."""
        self.logger = logging.getLogger(__name__)
        self.data_processor = IntelligentDataProcessor()
        
        # Initialize LLM with base schema
        self.llm = build_llm(
            output_mode='json',
            output_parser=PydanticOutputParser(pydantic_object=BaseErrorSchema),
            temperature=0.2
        )
        
        # Define error prompts
        self.error_prompts = {
            "analyze": """Analyze this error and provide a detailed explanation:
            
            Error: {error}
            Stack Trace: {stack_trace}
            Context: {context}
            
            Consider:
            1. Error type and severity
            2. Potential causes
            3. Impact on system
            4. Recommended actions
            
            Return a JSON analysis with:
            {{}
                "severity": str,  // "low", "medium", "high", "critical"
                "explanation": str,
                "potential_causes": [str],
                "impact": str,
                "recommended_actions": [str],
                "user_message": str  // User-friendly message
            }}
            """,
            "explain": """Explain this error analysis in {user_level} terms:
            
            Analysis: {analysis}
            
            Return a JSON explanation with:
            {{
                "title": str,  // Brief error title
                "summary": str,  // User-friendly summary
                "impact": str,  // Impact on user/system
                "next_steps": [str],  // What user should do
                "technical_details": str  // Optional technical details
            }}
            """,
            "recover": """Create a recovery plan based on this error analysis:
            
            Analysis: {analysis}
            System State: {system_state}
            
            Return a JSON recovery plan with:
            {{
                "immediate_actions": [str],  // Actions to take now
                "required_fixes": [str],  // Code/system fixes needed
                "validation_steps": [str],  // Steps to validate recovery
                "estimated_effort": str  // Estimated effort level
            }}
            """
        }
        
        # Define schema mappings
        self.schema_mappings = {
            "analyze": AnalysisSchema,
            "explain": ExplanationSchema,
            "recover": RecoverySchema
        }

    async def process_error(self, error: Exception, context: Dict[str, Any], user_level: str = "technical") -> Dict[str, Any]:
        """Process an error and generate analysis."""
        try:
            # Extract root cause
            root_cause = self._extract_root_cause(error)
            
            # Analyze error
            analysis = await self._analyze_error(
                error=str(error),
                stack_trace=traceback.format_exc(),
                context=context
            )
            
            # Generate explanation
            explanation = await self._explain_error(
                analysis=analysis,
                user_level=user_level
            )
            
            # Create recovery plan
            recovery_plan = await self._create_recovery_plan(
                analysis=analysis,
                system_state=context
            )
            
            # Combine results
            return {
                "error": str(error),
                "original_error": root_cause,
                "analysis": analysis,
                "explanation": explanation,
                "recovery_plan": recovery_plan,
                "timestamp": datetime.now(UTC).isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error in error processing: {str(e)}")
            return {
                "error": str(e),
                "original_error": str(error),
                "explanation": "Error occurred during error analysis",
                "analysis": {
                    "severity": "unknown",
                    "explanation": "Error occurred during error analysis",
                    "potential_causes": [],
                    "impact": "Unknown",
                    "recommended_actions": ["Contact support"],
                    "user_message": "An error occurred while processing your request"
                },
                "recovery": {
                    "immediate_actions": ["Stop affected operations"],
                    "long_term_fixes": ["Review system design"],
                    "validation_steps": ["Verify system stability"]
                },
                "fallback": True
            }

    def _extract_root_cause(self, error: Exception) -> str:
        """Extract the root cause from an error."""
        try:
            if hasattr(error, "__cause__") and error.__cause__:
                return str(error.__cause__)
            return str(error)
        except Exception as e:
            self.logger.error(f"Error extracting root cause: {str(e)}")
            return str(error)

    async def _analyze_error(
        self,
        error: str,
        stack_trace: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze an error using intelligent processing."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are an expert error analyst. Analyze errors and provide detailed insights."),
                HumanMessagePromptTemplate.from_template("""
                Analyze this error and provide insights:
                
                Error: {error}
                Context: {context}
                
                Provide a JSON response with:
                - error_type: The type of error
                - root_cause: Likely root cause
                - severity: Error severity (1-5)
                - impact: Potential impact
                - recommendations: Array of fix suggestions
                """)
            ])
            
            chain = create_chain(
                prompt=prompt,
                output_mode='json',
                schema_type="error_analysis"
            )
            
            return await chain.ainvoke({
                "error": error,
                "context": json.dumps(context) if context else "{}"
            })
            
        except Exception as e:
            logger.error(f"Error analyzing error: {e}")
            return {
                "error_type": "Unknown",
                "root_cause": str(e),
                "severity": 5,
                "impact": "Unable to analyze error",
                "recommendations": ["Manual investigation required"]
            }

    async def _explain_error(
        self,
        analysis: Dict[str, Any],
        user_level: str
    ) -> Dict[str, Any]:
        """Generate user-friendly error explanation."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are an expert at explaining technical errors in a user-friendly way."),
                HumanMessagePromptTemplate.from_template("""
                Generate a user-friendly error message:
                
                Error Details: {error}
                
                The message should:
                1. Be clear and concise
                2. Explain what went wrong
                3. Suggest next steps if applicable
                4. Use appropriate tone based on severity
                
                Return a JSON explanation with:
                {{s
                    "title": str,  // Brief error title
                    "summary": str,  // User-friendly summary
                    "impact": str,  // Impact on user/system
                    "next_steps": [str],  // What user should do
                    "technical_details": str  // Optional technical details
                }}
                """)
            ])
            
            chain = create_chain(
                prompt=prompt,
                output_mode='json',
                schema_type="error_explanation"
            )
            
            result = await chain.ainvoke({"error": json.dumps(analysis)})
            return result if isinstance(result, dict) else {
                "title": "Error Processing Failed",
                "summary": str(result),
                "impact": "Unable to process error details",
                "next_steps": ["Contact support"],
                "technical_details": None
            }
            
        except Exception as e:
            logger.error(f"Error generating message: {e}")
            return {
                "title": "Error Processing Failed",
                "summary": f"An error occurred: {analysis.get('error_type', 'Unknown error')}",
                "impact": "Unable to process error details",
                "next_steps": ["Contact support"],
                "technical_details": None
            }

    async def _create_recovery_plan(
        self,
        analysis: Dict[str, Any],
        system_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a recovery plan based on error analysis."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are an expert at creating recovery plans for technical errors."),
                HumanMessagePromptTemplate.from_template("""
                Create a recovery plan based on this error analysis:
                
                Analysis: {analysis}
                System State: {system_state}
                
                Return a JSON recovery plan with:
                {{
                    "immediate_actions": [str],  // Actions to take now
                    "required_fixes": [str],  // Code/system fixes needed
                    "validation_steps": [str],  // Steps to validate recovery
                    "estimated_effort": str  // Estimated effort level
                }}
                """)
            ])
            
            chain = create_chain(
                prompt=prompt,
                output_mode='json',
                schema_type="recovery_plan"
            )
            
            return await chain.ainvoke({
                "analysis": json.dumps(analysis),
                "system_state": json.dumps(system_state) if system_state else "{}"
            })
            
        except Exception as e:
            logger.error(f"Error creating recovery plan: {e}")
            raise

    def format_for_user(self, error_info: Dict[str, Any], format_type: str = "text") -> str:
        """Format error information for user display."""
        try:
            if format_type == "text":
                sections = [
                    "ERROR EXPLANATION",
                    "-----------------",
                    error_info.get("analysis", {}).get("explanation", "Error occurred during error analysis"),
                    "",
                    "SEVERITY",
                    "--------",
                    error_info.get("analysis", {}).get("severity", "unknown"),
                    "",
                    "IMPACT",
                    "------",
                    error_info.get("analysis", {}).get("impact", "Unknown"),
                    "",
                    "ACTION ITEMS",
                    "------------",
                    "\n".join(f"- {action}" for action in error_info.get("analysis", {}).get("recommended_actions", ["Contact support"])),
                    "",
                    "RECOVERY STEPS",
                    "--------------",
                    "\n".join(f"- {step}" for step in error_info.get("recovery", {}).get("immediate_actions", ["Stop affected operations"]))
                ]
                return "\n".join(sections)
            elif format_type == "markdown":
                sections = [
                    "# Error Explanation",
                    error_info.get("analysis", {}).get("explanation", "Error occurred during error analysis"),
                    "",
                    "## Severity",
                    error_info.get("analysis", {}).get("severity", "unknown"),
                    "",
                    "## Impact",
                    error_info.get("analysis", {}).get("impact", "Unknown"),
                    "",
                    "## Action Items",
                    "\n".join(f"* {action}" for action in error_info.get("analysis", {}).get("recommended_actions", ["Contact support"])),
                    "",
                    "## Recovery Steps",
                    "\n".join(f"* {step}" for step in error_info.get("recovery", {}).get("immediate_actions", ["Stop affected operations"]))
                ]
                return "\n".join(sections)
            else:
                return json.dumps(error_info, indent=2)
                
        except Exception as e:
            logger.error(f"Error formatting error message: {e}")
            return str(error_info)

    def _format_text(self, error_info: Dict[str, Any]) -> str:
        """Format error information as plain text."""
        try:
            lines = []
            
            # Add error message
            lines.extend([
                "ERROR EXPLANATION",
                "-----------------",
                error_info["analysis"]["explanation"],
                "",
                "SEVERITY",
                "--------",
                error_info["analysis"]["severity"],
                "",
                "IMPACT",
                "------",
                error_info["analysis"]["impact"],
                "",
                "RECOMMENDED ACTIONS",
                "-------------------"
            ])
            
            for action in error_info["analysis"]["recommended_actions"]:
                lines.append(f"- {action}")
                
            lines.extend([
                "",
                "RECOVERY STEPS",
                "--------------"
            ])
            
            for step in error_info["recovery"]["immediate_actions"]:
                lines.append(f"- {step}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error formatting text: {e}")
            return str(error_info)

    def _format_markdown(self, error_info: Dict[str, Any]) -> str:
        """Format error information as markdown."""
        try:
            lines = []
            
            # Add error message
            lines.extend([
                "# Error Explanation",
                "",
                error_info["analysis"]["explanation"],
                "",
                "## Severity",
                "",
                error_info["analysis"]["severity"],
                "",
                "## Impact",
                "",
                error_info["analysis"]["impact"],
                "",
                "## Recommended Actions",
                ""
            ])
            
            for action in error_info["analysis"]["recommended_actions"]:
                lines.append(f"* {action}")
                
            lines.extend([
                "",
                "## Recovery Steps",
                ""
            ])
            
            for step in error_info["recovery"]["immediate_actions"]:
                lines.append(f"* {step}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error formatting markdown: {e}")
            return str(error_info)

    async def generate_error_message(self, error: Dict[str, Any]) -> str:
        """Generate a user-friendly error message."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are an expert at explaining technical errors in a user-friendly way."),
                HumanMessagePromptTemplate.from_template("""
                Generate a user-friendly error message:
                
                Error: {error}
                
                The message should:
                1. Be clear and concise
                2. Explain what went wrong
                3. Suggest next steps if applicable
                4. Use appropriate tone based on severity
                """)
            ])
            
            chain = create_chain(
                prompt=prompt,
                output_mode='text'
            )
            
            result = await chain.ainvoke({"error": json.dumps(error)})
            return str(result)
            
        except Exception as e:
            logger.error(f"Error generating message: {e}")
            return f"An error occurred: {error.get('error_type', 'Unknown error')}"

async def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Handle an error with LLM assistance."""
    try:
        # Create error handling chain
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an expert error analyst. Analyze errors and provide detailed explanations and recovery steps."),
            HumanMessagePromptTemplate.from_template("""
            Analyze this error and provide explanation and recovery steps:
            
            Error: {error}
            Context: {context}
            
            Provide a detailed analysis including:
            1. Error explanation
            2. Impact assessment
            3. Recovery steps
            4. Validation steps
            """)
        ])
        
        chain = create_chain(
            prompt=prompt,
            output_mode="error",
            output_parser=PydanticOutputParser(pydantic_object=BaseErrorSchema)
        )
        
        # Execute chain
        result = await chain.ainvoke({
            "error": str(error),
            "context": context or {}
        })
        
        return result.model_dump()
        
    except Exception as e:
        logger.error(f"Error handling failed: {e}")
        # Return basic error info if handling fails
        return {
            "explanation": {
                "title": "Error Handling Failed",
                "summary": str(error),
                "impact": "Unknown",
                "next_steps": ["Contact support"]
            },
            "recovery": {
                "immediate_actions": ["Stop affected operations"],
                "required_fixes": ["Investigate error"],
                "validation_steps": ["Verify system state"],
                "estimated_effort": "Unknown"
            }
        } 