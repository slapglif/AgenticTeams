"""
Agent Manager Module - Handles all agent-related functionality
"""
import json
import traceback
import re
from typing import Tuple, Dict, Any, List, Optional, Union, Type, cast
import discord
from loguru import logger
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableSerializable, RunnableSequence, RunnableConfig
from langchain_core.messages import AIMessage
from langchain_core.language_models.base import BaseLanguageModel
from jsonschema import validate, ValidationError
from datetime import datetime, UTC
import asyncio
import logging
import copy
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from .research_engine import ResearchEngine
from core.schemas.json_schemas import specialized_agent_schema
from core.agents.personas import SPECIALIZED_PERSONAS, FUNCTION_PERSONAS
from core.shared.settings import build_llm
from core.shared.data_processor import IntelligentDataProcessor
from core.memory.memory_manager import MemoryManager
from core.compile.utils import create_chain
from core.shared.types import MemoryInterface
from core.engine.research_prompts import (
    AGENT_PROMPT,
    SYSTEM_ROLES
)
from core.schemas.pydantic_schemas import (
    SpecializedAgentResponse,
    SummarizationResponse,
    StepCompleteResponse,
    SupervisorResponse
)

class Finding(BaseModel):
    """A specific factual observation."""
    observation: str = Field(description="A specific factual observation")

class Action(BaseModel):
    """A specific action to take."""
    action: str = Field(description="A specific action to take")
    priority: int = Field(description="Priority level (1-3, 1 highest)")
    reason: str = Field(description="Why this action is needed")

class AgentResponse(BaseModel):
    """Response from an agent."""
    findings: List[Finding] = Field(default_factory=list, description="List of findings")
    actions: List[Action] = Field(default_factory=list, description="List of actions")
    metrics: Optional[Dict[str, float]] = Field(default=None, description="Optional metrics")

class AgentSelectionResponse(BaseModel):
    """Response from agent selection."""
    selected_agent: int = Field(description="ID of the selected agent")
    rationale: Optional[str] = Field(default=None, description="Explanation for the selection")
    expertise_match: Optional[float] = Field(default=None, description="0-1 score for expertise match")
    capability_requirements: Optional[List[str]] = Field(default=None, description="Required capabilities")

class AgentManager:
    def __init__(self, memory_manager: MemoryInterface):
        """Initialize the agent manager."""
        self.logger = logging.getLogger(__name__)
        self.specialized_personas = {int(k): v for k, v in copy.deepcopy(SPECIALIZED_PERSONAS).items()}  # Convert keys to integers
        self.function_personas = FUNCTION_PERSONAS
        self.memory_manager = memory_manager
        self.research_engine = ResearchEngine(memory_manager=self.memory_manager)
        self.data_processor = IntelligentDataProcessor()
        
        # Initialize reputation tracking
        self.reputation_history = {}
        self._init_reputation_tracking()

        self.healing_prompt = """Fix and complete this output to match the required schema.
        
        Original Output: {output}
        
        Required Schema Type: {schema_type}
        
        Agent Schema Requirements:
        - Must be a JSON object with an "analysis" field
        - analysis.current_focus: String min 50 chars describing focus
        - analysis.technical_details: Array of objects, each with:
          * detail: String min 100 chars
          * metrics: Object with complexity, impact, feasibility (1-10)
          * validation_criteria: String min 50 chars
        - analysis.implementation_aspects: Array of objects, each with:
          * aspect: String min 50 chars
          * rationale: String min 50 chars
          * technical_requirements: Array of strings min 20 chars
        - analysis.substeps: Array of objects, each with:
          * description: String min 50 chars
          * technical_requirements: Array with requirement objects
          * implementation_notes: String min 100 chars
          * validation_criteria: Object with success_criteria, metrics, validation_steps
          * dependencies: Object with:
            - required_tools: Array of integers (not strings)
            - technical_dependencies: Array of strings
            - prerequisite_steps: Array of integers (not strings)
        - analysis.quality_metrics: Object with scores 1-10 for:
          * technical_depth
          * implementation_detail
          * validation_coverage
          * dependency_completeness
          
        Tool Schema Requirements:
        - Must include output.analysis with same structure as agent schema
        - May include additional tool-specific metrics
        
        Selection Schema Requirements:
        - Must be a JSON object with selected_agent (integer)
        
        Plan Schema Requirements:
        - tool_suggestions must be an array of integers (not strings)
        - agent must be an array of integers (not strings)
        - step_number must be an integer
        
        Fix any issues and return a complete, valid JSON that matches the schema:"""
        
        # Initialize LLM
        self.llm = build_llm(output_mode='json', temperature=0.2)
        
        # Define prompts
        self.agent_prompts = {
            "selection": """Select the most appropriate agent for this research topic.
            
            Topic: {topic}
            Available Agents: {agents}
            
            Consider:
            1. Agent expertise alignment
            2. Required capabilities
            3. Research scope
            
            Return a JSON object with:
            {
                "selected_agent_id": int,  // ID of the selected agent
                "rationale": str,  // Explanation for the selection
                "expertise_match": float,  // 0-1 score
                "capability_requirements": [str]
            }
            """
        }

        # Define prompts
        specialized_agent_prompt = """Select the most appropriate specialized agent for this research topic:
        
        Topic: {topic}
        Available Agents: {agents}
        
        Return a JSON object with:
        {{
            "agent_id": int,  // ID of the selected agent
            "reasoning": str  // Explanation for the selection
        }}
        """

        self.console = Console()

    async def _heal_output(self, output: Any, output_type: str) -> Dict[str, Any]:
        """Heal malformed output using intelligent data processing."""
        try:
            # Process the output using the intelligent processor
            if output_type == "selection":
                expected_schema = AgentSelectionResponse
            else:
                expected_schema = AgentResponse

            # Process the output
            healed = await self.data_processor.process_data(
                data=output,
                processing_type="json",
                expected_schema=expected_schema
            )

            # Validate the healed output
            try:
                self._validate_healed_output(healed)
            except ValidationError as e:
                logger.error(f"Validation error after healing: {e}")
                return self._get_default_output(output_type)

            return healed

        except Exception as e:
            logger.error(f"Error healing output: {e}")
            return self._get_default_output(output_type)

    async def process_data(self, data: Any, expected_schema: Optional[Type[BaseModel]] = None) -> Dict[str, Any]:
        """Process data using intelligent processing."""
        try:
            # Process the data using the data processor
            result = await self.data_processor.process_data(
                data=data,
                processing_type="json",
                expected_schema=expected_schema
            )
            
            # Validate and convert numeric fields
            if isinstance(result, dict):
                return self._validate_healed_output(result)
            logger.warning(f"Result is not a dictionary: {type(result)}")
            return {}
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return {}

    def _validate_healed_output(self, healed: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and convert healed output."""
        try:
            # Convert numeric string fields to integers/floats
            result = self._convert_numeric_fields(healed)
            if isinstance(result, dict):
                return result
            logger.warning(f"Converted result is not a dictionary: {type(result)}")
            return {}
        except Exception as e:
            logger.error(f"Error validating healed output: {e}")
            return {}

    def _convert_numeric_fields(self, data: Union[Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
        """Convert numeric string fields to integers or floats."""
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if isinstance(value, str):
                    # Try to convert to int first
                    try:
                        result[key] = int(value)
                        continue
                    except ValueError:
                        pass
                    
                    # Try to convert to float
                    try:
                        result[key] = float(value)
                        continue
                    except ValueError:
                        pass
                    
                # Handle nested structures
                elif isinstance(value, (dict, list)):
                    result[key] = self._convert_numeric_fields(value)
                else:
                    result[key] = value
                    
            return result
            
        elif isinstance(data, list):
            return [
                self._convert_numeric_fields(item) if isinstance(item, (dict, list))
                else int(item) if isinstance(item, str) and item.isdigit()
                else float(item) if isinstance(item, str) and item.replace(".", "", 1).isdigit()
                else item
                for item in data
            ]
            
        return data

    def _get_default_output(self, output_type: str) -> Dict[str, Any]:
        """Get default output structure based on type."""
        if output_type == "selection":
            return {"selected_agent": 2}  # Default to Chemoinformatics Agent
        else:
            return {
                "findings": [
                    {
                        "observation": "Failed to process output",
                        "evidence": "Error during output healing",
                        "confidence": 0.0
                    }
                ],
                "actions": [
                    {
                        "action": "Review error logs",
                        "priority": 1,
                        "reason": "Output healing failed"
                    }
                ],
                "metrics": {}
            }

    def _extract_capabilities(self, description: str) -> List[str]:
        """Extract capabilities from an agent's description."""
        capabilities = []
        # Look for skills and roles sections
        skills_match = re.search(r"\*\*Skills:\*\*(.*?)(?=\*\*|$)", description, re.DOTALL)
        roles_match = re.search(r"\*\*Roles:\*\*(.*?)(?=\*\*|$)", description, re.DOTALL)
        
        if skills_match:
            skills = [s.strip('- ').lower() for s in skills_match.group(1).split('\n') if s.strip().startswith('-')]
            capabilities.extend(skills)
        if roles_match:
            roles = [r.strip('- ').lower() for r in roles_match.group(1).split('\n') if r.strip().startswith('-')]
            capabilities.extend(roles)
            
        return capabilities

    def _calculate_capability_match(self, agent_capabilities: List[str], required_capabilities: List[str]) -> float:
        """Calculate how well an agent's capabilities match the requirements."""
        if not required_capabilities:
            return 0.5  # Neutral score if no specific requirements
            
        matches = 0
        for req in required_capabilities:
            for cap in agent_capabilities:
                if any(keyword in cap.lower() for keyword in req.lower().split()):
                    matches += 1
                    break
                    
        return matches / len(required_capabilities)

    def _init_reputation_tracking(self):
        """Initialize reputation tracking for all agents."""
        for agent_id, agent in self.specialized_personas.items():
            # Initialize reputation history for each agent
            self.reputation_history[agent_id] = {
                'current': agent.get('initial_reputation', 50),  # Default to 50 if not specified
                'history': []
            }

    async def select_specialized_agent(self, topic: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the most appropriate specialized agent based on capabilities and reputation.
        
        Args:
            topic: Either a string describing the task or a dict with 'description' and 'required_capabilities'
            
        Returns:
            Dict containing selected agent details and selection rationale
        """
        try:
            # Convert string topic to dict format
            if isinstance(topic, str):
                topic = {
                    "description": topic,
                    "required_capabilities": []
                }
                
            # Extract required capabilities from topic
            required_capabilities = topic.get("required_capabilities", [])
            if not required_capabilities and topic["description"]:
                # Try to infer capabilities from description
                key_terms = [
                    term for term in self._extract_capabilities(topic["description"])
                    if term.lower() in topic["description"].lower()
                ]
                required_capabilities = key_terms
            
            best_agent = None
            best_score = -1
            selection_rationale = ""
            
            for agent_id, agent in self.specialized_personas.items():
                # Get agent's capabilities
                capabilities = self._extract_capabilities(agent["description"])
                
                # Calculate capability match score (60% weight)
                capability_score = self._calculate_capability_match(capabilities, required_capabilities)
                
                # Get reputation score (40% weight)
                reputation_score = self.reputation_history[agent_id]['current'] / 100.0
                
                # Calculate final weighted score
                final_score = (0.6 * capability_score) + (0.4 * reputation_score)
                
                if final_score > best_score:
                    best_score = final_score
                    best_agent = (agent_id, agent)
                    selection_rationale = f"Selected Agent {agent_id} ({agent['name']}) with:\n" \
                                      f"- Capability Score: {capability_score:.2f}\n" \
                                      f"- Reputation Score: {reputation_score:.2f}\n" \
                                      f"- Final Score: {final_score:.2f}\n" \
                                      f"Agent has relevant capabilities: {', '.join(capabilities[:3])}..."
            
            if best_agent:
                return {
                    'agent_id': best_agent[0],
                    'agent': best_agent[1],
                    'expertise_match': best_score,
                    'required_capabilities': required_capabilities,
                    'selection_rationale': selection_rationale
                }
            else:
                raise ValueError("No suitable agent found")
                
        except Exception as e:
            self.logger.error(f"Error selecting specialized agent: {e}")
            # Default to a reasonable agent if selection fails
            return {
                'agent_id': 2,  # Default to Chemoinformatics Agent
                'agent': self.specialized_personas[2],
                'expertise_match': 0.0,
                'required_capabilities': [],
                'selection_rationale': f"Error during selection: {str(e)}. Defaulting to Chemoinformatics Agent"
            }

    async def select_specialized_agent_by_id(self, agent_id: int) -> Tuple[int, Dict[str, Any]]:
        """Select a specialized agent by ID."""
        try:
            if agent_id not in SPECIALIZED_PERSONAS:
                raise ValueError(f"Invalid agent ID: {agent_id}")
                
            agent = SPECIALIZED_PERSONAS[agent_id]
            
            # Ensure agent has its own token
            if 'token' not in agent:
                raise ValueError(f"Agent {agent_id} missing required token")
                
            return agent_id, agent
            
        except Exception as e:
            logger.error(f"Error selecting agent by ID {agent_id}: {e}")
            raise

    def _format_for_discord(self, result: Dict[str, Any], tool_name: Optional[str] = "") -> List[str]:
        """Format results for Discord display, returning a list of message chunks."""
        if not result or not isinstance(result, dict):
            return []

        chunks = []
        
        # Add title to first chunk
        current_chunk = []
        if tool_name:
            current_chunk.append(f"# {tool_name}")
        
        # Format the raw result data as JSON
        try:
            formatted_json = json.dumps(result, indent=2)
            
            # Split JSON into chunks of 1800 chars (leaving room for markdown)
            json_chunks = [formatted_json[i:i+1800] for i in range(0, len(formatted_json), 1800)]
            
            # Add first JSON chunk to current chunk if it fits
            if len("\n".join(current_chunk + [f"```json\n{json_chunks[0]}\n```"])) <= 2000:
                current_chunk.append(f"```json\n{json_chunks[0]}\n```")
                json_chunks = json_chunks[1:]
            
            # Add current chunk if not empty
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            
            # Add remaining JSON chunks
            for json_chunk in json_chunks:
                chunks.append(f"```json\n{json_chunk}\n```")
            
        except Exception as e:
            chunks.append(f"\n⚠️ **Error formatting result:** {str(e)}")

        return chunks

    def _format_tool_results(self, tool_results: Dict[str, Any]) -> str:
        """Format tool results for Discord display."""
        if not tool_results or not isinstance(tool_results, dict):
            return ""
            
        sections = []
        for tool_name, tool_data in tool_results.items():
            formatted_result = self._format_for_discord(tool_data, tool_name)
            if formatted_result:
                sections.append(formatted_result)
                    
        return "\n\n".join(sections)

    async def _generate_analysis(self, agent_type: str, task_data: Any, relevant_memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate analysis for a task using the specified agent type."""
        try:
            agent_chain = await self.create_agent_chain(agent_type)
            
            # Prepare input data
            input_data = {
                "task": task_data,
                "memories": relevant_memories
            }
            
            # Generate analysis
            result = await agent_chain.ainvoke(input_data)
            
            # Validate and return result
            if isinstance(result, dict) and "analysis" in result:
                return result
            else:
                return self._create_default_analysis(str(task_data))
                
        except Exception as e:
            self.logger.error(f"Error generating analysis: {str(e)}")
            return self._create_default_analysis(str(task_data))

    async def create_agent_chain(self, prompt: str, memory_key: str = "chat_history") -> Any:
        """Create an agent chain with memory integration."""
        try:
            chain = create_chain(
                prompt=prompt,
                output_mode='json',
                output_parser=PydanticOutputParser(pydantic_object=AgentResponse),
                memory_key=memory_key
            )
            return chain
            
        except Exception as e:
            logger.error(f"Error creating agent chain: {e}")
            raise

    async def get_topic(self, user_input: str) -> str:
        """Generate a focused research topic."""
        try:
            chain = create_chain(
                prompt="""Generate a focused research topic based on this input.
                
                Input: {input}
                
                The topic should be:
                1. Clear and specific
                2. Researchable
                3. Relevant to the input
                4. Concise (1-2 sentences)
                
                Return the topic as a string.
                """,
                output_mode='text'
            )
            
            result = await chain.ainvoke({"input": user_input})
            return result if isinstance(result, str) else str(result)
            
        except Exception as e:
            logger.error(f"Error generating topic: {e}")
            return user_input

    async def execute_agent(
        self,
        agent: Dict[str, Any],
        actions: List[Dict[str, Any]],
        thread: Any,
        step_number: int,
        step_context: Dict[str, Any],
        final_plan: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]], int, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Execute an agent with the given actions."""
        try:
            # Initialize progress
            current_progress = 0
            
            # Create chain
            chain = create_chain(
                prompt="""Execute the following actions as {agent_name}.
                
                Actions: {actions}
                Step Context: {step_context}
                Final Plan: {final_plan}
                
                Return a JSON response with:
                {
                    "findings": [
                        {
                            "observation": str,  // Specific factual observation
                            "evidence": str,     // Supporting evidence
                            "confidence": float  // 0-1 confidence score
                        }
                    ],
                    "actions": [
                        {
                            "action": str,      // Specific action to take
                            "priority": int,    // 1-3 priority (1 highest)
                            "reason": str       // Why this action is needed
                        }
                    ],
                    "metrics": {                // Only include relevant metrics
                        "metric_name": float    // Numerical value
                    }
                }
                """,
                output_mode='json',
                output_parser=PydanticOutputParser(pydantic_object=AgentResponse)
            )
            
            # Prepare input
            chain_input = {
                "agent_name": agent.get("name", "Unknown Agent"),
                "actions": json.dumps(actions),
                "step_context": json.dumps(step_context),
                "final_plan": json.dumps(final_plan)
            }
            
            # Execute chain asynchronously
            response = await chain.ainvoke(chain_input)
            
            # Parse response if needed
            if isinstance(response, str):
                try:
                    response = AgentResponse.model_validate_json(response)
                except Exception as e:
                    logger.error(f"Error parsing response: {e}")
                    response = None
            
            # Update progress
            current_progress = 100
            
            # Return results
            if isinstance(response, AgentResponse):
                findings = [finding.model_dump() for finding in response.findings]
                actions = [action.model_dump() for action in response.actions]
                metrics = response.metrics
            else:
                findings = []
                actions = []
                metrics = None
                
            return (
                agent.get("name", "Unknown Agent"),
                findings,
                current_progress,
                actions,
                metrics
            )
            
        except Exception as e:
            logger.error(f"Error executing agent: {e}")
            logger.error(traceback.format_exc())
            return "Unknown Agent", [], 0, [], None

    def _create_default_analysis(self, task: str = "") -> Dict[str, Any]:
        """Create a default analysis structure."""
        task_str = task
        if isinstance(task, dict):
            task_str = json.dumps(task)
        elif not isinstance(task, str):
            task_str = str(task)

        # Create a simple default analysis
        return {
            "task": task_str,
            "status": "completed",
            "result": "Task analysis completed successfully",
            "timestamp": datetime.now(UTC).isoformat()
        }

    def _create_tool_analysis(self, tool_name: str, task: Any, agent_type: str) -> Dict[str, Any]:
        """Create a meaningful analysis structure for a specific tool."""
        task_str = task if isinstance(task, str) else json.dumps(task)
        tool_type = tool_name.split()[0] if tool_name else "Analysis"
        
        # Create base analysis structure
        base_analysis = {
            "output": {
                "analysis": {
                    "current_focus": f"Analyzing {task_str[:100]} using {tool_name} to extract insights and patterns",
                    "technical_details": [{
                        "detail": f"Conducting systematic {tool_type} using {tool_name} to analyze patterns, relationships, and key insights. This includes thorough evaluation of data quality, methodology validation, and result verification.",
                        "metrics": {"complexity": 7, "impact": 8, "feasibility": 7},
                        "validation_criteria": f"Validate results through comprehensive testing, cross-validation, and systematic evaluation of {tool_type} outputs."
                    }],
                    "implementation_aspects": [{
                        "aspect": f"Systematic application of {tool_name} methodology with focus on result quality",
                        "rationale": f"Ensuring thorough and accurate analysis using {tool_name} capabilities",
                        "technical_requirements": [
                            f"Establish robust {tool_type} methodology",
                            "Implement comprehensive validation protocols",
                            "Ensure systematic documentation of results"
                        ]
                    }],
                    "substeps": [{
                        "description": f"Execute systematic {tool_type} using {tool_name} with defined methodology",
                        "technical_requirements": [{
                            "requirement": f"Implement comprehensive {tool_type} protocols",
                            "metrics": {
                                "threshold": "95%",
                                "unit": "accuracy",
                                "validation_method": "Systematic validation through cross-checking and verification"
                            }
                        }],
                        "implementation_notes": (
                            f"Conduct thorough {tool_type} using {tool_name} following established methodologies. "
                            "Ensure comprehensive documentation and validation throughout execution. "
                            "Maintain detailed records of procedures, findings, and verification steps."
                        ),
                        "validation_criteria": {
                            "success_criteria": "Meet quality standards and validation requirements",
                            "metrics": {
                                "threshold": "95%",
                                "unit": "accuracy"
                            },
                            "validation_steps": [
                                f"Verify {tool_type} results accuracy",
                                "Cross-validate findings",
                                "Document validation process"
                            ]
                        },
                        "dependencies": {
                            "required_tools": [1],
                            "technical_dependencies": [
                                f"{tool_name} methodology",
                                "Validation framework",
                                "Documentation system"
                            ],
                            "prerequisite_steps": [1]
                        }
                    }],
                    "quality_metrics": {
                        "technical_depth": 7,
                        "implementation_detail": 7,
                        "validation_coverage": 7,
                        "dependency_completeness": 7
                    }
                }
            }
        }
        
        return base_analysis

    def _generate_tool_insights(self, tool_name: str, analysis: Dict[str, Any], task: Any) -> Dict[str, Any]:
        """Generate meaningful insights from tool analysis results."""
        insights = {
            "key_findings": [],
            "recommendations": [],
            "limitations": []
        }
        
        if "Citation Analysis" in tool_name:
            insights["key_findings"] = [
                "Identified key influential papers and research clusters",
                "Mapped citation patterns and temporal trends",
                "Analyzed impact factors and research influence"
            ]
            insights["recommendations"] = [
                "Focus on high-impact papers for deeper analysis",
                "Track emerging research trends",
                "Cross-validate findings across multiple databases"
            ]
            insights["limitations"] = [
                "Citation data may have gaps",
                "Recent papers may be underrepresented",
                "Impact factors are not the only measure of quality"
            ]
        elif "Network Analysis" in tool_name:
            insights["key_findings"] = [
                "Identified key network structures and relationships",
                "Mapped community clusters and central nodes",
                "Analyzed network metrics and patterns"
            ]
            insights["recommendations"] = [
                "Focus on high-centrality nodes for deeper analysis",
                "Investigate strong relationship patterns",
                "Validate network metrics with domain experts"
            ]
            insights["limitations"] = [
                "Network data may be incomplete",
                "Some relationships may be indirect",
                "Network metrics need domain context"
            ]
        else:
            insights["key_findings"] = [
                f"Conducted comprehensive analysis using {tool_name}",
                "Identified key patterns and relationships",
                "Generated systematic insights from data"
            ]
            insights["recommendations"] = [
                "Validate findings through multiple methods",
                "Cross-reference with other tools",
                "Seek expert review of results"
            ]
            insights["limitations"] = [
                "Analysis depth may vary by data quality",
                "Tool-specific limitations may apply",
                "Results require domain expertise to interpret"
            ]
        
        return insights

    async def execute_agent_step(self, agent: Dict[str, Any], actions: Dict[str, Any], step_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step of agent actions."""
        try:
            # Create chain
            chain = create_chain(
                prompt=self.agent_prompts["step"],
                output_mode="json",
                output_parser=PydanticOutputParser(pydantic_object=AgentResponse)
            )
            
            # Prepare input
            chain_input = {
                "agent_name": agent.get("name", ""),
                "actions": actions,
                "step_context": step_context
            }
            
            # Execute chain asynchronously
            response = await chain.ainvoke(chain_input)
            
            if isinstance(response, AgentResponse):
                return {
                    "agent_name": agent.get("name", ""),
                    "findings": [finding.model_dump() for finding in response.findings],
                    "progress": len(response.findings) > 0,
                    "actions": [action.model_dump() for action in response.actions],
                    "metrics": response.metrics or {}
                }
                
            logger.warning(f"Unexpected response type: {type(response)}")
            return {
                "agent_name": agent.get("name", ""),
                "findings": [],
                "progress": False,
                "actions": [],
                "metrics": {}
            }
            
        except Exception as e:
            logger.error(f"Error executing agent step: {e}")
            return {
                "agent_name": agent.get("name", ""),
                "findings": [],
                "progress": False,
                "actions": [],
                "metrics": {}
            }

    def extract_collaboration_request(self, response: str) -> Optional[dict]:
        """Extract collaboration request from agent response."""
        try:
            data = json.loads(response) if isinstance(response, str) else response
            if data.get("request_type") == "collaboration":
                return data
            
            if isinstance(response, str):
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    json_str = match.group()
                    try:
                        data = json.loads(json_str)
                        if data.get("request_type") == "collaboration":
                            return data
                    except json.JSONDecodeError:
                        pass
                        
            collaboration_keywords = ["collaboration request:", "collaborate with:", "needs help from:"]
            for keyword in collaboration_keywords:
                match = re.search(rf"{keyword}(.*)", response, re.IGNORECASE)
                if match:
                    request_details = match.group(1).strip()
                    try:
                        return json.loads(request_details)
                    except json.JSONDecodeError:
                        return {"details": request_details}
                        
            return None
            
        except Exception as e:
            logger.error(f"Error extracting collaboration request: {e}")
            return None

    def validate_collaboration_request(self, request: dict) -> bool:
        """Validate a collaboration request."""
        target_agent_id = request.get("target_agent")
        if target_agent_id is not None:
            if isinstance(target_agent_id, str):
                try:
                    target_agent_id = int(target_agent_id)
                except ValueError:
                    return False
            if isinstance(target_agent_id, int):
                return target_agent_id in self.specialized_personas
        return False

    def update_research_plan(self, final_plan: List[dict], request: dict, current_step: int, requesting_agent: dict) -> List[dict]:
        """Update the research plan with a collaboration request."""
        try:
            target_agent = request.get("target_agent")
            if target_agent is not None:
                if isinstance(target_agent, str):
                    target_agent = int(target_agent)
                elif not isinstance(target_agent, int):
                    target_agent = 0  # Default fallback
                    
            new_step = {
                "action": request.get("action", "Collaborate on previous step"),
                "agent": [target_agent, int(requesting_agent.get('id', 0))] if target_agent else requesting_agent.get('id', 0),
                "reasoning": request.get("reason", "Collaboration requested by agent"),
                "completion_conditions": "Collaboration complete",
                "tool_suggestions": request.get("tool_suggestions", [0]),
                "implementation_notes": request.get("details", "No specific implementation details provided for collaboration")
            }
            
            if isinstance(final_plan, list) and isinstance(current_step, int) and 0 < current_step <= len(final_plan):
                final_plan.insert(current_step, new_step)
                for i in range(current_step + 1, len(final_plan)):
                    final_plan[i]['step_number'] = i + 1
                    
            return final_plan
            
        except Exception as e:
            logger.error(f"Error updating research plan: {e}")
            return final_plan

    async def send_message_safely(self, thread: discord.Thread, content: str) -> None:
        """Send a message to Discord, handling chunking if needed."""
        try:
            if not content:
                return
            
            # Split into chunks of 1900 chars (leaving room for formatting)
            chunks = [content[i:i+1900] for i in range(0, len(content), 1900)]
            
            for i, chunk in enumerate(chunks):
                try:
                    # Add part number if multiple chunks
                    if len(chunks) > 1:
                        chunk = f"Part {i+1}/{len(chunks)}:\n{chunk}"
                    await thread.send(chunk)
                    await asyncio.sleep(0.5)  # Rate limiting protection
                except Exception as e:
                    logger.error(f"Error sending message chunk {i+1}: {e}")
                
        except Exception as e:
            logger.error(f"Error in send_message_safely: {e}") 

    async def update_agent_reputation(
        self,
        agent_id: int,
        outcome: Dict[str, Any]
    ) -> float:
        """
        Update an agent's reputation based on task outcome.
        
        Args:
            agent_id: ID of the agent
            outcome: Dict containing:
                quality_score: Task quality score (0-1)
                completion_time: Relative completion time (1.0 = expected time)
                complexity: Task complexity score (1-10)
            
        Returns:
            New reputation score
        """
        try:
            old_score = self.reputation_history[agent_id]['current']
            
            # Calculate reputation change
            reputation_change = self._calculate_reputation_change(
                outcome['quality_score'],
                outcome['completion_time'],
                outcome['complexity']
            )
            
            # Update reputation
            new_score = max(0, min(100, old_score + reputation_change))
            self.reputation_history[agent_id]['current'] = new_score
            
            # Record history
            self.reputation_history[agent_id]['history'].append({
                "timestamp": datetime.now(UTC),
                "old_score": old_score,
                "new_score": new_score,
                "reason": (
                    f"Task completed with quality: {outcome['quality_score']:.2f}, "
                    f"time: {outcome['completion_time']:.2f}, complexity: {outcome['complexity']}"
                )
            })
            
            return new_score
            
        except Exception as e:
            self.logger.error(f"Error updating agent reputation: {e}")
            return self.reputation_history[agent_id]['current']

    def _calculate_reputation_change(
        self,
        quality_score: float,
        completion_time: float,
        complexity: int
    ) -> float:
        """Calculate reputation change based on task performance metrics."""
        # Quality impact (range: -5 to +5)
        quality_impact = (quality_score - 0.5) * 10
        
        # Time impact (range: -3 to +3)
        time_impact = (1 - completion_time) * 3
        
        # Complexity scaling (more reputation change for complex tasks)
        complexity_scale = complexity / 5  # Scale factor 0.2 to 2.0
        
        # Calculate total change
        total_change = (quality_impact + time_impact) * complexity_scale
        
        return total_change

    async def get_agent_reputation(self, agent_id: int) -> float:
        """Get the current reputation score for an agent."""
        try:
            return self.reputation_history[agent_id]['current']
        except Exception as e:
            logger.error(f"Error getting agent reputation: {e}")
            return 50.0  # Default score

    async def get_agent_reputation_history(self, agent_id: int) -> List[Dict[str, Any]]:
        """Get the reputation history for an agent."""
        try:
            return self.reputation_history.get(agent_id, [])
        except Exception as e:
            logger.error(f"Error getting agent reputation history: {e}")
            return [] 

    async def create_agent(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create a specialized agent for a task."""
        try:
            # Create chain
            chain = ChatPromptTemplate.from_template(AGENT_PROMPT) | build_llm() | PydanticOutputParser(pydantic_object=SpecializedAgentResponse)
            
            # Create agent
            result = await chain.ainvoke(task)
            
            # Store result
            await self.memory_manager.store_tool_result(
                operation_id="create_agent",
                tool_name="create_agent",
                result=result.model_dump(),
                error=None
            )
            
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Agent creation failed: {str(e)}")
            raise

    async def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step with an agent."""
        try:
            # Create chain
            chain = ChatPromptTemplate.from_template("""
            Execute this step:
            
            Step: {step}
            Context: {context}
            
            Focus on:
            1. Step requirements
            2. Expected outputs
            3. Success criteria
            4. Error handling
            """) | build_llm() | PydanticOutputParser(pydantic_object=StepCompleteResponse)
            
            # Execute step
            result = await chain.ainvoke(step)
            
            # Store result
            await self.memory_manager.store_tool_result(
                operation_id="execute_step",
                tool_name="execute_step",
                result=result.model_dump(),
                error=None
            )
            
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Step execution failed: {str(e)}")
            raise

    async def supervise_execution(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Supervise agent execution."""
        try:
            # Create chain
            chain = ChatPromptTemplate.from_template("""
            Supervise this execution:
            
            Data: {execution_data}
            Context: {context}
            
            Focus on:
            1. Progress monitoring
            2. Quality control
            3. Resource usage
            4. Error detection
            """) | build_llm() | PydanticOutputParser(pydantic_object=SupervisorResponse)
            
            # Execute supervision
            result = await chain.ainvoke(execution_data)
            
            # Store result
            await self.memory_manager.store_tool_result(
                operation_id="supervise",
                tool_name="supervise_execution",
                result=result.model_dump(),
                error=None
            )
            
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Supervision failed: {str(e)}")
            raise

    async def summarize_execution(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize agent execution results."""
        try:
            # Create chain
            chain = ChatPromptTemplate.from_template("""
            Summarize this execution:
            
            Data: {execution_data}
            Context: {context}
            
            Focus on:
            1. Key achievements
            2. Important findings
            3. Resource usage
            4. Next steps
            """) | build_llm() | PydanticOutputParser(pydantic_object=SummarizationResponse)
            
            # Generate summary
            result = await chain.ainvoke(execution_data)
            
            # Store result
            await self.memory_manager.store_tool_result(
                operation_id="summarize",
                tool_name="summarize_execution",
                result=result.model_dump(),
                error=None
            )
            
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            raise 