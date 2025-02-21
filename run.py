"""
Runner Script - Orchestrates all components of the research system
"""
import asyncio
import json
import os
import re
import sys
import traceback
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, List, Tuple, Dict, Union, cast, Optional

import aiofiles
import aiohttp
import discord
from discord import Embed, Color, Attachment, TextChannel, ForumChannel, Message, Thread, DMChannel, VoiceChannel, StageChannel, PartialMessageable, GroupChannel
from discord.ext import commands
from discord.ext.commands import Bot as CommandsBot
from dotenv import load_dotenv
from jsonschema import validate, ValidationError
from langchain.callbacks.tracers import LangChainTracer
from langchain.globals import set_debug, set_llm_cache
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.output_parsers import PydanticOutputParser
from langsmith import Client
from loguru import logger
from rich.traceback import install
from rich.console import Console
from rich.table import Table
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

from core.agents.personas import CHANNEL_ID, FUNCTION_PERSONAS
from core.compile import State
from core.compile.utils import create_chain
from core.shared.types import MemoryInterface
from core.shared.settings import build_llm
from core.engine.research_prompts import (
    RESEARCH_PROMPT,
    DATA_ANALYSIS_PROMPT,
    GRAPH_ANALYSIS_PROMPT,
    CITATION_ANALYSIS_PROMPT,
    NETWORK_ANALYSIS_PROMPT,
    NETWORK_STRUCTURE_PROMPT,
    SYSTEM_ROLES
)
from core.compile.prompts import ANALYSIS_PROMPT
from core.schemas.pydantic_schemas import (
    SearchQuery,
    DataAnalysisResponse,
    GraphAnalysisResponse,
    CitationAnalysisResponse,
    NetworkAnalysisResponse
)
from core.engine.agent_manager import AgentManager
from core.engine.research_engine import ResearchEngine
from core.engine.discord_interface import DiscordInterface
from core.memory.memory_manager import MemoryManager
from core.agents.prompts import (
    cot32_prompt,
    final_plan_prompt,
    step_complete_prompt,
    supervisor_prompt,
    specialized_agent_prompt,
    summarization_prompt,
    topic_prompt
)
from core.schemas.json_schemas import topic_schema
from core.compile.models import Task, Tool

# Initialize logging and debugging
install()
load_dotenv()  # Load environment variables from .env file

# Configure logging
logger.remove()
logger.add(sys.stderr,
           level="TRACE",  # Most detailed logging level
           format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
           backtrace=True,  # Include full traceback
           diagnose=True  # Include variable values in tracebacks
           )
logger.add("dynamo.log",
           rotation="1 day",
           level="TRACE",
           format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
           backtrace=True,
           diagnose=True,
           enqueue=True  # Thread-safe logging
           )


def log_operation(operation: str, **kwargs):
    """Log operation details with all relevant information."""
    try:
        logger.debug(f"Operation: {operation}")
        for key, value in kwargs.items():
            if isinstance(value, (dict, list)):
                try:
                    logger.debug(f"{key}:\n{json.dumps(value, indent=2, default=str)}")
                except:
                    logger.debug(f"{key}: {str(value)}")
            else:
                logger.debug(f"{key}: {str(value)}")
    except Exception as e:
        logger.error(f"Error in log_operation: {e}")


# Configure LangCe(Nonhain
set_llm_cache(None)
set_debug(False)  # Enable LangChain debug mode

# Verify environment variables
required_vars = ['SEARX_HOST', 'OLLAMA_HOST']  # Remove DISCORD_TOKEN since it comes from personas
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

# Remove hardcoded environment variables since they're now in .env

# Initialize Redis chat history
redis_history = RedisChatMessageHistory(session_id="research_bot", url="redis://localhost:6379/0")

# Initialize LangSmith client with error handling and positional arguments
try:
    # Initialize LangSmith client first
    client = Client(api_key="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")
    tracer = LangChainTracer(project_name="AnunaAgentApp", client=client)
    logger.info("LangSmith tracer initialized successfully.")
except Exception as e:
    logger.warning(f"Failed to initialize LangSmith tracer: {e}")
    logger.warning("Continuing without tracing...")
    client = None
    tracer = None


class BenchmarkData:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.research_topic: str = ""
        self.user_input: str = ""
        self.final_plan: str = ""
        self.step_evaluations: List[Dict[str, Any]] = []
        self.final_metrics: Dict[str, Any] = {}
        self.conversation_history: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "research_topic": self.research_topic,
            "user_input": self.user_input,
            "final_plan": self.final_plan,
            "step_evaluations": self.step_evaluations,
            "final_metrics": self.final_metrics,
            "conversation_history": self.conversation_history
        }


class BenchmarkManager:
    def __init__(self):
        # Create necessary directories
        self.base_dir = Path("research_data")
        self.benchmarks_dir = self.base_dir / "benchmarks"
        self.detailed_dir = self.base_dir / "detailed_benchmarks"
        self.metrics_dir = self.base_dir / "metrics"

        # Create all directories
        for directory in [self.base_dir, self.benchmarks_dir, self.detailed_dir, self.metrics_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    async def save_benchmark(self, benchmark_data: BenchmarkData) -> Tuple[str, str]:
        """Save benchmark data to files."""
        run_id = benchmark_data.run_id
        timestamp = datetime.now().isoformat()

        # Create metrics for the benchmark
        metrics = self._calculate_metrics(benchmark_data)

        # Save detailed benchmark
        detailed_path = self.detailed_dir / f"detailed_benchmark_{run_id}.json"
        detailed_data = {
            **benchmark_data.to_dict(),
            "metrics": metrics,
            "timestamp": timestamp
        }

        # Save summary benchmark
        summary_path = self.benchmarks_dir / f"benchmark_{run_id}.json"
        summary_data = {
            "run_id": run_id,
            "research_topic": benchmark_data.research_topic,
            "metrics": metrics,
            "timestamp": timestamp
        }

        # Save files
        async with aiofiles.open(detailed_path, 'w') as f:
            await f.write(json.dumps(detailed_data, indent=2))

        async with aiofiles.open(summary_path, 'w') as f:
            await f.write(json.dumps(summary_data, indent=2))

        return str(detailed_path), str(summary_path)

    def _calculate_metrics(self, benchmark_data: BenchmarkData) -> Dict[str, Any]:
        """Calculate meaningful metrics from the benchmark data."""
        metrics = {
            "research_quality": {
                "depth": self._calculate_depth_score(benchmark_data),
                "breadth": self._calculate_breadth_score(benchmark_data),
                "citation_quality": self._calculate_citation_score(benchmark_data),
                "methodology": self._calculate_methodology_score(benchmark_data)
            },
            "execution_metrics": {
                "steps_completed": len(benchmark_data.step_evaluations),
                "average_step_duration": self._calculate_avg_step_duration(benchmark_data),
                "total_citations": self._count_total_citations(benchmark_data),
                "unique_sources": self._count_unique_sources(benchmark_data)
            },
            "content_metrics": {
                "topic_coverage": self._calculate_topic_coverage(benchmark_data),
                "source_diversity": self._calculate_source_diversity(benchmark_data),
                "temporal_distribution": self._calculate_temporal_distribution(benchmark_data)
            }
        }

        # Calculate overall scores
        metrics["overall_score"] = sum([
            metrics["research_quality"]["depth"] * 0.3,
            metrics["research_quality"]["breadth"] * 0.2,
            metrics["research_quality"]["citation_quality"] * 0.25,
            metrics["research_quality"]["methodology"] * 0.25
        ])

        return metrics

    # Metric calculation methods (placeholders)
    def _calculate_depth_score(self, data: BenchmarkData) -> float:
        return 0.0

    def _calculate_breadth_score(self, data: BenchmarkData) -> float:
        return 0.0

    def _calculate_citation_score(self, data: BenchmarkData) -> float:
        return 0.0

    def _calculate_methodology_score(self, data: BenchmarkData) -> float:
        return 0.0

    def _calculate_avg_step_duration(self, data: BenchmarkData) -> float:
        return 0.0

    def _count_total_citations(self, data: BenchmarkData) -> int:
        return 0

    def _count_unique_sources(self, data: BenchmarkData) -> int:
        return 0

    def _calculate_topic_coverage(self, data: BenchmarkData) -> float:
        return 0.0

    def _calculate_source_diversity(self, data: BenchmarkData) -> float:
        return 0.0

    def _calculate_temporal_distribution(self, data: BenchmarkData) -> Dict[str, int]:
        return {"recent": 0, "medium": 0, "old": 0}


# Initialize Discord intents
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True


# Initialize bot instances
class ResearchBot(CommandsBot):
    """Discord bot for research tasks."""

    def __init__(self, agent_data: Dict[str, Any]):
        """Initialize the bot."""
        # Set up intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.guilds = True

        # Get bot configuration from agent data
        self.name = str(agent_data.get('name', 'Research Bot'))
        self.description = str(agent_data.get('description', 'A bot for research tasks'))
        self.token = str(agent_data.get('token', ''))
        if not self.token:
            raise ValueError("Bot token not found in agent data")

        # Initialize base class
        super().__init__(command_prefix="!", intents=intents, description=self.description)

        # Initialize components
        self.memory_manager = MemoryManager()
        self.agent_manager = AgentManager(self.memory_manager)
        self.active_threads: Dict[str, Dict[str, Any]] = {}
        self.research_engine = ResearchEngine(memory_manager=self.memory_manager)
        self.discord_interface = DiscordInterface(CHANNEL_ID)
        self.benchmark_manager = BenchmarkManager()

        # Set Discord interface in research engine
        self.research_engine.set_discord_interface(self.discord_interface)

    async def on_ready(self):
        """Called when the bot is ready and connected."""
        logger.info(f"{self.name} ({self.user}) has connected to Discord!")

    async def setup_hook(self):
        """Called before the bot starts running."""
        logger.info(f"Setting up {self.name}...")

    async def get_topic(self, research_input: str) -> Dict[str, str]:
        """Generate a focused research topic from user input."""
        try:
            # Convert string prompts to ChatPromptTemplate
            topic_chain = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a research topic generator."),
                HumanMessagePromptTemplate.from_template(topic_prompt)
            ])

            # Get current time in UTC
            current_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

            # Generate topic with required variables
            topic_result = await self.research_engine.execute_chain(
                topic_chain,
                {
                    "user_input": research_input,
                    "current_time": current_time
                }
            )

            # Validate against schema
            validate(instance=topic_result, schema=topic_schema)

            return topic_result

        except ValidationError as e:
            logger.error(f"Topic validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error generating topic: {e}")
            raise

    def clean_thread_name(self, topic_str: str, run_id: str) -> str:
        """Clean and validate thread name to meet Discord requirements."""
        if not topic_str or len(topic_str.strip()) == 0:
            return f"Research-{run_id[:8]}"

        # Remove special characters and extra whitespace
        clean_name = re.sub(r'[^\w\s-]', '', topic_str)
        clean_name = re.sub(r'\s+', '-', clean_name).strip()

        # Ensure minimum length and valid start character
        if len(clean_name) < 1 or not clean_name[0].isalnum():
            clean_name = f"Research-{clean_name}"

        # Truncate if too long (leave room for uniqueness suffix)
        if len(clean_name) > 80:
            clean_name = clean_name[:77] + "..."

        # Ensure no double hyphens and clean up edges
        clean_name = re.sub(r'-+', '-', clean_name)
        clean_name = clean_name.strip('-')

        # Final validation - if still invalid, use fallback
        if len(clean_name) < 1:
            clean_name = f"Research-{run_id[:8]}"

        return clean_name

    @commands.command(name="research")
    async def start_research(self, ctx: commands.Context) -> None:
        """Start a new research thread and process."""
        run_id = str(uuid.uuid4())
        benchmark_data = BenchmarkData(run_id)

        if ctx.channel.id != CHANNEL_ID:
            return

        try:
            async with ctx.channel.typing():
                # Handle message content and attachments
                user_input = ctx.message.content.replace("!research", "").strip()
                if not user_input:
                    await ctx.send("Please provide a research topic or question.")
                    return

                # Handle attachments if present
                attachment_contents = []
                if ctx.message.attachments:
                    for attachment in ctx.message.attachments:
                        if attachment.filename.lower().endswith(('.txt', '.md')):
                            content = await self.read_text_attachment(attachment)
                            if content:
                                attachment_contents.append(content)

                # Combine message content with attachment contents
                combined_input = user_input
                if attachment_contents:
                    combined_input = user_input + "\n\nAttached Content:\n" + "\n---\n".join(attachment_contents)

                try:
                    research_topic = await self.get_topic(combined_input)
                except (ValueError, ValidationError) as e:
                    await ctx.send(f"Error generating research topic: {e}")
                    return

                # Get topic string and clean thread name
                topic_str = research_topic.get('topic', '') if isinstance(research_topic, dict) else str(research_topic)
                base_thread_name = self.clean_thread_name(topic_str, run_id)

                thinking_embed = Embed(
                    title="Research Initialization",
                    description="The research team is analyzing your request...",
                    color=Color.blue()
                )
                thinking_message = await ctx.message.channel.send(embed=thinking_embed)

                # Check if channel supports threads
                channel = ctx.message.channel
                if not isinstance(channel, (TextChannel, ForumChannel)):
                    await ctx.send("This command can only be used in text or forum channels.")
                    return

                # Create thread with name and auto_archive_duration
                thread = await channel.create_thread(
                    name=f"Research: {research_topic}",
                    auto_archive_duration=1440  # 24 hours
                )

                # Store thread ID for later use
                self.active_threads[str(thread.id)] = {
                    "topic": research_topic,
                    "user": ctx.author.id,
                    "created_at": datetime.now(UTC).isoformat()
                }

                await thinking_message.delete()

                # Just acknowledge the attachments without resending them
                if attachment_contents:
                    await thread.send(embed=Embed(
                        title="Processing Research Input",
                        description="Analyzing message and attached documents...",
                        color=Color.blue()
                    ))

                # Use the new agent selection process
                selected_agent = await self.agent_manager.select_specialized_agent(research_topic)

                # Send agent info in smaller chunks
                agent_info = {
                    "name": selected_agent.get('name', 'Unknown Agent'),
                    "expertise": selected_agent.get('description', 'No description available')
                }

                # Split long descriptions
                if len(agent_info["expertise"]) > 3000:  # Leave room for formatting
                    agent_info["expertise"] = agent_info["expertise"][:2997] + "..."

                await thread.send(
                    embed=Embed(
                        title="Selected Specialized Agent",
                        description=f"**Agent:** {agent_info['name']}\n\n**Expertise:**\n{agent_info['expertise']}",
                        color=Color.green()
                    )
                )

                # Initialize research context
                step_context_state: State = {
                    "messages": [],  # Required field for messages
                    "task": Task(  # Required field for task
                        id=str(uuid.uuid4()),
                        description=str(research_topic),  # Convert to string
                        tool=Tool(
                            name="research",
                            description="Execute research task",
                            parameters={}
                        ),
                        args={},
                        dependencies=[]
                    ),
                    "metrics": {  # Required field for metrics
                        "current_depth": 0,
                        "quality_history": [],
                        "completed_substeps": [],
                        "agent_responses": [],
                        "supervisor_feedback": [],
                        "depth_insights": []
                    }
                }

                # Convert State to Dict[str, Any] for execute_agent
                step_context = {
                    "messages": step_context_state["messages"],
                    "task": step_context_state["task"].model_dump(),
                    "metrics": step_context_state["metrics"]
                }

                # Store research topic in benchmark data
                benchmark_data.research_topic = json.dumps(research_topic)
                benchmark_data.user_input = json.dumps(combined_input)

                try:
                    # PHASE 1: Initial Brainstorming and Planning
                    # Convert string prompts to ChatPromptTemplate
                    topic_chain = ChatPromptTemplate.from_messages([
                        SystemMessage(content="You are a research topic generator."),
                        HumanMessagePromptTemplate.from_template(cot32_prompt)
                    ])

                    # Generate initial brainstorming
                    initial_plan = await self.research_engine.execute_chain(
                        topic_chain,
                        {
                            "research_topic": research_topic,
                            "task": combined_input,
                            "agent_indices": [f"{k}: {v.get('name', '')}" for k, v in
                                              self.agent_manager.specialized_personas.items()],
                            "previous_research_summary": ""
                        }
                    )

                    # Validate and format initial plan
                    initial_plan_embeds = await self.discord_interface.format_brainstorming_output(initial_plan,
                                                                                                   "Initial Research Plan")
                    for embed in initial_plan_embeds:
                        await thread.send(embed=embed)
                        await asyncio.sleep(0.5)

                    # PHASE 2: Create Detailed Research Plan
                    # Convert string prompts to ChatPromptTemplate
                    final_plan_chain = ChatPromptTemplate.from_messages([
                        SystemMessage(content="You are creating a detailed research plan."),
                        HumanMessagePromptTemplate.from_template(final_plan_prompt)
                    ])

                    # Generate final research plan
                    final_plan = await self.research_engine.execute_chain(
                        final_plan_chain,
                        {
                            "agent_indices": [f"{k}: {v.get('name')}" for k, v in
                                              self.agent_manager.specialized_personas.items()],
                            "tool_indices": [f"{k}: {v}" for k, v in self.research_engine.TOOLS.items()],
                            "initial_plan": json.dumps(initial_plan, indent=2),
                            "updates": "",
                            "agent_id": selected_agent.get('agent_id')
                        }
                    )

                    # Validate and format final plan
                    final_plan_embeds = await self.discord_interface.format_research_plan(final_plan,
                                                                                          "Detailed Research Plan")
                    for embed in final_plan_embeds:
                        await thread.send(embed=embed)
                        await asyncio.sleep(0.5)

                    # Store final plan in benchmark data
                    benchmark_data.final_plan = json.dumps(final_plan)

                    # PHASE 3: Execute Research Steps
                    for step in final_plan["plan"]:
                        result, evaluation = await self.execute_step(step, thread, step["step_number"],
                                                                     final_plan["plan"])

                        # Store step evaluation
                        benchmark_data.step_evaluations.append({
                            "step": step["step_number"],
                            "agent": selected_agent.get('name'),
                            "result": result,
                            "evaluation": evaluation
                        })

                        # Execute agent with depth tracking
                        agent_actions = [{
                            "reasoning": step["reasoning"],
                            "next_steps": [{
                                "action": step["action"],
                                "tool_suggestions": step["tool_suggestions"],
                                "implementation_notes": step["implementation_notes"]
                            }]
                        }]

                        agent, actions, current_progress, completed_substeps, collaboration_request = await self.agent_manager.execute_agent(
                            selected_agent,
                            agent_actions,
                            thread,
                            step["step_number"],
                            step_context,
                            final_plan["plan"]
                        )

                        if collaboration_request:
                            # Handle collaboration request by updating the research plan
                            final_plan["plan"] = self.agent_manager.update_research_plan(
                                final_plan["plan"],
                                collaboration_request,
                                current_progress,
                                selected_agent
                            )

                    # Save benchmark data
                    await self.benchmark_manager.save_benchmark(benchmark_data)

                except ValidationError as e:
                    logger.error(f"Validation error in research planning: {e}")
                    await thread.send(
                        embed=Embed(
                            title="Research Planning Error",
                            description=f"Invalid research plan format: {str(e)}",
                            color=Color.red()
                        )
                    )
                    return
                except Exception as e:
                    logger.error(f"Error in research planning/execution: {e}")
                    await thread.send(
                        embed=Embed(
                            title="Research Error",
                            description=f"An error occurred during research planning/execution: {str(e)}",
                            color=Color.red()
                        )
                    )
                    return

        except Exception as e:
            logger.error(f"Error in start_research: {e}")
            await ctx.send(f"An unexpected error occurred: {str(e)}")
            return

    async def format_and_send_chunks(self, thread: discord.Thread, content: Union[str, Dict[str, Any]], title: str = "",
                                     chunk_size: int = 1900) -> None:
        """Format and send content in chunks that respect Discord's message limits.
        
        Args:
            thread: Discord thread to send messages to
            content: Content to send (string or dictionary)
            title: Optional title for the message
            chunk_size: Maximum size of each chunk (default 1900 to leave room for formatting)
        """
        try:
            # Validate inputs
            if not thread:
                logger.error("No thread provided for sending message")
                return

            if not content:
                logger.warning("Empty content provided for message")
                return

            # Discord limits
            MAX_MESSAGE_LENGTH = 2000
            MAX_EMBED_LENGTH = 4096
            MAX_EMBED_TOTAL = 6000

            # Convert content to string if it's a dictionary
            if isinstance(content, dict):
                try:
                    content = json.dumps(content, indent=2, default=str)
                except Exception as e:
                    logger.error(f"Failed to convert dictionary to string: {e}")
                    content = str(content)
            else:
                content = str(content)

            # If content is small enough, try sending as a single embed first
            if len(content) <= MAX_EMBED_LENGTH and len(title) + len(content) <= MAX_EMBED_TOTAL:
                try:
                    embed = Embed(
                        title=title[:256] if title else None,  # Discord title limit
                        description=content,
                        color=Color.blue()
                    )
                    await thread.send(embed=embed)
                    return
                except discord.HTTPException as e:
                    logger.warning(f"Failed to send as embed, falling back to text chunks: {e}")

            # Calculate space needed for title/formatting
            title_space = len(title) + 4 if title else 0  # Add some padding for formatting
            effective_chunk_size = min(chunk_size, MAX_MESSAGE_LENGTH - title_space)

            # Split content into chunks
            chunks = []
            lines = content.split('\n')
            current_chunk = []
            current_length = 0

            for line in lines:
                line_length = len(line) + 1  # +1 for newline

                # If this single line is too long, split it
                if line_length > effective_chunk_size:
                    if current_chunk:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = []
                        current_length = 0

                    # Split long line into smaller pieces
                    for i in range(0, len(line), effective_chunk_size):
                        sub_line = line[i:i + effective_chunk_size]
                        if len(sub_line) >= effective_chunk_size:
                            chunks.append(sub_line)
                        else:
                            current_chunk = [sub_line]
                            current_length = len(sub_line) + 1

                # If adding this line would exceed chunk size, start new chunk
                elif current_length + line_length > effective_chunk_size:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_length = line_length
                else:
                    current_chunk.append(line)
                    current_length += line_length

            # Add final chunk if exists
            if current_chunk:
                chunks.append('\n'.join(current_chunk))

            # Ensure we have at least one chunk
            if not chunks:
                chunks = ["<empty message>"]

            # Send chunks with proper formatting
            for i, chunk in enumerate(chunks, 1):
                try:
                    # Create chunk title
                    chunk_title = f"{title} (Part {i}/{len(chunks)})" if len(chunks) > 1 else title

                    # Try sending as embed first if chunk is small enough
                    if len(chunk) <= MAX_EMBED_LENGTH and len(chunk_title or "") + len(chunk) <= MAX_EMBED_TOTAL:
                        try:
                            embed = Embed(
                                title=chunk_title[:256] if chunk_title else None,
                                description=chunk,
                                color=Color.blue()
                            )
                            await thread.send(embed=embed)
                            await asyncio.sleep(0.5)  # Rate limiting protection
                            continue
                        except discord.HTTPException:
                            pass  # Fall back to regular message

                    # Send as regular message
                    final_content = f"**{chunk_title}**\n{chunk}" if chunk_title else chunk
                    if len(final_content) > MAX_MESSAGE_LENGTH:
                        logger.warning(f"Message chunk {i} exceeds {MAX_MESSAGE_LENGTH} chars, truncating...")
                        final_content = final_content[:MAX_MESSAGE_LENGTH - 3] + "..."

                    await thread.send(final_content)
                    await asyncio.sleep(0.5)  # Rate limiting protection

                except discord.HTTPException as e:
                    logger.error(f"Discord API error sending chunk {i}: {e}")
                    # Try one last time with minimal content
                    try:
                        await thread.send(f"Error sending full message. Partial content:\n{chunk[:1000]}...")
                    except:
                        logger.error(f"Failed to send even truncated message for chunk {i}")

        except Exception as e:
            logger.error(f"Error in format_and_send_chunks: {e}")
            # Try to send a simple error message
            try:
                await thread.send("Error formatting message. Please check the logs for details.")
            except:
                logger.error("Failed to send error message")

    async def execute_step(self, step: Dict[str, Any], thread: discord.Thread, step_number: int,
                           final_plan: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute a single step of the research plan."""
        try:
            # Get agent for this step
            agent_id = step["agent"][0] if isinstance(step["agent"], list) else step["agent"]
            agent_id, agent = await self.agent_manager.select_specialized_agent_by_id(agent_id)

            # Initialize step context
            step_context: State = {
                "messages": [],  # Required field for messages
                "task": Task(  # Required field for task
                    id=str(uuid.uuid4()),
                    description=step["step_number"],
                    tool=Tool(
                        name="research",
                        description="Execute research task",
                        parameters={}
                    ),
                    args={},
                    dependencies=[]
                ),
                "metrics": {  # Required field for metrics
                    "current_depth": 0,
                    "quality_history": [],
                    "completed_substeps": [],
                    "agent_responses": [],
                    "supervisor_feedback": [],
                    "depth_insights": []
                }
            }

            # Execute agent
            result, actions, progress, substeps, collab = await self.execute_agent(
                agent,
                [step],
                thread,
                step_number,
                step_context,
                final_plan
            )

            # Return only the result and evaluation
            return result, {"progress": progress}

        except Exception as e:
            logger.error(f"Error executing step {step_number}: {e}")
            await self.format_and_send_chunks(thread, f"❌ Error executing step {step_number}: {str(e)}")
            return None, None

    async def read_text_attachment(self, attachment: Attachment) -> str:
        """Read content from a text file attachment."""
        try:
            if attachment.filename.lower().endswith(('.txt', '.md')):
                async with aiohttp.ClientSession() as session:
                    async with session.get(attachment.url) as response:
                        if response.status == 200:
                            content = await response.text()
                            return content
                        else:
                            logger.error(f"Failed to fetch attachment: {response.status}")
                            return ""
            return ""
        except Exception as e:
            logger.error(f"Error reading attachment: {e}")
            return ""

    async def execute_agent(self, agent: Dict[str, Any], actions: List[Dict[str, Any]], thread: discord.Thread,
                            step_number: int, step_context: Dict[str, Any], final_plan: List[Dict[str, Any]]) -> Tuple[
        Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Execute an agent with the given actions."""
        try:
            log_operation("AGENT_EXECUTION_START",
                          agent_name=agent.get('name'),
                          step_number=step_number,
                          actions=actions,
                          context=step_context
                          )

            logger.info(f"Executing agent {agent.get('name')} for task: {json.dumps(actions)}")

            # Send initial agent message
            await thread.send(f"🤖 **{agent.get('name')}** analyzing step {step_number}...")

            # Get relevant memories first
            implementation_notes = ""
            for action in actions:
                if "next_steps" in action:
                    for step in action["next_steps"]:
                        if "implementation_notes" in step:
                            implementation_notes = step["implementation_notes"]
                            break
                if implementation_notes:
                    break

            logger.info(f"Retrieving relevant memories for: {implementation_notes}")
            relevant_memories = await self.memory_manager.get_relevant_memories(implementation_notes)

            # Always perform a search first to gather context
            logger.info("Performing initial search for context")
            await thread.send("🔍 Gathering research context...")
            search_results = await self.research_engine.execute_tools([1], implementation_notes)

            if isinstance(search_results, dict) and "1" in search_results:
                search_data = search_results["1"]
                if isinstance(search_data, dict) and "results" in search_data:
                    # Format and send search results
                    search_embed = Embed(
                        title="Research Context",
                        color=Color.blue()
                    )
                    for result in search_data["results"][:5]:  # Show top 5 results
                        search_embed.add_field(
                            name=result.get("title", "Untitled"),
                            value=f"{result.get('snippet', 'No snippet available')}\n[Source]({result.get('url', '#')})",
                            inline=False
                        )
                    await thread.send(embed=search_embed)

                    # Store search results in memory
                    await self.memory_manager.store_tool_result(
                        tool_type="search",
                        result=search_data,
                        metadata={
                            "step": step_number,
                            "agent": agent.get("name"),
                            "timestamp": datetime.now(UTC).isoformat()
                        }
                    )

            # Extract tool suggestions from actions
            tool_suggestions = []
            for action in actions:
                if "next_steps" in action:
                    for step in action["next_steps"]:
                        if "tool_suggestions" in step:
                            tool_suggestions.extend(step["tool_suggestions"])

            # Add search tool if not already included
            if 1 not in tool_suggestions:
                tool_suggestions.insert(0, 1)  # Always include search

            log_operation("TOOL_EXECUTION_START",
                          tool_suggestions=tool_suggestions,
                          implementation_notes=implementation_notes
                          )

            # Execute tools if any are suggested
            tool_results = {}
            if tool_suggestions:
                try:
                    logger.info(f"Executing tools {tool_suggestions} with notes: {implementation_notes}")
                    await thread.send(f"🔍 Executing research tools...")
                    tool_results = await self.research_engine.execute_tools(tool_suggestions, implementation_notes)

                    log_operation("TOOL_RESULTS",
                                  results=tool_results
                                  )

                    # Format and send each tool result
                    for tool_id, result in tool_results.items():
                        tool_name = self.research_engine.TOOLS.get(tool_id, "Unknown Tool")
                        log_operation("FORMATTING_TOOL_RESULT",
                                      tool_id=tool_id,
                                      tool_name=tool_name,
                                      result=result
                                      )

                        embed = Embed(
                            title=f"Tool Results: {tool_name}",
                            color=Color.blue()
                        )

                        if isinstance(result, dict):
                            for key, value in result.items():
                                if isinstance(value, (list, dict)):
                                    embed.add_field(
                                        name=key.replace("_", " ").title(),
                                        value=json.dumps(value, indent=2)[:1024],
                                        inline=False
                                    )
                                else:
                                    embed.add_field(
                                        name=key.replace("_", " ").title(),
                                        value=str(value)[:1024],
                                        inline=True
                                    )

                        await thread.send(embed=embed)

                        # Store each tool result in memory
                        await self.memory_manager.store_tool_result(
                            tool_type=str(tool_id),
                            result=result,
                            metadata={
                                "step": step_number,
                                "agent": agent.get("name"),
                                "timestamp": datetime.now(UTC).isoformat()
                            }
                        )

                except Exception as e:
                    logger.error(f"Error executing tools: {e}", exc_info=True)
                    await thread.send(f"⚠️ Error executing tools: {str(e)}")
                    tool_results = {"error": str(e)}

            # Create base analysis structure
            base_analysis = {
                "technical_analysis": {
                    "depth": step_context["metrics"]["current_depth"] + 1,
                    "insights": [],
                    "dependencies": []
                },
                "implementation_aspects": [],
                "substeps": [],
                "quality_metrics": {
                    "technical_depth": 0,
                    "implementation_detail": 0,
                    "validation_coverage": 0,
                    "dependency_completeness": 0
                }
            }

            log_operation("AGENT_CHAIN_START",
                          agent_description=agent.get("description", ""),
                          step_number=step_number,
                          base_analysis=base_analysis
                          )

            # Execute agent chain through agent manager with memory context
            agent_chain = create_chain(
                """You are an expert agent specializing in {agent_description}.
                
                Current step: {step_number}
                Actions to take: {actions}
                Tool results: {tool_results}
                Current depth: {current_depth}
                Previous relevant context: {relevant_memories}
                
                Consider the previous context and tool results to provide:
                1. Detailed technical analysis incorporating search results and previous findings
                2. Implementation aspects considering past insights
                3. Required substeps that build on previous work
                4. Quality metrics for the analysis
                
                Format your response as a JSON object matching this structure:
                {base_analysis}
                """,
                output_mode='json'
            )

            result = await agent_chain.ainvoke({
                "agent_description": agent.get("description", ""),
                "step_number": step_number,
                "actions": json.dumps(actions, indent=2),
                "tool_results": json.dumps(tool_results, indent=2),
                "current_depth": step_context["metrics"]["current_depth"],
                "base_analysis": json.dumps(base_analysis, indent=2),
                "relevant_memories": json.dumps(relevant_memories, indent=2)
            })

            log_operation("AGENT_CHAIN_RESULT",
                          result=result
                          )

            # Store agent's analysis in memory
            await self.memory_manager.store_agent_response(
                agent_id=agent.get("id", 0),
                response=result,
                metadata={
                    "step": step_number,
                    "timestamp": datetime.now(UTC).isoformat()
                }
            )

            # Ensure result has required structure
            if not isinstance(result, dict):
                logger.error(f"Agent returned invalid result type: {type(result)}")
                result = base_analysis

            # Update step context with results
            step_context["metrics"]["current_depth"] += 1

            # Add new quality metrics
            quality_metrics = result.get("quality_metrics", {})
            quality_metrics.update({
                "step": step_number,
                "depth": step_context["metrics"]["current_depth"],
                "timestamp": datetime.now(UTC).isoformat()
            })
            step_context["metrics"]["quality_history"].append(quality_metrics)

            log_operation("CONTEXT_UPDATE",
                          new_depth=step_context["metrics"]["current_depth"],
                          quality_metrics=quality_metrics,
                          quality_history=step_context["metrics"]["quality_history"]
                          )

            # Update substeps
            step_context["metrics"]["completed_substeps"].extend(result.get("substeps", []))

            # Format and send agent analysis results
            analysis_embed = Embed(
                title=f"Analysis Results - {agent.get('name')}",
                color=Color.green()
            )

            # Add technical analysis
            tech_analysis = result.get("technical_analysis", {})
            if tech_analysis.get("insights"):
                analysis_embed.add_field(
                    name="Technical Insights",
                    value="\n".join(f"• {insight}" for insight in tech_analysis["insights"][:5])[:1024],
                    inline=False
                )

            # Add implementation aspects
            impl_aspects = result.get("implementation_aspects", [])
            if impl_aspects:
                analysis_embed.add_field(
                    name="Implementation Aspects",
                    value="\n".join(f"• {aspect}" for aspect in impl_aspects[:5])[:1024],
                    inline=False
                )

            # Add quality metrics
            metrics = result.get("quality_metrics", {})
            metrics_text = "\n".join(f"• {k.replace('_', ' ').title()}: {v:.2f}"
                                     for k, v in metrics.items()
                                     if isinstance(v, (int, float)))
            if metrics_text:
                analysis_embed.add_field(
                    name="Quality Metrics",
                    value=metrics_text[:1024],
                    inline=False
                )

            await thread.send(embed=analysis_embed)

            log_operation("AGENT_EXECUTION_COMPLETE",
                          result=result,
                          actions=actions,
                          context=step_context
                          )

            return result, actions, step_context, [], []

        except Exception as e:
            logger.error(f"Error executing agent: {e}", exc_info=True)
            await thread.send(f"❌ Error executing agent: {str(e)}")
            raise

    async def format_brainstorming_output(self, plan: Dict[str, Any], title: str) -> List[Embed]:
        """Format brainstorming output into multiple embeds if needed."""
        embeds = []

        # Split content into sections
        if isinstance(plan, dict):
            # Overview embed
            overview = Embed(
                title=f"{title} - Overview",
                color=Color.blue()
            )
            if 'topic' in plan:
                overview.add_field(
                    name="Research Topic",
                    value=plan['topic'][:1024],
                    inline=False
                )
            if 'reasoning' in plan:
                overview.add_field(
                    name="Rationale",
                    value=plan['reasoning'][:1024],
                    inline=False
                )
            embeds.append(overview)

            # Key aspects embed
            if 'key_aspects' in plan and plan['key_aspects']:
                aspects = plan['key_aspects']
                if isinstance(aspects, list):
                    # Split aspects into chunks of 5
                    for i in range(0, len(aspects), 5):
                        aspect_chunk = aspects[i:i + 5]
                        aspect_embed = Embed(
                            title=f"{title} - Key Aspects (Part {i // 5 + 1})",
                            color=Color.blue()
                        )
                        for j, aspect in enumerate(aspect_chunk, 1):
                            if isinstance(aspect, str):
                                aspect_embed.add_field(
                                    name=f"Aspect {i + j}",
                                    value=aspect[:1024],
                                    inline=False
                                )
                        embeds.append(aspect_embed)

            # Steps embed
            if 'steps' in plan and plan['steps']:
                steps = plan['steps']
                if isinstance(steps, list):
                    # Split steps into chunks of 3
                    for i in range(0, len(steps), 3):
                        step_chunk = steps[i:i + 3]
                        step_embed = Embed(
                            title=f"{title} - Research Steps (Part {i // 3 + 1})",
                            color=Color.blue()
                        )
                        for j, step in enumerate(step_chunk, 1):
                            if isinstance(step, dict):
                                step_desc = f"**Action:** {step.get('action', 'No action specified')}\n"
                                if 'reasoning' in step:
                                    step_desc += f"**Rationale:** {step['reasoning'][:500]}..."
                                step_embed.add_field(
                                    name=f"Step {i + j}",
                                    value=step_desc[:1024],
                                    inline=False
                                )
                        embeds.append(step_embed)

        # Ensure no embed exceeds limits
        for embed in embeds:
            total_length = len(embed.title or "")
            for field in embed.fields:
                total_length += len(field.name) + len(field.value)
                if len(field.value) > 1024:
                    field.value = field.value[:1021] + "..."
            if total_length > 6000:
                # Remove fields until under limit
                while total_length > 5500 and embed.fields:
                    field = embed.fields.pop()
                    total_length -= len(field.name) + len(field.value)

        return embeds

    async def format_research_plan(self, plan: Dict[str, Any], title: str) -> List[Embed]:
        """Format research plan into multiple embeds."""
        embeds = []

        if isinstance(plan, dict):
            # Overview embed
            overview = Embed(
                title=f"{title} - Overview",
                color=Color.blue()
            )
            if 'objective' in plan:
                overview.add_field(
                    name="Research Objective",
                    value=plan['objective'][:1024],
                    inline=False
                )
            embeds.append(overview)

            # Plan steps
            if 'plan' in plan and isinstance(plan['plan'], list):
                # Split steps into chunks of 2
                steps = plan['plan']
                for i in range(0, len(steps), 2):
                    step_chunk = steps[i:i + 2]
                    step_embed = Embed(
                        title=f"{title} - Steps {i + 1}-{min(i + 2, len(steps))}",
                        color=Color.blue()
                    )
                    for step in step_chunk:
                        if isinstance(step, dict):
                            step_desc = f"**Action:** {step.get('action', 'No action specified')}\n"
                            if 'reasoning' in step:
                                step_desc += f"**Rationale:** {step['reasoning'][:500]}\n"
                            if 'implementation_notes' in step:
                                step_desc += f"**Notes:** {step['implementation_notes'][:300]}..."
                            step_embed.add_field(
                                name=f"Step {step.get('step_number', '?')}",
                                value=step_desc[:1024],
                                inline=False
                            )
                    embeds.append(step_embed)

        # Ensure no embed exceeds limits
        for embed in embeds:
            total_length = len(embed.title or "")
            for field in embed.fields:
                total_length += len(field.name) + len(field.value)
                if len(field.value) > 1024:
                    field.value = field.value[:1021] + "..."
            if total_length > 6000:
                # Remove fields until under limit
                while total_length > 5500 and embed.fields:
                    field = embed.fields.pop()
                    total_length -= len(field.name) + len(field.value)

        return embeds

    async def get_memories(self, query: str) -> Optional[Dict[str, Any]]:
        """Get relevant memories based on a query."""
        try:
            return await self.memory_manager.get_memory(query)
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return None

    async def store_result(self, operation_id: str, tool_name: str, result: Any, error: Optional[str] = None) -> None:
        """Store a tool result in memory."""
        try:
            await self.memory_manager.store_tool_result(operation_id, tool_name, result, error)
        except Exception as e:
            logger.error(f"Error storing result: {e}")

    async def store_task_data(self, task_id: str, task_data: Dict[str, Any]) -> bool:
        """Store task data in memory."""
        try:
            return await self.memory_manager.store_task(task_id, task_data)
        except Exception as e:
            logger.error(f"Error storing task data: {e}")
            return False

    async def get_task_data(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task data from memory."""
        try:
            return await self.memory_manager.get_task(task_id)
        except Exception as e:
            logger.error(f"Error getting task data: {e}")
            return None

    def get_dict_value(self, d: Dict[str, Any], key: Union[str, int], default: Any = None) -> Any:
        """Safely get a value from a dictionary."""
        if isinstance(d, dict):
            return d.get(str(key), default)
        return default

    async def execute_agent(self, agent: Dict[str, Any], actions: List[Dict[str, Any]], thread: Any, step_number: int, step_context: Dict[str, Any], final_plan: Dict[str, Any]) -> None:
        """Execute an agent with the given actions."""
        try:
            # Convert final_plan to Dict[str, Any] if it's a list
            if isinstance(final_plan, list):
                final_plan = {"steps": final_plan}

            await self.agent_manager.execute_agent(agent, actions, thread, step_number, step_context, final_plan)
        except Exception as e:
            logger.error(f"Error executing agent: {e}")

    async def update_research_plan(self, requesting_agent: Dict[str, Any], feedback: str, error_analysis: Optional[str] = None) -> None:
        """Update the research plan based on agent feedback."""
        try:
            await self.agent_manager.update_research_plan(requesting_agent, feedback, error_analysis)
        except Exception as e:
            logger.error(f"Error updating research plan: {e}")


# Initialize main bot only - we won't run multiple instances
try:
    research_lead = FUNCTION_PERSONAS.get(1)
    if not research_lead:
        raise ValueError("Research Lead persona not found")
    bot = ResearchBot(research_lead)


    # Add the research command to the bot
    @bot.command(name="research")
    async def start_research(ctx: commands.Context) -> None:
        """Start a new research thread and process."""
        run_id = str(uuid.uuid4())
        benchmark_data = BenchmarkData(run_id)

        if ctx.channel.id != CHANNEL_ID:
            return

        try:
            async with ctx.channel.typing():
                # Handle message content and attachments
                user_input = ctx.message.content.replace("!research", "").strip()
                if not user_input:
                    await ctx.send("Please provide a research topic or question.")
                    return

                # Handle attachments if present
                attachment_contents = []
                if ctx.message.attachments:
                    for attachment in ctx.message.attachments:
                        if attachment.filename.lower().endswith(('.txt', '.md')):
                            content = await bot.read_text_attachment(attachment)
                            if content:
                                attachment_contents.append(content)

                # Combine message content with attachment contents
                combined_input = user_input
                if attachment_contents:
                    combined_input = user_input + "\n\nAttached Content:\n" + "\n---\n".join(attachment_contents)

                try:
                    research_topic = await bot.get_topic(combined_input)
                except (ValueError, ValidationError) as e:
                    await ctx.send(f"Error generating research topic: {e}")
                    return

                # Get topic string and clean thread name
                topic_str = research_topic.get('topic', '') if isinstance(research_topic, dict) else str(research_topic)
                base_thread_name = bot.clean_thread_name(topic_str, run_id)

                thinking_embed = Embed(
                    title="Research Initialization",
                    description="The research team is analyzing your request...",
                    color=Color.blue()
                )
                thinking_message = await ctx.message.channel.send(embed=thinking_embed)

                # Check if channel supports threads
                channel = ctx.message.channel
                if not isinstance(channel, (TextChannel, ForumChannel)):
                    await ctx.send("This command can only be used in text or forum channels.")
                    return

                # Create thread with name and auto_archive_duration
                thread = await channel.create_thread(
                    name=f"Research: {research_topic}",
                    auto_archive_duration=1440  # 24 hours
                )

                # Store thread ID for later use
                bot.active_threads[str(thread.id)] = {
                    "topic": research_topic,
                    "user": ctx.author.id,
                    "created_at": datetime.now(UTC).isoformat()
                }

                await thinking_message.delete()

                # Just acknowledge the attachments without resending them
                if attachment_contents:
                    await thread.send(embed=Embed(
                        title="Processing Research Input",
                        description="Analyzing message and attached documents...",
                        color=Color.blue()
                    ))

                # Use the new agent selection process
                selected_agent = await bot.agent_manager.select_specialized_agent(research_topic)

                # Send agent info in smaller chunks
                agent_info = {
                    "name": selected_agent.get('name', 'Unknown Agent'),
                    "expertise": selected_agent.get('description', 'No description available')
                }

                # Split long descriptions
                if len(agent_info["expertise"]) > 3000:  # Leave room for formatting
                    agent_info["expertise"] = agent_info["expertise"][:2997] + "..."

                await thread.send(
                    embed=Embed(
                        title="Selected Specialized Agent",
                        description=f"**Agent:** {agent_info['name']}\n\n**Expertise:**\n{agent_info['expertise']}",
                        color=Color.green()
                    )
                )

                # Initialize research context
                step_context_state: State = {
                    "messages": [],  # Required field for messages
                    "task": Task(  # Required field for task
                        id=str(uuid.uuid4()),
                        description=str(research_topic),  # Convert to string
                        tool=Tool(
                            name="research",
                            description="Execute research task",
                            parameters={}
                        ),
                        args={},
                        dependencies=[]
                    ),
                    "metrics": {  # Required field for metrics
                        "current_depth": 0,
                        "quality_history": [],
                        "completed_substeps": [],
                        "agent_responses": [],
                        "supervisor_feedback": [],
                        "depth_insights": []
                    }
                }

                # Convert State to Dict[str, Any] for execute_agent
                step_context = {
                    "messages": step_context_state["messages"],
                    "task": step_context_state["task"].model_dump(),
                    "metrics": step_context_state["metrics"]
                }

                # Store research topic in benchmark data
                benchmark_data.research_topic = json.dumps(research_topic)
                benchmark_data.user_input = json.dumps(combined_input)

                try:
                    # PHASE 1: Initial Brainstorming and Planning
                    # Convert string prompts to ChatPromptTemplate
                    topic_chain = ChatPromptTemplate.from_messages([
                        SystemMessage(content="You are a research topic generator."),
                        HumanMessagePromptTemplate.from_template(cot32_prompt)
                    ])

                    # Generate initial brainstorming
                    initial_plan = await bot.research_engine.execute_chain(
                        topic_chain,
                        {
                            "research_topic": research_topic,
                            "task": combined_input,
                            "agent_indices": [f"{k}: {v.get('name', '')}" for k, v in
                                              bot.agent_manager.specialized_personas.items()],
                            "previous_research_summary": ""
                        }
                    )

                    # Validate and format initial plan
                    initial_plan_embeds = await bot.discord_interface.format_brainstorming_output(initial_plan,
                                                                                                  "Initial Research Plan")
                    for embed in initial_plan_embeds:
                        await thread.send(embed=embed)
                        await asyncio.sleep(0.5)

                    # PHASE 2: Create Detailed Research Plan
                    # Convert string prompts to ChatPromptTemplate
                    final_plan_chain = ChatPromptTemplate.from_messages([
                        SystemMessage(content="You are creating a detailed research plan."),
                        HumanMessagePromptTemplate.from_template(final_plan_prompt)
                    ])

                    # Generate final research plan
                    final_plan = await bot.research_engine.execute_chain(
                        final_plan_chain,
                        {
                            "agent_indices": [f"{k}: {v.get('name')}" for k, v in
                                              bot.agent_manager.specialized_personas.items()],
                            "tool_indices": [f"{k}: {v}" for k, v in bot.research_engine.TOOLS.items()],
                            "initial_plan": json.dumps(initial_plan, indent=2),
                            "updates": "",
                            "agent_id": selected_agent.get('agent_id')
                        }
                    )

                    # Validate and format final plan
                    final_plan_embeds = await bot.discord_interface.format_research_plan(final_plan,
                                                                                         "Detailed Research Plan")
                    for embed in final_plan_embeds:
                        await thread.send(embed=embed)
                        await asyncio.sleep(0.5)

                    # Store final plan in benchmark data
                    benchmark_data.final_plan = json.dumps(final_plan)

                    # PHASE 3: Execute Research Steps
                    for step in final_plan["plan"]:
                        result, evaluation = await bot.execute_step(step, thread, step["step_number"],
                                                                    final_plan["plan"])

                        # Store step evaluation
                        benchmark_data.step_evaluations.append({
                            "step": step["step_number"],
                            "agent": selected_agent.get('name'),
                            "result": result,
                            "evaluation": evaluation
                        })

                        # Execute agent with depth tracking
                        agent_actions = [{
                            "reasoning": step["reasoning"],
                            "next_steps": [{
                                "action": step["action"],
                                "tool_suggestions": step["tool_suggestions"],
                                "implementation_notes": step["implementation_notes"]
                            }]
                        }]

                        agent, actions, current_progress, completed_substeps, collaboration_request = await bot.agent_manager.execute_agent(
                            selected_agent,
                            agent_actions,
                            thread,
                            step["step_number"],
                            step_context,
                            final_plan["plan"]
                        )

                        if collaboration_request:
                            # Handle collaboration request by updating the research plan
                            final_plan["plan"] = bot.agent_manager.update_research_plan(
                                final_plan["plan"],
                                collaboration_request,
                                current_progress,
                                selected_agent
                            )

                    # Save benchmark data
                    await bot.benchmark_manager.save_benchmark(benchmark_data)

                except ValidationError as e:
                    logger.error(f"Validation error in research planning: {e}")
                    await thread.send(
                        embed=Embed(
                            title="Research Planning Error",
                            description=f"Invalid research plan format: {str(e)}",
                            color=Color.red()
                        )
                    )
                    return
                except Exception as e:
                    logger.error(f"Error in research planning/execution: {e}")
                    await thread.send(
                        embed=Embed(
                            title="Research Error",
                            description=f"An error occurred during research planning/execution: {str(e)}",
                            color=Color.red()
                        )
                    )
                    return

        except Exception as e:
            logger.error(f"Error in start_research: {e}")
            await ctx.send(f"An unexpected error occurred: {str(e)}")
            return

except Exception as e:
    logger.error(f"Failed to initialize bot: {e}")
    sys.exit(1)

if __name__ == "__main__":
    try:
        # Run the bot with proper error handling
        asyncio.run(bot.start(bot.token))
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except discord.LoginFailure as e:
        logger.error(f"Failed to login to Discord: {e}")
    except Exception as e:
        logger.error(f"Program terminated due to error: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Ensure proper cleanup
        if not bot.is_closed():
            try:
                asyncio.run(bot.close())
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
