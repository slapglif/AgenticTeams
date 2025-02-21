"""
Discord Interface Module - Handles all Discord-related functionality
"""
import json
import asyncio
import aiohttp
import traceback
from typing import List, Dict, Any, Optional
import discord
from discord import Embed, Color, Attachment, Message
from discord.ext import commands
from loguru import logger
from jsonschema import validate
from core.schemas.json_schemas import cot32_schema, final_plan_schema

class DiscordInterface:
    # Discord's absolute maximum is 2000, we use 1800 for safety
    DISCORD_CHUNK_LIMIT = 1800
    DISCORD_EMBED_LIMIT = 1800  # Also limit embeds to 1800 for consistency
    
    def __init__(self, channel_id: int):
        self.channel_id = channel_id
        
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks that respect Discord's length limits."""
        if not text:
            return []
            
        if len(text) <= self.DISCORD_CHUNK_LIMIT:
            return [text]
            
        chunks = []
        current_chunk = ""
        
        # Split by lines first
        lines = text.split('\n')
        
        for line in lines:
            # If single line is too long, split it
            if len(line) > self.DISCORD_CHUNK_LIMIT:
                while line:
                    # Try to split at word boundary
                    split_point = self.DISCORD_CHUNK_LIMIT
                    if len(line) > self.DISCORD_CHUNK_LIMIT:
                        split_point = line[:self.DISCORD_CHUNK_LIMIT].rfind(' ')
                        if split_point == -1:
                            split_point = self.DISCORD_CHUNK_LIMIT
                    
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    
                    chunks.append(line[:split_point])
                    line = line[split_point:].lstrip()
            else:
                # Check if adding this line would exceed limit
                if len(current_chunk) + len(line) + 1 > self.DISCORD_CHUNK_LIMIT:
                    chunks.append(current_chunk)
                    current_chunk = line
                else:
                    current_chunk = current_chunk + '\n' + line if current_chunk else line
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

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

    async def send_message_safely(self, thread: discord.Thread, content: str, embed: Optional[discord.Embed] = None) -> None:
        """Safely send a message, chunking if necessary."""
        try:
            if embed:
                # Calculate total embed size
                total_size = len(embed.title or "") + len(embed.description or "")
                for field in embed.fields:
                    total_size += len(field.name) + len(field.value)
                
                if total_size <= self.DISCORD_EMBED_LIMIT:
                    await thread.send(embed=embed)
                else:
                    # Convert embed to text for chunking
                    text_content = []
                    if embed.title:
                        text_content.append(f"**{embed.title}**")
                    if embed.description:
                        text_content.append(embed.description)
                    for field in embed.fields:
                        text_content.append(f"**{field.name}**\n{field.value}")
                    
                    content = "\n\n".join(text_content)
                    for chunk in self.chunk_text(content):
                        await thread.send(chunk)
                        await asyncio.sleep(0.5)  # Rate limiting protection
            else:
                for chunk in self.chunk_text(content):
                    await thread.send(chunk)
                    await asyncio.sleep(0.5)  # Rate limiting protection
                    
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            error_msg = f"Error sending message: {str(e)}"
            for chunk in self.chunk_text(error_msg):
                await thread.send(chunk)

    async def send_as_bot(self, agent_data: dict, thread: discord.Thread, content: str, 
                         embed_title: str = None, color: Color = None) -> None:
        """Send a message as a specific bot using its token and persona."""
        temp_bot = None
        try:
            # Create a temporary bot instance with minimal intents
            intents = discord.Intents.default()
            intents.message_content = True
            intents.guilds = True
            intents.messages = True
            temp_bot = commands.Bot(command_prefix="!", intents=intents)
            await temp_bot.login(agent_data['token'])
            await temp_bot.wait_until_ready()
            
            try:
                # Directly fetch the channel without caching
                channel = await temp_bot.fetch_channel(thread.id)
                if not channel:
                    logger.error(f"Could not find channel {thread.id} for bot {agent_data['name']}")
                    return
                    
                # Create and send embed or content with proper chunking
                if embed_title:
                    embed = Embed(
                        title=embed_title,
                        description=content[:self.DISCORD_EMBED_LIMIT],
                        color=color or Color.default()
                    )
                    await self.send_message_safely(channel, "", embed)
                else:
                    await self.send_message_safely(channel, content)
                    
            except discord.NotFound:
                logger.error(f"Channel {thread.id} not found for bot {agent_data['name']}")
            except discord.Forbidden:
                logger.error(f"Bot {agent_data['name']} lacks permissions for channel {thread.id}")
            except Exception as e:
                logger.error(f"Error accessing channel: {e}")
                logger.error(traceback.format_exc())
                
        except Exception as e:
            logger.error(f"Error sending message as bot {agent_data['name']}: {e}")
            logger.error(traceback.format_exc())
        finally:
            # Always close the temporary bot connection
            if temp_bot and not temp_bot.is_closed():
                await temp_bot.close()

    async def send_analysis_to_discord(self, thread, analysis, title="Analysis Results", agent=None):
        """Send analysis results to Discord thread."""
        try:
            await thread.trigger_typing()  # Show typing indicator
            
            # Create main embed
            embed = discord.Embed(title=title, color=0x00ff00)
            
            if agent:
                embed.add_field(name="Agent", value=f"ID: {agent.get('id', 'N/A')}\nType: {agent.get('type', 'N/A')}")
            
            # Add analysis fields
            for key, value in analysis.items():
                if key != "error":
                    embed.add_field(name=key.replace("_", " ").title(), value=str(value), inline=False)
            
            await thread.send(embed=embed)
            
            # Send error separately if present
            if "error" in analysis:
                error_embed = discord.Embed(title="Error", description=analysis["error"], color=0xff0000)
                await thread.send(embed=error_embed)
                
        except Exception as e:
            logger.error(f"Error sending analysis to Discord: {e}")
            await thread.send(f"Error sending analysis: {str(e)}")

    async def format_brainstorming_output(self, data: Dict[str, Any], title: str) -> List[Embed]:
        """Format brainstorming output into Discord embeds."""
        try:
            validate(instance=data, schema=cot32_schema)
            embeds = []
            
            # Create main embed
            main_embed = Embed(
                title=title,
                description="Below are the key research ideas identified for this topic. Each idea is prioritized and includes specific requirements and related areas of research.",
                color=Color.blue()
            )
            embeds.append(main_embed)
            
            # Format brainstorming items
            for i, item in enumerate(data["brainstorming"], 1):
                item_text = f"**Research Idea {i}:**\n{item.get('idea', '')}\n\n"
                item_text += f"**Priority:**\n{item.get('priority', '')}\n\n"
                item_text += f"**Requirements:**\n{item.get('needs', '')}\n\n"
                item_text += f"**Related Areas:**\n{item.get('supplements', '')}"
                
                # Split long item text into chunks
                for chunk in self.chunk_text(item_text):
                    embed = Embed(
                        title=f"Brainstorming Item {i}",
                        description=chunk,
                        color=Color.blue()
                    )
                    embeds.append(embed)
            
            return embeds
            
        except Exception as e:
            logger.error(f"Error formatting brainstorming output: {e}")
            logger.error(traceback.format_exc())
            error_embed = Embed(
                title="Error Formatting Research Plan",
                description=f"An error occurred while formatting the research plan:\n```\n{str(e)}\n```\nPlease try again or contact support if the issue persists.",
                color=Color.red()
            )
            return [error_embed]

    async def format_research_plan(self, data: Dict[str, Any], title: str) -> List[Embed]:
        """Format research plan into Discord embeds."""
        try:
            validate(instance=data, schema=final_plan_schema)
            embeds = []
            
            # Create main embed
            main_embed = Embed(title=title, color=Color.green())
            embeds.append(main_embed)
            
            # Format each step
            for step in data["plan"]:
                step_text = f"**Step {step['step_number']}:** {step['action']}\n\n"
                step_text += f"**Agent:** {step['agent']}\n"
                step_text += f"**Reasoning:**\n{step['reasoning']}\n\n"
                step_text += f"**Completion Conditions:**\n{step['completion_conditions']}\n\n"
                step_text += f"**Tool Suggestions:**\n{', '.join(map(str, step['tool_suggestions']))}\n\n"
                step_text += f"**Implementation Notes:**\n{step['implementation_notes']}"
                
                # Split long step text into chunks
                for chunk in self.chunk_text(step_text):
                    embed = Embed(
                        title=f"Step {step['step_number']}",
                        description=chunk,
                        color=Color.green()
                    )
                    embeds.append(embed)
            
            return embeds
            
        except Exception as e:
            logger.error(f"Error formatting research plan: {e}")
            error_embed = Embed(title="Error", description=str(e), color=Color.red())
            return [error_embed] 