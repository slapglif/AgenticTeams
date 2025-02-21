"""Intelligent message formatter for Discord using LLMs."""

import json
from typing import List, Dict, Any, Optional, Union
from loguru import logger
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import discord
import asyncio
from pydantic import BaseModel, Field

from core.shared.settings import build_llm
from core.shared.data_processor import IntelligentDataProcessor
from core.compile.utils import create_chain

class ContentChunk(BaseModel):
    """Schema for content chunks."""
    content: str = Field(description="The content text")
    format: str = Field(description="Format type (plain, code, quote)")
    language: Optional[str] = Field(default=None, description="Language for code blocks")
    priority: int = Field(description="Priority (1-3)", ge=1, le=3)

class ContentStructure(BaseModel):
    """Schema for content structure."""
    chunks: List[ContentChunk] = Field(description="List of content chunks")
    metadata: Dict[str, Any] = Field(description="Content metadata")

class EnhancedContent(BaseModel):
    """Schema for enhanced content."""
    enhanced_content: str = Field(description="Enhanced content text")
    formatting_applied: List[str] = Field(description="Applied formatting")
    structure_changes: List[str] = Field(description="Structure changes made")
    metadata: Dict[str, float] = Field(description="Content metadata")

class MessageFormatter:
    """Handles formatting and chunking of messages for Discord."""
    
    # Discord limits
    DISCORD_MSG_LIMIT = 2000
    DISCORD_EMBED_LIMIT = 4096
    DISCORD_SAFE_CHUNK = 1800  # Leave room for formatting
    
    def __init__(self):
        """Initialize the message formatter."""
        self.data_processor = IntelligentDataProcessor()
        self.formatting_prompts = {
            "structure": """Structure this content for optimal Discord display.
            Consider:
            1. Message length limits and chunking needs
            2. Formatting for readability (headers, sections, etc)
            3. Important information hierarchy
            4. Code block and syntax highlighting needs
            
            Content: {content}
            Max Length: {max_length}
            Content Type: {content_type}
            
            Return a JSON structure with:
            {
                "chunks": [
                    {
                        "content": str,
                        "format": str,  // plain, code, quote, etc
                        "language": str,  // for code blocks
                        "priority": int   // 1-3, 1 is highest
                    }
                ],
                "metadata": {
                    "total_chunks": int,
                    "has_code": bool,
                    "has_embeds": bool,
                    "estimated_read_time": int  // seconds
                }
            }""",
            
            "summarize": """Create a concise summary of this content for Discord.
            Focus on:
            1. Key information and insights
            2. Critical details that must be preserved
            3. Logical flow and context
            4. Technical accuracy
            
            Content: {content}
            Target Length: {target_length}
            Context: {context}
            
            Return a JSON summary with:
            {
                "summary": str,
                "key_points": [str],
                "technical_details": [str],
                "metadata": {
                    "compression_ratio": float,
                    "information_preserved": float  // 0-1 score
                }
            }""",
            
            "enhance": """Enhance this content for better Discord presentation.
            Improve:
            1. Formatting and structure
            2. Technical clarity
            3. Information hierarchy
            4. Visual organization
            
            Content: {content}
            Format: {format}
            Context: {context}
            
            Return a JSON result with:
            {
                "enhanced_content": str,
                "formatting_applied": [str],
                "structure_changes": [str],
                "metadata": {
                    "readability_score": float,  // 0-1 score
                    "technical_accuracy": float  // 0-1 score
                }
            }"""
        }

    def format_content(self, content: Any, title: Optional[str] = None) -> List[str]:
        """Format content into Discord-friendly chunks."""
        try:
            chunks = []
            current_chunk = []
            
            # Add title if provided
            if title:
                current_chunk.append(f"# {title}\n")
            
            # Convert content to string if needed
            if isinstance(content, (dict, list)):
                try:
                    content = json.dumps(content, indent=2)
                except Exception as e:
                    logger.error(f"Error converting content to JSON: {e}")
                    content = str(content)
            else:
                content = str(content)
                
            # Split content into chunks
            content_chunks = [content[i:i+self.DISCORD_SAFE_CHUNK] 
                            for i in range(0, len(content), self.DISCORD_SAFE_CHUNK)]
            
            # Format first chunk with title if it fits
            if current_chunk and content_chunks:
                first_chunk = f"```\n{content_chunks[0]}\n```"
                if len("\n".join(current_chunk + [first_chunk])) <= self.DISCORD_MSG_LIMIT:
                    current_chunk.append(first_chunk)
                    content_chunks = content_chunks[1:]
                    chunks.append("\n".join(current_chunk))
                else:
                    # Title chunk and content chunk separately
                    if current_chunk:
                        chunks.append("\n".join(current_chunk))
                    chunks.append(f"```\n{content_chunks[0]}\n```")
                    content_chunks = content_chunks[1:]
            
            # Add remaining chunks
            for chunk in content_chunks:
                chunks.append(f"```\n{chunk}\n```")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error formatting content: {e}")
            return [f"Error formatting content: {str(e)}"]

    async def format_research_content(self, content: Dict[str, Any], title: Optional[str] = None) -> List[str]:
        """Format research content into Discord-friendly chunks with enhanced formatting.
        
        Args:
            content: Dictionary containing research content to format
            title: Optional title to include at the start
            
        Returns:
            List of formatted message chunks ready for Discord
        """
        try:
            # Get content structure
            structure = await self._get_content_structure(
                content=json.dumps(content, indent=2),
                content_type="research"
            )
            
            chunks = []
            current_chunk = []
            
            # Add title if provided
            if title:
                current_chunk.append(f"# {title}\n")
                
            # Process each content chunk
            for chunk_data in structure["chunks"]:
                processed = await self._process_chunk(
                    content=chunk_data["content"],
                    format_type=chunk_data["format"],
                    context={"priority": chunk_data["priority"]}
                )
                
                # Format based on chunk type
                if chunk_data["format"] == "code":
                    formatted = f"```{chunk_data.get('language', '')}\n{processed}\n```"
                elif chunk_data["format"] == "quote":
                    formatted = f"> {processed.replace('\n', '\n> ')}"
                else:
                    formatted = processed
                    
                # Add to chunks
                if current_chunk and len("\n".join(current_chunk + [formatted])) > self.DISCORD_MSG_LIMIT:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = [formatted]
                else:
                    current_chunk.append(formatted)
                    
            # Add any remaining content
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                
            return chunks
            
        except Exception as e:
            logger.error(f"Error formatting research content: {e}")
            # Fallback to basic formatting
            return self.format_content(content, title)

    def create_embed(self, title: str, description: str, 
                    fields: Optional[List[Dict[str, str]]] = None,
                    color: int = discord.Color.blue().value) -> discord.Embed:
        """Create a Discord embed with the given content."""
        try:
            embed = discord.Embed(
                title=title,
                description=description,
                color=color
            )
            
            if fields:
                for field in fields:
                    embed.add_field(
                        name=field.get("name", ""),
                        value=field.get("value", ""),
                        inline=bool(field.get("inline", False))
                    )
                    
            return embed
            
        except Exception as e:
            logger.error(f"Error creating embed: {e}")
            return discord.Embed(
                title="Error",
                description=f"Failed to create embed: {str(e)}",
                color=discord.Color.red().value
            )

    async def _get_content_structure(self, content: str, content_type: str) -> Dict[str, Any]:
        """Get structured content format."""
        try:
            # Create chain
            prompt = ChatPromptTemplate.from_template(self.formatting_prompts["structure"])
            chain = create_chain(
                prompt=prompt,
                output_mode="structure",
                output_parser=PydanticOutputParser(pydantic_object=ContentStructure)
            )
            
            # Execute chain
            result = await chain.ainvoke({
                "content": content,
                "max_length": self.DISCORD_MSG_LIMIT,
                "content_type": content_type
            })
            
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Failed to get content structure: {e}")
            raise

    async def _process_chunk(
        self,
        content: str,
        format_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process a content chunk with formatting."""
        try:
            # Create chain
            prompt = ChatPromptTemplate.from_template(self.formatting_prompts["enhance"])
            chain = create_chain(
                prompt=prompt,
                output_mode="enhance",
                output_parser=PydanticOutputParser(pydantic_object=EnhancedContent)
            )
            
            # Execute chain
            result = await chain.ainvoke({
                "content": content,
                "format": format_type,
                "context": context or {}
            })
            
            return result.enhanced_content
            
        except Exception as e:
            logger.error(f"Failed to process chunk: {e}")
            raise

    async def format_message(self, message: str, format_type: str = "structure") -> str:
        """Format a message using LLM."""
        try:
            # Create chain
            prompt = ChatPromptTemplate.from_template(self.formatting_prompts[format_type])
            chain = create_chain(
                prompt=prompt,
                output_mode=format_type,
                output_parser=PydanticOutputParser(pydantic_object=ContentStructure)
            )
            
            # Execute chain
            result = await chain.ainvoke({
                "content": message,
                "max_length": self.DISCORD_MSG_LIMIT,
                "content_type": "message"
            })
            
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Failed to format message: {e}")
            raise

    async def enhance_message(self, message: str) -> str:
        """Enhance a message using LLM."""
        try:
            # Create chain
            prompt = ChatPromptTemplate.from_template(self.formatting_prompts["enhance"])
            chain = create_chain(
                prompt=prompt,
                output_mode="enhance",
                output_parser=PydanticOutputParser(pydantic_object=EnhancedContent)
            )
            
            # Execute chain
            result = await chain.ainvoke({
                "content": message,
                "format": "message",
                "context": {}
            })
            
            return result.enhanced_content
            
        except Exception as e:
            logger.error(f"Failed to enhance message: {e}")
            raise

    async def summarize_content(self, content: str, target_length: int, context: str) -> str:
        """Summarize content using LLM."""
        try:
            # Create chain
            prompt = ChatPromptTemplate.from_template(self.formatting_prompts["summarize"])
            chain = create_chain(
                prompt=prompt,
                output_mode="summarize",
                output_parser=PydanticOutputParser(pydantic_object=ContentStructure)
            )
            
            # Execute chain
            result = await chain.ainvoke({
                "content": content,
                "target_length": target_length,
                "context": context
            })
            
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Failed to summarize content: {e}")
            raise 