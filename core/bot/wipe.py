"""Discord bot wipe module."""
import os
import discord
from discord.ext import commands
from loguru import logger

# Get bot token from environment variable
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("DISCORD_BOT_TOKEN environment variable is not set")

# Initialize bot with required intents
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    """Called when the bot is ready."""
    if bot.user:
        logger.info(f"Logged in as {bot.user.name}")
    else:
        logger.error("Bot user is None")

@bot.command()
@commands.has_permissions(manage_messages=True)
async def wipe(ctx, limit: int = 100):
    """Wipe messages from the channel."""
    deleted = 0
    async for message in ctx.channel.history(limit=limit):
        if not message.author.bot:
            try:
                await message.delete()
                deleted += 1
            except discord.errors.NotFound:
                pass
            except discord.errors.Forbidden:
                await ctx.send("I don't have permission to delete messages.")
                return
    
    await ctx.send(f"Deleted {deleted} messages sent by humans.")

bot.run(BOT_TOKEN)