# settings.py
import os
from typing import Dict, Any, Optional, Union, List
from loguru import logger
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pydantic import SecretStr, BaseModel, Field

from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langsmith import Client

API_BASE: str = ""
# Load environment variables (make sure to create a .env file with these variables)

COOLDOWN_TIME: int = 1
TERMINATE_KEYWORD: str = "TERMINATE"
SEARX_HOST: str = os.getenv("SEARX_HOST", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")  # Add your Tavily API key to environment variables
# Bot personas configuration

MEM0_KEY: str = os.getenv("MEM0_API_KEY", "")

# Use environment variables directly
DATABASE_URI: Optional[str] = os.getenv("DATABASE_URI")
GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")

# Flags to control web search and graph functionality
USE_WEB_SEARCH: bool = os.getenv("USE_WEB_SEARCH", "True").lower() == "true"
USE_GRAPH: bool = os.getenv("USE_GRAPH", "True").lower() == "true"

RATE_LIMIT_PER_SECOND: int = int(os.getenv("RATE_LIMIT_PER_SECOND", "10"))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "64"))  # For batched embeddings

PROVIDER: str = os.getenv("PROVIDER", "google")


BROWSERLESS_ENDPOINT: str = os.getenv("BROWSERLESS_ENDPOINT", "http://localhost:9222")
OUTPUT_FILE_PATH: str = os.getenv("OUTPUT_FILE_PATH", "dataset-test.jsonl")
NUM_WORKERS: int = int(os.getenv("NUM_WORKERS", "4"))
SEARCH_RESULTS_NUM: int = int(os.getenv("SEARCH_RESULTS_NUM", "1000"))
MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "1000"))


EMBEDDING_MODEL_OLLAMA: str = os.getenv("EMBEDDING_MODEL_OLLAMA", "bge-m3")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")

EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# Rate limiting setup
OLLAMA_SEMAPHORE_VALUE: int = int(os.getenv("OLLAMA_SEMAPHORE_VALUE", "5"))
OLLAMA_CALL_DELAY: int = int(os.getenv("OLLAMA_CALL_DELAY", "1"))
CSV_FOLDER_PATH: str = "C:\\Users\\Melody\\work\\datas"
# Chroma settings
PERSIST_DIRECTORY: str = os.getenv("PERSIST_DIRECTORY", "chroma_db")

# Agent Ops Configuration
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT_NAME: str = os.getenv("LANGSMITH_PROJECT_NAME", "AgentApp")
LANGSMITH_TAGS: list[str] = ["agent_app", "production", "agent_research", "deep_exploration"]

# Initialize LangSmith client for global use
try:
    langsmith_client: Optional[Client] = Client(api_key=LANGSMITH_API_KEY)
    # Set project tags after initialization if needed
    if LANGSMITH_TAGS:
        try:
            # Check if project exists first
            existing_projects = langsmith_client.list_projects()
            project_exists = any(p.name == LANGSMITH_PROJECT_NAME for p in existing_projects)
            
            if not project_exists:
                langsmith_client.create_project(
                    LANGSMITH_PROJECT_NAME,
                    metadata={"tags": LANGSMITH_TAGS}
                )
                logger.info(f"Created new LangSmith project: {LANGSMITH_PROJECT_NAME}")
            else:
                logger.info(f"LangSmith project {LANGSMITH_PROJECT_NAME} already exists")
        except Exception as project_error:
            logger.warning(f"Error managing LangSmith project: {project_error}")
            # Continue execution even if project creation fails
except Exception as e:
    logger.error(f"Error initializing LangSmith client: {e}")
    langsmith_client = None

# Agent Ops Monitoring Settings
ENABLE_TRACING: bool = os.getenv("ENABLE_TRACING", "True").lower() == "true"
TRACE_LEVEL: str = os.getenv("TRACE_LEVEL", "DEBUG")
TRACE_SAMPLE_RATE: float = float(os.getenv("TRACE_SAMPLE_RATE", "1.0"))

# Benchmarking Settings
BENCHMARK_METRICS: Dict[str, bool] = {
    "latency": True,
    "token_usage": True,
    "completion_tokens": True,
    "prompt_tokens": True,
    "total_tokens": True,
    "cost": True
}

# API key rotation settings
GOOGLE_API_KEYS = [
]

GEMINI_MODELS = [
    "gemini-2.0-flash-thinking-exp-01-21",  # Primary model
    "gemini-2.0-flash-exp",                 # First fallback
    "gemini-1.5-flash"                      # Second fallback
]

# Track API key usage and rate limits
_api_key_index = 0
_model_index = 0
_last_api_call = {}  # Track timestamps for each API key
_call_counts = {}    # Track call counts for each API key

def _get_next_api_key():
    """Get the next available API key with capacity."""
    global _api_key_index, _last_api_call, _call_counts
    
    current_time = datetime.now(UTC)
    
    # Initialize tracking for new minute if needed
    for key in GOOGLE_API_KEYS:
        if key not in _last_api_call or (current_time - _last_api_call[key]).total_seconds() >= 60:
            _call_counts[key] = 0
            _last_api_call[key] = current_time
    
    # Try each API key
    for _ in range(len(GOOGLE_API_KEYS)):
        key = GOOGLE_API_KEYS[_api_key_index]
        
        # Check if current key has capacity
        if _call_counts[key] < 10:  # 10 RPM limit
            _call_counts[key] += 1
            return key
            
        # Move to next key
        _api_key_index = (_api_key_index + 1) % len(GOOGLE_API_KEYS)
    
    # If all keys are at limit, return None
    return None

def _get_next_model():
    """Get the next fallback model."""
    global _model_index
    
    if _model_index < len(GEMINI_MODELS) - 1:
        _model_index += 1
        return GEMINI_MODELS[_model_index]
    return GEMINI_MODELS[-1]  # Stay on last model if all tried

def build_embeddings(provider: str = "ollama", **kwargs: Any) -> Union[GoogleGenerativeAIEmbeddings, OllamaEmbeddings]:
    """Build embeddings model based on provider."""
    provider = os.getenv("EMBEDDING_PROVIDER", provider)
    if provider == "google":
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "models/text-embedding-004"),
            transport="rest",
        )
    else:  # Default to Ollama embeddings
        _embeddings = OllamaEmbeddings(
            base_url=API_BASE,
            model=os.getenv("EMBEDDING_MODEL_OLLAMA", EMBEDDING_MODEL_OLLAMA),
            **kwargs,
        )
    return _embeddings


cached_embedder = build_embeddings(provider=os.getenv("EMBEDDING_PROVIDER", "ollama"))

# cached_embedder = CacheBackedEmbeddings(embeddings, InMemoryStore())


# cache = RedisSemanticCache(redis_url=REDIS_url, embedding=cached_embedder, score_threshold=0.5)
# set_llm_cache(cache)


class Models:
    """Model configuration class."""
    nemo_kraut: str = os.getenv("NEMO_KRAUT_MODEL", "cyberwald/sauerkrautlm-nemo-12b-instruct:q8_0")
    nemo: str = os.getenv("NEMO_MODEL", "mistral-nemo:12b-instruct-2407-q6_K")
    gemini_flash: str = os.getenv("GEMINI_FLASH_MODEL", "gemini-1.5-flash-latest")
    triplex: str = os.getenv("TRIPLEX_MODEL", "sciphi/triplex")
    text2cypher: str = os.getenv("TEXT2CYPHER_MODEL", "tomasonjo/codestral-text2cypher")
    nuextract: str = os.getenv("NUEXTRACT_MODEL", "nuextract:latest")
    generic: str = os.getenv("GENERIC_MODEL", "gpt-4-32k")
    phi_35_mini: str = os.getenv("PHI_35_MINI_MODEL", "phi3.5:3.8b-mini-instruct-q8_0")
    nemo_mini: str = os.getenv("NEMO_MINI_MODEL", "mistral-nemo:latest")
    smol: str = os.getenv("SMOL_MODEL", "smollm:1.7b-instruct-v0.2-fp16")


class ResearchResponse(BaseModel):
    """Base model for research responses."""
    analysis: str = Field(description="Detailed analysis of the research topic")
    findings: List[Dict[str, Any]] = Field(description="Key findings from the research")
    next_steps: List[str] = Field(description="Recommended next steps")
    confidence: float = Field(description="Confidence score for the findings", ge=0, le=1)

class TechnicalAnalysis(BaseModel):
    """Model for technical analysis responses."""
    current_focus: str = Field(description="Current analysis focus")
    technical_details: List[Dict[str, Any]] = Field(description="Technical details with methodology")
    implementation_aspects: List[Dict[str, Any]] = Field(description="Implementation considerations")
    quality_metrics: Dict[str, int] = Field(description="Quality metrics for the analysis")

def build_llm(output_mode: str = '', provider="google", **kwargs):
    """Build LLM with API key rotation and model fallback."""
    if provider == "ollama":
        # Extract and remove temperature from kwargs if it exists
        temperature = kwargs.pop("temperature", 0.5)
        model = kwargs.pop("model", "deepseek-r1:latest")
        
        if "num_ctx" not in kwargs:
            kwargs["num_ctx"] = int(os.getenv("OLLAMA_NUM_CTX", 14000))
            
        if "base_url" not in kwargs:
            kwargs["base_url"] = API_BASE
        
        # For Ollama, we use format='json' for JSON output
        if output_mode:
            kwargs["format"] = "json"
            
        llm = ChatOllama(
            model=model,
            temperature=temperature,
            **kwargs
        )
        return llm
        
    else:
        # Default to Google if not Ollama
        temperature = kwargs.pop("temperature", 0.1)
        
        while True:
            api_key = _get_next_api_key()
            
            # If no API key available, try fallback model
            if api_key is None:
                model = _get_next_model()
                logger.warning(f"All API keys at rate limit, falling back to model: {model}")
                # Reset API key index to try again with new model
                _api_key_index = 0
                continue
                
            try:
                # Set API key in environment
                os.environ["GOOGLE_API_KEY"] = api_key
                
                # Create base LLM
                llm = ChatGoogleGenerativeAI(
                    model=os.getenv("LLM_MODEL", GEMINI_MODELS[_model_index]),
                    temperature=temperature,
                    **kwargs
                )
                return llm
                    
            except Exception as e:
                if "quota exceeded" in str(e).lower() or "rate limit" in str(e).lower():
                    logger.warning(f"Rate limit hit for key {api_key}, trying next key/model")
                    continue
                raise  # Re-raise if it's not a rate limit error


models = Models()


