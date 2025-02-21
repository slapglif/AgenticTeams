"""
Default values and fallback configurations for research tools.

This module contains:
1. Default table styles and configurations
2. Fallback result templates
3. Default metrics and thresholds
4. Error message templates
"""
from typing import Dict, Any, List
from datetime import datetime, UTC

# Discord message limits
DISCORD_LIMITS = {
    "max_message_length": 2000,
    "max_embed_title": 256,
    "max_embed_description": 4096,
    "max_embed_fields": 25,
    "max_embed_field_name": 256,
    "max_embed_field_value": 1024,
    "max_embed_footer": 2048,
    "max_embed_author": 256,
    "max_embeds_per_message": 10,
    "max_file_size": 8388608  # 8MB
}

# Table styles and configurations
TABLE_STYLES = {
    "default": {
        "box": "SIMPLE",
        "header_style": "bold magenta",
        "row_styles": ["", "dim"],
        "padding": (0, 1),
        "title_style": "bold cyan",
        "caption_style": "italic"
    },
    "compact": {
        "box": "MINIMAL",
        "header_style": "bold blue",
        "row_styles": [""],
        "padding": (0, 1),
        "title_style": "bold white",
        "caption_style": "dim"
    },
    "detailed": {
        "box": "DOUBLE",
        "header_style": "bold green",
        "row_styles": ["", "dim"],
        "padding": (0, 2),
        "title_style": "bold yellow",
        "caption_style": "italic cyan"
    }
}

# Default table configurations
DEFAULT_TABLE_CONFIGS = {
    "topology": {
        "title": "Network Topology Analysis",
        "columns": ["Component", "Description", "Significance"],
        "style": "detailed"
    },
    "nodes": {
        "title": "Key Nodes Analysis",
        "columns": ["Node", "Role", "Impact"],
        "style": "detailed"
    },
    "interactions": {
        "title": "Node Interactions",
        "columns": ["Source", "Target", "Type", "Strength"],
        "style": "detailed"
    },
    "metrics": {
        "title": "Network Metrics",
        "columns": ["Metric", "Value", "Interpretation"],
        "style": "compact"
    },
    "findings": {
        "title": "Key Findings",
        "columns": ["Finding", "Evidence", "Implications"],
        "style": "default"
    },
    "citations": {
        "title": "Citation Analysis",
        "columns": ["Source", "Context", "Relevance"],
        "style": "default"
    },
    "domains": {
        "title": "Domain Analysis",
        "columns": ["Domain", "Description", "Significance"],
        "style": "default"
    },
    "data": {
        "title": "Data Analysis",
        "columns": ["Category", "Observation", "Interpretation"],
        "style": "default"
    }
}

# Empty result templates
def create_empty_network_result() -> Dict[str, Any]:
    """Create an empty network analysis result."""
    return {
        "topology": [],
        "nodes": [],
        "interactions": [],
        "metrics": [],
        "findings": [],
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "empty"
    }

def create_empty_network_structure_result() -> Dict[str, Any]:
    """Create an empty network structure result."""
    return {
        "structure": [],
        "nodes": [],
        "connections": [],
        "metrics": [],
        "findings": [],
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "empty"
    }

def create_empty_citation_result() -> Dict[str, Any]:
    """Create an empty citation analysis result."""
    return {
        "citations": [],
        "context": "",
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "empty"
    }

def create_empty_domain_result() -> Dict[str, Any]:
    """Create an empty domain analysis result."""
    return {
        "domains": [],
        "context": "",
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "empty"
    }

def create_empty_data_analysis_result() -> Dict[str, Any]:
    """Create an empty data analysis result."""
    return {
        "categories": [],
        "observations": [],
        "context": "",
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "empty"
    }

def create_empty_search_result() -> Dict[str, Any]:
    """Create an empty search result."""
    return {
        "query": "",
        "results": [],
        "context": "",
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "empty"
    }

def create_empty_graph_result() -> Dict[str, Any]:
    """Create an empty graph analysis result."""
    return {
        "nodes": [],
        "edges": [],
        "metrics": [],
        "findings": [],
        "context": "",
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "empty"
    }

def create_formatted_result(result: Dict[str, Any], tool_type: str, context: str = "") -> Dict[str, Any]:
    """Create a formatted result with standard fields.
    
    Args:
        result: Result data to include
        tool_type: Type of result (e.g. 'network', 'citations', etc.)
        context: Optional context string
        
    Returns:
        Formatted result dictionary
    """
    return {
        **result,
        "context": context,
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "success",
        "result_type": tool_type
    }

def create_error_response(error: str, tool_type: str) -> Dict[str, Any]:
    """Create an error response.
    
    Args:
        error: Error message
        tool_type: Type of tool that generated the error
        
    Returns:
        Error response dictionary
    """
    return {
        "error": str(error),
        "error_type": tool_type,
        "timestamp": datetime.now(UTC).isoformat(),
        "status": "error"
    }

# Default metrics thresholds
METRIC_THRESHOLDS = {
    "network": {
        "min_nodes": 3,
        "min_edges": 2,
        "min_density": 0.1,
        "min_clustering": 0.2,
        "max_diameter": 6
    },
    "citations": {
        "min_citations": 1,
        "min_relevance": 0.5,
        "min_context_length": 50
    },
    "domains": {
        "min_domains": 1,
        "min_significance": 0.5,
        "min_description_length": 50
    },
    "data": {
        "min_categories": 1,
        "min_observations": 3,
        "min_interpretation_length": 50
    }
}

# Error message templates
ERROR_MESSAGES = {
    "network_analysis": {
        "insufficient_data": "Insufficient data for network analysis. Need at least {min_nodes} nodes and {min_edges} edges.",
        "low_density": "Network density ({density}) below minimum threshold ({min_density}).",
        "high_diameter": "Network diameter ({diameter}) exceeds maximum threshold ({max_diameter}).",
        "execution_failed": "Network analysis execution failed: {error}"
    },
    "citation_analysis": {
        "no_citations": "No citations found in the provided content.",
        "low_relevance": "Citation relevance ({relevance}) below minimum threshold ({min_relevance}).",
        "insufficient_context": "Insufficient citation context. Minimum length: {min_context_length}",
        "execution_failed": "Citation analysis execution failed: {error}"
    },
    "domain_analysis": {
        "no_domains": "No domains identified in the provided content.",
        "low_significance": "Domain significance ({significance}) below minimum threshold ({min_significance}).",
        "insufficient_description": "Insufficient domain description. Minimum length: {min_description_length}",
        "execution_failed": "Domain analysis execution failed: {error}"
    },
    "data_analysis": {
        "insufficient_data": "Insufficient data for analysis. Need at least {min_categories} categories and {min_observations} observations.",
        "insufficient_interpretation": "Insufficient data interpretation. Minimum length: {min_interpretation_length}",
        "execution_failed": "Data analysis execution failed: {error}"
    }
}

# Default chain configurations
DEFAULT_CHAIN_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "stop_sequences": ["```", "---"],
    "retry_on_error": True,
    "max_retries": 3,
    "retry_delay": 1.0
}

# Default tool configurations
DEFAULT_TOOL_CONFIG = {
    "network_analysis": {
        "max_nodes": 100,
        "max_edges": 500,
        "include_metrics": True,
        "include_visualization": False
    },
    "citation_analysis": {
        "max_citations": 50,
        "include_context": True,
        "min_relevance": 0.5
    },
    "domain_analysis": {
        "max_domains": 20,
        "include_descriptions": True,
        "min_significance": 0.5
    },
    "data_analysis": {
        "max_categories": 20,
        "max_observations": 100,
        "include_interpretations": True
    }
}

# Default memory configurations
DEFAULT_MEMORY_CONFIG = {
    "max_entries": 1000,
    "ttl_seconds": 3600,
    "index_fields": ["task_id", "tool_name", "timestamp"],
    "compression": True
}

# Utility functions
def get_table_style(style_name: str) -> Dict[str, Any]:
    """Get a table style configuration.
    
    Args:
        style_name: Name of the style
        
    Returns:
        Style configuration
    """
    return TABLE_STYLES.get(style_name, TABLE_STYLES["default"])

def get_table_config(table_type: str) -> Dict[str, Any]:
    """Get a table configuration.
    
    Args:
        table_type: Type of table
        
    Returns:
        Table configuration
    """
    return DEFAULT_TABLE_CONFIGS.get(table_type, {
        "title": table_type.title(),
        "columns": ["Category", "Value", "Notes"],
        "style": "default"
    })

def get_metric_thresholds(analysis_type: str) -> Dict[str, float]:
    """Get metric thresholds for an analysis type.
    
    Args:
        analysis_type: Type of analysis
        
    Returns:
        Metric thresholds
    """
    return METRIC_THRESHOLDS.get(analysis_type, {})

def get_error_message(
    analysis_type: str,
    error_type: str,
    **kwargs: Any
) -> str:
    """Get a formatted error message.
    
    Args:
        analysis_type: Type of analysis
        error_type: Type of error
        **kwargs: Format arguments
        
    Returns:
        Formatted error message
    """
    template = ERROR_MESSAGES.get(analysis_type, {}).get(
        error_type,
        "Error in {analysis_type}: {error_type}"
    )
    return template.format(**kwargs)

def get_chain_config(
    chain_type: str,
    overrides: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Get chain configuration with optional overrides.
    
    Args:
        chain_type: Type of chain
        overrides: Optional configuration overrides
        
    Returns:
        Chain configuration
    """
    config = DEFAULT_CHAIN_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config

def get_tool_config(
    tool_name: str,
    overrides: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Get tool configuration with optional overrides.
    
    Args:
        tool_name: Name of the tool
        overrides: Optional configuration overrides
        
    Returns:
        Tool configuration
    """
    config = DEFAULT_TOOL_CONFIG.get(tool_name, {}).copy()
    if overrides:
        config.update(overrides)
    return config

def get_memory_config(
    overrides: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Get memory configuration with optional overrides.
    
    Args:
        overrides: Optional configuration overrides
        
    Returns:
        Memory configuration
    """
    config = DEFAULT_MEMORY_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config 