"""Research prompts module containing all prompts used by research tools.

Version: 1.0.0
Last Updated: 2024-03-19

This module contains all the prompts used by the research tools system. Each prompt is
documented with its purpose, expected input variables, and output format.
"""

# Domain Search Query Generation
DOMAIN_SEARCH_QUERY_PROMPT = """Generate domain-specific search queries for the following aspect:

Aspect: {aspect}

Generate:
1. Primary search query
2. Alternative formulations
3. Related subtopics

Format your response as a JSON object with:
- primary_query: str
- alternative_queries: List[str]
- subtopics: List[str]
"""

# Data Analysis
DATA_ANALYSIS_PROMPT = """You are a data analysis expert. Analyze the following research request and provide detailed insights:

Research Context:
{implementation_notes}

Memories:
{memories}

Your response must be a JSON object with the following structure:
{{
    "key_metrics": {{
        "relevance_score": 0.0,  // 0-1 score
        "coverage_score": 0.0,  // 0-1 score
        "recency_score": 0.0,  // 0-1 score
        "impact_score": 0.0  // 0-1 score
    }},
    "correlations": [
        {{
            "variables": [str],  // List of related variables
            "coefficient": float,  // Correlation strength
            "description": str  // Description of the relationship
        }}
    ],
    "significant_findings": [
        {{
            "finding": str,  // Description of the finding
            "confidence": float,  // 0-1 confidence score
            "supporting_evidence": [str]  // List of supporting evidence
        }}
    ],
    "analysis": {{
        "methodology": str,  // Description of analysis approach
        "limitations": [str],  // List of analysis limitations
        "recommendations": [str]  // List of recommendations
    }},
    "insights": [str],  // List of key insights
    "confidence": float,  // Overall confidence score (0-1)
    "metadata": {{
        "timestamp": str,  // ISO format timestamp
        "version": str,  // Analysis version
        "source": str  // Source of the analysis
    }}
}}

Focus on providing actual analysis results, not schema definitions.
Make sure your analysis is specific to the research context and provides actionable insights.
All fields except metadata are required."""

# Graph Analysis
GRAPH_ANALYSIS_PROMPT = """You are a graph theory and network analysis expert. Analyze the following research context and provide detailed graph insights:

Research Context:
{context}

Previous Related Work:
{memories}

Network Topology:
{topology}

Consider:
1. What are the key nodes and their relationships in this network?
2. What graph metrics would be most informative?
3. What communities or clusters can be identified?
4. What insights can we derive from the graph structure?

Your response must be a JSON object with the following structure:
{{
    "topology": {{
        "type": "string",  // e.g., "citation_network", "collaboration_network"
        "description": "string",  // Description of the network structure
        "key_properties": ["string"]  // List of important network properties
    }},
    "nodes": [
        {{
            "id": "string",
            "type": "string",
            "metrics": {{
                "centrality": 0.0,
                "degree": 0,
                "importance": 0.0
            }}
        }}
    ],
    "edges": [
        {{
            "source": "string",
            "target": "string",
            "weight": 0.0,
            "type": "string"
        }}
    ],
    "metrics": {{
        "clustering_coefficient": 0.0,
        "density": 0.0,
        "avg_path_length": 0.0,
        "centrality_measures": {{
            "degree": 0.0,
            "betweenness": 0.0,
            "closeness": 0.0
        }}
    }},
    "communities": [
        {{
            "id": "string",
            "members": ["string"],
            "description": "string",
            "cohesion": 0.0
        }}
    ],
    "findings": [
        {{
            "insight": "string",
            "confidence": 0.0,
            "supporting_evidence": ["string"]
        }}
    ]
}}

Focus on providing meaningful graph-based insights that are specific to the research context.
All fields are required and should contain actual analysis results, not schema definitions."""

# Citation Analysis
CITATION_ANALYSIS_PROMPT = """You are a citation analysis expert. Analyze the following research context and citations:

Research Context:
{research_context}

Citations:
{citations}

Your response must be a JSON object with the following structure:
{{
    "citations": [
        {{
            "source": "string",  // Source document title/name
            "validity": "string",  // Assessment of citation validity
            "credibility_score": 0.0,  // Float between 0-1
            "impact_factor": 0.0  // Float indicating impact
        }}
    ],
    "patterns": [
        {{
            "pattern_type": "string",  // Type of citation pattern
            "description": "string"  // Description of the pattern
        }}
    ],
    "recommendations": [
        {{
            "suggestion": "string",  // Recommendation based on citation analysis
            "rationale": "string"  // Reasoning behind the recommendation
        }}
    ],
    "synthesis": {{
        "sources": [
            {{
                "title": "string",  // Source document title
                "url": "string",  // Optional URL to source
                "relevance_score": 0.0  // Float between 0-1
            }}
        ],
        "key_insights": [
            {{
                "insight": "string",  // Key insight from analysis
                "supporting_sources": ["string"],  // List of supporting source titles
                "confidence": 0.0  // Float between 0-1
            }}
        ]
    }}
}}

Focus on providing actual analysis results, not schema definitions.
Make sure your analysis is specific to the research context and provides actionable insights.
All fields except metadata are required.
"""

# Network Analysis
NETWORK_ANALYSIS_PROMPT = """Analyze the network structure and relationships in the research context.

Implementation Notes: {implementation_notes}

Your task is to analyze the network and return a structured response with:

1. Network Topology:
   - Type of network structure (e.g., small-world, scale-free)
   - Description of the structure
   - Key properties and characteristics

2. Key Nodes:
   - Node ID
   - Node type (e.g., researcher, institution)
   - Importance score (0-1)
   - Additional attributes

3. Relationships:
   - Source and target node IDs
   - Connection type (e.g., collaboration, citation)
   - Connection strength (0-1)

4. Communities:
   - Community ID
   - Description
   - Member nodes

5. Metrics:
   - Network density
   - Average path length
   - Clustering coefficient

6. Key Findings:
   - Important insights
   - Confidence scores
   - Supporting evidence

Return the analysis in the following JSON format:
{{
    "network_topology": {{
        "type": "string",
        "description": "string",
        "key_properties": ["string"]
    }},
    "key_nodes": [{{
        "id": "string",
        "type": "string",
        "importance": 0.0,
        "attributes": {{"attr_name": "value"}}
    }}],
    "relationships": [{{
        "source": "string",
        "target": "string",
        "type": "string",
        "strength": 0.0
    }}],
    "communities": [{{
        "id": "string",
        "description": "string",
        "members": ["string"]
    }}],
    "metrics": {{
        "density": 0.0,
        "avg_path_length": 0.0,
        "clustering_coefficient": 0.0
    }},
    "findings": [{{
        "insight": "string",
        "confidence": 0.0,
        "supporting_evidence": ["string"]
    }}]
}}

Ensure all required fields are present and properly formatted."""

# Network Structure Analysis
NETWORK_STRUCTURE_PROMPT = """Analyze the following research context for network structure:

Context: {context}
Relevant Memories: {memories}

Identify:
1. Network structure and components
2. Node relationships and connections
3. Flow patterns and dynamics
4. Network resilience

Format your response as a JSON object with:
{{
    "structure": {{
        "type": str,
        "description": str,
        "components": List[str]
    }},
    "nodes": [
        {{
            "id": str,
            "type": str,
            "properties": Dict[str, Any]
        }}
    ],
    "connections": [
        {{
            "source": str,
            "target": str,
            "type": str,
            "weight": float,
            "properties": Dict[str, Any]
        }}
    ],
    "flows": [
        {{
            "path": List[str],
            "type": str,
            "volume": float
        }}
    ],
    "resilience": {{
        "score": float,
        "bottlenecks": List[str],
        "recommendations": List[str]
    }}
}}
"""

# Agent Prompt
AGENT_PROMPT = """You are a specialized research agent with expertise in {agent_description}.

Your task is to analyze the following research context:
{context}

Implementation Notes:
{implementation_notes}

Previous Related Work:
{memories}

Return your analysis in the following JSON format:
{{
    "interpretation": "Your detailed interpretation of the research context",
    "recommendations": "Your specific recommendations for next steps",
    "confidence_score": 0.0,  // Your confidence in your analysis (0-1)
    "key_insights": [
        "List of 3-5 key insights from your perspective"
    ],
    "concerns": [
        "List of any potential issues or concerns you identify"
    ],
    "collaboration_points": [
        "List of aspects where collaboration with other agents would be valuable"
    ]
}}

Ensure all required fields are present and properly formatted."""

# System role prompts
SYSTEM_ROLES = {
    "data_analysis": "You are a data analysis expert. Analyze the data and provide metrics.",
    "network_analysis": "You are a network analysis and complex systems expert. Analyze the data and provide network insights.",
    "citation_analysis": "You are a citation and reference analysis expert. Analyze the sources and their relationships.",
    "graph_analysis": "You are a graph theory and network analysis expert. Analyze the graph structure and metrics.",
    "search": "You are a domain-specific search expert. Generate targeted search queries."
}

# Prompt metadata
PROMPT_METADATA = {
    "DOMAIN_SEARCH_QUERY_PROMPT": {
        "purpose": "Generate domain-specific search queries from aspects",
        "input_vars": ["aspect"],
        "version": "1.0.0",
        "last_updated": "2024-03-19"
    },
    "DATA_ANALYSIS_PROMPT": {
        "purpose": "Analyze data and extract metrics, correlations, and findings",
        "input_vars": ["implementation_notes", "memories"],
        "version": "1.0.0",
        "last_updated": "2024-03-19"
    },
    "GRAPH_ANALYSIS_PROMPT": {
        "purpose": "Analyze graph structure and relationships",
        "input_vars": ["context", "memories"],
        "version": "1.0.0",
        "last_updated": "2024-03-19"
    },
    "CITATION_ANALYSIS_PROMPT": {
        "purpose": "Analyze citations and research sources",
        "input_vars": ["research_context"],
        "version": "1.0.0",
        "last_updated": "2024-03-19"
    },
    "NETWORK_ANALYSIS_PROMPT": {
        "purpose": "Analyze network topology, nodes, interactions, and findings",
        "input_vars": ["implementation_notes", "memories"],
        "version": "1.0.0",
        "last_updated": "2024-03-19"
    },
    "NETWORK_STRUCTURE_PROMPT": {
        "purpose": "Analyze network structure and components",
        "input_vars": ["context", "memories"],
        "version": "1.0.0",
        "last_updated": "2024-03-19"
    },
    "AGENT_PROMPT": {
        "purpose": "Analyze research context from a specialized agent perspective",
        "input_vars": ["agent_description", "context", "implementation_notes", "memories"],
        "version": "1.0.0",
        "last_updated": "2024-03-19"
    }
}

# Research Execution
RESEARCH_PROMPT = """Execute research on the following task:

Task: {task}

Consider:
1. What are the key aspects to research?
2. What specific queries would be most effective?
3. How should the research be structured?

Format your response as a JSON object with:
{{
    "query": str,  // Main research query
    "aspects": [  // List of aspects to explore
        {{
            "aspect": str,
            "priority": float,  // 0-1 priority score
            "queries": [str]  // List of specific queries for this aspect
        }}
    ],
    "structure": {{
        "approach": str,  // How to structure the research
        "phases": [str]  // List of research phases
    }}
}}
"""

# Research Analysis
ANALYSIS_PROMPT = """Analyze the following research data:

Research Data: {research_data}

Consider:
1. What are the key findings?
2. What patterns or trends emerge?
3. What are the implications?
4. What further research is needed?

Format your response as a JSON object with:
{{
    "key_metrics": {{
        "relevance_score": float,  // 0-1 score
        "coverage_score": float,  // 0-1 score
        "recency_score": float,  // 0-1 score
        "impact_score": float  // 0-1 score
    }},
    "correlations": [
        {{
            "variables": [str],  // List of related variables
            "coefficient": float,  // Correlation strength
            "description": str  // Description of the relationship
        }}
    ],
    "significant_findings": [
        {{
            "finding": str,  // Description of the finding
            "confidence": float,  // 0-1 confidence score
            "supporting_evidence": [str]  // List of supporting evidence
        }}
    ],
    "analysis": {{
        "methodology": str,  // Description of analysis approach
        "limitations": [str],  // List of analysis limitations
        "recommendations": [str]  // List of recommendations
    }},
    "insights": [str],  // List of key insights
    "confidence": float,  // Overall confidence score (0-1)
    "metadata": {{
        "timestamp": str,  // ISO format timestamp
        "version": str,  // Analysis version
        "source": str  // Source of the analysis
    }}
}}

Focus on providing actual analysis results, not schema definitions.
Make sure your analysis is specific to the research context and provides actionable insights.
"""

# Replan Prompt
REPLAN_PROMPT = """You are a research replanning expert. Your task is to analyze the current research state and suggest improvements.

Task Description:
{task}

Original Plan:
{original_plan}

Execution Results:
{execution_results}

Tool Descriptions:
{tool_descriptions}

Return a JSON object with:
{{
    "analysis": {{
        "issues": [
            {{
                "type": "error|inefficiency|gap",
                "description": "Description of the issue",
                "impact": "High|Medium|Low",
                "affected_steps": [str]
            }}
        ],
        "successful_parts": [str],
        "metrics_analysis": {{
            "key_findings": [str],
            "areas_for_improvement": [str]
        }}
    }},
    "revised_plan": {{
        "steps": [
            {{
                "step_number": int,
                "action": str,
                "tool": str,
                "expected_outcome": str,
                "dependencies": [int]
            }}
        ],
        "estimated_improvements": [str]
    }},
    "recommendations": [
        {{
            "focus": str,
            "rationale": str,
            "priority": "High|Medium|Low"
        {{
    ]
}}""" 