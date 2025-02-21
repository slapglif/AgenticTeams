from langchain.prompts import PromptTemplate

NETWORK_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["implementation_notes"],
    template="""Analyze the provided network data and generate a structured analysis in JSON format.

Context: {implementation_notes}

Please provide your analysis in the following JSON format:
{{
    "network_topology": {{
        "type": "string",
        "description": "string",
        "key_properties": ["string"]
    }},
    "key_nodes": [
        {{
            "id": "string",
            "type": "string",
            "importance": 0.0,
            "attributes": {{"attr_name": "value"}}
        }}
    ],
    "relationships": [
        {{
            "source": "string",
            "target": "string",
            "type": "string",
            "strength": 0.0
        }}
    ],
    "communities": [
        {{
            "id": "string",
            "description": "string",
            "members": ["string"]
        }}
    ],
    "metrics": {{
        "density": 0.0,
        "avg_path_length": 0.0,
        "clustering_coefficient": 0.0
    }},
    "findings": [
        {{
            "insight": "string",
            "confidence": 0.0,
            "supporting_evidence": ["string"]
        }}
    ]
}}"""
) 