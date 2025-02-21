"""
Intelligent data processor module that uses LLMs to parse and validate complex data structures.
"""

import json
from typing import Any, Dict, Optional, Union, Type
from loguru import logger
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from datetime import datetime, UTC
from jsonschema import validate, ValidationError
from pydantic import BaseModel
from core.shared.settings import logger
from core.shared.settings import build_llm
from core.compile.utils import create_chain

class IntelligentDataProcessor:
    def __init__(self):
        self.logger = logger
        self.healing_prompts = {
            "json": """Parse and heal this potentially malformed JSON data.
            If the data is valid JSON, return it unchanged.
            If not, intelligently reconstruct it while preserving the original structure and intent.
            
            Input Data: {data}
            Expected Schema: {schema}
            
            Return valid JSON that matches the schema.""",
            
            "type_conversion": """Convert data types in this JSON structure to match the schema.
            Pay special attention to:
            - Converting string numbers to actual numeric types
            - Handling arrays and nested objects
            - Maintaining data integrity
            
            Input Data: {data}
            Type Requirements: {type_info}
            
            Return the data with correct types.""",
            
            "schema_migration": """Adapt this data structure to match the target schema while preserving information.
            Handle:
            - Missing required fields (provide reasonable defaults)
            - Deprecated fields (map to new fields if possible)
            - Schema version differences
            
            Source Data: {data}
            Target Schema: {schema}
            
            Return data matching the target schema."""
        }

    async def process_data(
        self,
        data: Any,
        expected_schema: Optional[Type[BaseModel]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process data with intelligent analysis."""
        try:
            # Create chain
            chain = create_chain(
                prompt=ChatPromptTemplate.from_template("""
                Process and analyze this data:
                
                Data: {data}
                Context: {context}
                
                Return a structured analysis with:
                1. Key insights
                2. Important patterns
                3. Anomalies or issues
                4. Recommendations
                """),
                output_mode='json',
                output_parser=PydanticOutputParser(pydantic_object=expected_schema) if expected_schema else None
            )
            
            # Execute chain
            result = await chain.ainvoke({
                "data": str(data),
                "context": context or {}
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Data processing failed: {str(e)}")
            raise

    def _parse_metrics(self, metrics_str: str) -> Dict[str, float]:
        """Parse metrics string into a dictionary of float values."""
        if isinstance(metrics_str, dict):
            return metrics_str
            
        if not isinstance(metrics_str, str):
            return {}
            
        try:
            # Split metrics string and convert to dictionary
            metrics_dict = {}
            for metric in metrics_str.split(", "):
                if "=" in metric:
                    key, value = metric.split("=")
                    metrics_dict[key.strip()] = float(value.strip())
            return metrics_dict
        except Exception:
            return {}

    def _extract_type_info(self, schema: Dict[str, Any]) -> str:
        """Extract type information from JSON schema."""
        type_info = {}
        
        def extract_types(s: Dict[str, Any], path: str = "") -> None:
            if "type" in s:
                type_info[path] = s["type"]
            if "properties" in s:
                for prop, value in s["properties"].items():
                    new_path = f"{path}.{prop}" if path else prop
                    extract_types(value, new_path)
                    
        extract_types(schema)
        return json.dumps(type_info) 