# schemas.py
cot32_schema = {
    "type": "object",
    "properties": {
        "brainstorming": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idea": {"type": "string"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "needs": {"type": "string"},
                    "supplements": {"type": "string"}
                },
                "required": ["idea", "priority", "needs", "supplements"]
            }
        }
    },
    "required": ["brainstorming"]
}

final_plan_schema = {
    "type": "object",
    "properties": {
        "plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_number": {"type": "integer"},
                    "action": {"type": "string"},
                    "agent": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "integer"},
                                {"type": "string", "pattern": "^\\d+:\\s*.*$"}  # Matches patterns like "4: Agent Name"
                            ]
                        }
                    },
                    "reasoning": {"type": "string"},
                    "completion_conditions": {"type": "string"},
                    "tool_suggestions": {"type": "array", "items": {"type": "integer"}},
                    "implementation_notes": {"type": "string"}
                },
                "required": ["step_number", "action", "agent", "reasoning", "completion_conditions", "tool_suggestions",
                             "implementation_notes"]
            }
        }
    },
    "required": ["plan"]
}

specialized_agent_schema = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "object",
            "properties": {
                "current_focus": {
                    "type": "string",
                    "minLength": 50,
                    "description": "Clear description of the current analysis focus"
                },
                "technical_details": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "detail": {
                                "type": "string",
                                "minLength": 100,
                                "description": "Comprehensive technical detail with methodology and rationale"
                            },
                            "metrics": {
                                "type": "object",
                                "properties": {
                                    "complexity": {"type": "integer", "minimum": 1, "maximum": 10},
                                    "impact": {"type": "integer", "minimum": 1, "maximum": 10},
                                    "feasibility": {"type": "integer", "minimum": 1, "maximum": 10}
                                },
                                "required": ["complexity", "impact", "feasibility"],
                                "additionalProperties": False
                            },
                            "validation_criteria": {
                                "type": "string",
                                "minLength": 50,
                                "description": "Specific criteria to validate this technical aspect"
                            }
                        },
                        "required": ["detail", "metrics", "validation_criteria"],
                        "additionalProperties": False
                    },
                    "minItems": 1,
                    "description": "Array of technical details relevant to the analysis"
                },
                "implementation_aspects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "aspect": {
                                "type": "string",
                                "minLength": 50,
                                "description": "Description of implementation consideration"
                            },
                            "rationale": {
                                "type": "string",
                                "minLength": 50,
                                "description": "Justification for this implementation aspect"
                            },
                            "technical_requirements": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "minLength": 20
                                },
                                "minItems": 1,
                                "description": "List of specific technical requirements"
                            }
                        },
                        "required": ["aspect", "rationale", "technical_requirements"],
                        "additionalProperties": False
                    },
                    "minItems": 1,
                    "description": "Array of implementation considerations"
                },
                "substeps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "minLength": 50,
                                "description": "Clear description of the substep"
                            },
                            "technical_requirements": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "requirement": {
                                            "type": "string",
                                            "minLength": 30
                                        },
                                        "metrics": {
                                            "type": "object",
                                            "properties": {
                                                "threshold": {"type": "string"},
                                                "unit": {"type": "string"},
                                                "validation_method": {"type": "string", "minLength": 20}
                                            },
                                            "required": ["threshold", "unit", "validation_method"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["requirement", "metrics"],
                                    "additionalProperties": False
                                },
                                "minItems": 1
                            },
                            "implementation_notes": {
                                "type": "string",
                                "minLength": 100,
                                "description": "Detailed implementation guidance"
                            },
                            "validation_criteria": {
                                "type": "object",
                                "properties": {
                                    "success_criteria": {"type": "string", "minLength": 30},
                                    "metrics": {
                                        "type": "object",
                                        "properties": {
                                            "threshold": {"type": "string"},
                                            "unit": {"type": "string"}
                                        },
                                        "required": ["threshold", "unit"],
                                        "additionalProperties": False
                                    },
                                    "validation_steps": {
                                        "type": "array",
                                        "items": {"type": "string", "minLength": 20},
                                        "minItems": 1
                                    }
                                },
                                "required": ["success_criteria", "metrics", "validation_steps"],
                                "additionalProperties": False
                            },
                            "dependencies": {
                                "type": "object",
                                "properties": {
                                    "required_tools": {
                                        "type": "array",
                                        "items": {"type": "integer", "minimum": 1},
                                        "minItems": 1
                                    },
                                    "technical_dependencies": {
                                        "type": "array",
                                        "items": {"type": "string", "minLength": 10},
                                        "minItems": 1
                                    },
                                    "prerequisite_steps": {
                                        "type": "array",
                                        "items": {"type": "integer", "minimum": 1},
                                        "minItems": 1
                                    }
                                },
                                "required": ["required_tools", "technical_dependencies", "prerequisite_steps"],
                                "additionalProperties": False
                            }
                        },
                        "required": ["description", "technical_requirements", "implementation_notes", "validation_criteria", "dependencies"],
                        "additionalProperties": False
                    },
                    "minItems": 1,
                    "description": "Array of execution substeps"
                },
                "quality_metrics": {
                    "type": "object",
                    "properties": {
                        "technical_depth": {"type": "integer", "minimum": 1, "maximum": 10},
                        "implementation_detail": {"type": "integer", "minimum": 1, "maximum": 10},
                        "validation_coverage": {"type": "integer", "minimum": 1, "maximum": 10},
                        "dependency_completeness": {"type": "integer", "minimum": 1, "maximum": 10}
                    },
                    "required": ["technical_depth", "implementation_detail", "validation_coverage", "dependency_completeness"],
                    "additionalProperties": False
                }
            },
            "required": ["current_focus", "technical_details", "implementation_aspects", "substeps", "quality_metrics"],
            "additionalProperties": False
        },
        "request_type": {"type": ["string", "null"], "enum": ["collaboration", None]},
        "target_agent": {"type": ["integer", "string", "null"]},
        "action": {"type": ["string", "null"]},
        "reason": {"type": ["string", "null"]},
        "details": {"type": ["string", "null"]},
        "tool_suggestions": {"type": ["array", "null"], "items": {"type": "integer"}}
    },
    "required": ["analysis"],
    "additionalProperties": False
}


summarization_schema = {
    "type": "object",
    "properties": {
        "assessment": {"type": "string"},  # Overall assessment
        "components_to_retain": {"type": "array", "items": {"type": "string"}},
        "components_to_remove": {"type": "array", "items": {"type": "string"}},
        "components_to_adjust": {"type": "array", "items": {"type": "string"}},
        "next_planning_steps": {"type": "array", "items": {"type": "string"}},
        "actionable_insights": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["assessment", "next_planning_steps", "actionable_insights"],  # Other fields can be empty lists
}

step_complete_schema = {
    "type": "object",
    "properties": {
        "step_complete": {"type": "boolean"},
        "depth_sufficient": {"type": "boolean"},
        "quality_metrics": {
            "type": "object",
            "properties": {
                "technical_depth": {"type": "integer", "minimum": 0, "maximum": 10},
                "substep_quality": {"type": "integer", "minimum": 0, "maximum": 10},
                "implementation_detail": {"type": "integer", "minimum": 0, "maximum": 10},
                "progression": {"type": "integer", "minimum": 0, "maximum": 10}
            },
            "required": ["technical_depth", "substep_quality", "implementation_detail", "progression"],
            "additionalProperties": False
        },
        "missing_aspects": {
            "type": "array",
            "items": {"type": "string", "minLength": 10}
        }
    },
    "required": ["step_complete", "depth_sufficient", "quality_metrics", "missing_aspects"]
}


supervisor_schema = {
    "type": "object",
    "properties": {
        "research_complete": {"type": "boolean"},
        "next_agent": {"type": "integer"},
        "next_actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "tool": {"type": "integer"},
                    "details": {"type": "string"}
                },
                "required": ["action", "tool", "details"]
            }
        }
    },
    "required": ["research_complete", "next_agent", "next_actions"]
}


topic_schema = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"}
    },
    "required": ["topic"]
}


final_response_schema = {
    "type": "object",
    "properties": {
        "final_response": {"type": "string"}
    },
    "required": ["final_response"]
}

meta_review_schema = {
    "type": "object",
    "properties": {
        "requires_revision": {"type": "boolean"},
        "feedback": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section": {"type": "string"},
                    "issue": {"type": "string"},
                    "reason": {"type": "string"},
                    "suggested_fix": {"type": "string"}
                },
                "required": ["section", "issue", "reason", "suggested_fix"]
            }
        },
        "constraint_violations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["tool_unavailable", "human_interaction", "external_dependency"]},
                    "details": {"type": "string"},
                    "replacement": {"type": "string"}
                },
                "required": ["type", "details", "replacement"]
            }
        }
    },
    "required": ["requires_revision", "feedback", "constraint_violations"]
}

revised_response_schema = {
    "type": "object",
    "properties": {
        "original_content": {"type": "string"},
        "revisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tag": {"type": "string"},
                    "original_text": {"type": "string"},
                    "revised_text": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["tag", "original_text", "revised_text", "reason"]
            }
        },
        "final_content": {"type": "string"}
    },
    "required": ["original_content", "revisions", "final_content"]
}

# Tool Output Schemas
tool_data_analysis_schema = {
    "type": "object",
    "properties": {
        "basic_statistics": {
            "type": "object",
            "description": "Basic statistical measures of the data"
        },
        "correlations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "variables": {"type": "array", "items": {"type": "string"}},
                    "strength": {"type": "number"},
                    "significance": {"type": "string"}
                }
            }
        },
        "patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pattern_type": {"type": "string"},
                    "description": {"type": "string"},
                    "confidence": {"type": "number"}
                }
            }
        },
        "anomalies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                    "severity": {"type": "string"}
                }
            }
        },
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "rationale": {"type": "string"},
                    "priority": {"type": "string"}
                }
            }
        }
    },
    "required": ["basic_statistics", "correlations", "patterns", "anomalies", "recommendations"]
}

tool_graph_analysis_schema = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                    "metrics": {"type": "object"}
                }
            }
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "weight": {"type": "number"}
                }
            }
        },
        "metrics": {
            "type": "object",
            "properties": {
                "clustering_coefficient": {"type": "number"},
                "centrality_measures": {"type": "object"}
            }
        },
        "visualization": {
            "type": "object",
            "properties": {
                "layout": {"type": "string"},
                "recommendations": {"type": "array", "items": {"type": "string"}}
            }
        }
    },
    "required": ["nodes", "edges", "metrics", "visualization"]
}

tool_citation_analysis_schema = {
    "type": "object",
    "properties": {
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "validity": {"type": "string"},
                    "credibility_score": {"type": "number"},
                    "impact_factor": {"type": "number"}
                }
            }
        },
        "patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pattern_type": {"type": "string"},
                    "description": {"type": "string"}
                }
            }
        },
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "suggestion": {"type": "string"},
                    "rationale": {"type": "string"}
                }
            }
        }
    },
    "required": ["citations", "patterns", "recommendations"]
}

tool_network_analysis_schema = {
    "type": "object",
    "properties": {
        "topology": {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "description": {"type": "string"}
            }
        },
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                    "centrality": {"type": "number"}
                }
            }
        },
        "interactions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"type": "string"},
                    "strength": {"type": "number"}
                }
            }
        },
        "metrics": {
            "type": "object",
            "properties": {
                "density": {"type": "number"},
                "diameter": {"type": "number"},
                "average_path_length": {"type": "number"}
            }
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "finding": {"type": "string"},
                    "confidence": {"type": "number"}
                }
            }
        }
    },
    "required": ["topology", "nodes", "interactions", "metrics", "findings"]
}

# Memory Management Schemas
memory_node_schema = {
    "type": "object",
    "properties": {
        "node_id": {"type": "string"},
        "node_type": {
            "type": "string",
            "enum": [
                "agent",
                "tool_execution",
                "research_step",
                "insight",
                "decision",
                "artifact",
                "objective",
                "constraint",
                "requirement"
            ]
        },
        "content": {"type": "object"},
        "metadata": {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string", "format": "date-time"},
                "confidence": {"type": "number"},
                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                "relevance_score": {"type": "number"},
                "expiry": {"type": "string", "format": "date-time", "nullable": True}
            },
            "required": ["timestamp", "confidence", "priority", "relevance_score"]
        }
    },
    "required": ["node_id", "node_type", "content", "metadata"]
}

memory_relation_schema = {
    "type": "object",
    "properties": {
        "relation_id": {"type": "string"},
        "source_node": {"type": "string"},
        "target_node": {"type": "string"},
        "relation_type": {
            "type": "string",
            "enum": [
                "depends_on",
                "contributes_to",
                "influences",
                "precedes",
                "implements",
                "validates",
                "conflicts_with",
                "supports",
                "references",
                "uses",
                "produces",
                "modifies"
            ]
        },
        "metadata": {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string", "format": "date-time"},
                "strength": {"type": "number"},
                "bidirectional": {"type": "boolean"},
                "context": {"type": "string"}
            },
            "required": ["timestamp", "strength"]
        }
    },
    "required": ["relation_id", "source_node", "target_node", "relation_type", "metadata"]
}

memory_query_schema = {
    "type": "object",
    "properties": {
        "query_type": {
            "type": "string",
            "enum": [
                "context_retrieval",
                "decision_chain",
                "dependency_check",
                "impact_analysis",
                "relevance_search",
                "constraint_validation"
            ]
        },
        "filters": {
            "type": "object",
            "properties": {
                "node_types": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "relation_types": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "time_range": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "format": "date-time"},
                        "end": {"type": "string", "format": "date-time"}
                    }
                },
                "relevance_threshold": {"type": "number"},
                "confidence_threshold": {"type": "number"}
            }
        },
        "context": {
            "type": "object",
            "properties": {
                "current_objective": {"type": "string"},
                "agent_id": {"type": "string"},
                "step_number": {"type": "integer"}
            }
        }
    },
    "required": ["query_type", "context"]
}

memory_update_schema = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": ["add_node", "add_relation", "update_node", "update_relation", "delete_node", "delete_relation"]
        },
        "data": {
            "type": "object",
            "oneOf": [
                {"$ref": "#/definitions/memory_node_schema"},
                {"$ref": "#/definitions/memory_relation_schema"}
            ]
        },
        "metadata": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "confidence": {"type": "number"},
                "source": {"type": "string"}
            },
            "required": ["reason", "source"]
        }
    },
    "required": ["operation", "data", "metadata"]
}

memory_context_window_schema = {
    "type": "object",
    "properties": {
        "current_focus": {
            "type": "object",
            "properties": {
                "objective": {"type": "string"},
                "constraints": {"type": "array", "items": {"type": "string"}},
                "requirements": {"type": "array", "items": {"type": "string"}}
            }
        },
        "relevant_history": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "summary": {"type": "string"},
                    "relevance_score": {"type": "number"}
                }
            }
        },
        "active_dependencies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dependency_type": {"type": "string"},
                    "status": {"type": "string"},
                    "blocking": {"type": "boolean"}
                }
            }
        },
        "context_metadata": {
            "type": "object",
            "properties": {
                "window_size": {"type": "integer"},
                "token_count": {"type": "integer"},
                "last_updated": {"type": "string", "format": "date-time"}
            }
        }
    },
    "required": ["current_focus", "relevant_history", "context_metadata"]
}

atr_work_unit_schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 256,
            "description": "Name of the work unit"
        },
        "description": {
            "type": "string",
            "minLength": 50,
            "description": "Detailed description of the work unit"
        },
        "inputs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                    "required": {"type": "boolean"}
                },
                "required": ["name", "type", "description"]
            }
        },
        "outputs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["name", "type", "description"]
            }
        },
        "requirements": {
            "type": "object",
            "properties": {
                "agent_type": {"type": "string"},
                "capabilities": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "performance_metrics": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "target": {"type": "string"}
                        },
                        "required": ["type", "target"]
                    }
                }
            },
            "required": ["agent_type", "capabilities"]
        },
        "temporal_constraints": {
            "type": "object",
            "properties": {
                "start_time": {
                    "type": "string",
                    "format": "date-time"
                },
                "deadline": {
                    "type": "string",
                    "format": "date-time"
                },
                "max_duration": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Maximum duration in hours"
                }
            }
        },
        "payment": {
            "type": "object",
            "properties": {
                "token": {"type": "string"},
                "amount": {"type": "string"}
            },
            "required": ["token", "amount"]
        },
        "data_access_controls": {
            "type": "object",
            "properties": {
                "allowed_sources": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "allowed_queries": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    },
    "required": [
        "name",
        "description",
        "inputs",
        "outputs",
        "requirements"
    ]
}

responsibility_nft_schema = {
    "type": "object",
    "properties": {
        "token_id": {
            "type": "string",
            "description": "Unique identifier for the NFT"
        },
        "agent_id": {
            "type": "string",
            "description": "ID of the agent that owns the NFT"
        },
        "work_unit_id": {
            "type": "string",
            "description": "ID of the work unit this NFT relates to"
        },
        "action_type": {
            "type": "string",
            "description": "Type of action performed"
        },
        "timestamp": {
            "type": "string",
            "format": "date-time",
            "description": "When the action was performed"
        },
        "metadata": {
            "type": "object",
            "description": "Additional metadata about the action"
        }
    },
    "required": [
        "token_id",
        "agent_id", 
        "work_unit_id",
        "action_type",
        "timestamp"
    ]
}