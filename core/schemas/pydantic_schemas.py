"""
Pydantic models for all JSON schemas used in the application.
"""
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, constr

# Enums for constrained string fields
class NodeType(str, Enum):
    AGENT = "agent"
    TOOL_EXECUTION = "tool_execution"
    RESEARCH_STEP = "research_step"
    INSIGHT = "insight"
    DECISION = "decision"
    ARTIFACT = "artifact"
    OBJECTIVE = "objective"
    CONSTRAINT = "constraint"
    REQUIREMENT = "requirement"

class RelationType(str, Enum):
    DEPENDS_ON = "depends_on"
    CONTRIBUTES_TO = "contributes_to"
    INFLUENCES = "influences"
    PRECEDES = "precedes"
    IMPLEMENTS = "implements"
    VALIDATES = "validates"
    CONFLICTS_WITH = "conflicts_with"
    SUPPORTS = "supports"
    REFERENCES = "references"
    USES = "uses"
    PRODUCES = "produces"
    MODIFIES = "modifies"

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# Base Models
class Brainstorm(BaseModel):
    idea: str
    priority: Priority
    needs: str
    supplements: str

class BrainstormingResponse(BaseModel):
    brainstorming: List[Brainstorm]

class PlanStep(BaseModel):
    step_number: int
    action: str
    agent: List[Union[int, str]]  # Can be either int or string like "4: Agent Name"
    reasoning: str
    completion_conditions: str
    tool_suggestions: List[int]
    implementation_notes: str

class FinalPlan(BaseModel):
    plan: List[PlanStep]
    is_valid: bool = True
    errors: Optional[List[str]] = None

class Metrics(BaseModel):
    complexity: int = Field(ge=1, le=10)
    impact: int = Field(ge=1, le=10)
    feasibility: int = Field(ge=1, le=10)

class TechnicalDetail(BaseModel):
    detail: str = Field(min_length=100)
    metrics: Metrics
    validation_criteria: str = Field(min_length=50)

class TechnicalRequirement(BaseModel):
    requirement: str = Field(min_length=30)
    metrics: Dict[str, str]  # Contains threshold, unit, validation_method

class ValidationCriteria(BaseModel):
    success_criteria: str = Field(min_length=30)
    metrics: Dict[str, str]  # Contains threshold, unit
    validation_steps: List[str]

class Dependencies(BaseModel):
    required_tools: List[int]
    technical_dependencies: List[str]
    prerequisite_steps: List[int]

class SubStep(BaseModel):
    description: str = Field(min_length=50)
    technical_requirements: List[TechnicalRequirement]
    implementation_notes: str = Field(min_length=100)
    validation_criteria: ValidationCriteria
    dependencies: Dependencies

class ImplementationAspect(BaseModel):
    aspect: str = Field(min_length=50)
    rationale: str = Field(min_length=50)
    technical_requirements: List[str]

class QualityMetrics(BaseModel):
    technical_depth: int = Field(ge=1, le=10)
    implementation_detail: int = Field(ge=1, le=10)
    validation_coverage: int = Field(ge=1, le=10)
    dependency_completeness: int = Field(ge=1, le=10)

class Analysis(BaseModel):
    current_focus: str = Field(min_length=50)
    technical_details: List[TechnicalDetail]
    implementation_aspects: List[ImplementationAspect]
    substeps: List[SubStep]
    quality_metrics: QualityMetrics

class SpecializedAgentResponse(BaseModel):
    analysis: Analysis
    request_type: Optional[str] = Field(None)
    target_agent: Optional[Union[int, str]] = Field(None)
    action: Optional[str] = Field(None)
    reason: Optional[str] = Field(None)
    details: Optional[str] = Field(None)
    tool_suggestions: Optional[List[int]] = Field(None)

class SummarizationResponse(BaseModel):
    assessment: str
    components_to_retain: List[str]
    components_to_remove: List[str]
    components_to_adjust: List[str]
    next_planning_steps: List[str]
    actionable_insights: List[str]

class StepCompleteResponse(BaseModel):
    step_complete: bool
    depth_sufficient: bool
    quality_metrics: QualityMetrics
    missing_aspects: List[str]

class SupervisorAction(BaseModel):
    action: str
    tool: int
    details: str

class SupervisorResponse(BaseModel):
    research_complete: bool
    next_agent: int
    next_actions: List[SupervisorAction]

class TopicResponse(BaseModel):
    topic: str

class FinalResponse(BaseModel):
    final_response: str

class FeedbackItem(BaseModel):
    section: str
    issue: str
    reason: str
    suggested_fix: str

class ConstraintViolation(BaseModel):
    type: str
    details: str
    replacement: str

class MetaReviewResponse(BaseModel):
    requires_revision: bool
    feedback: List[FeedbackItem]
    constraint_violations: List[ConstraintViolation]

class Revision(BaseModel):
    tag: str
    original_text: str
    revised_text: str
    reason: str

class RevisedResponse(BaseModel):
    original_content: str
    revisions: List[Revision]
    final_content: str

# Tool Output Models
class Correlation(BaseModel):
    variables: List[str]
    strength: float
    significance: str

class Pattern(BaseModel):
    pattern_type: str
    description: str
    confidence: float

class Anomaly(BaseModel):
    type: str
    description: str
    severity: str

class Recommendation(BaseModel):
    action: str
    rationale: str
    priority: str

class ToolDataAnalysis(BaseModel):
    basic_statistics: Dict[str, Any]
    correlations: List[Correlation]
    patterns: List[Pattern]
    anomalies: List[Anomaly]
    recommendations: List[Recommendation]

class GraphNode(BaseModel):
    id: str
    type: str
    metrics: Dict[str, Any]

class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float

class GraphMetrics(BaseModel):
    clustering_coefficient: float
    centrality_measures: Dict[str, Any]

class GraphVisualization(BaseModel):
    layout: str
    recommendations: List[str]

class ToolGraphAnalysis(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    metrics: GraphMetrics
    visualization: GraphVisualization

class Citation(BaseModel):
    source: str
    validity: str
    credibility_score: float
    impact_factor: float

class CitationPattern(BaseModel):
    pattern_type: str
    description: str

class CitationRecommendation(BaseModel):
    suggestion: str
    rationale: str

class ToolCitationAnalysis(BaseModel):
    citations: List[Citation]
    patterns: List[CitationPattern]
    recommendations: List[CitationRecommendation]

class NetworkTopology(BaseModel):
    type: str
    description: str
    key_properties: List[str]

class NetworkNode(BaseModel):
    id: str
    type: str
    importance: float
    attributes: Dict[str, Any]

class NetworkRelationship(BaseModel):
    source: str
    target: str
    type: str
    strength: float

class NetworkCommunity(BaseModel):
    id: str
    description: str
    members: List[str]

class NetworkMetrics(BaseModel):
    density: float
    avg_path_length: float
    clustering_coefficient: float

class NetworkFinding(BaseModel):
    insight: str
    confidence: float
    supporting_evidence: List[str]

class NetworkAnalysisResponse(BaseModel):
    network_topology: NetworkTopology
    key_nodes: List[NetworkNode]
    relationships: List[NetworkRelationship]
    communities: List[NetworkCommunity]
    metrics: NetworkMetrics
    findings: List[NetworkFinding]

# Memory Management Models
class MemoryNodeMetadata(BaseModel):
    timestamp: datetime
    confidence: float
    priority: Priority
    relevance_score: float
    expiry: Optional[datetime] = None

class MemoryNode(BaseModel):
    node_id: str
    node_type: NodeType
    content: Dict[str, Any]
    metadata: MemoryNodeMetadata

class MemoryRelationMetadata(BaseModel):
    timestamp: datetime
    strength: float
    bidirectional: Optional[bool] = None
    context: Optional[str] = None

class MemoryRelation(BaseModel):
    relation_id: str
    source_node: str
    target_node: str
    relation_type: RelationType
    metadata: MemoryRelationMetadata

class TimeRange(BaseModel):
    start: datetime
    end: datetime

class QueryFilters(BaseModel):
    node_types: Optional[List[str]] = None
    relation_types: Optional[List[str]] = None
    time_range: Optional[TimeRange] = None
    relevance_threshold: Optional[float] = None
    confidence_threshold: Optional[float] = None

class QueryContext(BaseModel):
    current_objective: str
    agent_id: str
    step_number: int

class MemoryQuery(BaseModel):
    query_type: str
    filters: Optional[QueryFilters] = None
    context: QueryContext

class MemoryUpdateMetadata(BaseModel):
    reason: str
    confidence: Optional[float] = None
    source: str

class MemoryUpdate(BaseModel):
    operation: str
    data: Union[MemoryNode, MemoryRelation]
    metadata: MemoryUpdateMetadata

class CurrentFocus(BaseModel):
    objective: str
    constraints: List[str]
    requirements: List[str]

class RelevantHistoryItem(BaseModel):
    node_id: str
    summary: str
    relevance_score: float

class ActiveDependency(BaseModel):
    dependency_type: str
    status: str
    blocking: bool

class ContextMetadata(BaseModel):
    window_size: int
    token_count: int
    last_updated: datetime

class MemoryContextWindow(BaseModel):
    current_focus: CurrentFocus
    relevant_history: List[RelevantHistoryItem]
    active_dependencies: List[ActiveDependency]
    context_metadata: ContextMetadata

# ATR Work Unit Models
class WorkUnitInput(BaseModel):
    name: str
    type: str
    description: str
    required: bool

class WorkUnitOutput(BaseModel):
    name: str
    type: str
    description: str

class PerformanceMetric(BaseModel):
    type: str
    target: str

class WorkUnitRequirements(BaseModel):
    agent_type: str
    capabilities: List[str]
    performance_metrics: Optional[Dict[str, PerformanceMetric]] = None

class TemporalConstraints(BaseModel):
    start_time: Optional[datetime] = None
    deadline: Optional[datetime] = None
    max_duration: Optional[float] = None

class Payment(BaseModel):
    token: str
    amount: str

class DataAccessControls(BaseModel):
    allowed_sources: Optional[List[str]] = None
    allowed_queries: Optional[List[str]] = None

class ATRWorkUnit(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    description: str = Field(min_length=50)
    inputs: List[WorkUnitInput]
    outputs: List[WorkUnitOutput]
    requirements: WorkUnitRequirements
    temporal_constraints: Optional[TemporalConstraints] = None
    payment: Optional[Payment] = None
    data_access_controls: Optional[DataAccessControls] = None

class ResponsibilityNFT(BaseModel):
    token_id: str
    agent_id: str
    work_unit_id: str
    action_type: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

# Research Tool Models
class SearchQuery(BaseModel):
    primary_query: str
    alternative_queries: List[str]
    subtopics: List[str]

class KeyMetric(BaseModel):
    metric_name: str
    value: float

class DataCorrelation(BaseModel):
    variables: List[str]
    coefficient: float

class SignificantFinding(BaseModel):
    finding: str
    confidence: float

class DataAnalysisResponse(BaseModel):
    key_metrics: Dict[str, float]
    correlations: List[DataCorrelation]
    significant_findings: List[SignificantFinding]

class GraphTopology(BaseModel):
    type: str
    description: str

class GraphNodeRole(BaseModel):
    id: str
    role: str
    importance: float

class GraphInteraction(BaseModel):
    source: str
    target: str
    type: str
    strength: float

class GraphAnalysisResponse(BaseModel):
    topology: GraphTopology
    key_nodes: List[GraphNodeRole]
    interactions: List[GraphInteraction]
    metrics: Dict[str, float]
    findings: List[SignificantFinding]

class ResearchSource(BaseModel):
    title: str
    url: str
    relevance_score: float

class ResearchInsight(BaseModel):
    insight: str
    supporting_sources: List[str]
    confidence: float

class ResearchCitationPattern(BaseModel):
    pattern: str
    description: str
    significance: float

class CitationAnalysisResponse(BaseModel):
    sources: List[ResearchSource]
    key_insights: List[ResearchInsight]
    patterns: List[ResearchCitationPattern]

class JoinerActionContent(BaseModel):
    """Content for joiner action."""
    success: Optional[bool] = None
    quality_score: Optional[int] = Field(None, ge=0, le=100)
    summary: Optional[str] = None
    error_analysis: Optional[str] = None
    suggested_improvements: Optional[str] = None
    synthesis: Optional[Dict[str, Any]] = None
    document_path: Optional[str] = None

class JoinerAction(BaseModel):
    """Action for joiner response."""
    type: str = Field(..., pattern="^(final_response|replan)$")
    content: JoinerActionContent

class JoinerResponse(BaseModel):
    """Response from joiner component."""
    thought: str
    action: JoinerAction
    metrics: Optional[Dict[str, Any]] = None
    operations: Optional[List[Dict[str, Any]]] = None
    synthesis: Optional[Dict[str, Any]] = None
    document_path: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None