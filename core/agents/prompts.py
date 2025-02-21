# prompts.py
cot32_prompt = """
You are the Research Lead. Your task is to create a CONCEPTUAL research plan outline for deep exploration with a single specialized agent. Focus on brainstorming, prioritizing key areas, identifying research needs, and suggesting helpful conceptual supplements.

Research Topic:
{research_topic}

User Research Request:
{task}

Selected Agent:
{agent_indices}

Objective:
- Brainstorm potential research directions for deep exploration
- Prioritize the most crucial components of the user's request
- Identify any gaps in existing knowledge or resources
- Suggest helpful conceptual supplements (e.g., related research areas, theoretical frameworks)
- Assess the overall feasibility and potential impact of the research

Reasoning:
- Think deeply and systematically about the research topic
- Focus on thorough exploration of each aspect
- Emphasize depth over breadth
- Plan for progressive deepening of understanding

Previous Research Summary (if any):
{previous_research_summary}

Respond ONLY with a JSON object in this format:

{{
  "brainstorming": [
    {{
      "idea": "A concise description of a research idea.",
      "priority": "high",  // or "medium", "low"
      "needs": "Comma-separated list of specific conditions needed for this idea",
      "supplements": "Related concepts or areas of research.  This MUST be a single string."
    }},
    // ... more brainstorming ideas
  ]
}}

Be concise but thorough. Focus on deep exploration rather than broad coverage. Prioritize ideas that build upon each other for progressive understanding.
"""

final_plan_prompt = """
You are creating a detailed research plan based on the initial brainstorming ideas.
Your task is to organize these ideas into concrete, executable research steps.

Initial Plan:
{initial_plan}

Available Agents:
{agent_indices}

Available Tools:
{tool_indices}

Agent ID:
{agent_id}

Updates (if any):
{updates}

Your task is to:
1. Create a detailed, step-by-step research plan
2. Assign appropriate agents to each step
3. Specify completion conditions for each step
4. Suggest relevant tools for each step
5. Provide implementation notes

CRITICAL REQUIREMENTS:
1. Each step must be computational/analytical only (no physical actions)
2. Steps must build logically on each other
3. Include clear completion conditions
4. Provide detailed implementation notes
5. Ensure tool suggestions match available tools
6. Keep steps focused and achievable
7. ALL agent IDs must be integers (not strings)
8. ALL tool IDs must be integers (not strings)

Respond with a JSON object following this schema:
{{
    "plan": [
        {{
            "step_number": 1,  // Must be integer
            "action": "Detailed description of the research action",
            "agent": [1, 2],  // Must be array of integers (not strings)
            "reasoning": "Why this step is necessary and how it builds on previous steps",
            "completion_conditions": "Specific conditions that indicate step completion",
            "tool_suggestions": [1, 2],  // Must be array of integers (not strings)
            "implementation_notes": "Detailed notes on how to implement this step"
        }}
    ]
}}

VALIDATION RULES:
1. step_number must be integer
2. agent must be array of integers (not strings)
3. tool_suggestions must be array of integers (not strings)
4. No string numbers allowed (e.g. "1" is invalid, must be 1)
"""

step_complete_prompt = """
Evaluate if the provided agent response demonstrates sufficient depth and quality for the current step.

Agent: {agent_name}
Agent Response: {agent_response}
Action Description: {action_description}
Current Step: {current_step}
Completion Conditions: {completion_conditions}
Current Depth: {current_depth}
Max Depth: {max_depth}

Evaluation Criteria:
1. TECHNICAL DEPTH (Score 0-10)
- Are core technical concepts explained with implementation details? (2 points)
- Are specifications concrete with numbers and metrics? (2 points)
- Are edge cases addressed with mitigation strategies? (2 points)
- Is the analysis technically rigorous with examples? (2 points)
- Are security and performance considered with metrics? (2 points)

2. SUBSTEP QUALITY (Score 0-10)
- Are substeps clearly defined with specific scope? (2 points)
- Do they have measurable technical requirements? (2 points)
- Is implementation guidance detailed with patterns? (2 points)
- Are validation criteria measurable with thresholds? (2 points)
- Is the dependency graph complete and logical? (2 points)

3. IMPLEMENTATION DETAIL (Score 0-10)
- Are technical specifications complete with versions? (2 points)
- Are architectural decisions explained with tradeoffs? (2 points)
- Are tools and configurations specifically defined? (2 points)
- Are integration requirements detailed with protocols? (2 points)
- Are error handling and monitoring specified? (2 points)

4. PROGRESSION (Score 0-10)
- Does the analysis build on previous knowledge? (2 points)
- Is there clear progression in technical depth? (2 points)
- Are dependencies and relationships mapped? (2 points)
- Is the depth appropriate for the current level? (2 points)
- Are insights from previous steps incorporated? (2 points)

Quality Thresholds:
- Minimum score of 7 in each category to be considered sufficient
- All sections must have concrete examples and metrics
- Each substep must have at least 3 technical requirements
- Implementation notes must include architectural decisions
- Validation criteria must be measurable

Respond ONLY with a JSON object in the following format:
{{
    "step_complete": "bool",  // true only if ALL criteria are met
    "depth_sufficient": "bool",  // true if depth is appropriate for current level
    "quality_metrics": {{
        "technical_depth": "0-10",  // Score for technical depth
        "substep_quality": "0-10",  // Score for substep decomposition
        "implementation_detail": "0-10",  // Score for implementation specifics
        "progression": "0-10"  // Score for knowledge progression
    }},
    "missing_aspects": [  // List any missing or insufficient aspects
        "string"
    ]
}}
"""

supervisor_prompt = """
You are the Research Lead overseeing a deep exploration research project on:

{research_topic}

Your role is to assess the current progress, evaluate exploration depth, and guide the next steps for the selected agent.

Selected Agent:
{agent_indices}

Available Tools:
{tool_indices}

Final Research Plan:
{final_plan}

Current Progress:
- Steps Completed: {completed_steps}
- Conversation History: {conversation_history}

Current Step Context:
- Completed Sub-Steps: {completed_substeps}
- Agent Responses: {agent_responses}
- Previous Feedback: {supervisor_feedback}

Current Step Information:
- Action: {current_action}
- Implementation Notes: {implementation_notes}

Respond ONLY with a JSON object in the following format:

{{
    "research_complete": "bool",  // True if research is complete, False otherwise
    "next_agent": {agent_id},  // Always use the same agent ID to maintain focus
    "next_actions": [          // List of actions for deeper exploration
        {{
            "action": "string",  // Detailed action description
            "tool": "int",         // Index of tool to use (0 if no tool needed)
            "details": "string"  // Specific instructions for deep exploration
        }}
    ]
}}

**Important:**
- Maintain focus on the selected agent throughout
- Ensure each action builds on previous findings
- Guide progressively deeper exploration
- Use only available tool indices
- Provide detailed exploration instructions
- Consider current depth and progress
- Ensure valid JSON format
- Do not include collaboration suggestions
"""

specialized_agent_prompt = """
You are {agent_name}, a specialized AI agent with expertise in {agent_description}.

You are analyzing tool results from step {current_step} of a research process. The implementation context is:
{implementation_notes}

The tool results to analyze are:
{tool_results}

Analyze these results from your expert perspective. Your response should be a JSON object with the following structure:
{{
    "interpretation": "Your detailed interpretation of the tool results, focusing on aspects relevant to your expertise",
    "recommendations": "Your specific recommendations for next steps based on the results",
    "confidence_score": 0.0-1.0,  // Your confidence in your analysis
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

Ensure your analysis is thorough and reflects your specialized expertise."""

summarization_prompt = """
I want you to assess the request-aligned components of the conceptual plan. Ensure your output includes strong instructions
for continuing the next planning or research steps - remember your goal is not to enact or include plans to 
enact the goals provided, but simply to provide actionable insight via summary.

# Previous Outputs:

{input}

Respond in a well-structured, human-readable format:

**Assessment and Retention of Request-Aligned Components**
[List of aligned components]

**Components to Remove**
[List of components to remove]

**Components to Adjust**
[List of components to adjust]

**Next Planning Steps**
[List of next steps]

**Actionable Insights**
[List of actionable insights]
"""

action_extraction_prompt = """
Extract the next steps and supporting context from the summary to give the next agent a direct action plan to proceed. 
Focus on the actions needed to move the plan forward, ensuring the agent does not repeat previous steps. 

# Summary to extract from:

{input}

Respond ONLY with a bulleted list of actions, along with context if necessary:

* [Action 1] (Context: [Reasoning or details])
* [Action 2] (Context: [Reasoning or details])
* [Action 3] (Context: [Reasoning or details])
* ... 
"""

topic_prompt = """Given the user's research request, generate a focused research topic.

You must respond with a JSON object containing EXACTLY these fields:
{{
    "topic": "The focused research topic",
    "reasoning": "Your reasoning for choosing this topic",
    "key_aspects": ["List of key aspects to investigate"]
}}

IMPORTANT: The field MUST be named "topic", not "research_topic".

User Input: {user_input}
Current Time: {current_time}
"""

final_prompt = """
Given the following history and user input, provide a final resolution to the query.

# Conversation History:
{history}

# User Input:
{input}

Objective: 
- Carefully analyze the research plan and summarize its key findings.
- Consider the implications of the plan's recommendations.
- Evaluate the strengths and weaknesses of the plan.
- Identify potential areas for further research or refinement.

Reasoning:
- Provide a well-structured and detailed analysis of the plan, demonstrating a deep understanding of its content.
- Consider the context of the research topic and the goals of the research project.
- Explain the rationale behind your conclusions and recommendations.

Respond **ONLY** with a JSON object in the following format:

{{
    "final_response": "string"
}}

Ensure your response is valid JSON and follows the schema exactly.
"""

meta_review_prompt = """
You are a quality control reviewer ensuring research steps are realistic and constrained to available tools and capabilities.

Available Tools:
{tool_indices}

Response to Review:
{agent_response}

Your task is to:
1. Identify any unrealistic actions or requirements:
   - Human interactions (surveys, meetings, committees)
   - Unavailable tools or resources
   - External dependencies outside system capabilities
   
2. Assess technical quality and completeness:
   - Concrete vs vague specifications
   - Technical depth and detail
   - Implementation feasibility
   
3. Suggest specific improvements:
   - Replacements for unrealistic actions
   - Technical details to add
   - Areas needing more depth

Respond with a JSON object following this schema:
{{
    "requires_revision": "boolean",
    "feedback": [
        {{
            "section": "Specific section or aspect",
            "issue": "Description of the issue",
            "reason": "Why this is problematic",
            "suggested_fix": "Specific suggestion for improvement"
        }}
    ],
    "constraint_violations": [
        {{
            "type": "tool_unavailable|human_interaction|external_dependency",
            "details": "Description of the violation",
            "replacement": "Suggested replacement action within constraints"
        }}
    ]
}}
"""

revision_prompt = """
You are revising your previous response based on quality control feedback.

Original Response:
{original_response}

Feedback:
{feedback}

Constraint Violations:
{violations}

Your task is to revise the response:
1. Address each piece of feedback
2. Replace constraint violations with suggested alternatives
3. Tag revised sections with <revision> tags
4. Explain each revision

Respond with a JSON object following this schema:
{{
    "original_content": "Full original response",
    "revisions": [
        {{
            "tag": "Unique revision identifier",
            "original_text": "Text being replaced",
            "revised_text": "New text",
            "reason": "Explanation of the change"
        }}
    ],
    "final_content": "Complete revised response with <revision> tags"
}}
"""
