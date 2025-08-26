"""
LLM-D (Domains) Specialist Agent for EnMapper

This module implements the LLM-powered domain assignment specialist that enhances
the neural/rule-based domain detection with contextual reasoning, unknown column
analysis, and intelligent alias generation.

Key Features:
- Unknown column analysis using LLM reasoning
- Contextual domain suggestions based on table/data context
- Natural language explanations for domain assignments
- Auto-generation of domain aliases and patterns
- Fallback and error handling for LLM failures
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from .domain_catalog import DomainDefinition, DomainCatalog, ConfidenceBand
from .domain_assignment import ColumnInfo, DomainAssignment, Evidence
from .model_routing import get_best_model_for_task, TaskType, ModelTier
from settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class LLMDomainContext:
    """Context information for LLM domain analysis."""
    table_name: Optional[str] = None
    table_description: Optional[str] = None
    related_columns: List[str] = None
    business_domain: Optional[str] = None  # e.g., "e-commerce", "finance", "hr"
    data_source: Optional[str] = None  # e.g., "salesforce", "mysql", "csv"
    
    def __post_init__(self):
        if self.related_columns is None:
            self.related_columns = []


@dataclass 
class LLMDomainSuggestion:
    """LLM suggestion for domain assignment."""
    suggested_domain: str
    confidence: float  # 0.0 to 1.0
    reasoning: str
    suggested_aliases: List[str]
    suggested_patterns: List[str]
    context_factors: List[str]


class LLMDomainAgent:
    """LLM-powered domain assignment specialist."""
    
    def __init__(self, catalog: DomainCatalog, model_tier: ModelTier = ModelTier.BALANCED):
        self.catalog = catalog
        self.model_tier = model_tier
        self.settings = get_settings()
        
        # Get the best model for domain analysis
        self.model_name = get_best_model_for_task(TaskType.ANALYSIS, model_tier)
        
        logger.info(f"🤖 LLM Domain Agent initialized with model: {self.model_name}")
    
    async def analyze_unknown_column(self, 
                                   column: ColumnInfo, 
                                   context: LLMDomainContext = None) -> Optional[LLMDomainSuggestion]:
        """Analyze an unknown/ambiguous column using LLM reasoning."""
        
        if context is None:
            context = LLMDomainContext()
        
        try:
            # Build the analysis prompt
            prompt = self._build_unknown_analysis_prompt(column, context)
            
            # Call LLM for analysis
            response = await self._call_llm(prompt, "unknown_column_analysis")
            
            # Parse LLM response
            suggestion = self._parse_llm_suggestion(response)
            
            logger.info(f"🧠 LLM analyzed unknown column '{column.name}' → {suggestion.suggested_domain}")
            return suggestion
            
        except Exception as e:
            logger.error(f"❌ LLM analysis failed for column '{column.name}': {e}")
            return None
    
    async def enhance_domain_assignment(self, 
                                      assignment: DomainAssignment,
                                      column: ColumnInfo,
                                      context: LLMDomainContext = None) -> DomainAssignment:
        """Enhance domain assignment with LLM reasoning and explanations."""
        
        if context is None:
            context = LLMDomainContext()
        
        try:
            # Add LLM explanation for the assignment
            explanation = await self._generate_assignment_explanation(assignment, column, context)
            
            # Add explanation to assignment metadata (we'd need to extend the model)
            # For now, log it
            logger.info(f"🤖 LLM explanation for {column.name} → {assignment.domain_name}: {explanation}")
            
            # If confidence is borderline, ask LLM for second opinion
            if assignment.confidence_band == ConfidenceBand.BORDERLINE:
                second_opinion = await self._get_llm_second_opinion(assignment, column, context)
                if second_opinion:
                    logger.info(f"🔍 LLM second opinion: {second_opinion}")
            
            return assignment
            
        except Exception as e:
            logger.error(f"❌ LLM enhancement failed for column '{column.name}': {e}")
            return assignment
    
    async def suggest_domain_aliases(self, 
                                   domain: DomainDefinition,
                                   failed_columns: List[str]) -> List[str]:
        """Generate new aliases for a domain based on columns that failed to match."""
        
        try:
            prompt = self._build_alias_suggestion_prompt(domain, failed_columns)
            response = await self._call_llm(prompt, "alias_suggestion")
            
            # Parse suggested aliases
            aliases = self._parse_alias_suggestions(response)
            
            logger.info(f"🏷️ LLM suggested {len(aliases)} new aliases for {domain.name}: {aliases}")
            return aliases
            
        except Exception as e:
            logger.error(f"❌ LLM alias suggestion failed for domain '{domain.name}': {e}")
            return []
    
    async def analyze_table_context(self, 
                                  columns: List[ColumnInfo],
                                  context: LLMDomainContext) -> LLMDomainContext:
        """Analyze table structure to infer business domain and context."""
        
        try:
            prompt = self._build_context_analysis_prompt(columns, context)
            response = await self._call_llm(prompt, "context_analysis")
            
            # Parse and enhance context
            enhanced_context = self._parse_context_analysis(response, context)
            
            logger.info(f"🔍 LLM inferred business domain: {enhanced_context.business_domain}")
            return enhanced_context
            
        except Exception as e:
            logger.error(f"❌ LLM context analysis failed: {e}")
            return context
    
    def _build_unknown_analysis_prompt(self, column: ColumnInfo, context: LLMDomainContext) -> str:
        """Build prompt for analyzing unknown columns."""
        
        # Get available domains for reference
        domain_list = []
        for domain in self.catalog.domains.values():
            domain_list.append(f"- {domain.name}: {domain.description}")
        
        prompt = f"""
You are a data domain expert analyzing database columns. Analyze this unknown column and suggest the most appropriate semantic domain.

COLUMN INFORMATION:
- Name: {column.name}
- Data Type: {column.data_type}
- Sample Values: {column.sample_values[:5]}
- Null Count: {column.null_count}/{column.total_count}
- Unique Values: {column.unique_count}

CONTEXT:
- Table: {context.table_name or 'Unknown'}
- Business Domain: {context.business_domain or 'Unknown'}
- Related Columns: {', '.join(context.related_columns) if context.related_columns else 'None'}
- Data Source: {context.data_source or 'Unknown'}

AVAILABLE DOMAINS:
{chr(10).join(domain_list)}

TASK:
Analyze the column and provide your assessment in JSON format:

{{
    "suggested_domain": "domain.name or 'custom.new_domain'",
    "confidence": 0.85,
    "reasoning": "Detailed explanation of why this domain fits",
    "suggested_aliases": ["alias1", "alias2"],
    "suggested_patterns": ["regex_pattern1"],
    "context_factors": ["factor1", "factor2"]
}}

Consider:
1. Column name semantics and patterns
2. Data type compatibility
3. Sample values and their patterns
4. Business context and related columns
5. Common naming conventions

Respond with ONLY the JSON object.
"""
        return prompt.strip()
    
    def _build_alias_suggestion_prompt(self, domain: DomainDefinition, failed_columns: List[str]) -> str:
        """Build prompt for suggesting new domain aliases."""
        
        prompt = f"""
You are a data domain expert. A domain has failed to match some columns that should probably belong to it.

DOMAIN INFORMATION:
- Name: {domain.name}
- Description: {domain.description}
- Current Aliases: {domain.aliases}
- Example Values: {[ex.value for ex in domain.positive_examples[:3]]}

FAILED TO MATCH COLUMNS:
{chr(10).join(f"- {col}" for col in failed_columns)}

TASK:
Suggest additional aliases that would help this domain match the failed columns.
Consider common naming variations, abbreviations, and conventions.

Respond with a JSON array of suggested aliases:
["alias1", "alias2", "alias3"]

Be conservative - only suggest aliases you're confident belong to this domain.
"""
        return prompt.strip()
    
    def _build_context_analysis_prompt(self, columns: List[ColumnInfo], context: LLMDomainContext) -> str:
        """Build prompt for analyzing table context."""
        
        column_info = []
        for col in columns[:10]:  # Limit to first 10 columns
            column_info.append(f"- {col.name} ({col.data_type}): {col.sample_values[:2]}")
        
        prompt = f"""
You are a data analyst examining a database table structure to infer the business domain and context.

TABLE STRUCTURE:
{chr(10).join(column_info)}

CURRENT CONTEXT:
- Table Name: {context.table_name or 'Unknown'}
- Data Source: {context.data_source or 'Unknown'}

TASK:
Analyze the table structure and infer the business domain and context.

Respond in JSON format:
{{
    "business_domain": "e-commerce|finance|hr|crm|erp|marketing|healthcare|logistics|other",
    "likely_entity": "customers|orders|products|employees|transactions|users|other",
    "confidence": 0.85,
    "reasoning": "Why you think this based on column patterns"
}}

Consider column naming patterns, data types, and typical business entities.
"""
        return prompt.strip()
    
    async def _call_llm(self, prompt: str, operation: str) -> str:
        """Make LLM API call with error handling and retries."""
        
        try:
            # This is a placeholder - in real implementation, this would:
            # 1. Use the model routing system to get the right provider/model
            # 2. Make the actual API call (OpenAI, Anthropic, etc.)
            # 3. Handle retries, rate limiting, and errors
            # 4. Log the call for observability
            
            # For now, simulate an LLM response
            if "unknown_column_analysis" in operation:
                return self._simulate_unknown_analysis_response(prompt)
            elif "alias_suggestion" in operation:
                return '["user_id", "uid", "user_identifier"]'
            elif "context_analysis" in operation:
                return '{"business_domain": "e-commerce", "likely_entity": "customers", "confidence": 0.8, "reasoning": "Columns suggest customer/user data with email, phone, names"}'
            else:
                return "LLM analysis completed."
                
        except Exception as e:
            logger.error(f"LLM call failed for {operation}: {e}")
            raise
    
    def _simulate_unknown_analysis_response(self, prompt: str) -> str:
        """Simulate LLM response for unknown column analysis."""
        
        # Extract column name from prompt
        if "user_id" in prompt.lower():
            return json.dumps({
                "suggested_domain": "identifier.user",
                "confidence": 0.9,
                "reasoning": "Column name 'user_id' strongly suggests a user identifier, commonly used in customer/user management systems",
                "suggested_aliases": ["uid", "user_identifier", "customer_id"],
                "suggested_patterns": ["^[0-9]+$", "^usr_[0-9]+$"],
                "context_factors": ["column name pattern", "integer data type", "likely primary key"]
            })
        elif "status" in prompt.lower():
            return json.dumps({
                "suggested_domain": "status.generic",
                "confidence": 0.7,
                "reasoning": "Column name 'status' indicates state information, but specific domain depends on context",
                "suggested_aliases": ["state", "stage", "condition"],
                "suggested_patterns": ["^(active|inactive|pending)$", "^[A-Z_]+$"],
                "context_factors": ["enumerated values", "state tracking"]
            })
        else:
            return json.dumps({
                "suggested_domain": "unknown.data",
                "confidence": 0.3,
                "reasoning": "Insufficient information to determine specific domain",
                "suggested_aliases": [],
                "suggested_patterns": [],
                "context_factors": ["ambiguous column name", "mixed data patterns"]
            })
    
    def _parse_llm_suggestion(self, response: str) -> LLMDomainSuggestion:
        """Parse LLM response into domain suggestion."""
        
        try:
            data = json.loads(response)
            return LLMDomainSuggestion(
                suggested_domain=data.get("suggested_domain", "unknown"),
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", "No reasoning provided"),
                suggested_aliases=data.get("suggested_aliases", []),
                suggested_patterns=data.get("suggested_patterns", []),
                context_factors=data.get("context_factors", [])
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM suggestion: {e}")
            return LLMDomainSuggestion(
                suggested_domain="unknown",
                confidence=0.0,
                reasoning="Failed to parse LLM response",
                suggested_aliases=[],
                suggested_patterns=[],
                context_factors=[]
            )
    
    def _parse_alias_suggestions(self, response: str) -> List[str]:
        """Parse LLM alias suggestions."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error("Failed to parse alias suggestions")
            return []
    
    def _parse_context_analysis(self, response: str, original_context: LLMDomainContext) -> LLMDomainContext:
        """Parse LLM context analysis and enhance original context."""
        try:
            data = json.loads(response)
            return LLMDomainContext(
                table_name=original_context.table_name,
                table_description=data.get("reasoning"),
                related_columns=original_context.related_columns,
                business_domain=data.get("business_domain"),
                data_source=original_context.data_source
            )
        except json.JSONDecodeError:
            logger.error("Failed to parse context analysis")
            return original_context
    
    async def _generate_assignment_explanation(self, 
                                             assignment: DomainAssignment,
                                             column: ColumnInfo,
                                             context: LLMDomainContext) -> str:
        """Generate human-readable explanation for domain assignment."""
        
        prompt = f"""
Explain why the column '{column.name}' was assigned to domain '{assignment.domain_name}' 
with confidence {assignment.confidence_score:.3f} ({assignment.confidence_band.value}).

Evidence:
- Name similarity: {assignment.evidence.name_similarity:.3f}
- Regex strength: {assignment.evidence.regex_strength:.3f}
- Value similarity: {assignment.evidence.value_similarity:.3f}
- Unit compatibility: {assignment.evidence.unit_compatibility:.3f}

Provide a brief, clear explanation in 1-2 sentences.
"""
        
        try:
            response = await self._call_llm(prompt, "explanation")
            return response.strip()
        except:
            return f"Assigned based on {assignment.confidence_score:.1%} confidence from pattern matching and semantic analysis."
    
    async def _get_llm_second_opinion(self,
                                    assignment: DomainAssignment,
                                    column: ColumnInfo,
                                    context: LLMDomainContext) -> Optional[str]:
        """Get LLM second opinion on borderline assignments."""
        
        prompt = f"""
This column assignment has borderline confidence. Please review:

Column: {column.name}
Assigned Domain: {assignment.domain_name}
Confidence: {assignment.confidence_score:.3f}
Sample Values: {column.sample_values[:3]}

Do you agree with this assignment? Respond with:
- "AGREE" if the assignment seems correct
- "DISAGREE" if you think it's wrong
- "UNCERTAIN" if you need more context

Include a brief reason.
"""
        
        try:
            response = await self._call_llm(prompt, "second_opinion")
            return response.strip()
        except:
            return None


# Enhanced domain assignment with LLM integration
async def assign_domains_with_llm(columns: List[ColumnInfo],
                                context: LLMDomainContext = None,
                                catalog: DomainCatalog = None,
                                model_tier: ModelTier = ModelTier.BALANCED) -> Tuple[List[DomainAssignment], Dict[str, Any]]:
    """
    Enhanced domain assignment with LLM analysis for unknown columns.
    
    Returns:
        Tuple of (assignments, llm_insights)
    """
    
    if catalog is None:
        from .domain_catalog import get_domain_catalog
        catalog = get_domain_catalog()
    
    if context is None:
        context = LLMDomainContext()
    
    # Initialize LLM agent
    llm_agent = LLMDomainAgent(catalog, model_tier)
    
    # First, enhance context with table analysis
    enhanced_context = await llm_agent.analyze_table_context(columns, context)
    
    # Run standard domain assignment
    from .domain_assignment import DomainAssignmentEngine
    engine = DomainAssignmentEngine(catalog=catalog)
    assignments = engine.assign_domains(columns)
    
    # Enhance assignments with LLM analysis
    llm_insights = {
        "context_analysis": {
            "business_domain": enhanced_context.business_domain,
            "inferred_entity": getattr(enhanced_context, 'likely_entity', None)
        },
        "unknown_analyses": [],
        "borderline_reviews": [],
        "alias_suggestions": []
    }
    
    for i, assignment in enumerate(assignments):
        column = columns[i]
        
        # Enhance with LLM explanation
        enhanced_assignment = await llm_agent.enhance_domain_assignment(
            assignment, column, enhanced_context
        )
        assignments[i] = enhanced_assignment
        
        # Analyze unknown/low confidence columns with LLM
        if (assignment.domain_name is None or 
            assignment.confidence_band in [ConfidenceBand.LOW, ConfidenceBand.BORDERLINE]):
            
            llm_suggestion = await llm_agent.analyze_unknown_column(column, enhanced_context)
            if llm_suggestion:
                llm_insights["unknown_analyses"].append({
                    "column": column.name,
                    "suggestion": llm_suggestion.suggested_domain,
                    "confidence": llm_suggestion.confidence,
                    "reasoning": llm_suggestion.reasoning,
                    "aliases": llm_suggestion.suggested_aliases
                })
    
    logger.info(f"🤖 LLM-enhanced domain assignment completed: {len(llm_insights['unknown_analyses'])} unknown analyses")
    
    return assignments, llm_insights
