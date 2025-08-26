"""
Cost-Optimized Model Routing Configuration
Latest models and pricing as of August 2025
"""

from typing import Dict, List, Tuple
from enum import Enum

class TaskType(Enum):
    """Different types of tasks that require LLM inference."""
    DOMAIN_ASSIGNMENT = "domain_assignment"
    DATA_MAPPING = "data_mapping"
    CODE_GENERATION = "code_generation"
    VALIDATION = "validation"
    SCHEMA_INFERENCE = "schema_inference"
    PII_DETECTION = "pii_detection"
    ERROR_HANDLING = "error_handling"
    STANDARDIZATION = "standardization"
    SAMPLING_STRATEGY = "sampling_strategy"
    ARTIFACT_ANALYSIS = "artifact_analysis"
    ANALYSIS = "analysis"  # General analysis tasks

class ModelTier(Enum):
    """Model performance/cost tiers."""
    ULTRA_CHEAP = "ultra_cheap"      # $0.05-0.25 per M tokens
    ECONOMY = "economy"              # $0.25-1.25 per M tokens  
    BALANCED = "balanced"            # $1.25-5.00 per M tokens
    PERFORMANCE = "performance"      # $5.00+ per M tokens

# Latest 2025 Model Pricing (per million tokens)
MODEL_PRICING = {
    # OpenAI GPT-5 Series (2025) - Massive cost reduction!
    "gpt-5-nano": {"input": 0.05, "output": 0.40, "tier": ModelTier.ULTRA_CHEAP},
    "gpt-5-mini": {"input": 0.25, "output": 2.00, "tier": ModelTier.ULTRA_CHEAP},
    "gpt-5": {"input": 1.25, "output": 10.00, "tier": ModelTier.ECONOMY},
    
    # Claude 3.5 Series (Latest)
    "claude-3.5-haiku": {"input": 0.80, "output": 4.00, "tier": ModelTier.ECONOMY},
    "claude-3.5-sonnet": {"input": 3.00, "output": 15.00, "tier": ModelTier.BALANCED},
    "claude-3.5-opus": {"input": 15.00, "output": 75.00, "tier": ModelTier.PERFORMANCE},
    
    # Groq (Ultra-fast inference)
    "groq-llama-3.1-405b": {"input": 0.59, "output": 0.79, "tier": ModelTier.ECONOMY},
    "groq-llama-3.1-70b": {"input": 0.59, "output": 0.79, "tier": ModelTier.ECONOMY},
    "groq-llama-3.1-8b": {"input": 0.05, "output": 0.08, "tier": ModelTier.ULTRA_CHEAP},
    "groq-mixtral-8x7b": {"input": 0.24, "output": 0.24, "tier": ModelTier.ULTRA_CHEAP},
    
    # Local/Ollama (essentially free)
    "ollama-llama3.1": {"input": 0.00, "output": 0.00, "tier": ModelTier.ULTRA_CHEAP},
    "ollama-codellama": {"input": 0.00, "output": 0.00, "tier": ModelTier.ULTRA_CHEAP},
}

# Cost-Optimized Task → Model Routing
COST_OPTIMIZED_ROUTING: Dict[TaskType, List[Tuple[str, str]]] = {
    # Domain assignment: Needs reasoning, but GPT-5 Mini is 12x cheaper than GPT-4
    TaskType.DOMAIN_ASSIGNMENT: [
        ("gpt-5-mini", "Great reasoning at ultra-low cost"),
        ("claude-3.5-haiku", "Fast semantic understanding"),
        ("groq-llama-3.1-70b", "Lightning-fast inference"),
    ],
    
    # Data mapping: Structured tasks, Claude excels here
    TaskType.DATA_MAPPING: [
        ("claude-3.5-haiku", "Optimized for structured data"),
        ("gpt-5-mini", "Strong pattern recognition"),
        ("groq-llama-3.1-70b", "Fast bulk processing"),
    ],
    
    # Code generation: Claude 3.5 Sonnet excels at code, Groq for speed
    TaskType.CODE_GENERATION: [
        ("claude-3.5-sonnet", "Superior code generation quality"),
        ("groq-llama-3.1-70b", "Code-optimized, blazing fast"),
        ("ollama-codellama", "Local, zero cost"),
    ],
    
    # Validation: Simple yes/no, use cheapest possible
    TaskType.VALIDATION: [
        ("gpt-5-nano", "Dirt cheap, perfect for validation"),
        ("groq-llama-3.1-8b", "Ultra-fast simple tasks"),
        ("groq-mixtral-8x7b", "Reliable validation"),
    ],
    
    # Schema inference: Pattern recognition, cost-sensitive
    TaskType.SCHEMA_INFERENCE: [
        ("gpt-5-nano", "Pattern detection at minimal cost"),
        ("groq-llama-3.1-8b", "Fast schema analysis"),
        ("claude-3.5-haiku", "Structured data specialist"),
    ],
    
    # PII detection: Accuracy critical for compliance
    TaskType.PII_DETECTION: [
        ("gpt-5-mini", "Accuracy + cost balance"),
        ("claude-3.5-haiku", "Privacy-focused analysis"),
        ("groq-llama-3.1-70b", "Fast bulk PII scanning"),
    ],
    
    # Error handling: Quick responses needed
    TaskType.ERROR_HANDLING: [
        ("gpt-5-nano", "Ultra-cheap error analysis"),
        ("groq-llama-3.1-8b", "Instant error processing"),
        ("groq-mixtral-8x7b", "Reliable diagnostics"),
    ],
    
    # Standardization: Systematic data cleaning
    TaskType.STANDARDIZATION: [
        ("gpt-5-mini", "Systematic approach"),
        ("claude-3.5-haiku", "Data transformation expert"),
        ("groq-llama-3.1-70b", "Fast bulk standardization"),
    ],
    
    # Sampling strategy: Mathematical reasoning
    TaskType.SAMPLING_STRATEGY: [
        ("gpt-5-mini", "Statistical reasoning"),
        ("claude-3.5-sonnet", "Advanced analytics"),
        ("groq-llama-3.1-70b", "Fast strategy selection"),
    ],
    
    # Artifact analysis: Complex analysis
    TaskType.ARTIFACT_ANALYSIS: [
        ("claude-3.5-sonnet", "Deep analytical capabilities"),
        ("gpt-5", "Comprehensive analysis"),
        ("groq-llama-3.1-405b", "Massive model for complex tasks"),
    ],
    
    # General analysis: Flexible reasoning tasks
    TaskType.ANALYSIS: [
        ("gpt-5-mini", "General purpose analysis"),
        ("claude-3.5-haiku", "Structured reasoning"),
        ("claude-3.5-sonnet", "Advanced analytical thinking"),
    ],
}

# Emergency fallback chain (cheapest to most expensive)
FALLBACK_CHAIN = [
    "gpt-5-nano",           # $0.05 input - absolute cheapest
    "groq-llama-3.1-8b",   # $0.05 input - ultra-fast
    "groq-mixtral-8x7b",   # $0.24 input - reliable
    "gpt-5-mini",          # $0.25 input - solid choice
    "claude-3.5-haiku",    # $0.80 input - structured data expert
    "gpt-5",               # $1.25 input - when you need quality
]

def get_best_model_for_task(task_type: TaskType, budget_tier: ModelTier = ModelTier.ECONOMY) -> str:
    """
    Get the best model for a specific task within budget constraints.
    
    Args:
        task_type: The type of task to perform
        budget_tier: Maximum cost tier willing to pay
        
    Returns:
        Model name to use for the task
    """
    if task_type not in COST_OPTIMIZED_ROUTING:
        # Default to cheapest option for unknown tasks
        return "gpt-5-nano"
    
    # Get candidate models for this task
    candidates = COST_OPTIMIZED_ROUTING[task_type]
    
    # Filter by budget tier
    for model_name, reason in candidates:
        if model_name in MODEL_PRICING:
            model_tier = MODEL_PRICING[model_name]["tier"]
            # Check if this model fits the budget
            if (budget_tier == ModelTier.ULTRA_CHEAP and model_tier in [ModelTier.ULTRA_CHEAP]) or \
               (budget_tier == ModelTier.ECONOMY and model_tier in [ModelTier.ULTRA_CHEAP, ModelTier.ECONOMY]) or \
               (budget_tier == ModelTier.BALANCED and model_tier in [ModelTier.ULTRA_CHEAP, ModelTier.ECONOMY, ModelTier.BALANCED]) or \
               (budget_tier == ModelTier.PERFORMANCE):  # Performance tier allows any model
                return model_name
    
    # Fallback to cheapest option
    return "gpt-5-nano"

def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate cost for a model inference.
    
    Args:
        model_name: Name of the model
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Estimated cost in USD
    """
    if model_name not in MODEL_PRICING:
        # Unknown model, use GPT-5 pricing as default
        model_name = "gpt-5"
    
    pricing = MODEL_PRICING[model_name]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + output_cost

def get_cost_comparison() -> Dict[str, Dict[str, float]]:
    """Get cost comparison for all models."""
    comparison = {}
    
    # Standard task: 1000 input tokens, 500 output tokens
    test_input = 1000
    test_output = 500
    
    for model_name, pricing in MODEL_PRICING.items():
        cost = estimate_cost(model_name, test_input, test_output)
        savings_vs_gpt4 = ((30.0 * test_input + 60.0 * test_output) / 1_000_000) - cost
        savings_percentage = (savings_vs_gpt4 / ((30.0 * test_input + 60.0 * test_output) / 1_000_000)) * 100
        
        comparison[model_name] = {
            "cost_usd": round(cost, 6),
            "savings_vs_gpt4_usd": round(savings_vs_gpt4, 6),
            "savings_percentage": round(savings_percentage, 1),
            "tier": pricing["tier"].value
        }
    
    return comparison

# Print cost analysis for demonstration
if __name__ == "__main__":
    print("🚀 COST-OPTIMIZED MODEL ROUTING - 2025 EDITION")
    print("=" * 60)
    
    print("\n💰 Cost Analysis (1K input + 500 output tokens):")
    comparison = get_cost_comparison()
    
    for model, data in sorted(comparison.items(), key=lambda x: x[1]["cost_usd"]):
        print(f"  {model:20} | ${data['cost_usd']:8.6f} | {data['savings_percentage']:5.1f}% savings | {data['tier']}")
    
    print(f"\n🎯 Task Routing Examples:")
    for task in TaskType:
        best_model = get_best_model_for_task(task, ModelTier.ECONOMY)
        cost = estimate_cost(best_model, 1000, 500)
        print(f"  {task.value:20} → {best_model:15} (${cost:.6f})")
    
    print(f"\n🔥 Key Savings:")
    gpt4_cost = (30.0 * 1000 + 60.0 * 500) / 1_000_000  # $0.060000
    nano_cost = estimate_cost("gpt-5-nano", 1000, 500)   # $0.000250
    print(f"  GPT-4: ${gpt4_cost:.6f}")
    print(f"  GPT-5 Nano: ${nano_cost:.6f}")
    print(f"  💸 Savings: {((gpt4_cost - nano_cost) / gpt4_cost) * 100:.1f}% cheaper!")
