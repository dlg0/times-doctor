"""LLM cost estimation and controls."""

# Pricing per 1K tokens (as of 2025)
COST_PER_1K = {
    # OpenAI GPT-5 models
    "gpt-5-nano": {"input": 0.0001, "output": 0.0004},
    "gpt-5-mini": {"input": 0.0005, "output": 0.0015},
    "gpt-5": {"input": 0.005, "output": 0.015},
    "gpt-5-pro": {"input": 0.01, "output": 0.03},
    # Anthropic Claude models
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku-20241022": {"input": 0.001, "output": 0.005},
}


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars per token).

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4


def estimate_cost(input_text: str, output_text: str, model: str) -> tuple[int, int, float]:
    """Estimate cost for LLM API call.

    Args:
        input_text: Input/prompt text
        output_text: Expected output text
        model: Model identifier

    Returns:
        Tuple of (input_tokens, output_tokens, cost_usd)
    """
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text) if output_text else input_tokens // 2

    # Get pricing for model
    pricing = COST_PER_1K.get(model)
    if not pricing:
        # Try prefix match
        for key in COST_PER_1K:
            if model.startswith(key):
                pricing = COST_PER_1K[key]
                break

    if not pricing:
        # Default to conservative estimate
        pricing = {"input": 0.01, "output": 0.03}

    cost = (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]

    return input_tokens, output_tokens, cost


def check_cost_limit(estimated_cost: float, max_cost: float) -> bool:
    """Check if estimated cost is within limit.

    Args:
        estimated_cost: Estimated cost in USD
        max_cost: Maximum allowed cost in USD

    Returns:
        True if within limit, False otherwise
    """
    return estimated_cost <= max_cost
