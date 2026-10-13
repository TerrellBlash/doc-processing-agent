"""
Token prices for cost estimates.

USD per 1M tokens, standard tier, short context: (input, cached input,
output). Source: https://developers.openai.com/api/docs/pricing, checked
2026-10-12. Reasoning tokens are billed as output and are already counted
in output_tokens.
"""

from collections.abc import Mapping
from typing import Any

PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6-sol": (2.00, 0.20, 10.00),
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
}


def cost_usd(model_name: str, usage: Mapping[str, Any]) -> float | None:
    """
    Price one model's token usage.

    - Parameter model_name: the name the API reported, which may carry a
        date suffix (gpt-4o-mini-2024-07-18); the longest matching prefix wins.
    - Parameter usage: LangChain usage metadata for that model.
    - Returns: cost in USD, or None for a model with no listed price.
    """
    matches = [name for name in PRICES if model_name.startswith(name)]
    if not matches:
        return None
    input_price, cached_price, output_price = PRICES[max(matches, key=len)]
    cached = (usage.get("input_token_details") or {}).get("cache_read") or 0
    fresh = usage.get("input_tokens", 0) - cached
    return (fresh * input_price + cached * cached_price + usage.get("output_tokens", 0) * output_price) / 1_000_000
