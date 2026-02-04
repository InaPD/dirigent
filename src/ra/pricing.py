"""What a step cost.

Rates are US dollars per million tokens, taken from the Anthropic pricing reference dated
24 Jun 2026. Cache multipliers are the documented defaults: a cache write costs 1.25x the
base input rate and a cache read costs 0.1x.

Re-check these against the pricing page whenever a model is added or a rate changes. A
wrong number here makes every cost in every trace quietly wrong, which is worse than a
crash, so the table is small and explicit on purpose.
"""

from pydantic import BaseModel

from ra.schemas import Usage

PER_MILLION = 1_000_000


class Price(BaseModel):
    """Dollars per million tokens."""

    input: float
    output: float
    cache_write: float
    cache_read: float


PRICES_USD_PER_MTOK: dict[str, Price] = {
    "claude-sonnet-5": Price(input=2.00, output=10.00, cache_write=2.50, cache_read=0.20),
    "claude-haiku-4-5": Price(input=1.00, output=5.00, cache_write=1.25, cache_read=0.10),
}


class UnknownModel(ValueError):
    """A model with no price entry.

    A ValueError rather than a KeyError so that a Pydantic validator turns it into a
    readable ValidationError at startup, which is the only place it should ever surface.
    """

    def __init__(self, model: str) -> None:
        known = ", ".join(sorted(PRICES_USD_PER_MTOK))
        super().__init__(f"no price for {model!r}. Add it to pricing.py. Known models: {known}")


def price_for(model: str) -> Price:
    try:
        return PRICES_USD_PER_MTOK[model]
    except KeyError:
        raise UnknownModel(model) from None


def cost_usd(model: str, usage: Usage) -> float:
    """Dollar cost of one API call, from the token counts the response reported."""
    p = price_for(model)
    dollars = (
        usage.input_tokens * p.input
        + usage.output_tokens * p.output
        + usage.cache_creation_input_tokens * p.cache_write
        + usage.cache_read_input_tokens * p.cache_read
    ) / PER_MILLION
    return round(dollars, 6)
